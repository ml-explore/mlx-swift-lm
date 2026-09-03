// Copyright © 2025 Apple Inc.

/// Routes a model's decoded generation stream into reasoning (chain-of-thought)
/// vs response segments by scanning for the model's reasoning delimiters and
/// any protocol-declared implicit exits.
///
/// A value-type streaming scanner: feed it
/// each decoded chunk via ``process(_:)`` and it returns the routed segments,
/// holding back any partial delimiter that straddles a chunk boundary
/// (`pendingPrefix`). This makes detection robust to the detokenizer or
/// tool-call processor fragmenting a `<think>` across chunks.
///
/// **Primed state.** The headline reasoning families (Qwen3 with
/// thinking enabled, DeepSeek-R1) prefill the *opening* delimiter into the
/// rendered prompt, so the model's first generated token is already reasoning
/// content and it never emits an opening `<think>` in the stream — only the
/// closing `</think>`. Construct with `primedInside: true` for those, seeded by
/// inspecting the rendered prompt tail.
///
/// **State model.** `Outside -> Inside -> Outside`, plus a `Label` step for the
/// labeled-channel protocol (see ``ReasoningChannel``) where the role label
/// following the start delimiter decides which way the span routes. When outside,
/// the scanner watches for the start delimiter; when inside, the end delimiter
/// and any ``ReasoningConfig/implicitEndDelimiters``, earliest match winning.
/// A canonical end delimiter is framing and is consumed, while an implicit one
/// is answer/tool content and stays in the stream to be re-scanned outside.
/// `pendingPrefix` holds text that may be a delimiter split across a chunk
/// boundary. For a delimiter pair a start delimiter always (re)opens a reasoning
/// span - so multiple blocks each route, and the cost is a documented limitation:
/// a literal `<think>` appearing in answer text is misrouted (the deferred
/// token-ID detection is the real fix).
public struct ReasoningEventEmitter: Sendable {

    /// A routed slice of the decoded stream.
    public enum Segment: Sendable, Equatable {
        case reasoning(String)
        case response(String)
    }

    /// Which stream the text being scanned belongs to.
    private enum Route {
        case reasoning
        case response
    }

    private enum State {
        /// Not in a span: text is response, and the start delimiter is watched for.
        case outside
        /// The start delimiter has been consumed and the channel's role label is
        /// being read. Labeled channels only; nothing is emitted while here.
        case label
        /// Inside a span: text routes to the associated route and the end
        /// delimiter is watched for.
        case inside(Route)
    }

    private let startDelimiter: String
    private let endDelimiter: String
    private let endDelimiters: [String]

    /// The labeled-channel protocol, or `nil` for a plain delimiter pair.
    private let channel: ReasoningChannel?

    private var state: State

    /// Text held back because it may be the prefix of a delimiter split across a
    /// chunk boundary. Always a *proper* prefix of one of the currently-watched
    /// delimiters - or, in the `label` state, the unresolved label itself, which
    /// ``ReasoningChannel/maxLabelLength`` bounds.
    private var pendingPrefix: String = ""

    /// When set, the next non-empty emission has its leading whitespace trimmed.
    /// Set after consuming any delimiter, so the template newline(s) immediately
    /// following `<think>`/`</think>` are dropped (mirrors `unwrapToolCallMarkers`).
    private var pendingLeadingTrim: Bool = false

    /// True once a canonical or implicit end boundary has closed a *reasoning*
    /// span. Unlike ``isInsideReasoning``, this latches - so a caller (e.g. a
    /// think-then-call token collector) can detect a close even when an empty
    /// `<think></think>` resolves within a single ``process(_:)`` call, where
    /// sampling ``isInsideReasoning`` afterward reads `false` both before and
    /// after and the transient open is invisible.
    public private(set) var hasClosedReasoning: Bool = false

    public init(config: ReasoningConfig, primedInside: Bool) {
        self.startDelimiter = config.startDelimiter
        self.endDelimiter = config.endDelimiter
        self.endDelimiters =
            [config.endDelimiter]
            + config.implicitEndDelimiters.filter {
                $0 != config.endDelimiter
            }
        self.channel = config.channel
        // A prompt can only prime a *reasoning* span: prefilling the answer
        // channel would mean the template wrote the answer.
        self.state = primedInside ? .inside(.reasoning) : .outside
    }

    /// Whether a rendered prompt ends *inside* an open reasoning block — used to
    /// seed `primedInside`.
    ///
    /// The headline families (Qwen3 with thinking enabled, DeepSeek-R1) prefill
    /// the opening delimiter into the assistant generation prompt, so the model's
    /// first generated token is already reasoning content and it never emits an
    /// opening `<think>` — only the closing `</think>`. An emitter started
    /// `Outside` would misroute the entire thought block to `.response` and leak
    /// a bare `</think>`.
    ///
    /// The check must NOT be a naive `hasSuffix(startDelimiter)`: templates
    /// routinely append a trailing newline (`<think>\n`) after the prefill, so a
    /// strict suffix test returns false and silently misroutes 100% of reasoning.
    /// Instead: trim trailing whitespace, then test whether the last start
    /// delimiter is not followed by a matching end delimiter.
    public static func promptEndsInsideReasoning(
        renderedPromptTail tail: String, config: ReasoningConfig
    ) -> Bool {
        guard !config.startDelimiter.isEmpty else { return false }
        var trimmed = Substring(tail)
        while let last = trimmed.last, last.isWhitespace { trimmed = trimmed.dropLast() }
        guard let lastStart = trimmed.range(of: config.startDelimiter, options: .backwards) else {
            return false
        }
        let afterStart = trimmed[lastStart.upperBound...]
        return ([config.endDelimiter] + config.implicitEndDelimiters)
            .filter { !$0.isEmpty }
            .allSatisfy {
                afterStart.range(of: $0) == nil
            }
    }

    /// Whether a rendered prompt, given as its token ids, ends inside an open
    /// reasoning block.
    ///
    /// Decodes only the prompt tail: a prefilled delimiter is the last thing a
    /// generation prompt writes, so scanning further back cannot change the
    /// answer and would cost a full-prompt decode. Special tokens are kept:
    /// families whose delimiters are special tokens (Gemma 4, Qwen3) would
    /// otherwise decode to nothing.
    public static func promptEndsInsideReasoning(
        promptTokens: [Int], config: ReasoningConfig, tokenizer: any Tokenizer
    ) -> Bool {
        promptEndsInsideReasoning(
            renderedPromptTail: tokenizer.decode(
                tokenIds: Array(promptTokens.suffix(promptTailTokenCount))),
            config: config)
    }

    /// How many trailing prompt tokens ``promptEndsInsideReasoning(promptTokens:config:tokenizer:)``
    /// decodes. Generous next to the few tokens a prefill occupies.
    private static let promptTailTokenCount = 64

    /// Whether the scanner is currently inside a reasoning span.
    ///
    /// The generation loop reads this to attribute generated tokens to the
    /// reasoning token count (one `.token` = one token), since the emitter
    /// itself only sees decoded text, not token IDs.
    ///
    /// Reading a channel's label counts as inside: the channel is open, and the
    /// handful of label tokens are metadata that no answer will ever contain.
    public var isInsideReasoning: Bool {
        switch state {
        case .outside: false
        case .label: true
        case .inside(let route): route == .reasoning
        }
    }

    /// Ingests one decoded chunk and returns the segments it resolves to.
    ///
    /// May return zero segments (e.g. the chunk only advanced a partial
    /// delimiter), or several (e.g. a chunk containing a full `<think>…</think>`).
    public mutating func process(_ chunk: String) -> [Segment] {
        var output: [Segment] = []
        var working = Substring(pendingPrefix + chunk)
        pendingPrefix = ""

        scan: while true {
            switch state {
            case .label:
                guard let resolved = resolveLabel(&working) else {
                    // Nothing can be emitted until the label resolves, and it is
                    // short enough that the next chunk settles it: hold it all.
                    pendingPrefix = String(working)
                    break scan
                }
                state = resolved

            case .outside, .inside:
                let watched = watchedDelimiters
                guard let hit = earliestMatch(in: working, of: watched) else {
                    // No full delimiter. Hold back any suffix that could begin one
                    // on the next chunk; emit the rest in the current mode.
                    let tail = heldBackTailLength(working, delimiters: watched)
                    let splitIndex = working.index(working.endIndex, offsetBy: -tail)
                    appendSegment(
                        String(working[working.startIndex ..< splitIndex]),
                        trimmingTrailing: false, into: &output)
                    pendingPrefix = String(working[splitIndex...])
                    break scan
                }

                // Text before the marker belongs to the current mode; trim the
                // whitespace immediately preceding the marker.
                appendSegment(
                    String(working[working.startIndex ..< hit.range.lowerBound]),
                    trimmingTrailing: true, into: &output)

                // An implicit end delimiter is answer/tool content rather than
                // framing, so it stays in the stream to be re-scanned in the new
                // mode. Decided before `stateAfterDelimiter` advances the state.
                let isImplicitEnd: Bool
                if case .inside = state, hit.delimiter != endDelimiter {
                    isImplicitEnd = true
                } else {
                    isImplicitEnd = false
                }

                let next = stateAfterDelimiter(hit.delimiter)
                if isImplicitEnd {
                    working = working[hit.range.lowerBound...]
                } else {
                    // Consume the marker, trim whitespace immediately after it,
                    // and re-scan the remainder in the mode it put us in.
                    working = working[hit.range.upperBound...]
                    pendingLeadingTrim = true
                }
                state = next
            }
        }
        return output
    }

    /// Flushes any held-back text at end of generation.
    ///
    /// If the stream ends mid-reasoning (no closing delimiter ever arrived —
    /// e.g. a primed model that hit `maxTokens`), the leftover is emitted as
    /// `.reasoning`.
    public mutating func finalize() -> [Segment] {
        var output: [Segment] = []
        if case .label = state {
            // The label never terminated. We are still inside the channel, so keep
            // routing to reasoning rather than pretending the block ended - and
            // stop waiting for a label no further chunk will supply.
            state = .inside(.reasoning)
        }
        if !pendingPrefix.isEmpty {
            let leftover = pendingPrefix
            pendingPrefix = ""
            appendSegment(leftover, trimmingTrailing: true, into: &output)
        }
        return output
    }

    // MARK: - Private

    /// The delimiters watched in the current state, earliest match wins.
    private var watchedDelimiters: [String] {
        switch state {
        case .outside:
            // A labeled channel's delimiters are special tokens, so an end
            // delimiter arriving with no opener cannot be prose the model typed:
            // it means the PROMPT opened the channel. Watch for it here so it is
            // swallowed rather than leaked into the answer as a raw marker.
            channel == nil ? [startDelimiter] : [startDelimiter, endDelimiter]
        case .label:
            []
        case .inside:
            // Canonical close plus any protocol-declared implicit exits, e.g.
            // Qwen3.5 opening `<tool_call>` straight out of its thinking block.
            endDelimiters
        }
    }

    /// The state entered after consuming `delimiter` in the current state.
    private mutating func stateAfterDelimiter(_ delimiter: String) -> State {
        switch state {
        case .outside:
            guard channel != nil else { return .inside(.reasoning) }
            // Stray end delimiter (see `watchedDelimiters`): consumed, opens nothing.
            return delimiter == startDelimiter ? .label : .outside
        case .inside(let route):
            // Matching while inside means an *end* delimiter was consumed: a close.
            if route == .reasoning { hasClosedReasoning = true }
            return .outside
        case .label:
            // Unreachable: `.label` is resolved by `resolveLabel`, not by this scan.
            return state
        }
    }

    /// Consumes a channel's role label and decides where its body routes. Returns
    /// `nil` while the label is still incomplete.
    ///
    /// Every decision is made on where a terminator STARTS, never on how much text
    /// has been buffered, because the two disagree: buffered length counts a
    /// half-arrived end delimiter, so a rule phrased in terms of it classifies the
    /// same bytes differently depending on where the transport split them.
    private mutating func resolveLabel(_ working: inout Substring) -> State? {
        guard let channel else { return .inside(.reasoning) }

        let terminator = withinLabelWindow(
            working.range(of: channel.labelTerminator), in: working, channel: channel)
        let close = withinLabelWindow(
            working.range(of: endDelimiter), in: working, channel: channel)

        // Closed before any terminator: an empty channel whose label never ended.
        // There is no body and the label is metadata, so the whole thing goes.
        //
        // This still counts as a reasoning span closing. `isInsideReasoning` reports
        // true while the label is being read, so a caller that stops on
        // `hasClosedReasoning` (the think-then-call collector) would otherwise never
        // be released by a malformed opener.
        if let close, terminator.map({ close.lowerBound < $0.lowerBound }) ?? true {
            working = working[close.upperBound...]
            pendingLeadingTrim = true
            hasClosedReasoning = true
            return .outside
        }

        if let terminator {
            let label = trimmingWhitespace(working[working.startIndex ..< terminator.lowerBound])
            working = working[terminator.upperBound...]
            pendingLeadingTrim = true
            return .inside(channel.responseLabels.contains(label) ? .response : .reasoning)
        }

        // No terminator begins inside the window. Waiting is only pointless once
        // enough text has arrived that one starting at the very last in-window
        // position would have completed - before that, a terminator may still be
        // half here. Nothing is consumed: the body scan re-reads this text, so a
        // malformed opener shows its text rather than silently swallowing it.
        let settled =
            working.count
            >= channel.maxLabelLength + Swift.max(endDelimiter.count, channel.labelTerminator.count)
        return settled ? .inside(.reasoning) : nil
    }

    /// A terminator only ends the label if it begins within `maxLabelLength` of
    /// the start delimiter. Past that this is not a label, and its text is body
    /// that must stay visible.
    private func withinLabelWindow(
        _ range: Range<Substring.Index>?, in text: Substring, channel: ReasoningChannel
    ) -> Range<Substring.Index>? {
        guard let range else { return nil }
        return text.distance(from: text.startIndex, to: range.lowerBound) <= channel.maxLabelLength
            ? range : nil
    }

    private func trimmingWhitespace(_ text: Substring) -> String {
        var trimmed = text
        while let first = trimmed.first, first.isWhitespace { trimmed = trimmed.dropFirst() }
        while let last = trimmed.last, last.isWhitespace { trimmed = trimmed.dropLast() }
        return String(trimmed)
    }

    /// The earliest of `delimiters` present in `text`, or `nil` if none is.
    private func earliestMatch(in text: Substring, of delimiters: [String])
        -> (delimiter: String, range: Range<Substring.Index>)?
    {
        var best: (delimiter: String, range: Range<Substring.Index>)?
        for delimiter in delimiters {
            guard let range = text.range(of: delimiter) else { continue }
            if let current = best, current.range.lowerBound <= range.lowerBound { continue }
            best = (delimiter, range)
        }
        return best
    }

    /// Appends `text` as a segment in the current mode, applying the pending
    /// leading-trim and (optionally) trailing-trim, and skipping empties.
    private mutating func appendSegment(
        _ text: String, trimmingTrailing: Bool, into output: inout [Segment]
    ) {
        if text.isEmpty { return }
        var t = Substring(text)
        if pendingLeadingTrim {
            t = t.drop(while: { $0.isWhitespace })
        }
        if trimmingTrailing {
            while let last = t.last, last.isWhitespace { t.removeLast() }
        }
        // All-whitespace after trimming: emit nothing, keep the leading-trim
        // pending so it applies to the next real text.
        if t.isEmpty { return }
        pendingLeadingTrim = false
        switch state {
        case .inside(.reasoning), .label:
            output.append(.reasoning(String(t)))
        case .inside(.response), .outside:
            output.append(.response(String(t)))
        }
    }

    /// The longest hold-back any of `delimiters` requires.
    private func heldBackTailLength(_ text: Substring, delimiters: [String]) -> Int {
        delimiters.reduce(0) { Swift.max($0, heldBackTailLength(text, delimiter: $1)) }
    }

    /// The length of the longest suffix of `text` that is a *proper* prefix of
    /// `delimiter` (and therefore might complete into the delimiter on the next
    /// chunk). Returns 0 when no suffix could begin the delimiter.
    private func heldBackTailLength(_ text: Substring, delimiter: String) -> Int {
        guard !delimiter.isEmpty else { return 0 }
        let textChars = Array(text)
        let delimiterChars = Array(delimiter)
        var k = min(textChars.count, delimiterChars.count - 1)
        while k >= 1 {
            if textChars.suffix(k).elementsEqual(delimiterChars.prefix(k)) {
                return k
            }
            k -= 1
        }
        return 0
    }
}
