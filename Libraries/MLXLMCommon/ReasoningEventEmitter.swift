// Copyright © 2025 Apple Inc.

/// Routes a model's decoded generation stream into reasoning (chain-of-thought)
/// vs response segments by scanning for the model's reasoning delimiters.
///
/// A value-type streaming scanner modeled on ``WhitespaceRunTracker``: feed it
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
/// **State model.** Conceptually `Outside → Inside → Closed`, but represented
/// compactly as `inside: Bool` plus `pendingPrefix` (the diagram's
/// `PendingStart`/`PendingEnd` are "pendingPrefix is non-empty"; `Closed` is
/// "not inside, having produced reasoning"). When not inside, the scanner
/// watches for the start delimiter; when inside, the end delimiter. A start
/// delimiter always (re)opens a reasoning span — so multiple blocks each route,
/// and the cost is a documented limitation: a literal `<think>` appearing in
/// answer text is misrouted (the deferred token-ID detection is the real fix).
public struct ReasoningEventEmitter: Sendable {

    /// A routed slice of the decoded stream.
    public enum Segment: Sendable, Equatable {
        case reasoning(String)
        case response(String)
    }

    private let startDelimiter: String
    private let endDelimiter: String

    /// Whether the scanner is currently inside a reasoning span.
    private var inside: Bool

    /// Text held back because it may be the prefix of a delimiter split across a
    /// chunk boundary. Always a *proper* prefix of the currently-watched delimiter.
    private var pendingPrefix: String = ""

    /// When set, the next non-empty emission has its leading whitespace trimmed.
    /// Set after consuming any delimiter, so the template newline(s) immediately
    /// following `<think>`/`</think>` are dropped (mirrors `unwrapToolCallMarkers`).
    private var pendingLeadingTrim: Bool = false

    /// True once an end delimiter has been consumed, i.e. a reasoning span has
    /// closed at least once. Unlike ``isInsideReasoning``, this latches — so a
    /// caller (e.g. a think-then-call token collector) can detect a close even
    /// when an empty `<think></think>` resolves within a single ``process(_:)``
    /// call, where sampling ``isInsideReasoning`` afterward reads `false` both
    /// before and after and the transient open is invisible.
    public private(set) var hasClosedReasoning: Bool = false

    public init(config: ReasoningConfig, primedInside: Bool) {
        self.startDelimiter = config.startDelimiter
        self.endDelimiter = config.endDelimiter
        self.inside = primedInside
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
        var trimmed = Substring(tail)
        while let last = trimmed.last, last.isWhitespace { trimmed = trimmed.dropLast() }
        guard let lastStart = trimmed.range(of: config.startDelimiter, options: .backwards) else {
            return false
        }
        return trimmed[lastStart.upperBound...].range(of: config.endDelimiter) == nil
    }

    /// Whether the scanner is currently inside a reasoning span.
    ///
    /// The generation loop reads this to attribute generated tokens to the
    /// reasoning token count (one `.token` = one token), since the emitter
    /// itself only sees decoded text, not token IDs.
    public var isInsideReasoning: Bool { inside }

    /// Ingests one decoded chunk and returns the segments it resolves to.
    ///
    /// May return zero segments (e.g. the chunk only advanced a partial
    /// delimiter), or several (e.g. a chunk containing a full `<think>…</think>`).
    public mutating func process(_ chunk: String) -> [Segment] {
        var output: [Segment] = []
        var working = Substring(pendingPrefix + chunk)
        pendingPrefix = ""

        while true {
            let delimiter = inside ? endDelimiter : startDelimiter
            if let range = working.range(of: delimiter) {
                // Text before the marker belongs to the current mode; trim the
                // whitespace immediately preceding the marker.
                appendSegment(
                    String(working[working.startIndex ..< range.lowerBound]),
                    trimmingTrailing: true, into: &output)
                // Consume the marker and trim whitespace immediately after it.
                working = working[range.upperBound...]
                pendingLeadingTrim = true
                // Matching while `inside` means we just consumed an *end*
                // delimiter (`delimiter == endDelimiter`) — a close.
                if inside { hasClosedReasoning = true }
                inside.toggle()
                // Re-scan the remainder in the new mode.
            } else {
                // No full marker. Hold back any suffix that could begin one on
                // the next chunk; emit the rest in the current mode.
                let tail = heldBackTailLength(working, delimiter: delimiter)
                let splitIndex = working.index(working.endIndex, offsetBy: -tail)
                appendSegment(
                    String(working[working.startIndex ..< splitIndex]),
                    trimmingTrailing: false, into: &output)
                pendingPrefix = String(working[splitIndex...])
                break
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
        if !pendingPrefix.isEmpty {
            let leftover = pendingPrefix
            pendingPrefix = ""
            appendSegment(leftover, trimmingTrailing: true, into: &output)
        }
        return output
    }

    // MARK: - Private

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
        if inside {
            output.append(.reasoning(String(t)))
        } else {
            output.append(.response(String(t)))
        }
    }

    /// The length of the longest suffix of `text` that is a *proper* prefix of
    /// `delimiter` (and therefore might complete into the delimiter on the next
    /// chunk). Returns 0 when no suffix could begin the delimiter.
    private func heldBackTailLength(_ text: Substring, delimiter: String) -> Int {
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
