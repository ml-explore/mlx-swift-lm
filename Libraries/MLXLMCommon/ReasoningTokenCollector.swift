// Copyright © 2026 Apple Inc.

/// Drives a ``ReasoningEventEmitter`` over a raw generated-token stream,
/// accumulating the reasoning-span token IDs while routing decoded text to
/// reasoning/response segments.
///
/// This is the pure, model-free core of think-then-call **Phase 1**:
/// it owns a ``StreamingDetokenizer`` and an emitter, so the device-side
/// caller only supplies token IDs (from `generateTokens`) and forwards the
/// returned segments to its channel. Token IDs are carried verbatim — no
/// decode→re-encode round-trip — so the accumulated span prefills the
/// constrained Phase 2 exactly.
///
/// **Why a separate type.** The emitter routes text and never sees token IDs.
/// Phase 1 additionally needs to (a) retain the raw IDs for the hand-off and
/// (b) know when to stop generating. Keeping that here — rather than inline in the
/// executor — makes the logic host-testable with no model. The unconstrained
/// reasoning path uses it too.
///
/// When both framing delimiters are single tokens, they and any single-token implicit
/// end delimiter are matched by id and reported to the emitter as boundaries, so the
/// same characters spelled out as text are content.
public struct ReasoningTokenCollector {

    private var emitter: ReasoningEventEmitter
    private var detokenizer: any StreamingDetokenizer
    private let tokenizer: any Tokenizer

    /// Set when the delimiters are single tokens, which are then matched by id.
    private let delimiterTokens: ReasoningDelimiterTokens?

    /// Every token ingested so far, in order. Phase 2 prefills the model's
    /// prompt + these to continue from the completed reasoning span.
    ///
    /// Because the caller stops ingesting once ``shouldStopAfterReasoning`` is
    /// true, this ends at the closing-delimiter token. The *opening* delimiter is
    /// included when the model generates it (non-primed families, e.g. Qwen3);
    /// for primed families (e.g. DeepSeek-R1) the opening `<think>` lives in the
    /// prompt instead, so it is already part of the Phase-2 prefix.
    public private(set) var reasoningTokenIDs: [Int] = []

    public init(config: ReasoningConfig, primedInside: Bool, tokenizer: any Tokenizer) {
        let delimiterTokens = ReasoningDelimiterTokens(config: config, tokenizer: tokenizer)
        self.delimiterTokens = delimiterTokens
        self.emitter = ReasoningEventEmitter(
            config: config, primedInside: primedInside, delimiterTokens: delimiterTokens)
        self.tokenizer = tokenizer
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
    }

    /// Whether the scanner is currently inside a reasoning span.
    public var isInsideReasoning: Bool { emitter.isInsideReasoning }

    /// Whether a reasoning span has closed — the Phase 1 → Phase 2 boundary.
    ///
    /// Latches on the FIRST close (a later stray `<think>` re-opens the emitter,
    /// but the caller has already stopped). Crucially this detects an empty
    /// `<think></think>` that opens and closes within a single decoded chunk,
    /// which sampling ``isInsideReasoning`` after `ingest` cannot.
    public var shouldStopAfterReasoning: Bool { emitter.hasClosedReasoning }

    /// Ingest one generated token: append it to ``reasoningTokenIDs``, advance the
    /// detokenizer, and return the routed segments (forward these to the channel).
    /// Returns an empty array when the token only advanced an incomplete multibyte
    /// character or a partial delimiter held back across the chunk boundary.
    public mutating func ingest(_ token: Int) -> [ReasoningEventEmitter.Segment] {
        reasoningTokenIDs.append(token)
        if let boundary = delimiterTokens?.boundary(for: token), emitter.isBoundary(boundary) {
            // Text the detokenizer still holds precedes the boundary.
            let pending = detokenizer.finish().map { emitter.process($0) } ?? []
            detokenizer = tokenizer.makeStreamingDetokenizer()
            return pending + emitter.process(boundary: boundary)
        }
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return [] }
        return emitter.process(chunk)
    }

    /// Flush any held-back text at end of generation. If the stream ended
    /// mid-reasoning (no close ever arrived), the leftover routes as `.reasoning`.
    public mutating func finalize() -> [ReasoningEventEmitter.Segment] {
        let pending = detokenizer.finish().map { emitter.process($0) } ?? []
        return pending + emitter.finalize()
    }
}
