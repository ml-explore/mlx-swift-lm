// Copyright © 2026 Apple Inc.

/// Drives a ``ReasoningEventEmitter`` over a raw generated-token stream,
/// accumulating the reasoning-span token IDs while routing decoded text to
/// reasoning/response segments.
///
/// This is the pure, model-free core of think-then-call **Phase 1**:
/// it owns a ``NaiveStreamingDetokenizer`` and an emitter, so the device-side
/// caller only supplies token IDs (from `generateTokens`) and forwards the
/// returned segments to its channel. Token IDs are carried verbatim — no
/// decode→re-encode round-trip — so the accumulated span prefills the
/// constrained Phase 2 exactly.
///
/// **Why a separate type.** The emitter is intentionally text-only (it never
/// sees token IDs). Phase 1 additionally needs to (a) retain the raw IDs for the
/// hand-off and (b) know when to stop generating. Keeping that here — rather than
/// inline in the executor — makes the logic host-testable with no model, and lets
/// the unconstrained reasoning path adopt it later to share one loop.
public struct ReasoningTokenCollector {

    private var emitter: ReasoningEventEmitter
    private var detokenizer: NaiveStreamingDetokenizer

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
        self.emitter = ReasoningEventEmitter(config: config, primedInside: primedInside)
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
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
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return [] }
        return emitter.process(chunk)
    }

    /// Flush any held-back text at end of generation. If the stream ended
    /// mid-reasoning (no close ever arrived), the leftover routes as `.reasoning`.
    public mutating func finalize() -> [ReasoningEventEmitter.Segment] {
        emitter.finalize()
    }
}
