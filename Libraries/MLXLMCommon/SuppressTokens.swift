// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A language model that exposes token IDs that must never be sampled
/// during generation.
///
/// Multimodal models seed this set with the placeholder token IDs from their
/// configuration (image / audio / video soft tokens and their begin/end
/// markers). Those tokens are prompt-side markers that carry no meaning as
/// generated text, but they can occasionally be sampled if left unmasked and
/// then leak into the output verbatim (e.g. `<audio|>` read aloud by TTS).
/// Model factories additionally merge `suppress_tokens` from
/// `generation_config.json` into this set after the model is created.
///
/// Token iterators consult this set at initialization and chain a
/// ``SuppressTokensProcessor`` after the parameter-derived processor, masking
/// the corresponding logits to `-inf` before sampling. Only generated tokens
/// are affected; prompt tokens are never modified.
public protocol SuppressedTokensProviding: AnyObject {

    /// Token IDs whose logits are masked to `-inf` before sampling.
    var suppressedTokenIds: Set<Int> { get set }
}

/// Processor that masks a fixed set of token IDs to `-inf` so they can never
/// be sampled.
///
/// Port of `SuppressTokensLogitsProcessor` from Hugging Face `transformers`,
/// which is driven by `suppress_tokens` in `generation_config.json`.
public struct SuppressTokensProcessor: LogitProcessor {

    /// Suppressed vocabulary indices, shaped `[1, N]` for broadcasting
    /// against `[B, vocab]` logits.
    private let broadcastIndices: MLXArray

    /// `-inf` fill values, shaped `[1, N]` to match `broadcastIndices`.
    private let negInfValues: MLXArray

    /// Create a processor masking the given token IDs.
    ///
    /// Returns `nil` when `tokenIds` is empty (nothing to suppress).
    public init?(tokenIds: Set<Int>) {
        guard !tokenIds.isEmpty else { return nil }
        let sorted = tokenIds.sorted().map { UInt32($0) }
        self.broadcastIndices = MLXArray(sorted)[.newAxis, 0...]
        self.negInfValues =
            MLXArray([Float](repeating: -Float.infinity, count: sorted.count))[
                .newAxis, 0...]
    }

    public mutating func prompt(_ prompt: MLXArray) {
        // Stateless: suppression does not depend on the prompt.
    }

    public func process(logits: MLXArray) -> MLXArray {
        putAlong(
            logits, broadcastIndices, values: negInfValues.asType(logits.dtype), axis: -1)
    }

    public mutating func didSample(token: MLXArray) {
        // Stateless: nothing to track.
    }
}

/// Processor that applies multiple ``LogitProcessor``s in order.
public struct ChainedLogitProcessor: LogitProcessor {

    private var processors: [any LogitProcessor]

    public init(processors: [any LogitProcessor]) {
        self.processors = processors
    }

    public mutating func prompt(_ prompt: MLXArray) {
        for index in processors.indices {
            processors[index].prompt(prompt)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        processors.reduce(logits) { $1.process(logits: $0) }
    }

    public mutating func didSample(token: MLXArray) {
        for index in processors.indices {
            processors[index].didSample(token: token)
        }
    }
}

/// Merge `suppress_tokens` from `generation_config.json` into the model's
/// ``SuppressedTokensProviding/suppressedTokenIds``.
///
/// No-op when the configuration declares no suppressed tokens or the model
/// does not conform to ``SuppressedTokensProviding``. Called by the model
/// factories after the model is created.
public func mergeGenerationConfigSuppressedTokens(
    _ generationConfig: GenerationConfigFile?, into model: any LanguageModel
) {
    guard let suppressTokens = generationConfig?.suppressTokens?.values else { return }
    (model as? SuppressedTokensProviding)?.suppressedTokenIds.formUnion(suppressTokens)
}

/// Build the ``LogitProcessor`` for a generation run: the parameter-derived
/// penalty processor chained with a ``SuppressTokensProcessor`` when the
/// model advertises suppressed token IDs via ``SuppressedTokensProviding``.
func makeLogitProcessor(
    parameters: GenerateParameters, model: any LanguageModel
) -> LogitProcessor? {
    let base = parameters.processor()
    guard let provider = model as? SuppressedTokensProviding,
        let suppressor = SuppressTokensProcessor(tokenIds: provider.suppressedTokenIds)
    else {
        return base
    }
    guard let base else { return suppressor }
    return ChainedLogitProcessor(processors: [base, suppressor])
}
