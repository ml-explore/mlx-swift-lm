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

    /// Suppressed vocabulary indices, sorted ascending. IDs at or beyond the
    /// logits vocabulary size are dropped at ``process(logits:)`` time, so a
    /// configuration that inherits placeholder defaults larger than the
    /// model's vocabulary (e.g. tiny test configs) degrades to a no-op
    /// instead of indexing out of range.
    private let sortedIds: [Int]

    /// Suppressed vocabulary indices, shaped `[1, N]` for broadcasting
    /// against `[B, vocab]` logits. Fast path used when every ID fits the
    /// logits vocabulary.
    private let broadcastIndices: MLXArray

    /// `-inf` fill values, shaped `[1, N]` to match `broadcastIndices`.
    private let negInfValues: MLXArray

    /// Create a processor masking the given token IDs.
    ///
    /// Negative IDs are ignored. Returns `nil` when no valid IDs remain
    /// (nothing to suppress).
    public init?(tokenIds: Set<Int>) {
        let sorted = tokenIds.filter { $0 >= 0 }.sorted()
        guard !sorted.isEmpty else { return nil }
        self.sortedIds = sorted
        self.broadcastIndices = MLXArray(sorted.map { UInt32($0) })[.newAxis, 0...]
        self.negInfValues =
            MLXArray([Float](repeating: -Float.infinity, count: sorted.count))[
                .newAxis, 0...]
    }

    public mutating func prompt(_ prompt: MLXArray) {
        // Stateless: suppression does not depend on the prompt.
    }

    public func process(logits: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        if sortedIds.last! < vocabularySize {
            return putAlong(
                logits, broadcastIndices, values: negInfValues.asType(logits.dtype), axis: -1)
        }
        // Slow path: some IDs exceed this model's vocabulary — mask only the
        // in-range subset.
        let valid = sortedIds.prefix { $0 < vocabularySize }
        guard !valid.isEmpty else { return logits }
        let indices = MLXArray(valid.map { UInt32($0) })[.newAxis, 0...]
        let values = MLXArray([Float](repeating: -Float.infinity, count: valid.count))[
            .newAxis, 0...]
        return putAlong(logits, indices, values: values.asType(logits.dtype), axis: -1)
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

/// Side table associating suppressed token IDs with models that do not
/// conform to ``SuppressedTokensProviding``.
///
/// `suppress_tokens` from `generation_config.json` must be honored for every
/// model, not only those that adopt the protocol; this registry provides the
/// storage for the rest. Keys are held weakly, so entries disappear with
/// their model.
private final class SuppressedTokensRegistry: @unchecked Sendable {

    static let shared = SuppressedTokensRegistry()

    private let lock = NSLock()
    private let table = NSMapTable<AnyObject, NSSet>.weakToStrongObjects()

    func union(_ tokenIds: Set<Int>, for model: AnyObject) {
        guard !tokenIds.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        let existing = (table.object(forKey: model) as? Set<Int>) ?? []
        table.setObject(existing.union(tokenIds) as NSSet, forKey: model)
    }

    func tokenIds(for model: AnyObject) -> Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return (table.object(forKey: model) as? Set<Int>) ?? []
    }
}

/// Merge `suppress_tokens` from `generation_config.json` into the model's
/// suppressed-token set.
///
/// Models conforming to ``SuppressedTokensProviding`` receive the IDs in
/// ``SuppressedTokensProviding/suppressedTokenIds``; every other model is
/// tracked in a weak side table, so `generation_config.json` is honored
/// regardless of conformance. No-op when the configuration declares no
/// suppressed tokens. Called by the model factories after the model is
/// created.
public func mergeGenerationConfigSuppressedTokens(
    _ generationConfig: GenerationConfigFile?, into model: any LanguageModel
) {
    guard let suppressTokens = generationConfig?.suppressTokens?.values else { return }
    if let provider = model as? SuppressedTokensProviding {
        provider.suppressedTokenIds.formUnion(suppressTokens)
    } else {
        SuppressedTokensRegistry.shared.union(Set(suppressTokens), for: model)
    }
}

/// The full suppressed-token set for a model: protocol-advertised IDs plus
/// any `generation_config.json` IDs tracked in the side table.
func suppressedTokenIds(for model: any LanguageModel) -> Set<Int> {
    var ids = (model as? SuppressedTokensProviding)?.suppressedTokenIds ?? []
    ids.formUnion(SuppressedTokensRegistry.shared.tokenIds(for: model))
    return ids
}

/// Build the ``SuppressTokensProcessor`` for a model, or `nil` when it has
/// no suppressed tokens.
func makeSuppressTokensProcessor(model: any LanguageModel) -> SuppressTokensProcessor? {
    SuppressTokensProcessor(tokenIds: suppressedTokenIds(for: model))
}

/// Build the ``LogitProcessor`` for a generation run: the parameter-derived
/// penalty processor chained with a ``SuppressTokensProcessor`` when the
/// model has suppressed token IDs (via ``SuppressedTokensProviding`` or
/// `generation_config.json`).
func makeLogitProcessor(
    parameters: GenerateParameters, model: any LanguageModel
) -> LogitProcessor? {
    let base = parameters.processor()
    guard let suppressor = makeSuppressTokensProcessor(model: model) else {
        return base
    }
    guard let base else { return suppressor }
    return ChainedLogitProcessor(processors: [base, suppressor])
}

/// A ``LogitSampler`` that masks suppressed token IDs before delegating to a
/// base sampler.
///
/// Used for draft-token sampling paths that receive a sampler but not the
/// iterator's ``LogitProcessor`` chain (e.g.
/// ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:queryOffset:blockSize:sampler:)``).
/// Draft proposals of suppressed tokens would always be rejected by the
/// verifier — masking them at the source keeps speculative acceptance high.
/// ``SuppressTokensProcessor`` is stateless, so wrapping is safe.
struct SuppressTokensSampler: LogitSampler {

    let base: LogitSampler
    let suppressor: SuppressTokensProcessor

    func sample(logits: MLXArray) -> MLXArray {
        base.sample(logits: suppressor.process(logits: logits))
    }
}
