// Copyright © 2026 Apple Inc.
//
// The Qwen3.8-Flash-Next (qwen4_exp) MTP head, after the vLLM and SGLang Qwen4Exp MTP models.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Drafts one token per round for a ``Qwen4ExpModel`` target.
///
/// Like Qwen3.5's head, it pairs each token with the target state at the position
/// before it. The state is the target's hyper-connection streams before the final
/// mixer, and the head returns new streams for its next step.
public final class Qwen4ExpMTPDraftModel: Module, ResumableMTPDrafterModel {
    let configuration: Qwen4ExpTextConfiguration
    public let maximumBlockSize: Int? = 2
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true

    @ModuleInfo(key: "mtp") var mtp: Qwen4ExpMTPPredictor

    /// - Throws: `ModelFactoryError.invalidConfiguration` unless the config describes
    ///   one MTP layer.
    public init(_ configuration: Qwen4ExpConfiguration) throws {
        try configuration.validateModelConfiguration()
        let text = configuration.text
        guard text.mtpNumHiddenLayers == 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp MTP: mtp_num_hidden_layers must be 1, got \(text.mtpNumHiddenLayers)")
        }
        self.configuration = text
        _mtp.wrappedValue = Qwen4ExpMTPPredictor(text)
        super.init()
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard let target = target as? Qwen4ExpModel else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Qwen4ExpMTPDraftModel", expected: "a Qwen4ExpModel target",
                actual: String(describing: type(of: target)))
        }
        let other = target.configuration
        guard other.hiddenSize == configuration.hiddenSize,
            other.hcCount == configuration.hcCount,
            other.vocabularySize == configuration.vocabularySize
        else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Qwen4ExpMTPDraftModel",
                expected: "hidden size \(configuration.hiddenSize), hc_count "
                    + "\(configuration.hcCount) and vocabulary size \(configuration.vocabularySize)",
                actual: "hidden size \(other.hiddenSize), hc_count \(other.hcCount) and "
                    + "vocabulary size \(other.vocabularySize)")
        }
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    /// Run the head over `tokens` and the streams before them, then sample the seed of the
    /// next round from the last position.
    ///
    /// A long prompt runs in chunks: the indexer scores each query against every past key.
    private func extend(
        _ target: any LanguageModel, tokens: MLXArray, hyper: MLXArray,
        state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        let target = qwen4ExpTarget(target)
        let count = tokens.dim(1)
        let (mixed, next) = mtpForwardInChunks(count: count, cache: state.cache) { range in
            let ids = tokens[0..., range]
            return mtp(
                embeddings: target.embedTokens(ids), hyper: hyper[0..., range, 0...], ids: ids,
                cache: state.cache)
        }
        state.nextPosition += count
        state.seedHidden = next[0..., (-1)..., 0...]
        let logits = target.project(mixed[0..., (-1)..., 0...])
        state.seedToken = normalizedMTPColumn(sampler.sample(logits: logits[0..., -1, 0...]))
        state.proposalAppended = 0
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let prompt = normalizedMTPTokenBatch(promptTokens)
        guard prompt.dim(-1) > 0 else { return }
        let shifted = concatenated([prompt[0..., 1...], normalizedMTPColumn(firstBonus)], axis: 1)
        state.nextPosition = 0
        extend(
            target, tokens: shifted, hyper: targetHidden[0..., ..<shifted.dim(1), 0...],
            state: &state, sampler: sampler)
    }

    public func finalizeDrafterState(
        target _: any LanguageModel,
        targetBoundaryHidden _: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens: Int,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool {
        finalizeShiftedMTPDrafterState(
            targetProcessedTokenCount: targetProcessedTokenCount,
            discardedTargetTokens: discardedTargetTokens,
            state: &state)
    }

    public func resumeDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        targetBoundaryHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool {
        let suffix = normalizedMTPTokenBatch(suffixTokens)
        let suffixLength = suffix.dim(-1)
        guard suffixLength > 0,
            suffixTargetHidden.dim(1) >= suffixLength,
            targetBoundaryHidden.dim(1) == 1,
            state.proposalAppended == 0,
            mtpDrafterCacheIsAligned(state)
        else { return false }

        let tokens = concatenated([suffix, normalizedMTPColumn(firstBonus)], axis: 1)
        let hyper = concatenated(
            [targetBoundaryHidden, suffixTargetHidden[0..., ..<suffixLength, 0...]], axis: 1)
        extend(target, tokens: tokens, hyper: hyper, state: &state, sampler: sampler)
        return mtpDrafterCacheIsAligned(state)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        precondition(blockSize >= 2, "blockSize must be >= 2")
        if let seed = state.seedToken {
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            return seed
        }

        // RoPE positions come from the head's cache offset.
        let target = qwen4ExpTarget(target)
        var token = normalizedMTPColumn(lastToken)
        var hyper = lastHidden
        var tokens: [MLXArray] = []
        for _ in 0 ..< (blockSize - 1) {
            let (mixed, next) = mtp(
                embeddings: target.embedTokens(token), hyper: hyper, ids: token,
                cache: state.cache)
            hyper = next
            token = normalizedMTPColumn(
                sampler.sample(logits: target.project(mixed)[0..., -1, 0...]))
            tokens.append(token)
        }
        state.proposalAppended = blockSize - 1
        state.nextPosition += blockSize - 1
        return concatenated(tokens, axis: 1)
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }

        // Each verified token not yet in the head's cache, paired with the target
        // streams at the position before it.
        var tokens: [MLXArray] = []
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        let hyper = targetHidden[0..., keepAppended ..< (acceptedCount + 1), 0...]
        extend(
            target, tokens: concatenated(tokens, axis: 1), hyper: hyper, state: &state,
            sampler: sampler)
    }

    private func qwen4ExpTarget(_ target: any LanguageModel) -> Qwen4ExpModel {
        guard let target = target as? Qwen4ExpModel else {
            fatalError(
                "Qwen4ExpMTPDraftModel requires a Qwen4ExpModel target, got \(type(of: target))")
        }
        return target
    }

    // MARK: - Checkpoint

    /// The target's tensors in a full checkpoint, which the loader then never reads.
    ///
    /// `model` itself stays readable: the head may load from `model.mtp.`.
    public var excludedCheckpointNamespaces: [String] {
        [
            "language_model", "model.language_model", "model.embed_tokens", "model.layers",
            "model.hyper_connection_mixer", "lm_head", "model.visual", "visual", "vision_tower",
        ]
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    /// Keep the `mtp.` tensors: a full checkpoint also holds the target's.
    ///
    /// The published head stores its norms as `w` for a `1 + w` scale, as Hugging Face
    /// does. A file whose metadata declares the `scale` convention already holds `1 + w`.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames { name in
            if name.hasPrefix("mtp.") { return name }
            if name.hasPrefix("model.mtp.") { return String(name.dropFirst("model.".count)) }
            return nil
        }
        var weights = checkpoint.weights
        for key in weights.keys.sorted() {
            let value = weights[key]!
            if Qwen4ExpModel.splitExperts(key, value, into: &weights) {
                continue
            }
            let declared = checkpoint.metadata(forWeight: key)[Self.normMetadataKey]
            if declared != "scale", Self.foldOne.contains(where: { key.hasSuffix($0) }) {
                weights[key] = value + MLXArray(1, dtype: value.dtype)
            }
        }
        try Qwen4ExpModel.mergeBlockInjects(&weights, checkpoint: &checkpoint)
        checkpoint.weights = weights
        return checkpoint
    }

    static let normMetadataKey = Qwen35CheckpointPolicy.normMetadataKey

    private static let foldOne =
        Qwen4ExpModel.foldOne + ["pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"]
}

extension Qwen4ExpMTPDraftModel: ModelConversionMetadataProvider {
    public var modelConversionMetadata: [String: String] {
        [Self.normMetadataKey: "scale"]
    }
}
