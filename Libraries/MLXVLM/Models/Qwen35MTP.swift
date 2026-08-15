// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35VLMNextNPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35Language.DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35Configuration.TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize)
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35Language.DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func newCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        hiddenStates previousHidden: MLXArray,
        cache: [KVCache],
        positionOffset: Int,
        positionDeltas: MLXArray?
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        let positionIds = qwen35MTPPositionIds(
            offset: positionOffset, length: hiddenStates.dim(1),
            batchSize: hiddenStates.dim(0), positionDeltas: positionDeltas)
        for (layer, layerCache) in zip(layers, cache) {
            let maskMode = createAttentionMask(
                h: hiddenStates, cache: layerCache, returnArray: true)
            let attentionMask: MLXArray?
            if case .array(let arrayMask) = maskMode {
                attentionMask = arrayMask
            } else {
                attentionMask = nil
            }
            hiddenStates = layer(
                hiddenStates,
                attentionMask: attentionMask,
                ssmMask: nil,
                cache: layerCache,
                positionIds: positionIds)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35VLMNextNDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Qwen35Configuration.TextConfiguration
    public let maximumBlockSize: Int? = 2
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    private let preconvertedNorms: Bool

    @ModuleInfo(key: "mtp") var mtp: Qwen35VLMNextNPredictor

    public init(
        _ configuration: Qwen35Configuration.TextConfiguration,
        preconvertedNorms: Bool = false
    ) {
        self.configuration = configuration
        self.preconvertedNorms = preconvertedNorms
        _mtp.wrappedValue = Qwen35VLMNextNPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false
    ) {
        self.init(
            configuration.textConfiguration,
            preconvertedNorms: preconvertedNorms)
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard target is Qwen35 else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Qwen35VLMNextNDraftModel",
                expected: "a Qwen3.5 VLM target",
                actual: String(describing: type(of: target))
            )
        }
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
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
        guard let target = target as? Qwen35 else {
            fatalError("Qwen35VLMNextNDraftModel requires a Qwen35 VLM target")
        }
        let targetEmbedTokens = target.languageModel.model.embedTokens
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let prompt = normalizedMTPTokenBatch(promptTokens)
        let bonus = normalizedMTPColumn(firstBonus)
        guard prompt.dim(-1) > 0 else { return }

        let shifted = concatenated([prompt[0..., 1...], bonus], axis: 1)
        let hidden = targetHidden[0..., ..<shifted.dim(1), 0...]
        state.nextPosition = 0
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(shifted), hiddenStates: hidden,
            cache: state.cache, positionOffset: 0, positionDeltas: nil)
        state.nextPosition = shifted.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: target.languageModel.lmHead, sampler: sampler)
        state.proposalAppended = 0
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
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        guard let target = target as? Qwen35 else {
            fatalError(
                "Qwen35VLMNextNDraftModel requires a Qwen35 VLM target, got \(type(of: target))")
        }

        let targetEmbedTokens = target.languageModel.model.embedTokens
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let lmHead = target.languageModel.lmHead

        if let seed = state.seedToken {
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            return seed
        }

        state.proposalAppended = blockSize - 1
        let proposed = draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: lastToken,
            lastHidden: lastHidden,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            cache: state.cache
        ) { inputsEmbeds, hiddenStates, cache, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                positionOffset: positionOffset,
                positionDeltas: positionDeltas)
        }
        state.nextPosition += state.proposalAppended
        return proposed
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        guard let target = target as? Qwen35 else {
            fatalError("Qwen35VLMNextNDraftModel requires a Qwen35 VLM target")
        }
        let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
            hiddens.append(targetHidden[0..., index ..< (index + 1), 0...])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        hiddens.append(targetHidden[0..., acceptedCount ..< (acceptedCount + 1), 0...])

        let targetEmbedTokens = target.languageModel.model.embedTokens
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let committedTokens = concatenated(tokens, axis: 1)
        let committedHidden = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(committedTokens), hiddenStates: committedHidden,
            cache: state.cache, positionOffset: state.nextPosition,
            positionDeltas: positionDeltas)
        state.nextPosition += committedTokens.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: target.languageModel.lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        qwenMTPSanitizeWeights(
            weights: weights,
            mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts,
            shiftNormWeights: !preconvertedNorms
        )
    }
}

func qwen35MTPPositionIds(
    offset: Int,
    length: Int = 1,
    batchSize: Int,
    positionDeltas: MLXArray?
) -> MLXArray {
    var base = MLXArray(0 ..< length).asType(.int32)[.newAxis, 0...] + Int32(offset)
    base = broadcast(base, to: [batchSize, length])
    if var delta = positionDeltas {
        delta = delta.asType(.int32)
        if delta.ndim == 0 {
            delta = broadcast(delta, to: [batchSize])
        } else if delta.dim(0) < batchSize {
            let repeatCount = (batchSize + delta.dim(0) - 1) / delta.dim(0)
            delta = tiled(delta, repetitions: [repeatCount])
        }
        if delta.dim(0) > batchSize {
            delta = delta[0 ..< batchSize]
        }
        base = base + delta[0..., .newAxis]
    }
    return tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
}
