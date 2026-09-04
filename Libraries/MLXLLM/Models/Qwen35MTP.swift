// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35MTPPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1
        let layerTypes = Array(
            repeating: HybridAttentionSchedule.fullAttention, count: mtpArgs.hiddenLayers)
        mtpArgs.layerTypes = layerTypes
        mtpArgs.resolvedLayerTypes = layerTypes

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
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
        positionOffset: Int
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            let faMask = createAttentionMask(h: hiddenStates, cache: layerCache)
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: nil,
                cache: layerCache,
                positionOffset: positionOffset)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Qwen35TextConfiguration
    public let maximumBlockSize: Int? = 2
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    private let preconvertedNorms: Bool

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public init(
        _ configuration: Qwen35TextConfiguration,
        preconvertedNorms: Bool = false
    ) {
        self.configuration = configuration
        self.preconvertedNorms = preconvertedNorms
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false
    ) {
        self.init(configuration.textConfig, preconvertedNorms: preconvertedNorms)
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        target is Qwen35Model || target is Qwen35TextModel
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
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let prompt = normalizedMTPTokenBatch(promptTokens)
        let bonus = normalizedMTPColumn(firstBonus)
        guard prompt.dim(-1) > 0 else { return }

        let shifted = concatenated([prompt[0..., 1...], bonus], axis: 1)
        let hidden = targetHidden[0..., ..<shifted.dim(1), 0...]
        state.nextPosition = 0
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(shifted),
            hiddenStates: hidden,
            cache: state.cache,
            positionOffset: 0)
        state.nextPosition = shifted.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
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
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens

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
                positionOffset: positionOffset)
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

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
            hiddens.append(targetHidden[0..., index ..< (index + 1), 0...])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        hiddens.append(targetHidden[0..., acceptedCount ..< (acceptedCount + 1), 0...])

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let committedTokens = concatenated(tokens, axis: 1)
        let committedHidden = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(committedTokens),
            hiddenStates: committedHidden,
            cache: state.cache,
            positionOffset: state.nextPosition)
        state.nextPosition += committedTokens.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
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

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        if let model = target as? Qwen35Model {
            return (model.languageModel.model.embedTokens, model.languageModel.lmHead)
        }
        if let model = target as? Qwen35TextModel {
            return (model.model.embedTokens, model.lmHead)
        }
        fatalError(
            "Qwen35MTPDraftModel requires a Qwen35 target, got \(type(of: target))")
    }
}
