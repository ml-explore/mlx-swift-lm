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
        cache: KVCache?,
        stepIndex: Int,
        positionOffset: Int
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        let faMask = createAttentionMask(h: hiddenStates, cache: cache)
        let layer = layers[stepIndex % layers.count]

        hiddenStates = layer(
            hiddenStates,
            attentionMask: faMask,
            ssmMask: nil,
            cache: cache,
            positionOffset: positionOffset)

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, MTPDrafterModel {
    public let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public init(_ configuration: Qwen35TextConfiguration) {
        self.configuration = configuration
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(_ configuration: Qwen35Configuration) {
        self.init(configuration.textConfig)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        return draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: lastToken,
            lastHidden: lastHidden,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            newCache: mtp.newCache
        ) { inputsEmbeds, hiddenStates, cache, stepIndex, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                stepIndex: stepIndex,
                positionOffset: positionOffset)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        qwenMTPSanitizeWeights(
            weights: weights,
            mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts
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
