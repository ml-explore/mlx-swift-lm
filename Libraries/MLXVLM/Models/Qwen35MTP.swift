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
        cache: KVCache?,
        stepIndex: Int,
        positionOffset: Int,
        positionDeltas: MLXArray?
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        let maskMode = createAttentionMask(
            h: hiddenStates, cache: cache, returnArray: true)
        let attentionMask: MLXArray?
        if case .array(let arrayMask) = maskMode {
            attentionMask = arrayMask
        } else {
            attentionMask = nil
        }

        let layer = layers[stepIndex % layers.count]
        hiddenStates = layer(
            hiddenStates,
            attentionMask: attentionMask,
            ssmMask: nil,
            cache: cache,
            positionIds: qwen35MTPPositionIds(
                offset: positionOffset, batchSize: hiddenStates.dim(0),
                positionDeltas: positionDeltas))

        return norm(hiddenStates)
    }
}

public final class Qwen35VLMNextNDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Qwen35Configuration.TextConfiguration

    @ModuleInfo(key: "mtp") var mtp: Qwen35VLMNextNPredictor

    public init(_ configuration: Qwen35Configuration.TextConfiguration) {
        self.configuration = configuration
        _mtp.wrappedValue = Qwen35VLMNextNPredictor(configuration)
        super.init()
    }

    public convenience init(_ configuration: Qwen35Configuration) {
        self.init(configuration.textConfiguration)
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
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
        return draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: lastToken,
            lastHidden: lastHidden,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            cache: state.cache
        ) { inputsEmbeds, hiddenStates, cache, stepIndex, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                stepIndex: stepIndex,
                positionOffset: positionOffset,
                positionDeltas: positionDeltas)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        qwenMTPSanitizeWeights(
            weights: weights,
            mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts
        )
    }
}

func qwen35MTPPositionIds(
    offset: Int,
    batchSize: Int,
    positionDeltas: MLXArray?
) -> MLXArray {
    var base = MLXArray(Array(repeating: Int32(offset), count: batchSize), [batchSize, 1])
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
