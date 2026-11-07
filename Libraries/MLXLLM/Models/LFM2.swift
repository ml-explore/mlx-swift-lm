//
//  LFM2.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/7/12.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public typealias LFM2Configuration = MLXLMCommon.LFM2Configuration
package typealias LFM2ShortConv = MLXLMCommon.LFM2ShortConv

public class LFM2ModelInner: Module {
    let args: LFM2Configuration
    let vocabularySize: Int
    let numHiddenLayers: Int

    fileprivate let layers: [LFM2DecoderLayer]
    private let firstAttentionIndex: Int?
    private let firstConvIndex: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ args: LFM2Configuration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers

        precondition(vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< numHiddenLayers).map { i in
            LFM2DecoderLayer(args, layerIdx: i)
        }
        self.firstAttentionIndex = args.fullAttnIdxs.first
        self.firstConvIndex = layers.firstIndex(where: { !$0.isAttentionLayer })

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        ssmMask: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        forward(
            inputs,
            attentionMask: attentionMask,
            ssmMask: ssmMask,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            captureLayerIds: []
        ).hidden
    }

    /// Runs the target and optionally captures the output of selected decoder layers.
    /// DSpark checkpoints train against these exact pre-final-norm features.
    func forward(
        _ inputs: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        ssmMask: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
        captureLayerIds: Set<Int>
    ) -> (hidden: MLXArray, captured: [MLXArray]) {
        var h = inputEmbeddings ?? embedTokens(inputs)
        var captured = [MLXArray]()

        let attentionMask =
            attentionMask
            ?? {
                guard let firstAttentionIndex, let cache, firstAttentionIndex < cache.count else {
                    return createAttentionMask(h: h, cache: nil, windowSize: nil)
                }
                return createAttentionMask(h: h, cache: cache[firstAttentionIndex])
            }()

        let ssmMask =
            ssmMask
            ?? {
                guard let firstConvIndex, let cache, firstConvIndex < cache.count else {
                    return nil
                }
                return createSSMMask(h: h, cache: cache[firstConvIndex] as? MambaCache)
            }()

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
            if captureLayerIds.contains(i) {
                captured.append(h)
            }
        }

        return (embeddingNorm(h), captured)
    }
}

public class LFM2Model: Module, LLMModel, KVCacheDimensionProvider,
    SpeculativeCacheRewindModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM2ModelInner
    let configuration: LFM2Configuration
    public let maximumNativeTargetCacheRewind =
        LFM2RuntimeSupport.speculativeRollbackCapacity

    public init(_ args: LFM2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize

        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            args.fullAttnIdxs.contains(layerIdx) ? args.kvHeads : 0
        }

        self.model = LFM2ModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emitDrafterState = state?[mtpEmitFlagKey] ?? false
        let targetLayerIds = emitDrafterState ? (state?[mtpTargetLayerIdsKey] ?? []) : []
        precondition(
            Set(targetLayerIds).count == targetLayerIds.count
                && targetLayerIds.allSatisfy { (0 ..< configuration.hiddenLayers).contains($0) },
            "LFM2 drafter target layer ids must be unique and in range")

        let output = model.forward(
            input.tokens,
            ssmMask: input.mask,
            cache: cache,
            captureLayerIds: Set(targetLayerIds))
        let logits = model.embedTokens.asLinear(output.hidden)

        guard emitDrafterState else { return LMOutput(logits: logits) }

        var outState = state ?? LMOutput.State()
        outState[mtpLastHiddenStatesKey] =
            output.captured.isEmpty
            ? output.hidden : concatenated(output.captured, axis: -1)
        // LFM DSpark consumes hidden features, not target K/V. Keep the
        // generic block-drafter state contract explicit with empty maps.
        outState[mtpSharedKVStatesKey] = [:]
        outState[mtpSharedKVOffsetsKey] = [:]
        outState[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(logits: logits, state: outState)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try LFM2RuntimeSupport.removingLanguageModelScope(from: checkpoint)
        checkpoint.weights = sanitize(weights: checkpoint.weights)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        let shouldStripLanguageModelPrefix =
            weights.keys.contains(where: { $0.hasPrefix("language_model.model.") })
            && !weights.keys.contains(where: { $0.hasPrefix("model.") })

        for (name, param) in weights {
            let name =
                shouldStripLanguageModelPrefix && name.hasPrefix("language_model.")
                ? String(name.dropFirst("language_model.".count))
                : name
            var sanitizedParam = param

            if name.contains("conv.weight") {
                if param.shape[param.shape.count - 1] > param.dim(1) {
                    sanitizedParam = param.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[name] = sanitizedParam
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try LFM2RuntimeSupport.makeHybridCache(
            hiddenLayers: configuration.hiddenLayers,
            fullAttentionIndices: Set(configuration.fullAttnIdxs),
            convolutionStateLength: configuration.convLCache - 1,
            parameters: parameters)
    }
}

extension LFM2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension LFM2Model {
    public var toolCallFormat: ToolCallFormat? { .lfm2 }
}
