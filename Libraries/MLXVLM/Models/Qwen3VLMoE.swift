// Copyright © 2026 Apple Inc.
//
// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_vl_moe

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen3VLMoEError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

private let qwen3VLMoERopeDeltasKey = LMOutput.Key<MLXArray>("qwen3VLMoE.ropeDeltas")

public struct Qwen3VLMoEConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String
        public var numHiddenLayers: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numAttentionHeads: Int
        public var numExperts: Int
        public var numExpertsPerTok: Int
        public var decoderSparseStep: Int
        public var mlpOnlyLayers: [Int]
        public var moeIntermediateSize: Int
        public var rmsNormEps: Double
        public var vocabSize: Int
        private var _numKeyValueHeads: Int?
        public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
        public var headDim: Int
        public var ropeTheta: Double
        public var maxPositionEmbeddings: Int
        private var _ropeScaling: Qwen3VLConfiguration.RoPEScaling?
        public var ropeScaling: Qwen3VLConfiguration.RoPEScaling? { _ropeScaling }
        public var normTopKProb: Bool
        public var tieWordEmbeddings: Bool
        public var attentionBias: Bool
        public var hiddenAct: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case mlpOnlyLayers = "mlp_only_layers"
            case moeIntermediateSize = "moe_intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case vocabSize = "vocab_size"
            case _numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case _ropeScaling = "rope_scaling"
            case normTopKProb = "norm_topk_prob"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case hiddenAct = "hidden_act"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 0
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 0
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
            self.numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 1
            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.mlpOnlyLayers =
                try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.rmsNormEps =
                try container.decodeIfPresent(Double.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
            self._numKeyValueHeads = try container.decodeIfPresent(
                Int.self, forKey: ._numKeyValueHeads)
            self.headDim =
                try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? max(1, hiddenSize / max(1, numAttentionHeads))
            self.ropeTheta =
                try container.decodeIfPresent(Double.self, forKey: .ropeTheta) ?? 1_000_000
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32_768
            self._ropeScaling = try container.decodeIfPresent(
                Qwen3VLConfiguration.RoPEScaling.self, forKey: ._ropeScaling)
            self.normTopKProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.hiddenAct =
                try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 151_655 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 151_656 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 151_652 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 151_653 }
    private let _visionTokenId: Int?
    public var visionTokenId: Int { _visionTokenId ?? 151_654 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabSize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _visionTokenId = "vision_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

extension Qwen3VLMoE {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

enum Qwen3VLMoELanguage {

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: Qwen3VLLanguage.RotaryEmbedding

        init(_ config: Qwen3VLMoEConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = Qwen3VLLanguage.RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSequenceLength += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return wo(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        init(_ config: Qwen3VLMoEConfiguration.TextConfiguration) {
            self.normTopkProb = config.normTopKProb
            self.topK = config.numExpertsPerTok

            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let kth = gates.dim(-1) - topK
            let indices = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = MLX.takeAlong(gates, indices, axis: -1)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }

            let expertOutput = switchMLP(x, indices)
            return (expertOutput * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: Module

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLMoEConfiguration.TextConfiguration, layerIdx: Int) {
            _attention.wrappedValue = Attention(config)

            let isSparse =
                !config.mlpOnlyLayers.contains(layerIdx)
                && config.numExperts > 0
                && (layerIdx + 1) % config.decoderSparseStep == 0
            if isSparse {
                _mlp.wrappedValue = SparseMoeBlock(config)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: config.hiddenSize,
                    hiddenDimensions: config.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = (mlp as! UnaryLayer)(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLMoEConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
                DecoderLayer(config, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count, let visualMask {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let indices = maskIndices(visualMask)
            guard !indices.isEmpty else { return hiddenStates }

            let indexArray = MLXArray(indices.map { UInt32($0) })
            let result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds
            return result
        }

        private func maskIndices(_ mask: MLXArray) -> [Int] {
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            return indices
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLMoEConfiguration
        let textConfig: Qwen3VLMoEConfiguration.TextConfiguration
        var kvHeads: [Int]

        init(_ config: Qwen3VLMoEConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            state: LMOutput.State?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            var state = state ?? .init()

            if pixelValues != nil {
                state[qwen3VLMoERopeDeltasKey] = nil
            }

            var positionIds = providedPositionIds
            let inputOrEmbeddings = inputIds ?? inputEmbeddings!

            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                if (cache?.first?.offset ?? 0) == 0 || state[qwen3VLMoERopeDeltasKey] == nil
                    || cache == nil
                {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)

                        positionIds = computed
                        state[qwen3VLMoERopeDeltasKey] = deltas
                    } else if let cache, state[qwen3VLMoERopeDeltasKey] == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        base = base + MLXArray(currentOffset).asType(.int32)

                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = broadcast(positionIds!, to: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas = state[qwen3VLMoERopeDeltasKey] {
                    let batch = inputOrEmbeddings.dim(0)
                    let seqLength = inputOrEmbeddings.dim(1)
                    let lastCacheOffset = cache.last?.offset ?? 0

                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batch, seqLength])

                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }

                    base = base + delta
                    positionIds = broadcast(base[.newAxis, 0..., 0...], to: [3, batch, seqLength])
                }
            }

            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output, state: state)
        }
    }
}

public final class Qwen3VLMoE: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3VLMoELanguage.LanguageModel

    public let config: Qwen3VLMoEConfiguration

    public init(_ config: Qwen3VLMoEConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLMoELanguage.LanguageModel(config)
    }

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen3VLMoEError.featureTokenMismatch(
                expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)

        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if let pixelValues,
            let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = framesList.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let slices = layerFeatures.split(indices: splitIndices)
                    return concatenated(slices).asType(textEmbeds.dtype)
                }
            }
        }

        let typedCache = castCache(cache)

        let languageOutput = languageModel(
            inputIds,
            cache: typedCache,
            state: nil,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames)

        return .logits(languageOutput)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let typedCache = castCacheOptional(cache)

        return languageModel(
            input.tokens,
            cache: typedCache,
            state: state,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let adjusted = Self.sanitizeLanguageWeights(
            weights: weights,
            tieWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

        return visionModel.sanitize(weights: adjusted)
    }

    private static func sanitizeLanguageWeights(
        weights: [String: MLXArray],
        tieWordEmbeddings: Bool
    ) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = key

            if newKey.contains("model.visual") {
                newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if newKey.contains("model.language_model") {
                newKey = newKey.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if newKey.hasPrefix("model.") {
                newKey = "language_model.\(newKey)"
            } else if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if tieWordEmbeddings && newKey.contains(".lm_head.") {
                continue
            }

            if remapExpertWeight(key: newKey, value: value, into: &adjusted) {
                continue
            }

            adjusted[newKey] = value
        }

        return adjusted
    }

    private static func remapExpertWeight(
        key: String,
        value: MLXArray,
        into adjusted: inout [String: MLXArray]
    ) -> Bool {
        if let range = key.range(of: ".experts.gate_up_proj") {
            let prefix = String(key[..<range.lowerBound])
            let base = "\(prefix).switch_mlp"
            let mid = value.dim(-1) / 2
            var gate = value[.ellipsis, ..<mid]
            var up = value[.ellipsis, mid...]
            if gate.ndim == 3 {
                gate = gate.transposed(0, 2, 1)
                up = up.transposed(0, 2, 1)
            }
            adjusted["\(base).gate_proj.weight"] = gate
            adjusted["\(base).up_proj.weight"] = up
            return true
        }

        for name in ["gate_proj", "up_proj", "down_proj"] {
            if let range = key.range(of: ".experts.\(name)") {
                let prefix = String(key[..<range.lowerBound])
                var mapped = value
                if name == "down_proj", mapped.ndim == 3 {
                    mapped = mapped.transposed(0, 2, 1)
                }
                adjusted["\(prefix).switch_mlp.\(name).weight"] = mapped
                return true
            }
        }

        return false
    }
}
