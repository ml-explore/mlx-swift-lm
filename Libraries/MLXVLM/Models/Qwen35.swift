//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

private let precomputedPositionIdsKey = LMOutput.Key<MLXArray>(
    "qwen35.precomputedPositionIds")
private let ropeDeltasKey = LMOutput.Key<MLXArray>(
    "qwen35.ropeDeltas")

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
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
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)
            self.mropeSection =
                mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqsT = freqs[0, 0..., 0..., 0...]
            let dims = freqsT.dim(-1)
            var slices: [MLXArray] = []
            slices.reserveCapacity(dims)

            for idx in 0 ..< dims {
                var slice = freqsT[0..., 0..., idx]
                for (dim, offset) in [(1, 1), (2, 2)] {
                    let length = min(mropeSection[dim] * 3, dims)
                    if idx >= offset && idx < length && ((idx - offset) % 3 == 0) {
                        slice = freqs[dim, 0..., 0..., idx]
                        break
                    }
                }
                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                attentionMask = .array(mask[.ellipsis, 0 ..< kvSeqLen])
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
            .reshaped(B, L, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            var mixedQKV = inProjQKV(inputs)
            let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
            let b = inProjB(inputs)
            let a = inProjA(inputs)

            let convState: MLXArray
            if let cacheState = cache?[0] {
                convState = cacheState
            } else {
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            var out: MLXArray
            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )

            if let cache {
                cache[1] = state
                cache.advance(S)
            }

            out = norm(out, gate: z)
            return outProj(out.reshaped(B, S, -1))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let kth = gates.dim(-1) - topK
            let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = MLX.takeAlong(gates, inds, axis: -1)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores)

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration, layerIdx: Int) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds
                )
            }

            return norm(hiddenStates)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            state: LMOutput.State?,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil
        ) -> LMOutput {
            var state = state ?? .init()

            // Ensure inputs is 2D [batch, seq]. Text-only callers (e.g.
            // WiredMemoryUtils, TokenIterator) may pass 1D token arrays.
            let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs

            if pixelValues != nil {
                state[precomputedPositionIdsKey] = nil
                state[ropeDeltasKey] = nil
            }
            let precomputedPositionIds = state[precomputedPositionIdsKey]
            let ropeDeltas = state[ropeDeltasKey]

            var cacheOffset = 0
            if let cache, let faCache = cache[model.faIdx] {
                cacheOffset = faCache.offset
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                if (cache != nil && cache?[model.faIdx] != nil && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    if let precomputedPositionIds {
                        let seqLength = inputs.dim(1)
                        positionIds =
                            precomputedPositionIds[
                                0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                    } else {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        state[precomputedPositionIdsKey] = computed
                        state[ropeDeltasKey] = deltas
                    }
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    } else if delta.dim(0) < batchSize {
                        delta = repeated(delta, count: batchSize, axis: 0)
                    } else if delta.dim(0) > batchSize {
                        delta = delta[0 ..< batchSize]
                    }

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            var out = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds
            )

            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }

            return LMOutput(logits: out, state: state)
        }

        func makeCache(maxKVSize: Int?) -> [KVCache] {
            model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                if let maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Model

public class Qwen35: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Qwen35Language.LanguageModel

    public let config: Qwen35Configuration

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
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
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
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

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    /// Vision tower + image→token merge over the input, shared by `prepare`
    /// and `prepareContinuation` (mirrors `Qwen25VL.inputEmbeddings`).
    private func visionInputEmbeddings(
        _ input: LMInput
    ) throws -> (
        pixelValues: MLXArray?, imageFrames: [THW]?, videoFrames: [THW]?,
        inputEmbeddings: MLXArray?
    ) {
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?

        if let pixelValues,
            let frames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let inputIds = input.text.tokens
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValues, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        }

        return (pixelValues, imageFrames, videoFrames, inputEmbeddings)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        windowSize: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        // Windowed (chunked) prefill — the remaining #344 deferred item for
        // Qwen3.5 — with the same default as the sibling chunked prefills
        // (Gemma3/LLMModel: `windowSize ?? 512`). The windowed forward also
        // owns every warm continuation (multi-turn chat, tool restart,
        // restored prompt cache): a cold cache is just a continuation
        // anchored at offset 0, and a warm one anchors M-RoPE positions at
        // the cache offset plus the rope delta carried in `state` — never
        // back at zero. The windowed forward is single-sequence; batched
        // inputs keep the single-shot path below.
        let window = windowSize ?? 512
        if inputIds.ndim == 2, inputIds.dim(0) == 1, inputIds.dim(-1) > 0,
            faCacheOffset(cache) > 0 || inputIds.dim(-1) > window
        {
            return try prepareContinuation(
                input, cache: cache, state: state, windowSize: window)
        }

        let (pixelValues, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        let typedCache = castCache(cache)
        let output = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            languageModel(
                inputIds,
                inputsEmbeds: inputEmbeddings,
                cache: typedCache,
                state: state,
                mask: input.text.mask,
                positionIds: nil,
                pixelValues: pixelValues,
                imageGridTHW: imageFrames,
                videoGridTHW: videoFrames
            )
        }

        return .logits(output)
    }

    /// Offset of the first full-attention layer's cache — the model's notion
    /// of "how many tokens are already cached".
    private func faCacheOffset(_ cache: [any KVCache]) -> Int {
        let faIdx = languageModel.model.faIdx
        return cache.indices.contains(faIdx) ? cache[faIdx].offset : 0
    }

    /// Warm, windowed continuation through an image-bearing remainder — the
    /// windowed forward that `prepare` also delegates to for long prompts.
    ///
    /// It runs the vision tower and the image→token merge **once** over the
    /// remainder, computes the new image's M-RoPE positions **once** from the
    /// seeded **Position Anchor** (so the image's diverging t/h/w indices
    /// start at the anchor, not zero), then drives the language-model forward
    /// in chunks of `windowSize`. The full-attention scratch is bounded to
    /// `[heads, chunk, L]` instead of `[heads, L, L]`, so it cannot crash on
    /// a long prefix or a large image.
    ///
    /// `cache` must already be warmed to the restore offset `P` (a fresh cache
    /// means `P = 0`, a crash-safe cold prefill). `state` carries the anchor's
    /// rope delta in the `qwen35.ropeDeltas` slot (zero / absent for an
    /// image-free prefix). The returned state carries the rope delta the
    /// post-image text tail resumes with (`getRopeIndex` delta − `P`), so a
    /// caller threading state end-to-end continues that tail with the ordinary
    /// flat-continuation branch. Decode stays caller-owned.
    private func prepareContinuation(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        windowSize: Int
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let remainderLength = inputIds.dim(-1)
        precondition(remainderLength > 0, "prepareContinuation needs a non-empty remainder")

        // The Position Anchor: token offset already in the cache (P) plus the
        // rope delta the cached images accumulated, carried in `state`.
        let cacheOffset = faCacheOffset(cache)
        var anchorRopeDelta = 0
        if let seeded = state?[ropeDeltasKey] {
            anchorRopeDelta = seeded.asType(.int32).item(Int.self)
        }
        let positionOffset = cacheOffset + anchorRopeDelta

        // Vision tower + image→token merge — once, over the whole remainder;
        // chunks slice the merged embeddings below.
        let (_, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        // Offset-aware M-RoPE positions for the whole remainder — once. The new
        // image's t/h/w indices diverge from the anchor; the returned delta is
        // in the same offset frame.
        let (positionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: input.text.mask,
            positionOffset: positionOffset
        )

        // Chunk the forward. Each window forwards `chunk` query tokens against
        // the growing cache, so the full-attention scratch stays `[heads,
        // chunk, L]`. Intermediate logits are dropped un-evaluated (only the
        // cache is realized between windows), so `lm_head` never materializes a
        // `[1, L, vocab]` tensor. `asyncEval` bounds the un-evaluated graph
        // while letting the GPU run window i as the CPU builds window i+1
        // (same shape as the sibling chunked prefills, e.g. Gemma3).
        let typedCache = castCache(cache)
        let step = max(1, windowSize)
        var lastLogits: MLXArray
        var start = 0
        repeat {
            try Task.checkCancellation()
            let end = min(start + step, remainderLength)
            let chunkInputs = inputIds[0..., start ..< end]
            let chunkEmbeds = inputEmbeddings.map { $0[0..., start ..< end, 0...] }
            let chunkPositions = positionIds[0..., 0..., start ..< end]
            let output = languageModel(
                chunkInputs,
                inputsEmbeds: chunkEmbeds,
                cache: typedCache,
                state: nil,
                mask: nil,
                positionIds: chunkPositions,
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil
            )
            lastLogits = output.logits
            if let typedCache {
                asyncEval(typedCache)
            }
            start = end
        } while start < remainderLength
        if let typedCache {
            eval(typedCache)
        }

        // Seed the post-image text tail's anchor. The vendor's flat-continuation
        // branch positions tail token j at `tailCacheOffset + ropeDeltas + j`;
        // after this remainder `tailCacheOffset = P + remainderLength`, so the
        // delta the tail needs is the offset-frame `getRopeIndex` delta minus
        // `P` (which `getRopeIndex` implicitly counted into `remainderLength`).
        var resumeState = LMOutput.State()
        resumeState[ropeDeltasKey] = ropeDeltas - MLXArray(Int32(cacheOffset))

        return .logits(LMOutput(logits: lastLogits, state: resumeState))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            input.tokens,
            inputsEmbeds: nil,
            cache: typedCache,
            state: state,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            return weights
        }
        return sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Whether the checkpoint stores RMSNorm weights in the raw (un-shifted)
        // convention that needs the `+1` offset. Mirrors the MLXLLM Qwen35 gate
        // so the VLM and text paths agree: a pre-converted MLX checkpoint
        // (conv1d already sanitized, no MTP tensors) has already been shifted at
        // conversion time and must NOT be shifted again — doing so double-shifts
        // every layernorm and produces garbage tokens. Computed on the incoming
        // weights, before the `mtp.` filter below.
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        if config.textConfiguration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if key.hasPrefix("model.") {
                    // Unified Qwen 3.5 checkpoints (e.g. Qwen3.5-0.8B-MLX-4bit) ship
                    // language model tensors at bare `model.*` paths instead of
                    // `model.language_model.*`. Mirror the LLM-side fallback.
                    key = "language_model." + key
                }
            } else if key.contains("lm_head") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1
            {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return visionModel.sanitize(weights: sanitized)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}
