//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

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
        case ropeScaling = "rope_scaling"
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
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
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
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
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

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
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

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
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

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

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

    /// The decode-step GDN body with explicit state in/out — traced by the
    /// enclosing layer's per-layer decode trace or by a whole-step segment
    /// (see `Qwen35TextModelInner.decodeStep`), so every S == 1 decode
    /// reaches this through the layer above. Bit-identical to the unfused
    /// `callAsFunction` path for S == 1 / mask == nil; fusion only merges
    /// elementwise chains.
    func decodeForward(
        x: MLXArray, convState: MLXArray, recState: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)

        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(x)
        let a = inProjA(x)

        let convInput = concatenated([convState, qkv], axis: 1)
        let newConvState = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newRecState) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: recState,
            mask: nil
        )

        let gated = norm(out, gate: z)
        return (outProj(gated.reshaped(B, S, -1)), newConvState, newRecState)
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }

    /// Decode-step projections up to (not including) rope:
    /// `x` → (queries, gate, keys, values).
    ///
    /// Pure and static-shaped — it never touches the cache, and every shape
    /// depends only on the model config, so a trace of this body replays
    /// unchanged for every token. Byte-identical to the same prefix of
    /// `callAsFunction`.
    ///
    /// Rope is deliberately *not* included: its offset moves every token and,
    /// baked into a trace as the scalar it is, a replay would rotate every
    /// token to the position the trace was taken at. The caller applies the
    /// same `fast::RoPE` call with the same scalar offset outside the
    /// compiled region, so the arithmetic is untouched.
    func decodeProjectPreRope(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        return (queries, gate, keys, values)
    }

    /// The tail of the decode step: attention output → head merge → output
    /// gate → output projection. Static-shaped (the SDPA result is
    /// [B, heads, 1, headDim] whatever the cache length), so it traces.
    func decodeOutput(attention: MLXArray, gate: MLXArray) -> MLXArray {
        let merged =
            attention
            .transposed(0, 2, 1, 3)
            .reshaped(attention.dim(0), attention.dim(2), -1)
        return oProj(sigmoidMultiply(merged, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Decode runs through a compiled closure — the router chain
        // (takeAlong/sum/divide), shared-expert gating (sigmoid+multiply)
        // and the residuals fuse into fewer kernels, shortening the GPU
        // serial chain ~10 ops/layer/token. Fusion only merges elementwise
        // chains with per-op output rounding preserved, so the compiled
        // body is bit-identical to the unfused one; non-fusable primitives
        // (matmuls, gathers, custom kernels) tape through unchanged.
        // Traced once at warmup and replayed after.
        // Prefill takes the unfused body: it is GEMM-dominated (fusion
        // measured +0.3% at 8K) and the per-shape compile-trace cost is
        // real on short prompts (−5% at 128 ctx).
        if x.dim(1) != 1 {
            return forward(x)
        }
        if compiledForward == nil {
            // [self]: the closure lives only in `compiledForward` on
            // self, so it cannot outlive self — while a strong capture cycles
            // (self → compiledForward → CompiledFunction → closure → self)
            // and leaks the block, its expert weights, and the compiled mlx
            // tape on every model release. The trace bakes the weights
            // captured at first trace: swapping parameters on a live module
            // would silently replay stale weights — recreate the module
            // instead.
            compiledForward = compile { [unowned self] x in forward(x) }
        }
        return compiledForward!(x)
    }

    private var compiledForward: ((MLXArray) -> MLXArray)?

    /// The block body. Internal because the enclosing layer's decode trace
    /// inlines it directly rather than nesting this module's own compiled
    /// wrapper inside that trace.
    func forward(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
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

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        // The whole decode-step layer — norm → attention → residual → norm →
        // MLP → residual — runs as one traced function per layer instance
        // (two for full-attention layers, split where the KV cache is
        // written). This generalises the block-level compiled closures,
        // which compiled the MoE and GDN bodies but left the norms,
        // residuals and the whole full-attention layer to be rebuilt
        // op-by-op from Swift every token. Fusion only merges elementwise
        // chains, so outputs are bit-identical. Prefill, masked/array-mask
        // calls and cache kinds with their own attention route (quantized,
        // turbo) keep the path below unchanged.
        if x.dim(1) == 1, ssmMask == nil {
            if isLinear, let mambaCache = cache as? MambaCache {
                return decodeLinearLayer(x, cache: mambaCache)
            }
            if !isLinear, let cache, !(cache is QuantizedKVCacheProtocol),
                !(cache is TurboQuantKVCache)
            {
                return decodeAttentionLayer(x, mask: attentionMask, cache: cache)
            }
        }

        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    // MARK: - Compiled decode blocks

    private var compiledLinearLayer: (([MLXArray]) -> [MLXArray])?
    private var compiledAttentionPre: (([MLXArray]) -> [MLXArray])?
    private var compiledAttentionPost: (([MLXArray]) -> [MLXArray])?

    /// GDN (linear-attention) decode layer as a single traced function.
    ///
    /// Conv/recurrent state crosses the boundary explicitly — a compiled
    /// function has to be pure; the first decode token supplies the same
    /// explicit zero states `gatedDeltaUpdate` would have built internally.
    private func decodeLinearLayer(_ x: MLXArray, cache: MambaCache) -> MLXArray {
        let gdn = linearAttn!
        let convState =
            cache[0]
            ?? MLXArray.zeros(
                [x.dim(0), gdn.convKernelSize - 1, gdn.convDim], dtype: x.dtype)
        let recState =
            cache[1]
            ?? MLXArray.zeros(
                [x.dim(0), gdn.numVHeads, gdn.headVDim, gdn.headKDim], dtype: .float32)

        if compiledLinearLayer == nil {
            // [unowned self]: the closure lives only in this property on self,
            // so it cannot outlive self, while a strong capture would cycle
            // (self → property → CompiledFunction → closure → self) and leak
            // the layer, its weights and the compiled tape on every model
            // release. The trace also bakes the weights it
            // captured: swapping parameters on a live module would replay
            // stale weights — recreate the module instead.
            compiledLinearLayer = compile { [unowned self] args in
                let (out, newConvState, newRecState) = linearLayerBody(
                    x: args[0], convState: args[1], recState: args[2])
                return [out, newConvState, newRecState]
            }
        }

        let out = compiledLinearLayer!([x, convState, recState])
        cache[0] = out[1]
        cache[1] = out[2]
        cache.advance(1)
        return out[0]
    }

    /// Full-attention decode layer: two traced functions around the cache
    /// write. Everything traced here has static shapes — only the SDPA over
    /// the grown cache (and the rope whose offset moves) sits between them,
    /// so no shapeless compilation is involved.
    private func decodeAttentionLayer(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache
    ) -> MLXArray {
        let attn = selfAttn!

        if compiledAttentionPre == nil {
            // [unowned self]: see decodeLinearLayer.
            compiledAttentionPre = compile { [unowned self] args in
                let (queries, gate, keys, values) = attentionPreBody(x: args[0])
                return [queries, gate, keys, values]
            }
        }
        if compiledAttentionPost == nil {
            compiledAttentionPost = compile { [unowned self] args in
                [attentionPostBody(x: args[0], attention: args[1], gate: args[2])]
            }
        }

        let projected = compiledAttentionPre!([x])
        let ropeOffset = cache.ropeOffset
        let queries = applyRotaryPosition(attn.rope, to: projected[0], offset: ropeOffset)
        let keys = applyRotaryPosition(attn.rope, to: projected[2], offset: ropeOffset)

        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: projected[3])

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: attn.scale,
            mask: mask
        )

        return compiledAttentionPost!([x, attention, projected[1]])[0]
    }

    // MARK: - Layer bodies (traced by this layer, or by a whole-step segment)

    /// GDN layer body: norm → GDN with explicit state → residual → norm →
    /// MLP → residual.
    func linearLayerBody(x: MLXArray, convState: MLXArray, recState: MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let (r, newConvState, newRecState) = linearAttn!.decodeForward(
            x: inputLayerNorm(x), convState: convState, recState: recState)
        let h = x + r
        return (h + mlpForward(postAttentionLayerNorm(h)), newConvState, newRecState)
    }

    /// Full-attention layer up to the cache write: norm → projections → q/k
    /// norms (rope, the cache write and the SDPA follow outside).
    func attentionPreBody(x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        selfAttn!.decodeProjectPreRope(inputLayerNorm(x))
    }

    /// Full-attention layer from the SDPA result on: head merge → output gate
    /// → output projection → residual → norm → MLP → residual. `x` is this
    /// layer's input (the residual branch).
    func attentionPostBody(x: MLXArray, attention: MLXArray, gate: MLXArray) -> MLXArray {
        let r = selfAttn!.decodeOutput(attention: attention, gate: gate)
        let h = x + r
        return h + mlpForward(postAttentionLayerNorm(h))
    }

    /// The MLP body without its own compiled wrapper — the layer trace above
    /// already covers it, and a nested compile inside a trace would only add
    /// a second tape.
    private func mlpForward(_ x: MLXArray) -> MLXArray {
        if let moe = mlp as? Qwen35SparseMoeBlock {
            return moe.forward(x)
        }
        return (mlp as! UnaryLayer)(x)
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        // The whole decode step runs as a small
        // number of traced segments instead of ~40 per-layer traces plus Swift
        // glue. A segment runs from just after one full-attention layer's SDPA
        // to just before the next one's — the KV write is the only thing that
        // cannot live inside a trace, because the cache grows every token and
        // `cache.update`'s in-place slice_update has to stay outside a
        // compiled region. Everything else on a decode step is static-shaped
        // (GDN state shapes do not depend on context length), so the segments
        // compile concretely — no shapeless tracing is involved.
        if inputs.dim(1) == 1, let caches = cache, let step = decodeStep(inputs, caches) {
            return step
        }

        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
        }

        return norm(hiddenStates)
    }

    // MARK: - Whole-step decode schedule

    /// One traced piece of a decode step.
    ///
    /// The pieces tile the step in order: a segment optionally opens with the
    /// tail of the previous full-attention layer, then runs a run of GDN
    /// layers, then optionally closes with the head of the next
    /// full-attention layer (whose SDPA happens between this segment and the
    /// next one).
    private struct DecodeSegment {
        var opensModel = false
        var attentionPostLayer: Int?
        var linearLayers: [Int] = []
        var attentionPreLayer: Int?
        var closesModel = false
    }

    private var decodeSegments: [DecodeSegment] = []
    private var compiledSegments: [(([MLXArray]) -> [MLXArray])?] = []

    /// Input layout per segment:
    ///   [x] (or the token ids, for the opening segment)
    ///   + [attention, gate] when it opens with a full-attention tail
    ///   + [convState, recState] per GDN layer
    /// Output layout:
    ///   [x]  — the hidden state where the segment stops, which for a segment
    ///          closing with a full-attention head is that layer's *input*
    ///          (the residual branch the next segment needs)
    ///   + [newConvState, newRecState] per GDN layer
    ///   + [queries, gate, keys, values] when it closes with a head
    private func buildDecodeSchedule() {
        var segments: [DecodeSegment] = []
        var current = DecodeSegment(opensModel: true)
        for (i, layer) in layers.enumerated() {
            if layer.isLinear {
                current.linearLayers.append(i)
            } else {
                current.attentionPreLayer = i
                segments.append(current)
                current = DecodeSegment(attentionPostLayer: i)
            }
        }
        current.closesModel = true
        segments.append(current)
        decodeSegments = segments
        compiledSegments = Array(repeating: nil, count: segments.count)
    }

    private func segmentBody(_ segment: DecodeSegment, _ args: [MLXArray]) -> [MLXArray] {
        var hiddenStates = segment.opensModel ? embedTokens(args[0]) : args[0]
        var next = 1

        if let post = segment.attentionPostLayer {
            let attention = args[next]
            let gate = args[next + 1]
            next += 2
            hiddenStates = layers[post].attentionPostBody(
                x: hiddenStates, attention: attention, gate: gate)
        }

        var states: [MLXArray] = []
        for index in segment.linearLayers {
            let (out, newConvState, newRecState) = layers[index].linearLayerBody(
                x: hiddenStates, convState: args[next], recState: args[next + 1])
            next += 2
            hiddenStates = out
            states.append(newConvState)
            states.append(newRecState)
        }

        if let pre = segment.attentionPreLayer {
            let (queries, gate, keys, values) = layers[pre].attentionPreBody(x: hiddenStates)
            // hiddenStates is the attention layer's input: the next segment
            // needs it for the residual around the attention block.
            return [hiddenStates] + states + [queries, gate, keys, values]
        }

        if segment.closesModel {
            hiddenStates = norm(hiddenStates)
        }
        return [hiddenStates] + states
    }

    /// Run one decode step through the compiled segments, or return nil when
    /// this step is not the plain case the schedule is built for (any cache of
    /// an unexpected kind, a full-attention layer that wants a real mask, or a
    /// GDN layer whose SSM mask is set) — the caller then takes the general
    /// path.
    private func decodeStep(_ inputs: MLXArray, _ cache: [KVCache?]) -> MLXArray? {
        guard cache.count == layers.count else { return nil }
        // Same two masks the general path builds, asked of the same caches —
        // the schedule is only valid when both come out empty (the ordinary
        // single-token case).
        if (cache[ssmIdx] as? MambaCache)?.makeMask(N: 1) != nil { return nil }
        guard let faCache = cache[faIdx],
            case .none = faCache.makeMask(n: 1, windowSize: nil, returnArray: false)
        else { return nil }
        for (i, layer) in layers.enumerated() {
            if layer.isLinear {
                // A GDN layer with no state yet (a one-token prompt, i.e. no
                // prefill has run) takes the general path, which builds the
                // explicit zero states — that keeps the dtype question out of
                // here entirely.
                guard let mambaCache = cache[i] as? MambaCache, mambaCache[0] != nil,
                    mambaCache[1] != nil
                else { return nil }
            } else {
                guard let kv = cache[i], !(kv is QuantizedKVCacheProtocol),
                    !(kv is TurboQuantKVCache)
                else { return nil }
            }
        }

        if decodeSegments.isEmpty {
            buildDecodeSchedule()
        }

        var carry = inputs
        var pendingAttention: [MLXArray] = []

        for (segmentIndex, segment) in decodeSegments.enumerated() {
            var args: [MLXArray] = [carry] + pendingAttention
            for index in segment.linearLayers {
                let mambaCache = cache[index] as! MambaCache
                args.append(mambaCache[0]!)
                args.append(mambaCache[1]!)
            }

            if compiledSegments[segmentIndex] == nil {
                // [unowned self]: the closures live only in `compiledSegments`
                // on self, so they cannot outlive self, while a strong capture
                // would cycle and leak the model, its weights and the compiled
                // tapes on every release. The traces bake
                // the weights captured at first trace — recreate the model
                // rather than swapping parameters on a live one.
                compiledSegments[segmentIndex] = compile { [unowned self] args in
                    segmentBody(segment, args)
                }
            }
            let outputs = compiledSegments[segmentIndex]!(args)

            carry = outputs[0]
            var next = 1
            for index in segment.linearLayers {
                let mambaCache = cache[index] as! MambaCache
                mambaCache[0] = outputs[next]
                mambaCache[1] = outputs[next + 1]
                mambaCache.advance(1)
                next += 2
            }

            pendingAttention = []
            if let pre = segment.attentionPreLayer {
                let attn = layers[pre].selfAttn!
                let kvCache = cache[pre]!
                let ropeOffset = kvCache.ropeOffset
                let queries = applyRotaryPosition(attn.rope, to: outputs[next], offset: ropeOffset)
                let gate = outputs[next + 1]
                let keys = applyRotaryPosition(
                    attn.rope, to: outputs[next + 2], offset: ropeOffset)
                let values = outputs[next + 3]

                let (cachedKeys, cachedValues) = kvCache.update(keys: keys, values: values)
                let attention = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: cachedKeys,
                    values: cachedValues,
                    scale: attn.scale,
                    mask: .none
                )
                pendingAttention = [attention, gate]
            }
        }

        return carry
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
