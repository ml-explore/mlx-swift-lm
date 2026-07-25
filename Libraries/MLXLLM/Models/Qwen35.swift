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

// MARK: - Fused router top-k

/// One-kernel replacement for the decode router tail: `chainRouterTopK`
/// fully sorts all `E` experts (`ArgPartition::eval_gpu` delegates to
/// `gpu_merge_sort`) just to name `K` — three serial dispatches, three
/// encoder-wide barriers, where barrier-bound decode needs one.
///
/// Bit-identical to the chain by construction: the sort is stable
/// (`sort.h`'s `LessThan` compares values only, ties keep input order), so
/// counting the elements ranked strictly above `i` — with the index packed
/// into the low bits of a monotone bit key as the tie-break — reproduces
/// each winner's slot. `±0.0` normalises to one bit pattern (they compare
/// equal but differ bitwise), NaN maps above `+inf` (all NaNs tie), and the
/// sum accumulates sequentially in the output dtype from zero, in slot
/// order, matching `reduce.metal`'s `thread_reduce`.
private let routerTopKSource = """
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;

    threadgroup ulong sk[E_];
    threadgroup float top_v[K_];

    float v = static_cast<float>(gates[row * E_ + t]);
    uint b = (v == 0.0f) ? 0u : as_type<uint>(v);
    uint mono = isnan(v) ? 0xFFFFFFFFu : (b ^ ((uint)(((int)b) >> 31) | 0x80000000u));
    ulong key = (((ulong)mono) << 32) | (ulong)t;
    sk[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int above = 0;
    for (uint j = 0; j < E_; ++j) {
        above += (sk[j] > key) ? 1 : 0;
    }
    if (above < K_) {
        top_v[K_ - 1 - above] = v;
        inds[row * K_ + (K_ - 1 - above)] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        T acc = static_cast<T>(0);
        for (int q = 0; q < K_; ++q) {
            acc = static_cast<T>(top_v[q]) + acc;
        }
        for (int q = 0; q < K_; ++q) {
            T s = static_cast<T>(top_v[q]);
            scores[row * K_ + q] = NORM_ ? (s / acc) : s;
        }
    }
    """

private final class RouterTopKKernel: Sendable {
    static let shared = RouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "router_topk_norm",
            inputNames: ["gates"],
            outputNames: ["inds", "scores"],
            source: routerTopKSource
        )
    }
}

/// Metal's threads-per-threadgroup ceiling; one thread per expert, so past
/// this the dispatch is invalid, not just slow.
private let maxFusedRouterExperts = 1024

/// Top-`k` + optional normalisation over the last axis in one dispatch:
/// `(indices, scores)` shaped `[..., k]`, bit-identical to
/// `chainRouterTopK`, `uint32` indices included. One threadgroup per row
/// with an `O(E²)` rank count — callers gate this on the single-row decode
/// case. Internal so the bitwise test can reach it.
func fusedRouterTopK(_ gates: MLXArray, k: Int, normalize: Bool) -> (MLXArray, MLXArray) {
    let e = gates.dim(-1)
    let rows = gates.size / e
    let shape = Array(gates.shape.dropLast()) + [k]
    let out = RouterTopKKernel.shared.kernel(
        [gates],
        template: [
            ("T", gates.dtype), ("E_", e), ("K_", k), ("NORM_", normalize ? 1 : 0),
        ],
        grid: (e, rows, 1),
        threadGroup: (e, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, gates.dtype]
    )
    return (out[0], out[1])
}

/// The three-dispatch router tail the fused kernel replaces — the prefill
/// path, and the bitwise test's reference.
func chainRouterTopK(_ gates: MLXArray, k: Int, normalize: Bool) -> (MLXArray, MLXArray) {
    let kth = gates.dim(-1) - k
    let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
    var scores = MLX.takeAlong(gates, inds, axis: -1)
    if normalize {
        scores = scores / scores.sum(axis: -1, keepDims: true)
    }
    return (inds, scores)
}

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
        let convState =
            cache?[0] ?? zeroStates(batch: inputs.dim(0), dtype: inputs.dtype).conv
        let (out, newConvState, newRecState) = forward(
            inputs, convState: convState, recState: cache?[1], mask: mask)
        if let cache {
            cache[0] = newConvState
            cache[1] = newRecState
            cache.advance(inputs.dim(1))
        }
        return out
    }

    /// Zero conv/recurrent state — the shapes `callAsFunction` and
    /// `gatedDeltaUpdate` otherwise build implicitly, made explicit for the
    /// traced decode path.
    func zeroStates(batch: Int, dtype: DType) -> (conv: MLXArray, rec: MLXArray) {
        (
            MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype),
            MLXArray.zeros([batch, numVHeads, headVDim, headKDim], dtype: .float32)
        )
    }

    /// The GDN body with state passed explicitly in and out so it can be
    /// traced.
    func forward(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)

        var qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(x)
        let a = inProjA(x)

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convOut: MLXArray
        let newConvState: MLXArray
        if S == 1 {
            (convOut, newConvState) = decodeConv(convState: convState, qkv: qkv)
        } else {
            let convInput = concatenated([convState, qkv], axis: 1)
            newConvState = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
            convOut = silu(conv1d(convInput))
        }

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
            mask: mask
        )

        let gated = norm(out, gate: z)
        return (outProj(gated.reshaped(B, S, -1)), newConvState, newRecState)
    }

    /// The S == 1 depthwise conv as `convKernelSize` fused multiply-adds —
    /// elementwise ops `compile` folds into the surrounding segment, deleting
    /// a dispatch and a hazard barrier per GDN layer. f32 accumulation with a
    /// single final round reproduces MLX's `Convolution` kernel bit-for-bit
    /// (native-dtype accumulation differs in ~47% of channels);
    /// `Qwen35GDNDecodeBitwiseTests` pins the contract.
    func decodeConv(
        convState: MLXArray, qkv: MLXArray
    ) -> (convOut: MLXArray, newConvState: MLXArray) {
        var acc =
            convState[0..., 0, 0...].asType(.float32)
            * conv1d.weight[0..., 0, 0].asType(.float32)
        for tap in 1 ..< convKernelSize {
            let row =
                tap < convKernelSize - 1
                ? convState[0..., tap, 0...] : qkv[0..., 0, 0...]
            acc = acc + row.asType(.float32) * conv1d.weight[0..., tap, 0].asType(.float32)
        }
        let convOut = silu(acc.asType(qkv.dtype).reshaped(convState.dim(0), 1, convDim))
        return (convOut, concatenated([convState[0..., 1..., 0...], qkv], axis: 1))
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
        let (q, gate, k, values) = projectPreRope(x)

        let offset = cache?.ropeOffset
        let queries = applyRotaryPosition(rope, to: q, offset: offset)
        let keys = applyRotaryPosition(rope, to: k, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        return mergeHeadsAndProject(attention: output, gate: gate)
    }

    /// Projections up to (not including) rope: `x` → (queries, gate, keys,
    /// values). Rope stays outside the traced decode path on purpose: its
    /// offset moves every token, and a trace would bake it in as a constant.
    func projectPreRope(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
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

    /// Attention tail: head merge → output gate → output projection.
    func mergeHeadsAndProject(attention: MLXArray, gate: MLXArray) -> MLXArray {
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
        // Decode (S == 1) runs through a compiled trace: fusion merges the
        // elementwise chains into fewer kernels, bit-identically. Prefill
        // stays unfused — it is GEMM-bound and would pay a trace per shape.
        if x.dim(1) != 1 {
            return forward(x)
        }
        compileLock.lock()
        if compiledForward == nil {
            // [unowned self]: stored only on self, so it cannot outlive self;
            // a strong capture would cycle and leak the module, weights and
            // compiled tape (pinned by Qwen35CompiledDecodeLifecycleTests).
            // The trace bakes the weights it captured — recreate the module
            // rather than swapping parameters on a live one.
            compiledForward = compile { [unowned self] x in forward(x) }
        }
        let fn = compiledForward!
        compileLock.unlock()
        return fn(x)
    }

    /// Compiled functions are created on first decode, not at init (the
    /// weights aren't loaded yet), so the lazy assignment needs a lock.
    private let compileLock = NSLock()
    private var compiledForward: ((MLXArray) -> MLXArray)?

    /// The uncompiled body; an enclosing layer trace inlines it rather than
    /// nesting this block's own compiled wrapper.
    func forward(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let (inds, scores) = routerTopK(gates, k: topK)

        let y = switchMLP(x, inds)
        let combined = weightedExpertSum(y, scores)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }

    /// Decode (one row): the fused kernel, one dispatch instead of three
    /// barriers, bit-identical. Prefill: the chain — many rows make the
    /// `O(E²)` rank count the wrong shape, and there is no barrier to save.
    private func routerTopK(_ gates: MLXArray, k: Int) -> (MLXArray, MLXArray) {
        let e = gates.dim(-1)
        if gates.size == e, e <= maxFusedRouterExperts {
            return fusedRouterTopK(gates, k: k, normalize: normTopkProb)
        }
        return chainRouterTopK(gates, k: k, normalize: normTopkProb)
    }
}

// MARK: - Decoder Layer

/// Caches the compiled decode path can drive: their attention is the plain
/// `cache.update` + SDPA route in `attentionWithCacheUpdate`. Quantized and
/// turbo caches have their own routes and take the general path.
private func hasPlainAttentionRoute(_ cache: KVCache) -> Bool {
    !(cache is QuantizedKVCacheProtocol) && !(cache is TurboQuantKVCache)
}

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
        // Single-token unmasked decode runs the layer as one traced function
        // (two for full attention, split at the KV write). Everything else
        // takes the general body below.
        if x.dim(1) == 1, ssmMask == nil {
            if isLinear, let mambaCache = cache as? MambaCache {
                return decodeLinearLayer(x, cache: mambaCache)
            }
            if !isLinear, let cache, hasPlainAttentionRoute(cache) {
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

    // Lock rationale: see Qwen35SparseMoeBlock.compileLock.
    private let compileLock = NSLock()
    private var compiledLinearLayer: (([MLXArray]) -> [MLXArray])?
    private var compiledAttentionPre: (([MLXArray]) -> [MLXArray])?
    private var compiledAttentionPost: (([MLXArray]) -> [MLXArray])?

    /// GDN decode layer as one traced function. A compiled function must be
    /// pure, so conv/recurrent state crosses the boundary explicitly.
    private func decodeLinearLayer(_ x: MLXArray, cache: MambaCache) -> MLXArray {
        let zero = linearAttn!.zeroStates(batch: x.dim(0), dtype: x.dtype)
        let convState = cache[0] ?? zero.conv
        let recState = cache[1] ?? zero.rec

        compileLock.lock()
        if compiledLinearLayer == nil {
            // [unowned self]: see Qwen35SparseMoeBlock.callAsFunction.
            compiledLinearLayer = compile { [unowned self] args in
                let (out, newConvState, newRecState) = linearLayerBody(
                    x: args[0], convState: args[1], recState: args[2])
                return [out, newConvState, newRecState]
            }
        }
        let fn = compiledLinearLayer!
        compileLock.unlock()

        let out = fn([x, convState, recState])
        cache[0] = out[1]
        cache[1] = out[2]
        cache.advance(1)
        return out[0]
    }

    /// Full-attention decode layer: two traced functions around the KV write,
    /// which cannot live inside a trace because the cache grows every token.
    private func decodeAttentionLayer(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache
    ) -> MLXArray {
        compileLock.lock()
        if compiledAttentionPre == nil {
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
        let pre = compiledAttentionPre!
        let post = compiledAttentionPost!
        compileLock.unlock()

        let projected = pre([x])
        let attention = attentionCacheStep(
            queries: projected[0], keys: projected[2], values: projected[3],
            cache: cache, mask: mask)
        return post([x, attention, projected[1]])[0]
    }

    /// The part of a full-attention decode step that cannot be traced: rope
    /// (its offset moves every token), the KV write, and the SDPA over the
    /// grown cache.
    func attentionCacheStep(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        cache: KVCache, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attn = selfAttn!
        let offset = cache.ropeOffset
        return attentionWithCacheUpdate(
            queries: applyRotaryPosition(attn.rope, to: queries, offset: offset),
            keys: applyRotaryPosition(attn.rope, to: keys, offset: offset),
            values: values,
            cache: cache,
            scale: attn.scale,
            mask: mask
        )
    }

    // MARK: - Layer bodies

    func linearLayerBody(x: MLXArray, convState: MLXArray, recState: MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let (r, newConvState, newRecState) = linearAttn!.forward(
            inputLayerNorm(x), convState: convState, recState: recState, mask: nil)
        let h = x + r
        return (h + mlpForward(postAttentionLayerNorm(h)), newConvState, newRecState)
    }

    func attentionPreBody(x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        selfAttn!.projectPreRope(inputLayerNorm(x))
    }

    /// `x` is the layer input — the residual branch around the attention block.
    func attentionPostBody(x: MLXArray, attention: MLXArray, gate: MLXArray) -> MLXArray {
        let r = selfAttn!.mergeHeadsAndProject(attention: attention, gate: gate)
        let h = x + r
        return h + mlpForward(postAttentionLayerNorm(h))
    }

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

        let layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }
        self.layers = layers

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        let segments = Self.decodeSchedule(for: layers)
        self.decodeSegments = segments
        self.compiledSegments = Array(repeating: nil, count: segments.count)

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
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

    /// One traced piece of a decode step: the tail of the previous
    /// full-attention layer, a run of GDN layers, then the head of the next
    /// one (whose SDPA runs between this segment and the next).
    private struct DecodeSegment {
        var attentionPostLayer: Int?
        var linearLayers: [Int] = []
        var attentionPreLayer: Int?

        /// First conv/recurrent state slot in the input list (after `x` and
        /// any [attention, gate] pair).
        var stateInputOffset: Int { attentionPostLayer == nil ? 1 : 3 }
        /// First [queries, gate, keys, values] slot in the output list.
        var attentionOutputOffset: Int { 1 + 2 * linearLayers.count }
    }

    private let decodeSegments: [DecodeSegment]
    // Lock rationale: see Qwen35SparseMoeBlock.compileLock.
    private let compileLock = NSLock()
    private var compiledSegments: [(([MLXArray]) -> [MLXArray])?]

    private static func decodeSchedule(for layers: [Qwen35DecoderLayer]) -> [DecodeSegment] {
        var segments: [DecodeSegment] = []
        var current = DecodeSegment()
        for (i, layer) in layers.enumerated() {
            if layer.isLinear {
                current.linearLayers.append(i)
            } else {
                current.attentionPreLayer = i
                segments.append(current)
                current = DecodeSegment(attentionPostLayer: i)
            }
        }
        segments.append(current)
        return segments
    }

    /// Flat argument/result lists because `compile` takes `[MLXArray]`.
    /// In: `[x]` (token ids for segment 0), then `[attention, gate]` when
    /// opening with a full-attention tail, then `[convState, recState]` per
    /// GDN layer. Out: `[x]`, then `[newConvState, newRecState]` per GDN
    /// layer, then `[queries, gate, keys, values]` when closing with a head.
    private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
        let segment = decodeSegments[index]
        var hiddenStates = index == 0 ? embedTokens(args[0]) : args[0]

        if let post = segment.attentionPostLayer {
            hiddenStates = layers[post].attentionPostBody(
                x: hiddenStates, attention: args[1], gate: args[2])
        }

        var states: [MLXArray] = []
        for (i, layerIndex) in segment.linearLayers.enumerated() {
            let slot = segment.stateInputOffset + 2 * i
            let (out, newConvState, newRecState) = layers[layerIndex].linearLayerBody(
                x: hiddenStates, convState: args[slot], recState: args[slot + 1])
            hiddenStates = out
            states.append(newConvState)
            states.append(newRecState)
        }

        if let pre = segment.attentionPreLayer {
            let (queries, gate, keys, values) = layers[pre].attentionPreBody(x: hiddenStates)
            // The next segment needs the attention layer's input for its residual.
            return [hiddenStates] + states + [queries, gate, keys, values]
        }

        if index == decodeSegments.count - 1 {
            hiddenStates = norm(hiddenStates)
        }
        return [hiddenStates] + states
    }

    /// One decode step through the compiled segments, or nil when this is not
    /// the plain single-token case the schedule assumes — the caller then
    /// takes the general path. Segments split at each KV write: the cache
    /// update's in-place slice_update cannot live inside a trace, and
    /// everything else on a decode step is static-shaped, so the segments
    /// compile concretely.
    private func decodeStep(_ inputs: MLXArray, _ cache: [KVCache?]) -> MLXArray? {
        guard cache.count == layers.count else { return nil }
        // The schedule is only valid when the masks the general path would
        // build both come out empty.
        if createSSMMask(h: inputs, cache: cache[ssmIdx] as? MambaCache) != nil { return nil }
        guard let faCache = cache[faIdx],
            case .none = createAttentionMask(h: inputs, cache: faCache)
        else { return nil }

        // Cache kinds can change mid-generation (`maybeQuantizeKVCache` swaps
        // array entries), so eligibility is re-checked every step.
        var mambaCaches = [MambaCache?](repeating: nil, count: layers.count)
        for (i, layer) in layers.enumerated() {
            if layer.isLinear {
                // No GDN state yet (a single-token prompt): the general path
                // builds the zero states.
                guard let mambaCache = cache[i] as? MambaCache, mambaCache[0] != nil,
                    mambaCache[1] != nil
                else { return nil }
                mambaCaches[i] = mambaCache
            } else {
                guard let kv = cache[i], hasPlainAttentionRoute(kv) else { return nil }
            }
        }

        var carry = inputs
        var pendingAttention: [MLXArray] = []

        for (segmentIndex, segment) in decodeSegments.enumerated() {
            var args: [MLXArray] = [carry] + pendingAttention
            for layerIndex in segment.linearLayers {
                let mambaCache = mambaCaches[layerIndex]!
                args.append(mambaCache[0]!)
                args.append(mambaCache[1]!)
            }

            compileLock.lock()
            if compiledSegments[segmentIndex] == nil {
                // [unowned self]: see Qwen35SparseMoeBlock.callAsFunction.
                compiledSegments[segmentIndex] = compile { [unowned self] segmentArgs in
                    segmentBody(at: segmentIndex, segmentArgs)
                }
            }
            let fn = compiledSegments[segmentIndex]!
            compileLock.unlock()
            let outputs = fn(args)

            carry = outputs[0]
            for (i, layerIndex) in segment.linearLayers.enumerated() {
                let mambaCache = mambaCaches[layerIndex]!
                mambaCache[0] = outputs[1 + 2 * i]
                mambaCache[1] = outputs[2 + 2 * i]
                mambaCache.advance(1)
            }

            pendingAttention = []
            if let pre = segment.attentionPreLayer {
                let head = segment.attentionOutputOffset
                let attention = layers[pre].attentionCacheStep(
                    queries: outputs[head], keys: outputs[head + 2],
                    values: outputs[head + 3], cache: cache[pre]!, mask: .none)
                pendingAttention = [attention, outputs[head + 1]]
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
