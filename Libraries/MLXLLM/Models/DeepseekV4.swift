// Copyright © 2026 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseek_v4 (MIT)
// DeepSeek-V4 text model (no MTP): mHC hyper-connections, one KV head shared by keys and values,
// a sliding window plus compressed KV pools (ratio 4 or 128) with a lightning indexer, and
// sqrt-softplus or token-id hash routing. Module paths follow the MLX checkpoint layout.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    struct RopeScaling: Codable, Sendable {
        var type: String?
        var ropeType: String?
        var factor: Float?
        var originalMaxPositionEmbeddings: Int?
        var betaFast: Float?
        var betaSlow: Float?

        enum CodingKeys: String, CodingKey {
            case type
            case ropeType = "rope_type"
            case factor
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
        }
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var nSharedExperts: Int
    var nRoutedExperts: Int
    var routedScalingFactor: Float
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var numExpertsPerTok: Int
    var normTopkProb: Bool
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: RopeScaling?
    var attentionBias: Bool
    var headDim: Int
    var scoringFunc: String
    var compressRatios: [Int]
    var compressRopeTheta: Float
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float
    var numHashLayers: Int
    var swigluLimit: Float
    var slidingWindow: Int
    var oGroups: Int
    var oLoraRank: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    var indexTopk: Int
    /// Opt-in hierarchical (HISA) indexer selection, an mlx-vlm approximation that neither the
    /// checkpoint configuration nor DeepSeek's reference has: blocks of `indexBlock` pooled
    /// entries, of which the best `indexKeep` (by their mean entry) are scored in full. Only
    /// whole blocks are candidates, so the entries of a trailing partial block
    /// (`poolCount % indexBlock`) are never selected on this path. The default 0 selects the
    /// exact top-k over every visible entry, as the reference does.
    var indexBlock: Int
    var indexKeep: Int
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case scoringFunc = "scoring_func"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case numHashLayers = "num_hash_layers"
        case swigluLimit = "swiglu_limit"
        case slidingWindow = "sliding_window"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case indexBlock = "index_block"
        case indexKeep = "index_keep"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v4"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 129280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 43
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1024
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        scoringFunc =
            try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        compressRopeTheta =
            try c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160000
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        numHashLayers = try c.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 3
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        oGroups = try c.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        oLoraRank = try c.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1024
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 64
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 512
        indexBlock = try c.decodeIfPresent(Int.self, forKey: .indexBlock) ?? 0
        indexKeep = try c.decodeIfPresent(Int.self, forKey: .indexKeep) ?? 0
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Checkpoints list one ratio per layer plus the MTP stages; the text model takes the
        // first `num_hidden_layers`. There is no default layout to fall back on.
        guard let ratios = try c.decodeIfPresent([Int].self, forKey: .compressRatios) else {
            throw DecodingError.keyNotFound(
                CodingKeys.compressRatios,
                DecodingError.Context(
                    codingPath: c.codingPath,
                    debugDescription:
                        "deepseek_v4: compress_ratios is required (one ratio per layer)"))
        }
        compressRatios = Array(ratios.prefix(hiddenLayers))
    }
}

extension DeepseekV4Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "DeepseekV4Configuration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "deepseek_v4: \(message)"])
        }
        guard compressRatios.count == hiddenLayers else {
            throw fail(
                "compress_ratios has \(compressRatios.count) entries for \(hiddenLayers) layers")
        }
        let unsupported = compressRatios.filter { ![0, 4, 128].contains($0) }
        guard unsupported.isEmpty else { throw fail("unsupported compress ratios \(unsupported)") }
        guard ["sqrtsoftplus", "sigmoid", "softmax"].contains(scoringFunc) else {
            throw fail("unsupported scoring_func \(scoringFunc)")
        }
        guard attentionHeads % oGroups == 0 else {
            throw fail(
                "num_attention_heads \(attentionHeads) is not a multiple of o_groups \(oGroups)")
        }
        guard qkRopeHeadDim % 2 == 0, qkRopeHeadDim <= headDim, qkRopeHeadDim <= indexHeadDim
        else {
            throw fail("qk_rope_head_dim \(qkRopeHeadDim) exceeds head_dim or index_head_dim")
        }
        guard numExpertsPerTok > 0, numExpertsPerTok <= nRoutedExperts, hcMult > 0,
            slidingWindow > 0, indexTopk > 0, indexBlock >= 0, indexKeep >= 0
        else { throw fail("invalid expert, hc, window or indexer sizes") }
        guard !tieWordEmbeddings else {
            throw fail("tie_word_embeddings is not supported; the model has a separate head")
        }
        if let scaling = ropeScaling {
            let kind = scaling.type ?? scaling.ropeType ?? "default"
            guard ["yarn", "deepseek_yarn", "default"].contains(kind) else {
                throw fail("unsupported rope_scaling type \(kind)")
            }
            if kind != "default" {
                guard scaling.factor != nil, scaling.originalMaxPositionEmbeddings != nil else {
                    throw fail(
                        "yarn rope_scaling needs factor and original_max_position_embeddings")
                }
            }
        }
    }
}

// MARK: - Helpers

/// Masked scores take `finfo(dtype).min`, so they stay finite in their dtype.
private func deepseekV4MaskScores(_ scores: MLXArray, _ mask: MLXArray?) -> MLXArray {
    guard let mask else { return scores }
    return MLX.where(
        mask, scores, MLXArray(Float(scores.dtype.finfo!.min)).asType(scores.dtype))
}

private func deepseekV4MultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let quantized as QuantizedMultiLinear: quantized(x)
    case let dense as MultiLinear: dense(x)
    default: fatalError("Expected MultiLinear or QuantizedMultiLinear, got \(type(of: module))")
    }
}

/// SwiGLU with `gate` clamped from above and `up` clamped to `[-limit, limit]`, computed in
/// float32 and returned in the input dtype, as the reference's `Expert`.
private func deepseekV4LimitedSwiGLU(limit: Float) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    guard limit > 0 else {
        return compile(shapeless: true) { gate, up in
            (silu(gate.asType(.float32)) * up.asType(.float32)).asType(gate.dtype)
        }
    }
    return compile(shapeless: true) { gate, up in
        let (g, u) = (gate.asType(.float32), up.asType(.float32))
        return (silu(minimum(g, limit)) * clip(u, min: -limit, max: limit)).asType(gate.dtype)
    }
}

/// As the reference's `RMSNorm`: float32 input and weight, rounded once to the input dtype.
final class DeepseekV4RMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
            .asType(x.dtype)
    }
}

// MARK: - RoPE

/// Interleaved RoPE on the last `dims` channels, optional YaRN (no mscale) and an inverse mode.
/// With `freqScale`, pooled entry `i` of a call at `base` sits at token `base + i * freqScale`.
/// Not `YarnRoPE`: this rotates the trailing channels (interleaved pairs) and has an inverse mode.
/// The frequency tables of every rotated head width (`headDims`) are built here, forward and
/// inverse.
final class DeepseekV4RoPE {
    let dims: Int
    let freqScale: Int
    /// Keyed by head width, negated for the inverse rotation.
    private let tables: [Int: MLXArray]

    init(
        dims: Int, headDims: [Int], base: Float, scaling: DeepseekV4Configuration.RopeScaling?,
        freqScale: Int = 1
    ) {
        self.dims = dims
        self.freqScale = freqScale
        var invFreq =
            1.0
            / pow(MLXArray(base), MLXArray.arange(0, dims, step: 2, dtype: .float32) / Float(dims))
        let kind = scaling?.type ?? scaling?.ropeType
        if let scaling, kind == "yarn" || kind == "deepseek_yarn",
            let factor = scaling.factor, let original = scaling.originalMaxPositionEmbeddings
        {
            let betaFast = Double(scaling.betaFast ?? 32)
            let betaSlow = Double(scaling.betaSlow ?? 1)
            func correctionDim(_ rotations: Double) -> Double {
                Double(dims) * Foundation.log(Double(original) / (rotations * 2 * Double.pi))
                    / (2 * Foundation.log(Double(base)))
            }
            let low = max(Foundation.floor(correctionDim(betaFast)), 0)
            var high = min(Foundation.ceil(correctionDim(betaSlow)), Double(dims - 1))
            if low == high { high += 0.001 }
            let ramp =
                (MLXArray.arange(dims / 2, dtype: .float32) - Float(low)) / Float(high - low)
            let smooth = 1 - clip(ramp, min: 0, max: 1)
            invFreq = invFreq / factor * (1 - smooth) + invFreq * smooth
        }
        var freqs = 1.0 / invFreq
        if freqScale != 1 { freqs = freqs / Float(freqScale) }
        var tables = [Int: MLXArray]()
        for headDim in headDims {
            for inverse in [false, true] {
                var f = inverse ? -freqs : freqs
                let nopePairs = (headDim - dims) / 2
                if nopePairs > 0 {
                    f = concatenated([
                        MLXArray.full([nopePairs], values: MLXArray(Float.infinity)), f,
                    ])
                }
                tables[inverse ? -headDim : headDim] = f
            }
        }
        self.tables = tables
    }

    /// Rotate `[..., T, headDim]` at positions `offset ..< offset + T`.
    func callAsFunction(_ x: MLXArray, offset: Int, inverse: Bool = false) -> MLXArray {
        let headDim = x.dim(-1)
        guard let freqs = tables[inverse ? -headDim : headDim] else {
            fatalError("DeepseekV4RoPE has no table for head width \(headDim)")
        }
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: true, base: nil, scale: 1,
            offset: freqScale != 1 ? offset / freqScale : offset, freqs: freqs)
    }
}

// MARK: - Hyper-connections

/// `post * x + comb^T @ residual`: the residual update of one mHC branch.
/// Inputs: branch output `[B, L, D]`, residual `[B, L, hc, D]`, post `[B, L, hc]`,
/// comb `[B, L, hc, hc]`.
let deepseekV4HCExpand: @Sendable ([MLXArray]) -> [MLXArray] =
    compile(shapeless: true) { args in
        let (x, residual, post, comb) = (args[0], args[1], args[2], args[3])
        let expanded =
            expandedDimensions(post, axis: -1) * expandedDimensions(x, axis: 2).asType(.float32)
        let mixed = matmul(comb.swappedAxes(-1, -2), residual.asType(.float32))
        return [(expanded + mixed).asType(x.dtype)]
    }

/// For `hc_mult == 4`: the `pre`/`post` sigmoids, the Sinkhorn-normalized `comb` and the
/// `pre`-weighted collapse of the streams. Inputs: `x_in` `[B, L, 4, D]`, `mixes` `[B, L, 24]`,
/// float32 `scale` and `base`.
/// Metal source from mlx-vlm `hyper_connection.py` (Copyright © 2026 Apple Inc., MIT).
private let deepseekV4HCSinkhornCollapseKernel = MLXFast.metalKernel(
    name: "deepseek_v4_hc_sinkhorn_collapse",
    inputNames: ["x_in", "mixes", "scale", "base"],
    outputNames: ["collapsed", "post", "comb"],
    source: """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in + row * (HC * D);
        device U*       out_row = (device U*)collapsed + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = U4(result);
        }

        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                      + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
            out_row[d] = (U)val;
        }
        #endif
        """)

/// mHC (`hc_attn_*` / `hc_ffn_*`): mixes the `hc_mult` residual streams into one branch input
/// and returns the `post` and Sinkhorn-normalized `comb` weights for the residual update.
final class DeepseekV4HyperConnection: Module {
    let hcMult: Int
    let normEps: Float
    let hcEps: Float
    let sinkhornIters: Int

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    private let sinkhorn: @Sendable ([MLXArray]) -> [MLXArray]

    init(_ config: DeepseekV4Configuration) {
        let hc = config.hcMult
        let eps = config.hcEps
        let iterations = config.hcSinkhornIters
        self.hcMult = hc
        self.normEps = config.rmsNormEps
        self.hcEps = eps
        self.sinkhornIters = iterations

        let mix = (2 + hc) * hc
        _fn.wrappedValue = MLXArray.zeros([mix, hc * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([3], dtype: .float32)

        self.sinkhorn = compile(shapeless: false) { args in
            let mixes = args[0].asType(.float32)
            let scale = args[1].asType(.float32)
            let base = args[2].asType(.float32)
            let pre = sigmoid(mixes[.ellipsis, ..<hc] * scale[0] + base[..<hc]) + eps
            let post =
                2 * sigmoid(mixes[.ellipsis, hc ..< (2 * hc)] * scale[1] + base[hc ..< (2 * hc)])
            let leading = Array(mixes.shape.dropLast())
            var comb =
                mixes[.ellipsis, (2 * hc)...].reshaped(leading + [hc, hc]) * scale[2]
                + base[(2 * hc)...].reshaped(hc, hc)
            comb = softmax(comb, axis: -1, precise: true) + eps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            for _ in 0 ..< max(iterations - 1, 0) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            }
            return [pre, post, comb]
        }
        super.init()
    }

    /// Collapse `[B, L, hc, D]` to `[B, L, D]`; returns `(collapsed, post, comb)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = x.asType(.float32)
        let z = MLXFast.rmsNorm(y.reshaped(B, L, H * D), weight: MLXArray.mlxNone, eps: normEps)
        let mixes = matmul(z, fn.T)
        // As the reference: the fused kernel on the GPU outside training, else the op path.
        if hcMult == 4, !training, Device.defaultDevice().deviceType == .gpu {
            let out = deepseekV4HCSinkhornCollapseKernel(
                [x, mixes.asType(.float32), scale.asType(.float32), base.asType(.float32)],
                template: [
                    ("T", x.dtype), ("U", x.dtype), ("HC", hcMult), ("ITERS", sinkhornIters),
                    ("D", D), ("EPS_INT", Int((Double(hcEps) / 1e-9).rounded())),
                ],
                grid: (B * L * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[B, L, D], [B, L, hcMult], [B, L, hcMult, hcMult]],
                outputDTypes: [x.dtype, .float32, .float32])
            return (out[0], out[1], out[2])
        }
        let weights = sinkhorn([mixes, scale, base])
        let (pre, post, comb) = (weights[0], weights[1], weights[2])
        let collapsed = (expandedDimensions(pre, axis: -1) * y).sum(axis: 2).asType(x.dtype)
        return (collapsed, post, comb)
    }
}

/// `HyperHead`: a learned sigmoid-weighted collapse of the streams before the final norm.
final class DeepseekV4HyperHead: Module {
    let normEps: Float
    let hcEps: Float
    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: DeepseekV4Configuration) {
        self.normEps = config.rmsNormEps
        self.hcEps = config.hcEps
        _fn.wrappedValue = MLXArray.zeros(
            [config.hcMult, config.hcMult * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([config.hcMult], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([1], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = x.asType(.float32)
        let z = MLXFast.rmsNorm(y.reshaped(B, L, H * D), weight: MLXArray.mlxNone, eps: normEps)
        let pre = sigmoid(matmul(z, fn.T) * scale + base) + hcEps
        return (expandedDimensions(pre, axis: -1) * y).sum(axis: 2).asType(x.dtype)
    }
}

// MARK: - Feed-forward

final class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var gateProj: Linear
    @ModuleInfo(key: "w2") var downProj: Linear
    @ModuleInfo(key: "w3") var upProj: Linear
    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self.activation = deepseekV4LimitedSwiGLU(limit: swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x), upProj(x)))
    }
}

/// Router: `sqrt(softplus(logits))` scores. Hash layers take their experts from
/// `tid2eid[token]`, the others the top-k of `scores + bias`. The weights are the scores at the
/// chosen experts, renormalized and scaled.
final class DeepseekV4MoEGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray?

    let topK: Int
    let scoringFunc: String
    let scaling: Float
    let normalize: Bool

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        let experts = config.nRoutedExperts
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize])
        if layerIdx < config.numHashLayers {
            _tid2eid.wrappedValue = MLXArray.zeros(
                [config.vocabularySize, config.numExpertsPerTok], dtype: .int32)
        } else {
            _bias.wrappedValue = MLXArray.zeros([experts], dtype: .float32)
        }
        self.topK = config.numExpertsPerTok
        self.scoringFunc = config.scoringFunc
        self.scaling = config.routedScalingFactor
        self.normalize = config.normTopkProb
        super.init()
    }

    private func score(_ logits: MLXArray) -> MLXArray {
        switch scoringFunc {
        case "softmax": softmax(logits, axis: -1, precise: true)
        case "sigmoid": sigmoid(logits)
        default: sqrt(softplus(logits))
        }
    }

    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> (MLXArray, MLXArray) {
        // As the reference: the router logits are a float32 matmul.
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        let scores = score(logits)
        let indices: MLXArray
        if let tid2eid {
            indices = take(tid2eid, inputIDs, axis: 0).asType(.int32)
        } else {
            let biased = scores + bias!
            indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
                .asType(.int32)
        }
        var weights = takeAlong(scores, indices, axis: -1)
        if scoringFunc != "softmax", normalize {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * scaling)
    }
}

final class DeepseekV4MoE: Module {
    @ModuleInfo(key: "gate") var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _gate.wrappedValue = DeepseekV4MoEGate(config, layerIdx: layerIdx)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activationProduct: deepseekV4LimitedSwiGLU(limit: config.swigluLimit))
        _sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
        super.init()
    }

    /// Routed outputs weighted by the float32 router weights and summed with the shared expert
    /// in float32, as the reference, then returned in the input dtype.
    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, inputIDs: inputIDs)
        let routed = switchMLP(x, indices)
        let sum = weightedExpertSum(routed, weights.asType(.float32))
        return (sum + sharedExperts(x).asType(.float32)).asType(x.dtype)
    }
}

// MARK: - Compressed pool cache

/// The state of one compressed-attention branch: the raw kv/gate rows of the window still
/// filling up (fewer than `ratio`), on ratio-4 (overlapping) pools the raw rows of the last
/// completed window, and the pool of compressed entries, one per `ratio` tokens.
/// After mlx-vlm's `PoolingCache`. `offset` counts the tokens seen, like the window cache.
final class DeepseekV4PoolCache: BaseKVCache {
    let ratio: Int
    /// As the reference's `Compressor.overlap`: each ratio-4 entry also pools the window before
    /// it, however the tokens were split into calls.
    let overlap: Bool

    /// `[B, ratio, D]` raw rows; the first `remainder` are live.
    var bufferKV: MLXArray?
    var bufferGate: MLXArray?
    private(set) var remainder = 0

    /// `[B, ratio, D]` raw rows of the last completed window (the reference's
    /// `kv_state[:, :ratio]`), kept on overlapping pools; `nil` before the first window.
    private(set) var previousKV: MLXArray?
    private(set) var previousGate: MLXArray?

    /// `[B, capacity, D]` storage of the compressed entries in token order, grown `step` entries
    /// at a time like `KVCacheSimple`; the first `pooledCount` are live. `nil` before the first
    /// window.
    private var pooledStorage: MLXArray?
    private(set) var pooledCount = 0
    var step = 256

    init(ratio: Int) {
        precondition(ratio > 0)
        self.ratio = ratio
        self.overlap = ratio == 4
        super.init()
    }

    /// `[B, pooledCount, D]` live entries, or `nil` before the first window.
    var pooled: MLXArray? {
        guard let pooledStorage else { return nil }
        return pooledCount == pooledStorage.dim(1)
            ? pooledStorage : pooledStorage[0..., ..<pooledCount]
    }

    override func innerState() -> [MLXArray] {
        [bufferKV, bufferGate, previousKV, previousGate, pooledStorage].compactMap { $0 }
    }

    /// Rows of every window this call completes, `[B, W * ratio, D]` (empty when none), the
    /// token position of the first such row and, on overlapping pools, the `[B, ratio, D]` rows
    /// of the window before that one (`nil` for the first window of the sequence). `kv`/`gate`
    /// are `[B, L, D]` projections of the next `L` tokens.
    func accumulateWindows(kv: MLXArray, gate: MLXArray)
        -> (kv: MLXArray, gate: MLXArray, base: Int, previousKV: MLXArray?, previousGate: MLXArray?)
    {
        let ready = completeWindows(kv: kv, gate: gate)
        let n = ready.kv.dim(1)
        guard overlap, n > 0 else { return (ready.kv, ready.gate, ready.base, nil, nil) }
        let previous = (previousKV, previousGate)
        // The last window completed here precedes the next call's first one.
        previousKV = ready.kv[0..., (n - ratio)...]
        previousGate = ready.gate[0..., (n - ratio)...]
        return (ready.kv, ready.gate, ready.base, previous.0, previous.1)
    }

    private func completeWindows(kv: MLXArray, gate: MLXArray)
        -> (kv: MLXArray, gate: MLXArray, base: Int)
    {
        let (B, L) = (kv.dim(0), kv.dim(1))
        let (kvWidth, gateWidth) = (kv.dim(2), gate.dim(2))
        let offset = self.offset
        self.offset += L
        if bufferKV == nil {
            bufferKV = MLXArray.zeros([B, ratio, kvWidth], dtype: kv.dtype)
            bufferGate = MLXArray.zeros([B, ratio, gateWidth], dtype: gate.dtype)
        }
        let emptyKV = MLXArray.zeros([B, 0, kvWidth], dtype: kv.dtype)
        let emptyGate = MLXArray.zeros([B, 0, gateWidth], dtype: gate.dtype)

        if L > 1 {
            let total = L + remainder
            let usable = (total / ratio) * ratio
            let newRemainder = total % ratio
            var ready = (emptyKV, emptyGate, 0)
            if usable > 0 {
                let fresh = usable - remainder
                ready = (
                    concatenated([bufferKV![0..., ..<remainder], kv[0..., ..<fresh]], axis: 1),
                    concatenated([bufferGate![0..., ..<remainder], gate[0..., ..<fresh]], axis: 1),
                    offset - remainder
                )
                remainder = 0
            }
            if newRemainder > 0 {
                // The trailing rows of the call; all `L` of them when no window completed and
                // `remainder` rows were already buffered.
                let tail = L - (newRemainder - remainder)
                bufferKV![0..., remainder ..< newRemainder] = kv[0..., tail...]
                bufferGate![0..., remainder ..< newRemainder] = gate[0..., tail...]
            }
            remainder = newRemainder
            return ready
        }

        bufferKV![0..., remainder ..< (remainder + 1)] = kv
        bufferGate![0..., remainder ..< (remainder + 1)] = gate
        remainder = (remainder + 1) % ratio
        if remainder == 0 {
            return (bufferKV![.ellipsis], bufferGate![.ellipsis], offset - ratio + 1)
        }
        return (emptyKV, emptyGate, 0)
    }

    /// Append `[B, W, D]` new entries and return the whole pool (`[B, 0, D]` while empty).
    func appendPooled(_ entries: MLXArray) -> MLXArray {
        let (B, W, D) = (entries.dim(0), entries.dim(1), entries.dim(2))
        guard W > 0 else { return pooled ?? entries }
        let previous = pooledCount
        if pooledStorage == nil || previous + W > pooledStorage!.dim(1) {
            let grown = MLXArray.zeros([B, (W + step - 1) / step * step, D], dtype: entries.dtype)
            if let current = pooledStorage {
                pooledStorage = concatenated([current[0..., ..<previous], grown], axis: 1)
            } else {
                pooledStorage = grown
            }
        }
        pooledStorage![0..., previous ..< (previous + W)] = entries
        pooledCount = previous + W
        return pooledStorage![0..., ..<pooledCount]
    }

    /// `[L, P]` visibility of the pooled entries to the queries at `offset ..< offset + L`:
    /// entry `i` is visible to the query at `t` iff `i < (t + 1) / ratio`. `nil` for one query
    /// or an empty pool.
    func poolMask(queries L: Int, offset: Int) -> MLXArray? {
        guard pooledCount > 0, L > 1 else { return nil }
        let poolIndex = MLXArray.arange(pooledCount, dtype: .int32)
        let queryIndex = MLXArray.arange(offset + 1, offset + L + 1, dtype: .int32)
        return poolIndex .< expandedDimensions(queryIndex.floorDivide(Int32(ratio)), axis: 1)
    }

    /// The live arrays only: `[bufferKV, bufferGate]`, then `[previousKV, previousGate]` when
    /// kept, then the pool trimmed to `pooledCount` when not empty (0, 2, 3, 4 or 5 arrays).
    override var state: [MLXArray] {
        get { [bufferKV, bufferGate, previousKV, previousGate, pooled].compactMap { $0 } }
        set {
            guard [0, 2, 3, 4, 5].contains(newValue.count) else {
                fatalError("DeepseekV4PoolCache state must have 0, 2, 3, 4 or 5 arrays")
            }
            bufferKV = newValue.count >= 2 ? newValue[0] : nil
            bufferGate = newValue.count >= 2 ? newValue[1] : nil
            let hasPrevious = newValue.count >= 4
            previousKV = hasPrevious ? newValue[2] : nil
            previousGate = hasPrevious ? newValue[3] : nil
            pooledStorage = newValue.count % 2 == 1 ? newValue[newValue.count - 1] : nil
            pooledCount = pooledStorage?.dim(1) ?? 0
        }
    }

    override var metaState: [String] {
        get { [String(ratio), String(remainder), String(offset)] }
        set {
            guard newValue.count == 3, Int(newValue[0]) == ratio,
                let remainder = Int(newValue[1]), let offset = Int(newValue[2])
            else { fatalError("DeepseekV4PoolCache metaState mismatch: \(newValue)") }
            self.remainder = remainder
            self.offset = offset
        }
    }

    override func copy() -> any KVCache {
        let new = DeepseekV4PoolCache(ratio: ratio)
        new.bufferKV = bufferKV?[.ellipsis]
        new.bufferGate = bufferGate?[.ellipsis]
        new.previousKV = previousKV?[.ellipsis]
        new.previousGate = previousGate?[.ellipsis]
        new.pooledStorage = pooled?[.ellipsis]
        new.pooledCount = pooledCount
        new.step = step
        new.remainder = remainder
        new.offset = offset
        return new
    }
}

// MARK: - Compressor and indexer

private let deepseekV4SimpleCompress: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { kv, gate, ape in
        let weights = softmax(gate.asType(.float32) + ape, axis: -2).asType(kv.dtype)
        return (kv * weights).sum(axis: -2)
    }

/// Ratio-4 overlapping windows, as the reference's `overlap_transform`: each entry pools the
/// `2 * ratio` rows of the first half-channels of the window before it and the second
/// half-channels of its own window, `ape` added to both. Inputs: `kv`/`gate` `[B, W, R, 2D]`
/// windows, `ape` `[R, 2D]`, and the `[B, 1, R, 2D]` window before the first one (zeros and
/// gate `-inf` for the first window of the sequence, which has none).
private let deepseekV4OverlapCompress: @Sendable ([MLXArray]) -> [MLXArray] =
    compile { args in
        let (kv, gate, ape) = (args[0], args[1], args[2])
        let (previousKV, previousGate) = (args[3], args[4])
        let (W, half) = (kv.dim(1), kv.dim(3) / 2)
        let allKV = concatenated([previousKV, kv], axis: 1)
        let allGate = concatenated([previousGate, gate], axis: 1) + ape.asType(gate.dtype)
        let mixedKV = concatenated(
            [allKV[0..., ..<W, 0..., ..<half], allKV[0..., 1..., 0..., half...]], axis: 2)
        let mixedGate = concatenated(
            [allGate[0..., ..<W, 0..., ..<half], allGate[0..., 1..., 0..., half...]], axis: 2)
        let weights = softmax(mixedGate, axis: -2, precise: true)
        return [(mixedKV * weights).sum(axis: -2)]
    }

/// Pools every complete window of `ratio` tokens into one entry: the `softmax(gate + ape)`
/// weighted sum of the window's kv, RMSNorm, then RoPE at the window's start position.
final class DeepseekV4Compressor: Module {
    let ratio: Int
    let headDim: Int
    let overlap: Bool

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let rope: DeepseekV4RoPE

    init(_ config: DeepseekV4Configuration, ratio: Int, headDim: Int) {
        self.ratio = ratio
        self.headDim = headDim
        self.overlap = ratio == 4
        let outDim = headDim * (overlap ? 2 : 1)
        _wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        _wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        _ape.wrappedValue = MLXArray.zeros([ratio, outDim], dtype: .float32)
        _norm.wrappedValue = DeepseekV4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim, headDims: [headDim], base: config.compressRopeTheta,
            scaling: config.ropeScaling, freqScale: ratio)
        super.init()
    }

    /// The whole pool `[B, P, headDim]` after adding the windows that `x` (`[B, L, hidden]`)
    /// completes.
    func callAsFunction(_ x: MLXArray, cache: DeepseekV4PoolCache) -> MLXArray {
        let B = x.dim(0)
        // As the reference, projections, buffered rows and pooling are float32; the pooled
        // entries return to the input dtype before the norm.
        let x32 = x.asType(.float32)
        let ready = cache.accumulateWindows(kv: wkv(x32), gate: wgate(x32))
        guard ready.kv.dim(1) > 0 else {
            return cache.appendPooled(MLXArray.zeros([B, 0, headDim], dtype: x.dtype))
        }
        let windows = ready.kv.dim(1) / ratio
        let kv = ready.kv.reshaped(B, windows, ratio, -1)
        let gate = ready.gate.reshaped(B, windows, ratio, -1)
        var entries: MLXArray
        if overlap {
            let shape = [B, 1, ratio, kv.dim(-1)]
            let previousKV =
                ready.previousKV?.reshaped(shape) ?? MLXArray.zeros(shape, dtype: kv.dtype)
            let previousGate =
                ready.previousGate?.reshaped(shape)
                ?? MLXArray.full(shape, values: MLXArray(-Float.infinity), type: Float.self)
                .asType(gate.dtype)
            entries = deepseekV4OverlapCompress([kv, gate, ape, previousKV, previousGate])[0]
        } else {
            entries = deepseekV4SimpleCompress(kv, gate, ape)
        }
        entries = norm(entries.asType(x.dtype))
        entries = rope(expandedDimensions(entries, axis: 1), offset: ready.base).squeezed(axis: 1)
        return cache.appendPooled(entries)
    }
}

/// Lightning indexer: `sum_h relu(q_h . k) * scale * w_h(x) / sqrt(H)` over the ratio-4 index
/// pool, then the exact top `index_topk` entries per query. With the opt-in `index_block > 0`,
/// large enough pools use the two-stage (HISA) block filter instead.
final class DeepseekV4Indexer: Module {
    let heads: Int
    let headDim: Int
    let topK: Int
    let scale: Float
    let block: Int
    let keep: Int

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(_ config: DeepseekV4Configuration, ratio: Int) {
        self.heads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.scale = pow(Float(config.indexHeadDim), -0.5)
        self.block = config.indexBlock
        self.keep = config.indexKeep
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _compressor.wrappedValue = DeepseekV4Compressor(config, ratio: ratio, headDim: headDim)
        super.init()
    }

    /// `[B, L, k]` pooled indices, or `nil` while the pool is empty.
    func callAsFunction(
        _ x: MLXArray, qResidual: MLXArray, rope: DeepseekV4RoPE, cache: DeepseekV4PoolCache,
        offset: Int
    ) -> MLXArray? {
        let (B, L) = (x.dim(0), x.dim(1))
        let pooled = compressor(x, cache: cache)
        let poolCount = pooled.dim(1)
        guard poolCount > 0 else { return nil }

        var q = wqB(qResidual).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        let k = min(topK, poolCount)
        let poolMask = cache.poolMask(queries: L, offset: offset)
        let projected = weightsProj(x)
        let weights = projected.asType(.float32) * pow(Float(heads), -0.5)

        if block > 0, poolCount >= block * keep, keep * block >= k {
            if L == 1 {
                return hisaDecode(q: q, pooled: pooled, weights: weights, k: k)
            }
            let visible: MLXArray
            if let poolMask {
                visible = broadcast(
                    poolMask.asType(.int32).sum(axis: -1, keepDims: false)[.newAxis], to: [B, L])
            } else {
                visible = MLXArray.full([B, L], values: MLXArray(Int32(poolCount)))
            }
            return hisaPrefill(q: q, pooled: pooled, weights: weights, k: k, visible: visible)
        }

        // Exact top-k as the reference, in query tiles that bound the `[B, H, tile, P]` scores.
        // The reference scores in the activation dtype (`weights_proj` is bfloat16).
        let keys = expandedDimensions(pooled, axis: 1).swappedAxes(-1, -2)
        let headWeights = expandedDimensions(
            (projected * pow(Float(heads), -0.5)).swappedAxes(-1, -2), axis: -1)  // [B,H,L,1]
        let tile = max(
            1, exactQueryTile ?? Self.exactScoreBudget / max(heads * poolCount * B, 1))
        var parts = [MLXArray]()
        for start in stride(from: 0, to: L, by: tile) {
            let end = min(start + tile, L)
            var scores = matmul(q[0..., 0..., start ..< end], keys)
            scores = maximum(scores, 0) * scale
            scores = (scores * headWeights[0..., 0..., start ..< end]).sum(axis: 1)
            if let poolMask {
                scores = deepseekV4MaskScores(
                    scores, expandedDimensions(poolMask[start ..< end], axis: 0))
            }
            let selected = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            if tile < L { eval(selected) }
            parts.append(selected)
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    /// Float32 elements of one tile's `[B, H, tile, P]` exact scores.
    private static let exactScoreBudget = 1 << 26

    /// Queries per tile on the exact path; `nil` sizes tiles by `exactScoreBudget`.
    var exactQueryTile: Int?

    /// One query per row, every pooled entry visible. Only whole blocks are candidates.
    private func hisaDecode(q: MLXArray, pooled: MLXArray, weights: MLXArray, k: Int) -> MLXArray {
        let (B, poolCount, D) = (pooled.dim(0), pooled.dim(1), pooled.dim(2))
        let blocks = poolCount / block
        let usable = blocks * block
        let wq = expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)  // [B, H, 1, 1]
        let q32 = q.asType(.float32)

        let rep = pooled[0..., ..<usable].reshaped(B, blocks, block, D).mean(axis: 2)
        let coarse = maximum(
            matmul(q32, expandedDimensions(rep, axis: 1).swappedAxes(-1, -2).asType(.float32)), 0)
        let coarseScore = (coarse * scale * wq).sum(axis: 1)  // [B, 1, blocks]
        let kept = min(keep, blocks)
        let topBlocks = argPartition(-coarseScore, kth: kept - 1, axis: -1)[.ellipsis, ..<kept]
            .asType(.int32)

        let candidates = kept * block
        let positions =
            (expandedDimensions(topBlocks, axis: -1) * Int32(block)
            + MLXArray.arange(block, dtype: .int32)).reshaped(B, 1, candidates)
        let gatherIndex = broadcast(
            expandedDimensions(positions.reshaped(B, candidates), axis: -1),
            to: [B, candidates, D])
        let candidateKeys = takeAlong(pooled, gatherIndex, axis: 1)
        let fine = maximum(
            matmul(
                q32,
                expandedDimensions(candidateKeys, axis: 1).swappedAxes(-1, -2).asType(.float32)),
            0)
        let fineScore = (fine * scale * wq).sum(axis: 1)  // [B, 1, candidates]
        let selected = argPartition(-fineScore, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        return takeAlong(positions, selected, axis: -1)
    }

    /// Follows mlx-vlm's `hisa_select`: the fine stage runs in tiles of 512 queries, each
    /// evaluated before the next, to bound memory.
    private static let hisaQueryTile = 512

    /// `visible[b, l]` pooled entries are causally visible to each query.
    private func hisaPrefill(
        q: MLXArray, pooled: MLXArray, weights: MLXArray, k: Int, visible: MLXArray
    ) -> MLXArray {
        let (B, L) = (q.dim(0), q.dim(2))
        let (poolCount, D) = (pooled.dim(1), pooled.dim(2))
        let blocks = poolCount / block
        let usable = blocks * block
        let q32 = q.asType(.float32)
        let pooled32 = pooled.asType(.float32)
        let masked = MLXArray(Float(-1e30))

        let wk = weights * scale  // [B, L, H]
        let wkHeads = expandedDimensions(wk.transposed(0, 2, 1), axis: -1)  // [B, H, L, 1]
        let rep = pooled32[0..., ..<usable].reshaped(B, blocks, block, D).mean(axis: 2)
        let coarse = maximum(matmul(q32, expandedDimensions(rep, axis: 1).swappedAxes(-1, -2)), 0)
        var coarseScore = (coarse * wkHeads).sum(axis: 1)  // [B, L, blocks]
        let blockStart = MLXArray.arange(blocks, dtype: .int32) * Int32(block)
        coarseScore = MLX.where(
            blockStart.reshaped(1, 1, blocks) .< expandedDimensions(visible, axis: -1),
            coarseScore, masked)
        let kept = min(keep, blocks)
        let topBlocks = argPartition(-coarseScore, kth: kept - 1, axis: -1)[.ellipsis, ..<kept]
            .asType(.int32)

        let candidates = kept * block
        let within = MLXArray.arange(block, dtype: .int32)
        let tile = Self.hisaQueryTile
        var parts = [MLXArray]()
        for start in stride(from: 0, to: L, by: tile) {
            let end = min(start + tile, L)
            let n = end - start
            let positions =
                (expandedDimensions(topBlocks[0..., start ..< end], axis: -1)
                * Int32(block) + within).reshaped(B, n, candidates)
            let gatherIndex = broadcast(
                expandedDimensions(positions.reshaped(B, n * candidates), axis: -1),
                to: [B, n * candidates, D])
            let candidateKeys = takeAlong(pooled32, gatherIndex, axis: 1)
                .reshaped(B, n, candidates, D)
            let qTile = q32[0..., 0..., start ..< end].transposed(0, 2, 1, 3)  // [B, n, H, D]
            let fine = maximum(matmul(qTile, candidateKeys.swappedAxes(-1, -2)), 0)
            var score = (fine * expandedDimensions(wk[0..., start ..< end], axis: -1)).sum(axis: 2)
            score = MLX.where(
                positions .< expandedDimensions(visible[0..., start ..< end], axis: -1), score,
                masked)
            if tile < L { eval(score) }
            parts.append(score)
        }
        let fineScore = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        let selected = argPartition(-fineScore, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let positions = (expandedDimensions(topBlocks, axis: -1) * Int32(block) + within)
            .reshaped(B, L, candidates)
        return takeAlong(positions, selected, axis: -1)
    }
}

// MARK: - Attention

/// All query heads attend to one KV head (keys = values): the sliding window plus the compressed
/// pool (every entry, or the indexer's top-k once a ratio-4 pool exceeds `index_topk`), with
/// per-head sinks. The output is inverse-rotated and projected through `o_groups` groups.
final class DeepseekV4Attention: Module {
    let ratio: Int
    let heads: Int
    let headDim: Int
    let groups: Int
    let scale: Float
    let normEps: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_a") var woA: Module
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?
    let rope: DeepseekV4RoPE

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        self.ratio = config.compressRatios[layerIdx]
        self.heads = config.attentionHeads
        self.headDim = config.headDim
        self.groups = config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.normEps = config.rmsNormEps

        _wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        _qNorm.wrappedValue = DeepseekV4RMSNorm(
            dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kvNorm.wrappedValue = DeepseekV4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _woA.wrappedValue = MultiLinear(
            inputDims: heads * headDim / groups, outputDims: config.oLoraRank, numHeads: groups)
        _woB.wrappedValue = Linear(
            groups * config.oLoraRank, config.hiddenSize, bias: config.attentionBias)
        _attnSink.wrappedValue = MLXArray.zeros([heads], dtype: .float32)
        if ratio > 0 {
            _compressor.wrappedValue = DeepseekV4Compressor(config, ratio: ratio, headDim: headDim)
        }
        if ratio == 4 {
            _indexer.wrappedValue = DeepseekV4Indexer(config, ratio: ratio)
        }
        // Local layers rotate with the plain base; compressed layers and their indexer
        // queries with the YaRN-scaled compress base.
        self.rope =
            ratio == 0
            ? DeepseekV4RoPE(
                dims: config.qkRopeHeadDim, headDims: [headDim], base: config.ropeTheta,
                scaling: nil)
            : DeepseekV4RoPE(
                dims: config.qkRopeHeadDim,
                headDims: ratio == 4 ? [headDim, config.indexHeadDim] : [headDim],
                base: config.compressRopeTheta, scaling: config.ropeScaling)
        super.init()
    }

    /// `x` is `[B, L, hidden]` (normalized); `mask` the `[L, S]` sliding-window mask for
    /// multi-token calls (`nil` for one token).
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: DeepseekV4LayerCache) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let offset = cache.local.offset

        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual).reshaped(B, L, heads, headDim)
        // The reference normalizes the query heads in the activation dtype, not float32.
        q = q * rsqrt(mean(square(q), axis: -1, keepDims: true) + normEps)
        q = rope(q.transposed(0, 2, 1, 3), offset: offset)
        var kv = kvNorm(wkv(x)).reshaped(B, 1, L, headDim)
        kv = rope(kv, offset: offset)

        var out = attend(
            x, q: q, kv: kv, qResidual: qResidual, mask: mask, cache: cache, offset: offset)
        out = rope(out, offset: offset, inverse: true)

        out = out.reshaped(B, groups, heads / groups, L, headDim).transposed(0, 1, 3, 2, 4)
            .reshaped(B, groups, L, heads / groups * headDim)
        out = deepseekV4MultiLinear(woA, out)  // [B, G, L, R]
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return woB(out)
    }

    private func attend(
        _ x: MLXArray, q: MLXArray, kv: MLXArray, qResidual: MLXArray, mask: MLXArray?,
        cache: DeepseekV4LayerCache, offset: Int
    ) -> MLXArray {
        let L = x.dim(1)
        let (local, _) = cache.local.update(keys: kv, values: kv)
        let localMask = alignLocalMask(mask, length: local.dim(2))
        // MLX attention takes its sinks in the query dtype.
        let sinks = attnSink.asType(q.dtype)

        guard let compressor, let poolCache = cache.pool else {
            return sdpa(q, local, mask: localMask, sinks: sinks)
        }
        let pooled = compressor(x, cache: poolCache)
        let poolMask = poolCache.poolMask(queries: L, offset: offset)
        let topk: MLXArray?
        if let indexer, let indexCache = cache.indexPool {
            topk = indexer(x, qResidual: qResidual, rope: rope, cache: indexCache, offset: offset)
        } else {
            topk = nil
        }

        if pooled.dim(1) == 0 {
            return sdpa(q, local, mask: localMask, sinks: sinks)
        }
        if let topk, let indexer, pooled.dim(1) > indexer.topK {
            var sparseMask: MLXArray?
            if let poolMask {
                sparseMask = expandedDimensions(
                    takeAlong(expandedDimensions(poolMask, axis: 0), topk, axis: 2), axis: 1)
            }
            return sparsePooledAttention(
                q: q, local: local, pooled: pooled, topk: topk, localMask: localMask,
                pooledMask: sparseMask, sinks: attnSink.asType(.float32))
        }
        let full = concatenated([local, expandedDimensions(pooled, axis: 1)], axis: 2)
        return sdpa(
            q, full, mask: extendMask(localMask, poolMask, length: full.dim(2)), sinks: sinks)
    }

    private func sdpa(_ q: MLXArray, _ kv: MLXArray, mask: MLXArray?, sinks: MLXArray) -> MLXArray {
        if let mask {
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv, scale: scale, mask: .array(mask), sinks: sinks)
        }
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv, scale: scale, mask: .none, sinks: sinks)
    }

    /// Fit the mask's key axis to the window's length.
    private func alignLocalMask(_ mask: MLXArray?, length: Int) -> MLXArray? {
        guard let mask else { return nil }
        let current = mask.dim(-1)
        if current == length { return mask }
        if current > length { return mask[.ellipsis, (current - length)...] }
        var padShape = mask.shape
        padShape[padShape.count - 1] = length - current
        return concatenated([MLXArray.ones(padShape, dtype: .bool), mask], axis: -1)
    }

    /// Append the pool's `[L, P]` visibility (all visible when `nil`).
    private func extendMask(_ mask: MLXArray?, _ poolMask: MLXArray?, length: Int) -> MLXArray? {
        guard let mask else { return nil }
        let width = length - mask.dim(-1)
        let pool = poolMask ?? MLXArray.ones([mask.dim(-2), width], dtype: .bool)
        return concatenated([mask, pool], axis: -1)
    }

    /// One softmax over the window, the gathered top-k pooled entries and the sink.
    private func sparsePooledAttention(
        q: MLXArray, local: MLXArray, pooled: MLXArray, topk: MLXArray, localMask: MLXArray?,
        pooledMask: MLXArray?, sinks: MLXArray
    ) -> MLXArray {
        let (B, poolCount, D) = (pooled.dim(0), pooled.dim(1), pooled.dim(2))
        let (L, k) = (topk.dim(1), topk.dim(2))
        let gathered: MLXArray
        if B == 1 {
            gathered = take(pooled, topk.reshaped(-1), axis: 1).reshaped(B, L, k, D)
        } else {
            var rows = MLX.where(topk .< 0, topk + Int32(poolCount), topk)
            rows =
                rows + (MLXArray.arange(B, dtype: topk.dtype) * Int32(poolCount)).reshaped(B, 1, 1)
            gathered = take(pooled.reshaped(B * poolCount, D), rows.reshaped(-1), axis: 0)
                .reshaped(B, L, k, D)
        }

        // As the reference's sparse attention kernel: float32 scores, softmax and sink.
        let (local32, gathered32) = (local.asType(.float32), gathered.asType(.float32))
        let qScaled = q.asType(.float32) * scale
        let localScores = deepseekV4MaskScores(
            matmul(qScaled, local32.swappedAxes(-1, -2)), localMask)
        var normalizer = logSumExp(localScores, axis: -1, keepDims: true)
        let qByToken = qScaled.transposed(0, 2, 1, 3)
        let pooledScores = deepseekV4MaskScores(
            matmul(qByToken, gathered32.swappedAxes(-1, -2)).transposed(0, 2, 1, 3), pooledMask)
        normalizer = logAddExp(normalizer, logSumExp(pooledScores, axis: -1, keepDims: true))
        normalizer = logAddExp(normalizer, sinks.reshaped(1, -1, 1, 1))
        let localWeights = exp(localScores - normalizer)
        let pooledWeights = exp(pooledScores - normalizer)

        let out =
            matmul(localWeights, local32)
            + matmul(pooledWeights.transposed(0, 2, 1, 3), gathered32).transposed(0, 2, 1, 3)
        return out.asType(q.dtype)
    }
}

// MARK: - Decoder

final class DeepseekV4Block: Module {
    @ModuleInfo(key: "attn") var attn: DeepseekV4Attention
    @ModuleInfo(key: "ffn") var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _attn.wrappedValue = DeepseekV4Attention(config, layerIdx: layerIdx)
        _ffn.wrappedValue = DeepseekV4MoE(config, layerIdx: layerIdx)
        _attnNorm.wrappedValue = DeepseekV4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ffnNorm.wrappedValue = DeepseekV4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnHC.wrappedValue = DeepseekV4HyperConnection(config)
        _ffnHC.wrappedValue = DeepseekV4HyperConnection(config)
        super.init()
    }

    /// `h` is `[B, L, hc, hidden]`.
    func callAsFunction(
        _ h: MLXArray, mask: MLXArray?, cache: DeepseekV4LayerCache, inputIDs: MLXArray
    ) -> MLXArray {
        let (attnInput, attnPost, attnComb) = attnHC(h)
        let attnOut = attn(attnNorm(attnInput), mask: mask, cache: cache)
        let h = deepseekV4HCExpand([attnOut, h, attnPost, attnComb])[0]
        let (ffnInput, ffnPost, ffnComb) = ffnHC(h)
        let ffnOut = ffn(ffnNorm(ffnInput), inputIDs: inputIDs)
        return deepseekV4HCExpand([ffnOut, h, ffnPost, ffnComb])[0]
    }
}

/// A layer's caches: the sliding window, plus the attention pool on compressed layers and the
/// indexer pool on ratio-4 layers.
struct DeepseekV4LayerCache {
    let local: RotatingKVCache
    let pool: DeepseekV4PoolCache?
    let indexPool: DeepseekV4PoolCache?

    init(_ cache: KVCache, ratio: Int) {
        if ratio > 0, let list = cache as? CacheList {
            local = list[0] as! RotatingKVCache
            pool = (list[1] as! DeepseekV4PoolCache)
            indexPool = ratio == 4 ? (list[2] as! DeepseekV4PoolCache) : nil
        } else {
            local = cache as! RotatingKVCache
            pool = nil
            indexPool = nil
        }
    }
}

// MARK: - Model

public final class DeepseekV4Model: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: DeepseekV4Configuration

    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "head") var head: Linear

    public init(_ configuration: DeepseekV4Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _embed.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            DeepseekV4Block(configuration, layerIdx: $0)
        }
        _norm.wrappedValue = DeepseekV4RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _hcHead.wrappedValue = DeepseekV4HyperHead(configuration)
        _head.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(hiddenStates(inputs, cache: cache))
    }

    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let hidden = hiddenStates(input.tokens, cache: cache)
        return .init(
            logits: projectLogits(quantizedVocabularyProjectionInput(hidden, projection: head)))
    }

    /// The window is the architecture's own; the pools grow with the context (one entry per 4
    /// or 128 tokens), so `maxKVSize` and KV quantization do not apply.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        _ = try parameters?.kvCachePlan()
        return makeCache()
    }

    func makeCache() -> [KVCache] {
        layers.map { layer -> KVCache in
            let window = RotatingKVCache(maxSize: configuration.slidingWindow)
            switch layer.attn.ratio {
            case 0: return window
            case 4:
                return CacheList(
                    window, DeepseekV4PoolCache(ratio: 4), DeepseekV4PoolCache(ratio: 4))
            default:
                return CacheList(window, DeepseekV4PoolCache(ratio: layer.attn.ratio))
            }
        }
    }

    /// The multi-token prediction head is not loaded.
    public var excludedCheckpointNamespaces: [String] { ["mtp"] }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        DeepseekV4Weights.sanitize(weights, configuration: configuration)
    }

    public var loraLayers: [Module] { layers }
}

extension DeepseekV4Model: HiddenStateLanguageModel {
    /// Without a cache the call runs on fresh caches.
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let caches = cache ?? makeCache()
        let layerCaches = zip(caches, layers).map {
            DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
        }
        let L = inputs.dim(1)

        var h = embed(inputs)
        h = repeated(expandedDimensions(h, axis: 2), count: configuration.hcMult, axis: 2)

        // One window mask for every layer, from layer 0's window before any update.
        var mask: MLXArray?
        if L > 1,
            case .array(let array) = layerCaches[0].local.makeMask(
                n: L, windowSize: configuration.slidingWindow, returnArray: true)
        {
            mask = array
        }

        for (layer, layerCache) in zip(layers, layerCaches) {
            h = layer(h, mask: mask, cache: layerCache, inputIDs: inputs)
        }
        return norm(hcHead(h))
    }

    /// As the reference, the logits are a float32 projection of the normalized hidden states.
    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        head(hiddenStates.asType(.float32))
    }
}

// MARK: - Weight sanitization

enum DeepseekV4Weights {
    /// Map MLX checkpoint names to module paths: drop `mtp.*` and layers past
    /// `num_hidden_layers`, stack per-expert weights into `switch_mlp`, group `wo_a`, move the
    /// hyper-connection tensors under `attn_hc`/`ffn_hc`/`hc_head`, and make `tid2eid` int32.
    static func sanitize(
        _ weights: [String: MLXArray], configuration: DeepseekV4Configuration
    ) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        var experts = [String: [Int: MLXArray]]()
        let projections = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

        for (key, value) in weights {
            if key.hasPrefix("mtp.") { continue }
            let parts = key.split(separator: ".").map(String.init)
            if parts.count > 1, parts[0] == "layers", let layer = Int(parts[1]),
                layer >= configuration.hiddenLayers
            {
                continue
            }

            // layers.N.ffn.experts.E.wK.suffix
            if parts.count == 7, parts[0] == "layers", parts[2] == "ffn", parts[3] == "experts",
                let expert = Int(parts[4]), let projection = projections[parts[5]]
            {
                let stacked = "layers.\(parts[1]).ffn.switch_mlp.\(projection).\(parts[6])"
                experts[stacked, default: [:]][expert] = value
                continue
            }

            var mapped = key
            switch key {
            case "hc_head_fn": mapped = "hc_head.fn"
            case "hc_head_base": mapped = "hc_head.base"
            case "hc_head_scale": mapped = "hc_head.scale"
            default:
                for branch in ["attn", "ffn"] {
                    for parameter in ["fn", "base", "scale"] {
                        mapped = mapped.replacingOccurrences(
                            of: ".hc_\(branch)_\(parameter)", with: ".\(branch)_hc.\(parameter)")
                    }
                }
            }
            var tensor = value
            if mapped.hasSuffix(".tid2eid") {
                tensor = tensor.asType(.int32)
            }
            if mapped.contains(".attn.wo_a."), tensor.ndim == 2 {
                tensor = tensor.reshaped(configuration.oGroups, configuration.oLoraRank, -1)
            }
            result[mapped] = tensor
        }

        for (key, byExpert) in experts {
            guard byExpert.count == configuration.nRoutedExperts else {
                // Leave the gap for update(verify:) to report.
                for (expert, value) in byExpert { result["\(key).\(expert)"] = value }
                continue
            }
            result[key] = stacked((0 ..< configuration.nRoutedExperts).map { byExpert[$0]! })
        }
        return result
    }
}
