// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Depth attention

// Each threadgroup handles one row and Q depth queries.
private let depthAttentionSource = """
        constexpr int N = K + 1;
        uint row = threadgroup_position_in_grid.x;
        uint rows = threadgroups_per_grid.x;
        uint group = threadgroup_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        const device float* group_queries = queries + size_t(group) * Q * D;

        float acc[N * Q];
        for (int i = 0; i < N * Q; i++) {
            acc[i] = 0.0f;
        }
        for (int d = tid; d < D; d += TG) {
            float x[N];
            for (int i = 0; i < K; i++) {
                x[i] = blocks[(size_t(i) * rows + row) * D + d];
            }
            x[K] = partial[size_t(row) * D + d];
            for (int j = 0; j < Q; j++) {
                float qv = group_queries[j * D + d];
                for (int i = 0; i < N; i++) {
                    acc[i * Q + j] += x[i] * qv;
                }
            }
        }

        threadgroup float buf[N * Q][TG / 32];
        for (int i = 0; i < N * Q; i++) {
            float v = simd_sum(acc[i]);
            if (lane == 0) {
                buf[i][sg] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w[N * Q];
        for (int i = 0; i < N * Q; i++) {
            float t = 0.0f;
            for (int s = 0; s < TG / 32; s++) {
                t += buf[i][s];
            }
            w[i] = t * scale;
        }
        for (int j = 0; j < Q; j++) {
            float m = -INFINITY;
            for (int i = 0; i < N; i++) {
                m = max(m, w[i * Q + j]);
            }
            float z = 0.0f;
            for (int i = 0; i < N; i++) {
                w[i * Q + j] = metal::exp(w[i * Q + j] - m);
                z += w[i * Q + j];
            }
            for (int i = 0; i < N; i++) {
                w[i * Q + j] /= z;
            }
        }

        const uint total = threadgroups_per_grid.y * Q;
        for (int d = tid; d < D; d += TG) {
            float x[N];
            for (int i = 0; i < K; i++) {
                x[i] = blocks[(size_t(i) * rows + row) * D + d];
            }
            x[K] = partial[size_t(row) * D + d];
            for (int j = 0; j < Q; j++) {
                float o = 0.0f;
                for (int i = 0; i < N; i++) {
                    o += w[i * Q + j] * x[i];
                }
                out[(size_t(row) * total + group * Q + j) * D + d] = o;
            }
        }
    """

private final class Inkling2Kernels: Sendable {
    static let shared = Inkling2Kernels()

    let depthAttention: MLXFast.MLXFastKernel
    let relativeAttention: MLXFast.MLXFastKernel
    let expertRouting: MLXFast.MLXFastKernel
    let rowConvolution: MLXFast.MLXFastKernel
    let messagePair: MLXFast.MLXFastKernel
    let messageBlendNorm: MLXFast.MLXFastKernel
    let prefillPrologue: MLXFast.MLXFastKernel

    private init() {
        depthAttention = MLXFast.metalKernel(
            name: "inkling2_depth_attention",
            inputNames: ["blocks", "partial", "queries", "scale"],
            outputNames: ["out"],
            source: depthAttentionSource)
        relativeAttention = MLXFast.metalKernel(
            name: "inkling2_relative_attention",
            inputNames: ["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
            outputNames: ["out"],
            source: relativeAttentionSource,
            ensureRowContiguous: false)
        expertRouting = MLXFast.metalKernel(
            name: "inkling2_expert_routing",
            inputNames: ["logits", "bias", "eps", "scale"],
            outputNames: ["inds", "weights"],
            source: expertRoutingSource)
        rowConvolution = MLXFast.metalKernel(
            name: "inkling2_row_convolution",
            inputNames: [
                "x", "order", "weights", "state", "base", "conv_weight", "taps", "params",
                "divisor",
            ],
            outputNames: ["out", "state_out"], source: rowConvolutionSource,
            header: rowConvolutionHeader, ensureRowContiguous: false)
        messagePair = MLXFast.metalKernel(
            name: "inkling2_message_pair", inputNames: ["input", "messages", "first"],
            outputNames: ["pair"], source: messagePairSource)
        messageBlendNorm = MLXFast.metalKernel(
            name: "inkling2_message_blend_norm",
            inputNames: ["input", "messages", "first", "f", "g", "norm_weight", "eps"],
            outputNames: ["out"], source: messageBlendNormSource)
        prefillPrologue = MLXFast.metalKernel(
            name: "inkling2_prefill_prologue",
            inputNames: [
                "q", "r", "k", "v", "q_scale", "r_scale", "tau_in", "length", "thresholds",
            ],
            outputNames: ["q_out", "r_out", "k_out", "v_out"], source: prefillPrologueSource,
            ensureRowContiguous: false)
    }
}

/// Threads for a threadgroup that handles one row of `dim` values.
private func rowThreads(_ dim: Int) -> Int {
    min(1024, max(32, dim / 8 / 32 * 32))
}

/// Softmax attention of `queries` over the depth sources of each row.
///
/// - Parameters:
///   - blocks: float32 `[K, ..., D]`, the completed blocks.
///   - partial: float32 `[..., D]`, the partial block.
///   - queries: float32 `[Q, D]`.
///   - queriesPerGroup: queries each threadgroup handles; it must divide `Q`.
/// - Returns: float32 `[..., Q, D]`.
func inkling2DepthAttention(
    blocks: MLXArray, partial: MLXArray, queries: MLXArray, scale: Float,
    queriesPerGroup: Int? = nil
) -> MLXArray {
    let d = partial.dim(-1)
    let q = queries.dim(0)
    let perGroup = queriesPerGroup ?? q
    precondition(q % perGroup == 0, "queriesPerGroup must divide the query count")
    let threads = rowThreads(d)
    let rows = partial.size / d
    return Inkling2Kernels.shared.depthAttention(
        [blocks, partial, queries, MLXArray(scale)],
        template: [("K", blocks.dim(0)), ("D", d), ("Q", perGroup), ("TG", threads)],
        grid: (rows * threads, q / perGroup, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [Array(partial.shape.dropLast()) + [q, d]],
        outputDTypes: [.float32]
    )[0]
}

// MARK: - Relative attention

/// Queries per threadgroup, 8 for each of its 4 simdgroups.
private let relativeAttentionQueryBlock = 32
private let relativeAttentionThreads = 128

/// Causal flash attention with a relative position bias and a sink logit per
/// query. The online softmax of each row starts from its sink logit, so the
/// zero-value sink needs no key.
private let relativeAttentionSource = """
        constexpr float LOG2E = 1.4426950408889634f;
        constexpr int BK = 16;
        constexpr int LDK = BK + 8;
        constexpr int LDV = D + 8;
        const int L = sizes[0];
        const int S = sizes[1];
        const int keep = sizes[2];
        const int evicted = sizes[3];
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int batch = threadgroup_position_in_grid.z;
        const int head = threadgroup_position_in_grid.y;
        const int kv_head = head / (HQ / HKV);
        const int q0 = threadgroup_position_in_grid.x * BQ;
        const int start = key_start[batch];
        const float qscale = scale * LOG2E;

        // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
        const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
        const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
        const int row = q0 + int(sg) * 8 + fm;
        const bool row_ok = row < L;
        const int diag = S - L + row;

        // q and k are clipped and v has unit RMS, so half holds them.
        threadgroup half Ks[D * LDK];
        threadgroup half Vs[BK * LDV];

        simdgroup_half8x8 q_frag[D / 8];
        for (int db = 0; db < D / 8; ++db) {
            half2 qv = half2(0.0h);
            if (row_ok) {
                const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                    + row * q_strides[2] + db * 8 + fn;
                qv = half2(float(qp[0]), float(qp[1]));
            }
            q_frag[db].thread_elements()[0] = qv.x;
            q_frag[db].thread_elements()[1] = qv.y;
        }
        simdgroup_float8x8 acc[D / 8];
        for (int db = 0; db < D / 8; ++db) {
            acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
        float row_sum = 1.0f;
        const int bias_row = batch * L + (row_ok ? row : 0);
        const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

        const int first_diag = S - L + q0;
        const int last_diag = S - L + min(q0 + BQ, L) - 1;
        int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
        first_key = max(first_key, start);
        const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
        const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

        for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int c = tid; c < BK * D / 8; c += TG) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                if (key < S) {
                    auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                    auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                    kv[0] = kp[0];
                    kv[1] = kp[1];
                    vv[0] = vp[0];
                    vv[1] = vp[1];
                }
                for (int j = 0; j < 8; ++j) {
                    Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                    Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q0 + int(sg) * 8 >= L) {
                continue;
            }

            simdgroup_float8x8 scores[BK / 8];
            for (int kb = 0; kb < BK / 8; ++kb) {
                scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (int db = 0; db < D / 8; ++db) {
                    simdgroup_half8x8 k_frag;
                    simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                    simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
                }
            }

            float tile_max = -INFINITY;
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = diag - key + (key < keep ? evicted : 0);
                    const bool visible = row_ok && d >= 0 && key >= start
                        && (WINDOW == 0 || d < WINDOW);
                    float s = -INFINITY;
                    if (visible) {
                        s = scores[kb].thread_elements()[e] * qscale;
                        if (d < R) {
                            s += float(bias[d]) * LOG2E;
                        }
                    }
                    scores[kb].thread_elements()[e] = s;
                    tile_max = max(tile_max, s);
                }
            }
            // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
            const float new_max = max(row_max, tile_max);
            const float factor = fast::exp2(row_max - new_max);
            float tile_sum = 0.0f;
            simdgroup_half8x8 probs[BK / 8];
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                    probs[kb].thread_elements()[e] = half(p);
                    tile_sum += p;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, 1);
            tile_sum += simd_shuffle_xor(tile_sum, 8);
            row_sum = row_sum * factor + tile_sum;
            row_max = new_max;
            for (int db = 0; db < D / 8; ++db) {
                acc[db].thread_elements()[0] *= factor;
                acc[db].thread_elements()[1] *= factor;
                for (int kb = 0; kb < BK / 8; ++kb) {
                    simdgroup_half8x8 v_frag;
                    simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                    simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
                }
            }
        }

        if (row_ok) {
            device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
            for (int db = 0; db < D / 8; ++db) {
                op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
                op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
            }
        }
    """

/// Causal attention with a relative position bias and a sink per query.
///
/// - Parameters:
///   - q: bfloat16 `[B, Hq, L, D]`, the queries of the last `L` keys.
///   - k: bfloat16 `[B, Hkv, S, D]`.
///   - v: bfloat16 `[B, Hkv, S, D]`.
///   - rel: `[B, L, Hq, R]`, the bias at distances below `R`.
///   - sinks: `[B, L, Hq]`.
///   - window: the sliding window, or 0 for full attention.
///   - keyStart: int32 `[B]`, the first visible key of each row.
///   - keep: the pinned keys at the front, `evicted` positions older than their slots.
/// - Returns: `[B, L, Hq, D]`.
func inkling2RelativeAttention(
    q: MLXArray, k: MLXArray, v: MLXArray, rel: MLXArray, sinks: MLXArray,
    scale: Float, window: Int, keyStart: MLXArray, keep: Int, evicted: Int
) -> MLXArray {
    let (b, hq, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let (hkv, s) = (k.dim(1), k.dim(2))
    let block = relativeAttentionQueryBlock
    let threads = relativeAttentionThreads
    return Inkling2Kernels.shared.relativeAttention(
        [
            q, k, v, contiguous(rel), contiguous(sinks.asType(.float32)),
            contiguous(keyStart.asType(.int32)),
            MLXArray([Int32(l), Int32(s), Int32(keep), Int32(evicted)]),
            MLXArray(scale),
        ],
        template: [
            ("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("R", rel.dim(-1)),
            ("WINDOW", window), ("BQ", block), ("TG", threads),
        ],
        grid: ((l + block - 1) / block * threads, hq, b),
        threadGroup: (threads, 1, 1),
        outputShapes: [[b, l, hq, d]],
        outputDTypes: [q.dtype]
    )[0]
}

// MARK: - Prefill attention inputs

/// One simdgroup per head and row: the L2 clip of the queries, relative states and keys, the
/// log-length scaling of full layers and the RMS norm of the values.
private let prefillPrologueSource = """
        constexpr int E = DH / 32;
        const int unit = threadgroup_position_in_grid.x;
        const int row = threadgroup_position_in_grid.y;
        const int L = length[0];
        const int b = row / L;
        const int t = row % L;
        const uint lane = thread_position_in_threadgroup.x;
        const float tau = FULL ? tau_in[t] : 1.0f;

        if (unit < HQ) {
            const int h = unit;
            const auto src = q + b * q_strides[0] + t * q_strides[1];
            float xv[E];
            float ss = 0.0f;
            for (int e = 0; e < E; e++) {
                const int c = h * DH + lane * E + e;
                xv[e] = float(src[c * q_strides[2]]) * float(q_scale[c]);
                ss += xv[e] * xv[e];
            }
            const float f = min(thresholds[0] / (sqrt(simd_sum(ss)) + 1e-6f), 1.0f);
            for (int e = 0; e < E; e++) {
                T y = T(xv[e] * f);
                if (FULL) {
                    y = T(float(y) * tau);
                }
                q_out[(size_t(row) * HQ + h) * DH + lane * E + e] = y;
            }
            return;
        }
        if (unit < 2 * HQ) {
            const int h = unit - HQ;
            const auto src = r + b * r_strides[0] + t * r_strides[1];
            float xv = 0.0f;
            if (lane < DR) {
                const int c = h * DR + lane;
                xv = float(src[c * r_strides[2]]) * float(r_scale[c]);
            }
            const float f = min(thresholds[1] / (sqrt(simd_sum(xv * xv)) + 1e-6f), 1.0f);
            if (lane < DR) {
                T y = T(xv * f);
                if (FULL) {
                    y = T(float(y) * tau);
                }
                r_out[(size_t(row) * HQ + h) * DR + lane] = y;
            }
            return;
        }
        const bool is_key = unit < 2 * HQ + HKV;
        const int h = unit - 2 * HQ - (is_key ? 0 : HKV);
        const auto src = is_key ? k + b * k_strides[0] + t * k_strides[1]
                                : v + b * v_strides[0] + t * v_strides[1];
        const int stride = is_key ? k_strides[2] : v_strides[2];
        float xv[E];
        float ss = 0.0f;
        for (int e = 0; e < E; e++) {
            xv[e] = float(src[(h * DH + lane * E + e) * stride]);
            ss += xv[e] * xv[e];
        }
        const float total = simd_sum(ss);
        const float f = is_key
            ? min(thresholds[0] / (sqrt(total) + 1e-6f), 1.0f)
            : metal::precise::rsqrt(total / DH + 1e-6f);
        device T* out = is_key ? k_out : v_out;
        for (int e = 0; e < E; e++) {
            out[(size_t(row) * HKV + h) * DH + lane * E + e] = T(xv[e] * f);
        }
    """

/// The attention inputs of a prefill chunk from its projections and convolutions.
///
/// - Parameters:
///   - q: `[B, L, Hq * DH]`; `r`: `[B, L, Hq * DR]`; `k`, `v`: `[B, L, Hkv * DH]`.
///   - tau: float32 `[L]`, the log-length scaling of full layers.
/// - Returns: queries `[B, L, Hq, DH]`, relative states `[B, L, Hq, DR]`, keys and values
///   `[B, L, Hkv, DH]`.
func inkling2PrefillPrologue(
    q: MLXArray, r: MLXArray, k: MLXArray, v: MLXArray, qScale: MLXArray, rScale: MLXArray,
    tau: MLXArray?, heads: Int, kvHeads: Int, headDim: Int, relativeDim: Int,
    qkThreshold: Float, relativeThreshold: Float
) -> [MLXArray] {
    let (b, l) = (q.dim(0), q.dim(1))
    return Inkling2Kernels.shared.prefillPrologue(
        [
            q, r, k, v, qScale, rScale, tau ?? MLXArray([Float(1)]), MLXArray([Int32(l)]),
            MLXArray([qkThreshold, relativeThreshold]),
        ],
        template: [
            ("T", q.dtype), ("DH", headDim), ("HQ", heads), ("HKV", kvHeads),
            ("DR", relativeDim), ("FULL", tau != nil),
        ],
        grid: ((2 * heads + 2 * kvHeads) * 32, b * l, 1), threadGroup: (32, 1, 1),
        outputShapes: [
            [b, l, heads, headDim], [b, l, heads, relativeDim], [b, l, kvHeads, headDim],
            [b, l, kvHeads, headDim],
        ],
        outputDTypes: Array(repeating: q.dtype, count: 4))
}

// MARK: - Prefill message gates

/// The RMS-normalized input and message of one stream and row, side by side.
private let messagePairSource = """
        constexpr int NR = TG / 32;
        const uint unit = threadgroup_position_in_grid.x;
        const uint rows = threadgroups_per_grid.x / S;
        const uint s = unit / rows;
        const uint row = unit % rows;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const device float* in = input + (size_t(row) * Q + (Q == 1 ? 0 : s)) * D;
        const device float* msg = messages + (size_t(row) * MSTRIDE + first + s) * D;
        float a = 0.0f;
        float b = 0.0f;
        for (int d = tid; d < D; d += TG) {
            a += in[d] * in[d];
            b += msg[d] * msg[d];
        }
        threadgroup float red[2][NR];
        a = simd_sum(a);
        b = simd_sum(b);
        if (lane == 0) {
            red[0][sg] = a;
            red[1][sg] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a = 0.0f;
        b = 0.0f;
        for (int i = 0; i < NR; i++) {
            a += red[0][i];
            b += red[1][i];
        }
        const float ia = metal::precise::rsqrt(a / D + 1e-6f);
        const float ib = metal::precise::rsqrt(b / D + 1e-6f);
        device T* out = pair + size_t(unit) * 2 * D;
        for (int d = tid; d < D; d += TG) {
            out[d] = T(in[d] * ia);
            out[D + d] = T(msg[d] * ib);
        }
    """

/// `(1 - tanh(f)) * input + tanh(g) * message` of one stream and row, then its RMS norm with
/// the stream's weight.
private let messageBlendNormSource = """
        constexpr int NR = TG / 32;
        constexpr int PER = (D + TG - 1) / TG;
        const uint unit = threadgroup_position_in_grid.x;
        const uint rows = threadgroups_per_grid.x / S;
        const uint s = unit / rows;
        const uint row = unit % rows;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const device float* in = input + (size_t(row) * Q + (Q == 1 ? 0 : s)) * D;
        const device float* msg = messages + (size_t(row) * MSTRIDE + first + s) * D;
        float o[PER];
        float ss = 0.0f;
        for (int i = 0; i < PER; i++) {
            const int d = tid + i * TG;
            o[i] = 0.0f;
            if (d < D) {
                const size_t j = size_t(unit) * D + d;
                o[i] = (1.0f - metal::precise::tanh(f[j])) * in[d]
                    + metal::precise::tanh(g[j]) * msg[d];
                ss += o[i] * o[i];
            }
        }
        threadgroup float red[NR];
        ss = simd_sum(ss);
        if (lane == 0) {
            red[sg] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = 0.0f;
        for (int i = 0; i < NR; i++) {
            ss += red[i];
        }
        const float inv = metal::precise::rsqrt(ss / D + eps[0]);
        for (int i = 0; i < PER; i++) {
            const int d = tid + i * TG;
            if (d < D) {
                const float w = float(norm_weight[(NW == 1 ? 0 : s) * D + d]);
                out[size_t(unit) * D + d] = T(w * (o[i] * inv));
            }
        }
    """

/// The message gates of a prefill chunk as normalized sublayer inputs:
/// `norm((1 - tanh(h forget)) * input + tanh(h gain) * message)`, with `h` the down projection
/// of the RMS-normalized input and message.
///
/// - Parameters:
///   - inputs: float32 `[N, Q, D]`; stream `s` reads input `s`, or input 0 when `Q` is 1.
///   - messages: float32 `[N, M, D]`, the messages of all loop layers.
///   - first: the message of stream 0.
///   - down: bfloat16 `[S, R, 2D]`.
///   - forget: bfloat16 `[S, R, D]`.
///   - gain: bfloat16 `[S, R, D]`.
///   - normWeight: `[S or 1, D]`.
/// - Returns: `[S, N, D]` of `down`'s dtype.
func inkling2MessageGates(
    inputs: MLXArray, messages: MLXArray, first: Int, down: MLXArray, forget: MLXArray,
    gain: MLXArray, normWeight: MLXArray, eps: Float
) -> MLXArray {
    let (streams, rows, d) = (down.dim(0), messages.dim(0), messages.dim(2))
    let threads = 256
    let template: [(String, any KernelTemplateArg)] = [
        ("T", down.dtype), ("D", d), ("S", streams), ("Q", inputs.dim(1)), ("TG", threads),
        ("MSTRIDE", messages.dim(1)), ("NW", normWeight.dim(0)),
    ]
    let kernels = Inkling2Kernels.shared
    let firstArray = MLXArray(Int32(first))
    let pair = kernels.messagePair(
        [inputs, messages, firstArray], template: template,
        grid: (streams * rows * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[streams, rows, 2 * d]], outputDTypes: [down.dtype])[0]
    let h = matmul(pair, down.transposed(0, 2, 1)).asType(.float32)
    let f = matmul(h, forget.asType(.float32))
    let g = matmul(h, gain.asType(.float32))
    return kernels.messageBlendNorm(
        [inputs, messages, firstArray, f, g, normWeight, MLXArray([eps])], template: template,
        grid: (streams * rows * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[streams, rows, d]], outputDTypes: [down.dtype])[0]
}

// MARK: - Prefill convolution

/// The input of a short convolution at row `t` of batch row `b`. Rows before the chunk come from
/// the state; with `K` experts, a row is the weighted sum of its expert outputs.
private let rowConvolutionHeader = """
    template <typename T, int K, bool SORTED, bool HAS_STATE, typename X, typename O, typename W,
        typename S, typename XS>
    inline float inkling2_conv_input(
        X x, XS x_strides, O order, W weights, S state, int b, int t, int L, int C, int d) {
        if (t < 0) {
            return HAS_STATE ? float(state[(size_t(b) * 3 + 3 + t) * C + d]) : 0.0f;
        }
        if (K == 0) {
            return float(x[b * x_strides[0] + t * x_strides[1] + d * x_strides[2]]);
        }
        const size_t row = size_t(b) * L + t;
        float m = 0.0f;
        for (int k = 0; k < K; k++) {
            const size_t i = row * K + k;
            m += float(x[size_t(SORTED ? order[i] : i) * C + d]) * float(weights[i]);
        }
        return float(T(m));
    }
    """

/// One thread per channel and block of `RB` rows: `(conv(x) + x) / divisor`, plus `base` when
/// set, and the last three inputs as the next state. Dynamic taps add a projection of the layer
/// input, per row and head, to the static taps, and skip the rounding of the convolution.
private let rowConvolutionSource = """
        const int d = thread_position_in_grid.x;
        const int t0 = thread_position_in_grid.y * RB;
        const int b = thread_position_in_grid.z;
        const int L = params[0];
        if (d >= C || t0 >= L) {
            return;
        }
        #define INPUT(t) inkling2_conv_input<T, K, SORTED, HAS_STATE>( \\
            x, x_strides, order, weights, state, b, t, L, C, d)
        if (t0 == 0) {
            for (int j = 0; j < 3; j++) {
                state_out[(size_t(b) * 3 + j) * C + d] = T(INPUT(L - 3 + j));
            }
        }
        float w[4];
        for (int k = 0; k < 4; k++) {
            w[k] = DYN ? float(conv_weight[k * C + d]) : float(conv_weight[d * 4 + k]);
        }
        float p0 = INPUT(t0 - 3);
        float p1 = INPUT(t0 - 2);
        float p2 = INPUT(t0 - 1);
        for (int t = t0; t < min(L, t0 + RB); t++) {
            const float xt = INPUT(t);
            float y;
            if (DYN) {
                const auto tp = taps + b * taps_strides[0] + t * taps_strides[1]
                    + (d / (C / HEADS)) * 4 * taps_strides[2];
                float a = xt * (w[0] + float(tp[0]));
                a += p2 * (w[1] + float(tp[taps_strides[2]]));
                a += p1 * (w[2] + float(tp[2 * taps_strides[2]]));
                a += p0 * (w[3] + float(tp[3 * taps_strides[2]]));
                y = float(T((a + xt) / divisor[0]));
            } else {
                const float a = p0 * w[0] + p1 * w[1] + p2 * w[2] + xt * w[3];
                y = float(T((float(T(a)) + xt) / divisor[0]));
            }
            if (HAS_BASE) {
                y += float(base[b * base_strides[0] + t * base_strides[1] + d * base_strides[2]]);
            }
            out[(size_t(b) * L + t) * C + d] = OT(y);
            p0 = p1;
            p1 = p2;
            p2 = xt;
        }
        #undef INPUT
    """

/// The short convolution `(conv(x) + x) / sqrt(5)` of a prefill chunk, plus `base` when set.
///
/// - Parameters:
///   - x: `[B, L, C]`, or with `experts` the expert outputs `[B * L * K, C]`, whose weighted sum
///     per row is the input.
///   - state: the previous three inputs `[B, 3, C]`, or `nil` for zeros.
///   - weight: `[C, 4, 1]`, or with `dynamicTaps` the static taps `[4, C]`.
///   - experts: the routing weights `[B, L, K]` and, when the outputs are sorted, the row of
///     each output.
///   - dynamicTaps: `[B, L, heads * 4]`, added to the static taps of each head. The output is
///     then `(conv(x) + x) / 3`.
/// - Returns: `[B, L, C]` of `outputType`, and the next state `[B, 3, C]`.
func inkling2RowConvolution(
    _ x: MLXArray, state: MLXArray?, weight: MLXArray, base: MLXArray? = nil,
    outputType: DType, experts: (weights: MLXArray, order: MLXArray?)? = nil,
    dynamicTaps: MLXArray? = nil
) -> (output: MLXArray, state: MLXArray) {
    let (b, l, k) =
        experts.map { ($0.weights.dim(0), $0.weights.dim(1), $0.weights.dim(2)) }
        ?? (x.dim(0), x.dim(1), 0)
    let c = x.dim(-1)
    let dtype = x.dtype
    let rowsPerThread = 16
    let placeholder = MLXArray([Float(0)])
    let out = Inkling2Kernels.shared.rowConvolution(
        [
            experts == nil ? x : contiguous(x),
            experts?.order.map { contiguous($0) } ?? placeholder,
            experts.map { contiguous($0.weights) } ?? placeholder,
            state.map { contiguous($0) } ?? placeholder, base ?? placeholder, contiguous(weight),
            dynamicTaps ?? placeholder, MLXArray([Int32(l)]),
            MLXArray([Float(dynamicTaps == nil ? 5 : 9).squareRoot()]),
        ],
        template: [
            ("T", dtype), ("OT", outputType), ("C", c), ("K", k),
            ("SORTED", experts?.order != nil), ("HAS_STATE", state != nil),
            ("HAS_BASE", base != nil), ("DYN", dynamicTaps != nil),
            ("HEADS", (dynamicTaps?.dim(-1) ?? 4) / 4), ("RB", rowsPerThread),
        ],
        grid: (c, (l + rowsPerThread - 1) / rowsPerThread, b), threadGroup: (256, 1, 1),
        outputShapes: [[b, l, c], [b, 3, c]], outputDTypes: [outputType, dtype])
    return (out[0], out[1])
}

// MARK: - Decode attention

/// The distance of slot `j` from the newest key. The first `keep` slots are pinned and
/// `evicted` positions older than their slots. A wrapped ring over the other slots holds the
/// newest key at `ring - 1`; `ring` is negative when the keys are oldest first.
private let keyDistanceHeader = """
    inline int key_distance(int j, int S, int ring, int keep, int evicted) {
        if (j < keep) {
            return S - 1 - j + evicted;
        }
        return ring >= 0 ? (ring - 1 - j + S - keep) % (S - keep) : S - 1 - j;
    }
    """

/// One query per head over all keys. The bias of each distance comes from the head's relative
/// state and the projection table; the output is gated by `sigmoid(gate)`.
private let relativeDecodeSource = """
        constexpr int E = D / 32;
        const int bh = threadgroup_position_in_grid.x;
        const int b = bh / HQ;
        const int h = bh % HQ;
        const int kv_head = h / (HQ / HKV);
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int S = params[0];
        const int ring = params[1];
        const int keep = params[2];
        const int evicted = params[3];

        threadgroup float bias[R];
        const device T* rq = rel + size_t(bh) * DR;
        for (int d = tid; d < R; d += NSG * 32) {
            float s = 0.0f;
            for (int r = 0; r < DR; r++) {
                s += float(rq[r]) * float(table[r * R + d]);
            }
            bias[d] = float(T(s));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qv[E];
        for (int i = 0; i < E; i++) {
            qv[i] = float(q[size_t(bh) * D + lane * E + i]) * scale;
        }
        const device T* k_head = k + b * k_strides[0] + kv_head * k_strides[1] + lane * E;
        const device T* v_head = v + b * v_strides[0] + kv_head * v_strides[1] + lane * E;

        float m = -INFINITY;
        float l = 0.0f;
        float acc[E];
        for (int i = 0; i < E; i++) {
            acc[i] = 0.0f;
        }
        for (int j = int(sg); j < S; j += NSG) {
            const int d = key_distance(j, S, ring, keep, evicted);
            if (WINDOW > 0 && d >= WINDOW) {
                continue;
            }
            const device T* kp = k_head + j * k_strides[2];
            float s = 0.0f;
            for (int i = 0; i < E; i++) {
                s += qv[i] * float(kp[i]);
            }
            s = simd_sum(s);
            if (d < R) {
                s += bias[d];
            }
            const float mn = max(m, s);
            const float f = fast::exp(m - mn);
            const float p = fast::exp(s - mn);
            l = l * f + p;
            const device T* vp = v_head + j * v_strides[2];
            for (int i = 0; i < E; i++) {
                acc[i] = acc[i] * f + p * float(vp[i]);
            }
            m = mn;
        }

        threadgroup float tm[NSG];
        threadgroup float tl[NSG];
        threadgroup float tacc[NSG * D];
        if (lane == 0) {
            tm[sg] = m;
            tl[sg] = l;
        }
        for (int i = 0; i < E; i++) {
            tacc[sg * D + lane * E + i] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint t = tid;
        if (t >= D) {
            return;
        }
        const float sink = float(sinks[b * sinks_strides[0] + h]);
        float big = sink;
        for (int g = 0; g < NSG; g++) {
            big = max(big, tm[g]);
        }
        float total = fast::exp(sink - big);
        float o = 0.0f;
        for (int g = 0; g < NSG; g++) {
            const float w = fast::exp(tm[g] - big);
            total += tl[g] * w;
            o += tacc[g * D + t] * w;
        }
        const float x = float(gate[b * gate_strides[0] + h * D + t]);
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
        out[size_t(bh) * D + t] = T(float(T(o / total)) * (x < 0 ? y : 1.0f - y));
    """

/// All query heads of one key head over one split of `PER` keys, so each key is read once.
/// The relative bias is computed only for the split's keys.
private let groupedDecodeSplitSource = """
        constexpr int E = D / 32;
        constexpr int G = HQ / HKV;
        const int bk = threadgroup_position_in_grid.x;
        const int split = threadgroup_position_in_grid.y;
        const int b = bk / HKV;
        const int kv_head = bk % HKV;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int S = params[0];
        const int ring = params[1];
        const int keep = params[2];
        const int evicted = params[3];
        const int k0 = split * PER;
        const int k1 = min(S, k0 + PER);

        threadgroup float bias[G * PER];
        for (int t = tid; t < G * PER; t += NSG * 32) {
            const int j = k0 + t % PER;
            const int d = key_distance(j, S, ring, keep, evicted);
            if (j < k1 && d < R) {
                const device T* rq = rel + (size_t(b) * HQ + kv_head * G + t / PER) * DR;
                float s = 0.0f;
                for (int r = 0; r < DR; r++) {
                    s += float(rq[r]) * float(table[r * R + d]);
                }
                bias[t] = float(T(s));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qv[G][E];
        float m[G];
        float l[G];
        float acc[G][E];
        for (int g = 0; g < G; g++) {
            m[g] = -INFINITY;
            l[g] = 0.0f;
            for (int i = 0; i < E; i++) {
                qv[g][i] = float(q[(size_t(b) * HQ + kv_head * G + g) * D + lane * E + i]) * scale;
                acc[g][i] = 0.0f;
            }
        }
        const device T* k_head = k + b * k_strides[0] + kv_head * k_strides[1] + lane * E;
        const device T* v_head = v + b * v_strides[0] + kv_head * v_strides[1] + lane * E;
        for (int j = k0 + int(sg); j < k1; j += NSG) {
            const int d = key_distance(j, S, ring, keep, evicted);
            if (WINDOW > 0 && d >= WINDOW) {
                continue;
            }
            float kf[E];
            float vf[E];
            for (int i = 0; i < E; i++) {
                kf[i] = float(k_head[j * k_strides[2] + i]);
                vf[i] = float(v_head[j * v_strides[2] + i]);
            }
            for (int g = 0; g < G; g++) {
                float s = 0.0f;
                for (int i = 0; i < E; i++) {
                    s += qv[g][i] * kf[i];
                }
                s = simd_sum(s);
                if (d < R) {
                    s += bias[g * PER + j - k0];
                }
                const float mn = max(m[g], s);
                const float f = fast::exp(m[g] - mn);
                const float p = fast::exp(s - mn);
                l[g] = l[g] * f + p;
                for (int i = 0; i < E; i++) {
                    acc[g][i] = acc[g][i] * f + p * vf[i];
                }
                m[g] = mn;
            }
        }

        threadgroup float tm[NSG];
        threadgroup float tl[NSG];
        threadgroup float tacc[NSG * D];
        for (int g = 0; g < G; g++) {
            if (g > 0) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                tm[sg] = m[g];
                tl[sg] = l[g];
            }
            for (int i = 0; i < E; i++) {
                tacc[sg * D + lane * E + i] = acc[g][i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < D) {
                float big = -INFINITY;
                for (int n = 0; n < NSG; n++) {
                    big = max(big, tm[n]);
                }
                float total = 0.0f;
                float o = 0.0f;
                for (int n = 0; n < NSG; n++) {
                    const float w = big == -INFINITY ? 0.0f : fast::exp(tm[n] - big);
                    total += tl[n] * w;
                    o += tacc[n * D + tid] * w;
                }
                const size_t row = (size_t(b) * HQ + kv_head * G + g) * SPLITS + split;
                partial[row * D + tid] = o;
                if (tid == 0) {
                    stats[row * 2] = big;
                    stats[row * 2 + 1] = total;
                }
            }
        }
    """

private let relativeDecodeReduceSource = """
        const int bh = threadgroup_position_in_grid.x;
        const int b = bh / HQ;
        const int h = bh % HQ;
        const uint t = thread_position_in_threadgroup.x;
        const float sink = float(sinks[b * sinks_strides[0] + h]);
        float big = sink;
        for (int s = 0; s < SPLITS; s++) {
            big = max(big, stats[(size_t(bh) * SPLITS + s) * 2]);
        }
        float total = fast::exp(sink - big);
        float o = 0.0f;
        for (int s = 0; s < SPLITS; s++) {
            const size_t row = size_t(bh) * SPLITS + s;
            const float w = fast::exp(stats[row * 2] - big);
            total += stats[row * 2 + 1] * w;
            o += partial[row * D + t] * w;
        }
        const float x = float(gate[b * gate_strides[0] + h * D + t]);
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
        out[size_t(bh) * D + t] = T(float(T(o / total)) * (x < 0 ? y : 1.0f - y));
    """

/// Gated attention of one query per head with a relative position bias and a sink.
///
/// - Parameters:
///   - q: `[B, Hq, D]`, the clipped queries.
///   - k: `[B, Hkv, S, D]` with contiguous rows.
///   - v: `[B, Hkv, S, D]` with contiguous rows.
///   - relative: `[B, Hq, DR]`, the clipped relative states.
///   - table: `[DR, R]`, the bias of each distance below `R` is `relative @ table`.
///   - sinks: `[B, 1, Hq]` with a contiguous last axis.
///   - gate: `[B, 1, Hq * D]` with a contiguous last axis.
///   - ringIndex: the slot after the newest key when the keys are a wrapped ring.
///   - window: the sliding window, or 0 for full attention.
///   - keep: the pinned keys at the front, `evicted` positions older than their slots.
/// - Returns: `[B, Hq * D]`.
func inkling2DecodeAttention(
    q: MLXArray, k: MLXArray, v: MLXArray, relative: MLXArray, table: MLXArray,
    sinks: MLXArray, gate: MLXArray, scale: Float, ringIndex: Int?, window: Int, keep: Int,
    evicted: Int
) -> MLXArray {
    let (b, hq, d) = (q.dim(0), q.dim(1), q.dim(2))
    let (hkv, s) = (k.dim(1), k.dim(2))
    let kernels = Inkling2DecodeKernels.shared
    let params = MLXArray([Int32(s), Int32(ringIndex ?? -1), Int32(keep), Int32(evicted)])
    let inputs: [MLXArray] = [
        q, k, v, relative, table, sinks, gate, params, MLXArray(scale),
    ]
    var template: [(String, any KernelTemplateArg)] = [
        ("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("R", table.dim(1)),
        ("DR", table.dim(0)), ("WINDOW", window),
    ]
    // Up to a sliding window, one threadgroup per head is fastest; longer caches are split.
    if s <= 1024 {
        let simdgroups = 32
        return kernels.single(
            inputs, template: template + [("NSG", simdgroups)],
            grid: (b * hq * simdgroups * 32, 1, 1), threadGroup: (simdgroups * 32, 1, 1),
            outputShapes: [[b, hq * d]], outputDTypes: [q.dtype]
        )[0]
    }
    let (simdgroups, keysPerSplit) = (8, 256)
    let splits = (s + keysPerSplit - 1) / keysPerSplit
    template += [("NSG", simdgroups), ("PER", keysPerSplit), ("SPLITS", splits)]
    let partial = kernels.split(
        inputs, template: template, grid: (b * hkv * simdgroups * 32, splits, 1),
        threadGroup: (simdgroups * 32, 1, 1),
        outputShapes: [[b * hq * splits, d], [b * hq * splits, 2]],
        outputDTypes: [.float32, .float32])
    return kernels.reduce(
        [partial[0], partial[1], sinks, gate],
        template: [("T", q.dtype), ("D", d), ("HQ", hq), ("SPLITS", splits)],
        grid: (b * hq * d, 1, 1), threadGroup: (d, 1, 1), outputShapes: [[b, hq * d]],
        outputDTypes: [q.dtype]
    )[0]
}

// MARK: - Depth mix

/// One row per threadgroup: the residual `base + conv(x)`, its RMS norm, the attention of
/// `Q` queries over the blocks and the normalized residual, then the RMS norm of each
/// readout. Each thread reads back only the values it wrote.
private let depthMixSource = """
        constexpr int N = K + 1;
        constexpr int NR = TG / 32;
        const uint row = threadgroup_position_in_grid.x;
        const uint rows = threadgroups_per_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const float scale = params[0];
        const float eps = params[1];
        const float divisor = params[2];
        const size_t base_row = size_t(row) * D;

        float acc[N * Q];
        for (int i = 0; i < N * Q; i++) {
            acc[i] = 0.0f;
        }
        float ss = 0.0f;
        for (int d = tid; d < D; d += TG) {
            float r = 0.0f;
            if (HAS_BASE) {
                r = float(base[base_row + d]);
            }
            if (HAS_CONV) {
                const size_t sb = size_t(row) * 3 * D + d;
                const float s0 = float(state[sb]);
                const float s1 = float(state[sb + D]);
                const float s2 = float(state[sb + 2 * D]);
                const float xv = float(x[base_row + d]);
                auto w = conv_weight + size_t(d) * 4;
                const float a = s0 * float(w[0]) + s1 * float(w[1]) + s2 * float(w[2])
                    + xv * float(w[3]);
                r += float(T((float(T(a)) + xv) / divisor));
                state_out[sb] = state[sb + D];
                state_out[sb + D] = state[sb + 2 * D];
                state_out[sb + 2 * D] = x[base_row + d];
            }
            residual_out[base_row + d] = r;
            ss += r * r;
            for (int j = 0; j < Q; j++) {
                const float qv = queries[j * D + d];
                for (int i = 0; i < K; i++) {
                    acc[i * Q + j] += blocks[(size_t(i) * rows + row) * D + d] * qv;
                }
                acc[K * Q + j] += r * qv;
            }
        }

        threadgroup float red[N * Q + 1][NR];
        for (int i = 0; i < N * Q; i++) {
            const float v = simd_sum(acc[i]);
            if (lane == 0) {
                red[i][sg] = v;
            }
        }
        const float ssv = simd_sum(ss);
        if (lane == 0) {
            red[N * Q][sg] = ssv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One thread per sum, so no thread reads every partial.
        threadgroup float totals[N * Q + 1];
        if (tid <= N * Q) {
            float t = 0.0f;
            for (int s = 0; s < NR; s++) {
                t += red[tid][s];
            }
            totals[tid] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv = metal::precise::rsqrt(totals[N * Q] / D + eps);
        float w[N * Q];
        for (int i = 0; i < N * Q; i++) {
            w[i] = totals[i] * scale * (i >= K * Q ? inv : 1.0f);
        }
        for (int j = 0; j < Q; j++) {
            float m = -INFINITY;
            for (int i = 0; i < N; i++) {
                m = max(m, w[i * Q + j]);
            }
            float z = 0.0f;
            for (int i = 0; i < N; i++) {
                w[i * Q + j] = metal::exp(w[i * Q + j] - m);
                z += w[i * Q + j];
            }
            for (int i = 0; i < N; i++) {
                w[i * Q + j] /= z;
            }
        }

        threadgroup float outs[NORM_OUT ? Q * D : 1];
        float sq[Q];
        for (int j = 0; j < Q; j++) {
            sq[j] = 0.0f;
        }
        for (int d = tid; d < D; d += TG) {
            const float p = residual_out[base_row + d] * inv;
            normed_out[base_row + d] = p;
            for (int j = 0; j < Q; j++) {
                float o = w[K * Q + j] * p;
                for (int i = 0; i < K; i++) {
                    o += w[i * Q + j] * blocks[(size_t(i) * rows + row) * D + d];
                }
                if (NORM_OUT) {
                    outs[j * D + d] = o;
                    sq[j] += o * o;
                } else {
                    readout[(size_t(row) * Q + j) * D + d] = TO(o);
                }
            }
        }
        if (!NORM_OUT) {
            return;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < Q; j++) {
            const float v = simd_sum(sq[j]);
            if (lane == 0) {
                red[j][sg] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < Q) {
            float t = 0.0f;
            for (int s = 0; s < NR; s++) {
                t += red[tid][s];
            }
            totals[tid] = metal::precise::rsqrt(t / D + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float invs[Q];
        for (int j = 0; j < Q; j++) {
            invs[j] = totals[j];
        }
        for (int d = tid; d < D; d += TG) {
            for (int j = 0; j < Q; j++) {
                const float g = float(norm_weight[(NW == 1 ? 0 : j) * D + d]);
                const T y = T(g * (outs[j * D + d] * invs[j]));
                readout[(size_t(row) * Q + j) * D + d] = TO(y);
                if (F32_COPY) {
                    readout_f32[(size_t(row) * Q + j) * D + d] = float(y);
                }
            }
        }
    """

struct Inkling2DepthMix {
    /// float32 `[B, D]`, `base + conv(x)`.
    var residual: MLXArray
    /// float32 `[B, D]`, the residual over its RMS.
    var normed: MLXArray
    /// `[B, Q, D]`: bfloat16 and normalized with `normWeight`, otherwise the float32 readout.
    var readout: MLXArray
    /// float32 `[B, Q, D]`, the normalized readout read back, when requested.
    var readoutF32: MLXArray?
    /// The next convolution state.
    var state: MLXArray?
}

/// The attention of `queries` over the blocks and the normalized residual, for one token per
/// row.
///
/// - Parameters:
///   - blocks: float32 `[K, B, D]`.
///   - base: `[B, D]` added to the residual.
///   - convolution: bfloat16 input `[B, D]`, state `[B, 3, D]` and weight `[D, 4, 1]` of a
///     short convolution added to the residual.
///   - queries: float32 `[Q, D]`.
///   - normWeight: bfloat16 `[1 or Q, D]`, or `nil` for the float32 readout.
func inkling2DepthMix(
    blocks: MLXArray, base: MLXArray?,
    convolution: (x: MLXArray, state: MLXArray, weight: MLXArray)?,
    queries: MLXArray, normWeight: MLXArray?, f32Copy: Bool = false, eps: Float
) -> Inkling2DepthMix {
    let d = queries.dim(-1)
    let q = queries.dim(0)
    let rows = blocks.dim(1)
    let threads = 1024
    let dummy = MLXArray.zeros([1], dtype: .bfloat16)
    let norm = normWeight != nil
    let out = Inkling2DecodeKernels.shared.depthMix(
        [
            blocks, base ?? dummy, convolution?.x ?? dummy, convolution?.state ?? dummy,
            convolution?.weight ?? dummy, queries, normWeight ?? dummy,
            MLXArray([1 / Float(d).squareRoot(), eps, Float(5).squareRoot()]),
        ],
        template: [
            ("T", DType.bfloat16), ("TO", norm ? DType.bfloat16 : DType.float32),
            ("K", blocks.dim(0)), ("D", d), ("Q", q), ("TG", threads),
            ("NW", normWeight?.dim(0) ?? 1), ("HAS_BASE", base != nil),
            ("HAS_CONV", convolution != nil), ("NORM_OUT", norm), ("F32_COPY", f32Copy),
        ],
        grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [
            [rows, d], [rows, d], [rows, q, d], f32Copy ? [rows, q, d] : [1],
            convolution != nil ? [rows, 3, d] : [1],
        ],
        outputDTypes: [.float32, .float32, norm ? .bfloat16 : .float32, .float32, .bfloat16])
    return Inkling2DepthMix(
        residual: out[0], normed: out[1], readout: out[2], readoutF32: f32Copy ? out[3] : nil,
        state: convolution != nil ? out[4] : nil)
}

// MARK: - Attention prologue

/// One simdgroup per head of the queries, the relative states, the keys and the values.
/// Keys and values first go through their short convolution, static or with dynamic taps.
private let attentionPrologueSource = """
        constexpr int E = DH / 32;
        constexpr int KV = HKV * DH;
        const int unit = threadgroup_position_in_grid.x;
        const int row = threadgroup_position_in_grid.y;
        const uint lane = thread_position_in_threadgroup.x;
        const float tau = FULL ? tau_in[0] : 1.0f;
        const float qk_threshold = params[0];
        const float rel_threshold = params[1];
        const float divisor = params[2];

        if (unit < HQ) {
            const int h = unit;
            float xv[E];
            float ss = 0.0f;
            for (int e = 0; e < E; e++) {
                const int c = h * DH + lane * E + e;
                xv[e] = float(q[row * q_strides[0] + c]) * float(q_scale[c]);
                ss += xv[e] * xv[e];
            }
            const float f = min(qk_threshold / (sqrt(simd_sum(ss)) + 1e-6f), 1.0f);
            for (int e = 0; e < E; e++) {
                T y = T(xv[e] * f);
                if (FULL) {
                    y = T(float(y) * tau);
                }
                q_out[(size_t(row) * HQ + h) * DH + lane * E + e] = y;
            }
            return;
        }
        if (unit < 2 * HQ) {
            const int h = unit - HQ;
            float xv = 0.0f;
            if (lane < DR) {
                const int c = h * DR + lane;
                xv = float(r[row * r_strides[0] + c]) * float(r_scale[c]);
            }
            const float f = min(rel_threshold / (sqrt(simd_sum(xv * xv)) + 1e-6f), 1.0f);
            if (lane < DR) {
                T y = T(xv * f);
                if (FULL) {
                    y = T(float(y) * tau);
                }
                r_out[(size_t(row) * HQ + h) * DR + lane] = y;
            }
            return;
        }

        const bool is_key = unit < 2 * HQ + HKV;
        const int h = unit - 2 * HQ - (is_key ? 0 : HKV);
        const device T* src = is_key ? k : v;
        const size_t src_row = row * (is_key ? k_strides[0] : v_strides[0]);
        const device T* state = is_key ? k_state : v_state;
        device T* state_next = is_key ? k_state_out : v_state_out;
        float yv[E];
        float ss = 0.0f;
        for (int e = 0; e < E; e++) {
            const int c = h * DH + lane * E + e;
            const size_t sb = size_t(row) * 3 * KV + c;
            const float s0 = float(state[sb]);
            const float s1 = float(state[sb + KV]);
            const float s2 = float(state[sb + 2 * KV]);
            const float xv = float(src[src_row + c]);
            float y;
            if (FULL) {
                auto sw = is_key ? k_weight : v_weight;
                auto taps = is_key ? k_taps : v_taps;
                const size_t tap_row = row * (is_key ? k_taps_strides[0] : v_taps_strides[0])
                    + h * 4;
                float a = xv * (float(sw[c]) + float(taps[tap_row]));
                a += s2 * (float(sw[KV + c]) + float(taps[tap_row + 1]));
                a += s1 * (float(sw[2 * KV + c]) + float(taps[tap_row + 2]));
                a += s0 * (float(sw[3 * KV + c]) + float(taps[tap_row + 3]));
                y = float(T((a + xv) / divisor));
            } else {
                auto w = (is_key ? k_weight : v_weight) + size_t(c) * 4;
                const float a = s0 * float(w[0]) + s1 * float(w[1]) + s2 * float(w[2])
                    + xv * float(w[3]);
                y = float(T((float(T(a)) + xv) / divisor));
            }
            state_next[sb] = state[sb + KV];
            state_next[sb + KV] = state[sb + 2 * KV];
            state_next[sb + 2 * KV] = src[src_row + c];
            yv[e] = y;
            ss += y * y;
        }
        const float total = simd_sum(ss);
        device T* out = is_key ? k_out : v_out;
        const float f = is_key
            ? min(qk_threshold / (sqrt(total) + 1e-6f), 1.0f)
            : metal::precise::rsqrt(total / DH + 1e-6f);
        for (int e = 0; e < E; e++) {
            out[(size_t(row) * HKV + h) * DH + lane * E + e] = T(yv[e] * f);
        }
    """

/// The decode inputs of the attention from its projections.
///
/// - Parameters:
///   - projections: `q`, `r`, `k` and `v` as `[B, 1, n]` with a contiguous last axis.
///   - taps: the dynamic convolution taps `[B, 1, Hkv * 4]` of full layers.
///   - weights: the key and value convolution weights, `[C, 4, 1]` or, with taps, `[4, C]`.
///   - tau: float32 `[1]`, the log-length scaling of full layers.
/// - Returns: queries `[B, Hq, DH]`, relative states `[B, Hq, DR]`, keys and values
///   `[B, Hkv, 1, DH]` and the next key and value convolution states.
func inkling2AttentionPrologue(
    q: MLXArray, r: MLXArray, k: MLXArray, v: MLXArray,
    taps: (k: MLXArray, v: MLXArray)?, weights: (k: MLXArray, v: MLXArray),
    states: (k: MLXArray, v: MLXArray), qScale: MLXArray, rScale: MLXArray,
    tau: MLXArray?, heads: Int, kvHeads: Int, headDim: Int, relativeDim: Int,
    qkThreshold: Float, relativeThreshold: Float
) -> [MLXArray] {
    let rows = q.dim(0)
    let kv = kvHeads * headDim
    let full = taps != nil
    let dummy = MLXArray.zeros([1], dtype: q.dtype)
    let divisor = Float(full ? 9 : 5).squareRoot()
    return Inkling2DecodeKernels.shared.attentionPrologue(
        [
            q, r, k, v, taps?.k ?? dummy, taps?.v ?? dummy, weights.k, weights.v, states.k,
            states.v, qScale, rScale, tau ?? MLXArray([Float(1)]),
            MLXArray([qkThreshold, relativeThreshold, divisor]),
        ],
        template: [
            ("T", q.dtype), ("DH", headDim), ("HQ", heads), ("HKV", kvHeads),
            ("DR", relativeDim), ("FULL", full),
        ],
        grid: ((2 * heads + 2 * kvHeads) * 32, rows, 1), threadGroup: (32, 1, 1),
        outputShapes: [
            [rows, heads, headDim], [rows, heads, relativeDim], [rows, kvHeads, 1, headDim],
            [rows, kvHeads, 1, headDim], [rows, 3, kv], [rows, 3, kv],
        ],
        outputDTypes: Array(repeating: q.dtype, count: 6))
}

// MARK: - Message controllers

/// The rank-`R` projection `h` of the RMS-normalized `[input, message]` pair of each stream.
/// Each threadgroup normalizes one pair and computes `TG / 32` of its outputs.
private let messageDownSource = """
        constexpr int NR = TG / 32;
        const uint unit = threadgroup_position_in_grid.x;
        const uint chunk = threadgroup_position_in_grid.y;
        const uint rows = threadgroups_per_grid.x / S;
        const uint s = unit / rows;
        const uint row = unit % rows;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const device float* in = input + (size_t(SHARED ? 0 : s) * rows + row) * D;
        const device float* msg = messages + (size_t(row) * MSTRIDE + first + s) * D;

        float a = 0.0f;
        float b = 0.0f;
        for (int d = tid; d < D; d += TG) {
            a += in[d] * in[d];
            b += msg[d] * msg[d];
        }
        threadgroup float red[2][NR];
        a = simd_sum(a);
        b = simd_sum(b);
        if (lane == 0) {
            red[0][sg] = a;
            red[1][sg] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a = 0.0f;
        b = 0.0f;
        for (int i = 0; i < NR; i++) {
            a += red[0][i];
            b += red[1][i];
        }
        const float ia = metal::precise::rsqrt(a / D + 1e-6f);
        const float ib = metal::precise::rsqrt(b / D + 1e-6f);
        threadgroup T pair[2 * D];
        for (int d = tid; d < D; d += TG) {
            pair[d] = T(in[d] * ia);
            pair[D + d] = T(msg[d] * ib);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int o = chunk * NR + sg;
        if (o >= R) {
            return;
        }
        const device T* w = down + (size_t(s) * R + o) * 2 * D;
        float acc = 0.0f;
        for (int k = lane * 4; k < 2 * D; k += 128) {
            for (int e = 0; e < 4; e++) {
                acc += float(pair[k + e]) * float(w[k + e]);
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            h[(size_t(s) * rows + row) * R + o] = float(T(acc));
        }
    """

/// `(1 - tanh(forget(h))) * input + tanh(gain(h)) * message` for `TG` channels of one stream,
/// and the sum of their squares.
private let messageBlendSource = """
        const uint unit = threadgroup_position_in_grid.x;
        const uint chunk = threadgroup_position_in_grid.y;
        const uint chunks = threadgroups_per_grid.y;
        const uint rows = threadgroups_per_grid.x / S;
        const uint s = unit / rows;
        const uint row = unit % rows;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const size_t unit_row = size_t(s) * rows + row;
        const device float* in = input + (size_t(SHARED ? 0 : s) * rows + row) * D;
        const device float* msg = messages + (size_t(row) * MSTRIDE + first + s) * D;

        threadgroup float hv[R];
        for (int r = tid; r < R; r += TG) {
            hv[r] = h[unit_row * R + r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int d = chunk * TG + tid;
        float o = 0.0f;
        if (d < D) {
            const device T* wf = forget + size_t(s) * R * D + d;
            const device T* wg = gain + size_t(s) * R * D + d;
            float f = 0.0f;
            float g = 0.0f;
            for (int r = 0; r < R; r++) {
                f += hv[r] * float(wf[r * D]);
                g += hv[r] * float(wg[r * D]);
            }
            o = (1.0f - metal::precise::tanh(f)) * in[d] + metal::precise::tanh(g) * msg[d];
            blend[unit_row * D + d] = o;
        }
        threadgroup float red[TG / 32];
        const float ss = simd_sum(o * o);
        if (lane == 0) {
            red[sg] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float t = 0.0f;
            for (int i = 0; i < TG / 32; i++) {
                t += red[i];
            }
            sums[unit_row * chunks + chunk] = t;
        }
    """

/// The RMS norm of each blended stream with the weight of the stream.
private let messageNormSource = """
        const uint unit = threadgroup_position_in_grid.x;
        const uint rows = threadgroups_per_grid.x / S;
        const uint s = unit / rows;
        const uint tid = thread_position_in_threadgroup.x;
        float t = 0.0f;
        for (int c = 0; c < C; c++) {
            t += sums[size_t(unit) * C + c];
        }
        const float inv = metal::precise::rsqrt(t / D + eps[0]);
        for (int d = tid; d < D; d += TG) {
            const size_t i = size_t(unit) * D + d;
            const T y = T(float(norm_weight[(NW == 1 ? 0 : s) * D + d]) * (blend[i] * inv));
            out[i] = y;
            if (F32_COPY) {
                out_f32[i] = float(y);
            }
        }
    """

/// The message controllers of one sublayer side, as normalized sublayer inputs.
///
/// - Parameters:
///   - inputs: float32 `[S or 1, B, D]`; one input is shared by every stream.
///   - messages: float32 `[B, M, D]`, the messages of all loop layers.
///   - first: the message of stream 0.
///   - down: bfloat16 `[S, R, 2D]`.
///   - forget: bfloat16 `[S, R, D]`.
///   - gain: bfloat16 `[S, R, D]`.
///   - normWeight: bfloat16 `[S or 1, D]`.
/// - Returns: bfloat16 `[S, B, D]` and its float32 copy when requested.
func inkling2MessageControllers(
    inputs: MLXArray, messages: MLXArray, first: Int, down: MLXArray, forget: MLXArray,
    gain: MLXArray, normWeight: MLXArray, eps: Float, f32Copy: Bool
) -> (MLXArray, MLXArray?) {
    let (streams, rank) = (down.dim(0), down.dim(1))
    let (rows, d) = (messages.dim(0), messages.dim(2))
    // Small threadgroups spread the weight reads over every GPU core.
    let threads = 256
    let chunks = (d + threads - 1) / threads
    let template: [(String, any KernelTemplateArg)] = [
        ("T", down.dtype), ("D", d), ("S", streams), ("R", rank), ("TG", threads),
        ("SHARED", inputs.dim(0) == 1 && streams > 1), ("MSTRIDE", messages.dim(1)),
    ]
    let kernels = Inkling2DecodeKernels.shared
    let firstArray = MLXArray(Int32(first))
    let h = kernels.messageDown(
        [inputs, messages, firstArray, down], template: template,
        grid: (streams * rows * threads, (rank + threads / 32 - 1) / (threads / 32), 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[streams, rows, rank]], outputDTypes: [.float32])[0]
    let blend = kernels.messageBlend(
        [inputs, messages, firstArray, h, forget, gain], template: template,
        grid: (streams * rows * threads, chunks, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[streams, rows, d], [streams, rows, chunks]],
        outputDTypes: [.float32, .float32])
    let out = kernels.messageNorm(
        [blend[0], blend[1], normWeight, MLXArray([eps])],
        template: [
            ("T", down.dtype), ("D", d), ("S", streams), ("C", chunks), ("TG", threads),
            ("NW", normWeight.dim(0)), ("F32_COPY", f32Copy),
        ],
        grid: (streams * rows * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[streams, rows, d], f32Copy ? [streams, rows, d] : [1]],
        outputDTypes: [down.dtype, .float32])
    return (out[0], f32Copy ? out[1] : nil)
}

// MARK: - Expert epilogue

/// The short convolution of the expert sum `m` of `row` at channel `d`, and the residual add.
private let expertEpilogueTail = """
        const T mb = T(m);
        const size_t sb = size_t(row) * 3 * D + d;
        const float s0 = float(state[sb]);
        const float s1 = float(state[sb + D]);
        const float s2 = float(state[sb + 2 * D]);
        const device T* cw = conv_weight + size_t(d) * 4;
        const float a = s0 * float(cw[0]) + s1 * float(cw[1]) + s2 * float(cw[2])
            + float(mb) * float(cw[3]);
        const float y = float(T((float(T(a)) + float(mb)) / divisor[0]));
        const size_t i = size_t(row) * D + d;
        hidden[i] = T(residual[i] + y);
        state_out[sb] = state[sb + D];
        state_out[sb + D] = state[sb + 2 * D];
        state_out[sb + 2 * D] = mb;
    """

/// The weighted sum of the experts, its short convolution and the residual add.
private let expertEpilogueSource =
    """
        const uint row = thread_position_in_grid.x / D;
        const uint d = thread_position_in_grid.x % D;
        float m = 0.0f;
        for (int j = 0; j < KE; j++) {
            m += float(outputs[(size_t(row) * KE + j) * D + d]) * weights[row * KE + j];
        }
    """ + expertEpilogueTail

/// The layer output `residual + conv(sum(weights * outputs))` and the next convolution state.
///
/// - Parameters:
///   - outputs: bfloat16 `[B, K, D]`, the outputs of the routed experts.
///   - weights: float32 `[B, K]`.
///   - residual: float32 `[B, D]`.
func inkling2ExpertEpilogue(
    outputs: MLXArray, weights: MLXArray, residual: MLXArray, state: MLXArray,
    convolutionWeight: MLXArray
) -> (hidden: MLXArray, state: MLXArray) {
    let (rows, k, d) = (outputs.dim(0), outputs.dim(1), outputs.dim(2))
    let out = Inkling2DecodeKernels.shared.expertEpilogue(
        [outputs, weights, residual, state, convolutionWeight, MLXArray([Float(5).squareRoot()])],
        template: [("T", outputs.dtype), ("D", d), ("KE", k)],
        grid: (rows * d, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[rows, d], [rows, 3, d]], outputDTypes: [outputs.dtype, outputs.dtype])
    return (out[0], out[1])
}

// MARK: - Routed experts

/// Dot products of quantized rows with a vector, 16 weights per lane and step. The fp modes
/// decode e2m1 and e4m3 through `half`, as MLX does.
private let expertHeader = """
    template <int GS, bool AFFINE, typename S>
    inline float inkling2_group_scale(S s) {
        if (AFFINE) {
            return float(s);
        }
        const uint8_t b = uint8_t(s);
        if (GS == 16) {
            const ushort v = b & 127;
            const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort(b & 128) << 8);
            return float(as_type<half>(u)) * 256.0f;
        }
        return as_type<float>(b == 0 ? 0x400000u : (uint(b) << 23));
    }

    // The dot product of 16 weights with x, before the group scale.
    template <int BITS, bool AFFINE>
    inline float inkling2_unit_dot(const device uint8_t* w, thread const float* x) {
        float s = 0.0f;
        if (BITS == 4) {
            const uint2 q = *(const device uint2*)w;
            for (int i = 0; i < 8; i++) {
                const uint n = (i < 4 ? q.x : q.y) >> ((i % 4) * 8);
                if (AFFINE) {
                    s += x[2 * i] * float(n & 15) + x[2 * i + 1] * float((n >> 4) & 15);
                } else {
                    const half lo = as_type<half>(ushort(((n & 8) << 12) | ((n & 7) << 9)));
                    const half hi = as_type<half>(ushort(((n & 128) << 8) | ((n & 112) << 5)));
                    s += x[2 * i] * float(lo) + x[2 * i + 1] * float(hi);
                }
            }
            return AFFINE ? s : s * 16384.0f;
        }
        const uint4 q = *(const device uint4*)w;
        for (int i = 0; i < 16; i++) {
            const uint b = (q[i / 4] >> ((i % 4) * 8)) & 255;
            if (AFFINE) {
                s += x[i] * float(b);
            } else {
                const ushort v = b & 127;
                const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort(b & 128) << 8);
                s += x[i] * float(as_type<half>(u));
            }
        }
        return AFFINE ? s : s * 256.0f;
    }

    // Each lane's share of the dot products of R consecutive rows with x.
    template <int BITS, int GS, bool AFFINE, int K, int R, typename T, typename S, typename B>
    inline void inkling2_rows_dot(
        const device uint8_t* w, const device S* scales, B biases, const device T* x, uint lane,
        thread float* acc) {
        for (int u = lane; u < K / 16; u += 32) {
            float xv[16];
            float xs = 0.0f;
            for (int i = 0; i < 16; i++) {
                xv[i] = float(x[u * 16 + i]);
                xs += xv[i];
            }
            const int g = u * 16 / GS;
            for (int r = 0; r < R; r++) {
                const float dot = inkling2_unit_dot<BITS, AFFINE>(w + (r * K + u * 16) * BITS / 8, xv);
                acc[r] += inkling2_group_scale<GS, AFFINE>(scales[r * (K / GS) + g]) * dot;
                if (AFFINE) {
                    acc[r] += float(biases[r * (K / GS) + g]) * xs;
                }
            }
        }
    }
    """

/// One simdgroup per channel of one routed expert: its gate and up rows, then the softcapped
/// SwiGLU. Each expert output rounds to `T` before the activation.
private let expertGateUpSource = """
        const uint lane = thread_index_in_simdgroup;
        const int bj = threadgroup_position_in_grid.y;
        const int c = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
        const uint e = indices[bj];
        const size_t row = size_t(e) * 2 * I + 2 * c;
        float acc[2] = {0.0f, 0.0f};
        inkling2_rows_dot<BITS, GS, AFFINE, DIN, 2>(
            (const device uint8_t*)w + row * (DIN * BITS / 8), scales + row * (DIN / GS),
            biases + (AFFINE ? row * (DIN / GS) : 0), x + size_t(bj / KE) * DIN, lane, acc);
        const float g = simd_sum(acc[0]);
        const float u = simd_sum(acc[1]);
        if (lane == 0) {
            const float s = HAS_GS ? global_scale[e] / (448.0f * 6.0f) : 1.0f;
            const float gate = float(T(g * s));
            const float up = float(T(u * s));
            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gate)));
            const float sig = gate < 0 ? y : 1.0f - y;
            const float gc = metal::precise::tanh(gate / caps[0]) * caps[0];
            const float uc = metal::precise::tanh(up / caps[1]) * caps[1];
            act[size_t(bj) * I + c] = T(gc * sig * uc);
        }
    """

/// One simdgroup per routed expert over `ROWS` output channels, then the weighted sum of the
/// experts, its short convolution and the residual add.
private let expertDownSource =
    """
        const uint lane = thread_index_in_simdgroup;
        const uint j = simdgroup_index_in_threadgroup;
        const int row = threadgroup_position_in_grid.y;
        const int d0 = threadgroup_position_in_grid.x * ROWS;
        const uint e = indices[row * KE + j];
        const float s = HAS_GS ? global_scale[e] / (448.0f * 6.0f) : 1.0f;
        threadgroup float outputs[KE * ROWS];
        for (int r = 0; r < ROWS; r++) {
            const size_t n = size_t(e) * D + d0 + r;
            float acc[1] = {0.0f};
            inkling2_rows_dot<BITS, GS, AFFINE, K, 1>(
                (const device uint8_t*)w + n * (K * BITS / 8), scales + n * (K / GS),
                biases + (AFFINE ? n * (K / GS) : 0), act + size_t(row * KE + j) * K, lane, acc);
            const float o = simd_sum(acc[0]);
            if (lane == 0) {
                outputs[j * ROWS + r] = float(T(o * s));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = thread_position_in_threadgroup.x;
        if (t >= ROWS) {
            return;
        }
        const int d = d0 + t;
        float m = 0.0f;
        for (int k = 0; k < KE; k++) {
            m += outputs[k * ROWS + t] * weights[row * KE + k];
        }
    """ + expertEpilogueTail

/// A quantized projection of all experts, `[E, N, K * bits / 32]`.
struct Inkling2ExpertTensors {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let globalScale: MLXArray?
    let format: Inkling2ExpertFormat

    /// Whether the fused expert kernels read this format with `inputDims` per row.
    func supportsFusedDecode(inputDims: Int) -> Bool {
        let modes: [QuantizationMode] = [.affine, .nvfp4, .mxfp4, .mxfp8]
        return modes.contains(format.mode) && [4, 8].contains(format.bits)
            && format.groupSize % 16 == 0 && inputDims % 16 == 0
            && (format.mode == .affine) == (biases != nil)
    }

    fileprivate func inputs() -> [MLXArray] {
        [weight, scales, biases ?? MLXArray([Float(0)]), globalScale ?? MLXArray([Float(1)])]
    }

    fileprivate func template(_ dtype: DType) -> [(String, any KernelTemplateArg)] {
        [
            ("T", dtype), ("BITS", format.bits), ("GS", format.groupSize),
            ("AFFINE", format.mode == .affine), ("HAS_GS", globalScale != nil),
        ]
    }
}

/// The softcapped SwiGLU of the gate and up rows of each routed expert, which alternate.
///
/// - Parameters:
///   - x: `[B, D]`.
///   - indices: uint32 `[B, K]`.
///   - softcaps: float32 `[2]`, the gate and up softcaps.
/// - Returns: `[B, K, I]`.
func inkling2ExpertGateUp(
    _ x: MLXArray, experts: Inkling2ExpertTensors, indices: MLXArray, softcaps: MLXArray
) -> MLXArray {
    let (rows, k) = (indices.dim(0), indices.dim(1))
    let (din, channels) = (x.dim(-1), experts.weight.dim(1) / 2)
    let simdgroups = 4
    return Inkling2DecodeKernels.shared.expertGateUp(
        [x] + experts.inputs() + [indices, softcaps],
        template: experts.template(x.dtype) + [
            ("DIN", din), ("I", channels), ("KE", k), ("NSG", simdgroups),
        ],
        grid: (channels * 32, rows * k, 1), threadGroup: (simdgroups * 32, 1, 1),
        outputShapes: [[rows, k, channels]], outputDTypes: [x.dtype])[0]
}

/// ``inkling2ExpertEpilogue(outputs:weights:residual:state:convolutionWeight:)`` of the down
/// projections of `activations`, without materializing the expert outputs.
///
/// - Parameters:
///   - activations: `[B, K, I]`.
///   - weights: float32 `[B, K]`.
///   - residual: float32 `[B, D]`.
func inkling2ExpertDown(
    _ activations: MLXArray, experts: Inkling2ExpertTensors, indices: MLXArray,
    weights: MLXArray, residual: MLXArray, state: MLXArray, convolutionWeight: MLXArray
) -> (hidden: MLXArray, state: MLXArray) {
    let (rows, k, inner) = (activations.dim(0), activations.dim(1), activations.dim(2))
    let d = experts.weight.dim(1)
    let tile = 2
    let out = Inkling2DecodeKernels.shared.expertDown(
        [activations] + experts.inputs() + [
            indices, weights, residual, state, convolutionWeight,
            MLXArray([Float(5).squareRoot()]),
        ],
        template: experts.template(activations.dtype) + [
            ("K", inner), ("D", d), ("KE", k), ("ROWS", tile),
        ],
        grid: (d / tile * k * 32, rows, 1), threadGroup: (k * 32, 1, 1),
        outputShapes: [[rows, d], [rows, 3, d]],
        outputDTypes: [activations.dtype, activations.dtype])
    return (out[0], out[1])
}

private final class Inkling2DecodeKernels: @unchecked Sendable {
    static let shared = Inkling2DecodeKernels()

    let single: MLXFast.MLXFastKernel
    let split: MLXFast.MLXFastKernel
    let reduce: MLXFast.MLXFastKernel
    let attentionPrologue: MLXFast.MLXFastKernel
    let expertEpilogue: MLXFast.MLXFastKernel
    let expertGateUp: MLXFast.MLXFastKernel
    let expertDown: MLXFast.MLXFastKernel
    let depthMix: MLXFast.MLXFastKernel
    let messageDown: MLXFast.MLXFastKernel
    let messageBlend: MLXFast.MLXFastKernel
    let messageNorm: MLXFast.MLXFastKernel

    private init() {
        let inputs = ["q", "k", "v", "rel", "table", "sinks", "gate", "params", "scale"]
        single = MLXFast.metalKernel(
            name: "inkling2_decode_attention", inputNames: inputs, outputNames: ["out"],
            source: relativeDecodeSource, header: keyDistanceHeader, ensureRowContiguous: false)
        split = MLXFast.metalKernel(
            name: "inkling2_decode_attention_split", inputNames: inputs,
            outputNames: ["partial", "stats"], source: groupedDecodeSplitSource,
            header: keyDistanceHeader, ensureRowContiguous: false)
        reduce = MLXFast.metalKernel(
            name: "inkling2_decode_attention_reduce",
            inputNames: ["partial", "stats", "sinks", "gate"], outputNames: ["out"],
            source: relativeDecodeReduceSource, ensureRowContiguous: false)
        attentionPrologue = MLXFast.metalKernel(
            name: "inkling2_attention_prologue",
            inputNames: [
                "q", "r", "k", "v", "k_taps", "v_taps", "k_weight", "v_weight", "k_state",
                "v_state", "q_scale", "r_scale", "tau_in", "params",
            ],
            outputNames: ["q_out", "r_out", "k_out", "v_out", "k_state_out", "v_state_out"],
            source: attentionPrologueSource, ensureRowContiguous: false)
        expertEpilogue = MLXFast.metalKernel(
            name: "inkling2_expert_epilogue",
            inputNames: ["outputs", "weights", "residual", "state", "conv_weight", "divisor"],
            outputNames: ["hidden", "state_out"], source: expertEpilogueSource)
        let experts = ["w", "scales", "biases", "global_scale", "indices"]
        expertGateUp = MLXFast.metalKernel(
            name: "inkling2_expert_gate_up", inputNames: ["x"] + experts + ["caps"],
            outputNames: ["act"], source: expertGateUpSource, header: expertHeader)
        expertDown = MLXFast.metalKernel(
            name: "inkling2_expert_down",
            inputNames: ["act"] + experts + [
                "weights", "residual", "state", "conv_weight", "divisor",
            ],
            outputNames: ["hidden", "state_out"], source: expertDownSource, header: expertHeader)
        depthMix = MLXFast.metalKernel(
            name: "inkling2_depth_mix",
            inputNames: [
                "blocks", "base", "x", "state", "conv_weight", "queries", "norm_weight", "params",
            ],
            outputNames: ["residual_out", "normed_out", "readout", "readout_f32", "state_out"],
            source: depthMixSource)
        messageDown = MLXFast.metalKernel(
            name: "inkling2_message_down", inputNames: ["input", "messages", "first", "down"],
            outputNames: ["h"], source: messageDownSource)
        messageBlend = MLXFast.metalKernel(
            name: "inkling2_message_blend",
            inputNames: ["input", "messages", "first", "h", "forget", "gain"],
            outputNames: ["blend", "sums"], source: messageBlendSource)
        messageNorm = MLXFast.metalKernel(
            name: "inkling2_message_norm", inputNames: ["blend", "sums", "norm_weight", "eps"],
            outputNames: ["out", "out_f32"], source: messageNormSource)
    }
}

// MARK: - Expert routing

/// Sigmoid scores, top-k on the bias-corrected scores, and normalized weights
/// of the selected scores, one simdgroup per token.
private let expertRoutingSource = """
        constexpr int R = (E + 31) / 32;
        uint token = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        auto row = logits + size_t(token) * E;

        float score[R];
        float biased[R];
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            score[i] = 0.0f;
            biased[i] = -INFINITY;
            if (e < E) {
                float x = row[e];
                float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                score[i] = x < 0 ? y : 1.0f - y;
                biased[i] = score[i] + bias[e];
            }
        }

        float w[K];
        for (int k = 0; k < K; k++) {
            float best = -INFINITY;
            int best_e = E;
            for (int i = 0; i < R; i++) {
                int e = lane + 32 * i;
                if (biased[i] > best || (biased[i] == best && e < best_e)) {
                    best = biased[i];
                    best_e = e;
                }
            }
            float top = simd_max(best);
            int winner = simd_min(best == top ? best_e : E);
            float s = 0.0f;
            for (int i = 0; i < R; i++) {
                if (lane + 32 * i == winner) {
                    s = score[i];
                    biased[i] = -INFINITY;
                }
            }
            w[k] = simd_sum(s);
            if (lane == 0) {
                inds[size_t(token) * K + k] = winner;
            }
        }

        float total = 0.0f;
        for (int k = 0; k < K; k++) {
            total = w[k] + total;
        }
        if (lane == 0) {
            for (int k = 0; k < K; k++) {
                weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
            }
        }
    """

/// The experts and their weights from float32 router logits `[..., E]`.
///
/// - Returns: uint32 indices and float32 weights, both `[..., topK]`.
func inkling2ExpertRouting(
    logits: MLXArray, bias: MLXArray, topK: Int, scoreEps: Float, routeScale: Float
) -> (indices: MLXArray, weights: MLXArray) {
    let e = logits.dim(-1)
    let tokens = logits.size / e
    let shape = Array(logits.shape.dropLast()) + [topK]
    let out = Inkling2Kernels.shared.expertRouting(
        [logits, bias, MLXArray(scoreEps), MLXArray(routeScale)],
        template: [("E", e), ("K", topK)],
        grid: (tokens * 32, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, .float32])
    return (out[0], out[1])
}
