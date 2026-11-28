// Copyright © 2026 Apple Inc.
//
// Decode kernels of Raon-1 mini. Adapted from the kernels KRAFTON ships with the MLX checkpoint
// (`raon_mlx_model.py`, Apache 2.0).

import Foundation
import MLX

/// One mHC site in two kernels instead of about twenty operations.
///
/// The first kernel mixes the pending branch into the rows and accumulates their selector
/// products and squared sum per block. The second reduces the blocks, computes the stream
/// coefficients, aggregates the branch input and applies the site's RMS norm.
enum Raon1MHCKernels {
    private static let threads = 512
    private static let block = 1024

    /// Whether the kernels support `rows` `[N, n, H]`: bfloat16, and a hidden size the norm's
    /// threadgroup covers four values per thread.
    static func supports(_ rows: MLXArray) -> Bool {
        rows.dtype == .bfloat16 && rows.dim(2) % 4 == 0 && rows.dim(2) <= 4 * 1024
    }

    /// - Parameters:
    ///   - rows: the streams `[N, n, H]`, bfloat16.
    ///   - pending: the branch `[N, H]` with its float32 `post` `[N, n]` and `residual`
    ///     `[N, n, n]`, or `nil` before the first branch.
    ///   - selector: `phi_pre`, `phi_post` and `phi_z` stacked, `[2n + (n - 1)², n H]`.
    ///   - parameters: the site's packed scalars, see ``Raon1MHC``.
    ///   - normWeight: the RMS norm weight `[H]` and its `eps`.
    /// - Returns: the next rows, the normed branch input `[N, H]`, `post` and `residual`.
    static func enter(
        rows: MLXArray, pending: (branch: MLXArray, post: MLXArray, residual: MLXArray)?,
        selector: MLXArray, parameters: MLXArray, normWeight: MLXArray, eps: MLXArray
    ) -> (rows: MLXArray, x: MLXArray, post: MLXArray, residual: MLXArray) {
        let (n, streams, hidden) = (rows.dim(0), rows.dim(1), rows.dim(2))
        let blocks = (streams * hidden + block - 1) / block
        let products = 2 * streams + (streams - 1) * (streams - 1)
        let mixes = pending != nil
        let pending =
            pending
            ?? (
                MLXArray.zeros([n, hidden], dtype: .bfloat16),
                MLXArray.zeros([n, streams], dtype: .float32),
                MLXArray.zeros([n, streams, streams], dtype: .float32)
            )
        let first = mixKernel(
            [rows, pending.branch, pending.post, pending.residual, selector],
            template: [
                ("S", streams), ("H_DIM", hidden), ("DO_MIX", mixes),
                ("BLK", block), ("NBLK", blocks),
            ],
            grid: (threads * blocks, n, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[n, streams, hidden], [n, blocks, products + 1]],
            outputDTypes: [.bfloat16, .float32])

        let normThreads = 32 * ((hidden + 127) / 128)
        let second = routeKernel(
            [first[1], first[0], parameters, normWeight, eps],
            template: [("S", streams), ("H_DIM", hidden), ("NBLK", blocks)],
            grid: (normThreads, n, 1),
            threadGroup: (normThreads, 1, 1),
            outputShapes: [[n, hidden], [n, streams], [n, streams, streams]],
            outputDTypes: [.bfloat16, .float32, .float32])
        return (first[0], second[0], second[1], second[2])
    }

    private static let mixKernel = MLXFast.metalKernel(
        name: "raon1_mhc_mix",
        inputNames: ["rows", "branch", "post_in", "res_in", "phi"],
        outputNames: ["new_rows", "partial"],
        source: """
                const uint row = threadgroup_position_in_grid.y;
                const uint blk = threadgroup_position_in_grid.x;
                const uint tid = thread_position_in_threadgroup.x;
                const uint T = threads_per_threadgroup.x;
                constexpr int H = H_DIM;
                constexpr int K = 2 * S + (S - 1) * (S - 1);
                constexpr int D = S * H;
                threadgroup float red[32][K + 1];
                float acc[K + 1];
                for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
                const device bfloat16_t* rin = rows + row * D;
                device bfloat16_t* rout = new_rows + row * D;
                const device bfloat16_t* br = branch + row * H;
                const int start = blk * BLK;
                const int stop = metal::min(start + BLK, D);
                for (int idx = start + tid; idx < stop; idx += T) {
                    const int s = idx / H;
                    const int h = idx - s * H;
                    float v;
                    if (DO_MIX) {
                        float m = 0.0f;
                        for (int j = 0; j < S; ++j)
                            m += res_in[(row * S + s) * S + j] * (float)rin[j * H + h];
                        m += post_in[row * S + s] * (float)br[h];
                        const bfloat16_t mb = (bfloat16_t)m;
                        rout[idx] = mb;
                        v = (float)mb;
                    } else {
                        v = (float)rin[idx];
                        rout[idx] = rin[idx];
                    }
                    acc[K] += v * v;
                    for (int k = 0; k < K; ++k) acc[k] += v * (float)phi[k * D + idx];
                }
                const uint lane = tid % 32;
                const uint sg = tid / 32;
                for (int k = 0; k <= K; ++k) {
                    const float t = simd_sum(acc[k]);
                    if (lane == 0) red[sg][k] = t;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid <= (uint)K) {
                    float t = 0.0f;
                    for (uint g = 0; g < (T + 31) / 32; ++g) t += red[g][tid];
                    partial[(row * NBLK + blk) * (K + 1) + tid] = t;
                }
            """)

    /// `prm`: eps, post factor, α_pre, b_pre (S), α_post, b_post (S), α_z, b_z ((S-1)²),
    /// γ = sigmoid(η), the Helmert basis (S × (S-1)) and the chart center (S × S).
    private static let routeKernel = MLXFast.metalKernel(
        name: "raon1_mhc_route_norm",
        inputNames: ["partial", "new_rows", "prm", "w", "eps"],
        outputNames: ["normed", "post_out", "res_out"],
        source: """
                const uint row = threadgroup_position_in_grid.y;
                const uint tid = thread_position_in_threadgroup.x;
                constexpr int H = H_DIM;
                constexpr int M = (S - 1) * (S - 1);
                constexpr int K = 2 * S + M;
                constexpr int D = S * H;
                constexpr int A_PRE = 2, B_PRE = 3, A_POST = 3 + S, B_POST = 4 + S;
                constexpr int A_Z = 4 + 2 * S, B_Z = 5 + 2 * S, GAMMA = 5 + 2 * S + M;
                constexpr int Q = GAMMA + 1, C = Q + S * (S - 1);
                threadgroup float tots[K + 1];
                threadgroup float zs[K];
                threadgroup float pre[S];
                threadgroup float cs[M];
                threadgroup float es[S * S];
                threadgroup float rhos[1];
                if (tid <= (uint)K) {
                    float t = 0.0f;
                    for (int b = 0; b < NBLK; ++b) t += partial[(row * NBLK + b) * (K + 1) + tid];
                    tots[tid] = t;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float inv_t = metal::rsqrt(tots[K] / (float)D + prm[0]);
                if (tid < (uint)K) zs[tid] = (float)(bfloat16_t)tots[tid] * inv_t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < (uint)S) {
                    pre[tid] = 1.0f / (1.0f + metal::exp(-(prm[A_PRE] * zs[tid] + prm[B_PRE + tid])));
                    post_out[row * S + tid] =
                        prm[1] / (1.0f + metal::exp(-(prm[A_POST] * zs[S + tid] + prm[B_POST + tid])));
                } else if (tid < (uint)(S + M)) {
                    const int k = tid - S;
                    cs[k] = prm[A_Z] * zs[2 * S + k] + prm[B_Z + k];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < (uint)(S * S)) {
                    const int i = tid / S, j = tid % S;
                    float t = 0.0f;
                    for (int a = 0; a < S - 1; ++a)
                        for (int b = 0; b < S - 1; ++b)
                            t += prm[Q + i * (S - 1) + a] * cs[a * (S - 1) + b] * prm[Q + j * (S - 1) + b];
                    es[tid] = t;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float rho = 0.0f;
                    for (int ij = 0; ij < S * S; ++ij) rho = metal::max(rho, -es[ij] / prm[C + ij]);
                    rhos[0] = rho;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < (uint)(S * S)) {
                    const int i = tid / S, j = tid % S;
                    const float g = prm[GAMMA];
                    res_out[row * S * S + tid] =
                        (1.0f - g) * (i == j ? 1.0f : 0.0f) + g * (prm[C + tid] + es[tid] / (1.0f + rhos[0]));
                }
                threadgroup float local_sums[32];
                threadgroup float local_inv[1];
                const uint lane = tid % 32;
                const uint sg = tid / 32;
                if (sg == 0) local_sums[lane] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float xs[4];
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const int h = tid * 4 + i;
                    float x = 0.0f;
                    if (h < H) {
                        float a = 0.0f;
                        for (int s = 0; s < S; ++s) a += (float)new_rows[row * D + s * H + h] * pre[s];
                        x = (float)(bfloat16_t)a;
                    }
                    xs[i] = x;
                    acc += x * x;
                }
                acc = simd_sum(acc);
                if (lane == 0) local_sums[sg] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    acc = simd_sum(local_sums[lane]);
                    if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / (float)H + eps[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = 0; i < 4; ++i) {
                    const int h = tid * 4 + i;
                    if (h < H) normed[row * H + h] = w[h] * static_cast<bfloat16_t>(xs[i] * local_inv[0]);
                }
            """)
}

/// One-token RDA in two kernels: the convolution, normalization and gates, then the recurrence
/// with the output norm and gate.
enum Raon1RDAKernels {
    /// Whether the kernels support a step of `x` `[B, L, H]` with heads of `headDim`.
    static func supports(_ x: MLXArray, headDim: Int) -> Bool {
        x.dim(1) == 1 && x.dtype == .bfloat16 && headDim == 128
    }

    /// The constants the kernels read, derived once from the layer's parameters.
    struct Constants {
        /// The convolution weight, float32 `[3P, K]`.
        let convolution: MLXArray
        /// `dt_bias`, float32 `[heads, headDim]`.
        let dtBias: MLXArray
        /// `exp(A_log)`, float32 `[heads]`.
        let decayScale: MLXArray
        /// `[1 / sqrt(headDim), gate lower bound]`, float32.
        let scalars: MLXArray
        /// The output norm weight, float32 `[headDim]`.
        let normWeight: MLXArray
        /// The output norm eps, float32 `[1]`.
        let eps: MLXArray
    }

    /// - Parameters:
    ///   - q, k, v: the projections `[B, 1, P]` before the convolution.
    ///   - state: the convolution state `[B, K - 1, 3P]`.
    ///   - b, w: the write gate projections `[B, 1, heads]`.
    ///   - decay: the decay projection `[B, 1, P]`.
    /// - Returns: `q`, `k`, `v` `[B, 1, heads, headDim]`, the float32 decay `g`, `beta`
    ///   `[B, 1, heads]` and the next convolution state.
    static func prepare(
        q: MLXArray, k: MLXArray, v: MLXArray, state: MLXArray, b: MLXArray, w: MLXArray,
        decay: MLXArray, constants: Constants, heads: Int, headDim: Int, kernelSize: Int
    ) -> [MLXArray] {
        let batch = q.dim(0)
        let shape = [batch, 1, heads, headDim]
        let out = prepareKernel(
            [
                q, k, v, state, constants.convolution, b, w, decay, constants.dtBias,
                constants.decayScale, constants.scalars,
            ],
            template: [("HEAD_DIM", headDim), ("N_HEADS", heads), ("KSIZE", kernelSize)],
            grid: (heads * headDim, batch, 1),
            threadGroup: (headDim, 1, 1),
            outputShapes: [shape, shape, shape, shape, [batch, 1, heads], state.shape],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .float32, .bfloat16, state.dtype])
        return out
    }

    /// - Returns: the gated, normed output `[B, 1, heads, headDim]` and the next state.
    static func recurrence(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
        gate: MLXArray, constants: Constants
    ) -> (MLXArray, MLXArray) {
        let (batch, heads, headDim) = (q.dim(0), q.dim(2), q.dim(3))
        let rows = 4
        let out = recurrenceKernel(
            [q, k, v, g, beta, state, constants.normWeight, gate, constants.eps],
            template: [
                ("InT", q.dtype), ("StT", state.dtype), ("Dk", headDim), ("Dv", headDim),
                ("Hk", heads), ("Hv", heads), ("ROWS", rows),
            ],
            grid: (32, headDim / rows, batch * heads),
            threadGroup: (32, headDim / rows, 1),
            outputShapes: [q.shape, state.shape],
            outputDTypes: [.bfloat16, state.dtype])
        return (out[0], out[1])
    }

    /// The causal depthwise convolution of a prompt's q, k and v projections `[B, L, P]`
    /// followed by SiLU, accumulated in float32.
    ///
    /// - Parameters:
    ///   - state: the last `K - 1` inputs before the prompt, `[B, K - 1, 3P]`.
    ///   - weight: the float32 convolution weight `[3P, K]`.
    /// - Returns: the activations of q, k and v, each `[B, L, P]`.
    static func convolution(
        q: MLXArray, k: MLXArray, v: MLXArray, state: MLXArray, weight: MLXArray
    ) -> [MLXArray] {
        let (batch, length, width) = (q.dim(0), q.dim(1), q.dim(2))
        return convolutionKernel(
            [q, k, v, state, weight],
            template: [("P", width), ("KS", weight.dim(1))],
            grid: (3 * width, length, batch),
            threadGroup: (256, 1, 1),
            outputShapes: [q.shape, q.shape, q.shape],
            outputDTypes: [q.dtype, q.dtype, q.dtype])
    }

    private static let convolutionKernel = MLXFast.metalKernel(
        name: "raon1_rda_short_convolution",
        inputNames: ["xq", "xk", "xv", "state", "w32"],
        outputNames: ["q_out", "k_out", "v_out"],
        source: """
                const uint c = thread_position_in_grid.x;
                const uint l = thread_position_in_grid.y;
                const uint b = thread_position_in_grid.z;
                const uint L = threads_per_grid.y;
                constexpr int KEEP = KS - 1;
                const uint part = c / P;
                const uint cc = c - part * P;
                const device bfloat16_t* x = part == 0 ? xq : (part == 1 ? xk : xv);
                device bfloat16_t* out = part == 0 ? q_out : (part == 1 ? k_out : v_out);
                float acc = 0.0f;
                for (int t = 0; t < KS; ++t) {
                    const int pos = (int)l + t - KEEP;
                    const float value = pos >= 0
                        ? (float)x[(b * L + pos) * P + cc]
                        : (float)state[(b * KEEP + KEEP + pos) * 3 * P + c];
                    acc = metal::fma(value, w32[c * KS + t], acc);
                }
                out[(b * L + l) * P + cc] = (bfloat16_t)(acc * raon_sigmoid(acc));
            """,
        header: header)

    /// Whether ``prompt(q:k:v:g:beta:state:)`` supports these inputs.
    static func supportsPrompt(q: MLXArray, v: MLXArray, g: MLXArray, state: MLXArray?) -> Bool {
        q.dim(1) > 1 && q.dim(3) == 128 && v.dim(3) % 8 == 0 && g.ndim == 4
            && g.dtype == .float32 && (state?.dtype ?? .float32) == .float32
    }

    /// The gated delta rule with one decay per key dimension over a prompt chunk, for 128-wide
    /// keys. Four lanes share a value row, so a SIMD group updates eight rows at once.
    static func prompt(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (batch, length, keyHeads, keyDim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (valueHeads, valueDim) = (v.dim(2), v.dim(3))
        let out = promptKernel(
            [q, k, v, g, beta, state, length],
            template: [
                ("InT", q.dtype), ("StT", state.dtype), ("Dk", keyDim), ("Dv", valueDim),
                ("Hk", keyHeads), ("Hv", valueHeads),
            ],
            grid: (32, valueDim / 8, batch * valueHeads),
            threadGroup: (32, 2, 1),
            outputShapes: [[batch, length, valueHeads, valueDim], state.shape],
            outputDTypes: [q.dtype, state.dtype])
        return (out[0], out[1])
    }

    private static let promptKernel = MLXFast.metalKernel(
        name: "raon1_gated_delta_prompt",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
                constexpr int lanes_per_row = 4;
                constexpr int rows_per_simdgroup = 32 / lanes_per_row;
                constexpr int values_per_lane = Dk / lanes_per_row;
                constexpr int partials_per_lane = values_per_lane / 4;
                auto n = thread_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);
                auto lane = thread_index_in_simdgroup;
                auto lane_in_row = lane & (lanes_per_row - 1);
                auto dv_idx = thread_position_in_grid.y * rows_per_simdgroup + lane / lanes_per_row;
                auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk + lane_in_row * values_per_lane;
                auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
                y += (b_idx * T * Hv + hv_idx) * Dv;
                auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
                auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
                float state[values_per_lane];
                for (int i = 0; i < values_per_lane; ++i) state[i] = static_cast<float>(i_state[i]);
                auto beta_ = beta + b_idx * T * Hv;
                for (int t = 0; t < T; ++t) {
                    float part[partials_per_lane];
                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] * static_cast<float>(g_[e]);
                            acc += state[e] * static_cast<float>(k_[e]);
                        }
                        part[pb] = acc;
                    }
                    float kv_mem = ((part[0] + part[1]) + (part[2] + part[3]))
                        + ((part[4] + part[5]) + (part[6] + part[7]));
                    kv_mem += simd_shuffle_xor(kv_mem, 1);
                    kv_mem += simd_shuffle_xor(kv_mem, 2);
                    auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                        * static_cast<float>(beta_[hv_idx]);
                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
                            acc += state[e] * static_cast<float>(q_[e]);
                        }
                        part[pb] = acc;
                    }
                    float out = ((part[0] + part[1]) + (part[2] + part[3]))
                        + ((part[4] + part[5]) + (part[6] + part[7]));
                    out += simd_shuffle_xor(out, 1);
                    out += simd_shuffle_xor(out, 2);
                    if (lane_in_row == 0) y[dv_idx] = static_cast<InT>(out);
                    q_ += Hk * Dk;
                    k_ += Hk * Dk;
                    g_ += Hv * Dk;
                    v_ += Hv * Dv;
                    y += Hv * Dv;
                    beta_ += Hv;
                }
                for (int i = 0; i < values_per_lane; ++i) o_state[i] = static_cast<StT>(state[i]);
            """)

    /// The activations of MLX, so the results match the operations.
    private static let header = """
        inline float raon_sigmoid(float z) {
          float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
          return (z < 0.0f) ? y : 1.0f - y;
        }
        inline float raon_log1p(float z) {
          float xp1 = 1.0f + z;
          if (xp1 == 1.0f) return z;
          return z * (metal::log(xp1) / (xp1 - 1.0f));
        }
        inline float raon_softplus(float z) {
          float hi = metal::max(z, 0.0f), lo = metal::min(z, 0.0f);
          return (metal::isinf(lo) || metal::isinf(hi)) ? hi : hi + raon_log1p(metal::exp(lo - hi));
        }
        """

    private static let prepareKernel = MLXFast.metalKernel(
        name: "raon1_rda_prepare",
        inputNames: [
            "xq", "xk", "xv", "state", "w32", "braw", "delta", "decay", "dt32", "a_exp", "fp",
        ],
        outputNames: ["q_out", "k_out", "v_out", "g_out", "beta_out", "new_state"],
        source: """
                const uint b = threadgroup_position_in_grid.y;
                const uint h = threadgroup_position_in_grid.x;
                const uint d = thread_position_in_threadgroup.x;
                constexpr int D = HEAD_DIM;
                constexpr int NH = N_HEADS;
                constexpr int P = NH * D;
                constexpr int KS = KSIZE;
                constexpr int KEEP = KS - 1;

                float outv[3];
                for (int part = 0; part < 3; ++part) {
                    const device bfloat16_t* x = part == 0 ? xq : (part == 1 ? xk : xv);
                    const int c = part * P + h * D + d;
                    float acc = 0.0f;
                    for (int t = 0; t < KEEP; ++t)
                        acc += (float)state[(b * KEEP + t) * 3 * P + c] * w32[c * KS + t];
                    const bfloat16_t xin = x[b * P + h * D + d];
                    acc += (float)xin * w32[c * KS + KEEP];
                    for (int t = 0; t < KEEP - 1; ++t)
                        new_state[(b * KEEP + t) * 3 * P + c] = state[(b * KEEP + t + 1) * 3 * P + c];
                    new_state[(b * KEEP + KEEP - 1) * 3 * P + c] = xin;
                    const float ys = 1.0f / (1.0f + metal::precise::exp(metal::abs(acc)));
                    outv[part] = (float)(bfloat16_t)(acc * ((acc < 0.0f) ? ys : 1.0f - ys));
                }
                // Sum in the order of the MLX row reduction, 32 lanes of D / 32 values.
                threadgroup float sq[2][HEAD_DIM];
                threadgroup float tot[2];
                sq[0][d] = outv[0] * outv[0];
                sq[1][d] = outv[1] * outv[1];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (d < 32) {
                    float pq = 0.0f, pk = 0.0f;
                    for (int i = 0; i < D / 32; ++i) {
                        pq = sq[0][(D / 32) * d + i] + pq;
                        pk = sq[1][(D / 32) * d + i] + pk;
                    }
                    pq = simd_sum(pq);
                    pk = simd_sum(pk);
                    if (d == 0) { tot[0] = pq; tot[1] = pk; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float qn = (float)(bfloat16_t)metal::precise::divide(
                    outv[0], metal::max(metal::precise::sqrt(tot[0]), 1e-6f));
                const float kn = (float)(bfloat16_t)metal::precise::divide(
                    outv[1], metal::max(metal::precise::sqrt(tot[1]), 1e-6f));
                const int o = (b * NH + h) * D + d;
                q_out[o] = (bfloat16_t)(qn * (float)(bfloat16_t)fp[0]);
                k_out[o] = (bfloat16_t)kn;
                const float br = (float)braw[b * NH + h];
                const float dl = (float)delta[b * NH + h];
                const float ratio = metal::precise::exp(-raon_softplus(-br - dl) + raon_softplus(-br));
                v_out[o] = (bfloat16_t)(outv[2] * ratio);
                const float dec = ((float)decay[b * P + h * D + d] + dt32[h * D + d]) * a_exp[h];
                g_out[o] = metal::precise::exp(fp[1] * raon_sigmoid(dec));
                if (d == 0) beta_out[b * NH + h] = (bfloat16_t)raon_sigmoid(br);
            """,
        header: header)

    private static let recurrenceKernel = MLXFast.metalKernel(
        name: "raon1_rda_recurrence_out",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "norm_w", "gate", "eps"],
        outputNames: ["gated", "state_out"],
        source: """
                auto n = threadgroup_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);
                constexpr int n_per_t = Dk / 32;
                threadgroup float ys[Dv];
                auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
                auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
                auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
                auto g_ = g + (b_idx * Hv + hv_idx) * Dk;
                const float bt = beta[b_idx * Hv + hv_idx];
                auto dk_idx = thread_position_in_threadgroup.x;
                auto rg = thread_position_in_threadgroup.y;
                float qv[n_per_t], kv[n_per_t], gv[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    qv[i] = q_[s_idx];
                    kv[i] = k_[s_idx];
                    gv[i] = g_[s_idx];
                }
                for (int r = 0; r < ROWS; ++r) {
                    auto dv_idx = rg * ROWS + r;
                    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
                    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
                    float state[n_per_t];
                    for (int i = 0; i < n_per_t; ++i)
                        state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
                    float kv_mem = 0.0f;
                    for (int i = 0; i < n_per_t; ++i) {
                        state[i] = state[i] * gv[i];
                        kv_mem += state[i] * kv[i];
                    }
                    kv_mem = simd_sum(kv_mem);
                    auto delta = (v_[dv_idx] - kv_mem) * bt;
                    float out = 0.0f;
                    for (int i = 0; i < n_per_t; ++i) {
                        state[i] = state[i] + kv[i] * delta;
                        out += state[i] * qv[i];
                    }
                    out = simd_sum(out);
                    if (dk_idx == 0) ys[dv_idx] = static_cast<float>(static_cast<InT>(out));
                    for (int i = 0; i < n_per_t; ++i)
                        o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (rg == 0) {
                    float xs[4];
                    float acc = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        const int h = dk_idx * 4 + i;
                        xs[i] = (h < Dv) ? ys[h] : 0.0f;
                        acc += xs[i] * xs[i];
                    }
                    acc = simd_sum(acc);
                    acc = simd_sum(dk_idx == 0 ? acc : 0.0f);
                    const float inv = metal::precise::rsqrt(acc / (float)Dv + eps[0]);
                    auto gate_ = gate + (b_idx * Hv + hv_idx) * Dv;
                    auto o_ = gated + (b_idx * Hv + hv_idx) * Dv;
                    for (int i = 0; i < 4; ++i) {
                        const int h = dk_idx * 4 + i;
                        if (h < Dv) {
                            const float nv = norm_w[h] * (xs[i] * inv);
                            const float z = static_cast<float>(gate_[h]);
                            const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
                            o_[h] = static_cast<bfloat16_t>(nv * ((z < 0.0f) ? e : 1.0f - e));
                        }
                    }
                }
            """)
}

/// One-token MoE in three kernels: the router, the gated up projections of the routed and shared
/// experts, and the down projections summed with the routing weights.
///
/// The products repeat the arithmetic of the MLX NVFP4 matrix-vector kernels, so the projections
/// round as the operation path does.
enum Raon1MoEKernels {
    /// Inputs with at most this many rows take the kernels.
    static let maximumRows = 8

    private static let rowsPerSIMDGroup = 4
    private static let simdGroups = 2
    private static let routeThreads = 128

    /// An NVFP4 projection: packed weight, E4M3 block scales and float32 global scale.
    typealias Weights = (weight: MLXArray, scales: MLXArray, globalScale: MLXArray)

    /// The three projections of the routed experts, with MLX global scales `[E]`, or of the
    /// shared expert, with per-tensor output scales `[1]`.
    struct Projections {
        let up: Weights
        let gate: Weights
        let down: Weights
    }

    /// Whether the kernels support `x` `[..., H]` with `experts` routed experts of width
    /// `intermediate`.
    static func supports(_ x: MLXArray, experts: Int, intermediate: Int, topK: Int) -> Bool {
        let hidden = x.dim(-1)
        let tile = rowsPerSIMDGroup * simdGroups
        return x.dtype == .bfloat16 && x.size / hidden <= maximumRows && hidden % 512 == 0
            && hidden % tile == 0 && intermediate % tile == 0 && intermediate % 16 == 0
            && experts % 32 == 0 && experts <= 1024 && topK <= 32
    }

    /// - Parameters:
    ///   - x: the rows `[N, H]`, bfloat16.
    ///   - router: the float32 router weight `[E, H]` and expert bias `[E]`.
    ///   - parameters: float32 `[SiTU beta, SiTU linear beta, routed scale]`.
    /// - Returns: the MoE output `[N, H]`.
    static func apply(
        _ x: MLXArray, router: (weight: MLXArray, bias: MLXArray), routed: Projections,
        shared: Projections?, topK: Int, renormalize: Bool, parameters: MLXArray
    ) -> MLXArray {
        let (rows, hidden) = (x.dim(0), x.dim(1))
        let experts = router.weight.dim(0)
        let intermediate = routed.up.weight.dim(1)
        let slots = topK + (shared == nil ? 0 : 1)
        let tile = rowsPerSIMDGroup * simdGroups
        let threads = 32 * simdGroups

        let logits = routeKernel(
            [x, router.weight],
            template: [("E", experts), ("H", hidden)],
            grid: (routeThreads * experts, rows, 1),
            threadGroup: (routeThreads, 1, 1),
            outputShapes: [[rows, experts]],
            outputDTypes: [.float32])[0]
        let selection: [(String, any KernelTemplateArg)] = [
            ("E", experts), ("TOPK", topK), ("RENORM", renormalize),
            ("HAS_SHARED", shared != nil), ("R", rowsPerSIMDGroup), ("NSG", simdGroups),
            ("H", hidden), ("I", intermediate),
        ]

        // Inputs under eight elements bind as `constant`, so the placeholders take eight.
        let placeholder = MLXArray.zeros([8], dtype: .uint32)
        let shared8 = MLXArray.zeros([8], dtype: .uint8)
        let one = MLXArray.ones([1], dtype: .float32)
        let upInputs: [MLXArray] = [
            x, logits, router.bias, parameters,
            routed.up.weight, routed.up.scales, routed.up.globalScale,
            routed.gate.weight, routed.gate.scales, routed.gate.globalScale,
            shared?.up.weight ?? placeholder, shared?.up.scales ?? shared8,
            shared?.up.globalScale ?? one,
            shared?.gate.weight ?? placeholder, shared?.gate.scales ?? shared8,
            shared?.gate.globalScale ?? one,
        ]
        let activated = upKernel(
            upInputs,
            template: selection,
            grid: (threads, intermediate / tile, rows * slots),
            threadGroup: (threads, 1, 1),
            outputShapes: [[rows, slots, intermediate]],
            outputDTypes: [.bfloat16])[0]

        return downKernel(
            [
                activated, logits, router.bias, parameters,
                routed.down.weight, routed.down.scales, routed.down.globalScale,
                shared?.down.weight ?? placeholder, shared?.down.scales ?? shared8,
                shared?.down.globalScale ?? one,
            ],
            template: selection,
            grid: (threads, hidden / tile, rows),
            threadGroup: (threads, 1, 1),
            outputShapes: [[rows, hidden]],
            outputDTypes: [.bfloat16])[0]
    }

    private static let header = """
        inline float raon_fp4(uint16_t bits) {
          half v = as_type<half>(ushort((bits & 7) << 9)) * 16384.0h;
          return static_cast<float>(bits & 8 ? -v : v);
        }
        inline float raon_e4m3(uint8_t bits) {
          uint16_t v = bits & 127;
          uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | (((uint16_t)((bits >> 7) & 1)) << 15);
          return static_cast<float>(as_type<half>(u) * 256.0h);
        }
        template <int N>
        inline float raon_qdot(const device uint8_t* w, const thread float* x, float scale) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0;
          for (int i = 0; i < N / 4; i++) {
            accum += (x[4 * i] * raon_fp4(ws[i]) + x[4 * i + 1] * raon_fp4(ws[i] >> 4) +
                      x[4 * i + 2] * raon_fp4(ws[i] >> 8) + x[4 * i + 3] * raon_fp4(ws[i] >> 12));
          }
          return scale * accum;
        }
        inline float raon_sigmoid(float z) {
          float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
          return (z < 0.0f) ? y : 1.0f - y;
        }
        // One SIMD group ranks sigmoid(logits) + bias and writes the top k experts and their
        // routing weights.
        template <int E, int TOPK, bool RENORM>
        inline void raon_select(
            const device float* logits, const device float* bias, float scale, uint lane,
            threadgroup uint* chosen, threadgroup float* weights) {
          constexpr int PER_LANE = E / 32;
          float score[PER_LANE], ranked[PER_LANE];
          for (int j = 0; j < PER_LANE; ++j) {
            score[j] = raon_sigmoid(logits[lane * PER_LANE + j]);
            ranked[j] = score[j] + bias[lane * PER_LANE + j];
          }
          float picked[TOPK];
          float total = 0.0f;
          for (int k = 0; k < TOPK; ++k) {
            float best = -INFINITY, best_score = 0.0f;
            uint where = E;
            for (int j = 0; j < PER_LANE; ++j) {
              if (ranked[j] > best) { best = ranked[j]; best_score = score[j]; where = lane * PER_LANE + j; }
            }
            const uint e = simd_min(best == simd_max(best) ? where : (uint)E);
            picked[k] = simd_shuffle(best_score, (ushort)(e / PER_LANE));
            if (e / PER_LANE == lane) ranked[e % PER_LANE] = -INFINITY;
            total += picked[k];
            if (lane == 0) chosen[k] = e;
          }
          if (lane == 0) {
            for (int k = 0; k < TOPK; ++k)
              weights[k] = (RENORM ? picked[k] / (total + 1e-20f) : picked[k]) * scale;
          }
        }
        """

    /// Each threadgroup scores one expert: the router logits `[N, E]`.
    private static let routeKernel = MLXFast.metalKernel(
        name: "raon1_moe_route",
        inputNames: ["x", "w"],
        outputNames: ["logits"],
        source: """
                const uint e = threadgroup_position_in_grid.x;
                const uint n = threadgroup_position_in_grid.y;
                const uint t = thread_position_in_threadgroup.x;
                const uint T = threads_per_threadgroup.x;
                const device bfloat16_t* xr = x + n * H;
                const device float* wr = w + e * H;
                float acc = 0.0f;
                for (int h = 4 * t; h < H; h += 4 * T) {
                    const float4 wv = *(const device float4*)(wr + h);
                    acc = metal::fma((float)xr[h], wv.x, acc);
                    acc = metal::fma((float)xr[h + 1], wv.y, acc);
                    acc = metal::fma((float)xr[h + 2], wv.z, acc);
                    acc = metal::fma((float)xr[h + 3], wv.w, acc);
                }
                acc = simd_sum(acc);
                threadgroup float partial[32];
                if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t == 0) {
                    float total = 0.0f;
                    for (uint g = 0; g < T / 32; ++g) total += partial[g];
                    logits[n * E + e] = total;
                }
            """)

    /// Each SIMD group computes `R` rows of one expert's up and gate projections and their SiTU.
    private static let upKernel = MLXFast.metalKernel(
        name: "raon1_moe_up_gate",
        inputNames: [
            "x", "logits", "bias", "prm", "wu", "su", "gu", "wg", "sg", "gg", "swu", "ssu", "sgu",
            "swg", "ssg", "sgg",
        ],
        outputNames: ["act"],
        source: """
                constexpr int SLOTS = TOPK + (HAS_SHARED ? 1 : 0);
                constexpr int BLK = 512;
                constexpr int KW = H / 2;
                constexpr int KG = H / 16;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const uint z = threadgroup_position_in_grid.z;
                const uint n = z / SLOTS;
                const uint slot = z - n * SLOTS;
                const int row0 = (threadgroup_position_in_grid.y * NSG + group) * R;
                const bool is_shared = HAS_SHARED && slot == TOPK;
                threadgroup uint chosen[TOPK];
                threadgroup float route_w[TOPK];
                if (group == 0 && !is_shared)
                    raon_select<E, TOPK, RENORM>(logits + n * E, bias, prm[2], lane, chosen, route_w);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint e = is_shared ? 0 : chosen[slot];
                const size_t expert_w = (size_t)e * I * KW, expert_s = (size_t)e * I * KG;
                const device uint8_t* bu = (is_shared ? (const device uint8_t*)swu : (const device uint8_t*)wu + expert_w)
                    + row0 * KW + lane * 8;
                const device uint8_t* bg = (is_shared ? (const device uint8_t*)swg : (const device uint8_t*)wg + expert_w)
                    + row0 * KW + lane * 8;
                const device uint8_t* cu = (is_shared ? ssu : su + expert_s) + row0 * KG + lane;
                const device uint8_t* cg = (is_shared ? ssg : sg + expert_s) + row0 * KG + lane;
                const device bfloat16_t* xr = x + n * H + lane * 16;
                float au[R] = {0}, ag[R] = {0};
                for (int k = 0; k < H; k += BLK) {
                    float xt[16];
                    for (int i = 0; i < 16; ++i) xt[i] = xr[k + i];
                    for (int r = 0; r < R; ++r) {
                        au[r] += raon_qdot<16>(bu + r * KW + k / 2, xt, raon_e4m3(cu[r * KG + k / 16]));
                        ag[r] += raon_qdot<16>(bg + r * KW + k / 2, xt, raon_e4m3(cg[r * KG + k / 16]));
                    }
                }
                const float beta = prm[0], linear_beta = prm[1];
                for (int r = 0; r < R; ++r) {
                    const float u = simd_sum(au[r]);
                    const float g = simd_sum(ag[r]);
                    if (lane == 0) {
                        float uv, gv;
                        if (is_shared) {
                            uv = (float)(bfloat16_t)((float)(bfloat16_t)u * sgu[0]);
                            gv = (float)(bfloat16_t)((float)(bfloat16_t)g * sgg[0]);
                        } else {
                            uv = (float)(bfloat16_t)(u * (gu[e] / (448.0f * 6.0f)));
                            gv = (float)(bfloat16_t)(g * (gg[e] / (448.0f * 6.0f)));
                        }
                        act[(size_t)z * I + row0 + r] = (bfloat16_t)(beta * metal::precise::tanh(gv / beta)
                            * raon_sigmoid(gv) * (linear_beta * metal::precise::tanh(uv / linear_beta)));
                    }
                }
            """,
        header: header)

    /// Each SIMD group computes `R` output rows over all experts and sums them with the routing
    /// weights and the shared expert.
    private static let downKernel = MLXFast.metalKernel(
        name: "raon1_moe_down_combine",
        inputNames: ["act", "logits", "bias", "prm", "wd", "sd", "gd", "swd", "ssd", "sgd"],
        outputNames: ["y"],
        source: """
                constexpr int SLOTS = TOPK + (HAS_SHARED ? 1 : 0);
                constexpr int BLK = 256;
                constexpr int NIT = (I - BLK + BLK - 1) / BLK;
                constexpr int KT = NIT * BLK;
                constexpr int KW = I / 2;
                constexpr int KG = I / 16;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const uint n = threadgroup_position_in_grid.z;
                const int row0 = (threadgroup_position_in_grid.y * NSG + group) * R;
                const int tail = metal::clamp(I - KT - (int)lane * 8, 0, 8);
                threadgroup uint chosen[TOPK];
                threadgroup float route_w[TOPK];
                if (group == 0)
                    raon_select<E, TOPK, RENORM>(logits + n * E, bias, prm[2], lane, chosen, route_w);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total[R] = {0}, shared_out[R] = {0};
                for (int slot = 0; slot < SLOTS; ++slot) {
                    const bool is_shared = HAS_SHARED && slot == TOPK;
                    const uint e = is_shared ? 0 : chosen[slot];
                    const device uint8_t* wb = (is_shared ? (const device uint8_t*)swd
                        : (const device uint8_t*)wd + (size_t)e * H * KW) + row0 * KW + lane * 4;
                    const device uint8_t* sb = (is_shared ? ssd : sd + (size_t)e * H * KG)
                        + row0 * KG + lane / 2;
                    const device bfloat16_t* xr = act + ((size_t)n * SLOTS + slot) * I + lane * 8;
                    float xt[NIT + 1][8];
                    for (int it = 0; it < NIT; ++it)
                        for (int i = 0; i < 8; ++i) xt[it][i] = xr[it * BLK + i];
                    for (int i = 0; i < 8; ++i) xt[NIT][i] = i < tail ? (float)xr[KT + i] : 0.0f;
                    for (int r = 0; r < R; ++r) {
                        float acc = 0.0f;
                        for (int it = 0; it < NIT; ++it)
                            acc += raon_qdot<8>(wb + r * KW + it * (BLK / 2), xt[it],
                                raon_e4m3(sb[r * KG + it * (BLK / 16)]));
                        if (tail > 0)
                            acc += raon_qdot<8>(wb + r * KW + NIT * (BLK / 2), xt[NIT],
                                raon_e4m3(sb[r * KG + NIT * (BLK / 16)]));
                        const float v = simd_sum(acc);
                        if (is_shared) {
                            shared_out[r] = (float)(bfloat16_t)((float)(bfloat16_t)v * sgd[0]);
                        } else {
                            total[r] += (float)(bfloat16_t)(v * (gd[e] / (448.0f * 6.0f)))
                                * route_w[slot];
                        }
                    }
                }
                if (lane == 0) {
                    for (int r = 0; r < R; ++r)
                        y[(size_t)n * H + row0 + r] = (bfloat16_t)((float)(bfloat16_t)total[r] + shared_out[r]);
                }
            """,
        header: header)
}
