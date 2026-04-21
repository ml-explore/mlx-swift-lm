// Copyright © 2026 Apple Inc.

import Foundation
import MLX

enum TurboQuantMetalKernels {

    /// Fused encode: norm, rotate, quantize, pack, and norm-correct in one dispatch.
    static let fusedEncodeSource = """
        constexpr uint LEVELS = 1u << Bits;

        uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
        uint row = thread_position_in_grid.y;         // vector index (B*H*T)

        // --- Step 1: Load input value ---
        float val = input[row * Dim + d];

        // --- Step 2: Compute L2 norm (SIMD reduction) ---
        float sq = val * val;
        float norm_sq = simd_sum(sq);
        // For Dim > 32, need threadgroup reduction
        threadgroup float shared_norm[4];  // up to 4 SIMD groups
        uint sg_id = d / 32;
        if (d % 32 == 0) {
            shared_norm[sg_id] = norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_norm_sq = 0;
        uint num_groups = (Dim + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            total_norm_sq += shared_norm[i];
        }
        float norm_val = sqrt(total_norm_sq);
        float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

        // --- Step 3: Normalize ---
        float unit_val = val * inv_norm;

        // --- Step 4: Rotate (y = Π · x_unit) via shared memory matmul ---
        // Each thread d computes: y[d] = Σ_j rotation[d * Dim + j] * x_unit[j]
        threadgroup float shared_unit[1024];  // max Dim = 1024
        shared_unit[d] = unit_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rotated = 0.0f;
        for (uint j = 0; j < Dim; j++) {
            rotated += rotation[d * Dim + j] * shared_unit[j];
        }

        // --- Step 5: Quantize via branchless boundary comparison ---
        // V2.1 optimization: use arithmetic sum of comparisons instead of branching.
        // Metal compiles (rotated > boundaries[b]) to a predicated 0/1 — summing these
        // is branchless and avoids SIMD lane divergence.
        uint idx = 0;
        for (uint b = 0; b < LEVELS - 1; b++) {
            idx += (uint)(rotated > boundaries[b]);
        }

        // --- Step 6: Pack bits into uint32 word (atomic OR) ---
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint masked = idx & ((1u << Bits) - 1u);

        // Pack bits — use threadgroup shared memory to avoid atomic contention
        // Each thread writes its index bits to shared, then thread 0 per word writes output
        threadgroup uint shared_packed[64];  // max PackedWidth = 64 words
        if (d < PackedWidth) shared_packed[d] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each dimension contributes its bits via atomic OR on threadgroup memory
        uint primary_val = masked << shift;
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            uint spill_val = masked >> ((uint)Bits - (uint)spill);
            atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write packed words to output (one thread per word)
        if (d < PackedWidth) {
            packed_out[row * PackedWidth + d] = shared_packed[d];
        }

        // --- Step 7: Norm correction ---
        // Compute reconstruction norm: ||codebook[idx]||₂ for the quantized unit vector.
        // Store corrected_norm = original_norm / recon_norm so that
        // decode(centroid[idx] * corrected_norm) better approximates the original vector.
        float centroid_val = codebook[idx];
        float recon_sq = centroid_val * centroid_val;
        float recon_norm_sq = simd_sum(recon_sq);
        // Threadgroup reduction for Dim > 32
        if (d % 32 == 0) {
            shared_norm[sg_id] = recon_norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_recon_sq = 0;
        for (uint i = 0; i < num_groups; i++) {
            total_recon_sq += shared_norm[i];
        }
        float recon_norm = sqrt(total_recon_sq);
        float corrected_norm = (recon_norm > 1e-8f) ? (norm_val / recon_norm) : norm_val;

        if (d == 0) {
            norms_out[row] = corrected_norm;
        }
        """

    /// Fused WHT encode: norm, WHT rotation, quantize, and pack (no norm correction).
    static let fusedEncodeWHTSource = """
        constexpr uint LEVELS = 1u << Bits;

        uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
        uint row = thread_position_in_grid.y;         // vector index (B*H*T)

        // --- Step 1: Load input value ---
        float val = input[row * Dim + d];

        // --- Step 2: Compute L2 norm (SIMD reduction) ---
        float sq = val * val;
        float norm_sq = simd_sum(sq);
        threadgroup float shared_norm[4];
        uint sg_id = d / 32;
        if (d % 32 == 0) {
            shared_norm[sg_id] = norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_norm_sq = 0;
        uint num_groups = (Dim + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            total_norm_sq += shared_norm[i];
        }
        float norm_val = sqrt(total_norm_sq);
        float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

        // --- Step 3: Normalize + sign flip (fused) ---
        // V2.1 optimization: pre-compute inv_norm * sign to eliminate one multiply per element.
        // Instead of: unit_val = val * inv_norm; wht_val = sign * unit_val (2 muls)
        // We do:      wht_val = val * (inv_norm * sign) (1 mul + 1 FMA-friendly product)
        float inv_norm_sign = inv_norm * wht_signs[d];
        float wht_val = val * inv_norm_sign;

        // --- Step 4: WHT rotation via cooperative SIMD shuffle ---
        // V2.1 optimization: use simd_shuffle_xor for intra-SIMD butterfly stages
        // (register-to-register, no shared memory or barriers needed for first 5 stages)

        // Phase 1: Intra-SIMD butterfly via simd_shuffle_xor (stages 0..min(LogDim,5)-1)
        // Each stage s XORs lane indices at distance 2^s — effectively free on Apple GPU
        // Use metal::min: MLX-injected headers add overloads named `min` (bf16_math.h), so
        // unqualified min(LogDim, 5u) is ambiguous vs metal::min on newer toolchains.
        uint log_dim_u = static_cast<uint>(LogDim);
        uint simd_stages = metal::min(log_dim_u, 5u);  // 5 stages covers 32 lanes (2^5 = 32)
        uint lane_in_simd = d % 32;
        for (uint s = 0; s < simd_stages; s++) {
            uint step = 1u << s;
            float other = simd_shuffle_xor(wht_val, step);
            wht_val = (lane_in_simd & step) ? (other - wht_val) : (other + wht_val);
        }

        // Phase 2: Cross-SIMD-group butterfly via shared memory (stages 5..LogDim-1)
        // Only needed when Dim > 32 — these stages cross SIMD group boundaries
        threadgroup float shared_buf[1024];  // max Dim = 1024
        if (log_dim_u > 5u) {
            shared_buf[d] = wht_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = simd_stages; s < log_dim_u; s++) {
                uint half_block = 1u << s;
                uint block_size = half_block << 1;
                uint block_id = d / block_size;
                uint pos_in_block = d % block_size;

                float a, b;
                if (pos_in_block < half_block) {
                    a = shared_buf[block_id * block_size + pos_in_block];
                    b = shared_buf[block_id * block_size + pos_in_block + half_block];
                    shared_buf[d] = a + b;
                } else {
                    a = shared_buf[block_id * block_size + pos_in_block - half_block];
                    b = shared_buf[block_id * block_size + pos_in_block];
                    shared_buf[d] = a - b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            wht_val = shared_buf[d];
        }

        // Normalize: WHT has scale factor sqrt(Dim)
        float inv_sqrt_dim = 1.0f / sqrt((float)Dim);
        float rotated = wht_val * inv_sqrt_dim;

        // --- Step 5: Quantize via branchless boundary comparison ---
        // V2.1 optimization: arithmetic sum avoids SIMD lane divergence
        uint idx = 0;
        for (uint b = 0; b < LEVELS - 1; b++) {
            idx += (uint)(rotated > boundaries[b]);
        }

        // --- Step 6: Pack bits into uint32 word (atomic OR) ---
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint masked = idx & ((1u << Bits) - 1u);

        threadgroup uint shared_packed[64];
        if (d < PackedWidth) shared_packed[d] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint primary_val = masked << shift;
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            uint spill_val = masked >> ((uint)Bits - (uint)spill);
            atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < PackedWidth) {
            packed_out[row * PackedWidth + d] = shared_packed[d];
        }

        // --- Step 7: Store raw norm (WHT is orthogonal — no norm correction needed) ---
        // WHT preserves norms: ||WHT(x)||₂ = ||x||₂. Reconstruction norm ≈ original norm,
        // so the correction ratio ≈ 1.0. Skipping saves codebook lookup + norm + division.
        if (d == 0) {
            norms_out[row] = norm_val;
        }
        """

    /// Flash attention pass 1 (causal): per-block partial attention with causal masking.
    static let turboFlashPass1CausalSource = """
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        // Runtime params from input buffers
        uint token_count = uint(tc_buf[0]);
        uint repeat_count = uint(rc_buf[0]);
        uint num_blocks = uint(nb_buf[0]);
        uint BlockSize = uint(bs_buf[0]);
        uint L = uint(L_buf[0]);
        uint q_offset = uint(qo_buf[0]);

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint block_idx = thread_position_in_grid.z; // token block index

        uint q_within_L = q_idx % L;
        uint q_head_idx = q_idx / L;
        uint kv_idx = q_head_idx / repeat_count;

        uint q_abs = q_offset + q_within_L;

        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > token_count) t_end = token_count;

        // Early exit: entire block is future-masked
        if (t_start > q_abs) {
            uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) o_partials[partial_base + d] = 0.0f;
            }
            if (lane == 0) {
                uint ml_idx = q_idx * num_blocks + block_idx;
                m_partials[ml_idx] = -INFINITY;
                l_partials[ml_idx] = 0.0f;
            }
            return;
        }

        // Clamp t_end to causal boundary
        if (t_end > q_abs + 1) t_end = q_abs + 1;

        // Load key codebook into registers
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) {
            key_cb[i] = key_codebook[i];
        }

        // Load value codebook into registers
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for this lane's dimensions
        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        // Online softmax state for this block
        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        // Process tokens in this block (up to causal boundary)
        for (uint t = t_start; t < t_end; t++) {
            // --- Score: Q×K dot product ---
            const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            float k_norm = key_norms[kv_idx * token_count + t];

            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;
                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;

                dot_partial += q_vals[i] * key_cb[k_value];
            }

            float score = simd_sum(dot_partial) * k_norm;

            // --- Online softmax update + V accumulation ---
            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];

            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;

                o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
            }

            l = l * exp_diff + exp_score;
            m = new_m;
        }

        // Write partial results: o[D], m, l
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o_partials[partial_base + d] = o[i];
            }
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = m;
            l_partials[ml_idx] = l;
        }
        """

    /// Flash attention pass 1 NR0 (causal): multi-row amortized KV dequant with per-row masking.
    static let turboFlashPass1NR0CausalSource = """
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        // Runtime params from input buffers (avoids per-token pipeline recompilation)
        uint token_count = uint(tc_buf[0]);
        uint repeat_count = uint(rc_buf[0]);
        uint num_blocks = uint(nb_buf[0]);
        uint BlockSize = uint(bs_buf[0]);
        uint L = uint(L_buf[0]);
        uint q_offset = uint(qo_buf[0]);

        uint lane = thread_position_in_grid.x;
        uint query_group = thread_position_in_grid.y;
        uint block_idx = thread_position_in_grid.z;

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > token_count) t_end = token_count;

        // Compute per-row causal boundaries and find the maximum (most permissive)
        // for the shared token loop. Per-row masking happens inside the score loop.
        uint q_abs[NR0];
        uint max_q_abs = 0;
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            uint q_within_L = q_idx % L;
            q_abs[r] = q_offset + q_within_L;
            if (q_abs[r] > max_q_abs) max_q_abs = q_abs[r];
        }

        // Early exit: entire block is future-masked for ALL NR0 queries
        if (t_start > max_q_abs) {
            for (uint r = 0; r < NR0; r++) {
                uint q_idx = query_group * NR0 + r;
                uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    uint d = lane + i * 32;
                    if (d < Dim) o_partials[partial_base + d] = 0.0f;
                }
                if (lane == 0) {
                    uint ml_idx = q_idx * num_blocks + block_idx;
                    m_partials[ml_idx] = -INFINITY;
                    l_partials[ml_idx] = 0.0f;
                }
            }
            return;
        }

        // Clamp t_end to the most permissive causal boundary
        if (t_end > max_q_abs + 1) t_end = max_q_abs + 1;

        // Load codebooks (shared across all NR0 queries)
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) key_cb[i] = key_codebook[i];
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) val_cb[i] = val_codebook[i];

        // Load query values for all NR0 rows
        float q_vals[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                q_vals[r * DIMS_PER_LANE + i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
            }
        }

        // KV head mapping (use first query's head — same assumption as non-causal NR0)
        uint q_head_idx_0 = (query_group * NR0) / L;
        uint kv_idx = q_head_idx_0 / repeat_count;

        // Online softmax state — NR0 independent streams
        float m_state[NR0];
        float l_state[NR0];
        float o_state[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            m_state[r] = -INFINITY;
            l_state[r] = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) o_state[r * DIMS_PER_LANE + i] = 0.0f;
        }

        // Process tokens — KV dequant once, score per-row with causal mask
        for (uint t = t_start; t < t_end; t++) {
            // Dequant K once
            float k_decoded[DIMS_PER_LANE];
            const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { k_decoded[i] = 0.0f; continue; }
                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;
                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;
                k_decoded[i] = key_cb[k_value];
            }
            float k_norm = key_norms[kv_idx * token_count + t];

            // Dequant V once
            float v_decoded[DIMS_PER_LANE];
            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { v_decoded[i] = 0.0f; continue; }
                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;
                v_decoded[i] = val_cb[v_value] * v_norm;
            }

            // Score + softmax + V for each query row (with per-row causal mask)
            for (uint r = 0; r < NR0; r++) {
                // Per-row causal: skip if this token is future for this specific query
                if (t > q_abs[r]) continue;

                float dot_partial = 0.0f;
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    dot_partial += q_vals[r * DIMS_PER_LANE + i] * k_decoded[i];
                }
                float score = simd_sum(dot_partial) * k_norm;

                float new_m = max(m_state[r], score);
                float exp_diff = exp(m_state[r] - new_m);
                float exp_score = exp(score - new_m);

                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    o_state[r * DIMS_PER_LANE + i] = o_state[r * DIMS_PER_LANE + i] * exp_diff + exp_score * v_decoded[i];
                }
                l_state[r] = l_state[r] * exp_diff + exp_score;
                m_state[r] = new_m;
            }
        }

        // Write partial results for all NR0 queries
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) o_partials[partial_base + d] = o_state[r * DIMS_PER_LANE + i];
            }
            if (lane == 0) {
                uint ml_idx = q_idx * num_blocks + block_idx;
                m_partials[ml_idx] = m_state[r];
                l_partials[ml_idx] = l_state[r];
            }
        }
        """

    /// Flash attention pass 2: cross-block reduction of partial softmax states.
    static let turboFlashPass2Source = """
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        // Runtime params from input buffers (avoids per-token pipeline recompilation)
        uint num_blocks = uint(nb_buf[0]);

        uint lane = thread_position_in_grid.x;
        uint q_idx = thread_position_in_grid.y;

        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        for (uint b = 0; b < num_blocks; b++) {
            uint ml_idx = q_idx * num_blocks + b;

            // All lanes read the same m/l (broadcast read from device memory)
            float block_m = m_partials[ml_idx];
            float block_l = l_partials[ml_idx];

            // Skip empty blocks
            if (block_l == 0.0f) continue;

            float new_m = max(m, block_m);
            float exp_old = exp(m - new_m);
            float exp_block = exp(block_m - new_m);

            uint partial_base = (q_idx * num_blocks + b) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
                }
            }

            l = l * exp_old + block_l * exp_block;
            m = new_m;
        }

        // Write normalized output
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                output[q_idx * Dim + d] = o[i] * inv_l;
            }
        }
        """

    /// Flash attention pass 2 with fused inverse value rotation.
    static let turboFlashPass2FusedRotSource = """
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        // Runtime params from input buffers (avoids per-token pipeline recompilation)
        uint num_blocks = uint(nb_buf[0]);

        uint lane = thread_position_in_grid.x;
        uint q_idx = thread_position_in_grid.y;

        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        for (uint b = 0; b < num_blocks; b++) {
            uint ml_idx = q_idx * num_blocks + b;

            float block_m = m_partials[ml_idx];
            float block_l = l_partials[ml_idx];

            if (block_l == 0.0f) continue;

            float new_m = max(m, block_m);
            float exp_old = exp(m - new_m);
            float exp_block = exp(block_m - new_m);

            uint partial_base = (q_idx * num_blocks + b) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
                }
            }

            l = l * exp_old + block_l * exp_block;
            m = new_m;
        }

        // Normalize
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

        // Gather normalized output into threadgroup shared memory for rotation
        threadgroup float shared_out[Dim];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                shared_out[d] = o[i] * inv_l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply inverse value rotation: output[d] = Σ_j shared_out[j] * Π_val[j][d]
        // matmul(x, Π_val) reads column d of Π_val for output dimension d.
        // Π_val is stored row-major [Dim, Dim], so column d = val_rotation[j * Dim + d]
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                float acc = 0.0f;
                for (uint j = 0; j < Dim; j++) {
                    acc += shared_out[j] * val_rotation[j * Dim + d];
                }
                output[q_idx * Dim + d] = acc;
            }
        }
        """

    /// Sparse V skip threshold. Override via `TURBO_SPARSE_V_THRESHOLD` env var.
    static let sparseVThreshold: Float = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_SPARSE_V_THRESHOLD"],
            let parsed = Float(envValue)
        {
            return parsed
        }
        return 1e-6
    }()

    /// Value aggregation: weighted sum of codebook-quantized values in rotated space.
    static var valueKernelSource: String {
        let threshold = String(format: "%e", sparseVThreshold)
        return """
            constexpr uint MASK = (1u << Bits) - 1u;
            constexpr uint LEVELS = 1u << Bits;

            // Runtime params from input buffers (avoids per-token pipeline recompilation)
            uint token_count = uint(tc_buf[0]);
            uint repeat_count = uint(rc_buf[0]);

            uint lane = thread_position_in_grid.x;
            uint head_idx = thread_position_in_grid.y;
            uint dim_block = thread_position_in_grid.z;

            uint d = dim_block * 32 + lane;
            if (d >= Dim) return;

            uint kv_head = head_idx / repeat_count;

            // Load codebook
            float cb[LEVELS];
            for (uint i = 0; i < LEVELS; i++) {
                cb[i] = codebook[i];
            }

            float acc = 0.0f;
            for (uint t = 0; t < token_count; t++) {
                float w = weights[head_idx * token_count + t];
                if (w < \(threshold)f) continue;  // Sparse V: skip negligible attention weights

                float norm_val = norms[kv_head * token_count + t];
                const device uint32_t* packed_ptr = packed + kv_head * token_count * PackedWidth + t * PackedWidth;

                uint bit_offset = d * Bits;
                uint word_idx = bit_offset / 32;
                uint shift = bit_offset % 32;
                uint value = (packed_ptr[word_idx] >> shift);

                int spill = (int)shift + (int)Bits - 32;
                if (spill > 0) {
                    value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
                }
                value &= MASK;

                acc += w * norm_val * cb[value];
            }

            output[head_idx * Dim + d] = acc;
            """
    }
}

public enum TurboQuantKernelOps {
    nonisolated(unsafe) private static var valueKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    /// Fused encode with dense rotation.
    nonisolated(unsafe) private static var encodeKernelCache: [String: MLXFast.MLXFastKernel] = [:]
    private static let encodeLock = NSLock()

    private static func getEncodeKernel(bits: Int, dim: Int) -> MLXFast.MLXFastKernel {
        let key = "encode_\(bits)_\(dim)"
        encodeLock.lock()
        if let cached = encodeKernelCache[key] {
            encodeLock.unlock()
            return cached
        }
        let kernel = MLXFast.metalKernel(
            name: "turbo_fused_encode_\(bits)_\(dim)",
            inputNames: ["input", "rotation", "boundaries", "codebook"],
            outputNames: ["packed_out", "norms_out"],
            source: TurboQuantMetalKernels.fusedEncodeSource
        )
        encodeKernelCache[key] = kernel
        encodeLock.unlock()
        return kernel
    }

    public static func fusedEncode(
        input: MLXArray,
        rotation: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let numRows = input.dim(0)
        let packedWidth = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let kernel = getEncodeKernel(bits: bits, dim: dim)

        let results = kernel(
            [input, rotation, boundaries, codebook],
            template: [
                ("Bits", bits),
                ("Dim", dim),
                ("PackedWidth", packedWidth),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, packedWidth], [numRows]],
            outputDTypes: [.uint32, .float32]
        )
        return (packed: results[0], norms: results[1])
    }

    /// Fused encode with WHT rotation (power-of-2 dims only).
    nonisolated(unsafe) private static var encodeWHTKernelCache: [String: MLXFast.MLXFastKernel] =
        [:]
    private static let encodeWHTLock = NSLock()

    private static func getEncodeWHTKernel(bits: Int, dim: Int) -> MLXFast.MLXFastKernel {
        let key = "encode_wht_\(bits)_\(dim)"
        encodeWHTLock.lock()
        if let cached = encodeWHTKernelCache[key] {
            encodeWHTLock.unlock()
            return cached
        }
        let kernel = MLXFast.metalKernel(
            name: "turbo_fused_encode_wht_\(bits)_\(dim)",
            inputNames: ["input", "wht_signs", "boundaries"],
            outputNames: ["packed_out", "norms_out"],
            source: TurboQuantMetalKernels.fusedEncodeWHTSource
        )
        encodeWHTKernelCache[key] = kernel
        encodeWHTLock.unlock()
        return kernel
    }

    public static func fusedEncodeWHT(
        input: MLXArray,
        whtSigns: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let numRows = input.dim(0)
        let packedWidth = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let kernel = getEncodeWHTKernel(bits: bits, dim: dim)

        let results = kernel(
            [input, whtSigns, boundaries],
            template: [
                ("Bits", bits),
                ("Dim", dim),
                ("PackedWidth", packedWidth),
                ("LogDim", Int(log2(Double(dim)))),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, packedWidth], [numRows]],
            outputDTypes: [.uint32, .float32]
        )
        return (packed: results[0], norms: results[1])
    }

    nonisolated(unsafe) private static var flashPass1Kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var flashPass2Kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// Query rows per SIMD group. Override via `TURBO_FLASH_NR0` env var.
    public static let flashNR0: Int = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_FLASH_NR0"],
            let parsed = Int(envValue), parsed > 0, (parsed & (parsed - 1)) == 0
        {
            return parsed
        }
        return 2
    }()

    /// Tokens per block in two-pass flash attention. Override via `TURBO_FLASH_BLOCK_SIZE` env var.
    public static let flashBlockSize: Int = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_FLASH_BLOCK_SIZE"],
            let parsed = Int(envValue), parsed > 0
        {
            return parsed
        }
        return 64
    }()

    /// Current sparse V skip threshold.
    public static var sparseVThreshold: Float { TurboQuantMetalKernels.sparseVThreshold }

    private static let flashPass1Lock = NSLock()

    private static func getFlashPass1Kernel(
        source: String, cachePrefix: String,
        keyBits: Int, valueBits: Int, dim: Int,
        extraInputNames: [String]
    ) -> MLXFast.MLXFastKernel {
        let key = "\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)"
        flashPass1Lock.lock()
        if let cached = flashPass1Kernels[key] {
            flashPass1Lock.unlock()
            return cached
        }
        let baseInputs = [
            "q_rot", "key_packed", "key_norms", "key_codebook",
            "val_packed", "val_norms", "val_codebook",
            "tc_buf", "rc_buf", "nb_buf", "bs_buf",
        ]
        let kernel = MLXFast.metalKernel(
            name: "turbo_flash_p1_\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)",
            inputNames: baseInputs + extraInputNames,
            outputNames: ["o_partials", "m_partials", "l_partials"],
            source: source
        )
        flashPass1Kernels[key] = kernel
        flashPass1Lock.unlock()
        return kernel
    }

    private static func dispatchFlashPass1(
        source: String, cachePrefix: String,
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int,
        extraInputNames: [String] = [],
        extraInputBuffers: [MLXArray] = []
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let kernel = getFlashPass1Kernel(
            source: source, cachePrefix: cachePrefix,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            extraInputNames: extraInputNames)

        let runtimeBufs: [MLXArray] = [
            MLXArray([Int32(tokenCount)]), MLXArray([Int32(repeatCount)]),
            MLXArray([Int32(numBlocks)]), MLXArray([Int32(blockSize)]),
        ]
        let keyPW = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let valPW = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)

        let results = kernel(
            [
                rotatedQueries, keyPacked, keyNorms, keyCodebook,
                valPacked, valNorms, valCodebook,
            ] + runtimeBufs + extraInputBuffers,
            template: [
                ("KeyBits", keyBits),
                ("ValueBits", valueBits),
                ("Dim", dim),
                ("KeyPackedWidth", keyPW),
                ("ValuePackedWidth", valPW),
            ],
            grid: (32, totalQ, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )
        return (oPartials: results[0], mPartials: results[1], lPartials: results[2])
    }

    /// NR0 multi-row causal pass 1 dispatch.
    private static func dispatchFlashPass1NR0Causal(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int, nr0: Int,
        queryChunkLength: Int, queryOffset: Int
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let kernel = getFlashPass1Kernel(
            source: TurboQuantMetalKernels.turboFlashPass1NR0CausalSource,
            cachePrefix: "nr0_causal", keyBits: keyBits, valueBits: valueBits, dim: dim,
            extraInputNames: ["L_buf", "qo_buf"])
        let runtimeBufs: [MLXArray] = [
            MLXArray([Int32(tokenCount)]), MLXArray([Int32(repeatCount)]),
            MLXArray([Int32(numBlocks)]), MLXArray([Int32(blockSize)]),
        ]
        let extraBufs = [MLXArray([Int32(queryChunkLength)]), MLXArray([Int32(queryOffset)])]
        let keyPW = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let valPW = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let results = kernel(
            [
                rotatedQueries, keyPacked, keyNorms, keyCodebook,
                valPacked, valNorms, valCodebook,
            ] + runtimeBufs + extraBufs,
            template: [
                ("KeyBits", keyBits), ("ValueBits", valueBits), ("Dim", dim),
                ("KeyPackedWidth", keyPW), ("ValuePackedWidth", valPW), ("NR0", nr0),
            ],
            grid: (32, totalQ / nr0, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )
        return (oPartials: results[0], mPartials: results[1], lPartials: results[2])
    }

    /// Pass 2 dispatch with optional fused output rotation.
    private static let flashPass2Lock = NSLock()

    private static func getFlashPass2Kernel(fused: Bool, dim: Int) -> MLXFast.MLXFastKernel {
        let key = "p2_\(fused ? "fused" : "plain")_\(dim)"
        flashPass2Lock.lock()
        if let cached = flashPass2Kernels[key] {
            flashPass2Lock.unlock()
            return cached
        }
        let source =
            fused
            ? TurboQuantMetalKernels.turboFlashPass2FusedRotSource
            : TurboQuantMetalKernels.turboFlashPass2Source
        let inputs =
            fused
            ? ["o_partials", "m_partials", "l_partials", "val_rotation", "nb_buf"]
            : ["o_partials", "m_partials", "l_partials", "nb_buf"]
        let kernel = MLXFast.metalKernel(
            name: "turbo_flash_p2_\(key)",
            inputNames: inputs,
            outputNames: ["output"],
            source: source
        )
        flashPass2Kernels[key] = kernel
        flashPass2Lock.unlock()
        return kernel
    }

    private static func dispatchFlashPass2(
        oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray,
        dim: Int, numBlocks: Int, totalQ: Int,
        valRotation: MLXArray? = nil
    ) -> MLXArray {
        let fused = valRotation != nil
        let kernel = getFlashPass2Kernel(fused: fused, dim: dim)
        var inputs: [MLXArray] = [oPartials, mPartials, lPartials]
        if let valRotation { inputs.append(valRotation) }
        inputs.append(MLXArray([Int32(numBlocks)]))

        let results = kernel(
            inputs,
            template: [("Dim", dim)],
            grid: (dim, totalQ, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[totalQ, dim]],
            outputDTypes: [.float32]
        )
        return results[0]
    }

    /// Two-pass flash attention with causal masking.
    public static func turboFlashAttentionCausal(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valueBits: Int,
        dim: Int,
        queryChunkLength: Int,
        queryOffset: Int,
        valRotation: MLXArray? = nil,
        blockSize: Int? = nil
    ) -> MLXArray {
        let blockSize = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let nr0 = flashNR0

        let useNR0 = nr0 > 1 && totalQ % nr0 == 0 && totalQ >= nr0

        let oPartials: MLXArray
        let mPartials: MLXArray
        let lPartials: MLXArray

        if useNR0 {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1NR0Causal(
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize, nr0: nr0,
                queryChunkLength: queryChunkLength, queryOffset: queryOffset
            )
        } else {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1(
                source: TurboQuantMetalKernels.turboFlashPass1CausalSource,
                cachePrefix: "flash_p1_causal",
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize,
                extraInputNames: ["L_buf", "qo_buf"],
                extraInputBuffers: [
                    MLXArray([Int32(queryChunkLength)]), MLXArray([Int32(queryOffset)]),
                ]
            )
        }

        return dispatchFlashPass2(
            oPartials: oPartials, mPartials: mPartials, lPartials: lPartials,
            dim: dim, numBlocks: numBlocks, totalQ: totalQ,
            valRotation: valRotation
        )
    }

    /// Weighted sum of packed codebook values. Result is in rotated space.
    private static let valueLock = NSLock()

    public static func mseWeightedSum(
        weights: MLXArray,
        packed: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let key = "value_\(bits)_\(dim)"
        valueLock.lock()
        if valueKernels[key] == nil {
            valueKernels[key] = MLXFast.metalKernel(
                name: "turbo_value_\(bits)_\(dim)",
                inputNames: ["weights", "packed", "norms", "codebook", "tc_buf", "rc_buf"],
                outputNames: ["output"],
                source: TurboQuantMetalKernels.valueKernelSource
            )
        }
        let kernel = valueKernels[key]!
        valueLock.unlock()

        let totalHeads = weights.dim(0)
        let packedWidth = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let results = kernel(
            [
                weights, packed, norms, codebook,
                MLXArray([Int32(tokenCount)]), MLXArray([Int32(repeatCount)]),
            ],
            template: [("Bits", bits), ("Dim", dim), ("PackedWidth", packedWidth)],
            grid: (32, totalHeads, (dim + 31) / 32),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalHeads, dim]],
            outputDTypes: [.float32]
        )
        return results[0]
    }
}
