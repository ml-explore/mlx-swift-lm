// Copyright © 2026

// Microbench / sanity test: does MLX's matmul kernel produce the same numerical
// output for `[1, 1, H] @ W` and `[1, L, H] @ W` at the corresponding row?
//
// This is the suspected root cause of the MTP-vs-baseline drift on small bf16
// targets — the verify path computes K/V at L=4 while the no-drafter baseline
// computes them at L=1, and any L-dependence in the matmul kernel propagates
// through SDPA into different argmaxes for tokens with narrow logit margins.
//
// Run with:
//   xcodebuild test -scheme mlx-swift-lm-Package \
//     -destination 'platform=macOS,arch=arm64' \
//     -only-testing:MLXLMTests/MatmulInvarianceTests -skipMacroValidation

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class MatmulInvarianceTests: XCTestCase {

    private func runOne(
        H: Int = 1024,
        O: Int = 256,
        L: Int = 4,
        dtype: DType = .bfloat16
    ) -> (maxDiff: Float, sampleSingle: [Float], sampleMulti: [Float]) {
        let weight = MLX.MLXRandom.normal([O, H]).asType(dtype)
        let row0 = MLX.MLXRandom.normal([H]).asType(dtype)
        var dummies: [MLXArray] = [row0]
        for _ in 1 ..< L {
            dummies.append(MLX.MLXRandom.normal([H]).asType(dtype))
        }
        let single = row0.reshaped([1, 1, H])
        let multi = stacked(dummies, axis: 0).reshaped([1, L, H])

        // Plain matmul: out = x @ W.T
        let wT = weight.transposed()
        let outSingle = matmul(single, wT)  // [1, 1, O]
        let outMulti = matmul(multi, wT)  // [1, L, O]
        eval(outSingle, outMulti)

        let s0 = outSingle[0, 0, 0...].asType(.float32)
        let m0 = outMulti[0, 0, 0...].asType(.float32)
        let diff = abs(s0 - m0).max().item(Float.self)
        let sSample = Array(s0[0 ..< 5].asArray(Float.self))
        let mSample = Array(m0[0 ..< 5].asArray(Float.self))
        return (diff, sSample, mSample)
    }

    func testMatmulRow0InvariantBF16() {
        let r = runOne(H: 1024, O: 256, L: 4, dtype: .bfloat16)
        print("[bf16] max abs diff (single row 0 vs multi row 0): \(r.maxDiff)")
        print("[bf16] single[0..5]: \(r.sampleSingle)")
        print("[bf16] multi[0..5]:  \(r.sampleMulti)")
        // Document the observed delta; do not strictly assert since MLX matmul
        // might not be bit-identical across M=1 vs M=L. Diff > 0 confirms the
        // hypothesis.
    }

    func testMatmulRow0InvariantFP32() {
        let r = runOne(H: 1024, O: 256, L: 4, dtype: .float32)
        print("[fp32] max abs diff (single row 0 vs multi row 0): \(r.maxDiff)")
        print("[fp32] single[0..5]: \(r.sampleSingle)")
        print("[fp32] multi[0..5]:  \(r.sampleMulti)")
    }

    func testMatmulLargerL() {
        for L in [2, 4, 8, 16, 32] {
            let r = runOne(H: 1024, O: 256, L: L, dtype: .bfloat16)
            print("[bf16 L=\(L)] max abs diff: \(r.maxDiff)")
        }
    }

    /// Test the suspected MTP-drift source directly: SDPA with the *same* Q/K/V at
    /// the corresponding query position, compared between L=1 (no causal mask) and
    /// L=N (with causal mask). For the LAST query position, the causal limit
    /// allows all keys, so the math should be identical.
    private func runSDPAOne(
        N: Int = 64,
        H: Int = 8,
        D: Int = 128,
        L: Int = 4,
        dtype: DType = .bfloat16
    ) -> Float {
        // Shared Q/K/V history: K, V have shape [B, H, N, D]
        let kAll = MLX.MLXRandom.normal([1, H, N + L, D]).asType(dtype)
        let vAll = MLX.MLXRandom.normal([1, H, N + L, D]).asType(dtype)

        // For the SINGLE-position case: query is at position N (last), K/V is
        // [..N+1] (= N+1 positions). Use sdpa with .none mask.
        let qSingle = MLX.MLXRandom.normal([1, H, 1, D]).asType(dtype)
        let kSingle = kAll[0..., 0..., 0 ..< (N + 1), 0...]
        let vSingle = vAll[0..., 0..., 0 ..< (N + 1), 0...]

        let outSingle = MLXFast.scaledDotProductAttention(
            queries: qSingle,
            keys: kSingle,
            values: vSingle,
            scale: 1.0 / Float(D).squareRoot(),
            mask: .none
        )

        // For the MULTI-position case: queries are at positions [N, N+1, ..., N+L-1].
        // Build qMulti with row 0 = qSingle's row, dummies for rest.
        var qRows: [MLXArray] = [qSingle.squeezed(axis: 2)]  // [1, H, D]
        for _ in 1 ..< L {
            qRows.append(MLX.MLXRandom.normal([1, H, D]).asType(dtype))
        }
        let qMulti = stacked(qRows, axis: 2)  // [1, H, L, D]
        let kMulti = kAll[0..., 0..., 0 ..< (N + L), 0...]
        let vMulti = vAll[0..., 0..., 0 ..< (N + L), 0...]

        let outMulti = MLXFast.scaledDotProductAttention(
            queries: qMulti,
            keys: kMulti,
            values: vMulti,
            scale: 1.0 / Float(D).squareRoot(),
            mask: .causal
        )

        eval(outSingle, outMulti)

        // Compare output[..., 0, ...] of multi (= attention at position N, which
        // attends to keys [..N], same as single's [..N+1] minus position N+1 onwards).
        // Wait — single's K has positions [0..N], with Q at position N. Single Q
        // attends to all N+1 keys (positions 0..N).
        // Multi Q at row 0 is at position N, attends to causal: keys 0..N (since
        // q_seq_idx=0 of multi corresponds to absolute position N due to KV offset).
        // Hmm — in MLX SDPA's causal interpretation, do_causal for multi at row 0
        // would attend to keys 0..(N + 0 - (L - 1)) = 0..(N - L + 1)? That's not
        // quite right.
        //
        // Actually for fair comparison, a non-trivial test would need to simulate
        // the full MTP cache state. For this microbench we just measure how much
        // the output at row 0 of multi differs from single, ignoring causal-mask
        // semantic alignment — what matters is whether the kernel is L-invariant
        // when fed the same Q[0] and same K/V history.
        let s = outSingle[0, 0..., 0, 0...].asType(.float32)
        let m = outMulti[0, 0..., 0, 0...].asType(.float32)
        return abs(s - m).max().item(Float.self)
    }

    func testSDPALInvariance() {
        for L in [1, 2, 4, 8] {
            let diff = runSDPAOne(N: 64, H: 8, D: 128, L: L, dtype: .bfloat16)
            print("[SDPA bf16 L=\(L)] max abs diff at row 0: \(diff)")
        }
    }

    /// More targeted: run sdpa with SAME [B, H, S, D] keys/values and SAME Q at
    /// the LAST query position. With L=1 (no mask) and L=N (causal mask), the
    /// last-position output should be mathematically identical (last query sees
    /// all keys regardless of mask). Diff measures the MLXFast kernel-path drift.
    /// Verify MaskedEmbedder dtype preservation + correctness of cluster routing.
    /// CoreML-LLM hit three bugs where token_ordering / topk indices were silently
    /// demoted to fp16 or uint16, truncating token IDs >32767 (vocab=262144).
    /// This test checks the MLX-Swift equivalents.
    func testMaskedEmbedderDtypeAndCorrectness() throws {
        let configJSON = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 64,
                "use_ordered_embeddings": true,
                "num_centroids": 8,
                "centroid_intermediate_top_k": 2,
                "tie_word_embeddings": true,
                "block_size": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "vocab_size": 64,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 0,
                    "sliding_window": 16,
                    "sliding_window_pattern": 5,
                    "tie_word_embeddings": true,
                    "use_double_wide_mlp": false,
                    "hidden_size_per_layer_input": 0,
                    "rms_norm_eps": 1.0e-6
                }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: configJSON)
        let drafter = Gemma4AssistantDraftModel(config)

        // Apply sanitize on a synthetic int64 token_ordering to mimic load path.
        let int64Ordering = MLXArray.zeros([64], dtype: .int64) + 1
        let sanitized = drafter.sanitize(weights: [
            "masked_embedding.token_ordering": int64Ordering
        ])
        let toAfter = sanitized["masked_embedding.token_ordering"]!
        XCTAssertEqual(
            toAfter.dtype, .int32,
            "sanitize must cast token_ordering int64 → int32 (CoreML-LLM bug #1)")

        // CoreML-LLM bug #2 analog: argPartition output dtype on bf16 input.
        let centroidLogits = MLX.MLXRandom.normal([1, 1, 8]).asType(.bfloat16)
        let topkIdx = MLX.argPartition(centroidLogits, kth: 6, axis: -1)
        eval(topkIdx)
        print("[MaskedEmbedder] argPartition output dtype: \(topkIdx.dtype)")
        // Verify it's an integer dtype that can hold values up to vocab_size (262144)
        XCTAssertTrue(
            [DType.int32, DType.int64, DType.uint32, DType.uint64].contains(topkIdx.dtype),
            "argPartition output must be int32/int64 (CoreML-LLM bug #2: not uint16). Got \(topkIdx.dtype)"
        )

        // CoreML-LLM bug #1/#3 analog: large token IDs (>32767) must survive
        // gather without truncation. MLX-Swift int32 = 4 bytes; verify dtype
        // preservation through indexing.
        let orderingValues: [Int32] = [0, 50000, 200000, 262143]
        let testOrdering = MLXArray(orderingValues)
        XCTAssertEqual(testOrdering.dtype, .int32)
        let pickIdx = MLXArray([Int32(2)])
        let gathered = testOrdering[pickIdx]
        eval(gathered)
        XCTAssertEqual(gathered.dtype, .int32, "Gather must preserve int32")
        XCTAssertEqual(
            gathered.item(Int32.self), Int32(200000),
            "Large index (>32767) must survive gather without uint16 truncation")
    }

    /// Quantized 4-bit Linear: is its matmul L-invariant? Target Gemma 4 is
    /// loaded as 4-bit, so q/k/v_proj actually call `quantizedMatmul` under
    /// the hood — that kernel may dispatch differently for M=1 vs M>1.
    func testQuantizedMatmulLInvariance() {
        let H = 1024
        let O = 256
        let weight = MLX.MLXRandom.normal([O, H]).asType(.bfloat16)
        let q = MLX.quantized(weight, groupSize: 64, bits: 4)
        let wq = q.wq
        let scales = q.scales
        let biases = q.biases

        for L in [2, 4, 8, 16] {
            let row0 = MLX.MLXRandom.normal([H]).asType(.bfloat16)
            var rows = [row0]
            for _ in 1 ..< L {
                rows.append(MLX.MLXRandom.normal([H]).asType(.bfloat16))
            }
            let single = row0.reshaped([1, 1, H])
            let multi = stacked(rows, axis: 0).reshaped([1, L, H])

            let outSingle = MLX.quantizedMatmul(
                single, wq, scales: scales, biases: biases, transpose: true,
                groupSize: 64, bits: 4)
            let outMulti = MLX.quantizedMatmul(
                multi, wq, scales: scales, biases: biases, transpose: true,
                groupSize: 64, bits: 4)
            eval(outSingle, outMulti)

            let s = outSingle[0, 0, 0...].asType(.float32)
            let m = outMulti[0, 0, 0...].asType(.float32)
            let diff = abs(s - m).max().item(Float.self)
            print("[quantized 4-bit L=\(L)] max abs diff at row 0: \(diff)")
            if L == 2 {
                print("  single[0..5]: \(Array(s[0 ..< 5].asArray(Float.self)))")
                print("  multi[0..5]:  \(Array(m[0 ..< 5].asArray(Float.self)))")
            }
        }
    }

    /// Critical test: SDPA at a NON-last position in multi-token forward,
    /// where causal mask DOES skip some keys (the ones beyond this query's
    /// reach). Compared against a single-token forward at the corresponding
    /// cache state (= same keys minus the skipped ones).
    /// RoPE L-invariance: does rope(x, offset=P) on x.shape=[B, H, 1, D] and
    /// rope(x_concat, offset=P) on x_concat.shape=[B, H, L, D] (where row 0 of
    /// x_concat == x) produce identical row 0 output? Position 0 of multi
    /// rotates at offset P + 0 = P, same as single.
    func testRoPELInvariance() {
        let H = 8
        let D = 128
        let offset = 64

        for L in [2, 4, 8] {
            let rowSingle = MLX.MLXRandom.normal([H, D]).asType(.bfloat16)
            var rows = [rowSingle]
            for _ in 1 ..< L {
                rows.append(MLX.MLXRandom.normal([H, D]).asType(.bfloat16))
            }
            let xSingle = rowSingle.reshaped([1, H, 1, D])
            let xMulti = stacked(rows, axis: 1).reshaped([1, H, L, D])

            // MLXFast.rope: apply rotary positional encoding.
            let outSingle = MLXFast.RoPE(
                xSingle, dimensions: D, traditional: false, base: 10000.0,
                scale: 1.0, offset: offset)
            let outMulti = MLXFast.RoPE(
                xMulti, dimensions: D, traditional: false, base: 10000.0,
                scale: 1.0, offset: offset)
            eval(outSingle, outMulti)

            let s = outSingle[0, 0..., 0, 0...].asType(.float32)
            let m = outMulti[0, 0..., 0, 0...].asType(.float32)
            let diff = abs(s - m).max().item(Float.self)
            print("[bf16 RoPE L=\(L)] row 0 diff: \(diff)")
        }
    }

    func testSDPAMidPositionDrift() {
        let H = 8
        let D = 128
        let basePromptLen = 64

        for L in [2, 4, 8] {
            for queryIdx in 0 ..< L {
                // Build a "real-shaped" scenario:
                // - Multi forward: Q at L positions, K/V has basePromptLen + L keys
                // - Single forward at the corresponding query position:
                //   Q at q_idx of multi corresponds to absolute position
                //   basePromptLen + q_idx; cache for single has keys
                //   0..(basePromptLen + q_idx) (= basePromptLen + q_idx + 1 keys).

                // Generate full Q sequence and K/V.
                let qFull = MLX.MLXRandom.normal([1, H, L, D]).asType(.bfloat16)
                let kFull = MLX.MLXRandom.normal([1, H, basePromptLen + L, D]).asType(
                    .bfloat16)
                let vFull = MLX.MLXRandom.normal([1, H, basePromptLen + L, D]).asType(
                    .bfloat16)

                // Multi SDPA: causal, see all of K (kernel masks within).
                let outMulti = MLXFast.scaledDotProductAttention(
                    queries: qFull,
                    keys: kFull,
                    values: vFull,
                    scale: 1.0 / Float(D).squareRoot(),
                    mask: .causal
                )

                // Single SDPA: extract Q at queryIdx (=> [1, H, 1, D]), K/V up to
                // basePromptLen + queryIdx + 1 keys. Mask=.none (Q at last attends
                // to all).
                let qSingle = qFull[0..., 0..., queryIdx ... queryIdx, 0...]
                let kSingle = kFull[0..., 0..., 0 ..< (basePromptLen + queryIdx + 1), 0...]
                let vSingle = vFull[0..., 0..., 0 ..< (basePromptLen + queryIdx + 1), 0...]
                let outSingle = MLXFast.scaledDotProductAttention(
                    queries: qSingle,
                    keys: kSingle,
                    values: vSingle,
                    scale: 1.0 / Float(D).squareRoot(),
                    mask: .none
                )

                eval(outMulti, outSingle)
                let m = outMulti[0, 0..., queryIdx, 0...].asType(.float32)
                let s = outSingle[0, 0..., 0, 0...].asType(.float32)
                let diff = abs(s - m).max().item(Float.self)
                if diff > 0 {
                    print("[bf16 L=\(L) q_idx=\(queryIdx)] DRIFT: \(diff)")
                } else {
                    print("[bf16 L=\(L) q_idx=\(queryIdx)] match")
                }
            }
        }
    }

    /// **Real-model integration test**: build a small (random-weights) Gemma4
    /// text model and verify that running L single-token forwards produces the
    /// same cache K/V and same per-position logits as one multi-token forward
    /// of length L on the same input sequence.
    /// cache.update with L=1 vs L=4: does writing K[0..1] vs K[0..4] produce the
    /// same stored value at slot 0 in the cache buffer? If matmul / RoPE all
    /// produce identical row 0, but cache.state[..., 0, ...] differs between
    /// the two paths, the L-dependence is in the slice-assignment kernel.
    func testCacheUpdateLInvariance() {
        let kvHeads = 1
        let headDim = 16

        // K1 = [1, kvHeads, 1, headDim], K4 = [1, kvHeads, 4, headDim]
        // with K4[..., 0, ...] == K1[..., 0, ...] by construction.
        let row0 = MLX.MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float32)
        let extra3 = MLX.MLXRandom.normal([1, kvHeads, 3, headDim]).asType(.float32)
        let v0 = MLX.MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float32)
        let vExtra3 = MLX.MLXRandom.normal([1, kvHeads, 3, headDim]).asType(.float32)
        let K4 = concatenated([row0, extra3], axis: 2)
        let V4 = concatenated([v0, vExtra3], axis: 2)

        // Path A: 4 single updates with K[0], K[1], K[2], K[3] separately.
        let cacheA = KVCacheSimple()
        let (_, _) = cacheA.update(keys: row0, values: v0)
        let (_, _) = cacheA.update(
            keys: extra3[0..., 0..., 0 ..< 1, 0...],
            values: vExtra3[0..., 0..., 0 ..< 1, 0...])
        let (_, _) = cacheA.update(
            keys: extra3[0..., 0..., 1 ..< 2, 0...],
            values: vExtra3[0..., 0..., 1 ..< 2, 0...])
        let (_, _) = cacheA.update(
            keys: extra3[0..., 0..., 2 ..< 3, 0...],
            values: vExtra3[0..., 0..., 2 ..< 3, 0...])

        // Path B: 1 multi update with all 4 K's at once.
        let cacheB = KVCacheSimple()
        let (_, _) = cacheB.update(keys: K4, values: V4)

        let stateA = cacheA.state
        let stateB = cacheB.state
        eval(stateA[0], stateB[0])

        let kA = stateA[0].asType(.float32)
        let kB = stateB[0].asType(.float32)
        print("[cache.update] kA shape: \(kA.shape), kB shape: \(kB.shape)")
        let kDiff = abs(kA - kB).max().item(Float.self)
        let vDiff = abs(stateA[1].asType(.float32) - stateB[1].asType(.float32)).max().item(
            Float.self)
        print("[cache.update] K diff: \(kDiff), V diff: \(vDiff)")
    }

    func testRMSNormLInvariance() {
        let H = 64
        let weight = MLX.MLXRandom.normal([H]).asType(.float32) * 0.1 + 1.0  // typical RMSNorm weight near 1
        for L in [2, 4, 8] {
            let row0 = MLX.MLXRandom.normal([H]).asType(.float32)
            var rows = [row0]
            for _ in 1 ..< L { rows.append(MLX.MLXRandom.normal([H]).asType(.float32)) }
            let single = row0.reshaped([1, 1, H])
            let multi = stacked(rows, axis: 0).reshaped([1, L, H])

            let outSingle = MLXFast.rmsNorm(single, weight: weight, eps: 1e-6)
            let outMulti = MLXFast.rmsNorm(multi, weight: weight, eps: 1e-6)
            eval(outSingle, outMulti)
            let s = outSingle[0, 0, 0...].asType(.float32)
            let m = outMulti[0, 0, 0...].asType(.float32)
            let diff = abs(s - m).max().item(Float.self)
            print("[fp32 RMSNorm L=\(L)] row 0 diff: \(diff)")
        }
    }

    func testGemma4FullModelLInvariance() throws {
        // Build a tiny config so the test runs fast and doesn't need a real
        // model download.
        let configJSON = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 32,
                "vocab_size": 512,
                "vocab_size_per_layer_input": 0,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 128,
                "sliding_window_pattern": 5,
                "tie_word_embeddings": true,
                "use_double_wide_mlp": false,
                "rms_norm_eps": 1.0e-6,
                "final_logit_softcapping": 30.0
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configJSON)
        let model = Gemma4TextModel(config)
        eval(model)  // realize random init

        let promptLen = 6
        let extraLen = 4
        let totalLen = promptLen + extraLen
        // Random token IDs in vocab range.
        let allTokens = MLX.MLXRandom.randInt(low: 1, high: 511, [totalLen]).asType(.int32)
        let promptTokens = allTokens[0 ..< promptLen].reshaped([1, promptLen])

        // Path A: prompt prefill + single-token forwards for the next extraLen tokens.
        let cacheA = model.newCache(parameters: nil)
        _ = model(promptTokens, cache: cacheA)
        for i in 0 ..< extraLen {
            let tok = allTokens[promptLen + i ..< promptLen + i + 1].reshaped([1, 1])
            _ = model(tok, cache: cacheA)
        }
        eval(cacheA[0].state.first ?? MLXArray(0))

        // Path B: prompt prefill + ONE multi-token forward of the next extraLen tokens.
        let cacheB = model.newCache(parameters: nil)
        _ = model(promptTokens, cache: cacheB)
        let multiInput = allTokens[promptLen ..< totalLen].reshaped([1, extraLen])
        let logitsB = model(multiInput, cache: cacheB)
        eval(logitsB)

        // Compare cache K/V at each layer.
        XCTAssertEqual(cacheA.count, cacheB.count, "cache layer counts differ")
        for layerIdx in 0 ..< cacheA.count {
            let stateA = cacheA[layerIdx].state
            let stateB = cacheB[layerIdx].state
            guard stateA.count == 2, stateB.count == 2 else { continue }
            let kA = stateA[0].asType(.float32)
            let kB = stateB[0].asType(.float32)
            let vA = stateA[1].asType(.float32)
            let vB = stateB[1].asType(.float32)

            let kDiff = abs(kA - kB).max().item(Float.self)
            let vDiff = abs(vA - vB).max().item(Float.self)
            print("[layer \(layerIdx)] K diff: \(kDiff), V diff: \(vDiff)")
        }

        // Compare logits at the last position.
        // Path A's last logits are not directly accessible; redo a single forward
        // that captures the last logits.
        let cacheALogits = model.newCache(parameters: nil)
        _ = model(promptTokens, cache: cacheALogits)
        for i in 0 ..< extraLen - 1 {
            let tok = allTokens[promptLen + i ..< promptLen + i + 1].reshaped([1, 1])
            _ = model(tok, cache: cacheALogits)
        }
        let lastTokA = allTokens[totalLen - 1 ..< totalLen].reshaped([1, 1])
        let logitsA_last = model(lastTokA, cache: cacheALogits)
        eval(logitsA_last)

        let logitsBLast = logitsB[0..., -1, 0...]
        let logitsALast = logitsA_last[0..., 0, 0...]
        let logitsDiff = abs(logitsALast.asType(.float32) - logitsBLast.asType(.float32))
            .max()
            .item(Float.self)
        let argA = argMax(logitsALast, axis: -1).item(Int.self)
        let argB = argMax(logitsBLast, axis: -1).item(Int.self)
        print("[final logits] max abs diff: \(logitsDiff), argmaxA=\(argA), argmaxB=\(argB)")
    }

    func testSDPALastPositionDrift() {
        for L in [2, 4, 8, 16] {
            let H = 8
            let D = 128
            let S = 64

            let q = MLX.MLXRandom.normal([1, H, 1, D]).asType(.bfloat16)
            let k = MLX.MLXRandom.normal([1, H, S, D]).asType(.bfloat16)
            let v = MLX.MLXRandom.normal([1, H, S, D]).asType(.bfloat16)

            // Single-position SDPA: q (last position), k/v with all S keys.
            let outSingle = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: 1.0 / Float(D).squareRoot(),
                mask: .none
            )

            // Multi-position with q at LAST position, dummy queries before.
            var qRows: [MLXArray] = []
            for _ in 0 ..< (L - 1) {
                qRows.append(MLX.MLXRandom.normal([1, H, D]).asType(.bfloat16))
            }
            qRows.append(q.squeezed(axis: 2))
            let qMulti = stacked(qRows, axis: 2)  // [1, H, L, D]

            // Causal mask: last query sees all S keys (since L-1 + offset_into_kv
            // < S). Need to ensure kv_len >= L for the causal logic. Use S keys
            // unchanged; multi's last query sees up to (S - L + (L-1)) = S - 1
            // keys with do_causal — which is fine for first S-1 keys but
            // EXCLUDES key at index S-1.
            //
            // To keep things simple, call with .causal and see if last-row
            // output matches.
            let outMulti = MLXFast.scaledDotProductAttention(
                queries: qMulti,
                keys: k,
                values: v,
                scale: 1.0 / Float(D).squareRoot(),
                mask: .causal
            )

            eval(outSingle, outMulti)

            let s = outSingle[0, 0..., 0, 0...].asType(.float32)
            // Last row of multi
            let m = outMulti[0, 0..., L - 1, 0...].asType(.float32)
            let diff = abs(s - m).max().item(Float.self)
            print("[bf16 L=\(L)] SDPA last-row diff vs single: \(diff)")
        }
    }
}
