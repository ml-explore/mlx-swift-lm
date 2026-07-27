// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Unit tests for `Gemma4.prepare(_:cache:state:prefill:)` chunked prefill.
///
/// The core property under test is **chunk-size invariance**: prefilling the
/// same prompt with a small `windowSize` (many chunks) and a `windowSize`
/// larger than the prompt (single pass) must agree on the final logits and
/// leave the KV caches at the same offsets. The synthetic config deliberately
/// covers Gemma 4's tricky structures: sliding-window layers whose rotating
/// caches wrap across chunk boundaries, a global-attention layer, KV-shared
/// tail layers, and per-layer inputs.
struct Gemma4ChunkedPrefillTests {

    /// Tiny Gemma4 built from a sparse JSON config (all other fields take
    /// the decoder defaults). 6 text layers with sliding_window_pattern 3
    /// → [sliding, sliding, full, sliding, sliding, full]; the last 2 are
    /// KV-shared, so 4 caches (3 rotating + 1 standard). sliding_window 8
    /// is much smaller than the test prompt, forcing rotation.
    private static func makeTinyModel() throws -> Gemma4 {
        let json = """
            {
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 6,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 32,
                    "vocab_size": 200,
                    "vocab_size_per_layer_input": 200,
                    "num_kv_shared_layers": 2,
                    "hidden_size_per_layer_input": 8,
                    "sliding_window": 8,
                    "sliding_window_pattern": 3,
                    "max_position_embeddings": 4096
                },
                "vision_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "head_dim": 8,
                    "position_embedding_size": 64
                }
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(json.utf8))
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // tests in this suite run concurrently and share the global RNG.
        let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
            Gemma4(config)
        }
        // Run in f16, as real models do.
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
        return model
    }

    /// Token ids well inside the tiny vocabulary.
    private static func makePrompt(count: Int) -> [Int] {
        (0 ..< count).map { ($0 * 7 + 3) % 199 }
    }

    private static func prefillLogits(
        model: Gemma4, tokens: [Int], windowSize: Int
    ) throws -> (logits: MLXArray, cacheOffsets: [Int]) {
        let cache = model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: .init(stepSize: windowSize))
        guard case .logits(let output) = result else {
            Issue.record("Expected .logits from Gemma4.prepare, got .tokens")
            return (MLXArray(0), [])
        }
        // Final-position logits regardless of how many positions the last
        // forward covered.
        let logits = output.logits[0..., -1, 0...]
        eval(logits)
        return (logits, cache.map { $0.offset })
    }

    @Test(
        "Chunked and single-pass prefill agree on logits and cache offsets",
        arguments: [3, 5, 8, 16])
    func chunkSizeInvariance(chunkSize: Int) throws {
        let model = try Self.makeTinyModel()
        let tokens = Self.makePrompt(count: 37)

        let chunked = try Self.prefillLogits(
            model: model, tokens: tokens, windowSize: chunkSize)
        let singlePass = try Self.prefillLogits(
            model: model, tokens: tokens, windowSize: 1024)

        #expect(chunked.cacheOffsets == singlePass.cacheOffsets)
        #expect(chunked.cacheOffsets.allSatisfy { $0 == tokens.count })

        // Max absolute difference, not allClose: relative error is meaningless
        // here because logits pass through zero.
        let drift = abs(
            chunked.logits.asType(.float32) - singlePass.logits.asType(.float32)
        ).max()
        eval(drift)
        #expect(
            drift.item(Float.self) < 5e-2,
            """
            Final-position logits differ between windowSize=\(chunkSize) and \
            single-pass prefill by \(drift.item(Float.self)) — chunked prefill \
            must be numerically equivalent under causal masking.
            """)
    }

    @Test("Prompt shorter than windowSize takes the single-chunk path")
    func shortPromptSingleChunk() throws {
        let model = try Self.makeTinyModel()
        let tokens = Self.makePrompt(count: 5)

        let result = try Self.prefillLogits(model: model, tokens: tokens, windowSize: 512)
        #expect(result.cacheOffsets.allSatisfy { $0 == tokens.count })
        #expect(result.logits.shape == [1, 200])
    }

    @Test("Unbatched (1-D) token input is accepted on the text path")
    func unbatchedTokensAccepted() throws {
        let model = try Self.makeTinyModel()
        let tokens = Self.makePrompt(count: 21)

        let cache = model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray(tokens))  // 1-D, no batch axis
        let result = try model.prepare(input, cache: cache, state: nil, prefill: .init(stepSize: 6))
        guard case .logits(let output) = result else {
            Issue.record("Expected .logits from Gemma4.prepare")
            return
        }
        eval(output.logits)
        #expect(cache.allSatisfy { $0.offset == tokens.count })

        // And it matches the batched run. Same shapes and same chunking, so the
        // two runs are bitwise equal; the tolerance only covers one f16 ulp,
        // which is ~2e-3 at these logit magnitudes.
        let batched = try Self.prefillLogits(model: model, tokens: tokens, windowSize: 6)
        let drift = abs(
            output.logits[0..., -1, 0...].asType(.float32) - batched.logits.asType(.float32)
        ).max()
        eval(drift)
        #expect(
            drift.item(Float.self) < 5e-3,
            "1-D and [1, N] token inputs must produce identical logits.")
    }
}
