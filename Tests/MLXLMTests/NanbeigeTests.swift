// Copyright © 2026 Apple Inc.
//
// Nanbeige (Looped Transformer) tests: the layer stack runs
// `effective_num_loops` times with shared weights, and each loop pass owns
// its own slice of the KV cache array. On a tiny random-weight model so they
// run in CI without downloads.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class NanbeigeTests: XCTestCase {

    // MARK: - Tiny model

    private func makeConfig(
        loopLossWeights: String = "[]", extra: String = ""
    ) throws -> NanbeigeConfiguration {
        let json = """
            {
                "model_type": "nanbeige",
                "hidden_size": 64,
                "num_hidden_layers": 3,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-5,
                "vocab_size": 512,
                "rope_theta": 70000000.0,
                "max_position_embeddings": 4096,
                "num_loops": 2,
                "loop_loss_weights": \(loopLossWeights),
                "skip_loop_final_norm": false\(extra)
            }
            """
        return try JSONDecoder().decode(NanbeigeConfiguration.self, from: Data(json.utf8))
    }

    /// Deterministic pseudo-random tokens, 1-D — the shape the default
    /// `LLMModel.prepare` chunked-prefill path expects.
    private func textTokens(_ count: Int, seed: Int = 0) -> MLXArray {
        var values: [Int32] = []
        for i in 0 ..< count {
            values.append(Int32((i * 13 + 7 + seed) % 512))
        }
        return MLXArray(values)
    }

    /// Prefill `tokens` into `cache` via `prepare` (the TokenIterator flow)
    /// and return the next-token logits from evaluating the remainder.
    private func prefillLogits(
        _ model: NanbeigeModel, _ tokens: MLXArray, cache: [KVCache]
    ) throws -> MLXArray {
        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: cache, state: nil, windowSize: 16)
        switch result {
        case .tokens(let remainder):
            let out = model(remainder.tokens[.newAxis], cache: cache)
            return out[0..., -1, 0...]
        case .logits(let out):
            return out.logits[0..., -1, 0...]
        }
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: - Configuration

    func testConfigDecodeAndEffectiveLoops() throws {
        let config = try makeConfig()
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.numLoops, 2)
        // Empty loop_loss_weights (the released Nanbeige4.2-3B config) must
        // fall through to num_loops, matching the Python truthiness check.
        XCTAssertEqual(config.effectiveNumLoops, 2)
    }

    func testEffectiveLoopsDerivedFromLossWeights() throws {
        let config = try makeConfig(loopLossWeights: "[0.5, 0.5]")
        // One weight per extra loop: 2 weights -> 3 loops, overriding num_loops.
        XCTAssertEqual(config.effectiveNumLoops, 3)
    }

    func testUnsupportedReferenceFeaturesThrow() throws {
        let config = try makeConfig(extra: ",\n\"enable_depth_attention\": true")
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testSupportedConfigValidates() throws {
        XCTAssertNoThrow(try makeConfig().validateModelConfiguration())
    }

    // MARK: - Cache layout

    func testNewCacheAllocatesOneCachePerLoopLayerPair() throws {
        let model = NanbeigeModel(try makeConfig())
        XCTAssertEqual(model.newCache(parameters: nil).count, 6)
    }

    /// The released Nanbeige4.2-3B shape: 22 hidden layers, 2 loops → 44
    /// caches (tiny dims so module init stays cheap).
    func testReleasedCheckpointShapeAllocates44Caches() throws {
        let json = """
            {
                "model_type": "nanbeige",
                "hidden_size": 16,
                "num_hidden_layers": 22,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 64,
                "num_loops": 2,
                "loop_loss_weights": [],
                "skip_loop_final_norm": false
            }
            """
        let config = try JSONDecoder().decode(NanbeigeConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(NanbeigeModel(config).newCache(parameters: nil).count, 44)
    }

    // MARK: - Sanitize

    func testSanitizeDropsRotaryFreqsAndTiedHead() throws {
        let tied = try makeConfig(extra: ",\n\"tie_word_embeddings\": true")
        let model = NanbeigeModel(tied)
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([Float(1)]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([Float(1)]),
            "lm_head.weight": MLXArray([Float(1)]),
        ])
        XCTAssertEqual(
            Set(sanitized.keys), ["model.layers.0.self_attn.q_proj.weight"])
    }

    // MARK: - Looped forward

    /// The loop must change the computation: a 2-loop forward differs from a
    /// 1-loop forward of the same weights (guards against a port that ignores
    /// `num_loops` and silently degenerates to a single pass).
    func testSecondLoopChangesLogits() throws {
        MLXRandom.seed(11)
        let looped = NanbeigeModel(try makeConfig())
        MLXRandom.seed(11)
        var singleConfig = try makeConfig()
        singleConfig.numLoops = 1
        let single = NanbeigeModel(singleConfig)

        let tokens = textTokens(9)[.newAxis]
        let loopedOut = looped(tokens, cache: nil)[0..., -1, 0...]
        let singleOut = single(tokens, cache: nil)[0..., -1, 0...]
        XCTAssertGreaterThan(maxAbsDiff(loopedOut, singleOut), 1e-4)
    }

    /// Warm continuation (prefix in cache, remainder prefilled on top) must
    /// match one cold prefill of the concatenation. This is the invariant
    /// that breaks if the per-loop cache slices are mis-indexed — loop 2
    /// reading loop 1's keys shows up here immediately.
    func testWarmContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(7)
        let model = NanbeigeModel(try makeConfig())
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)
        let full = concatenated([t1, t2], axis: 0)

        let cacheF = model.newCache(parameters: nil)
        let logitsF = try prefillLogits(model, full, cache: cacheF)

        let cacheW = model.newCache(parameters: nil)
        _ = try prefillLogits(model, t1, cache: cacheW)
        let logitsW = try prefillLogits(model, t2, cache: cacheW)

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "warm continuation diverged from cold full prefill")
    }
}
