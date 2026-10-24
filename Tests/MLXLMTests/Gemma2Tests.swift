// Copyright © 2026 Apple Inc.
//
// Gemma2 mask regressions on a tiny random-weight model. Gemma2 computes
// attention by hand and applies the mask itself, so two separate mistakes
// hide in that one line: sizing the mask from the absolute cache offset
// (wrong once the cache starts evicting) and adding a boolean mask to the
// scores (which keeps masked keys alive instead of suppressing them).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class Gemma2Tests: XCTestCase {

    // MARK: - Tiny model

    private func makeTinyModel() throws -> Gemma2Model {
        let json = """
            {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "head_dim": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 512,
                "num_key_value_heads": 2,
                "rope_theta": 10000.0,
                "rope_traditional": false,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "query_pre_attn_scalar": 144.0
            }
            """
        let config = try JSONDecoder().decode(Gemma2Configuration.self, from: Data(json.utf8))
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // parallel tests must not share (or perturb) the global random stream.
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Gemma2Model(config) }
    }

    /// Deterministic pseudo-random tokens, batched `[1, count]`, away from ids
    /// that could collide with special tokens.
    private func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        let values = (0 ..< count).map { Int32(($0 * 13 + 7 + Int(seed)) % 480) }
        return MLXArray(values).expandedDimensions(axis: 0)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: - Mask application

    /// One multi-token forward must match token-by-token decode: the causal
    /// mask has to suppress future keys, not boost the kept ones.
    func testCausalPrefillMatchesTokenByTokenDecode() throws {
        let model = try makeTinyModel()
        let tokens = textTokens(12)

        let cacheFull = try model.newCache(parameters: nil)
        let full = model(tokens, cache: cacheFull)[0..., -1, 0...]
        eval(full)

        let cacheStep = try model.newCache(parameters: nil)
        var last = MLXArray(0)
        for index in 0 ..< tokens.dim(1) {
            last = model(tokens[0..., index ..< index + 1], cache: cacheStep)[0..., -1, 0...]
            eval(last)
        }

        XCTAssertLessThanOrEqual(maxAbsDiff(full, last), MatmulPrecision.splitTolerance)
    }

    // MARK: - Warm continuation

    /// With a capacity-limited cache the mask must span the keys the cache
    /// still presents: after eviction the warm multi-token continuation used
    /// to size its mask from the absolute offset and the score broadcast
    /// failed. The suffix cases include single-token and over-window
    /// continuations.
    func testWarmContinuationAfterCacheRotationMatchesDecode() throws {
        for (capacity, suffixLength) in [(16, 6), (16, 1), (16, 23), (1, 6)] {
            let model = try makeTinyModel()
            let parameters = GenerateParameters(
                kvCache: KVCacheConfiguration(
                    capacity: try .init(maxTokens: capacity, preservedPrefixTokens: 0)))
            let cache = try model.newCache(parameters: parameters)
            eval(model(textTokens(capacity), cache: cache))
            let decoded = textTokens(5, seed: 3)
            for index in 0 ..< decoded.dim(1) {
                eval(model(decoded[0..., index ..< index + 1], cache: cache))
            }
            XCTAssertTrue(cache.allSatisfy { $0.offset == capacity + 5 })

            let referenceCache = cache.map { $0.copy() }
            let suffix = textTokens(suffixLength, seed: 9)
            var expected = [MLXArray]()
            for index in 0 ..< suffix.dim(1) {
                let output = model(suffix[0..., index ..< index + 1], cache: referenceCache)
                eval(output)
                expected.append(output)
            }

            let output = model(suffix, cache: cache)
            eval(output)
            XCTAssertLessThanOrEqual(
                maxAbsDiff(output, concatenated(expected, axis: 1)),
                MatmulPrecision.splitTolerance)
            XCTAssertTrue(cache.allSatisfy { $0.offset == capacity + 5 + suffixLength })
        }
    }
}
