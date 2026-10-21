// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class PromptCheckpointNumericsTests: XCTestCase {
    private func mellum() throws -> any LanguageModel {
        let json = #"""
            {
                "model_type": "mellum", "hidden_size": 32, "num_hidden_layers": 3,
                "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "num_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 32, "norm_topk_prob": true, "rms_norm_eps": 1e-6,
                "vocab_size": 128, "max_position_embeddings": 16384, "sliding_window": 32,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
                "mlp_layer_types": ["sparse", "dense", "sparse"],
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                }
            }
            """#
        return MellumModel(
            try JSONDecoder().decode(MellumConfiguration.self, from: Data(json.utf8)))
    }

    private func qwen() throws -> any LanguageModel {
        let json = #"""
            {
                "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "vocab_size": 128, "full_attention_interval": 2,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "max_position_embeddings": 16384,
                "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                    "partial_rotary_factor": 1.0, "type": "default"}
            }
            """#
        return Qwen35Model(
            try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }

    private func tokens(_ count: Int, seed: Int = 3) -> [Int] {
        (0 ..< count).map { ($0 * 7 + seed) % 128 }
    }

    private func storage(_ model: any LanguageModel) throws -> KVCacheStorage {
        KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)
    }

    private func prefill(
        _ tokens: [Int], model: any LanguageModel, cache: KVCacheStorage,
        state: LMOutput.State? = nil
    ) throws -> LMOutput.State? {
        try model.prefill(
            LMInput(tokens: MLXArray(tokens)), into: cache, state: state,
            parameters: .init(stepSize: 16))
    }

    private func lastLogits(
        _ tokens: [Int], model: any LanguageModel, cache: KVCacheStorage,
        state: LMOutput.State? = nil
    ) throws -> MLXArray {
        let state =
            tokens.count > 1
            ? try prefill(Array(tokens.dropLast()), model: model, cache: cache, state: state)
            : state
        let output = model.nextTokenLogits(
            .init(tokens: MLXArray([tokens.last!])[.newAxis]), cache: cache.cache, state: state)
        cache.commitProcessedTokens(1)
        eval(output.logits, cache.cache)
        return output.logits[0..., -1, 0...]
    }

    func testRestoredMellumAndRecurrentQwenMatchFreshPrefill() throws {
        for compressed in [false, true] {
            MLXRandom.seed(42)
            for model in [try mellum(), try qwen()] {
                if compressed {
                    quantize(model: model, groupSize: 32, bits: 4)
                    model.apply { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 }
                }
                let tolerance: Double = compressed ? 0.02 : 1e-4
                eval(model)
                let prefix = tokens(79)
                let live = try storage(model)
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prefix + [12])), model: model,
                    cacheStorage: live,
                    parameters: GenerateParameters(temperature: 0, prefill: .init(stepSize: 16)),
                    checkpointAt: prefix.count)
                let checkpoint = try XCTUnwrap(iterator.prefillCheckpoint)
                _ = try prefill(
                    tokens(96, seed: 11), model: model, cache: live, state: iterator.state)

                for count in [1, 17, 65] {
                    XCTAssertTrue(live.restore(checkpoint.cache))
                    let suffix = tokens(count, seed: count)
                    let actual = try lastLogits(
                        suffix, model: model, cache: live, state: checkpoint.state)
                    let replay = try storage(model)
                    let replayState = try prefill(prefix, model: model, cache: replay)
                    let expected = try lastLogits(
                        suffix, model: model, cache: replay, state: replayState)
                    let diagnostic = "\(type(of: model)) quantized=\(compressed) suffix=\(count)"
                    XCTAssertTrue(
                        allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self), diagnostic)
                    XCTAssertEqual(live.processedTokenCount, replay.processedTokenCount)
                    let actualLeaves = KVCacheTree.leaves(in: live.cache)
                    let replayLeaves = KVCacheTree.leaves(in: replay.cache)
                    for (a, b) in zip(actualLeaves, replayLeaves) where !a.isAttentionCache {
                        XCTAssertEqual(a.cache.state.count, b.cache.state.count)
                        for (a, b) in zip(a.cache.state, b.cache.state) {
                            XCTAssertTrue(
                                allClose(a, b, rtol: 0, atol: 0).item(Bool.self), diagnostic)
                        }
                    }

                    // BF16 may round differently when a cold prefill changes chunk boundaries.
                    let fresh = try storage(model)
                    let cold = try lastLogits(prefix + suffix, model: model, cache: fresh)
                    let delta = actual.asType(.float32) - cold.asType(.float32)
                    let relative =
                        (sqrt(mean(square(delta)))
                        / maximum(sqrt(mean(square(cold.asType(.float32)))), MLXArray(1e-8))).item(
                            Float.self)
                    XCTAssertLessThan(Double(relative), tolerance, diagnostic)
                }
            }
        }
    }

    func testOversizedPrefillCheckpointPreservesMellumLogits() throws {
        let model = try mellum()
        eval(model)
        let prefix = tokens(512)
        let live = try storage(model)
        let iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prefix + [12])), model: model, cacheStorage: live,
            parameters: GenerateParameters(temperature: 0), checkpointAt: prefix.count)
        let checkpoint = try XCTUnwrap(iterator.prefillCheckpoint)
        XCTAssertEqual(
            checkpoint.cache.retainedBytes, 2 * 2 * 2 * 32 * 32 * MemoryLayout<Float>.size)
        _ = try prefill(tokens(65), model: model, cache: live)
        XCTAssertTrue(live.restore(checkpoint.cache))
        let actual = try lastLogits([3], model: model, cache: live, state: checkpoint.state)
        let fresh = try storage(model)
        let state = try model.prefill(
            LMInput(tokens: MLXArray(prefix)), into: fresh, state: nil, parameters: .init())
        let expected = try lastLogits([3], model: model, cache: fresh, state: state)
        XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    func testCheckpointBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MLX_RUN_CHECKPOINT_BENCHMARKS"] == "1" else {
            return
        }
        for (name, model) in [("Mellum", try mellum()), ("Qwen35", try qwen())] {
            quantize(model: model, groupSize: 32, bits: 4)
            model.apply { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 }
            eval(model)
            for count in [512, 2048] {
                let prefix = tokens(count)
                let suffix = tokens(33, seed: 11)
                let live = try storage(model)
                let parameters = GenerateParameters(temperature: 0)
                func run(
                    _ input: [Int], cache: KVCacheStorage, checkpointAt position: Int? = nil,
                    state: LMOutput.State? = nil
                )
                    throws -> TokenIterator
                {
                    let iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(input)), model: model, cacheStorage: cache,
                        state: state, parameters: parameters, checkpointAt: position)
                    _ = iterator.y.tokens.item(Int.self)
                    return iterator
                }
                let iterator = try run(prefix + [12], cache: live, checkpointAt: count)
                let checkpoint = try XCTUnwrap(iterator.prefillCheckpoint)
                _ = try run(tokens(64, seed: 19), cache: live)
                var cold: [Double] = []
                var warm: [Double] = []
                var split: [Double] = []
                var unsplit: [Double] = []
                for _ in 0 ..< 4 {
                    let fresh = try storage(model)
                    let coldStart = Date.timeIntervalSinceReferenceDate
                    _ = try run(prefix + suffix, cache: fresh)
                    cold.append(Date.timeIntervalSinceReferenceDate - coldStart)
                    let warmStart = Date.timeIntervalSinceReferenceDate
                    XCTAssertTrue(live.restore(checkpoint.cache))
                    _ = try run(suffix, cache: live, state: checkpoint.state)
                    warm.append(Date.timeIntervalSinceReferenceDate - warmStart)
                    let splitCache = try storage(model)
                    let splitStart = Date.timeIntervalSinceReferenceDate
                    _ = try run(prefix + [12], cache: splitCache, checkpointAt: count)
                    split.append(Date.timeIntervalSinceReferenceDate - splitStart)
                    let unsplitCache = try storage(model)
                    let unsplitStart = Date.timeIntervalSinceReferenceDate
                    _ = try run(prefix + [12], cache: unsplitCache)
                    unsplit.append(Date.timeIntervalSinceReferenceDate - unsplitStart)
                }
                func median(_ samples: [Double]) -> Double {
                    samples.dropFirst().sorted()[1] * 1000
                }
                print(
                    String(
                        format:
                            "[CHECKPOINT] %@ prefix=%d cold=%.3fms restore=%.3fms split=%.3fms unsplit=%.3fms saved=%d bytes",
                        name, count, median(cold), median(warm), median(split), median(unsplit),
                        checkpoint.cache.retainedBytes))
            }
        }
    }
}
