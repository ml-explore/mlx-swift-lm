// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen35PrefillLogitsTests: XCTestCase {
    private func configuration(tied: Bool = false, moe: Bool = false) -> Data {
        Data(
            """
            {
                "model_type": "qwen3_5", "vocab_size": 512,
                "image_token_id": 500, "video_token_id": 501,
                "vision_start_token_id": 502, "vision_end_token_id": 503,
                "text_config": {
                    "model_type": "qwen3_5_text", "hidden_size": 64,
                    "num_hidden_layers": 2, "intermediate_size": 128,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                    "vocab_size": 512, "full_attention_interval": 2,
                    "tie_word_embeddings": \(tied), "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1, "linear_key_head_dim": 32,
                    "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                    "num_experts": \(moe ? 4 : 0), "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
                    "max_position_embeddings": 4096,
                    "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                        "partial_rotary_factor": 1.0, "type": "default"}
                },
                "vision_config": {"model_type": "qwen3_vl", "depth": 1,
                    "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
                    "num_heads": 2, "patch_size": 16, "spatial_merge_size": 2,
                    "temporal_patch_size": 2, "num_position_embeddings": 64}
            }
            """.utf8)
    }

    private func prepareWeights(_ model: Module, bits: Int? = 4, dtype: DType = .bfloat16) {
        if let bits { quantize(model: model, groupSize: 32, bits: bits) }
        model.apply { $0.dtype == .float32 ? $0.asType(dtype) : $0 }
        eval(model)
    }

    private func text(_ count: Int, batch: Int = 1) -> LMInput.Text {
        .init(
            tokens: MLXArray((0 ..< count * batch).map { ($0 * 13 + 7) % 480 }).reshaped(
                batch, count))
    }

    private func equal(
        _ a: MLXArray, _ b: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        XCTAssertEqual(a.dtype, b.dtype, file: file, line: line)
        XCTAssertTrue(allClose(a, b, rtol: 0, atol: 0).item(Bool.self), file: file, line: line)
    }

    private func equalCache(
        _ a: [KVCache], _ b: [KVCache], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.offset, y.offset, file: file, line: line)
            XCTAssertEqual(x.state.count, y.state.count, file: file, line: line)
            for (u, v) in zip(x.state, y.state) { equal(u, v, file: file, line: line) }
        }
    }

    func testTextDenseAndMoEPreserveLogitsAndRecurrentCache() throws {
        for tied in [false, true] {
            for moe in [false, true] {
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    let config = try JSONDecoder().decode(
                        MLXLLM.Qwen35Configuration.self, from: configuration(tied: tied, moe: moe))
                    let model = withRandomState(MLXRandom.RandomState(seed: 17)) {
                        Qwen35Model(config)
                    }
                    prepareWeights(model, dtype: dtype)
                    for count in [
                        1, 31, 32, 63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257, 511, 512,
                        513,
                    ] {
                        try autoreleasepool {
                            let a = try model.newCache(parameters: nil)
                            let b = try model.newCache(parameters: nil)
                            let expected = model(text(count), cache: a, state: nil).logits
                            let actual = model.nextTokenLogits(text(count), cache: b, state: nil)
                                .logits
                            XCTAssertEqual(expected.dim(1), count)
                            if count >= 96 { XCTAssertLessThanOrEqual(actual.dim(1), 95) }
                            equal(actual[0..., -1, 0...], expected[0..., -1, 0...])
                            equalCache(a, b)
                        }
                    }
                }
            }
        }
    }

    func testTextWarmContinuationAndMTPState() throws {
        let model = Qwen35Model(
            try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: configuration()))
        prepareWeights(model, bits: 8)
        for parameters in [
            GenerateParameters(), GenerateParameters(maxKVSize: 128),
            GenerateParameters(kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0),
        ] {
            let prefix = try model.newCache(parameters: parameters)
            eval(model(text(97), cache: prefix, state: nil).logits, prefix)
            let a = prefix.map { $0.copy() }
            let b = prefix.map { $0.copy() }
            for count in [129, 1, 1, 96, 1] {
                let expected = model(text(count), cache: a, state: nil).logits
                let actual = model.nextTokenLogits(text(count), cache: b, state: nil).logits
                equal(actual[0..., -1, 0...], expected[0..., -1, 0...])
                equalCache(a, b)
            }
        }
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let a = try model.newCache(parameters: nil)
        let b = try model.newCache(parameters: nil)
        let expected = model(text(128), cache: a, state: state)
        let actual = model.nextTokenLogits(text(128), cache: b, state: state)
        equal(actual.logits, expected.logits)
        equal(
            try XCTUnwrap(actual.state?[mtpLastHiddenStatesKey]),
            try XCTUnwrap(expected.state?[mtpLastHiddenStatesKey]))
        XCTAssertEqual(actual.state?[mtpSharedKVOffsetsKey], expected.state?[mtpSharedKVOffsetsKey])
        equalCache(a, b)
    }

    func testTextFallbacksKeepFullLogits() throws {
        for (bits, batch) in [(nil, 1), (Optional(2), 1), (Optional(4), 2)] {
            let model = Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: configuration()))
            prepareWeights(model, bits: bits)
            equal(
                model.nextTokenLogits(text(128, batch: batch), cache: nil, state: nil).logits,
                model(text(128, batch: batch), cache: nil, state: nil).logits)
        }
    }

    func testLocalCheckpointTextAndVisionParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_QWEN35_PREFILL_MODEL"],
            !path.isEmpty
        else {
            throw XCTSkip("set MLX_QWEN35_PREFILL_MODEL to validate a local Qwen3.5 checkpoint")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        for vision in [false, true] {
            try autoreleasepool {
                let model: any LanguageModel
                if vision {
                    model = Qwen35(
                        try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
                } else {
                    model = Qwen35Model(
                        try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
                }
                try loadWeights(
                    modelDirectory: directory, model: model,
                    perLayerQuantization: base.perLayerQuantization)
                for count in [128, 512] {
                    let a = try model.newCache(parameters: nil)
                    let b = try model.newCache(parameters: nil)
                    var fullState = LMOutput.State()
                    fullState[mtpEmitFlagKey] = true
                    let expected = withPreparedCache(a, lengths: text(count).sequenceLengths) {
                        model(text(count), cache: a, state: fullState)
                    }
                    let actual: LMOutput
                    if vision {
                        guard
                            case .logits(let output) = try model.prepare(
                                LMInput(text: text(count)), cache: b, state: nil, prefill: .init())
                        else { return XCTFail("Expected prefill logits") }
                        actual = output
                    } else {
                        actual = withPreparedCache(b, lengths: text(count).sequenceLengths) {
                            model.nextTokenLogits(text(count), cache: b, state: nil)
                        }
                    }
                    equal(actual.logits[0..., -1, 0...], expected.logits[0..., -1, 0...])
                    equalCache(a, b)
                    var stateA = expected.state
                    stateA?[mtpEmitFlagKey] = false
                    var stateB = actual.state
                    var token = argMax(expected.logits[0..., -1, 0...], axis: -1).item(Int.self)
                    for _ in 0 ..< 16 {
                        let input = LMInput.Text(tokens: MLXArray([token])[.newAxis])
                        let expected = model(input, cache: a, state: stateA)
                        let actual = model.nextTokenLogits(input, cache: b, state: stateB)
                        equal(actual.logits, expected.logits)
                        token = argMax(expected.logits[0..., -1, 0...], axis: -1).item(Int.self)
                        stateA = expected.state
                        stateB = actual.state
                    }
                    equalCache(a, b)
                }
            }
            Memory.clearCache()
        }
    }

    func testVisionPreparationPreservesImageStateAndContinuation() throws {
        let model = Qwen35(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: configuration()))
        prepareWeights(model)
        let fixtures = ContinuationAssertions(imageTokenId: 500, visionStartTokenId: 502)
        for withImage in [false, true] {
            let tokens =
                withImage
                ? concatenated([fixtures.imageRun(), fixtures.textTokens(123)], axis: 1)
                : fixtures.textTokens(128)
            let input = LMInput(
                text: .init(tokens: tokens), image: withImage ? fixtures.image() : nil)
            let a = try model.newCache(parameters: nil)
            let b = try model.newCache(parameters: nil)
            var fullState = LMOutput.State()
            fullState[mtpEmitFlagKey] = true
            guard
                case .logits(let expected) = try model.prepare(
                    input, cache: a, state: fullState, prefill: .init()),
                case .logits(let actual) = try model.prepare(
                    input, cache: b, state: nil, prefill: .init())
            else {
                return XCTFail("Expected prefill logits")
            }
            XCTAssertEqual(expected.logits.dim(1), 128)
            XCTAssertEqual(actual.logits.dim(1), 64)
            equal(actual.logits[0..., -1, 0...], expected.logits[0..., -1, 0...])
            equalCache(a, b)
            let ropeKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")
            equal(try XCTUnwrap(actual.state?[ropeKey]), try XCTUnwrap(expected.state?[ropeKey]))
            var stateA = expected.state
            stateA?[mtpEmitFlagKey] = false
            var stateB = actual.state
            for _ in 0 ..< 4 {
                let expected = model(text(1), cache: a, state: stateA)
                let actual = model(text(1), cache: b, state: stateB)
                equal(actual.logits, expected.logits)
                equalCache(a, b)
                stateA = expected.state
                stateB = actual.state
            }
        }
    }
}
