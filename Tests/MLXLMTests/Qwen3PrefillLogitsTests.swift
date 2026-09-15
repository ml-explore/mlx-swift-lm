// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class Qwen3PrefillLogitsTests: XCTestCase {
    private func makeModel(tied: Bool = true, bits: Int? = 4, dtype: DType = .bfloat16)
        throws -> Qwen3Model
    {
        let configuration = """
            {
                "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 32,
                "rms_norm_eps": 0.000001, "vocab_size": 256,
                "tie_word_embeddings": \(tied)
            }
            """
        MLXRandom.seed(17)
        let model = Qwen3Model(
            try JSONDecoder().decode(Qwen3Configuration.self, from: Data(configuration.utf8)))
        if let bits {
            quantize(model: model, groupSize: 32, bits: bits)
        }
        model.apply { $0.dtype == .float32 ? $0.asType(dtype) : $0 }
        eval(model)
        return model
    }

    private func input(_ count: Int, batch: Int = 1) -> LMInput.Text {
        .init(
            tokens: MLXArray((0 ..< count * batch).map { ($0 * 37 + 11) % 256 })
                .reshaped(batch, count))
    }

    private func assertEqual(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertTrue(
            allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self),
            "Expected exact equality", file: file, line: line)
    }

    private func assertCacheEqual(
        _ actual: [KVCache], _ expected: [KVCache],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(a.offset, b.offset, file: file, line: line)
            XCTAssertEqual(a.state.count, b.state.count, file: file, line: line)
            for (x, y) in zip(a.state, b.state) {
                assertEqual(x, y, file: file, line: line)
            }
        }
    }

    func testQuantizedProjectionPreservesExactLogitsAndCacheAtBoundaries() throws {
        let lengths = [
            1, 8, 9, 31, 32, 33, 63, 64, 65, 95, 96, 97,
            127, 128, 129, 255, 256, 257, 511, 512, 513,
        ]
        for dtype: DType in [.float32, .float16, .bfloat16] {
            for bits in [4, 8] {
                let model = try makeModel(bits: bits, dtype: dtype)
                let languageModel: any LanguageModel = model
                for length in lengths {
                    try autoreleasepool {
                        let expectedCache = try model.newCache(parameters: nil)
                        let actualCache = try model.newCache(parameters: nil)
                        let text = input(length)
                        let expected = languageModel(text, cache: expectedCache, state: nil)
                        let actual = languageModel.nextTokenLogits(
                            text, cache: actualCache, state: nil)
                        XCTAssertEqual(expected.logits.shape, [1, length, 256])
                        if length >= 96 {
                            XCTAssertLessThanOrEqual(actual.logits.dim(1), 95)
                        }
                        assertEqual(actual.logits[0..., -1, 0...], expected.logits[0..., -1, 0...])
                        assertCacheEqual(actualCache, expectedCache)
                    }
                }
            }
        }
    }

    func testUnquantizedUntiedAndBatchedForwardsKeepAllRows() throws {
        for (tied, bits, batch) in [
            (true, nil, 1), (false, nil, 1),
            (false, Optional(4), 1), (true, Optional(2), 1), (true, Optional(4), 2),
        ] {
            let model = try makeModel(tied: tied, bits: bits)
            let text = input(128, batch: batch)
            let expected = model(text, cache: nil, state: nil)
            let actual = model.nextTokenLogits(text, cache: nil, state: nil)
            assertEqual(actual.logits, expected.logits)
            XCTAssertEqual(model(text.tokens, cache: nil).shape, [batch, 128, 256])
        }
    }

    func testWarmSuffixAndContinuationPreserveEntireCache() throws {
        let model = try makeModel()
        for parameters in [
            GenerateParameters(), GenerateParameters(maxKVSize: 128),
            GenerateParameters(kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0),
        ] {
            let prefix = try model.newCache(parameters: parameters)
            eval(model(input(97), cache: prefix, state: nil).logits, prefix)
            let expectedCache = prefix.map { $0.copy() }
            let actualCache = prefix.map { $0.copy() }
            for length in [129, 1, 1, 96, 1] {
                let text = input(length)
                let expected = model(text, cache: expectedCache, state: nil)
                let actual = model.nextTokenLogits(text, cache: actualCache, state: nil)
                assertEqual(actual.logits[0..., -1, 0...], expected.logits[0..., -1, 0...])
                assertCacheEqual(actualCache, expectedCache)
            }
        }
    }

    func testOrdinaryIteratorUsesOptimizedEntryPointAndSpeculationKeepsFullForward() throws {
        let model = try makeModel()
        let recording = RecordingQwen(model)
        let parameters = GenerateParameters(maxTokens: 6, temperature: 0)
        var iterator = try TokenIterator(
            input: LMInput(tokens: input(128).tokens.squeezed(axis: 0)),
            model: recording, parameters: parameters)
        while iterator.next() != nil {}
        XCTAssertGreaterThan(recording.nextTokenCalls, 0)
        XCTAssertEqual(recording.fullForwardCalls, 0)

        recording.nextTokenCalls = 0
        var speculative = try SpeculativeTokenIterator(
            input: LMInput(tokens: input(128).tokens.squeezed(axis: 0)),
            mainModel: recording, draftModel: model, parameters: parameters, numDraftTokens: 2)
        while speculative.next() != nil {}
        XCTAssertEqual(recording.nextTokenCalls, 0)
        XCTAssertGreaterThan(recording.fullForwardCalls, 0)
    }

    func testLocalCheckpointPreservesLogitsCacheAndContinuation() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_QWEN3_PREFILL_MODEL"],
            !path.isEmpty
        else {
            throw XCTSkip("set MLX_QWEN3_PREFILL_MODEL to validate a local Qwen3 checkpoint")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = Qwen3Model(try JSONDecoder().decode(Qwen3Configuration.self, from: data))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)

        for length in [96, 128, 512, 513] {
            try autoreleasepool {
                let prefix = try model.newCache(parameters: nil)
                eval(model(input(97), cache: prefix, state: nil).logits, prefix)
                let expectedCache = prefix.map { $0.copy() }
                let actualCache = prefix.map { $0.copy() }
                var text = input(length)
                for step in 0 ... 16 {
                    let expected = model(text, cache: expectedCache, state: nil).logits[
                        0..., -1, 0...]
                    let actual = model.nextTokenLogits(text, cache: actualCache, state: nil).logits[
                        0..., -1, 0...]
                    assertEqual(actual, expected)
                    if step == 0 || step == 16 {
                        assertCacheEqual(actualCache, expectedCache)
                    }
                    let expectedToken = argMax(expected, axis: -1).item(Int.self)
                    XCTAssertEqual(argMax(actual, axis: -1).item(Int.self), expectedToken)
                    text = .init(tokens: MLXArray([expectedToken])[.newAxis])
                }
            }
        }
    }

    func testDefaultEntryPointPreservesFullLogitsAndState() throws {
        let model: any LanguageModel = StatefulFullLogitModel()
        var state = LMOutput.State()
        state[StatefulFullLogitModel.key] = 7
        let text = input(128)
        let output = model.nextTokenLogits(text, cache: nil, state: state)
        XCTAssertEqual(output.logits.shape, [1, 128, 1])
        XCTAssertEqual(output.state?[StatefulFullLogitModel.key], 8)
        assertEqual(output.logits, text.tokens.asType(.float32)[.ellipsis, .newAxis])
    }
}

private final class RecordingQwen: Module, LLMModel {
    var loraLayers: [Module] { wrapped.loraLayers }
    let wrapped: Qwen3Model
    var nextTokenCalls = 0
    var fullForwardCalls = 0
    var vocabularySize: Int { wrapped.vocabularySize }

    init(_ wrapped: Qwen3Model) { self.wrapped = wrapped }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try wrapped.newCache(parameters: parameters)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        fullForwardCalls += 1
        return wrapped(input, cache: cache, state: state)
    }

    func nextTokenLogits(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        nextTokenCalls += 1
        return wrapped.nextTokenLogits(input, cache: cache, state: state)
    }
}

private final class StatefulFullLogitModel: Module, LLMModel {
    var loraLayers: [Module] { [] }
    static let key = LMOutput.Key<Int>("step")
    let vocabularySize = 1

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        var next = state ?? .init()
        next[Self.key] = (state?[Self.key] ?? 0) + 1
        return .init(logits: input.tokens.asType(.float32)[.ellipsis, .newAxis], state: next)
    }
}
