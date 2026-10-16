// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class Qwen3PrefillLogitsTests: XCTestCase {
    private func makeModel(
        tied: Bool = true, bits: Int? = 4, dtype: DType = .bfloat16,
        heads: Int = 2, kvHeads: Int = 1, headDim: Int = 32
    ) throws -> Qwen3Model {
        let configuration = """
            {
                "hidden_size": \(heads * headDim), "num_hidden_layers": 2,
                "intermediate_size": \(2 * heads * headDim), "num_attention_heads": \(heads),
                "num_key_value_heads": \(kvHeads), "head_dim": \(headDim),
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
            127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 1025,
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
                        if length >= 97 && length <= 1024 {
                            XCTAssertLessThanOrEqual(actual.logits.dim(1), 96)
                        } else if length == 1025 {
                            XCTAssertEqual(actual.logits.dim(1), length)
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

    func testEagerQuantizedCacheMatchesDynamicConversion() throws {
        let model = try makeModel(bits: nil, dtype: .float32)
        let parameters = GenerateParameters(kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)
        let prompt = input(97).tokens

        // Eager: the cache is quantized before the first token lands, chunk by chunk.
        let eager = try model.newCache(parameters: parameters)
        XCTAssertTrue(eager.allSatisfy { $0 is QuantizedKVCache })
        for range in [0 ..< 64, 64 ..< 97] {
            _ = model(.init(tokens: prompt[0..., range]), cache: eager, state: nil)
        }

        // Dynamic: the whole prompt at full precision, then one conversion.
        var dynamic = try model.newCache(parameters: nil)
        XCTAssertTrue(dynamic.allSatisfy { $0 is KVCacheSimple })
        _ = model(.init(tokens: prompt), cache: dynamic, state: nil)
        maybeQuantizeKVCache(cache: &dynamic, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)
        XCTAssertTrue(dynamic.allSatisfy { $0 is QuantizedKVCache })

        // Layer 0 keys and values depend only on the embeddings, so both routes store the
        // same quantized tokens. Deeper layers differ only by the second chunk attending to
        // 8-bit instead of full-precision history.
        assertCacheEqual([eager[0]], [dynamic[0]])
        let next = input(98).tokens[0..., 97...]
        let eagerLogits = model.nextTokenLogits(.init(tokens: next), cache: eager, state: nil)
        let dynamicLogits = model.nextTokenLogits(
            .init(tokens: next), cache: dynamic, state: nil)
        XCTAssertTrue(
            allClose(eagerLogits.logits, dynamicLogits.logits, rtol: 1e-2, atol: 1e-2)
                .item(Bool.self))
    }

    func testEagerFourBitCachePrefillsThroughFusedAttention() throws {
        // Head dim 64 with 64-token chunks: every prefill chunk lands in a 4-bit cache and
        // attends through the dequantized fused route, as a real model would.
        let model = try makeModel(bits: nil, dtype: .bfloat16, heads: 4, kvHeads: 2, headDim: 64)
        let parameters = GenerateParameters(kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        let prompt = input(128).tokens
        XCTAssertTrue(
            usesDequantizedPrefillAttention(
                queries: MLXArray.zeros([1, 4, 64, 64], dtype: .bfloat16), valueHeadDim: 64))

        let eager = try model.newCache(parameters: parameters)
        XCTAssertTrue(eager.allSatisfy { $0 is QuantizedKVCache })
        for range in [0 ..< 64, 64 ..< 128] {
            _ = model(.init(tokens: prompt[0..., range]), cache: eager, state: nil)
        }
        XCTAssertTrue(eager.allSatisfy { $0 is QuantizedKVCache && $0.offset == 128 })

        var dynamic = try model.newCache(parameters: nil)
        _ = model(.init(tokens: prompt), cache: dynamic, state: nil)
        maybeQuantizeKVCache(cache: &dynamic, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        XCTAssertTrue(dynamic.allSatisfy { $0 is QuantizedKVCache && $0.offset == 128 })

        let next = input(129).tokens[0..., 128...]
        let reference = try model.newCache(parameters: nil)
        _ = model(.init(tokens: prompt), cache: reference, state: nil)
        func logits(_ cache: [KVCache]) -> MLXArray {
            model.nextTokenLogits(.init(tokens: next), cache: cache, state: nil).logits
                .asType(.float32)
        }
        func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            sqrt((mean(square(actual - expected)) / mean(square(expected))).item(Float.self))
        }
        let (eagerLogits, dynamicLogits, referenceLogits) = (
            logits(eager), logits(dynamic), logits(reference)
        )
        XCTAssertTrue(all(isFinite(eagerLogits)).item(Bool.self))
        XCTAssertTrue(eager.allSatisfy { $0.offset == 129 })

        // The second chunk attended over 4-bit history instead of bfloat16, so eager sits
        // in the same 4-bit noise band as dynamic conversion, measured against the
        // never-quantized reference (about 0.13 vs 0.11 on this tiny random model).
        let dynamicError = relativeError(dynamicLogits, referenceLogits)
        XCTAssertLessThan(relativeError(eagerLogits, referenceLogits), 1.5 * dynamicError)
        XCTAssertLessThan(relativeError(eagerLogits, dynamicLogits), 1.5 * dynamicError)
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
