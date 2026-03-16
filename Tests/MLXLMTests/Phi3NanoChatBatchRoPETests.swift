// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
@preconcurrency @testable import MLXLMCommon
import XCTest

final class Phi3NanoChatBatchRoPETests: XCTestCase {

    private let prefillPrompts: [[Int32]] = [
        [11, 12, 13, 14, 15],
        [21, 22, 23],
    ]

    private let decodeTokens: [Int32] = [31, 32]

    func testPhi3BatchPrefillMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makePhi3Model(seed: 100)
        try assertPrefillMatchesSingle(model: model, prompts: prefillPrompts)
    }

    func testPhi3BatchDecodeMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makePhi3Model(seed: 101)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prefillPrompts,
            decodeTokens: decodeTokens
        )
    }

    func testNanoChatBatchPrefillMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makeNanoChatModel(seed: 200)
        try assertPrefillMatchesSingle(model: model, prompts: prefillPrompts)
    }

    func testNanoChatBatchDecodeMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makeNanoChatModel(seed: 201)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prefillPrompts,
            decodeTokens: decodeTokens
        )
    }

    func testPhi3IsBatchCompatibleForTextOnlyRequests() throws {
        try skipIfMetalUnavailable()

        let model = try makePhi3Model(seed: 300)
        assertSchedulerBatchCompatibility(model: model)
    }

    func testNanoChatIsBatchCompatibleForTextOnlyRequests() throws {
        try skipIfMetalUnavailable()

        let model = try makeNanoChatModel(seed: 301)
        assertSchedulerBatchCompatibility(model: model)
    }

    private func makePhi3Model(seed: UInt64) throws -> Phi3Model {
        let config: Phi3Configuration = try decodeConfig(
            """
            {
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "rms_norm_eps": 0.00001,
              "vocab_size": 64,
              "num_key_value_heads": 2,
              "rope_theta": 10000.0,
              "rope_traditional": false,
              "partial_rotary_factor": 1.0,
              "max_position_embeddings": 128,
              "original_max_position_embeddings": 128,
              "tie_word_embeddings": false
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Phi3Model(config)
            eval(model)
            return model
        }
    }

    private func makeNanoChatModel(seed: UInt64) throws -> NanoChatModel {
        let config: NanoChatConfiguration = try decodeConfig(
            """
            {
              "model_type": "nanochat",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "vocab_size": 64,
              "max_position_embeddings": 128,
              "intermediate_size": 32,
              "rope_theta": 10000.0,
              "rms_norm_eps": 0.00001,
              "logits_softcap": 15.0
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = NanoChatModel(config)
            eval(model)
            return model
        }
    }

    private func decodeConfig<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func assertSchedulerBatchCompatibility<M: LanguageModel>(
        model: M,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)

        XCTAssertTrue(
            InferenceScheduler.isBatchCompatible(
                input: input,
                parameters: parameters,
                cache: nil,
                model: model
            ),
            file: file,
            line: line
        )
    }

    private func assertPrefillMatchesSingle<M: LanguageModel & KVCacheDimensionProvider>(
        model: M,
        prompts: [[Int32]],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let singleResults = prompts.map { prompt in
            prefillSingle(model: model, prompt: prompt)
        }
        let batched = prefillBatch(model: model, prompts: prompts)

        for (index, prompt) in prompts.enumerated() {
            let pad = batched.leftPadding[index]
            let batchValid = batched.logits[index ..< (index + 1), pad..., 0...].asType(.float32)
            let single = singleResults[index].logits.asType(.float32)

            XCTAssertEqual(batchValid.shape, single.shape, file: file, line: line)
            let diff = maxAbsDifference(batchValid, single)
            XCTAssertLessThanOrEqual(
                diff,
                0.01,
                "Prefill logits diverged for prompt \(prompt)",
                file: file,
                line: line
            )
        }
    }

    private func assertDecodeMatchesSingle<M: LanguageModel & KVCacheDimensionProvider>(
        model: M,
        prompts: [[Int32]],
        decodeTokens: [Int32],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let singleResults = prompts.enumerated().map { index, prompt in
            var result = prefillSingle(model: model, prompt: prompt)
            let decodeInput = MLXArray([decodeTokens[index]])[.newAxis, .ellipsis]
            let decodeLogits = model.callAsFunction(decodeInput, cache: result.cache)
            materialize(arrays: [decodeLogits], cache: result.cache)
            result.logits = decodeLogits
            return result
        }

        var batched = prefillBatch(model: model, prompts: prompts)
        let batchedDecodeInput = MLXArray(decodeTokens, [decodeTokens.count, 1])
        let batchedDecodeLogits = model.callAsFunction(batchedDecodeInput, cache: batched.cache)
        materialize(arrays: [batchedDecodeLogits], cache: batched.cache)
        batched.logits = batchedDecodeLogits

        for index in prompts.indices {
            let batchRow = batched.logits[index ..< (index + 1), 0..., 0...].asType(.float32)
            let single = singleResults[index].logits.asType(.float32)

            XCTAssertEqual(batchRow.shape, single.shape, file: file, line: line)
            let diff = maxAbsDifference(batchRow, single)
            XCTAssertLessThanOrEqual(
                diff,
                0.01,
                "Decode logits diverged for prompt index \(index)",
                file: file,
                line: line
            )
        }
    }

    private func prefillSingle<M: LanguageModel & KVCacheDimensionProvider>(
        model: M,
        prompt: [Int32]
    ) -> (logits: MLXArray, cache: [KVCache]) {
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(prompt)[.newAxis, .ellipsis]
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache)
    }

    private func prefillBatch<M: LanguageModel & KVCacheDimensionProvider>(
        model: M,
        prompts: [[Int32]]
    ) -> (logits: MLXArray, cache: [KVCache], leftPadding: [Int]) {
        let maxLength = prompts.map(\.count).max() ?? 0
        let leftPadding = prompts.map { maxLength - $0.count }

        let flat = zip(prompts, leftPadding).flatMap { prompt, pad in
            Array(repeating: Int32(0), count: pad) + prompt
        }
        let input = MLXArray(flat, [prompts.count, maxLength])
        let cache: [KVCache] = model.kvHeads.map { _ in
            BatchKVCache(leftPadding: leftPadding)
        }
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache, leftPadding)
    }

    private func materialize(arrays: [MLXArray], cache: [KVCache]) {
        eval(arrays)
        let cacheState = cache.flatMap { $0.state }
        if !cacheState.isEmpty {
            eval(cacheState)
        }
    }

    private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }
}
