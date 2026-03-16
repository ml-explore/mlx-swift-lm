// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLLM
@preconcurrency @testable import MLXLMCommon
import XCTest

final class Gemma2FalconH1BatchMaskTests: XCTestCase {

    private let prefillPrompts: [[Int32]] = [
        [11, 12, 13, 14, 15],
        [21, 22, 23],
    ]

    private let decodeTokens: [Int32] = [31, 32]

    func testGemma2BatchPrefillMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makeGemma2Model(seed: 100)
        try assertPrefillMatchesSingle(model: model, prompts: prefillPrompts)
    }

    func testGemma2BatchDecodeMatchesSingle() throws {
        try skipIfMetalUnavailable()

        let model = try makeGemma2Model(seed: 101)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prefillPrompts,
            decodeTokens: decodeTokens
        )
    }

    func testGemma2IsBatchCompatibleForTextOnlyRequests() throws {
        try skipIfMetalUnavailable()

        let model = try makeGemma2Model(seed: 102)
        assertSchedulerBatchCompatibility(model: model)
    }

    func testFalconH1AttentionBatchDecodeMatchesMergedSingles() throws {
        try skipIfMetalUnavailable()

        let config = try makeFalconH1Configuration()
        let attention = withRandomState(MLXRandom.RandomState(seed: 200)) {
            let attention = FalconH1Attention(config)
            eval(attention)
            return attention
        }

        try assertFalconAttentionDecodeMatchesMergedSingles(
            attention: attention,
            hiddenSize: config.hiddenSize,
            promptLengths: prefillPrompts.map(\.count)
        )
    }

    func testFalconH1IsBatchIncompatibleForTextOnlyRequests() throws {
        try skipIfMetalUnavailable()

        let model = try makeFalconH1Model(seed: 201)
        assertSchedulerBatchIncompatibility(model: model)
    }

    private func makeGemma2Model(seed: UInt64) throws -> Gemma2Model {
        let config: Gemma2Configuration = try decodeConfig(
            """
            {
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "head_dim": 4,
              "rms_norm_eps": 0.00001,
              "vocab_size": 64,
              "num_key_value_heads": 2,
              "rope_theta": 10000.0,
              "rope_traditional": false,
              "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0,
              "query_pre_attn_scalar": 16.0
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Gemma2Model(config)
            eval(model)
            return model
        }
    }

    private func makeFalconH1Configuration() throws -> FalconH1Configuration {
        try decodeConfig(
            """
            {
              "model_type": "falcon_h1",
              "hidden_size": 16,
              "vocab_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 4,
              "max_position_embeddings": 128,
              "intermediate_size": 32,
              "mamba_d_ssm": 8,
              "mamba_d_state": 4,
              "mamba_n_heads": 2,
              "mamba_d_head": 4,
              "mamba_d_conv": 4,
              "rope_theta": 10000.0,
              "rope_traditional": false
            }
            """
        )
    }

    private func makeFalconH1Model(seed: UInt64) throws -> FalconH1Model {
        let config = try makeFalconH1Configuration()

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = FalconH1Model(config)
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

    private func assertSchedulerBatchIncompatibility<M: LanguageModel>(
        model: M,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)

        XCTAssertFalse(
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

    private func assertFalconAttentionDecodeMatchesMergedSingles(
        attention: FalconH1Attention,
        hiddenSize: Int,
        promptLengths: [Int],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let singleCaches: [KVCacheSimple] = promptLengths.enumerated().map { index, length in
            let cache = KVCacheSimple()
            let hidden = makeHiddenStates(length: length, hiddenSize: hiddenSize, base: Float(index + 1))
            let mask = createAttentionMask(h: hidden, cache: cache)
            let output = attention(hidden, mask: mask, cache: cache)
            materialize(arrays: [output], cache: [cache])
            return cache
        }

        let batchCache = BatchKVCache.merge(singleCaches.map { $0 as KVCache })
        let decodeInputs = promptLengths.indices.map { index in
            makeHiddenStates(length: 1, hiddenSize: hiddenSize, base: Float(100 + index))
        }

        let singleOutputs = decodeInputs.enumerated().map { index, decodeInput in
            let mask = createAttentionMask(h: decodeInput, cache: singleCaches[index])
            let output = attention(decodeInput, mask: mask, cache: singleCaches[index])
            materialize(arrays: [output], cache: [singleCaches[index]])
            return output
        }

        let batchedDecodeInput = concatenated(decodeInputs, axis: 0)
        let batchedMask = createAttentionMask(h: batchedDecodeInput, cache: batchCache)
        let batchedOutput = attention(batchedDecodeInput, mask: batchedMask, cache: batchCache)
        materialize(arrays: [batchedOutput], cache: [batchCache])

        for index in promptLengths.indices {
            let batchRow = batchedOutput[index ..< (index + 1), 0..., 0...].asType(.float32)
            let single = singleOutputs[index].asType(.float32)

            XCTAssertEqual(batchRow.shape, single.shape, file: file, line: line)
            let diff = maxAbsDifference(batchRow, single)
            XCTAssertLessThanOrEqual(
                diff,
                0.01,
                "FalconH1 attention decode diverged for prompt index \(index)",
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

    private func makeHiddenStates(length: Int, hiddenSize: Int, base: Float) -> MLXArray {
        let values = (0 ..< (length * hiddenSize)).map { index in
            base + Float(index) / 100.0
        }
        return MLXArray(values, [1, length, hiddenSize])
    }

    private func materialize(arrays: [MLXArray], cache: [KVCache]) {
        if !arrays.isEmpty {
            eval(arrays)
        }
        let cacheState = cache.flatMap { $0.state }
        if !cacheState.isEmpty {
            eval(cacheState)
        }
    }

    private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }
}
