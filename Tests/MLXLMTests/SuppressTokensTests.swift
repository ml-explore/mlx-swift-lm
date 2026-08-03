// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Minimal `LanguageModel` whose logits always peak at `peakToken` with
/// `runnerUpToken` in second place. Conforms to `SuppressedTokensProviding`
/// so tests can verify that suppressed tokens are never sampled.
private final class SuppressingMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    SuppressedTokensProviding
{
    var kvHeads: [Int] { [1] }
    var suppressedTokenIds: Set<Int>

    let vocabularySize = 16
    let peakToken: Int
    let runnerUpToken: Int

    init(peakToken: Int, runnerUpToken: Int, suppressedTokenIds: Set<Int>) {
        self.peakToken = peakToken
        self.runnerUpToken = runnerUpToken
        self.suppressedTokenIds = suppressedTokenIds
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        var data = [Float](repeating: 0, count: positions * vocabularySize)
        for position in 0 ..< positions {
            data[position * vocabularySize + peakToken] = 10
            data[position * vocabularySize + runnerUpToken] = 5
        }
        return MLXArray(data, [1, positions, vocabularySize])
    }
}

/// Like `SuppressingMockModel`, but does NOT conform to
/// `SuppressedTokensProviding` — used to verify that `suppress_tokens` from
/// `generation_config.json` is honored for models without the conformance.
private final class PlainMockModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }

    let vocabularySize = 16
    let peakToken: Int
    let runnerUpToken: Int

    init(peakToken: Int, runnerUpToken: Int) {
        self.peakToken = peakToken
        self.runnerUpToken = runnerUpToken
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        var data = [Float](repeating: 0, count: positions * vocabularySize)
        for position in 0 ..< positions {
            data[position * vocabularySize + peakToken] = 10
            data[position * vocabularySize + runnerUpToken] = 5
        }
        return MLXArray(data, [1, positions, vocabularySize])
    }
}

public class SuppressTokensTests: XCTestCase {

    // MARK: - generation_config.json parsing

    func testGenerationConfigDecodesSuppressTokensArray() throws {
        // Mirrors mlx-community/gemma-4-12B-it-4bit's generation_config.json
        let json = """
            {"eos_token_id": [1, 106], "suppress_tokens": [258883, 258882]}
            """
        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: Data(json.utf8))
        XCTAssertEqual(config.suppressTokens?.values, [258883, 258882])
    }

    func testGenerationConfigDecodesSuppressTokensSingleInt() throws {
        let json = """
            {"suppress_tokens": 258882}
            """
        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: Data(json.utf8))
        XCTAssertEqual(config.suppressTokens?.values, [258882])
    }

    func testGenerationConfigMissingSuppressTokensIsNil() throws {
        let json = """
            {"eos_token_id": 106}
            """
        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: Data(json.utf8))
        XCTAssertNil(config.suppressTokens)
    }

    func testGenerationConfigNullSuppressTokensIsNil() throws {
        let json = """
            {"suppress_tokens": null}
            """
        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: Data(json.utf8))
        XCTAssertNil(config.suppressTokens)
    }

    // MARK: - SuppressTokensProcessor

    func testSuppressTokensProcessorMasksOnlyGivenIds() throws {
        let processor = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [1, 3]))
        let logits = MLXArray([0.5, 5.0, 1.0, 9.0, 2.0] as [Float])[.newAxis, .ellipsis]

        let processed = processor.process(logits: logits)

        XCTAssertEqual(processed[0, 1].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 3].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 0].item(Float.self), 0.5)
        XCTAssertEqual(processed[0, 2].item(Float.self), 1.0)
        XCTAssertEqual(processed[0, 4].item(Float.self), 2.0)
    }

    func testSuppressTokensProcessorEmptySetIsNil() {
        XCTAssertNil(SuppressTokensProcessor(tokenIds: []))
    }

    func testSuppressedTokenIsNeverArgMax() throws {
        let processor = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [3]))
        let logits = MLXArray([0.5, 5.0, 1.0, 9.0, 2.0] as [Float])[.newAxis, .ellipsis]

        let token = ArgMaxSampler().sample(logits: processor.process(logits: logits))

        XCTAssertEqual(token.item(Int.self), 1)
    }

    func testSuppressTokensProcessorIgnoresOutOfRangeIds() throws {
        // IDs beyond the logits vocabulary (e.g. a tiny test config
        // inheriting 255k-range boi/boa defaults) must be dropped, not
        // passed to putAlong.
        let processor = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [1, 258882, 258883]))
        let logits = MLXArray([0.5, 5.0, 1.0, 9.0, 2.0] as [Float])[.newAxis, .ellipsis]

        let processed = processor.process(logits: logits)

        XCTAssertEqual(processed[0, 1].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 3].item(Float.self), 9.0)
    }

    func testSuppressTokensProcessorAllOutOfRangeIsNoOp() throws {
        let processor = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [258882, 258883]))
        let logits = MLXArray([0.5, 5.0, 1.0] as [Float])[.newAxis, .ellipsis]

        let processed = processor.process(logits: logits)

        XCTAssertEqual(processed[0, 1].item(Float.self), 5.0)
    }

    func testSuppressTokensProcessorIgnoresNegativeIds() {
        // Negative IDs are dropped at init; all-negative means nothing to
        // suppress.
        XCTAssertNil(SuppressTokensProcessor(tokenIds: [-1, -106]))
    }

    // MARK: - ChainedLogitProcessor

    func testChainedLogitProcessorAppliesAllProcessors() throws {
        let first = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [1]))
        let second = try XCTUnwrap(SuppressTokensProcessor(tokenIds: [3]))
        let chained = ChainedLogitProcessor(processors: [first, second])
        let logits = MLXArray([0.5, 5.0, 1.0, 9.0, 2.0] as [Float])[.newAxis, .ellipsis]

        let processed = chained.process(logits: logits)

        XCTAssertEqual(processed[0, 1].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 3].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 4].item(Float.self), 2.0)
    }

    // MARK: - TokenIterator integration

    private func generate(model: SuppressingMockModel, parameters: GenerateParameters) throws
        -> [Int]
    {
        let input = LMInput(tokens: MLXArray([1, 2, 3]))
        var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }
        return tokens
    }

    func testTokenIteratorNeverSamplesSuppressedToken() throws {
        let model = SuppressingMockModel(
            peakToken: 5, runnerUpToken: 7, suppressedTokenIds: [5])
        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)

        let tokens = try generate(model: model, parameters: parameters)

        XCTAssertEqual(tokens, [7, 7, 7, 7, 7])
    }

    func testTokenIteratorWithoutSuppressionSamplesPeakToken() throws {
        // Control: with nothing suppressed, the argmax token is the peak.
        let model = SuppressingMockModel(
            peakToken: 5, runnerUpToken: 7, suppressedTokenIds: [])
        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)

        let tokens = try generate(model: model, parameters: parameters)

        XCTAssertEqual(tokens, [5, 5, 5, 5, 5])
    }

    func testTokenIteratorSuppressionChainsWithPenaltyProcessor() throws {
        // With a repetition penalty active, suppression is chained after the
        // penalty processor and must still mask the suppressed token.
        let model = SuppressingMockModel(
            peakToken: 5, runnerUpToken: 7, suppressedTokenIds: [5])
        let parameters = GenerateParameters(
            maxTokens: 5, temperature: 0, repetitionPenalty: 1.5, repetitionContextSize: 4)

        let tokens = try generate(model: model, parameters: parameters)

        XCTAssertFalse(tokens.contains(5))
        XCTAssertEqual(tokens, [7, 7, 7, 7, 7])
    }

    func testTokenIteratorWithOutOfVocabularySuppressedIdsDoesNotCrash() throws {
        // Regression: a model whose suppressed set mixes in-range and
        // out-of-vocabulary IDs (tiny vocab + inherited 255k-range
        // placeholder defaults) must generate normally, suppressing only
        // the in-range IDs.
        let model = SuppressingMockModel(
            peakToken: 5, runnerUpToken: 7, suppressedTokenIds: [5, 258882, 258883])
        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)

        let tokens = try generate(model: model, parameters: parameters)

        XCTAssertEqual(tokens, [7, 7, 7, 7, 7])
    }

    // MARK: - generation_config.json suppression without protocol conformance

    func testGenerationConfigSuppressionAppliesToNonConformingModel() throws {
        // suppress_tokens from generation_config.json must be honored even
        // when the model does not adopt SuppressedTokensProviding.
        let model = PlainMockModel(peakToken: 5, runnerUpToken: 7)
        let generationConfig = GenerationConfigFile(suppressTokens: IntOrIntArray([5]))

        mergeGenerationConfigSuppressedTokens(generationConfig, into: model)

        let input = LMInput(tokens: MLXArray([1, 2, 3]))
        var iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: 5, temperature: 0))
        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        XCTAssertEqual(tokens, [7, 7, 7, 7, 7])
    }

    // MARK: - Gemma4Unified config-derived suppressed tokens

    func testGemma4UnifiedSeedsSuppressedTokenIdsFromConfig() throws {
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "gemma4_unified",
                  "vocab_size": 32,
                  "image_token_id": 31,
                  "audio_token_id": 30,
                  "video_token_id": 29,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 8,
                    "global_head_dim": 8,
                    "vocab_size": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "attention_k_eq_v": true,
                    "use_double_wide_mlp": false,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": true
                  },
                  "vision_config": null,
                  "audio_config": null
                }
                """.utf8))

        let model = Gemma4Unified(config)

        // image / audio / video placeholder tokens from the config, plus the
        // boi / eoi / boa defaults for gemma4_unified.
        XCTAssertTrue(model.suppressedTokenIds.isSuperset(of: [31, 30, 29]))
        XCTAssertTrue(model.suppressedTokenIds.contains(config.boiTokenId))
        XCTAssertTrue(model.suppressedTokenIds.contains(config.boaTokenId))
        if let eoiTokenId = config.eoiTokenId {
            XCTAssertTrue(model.suppressedTokenIds.contains(eoiTokenId))
        }
    }
}
