// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class StopStringTests: XCTestCase {

    func testGenerationConfigDecodesStopStringsArray() throws {
        let data = Data(
            """
            {
              "eos_token_id": [1, 2],
              "stop_strings": ["<|im_end|>", "<end_of_turn>"]
            }
            """.utf8)

        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: data)

        XCTAssertEqual(config.eosTokenIds?.values, [1, 2])
        XCTAssertEqual(config.stopStrings, ["<|im_end|>", "<end_of_turn>"])
    }

    func testGenerationConfigDecodesSingleStopStringAndStopAlias() throws {
        let data = Data(
            """
            {
              "stop_strings": "<turn|>",
              "stop": ["<fallback>"]
            }
            """.utf8)

        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: data)

        XCTAssertEqual(config.stopStrings, ["<turn|>", "<fallback>"])
    }

    func testModelConfigurationResolutionPreservesStopStrings() {
        let config = ModelConfiguration(
            id: "org/model",
            extraEOSTokens: ["<extra>"],
            stopStrings: ["<stop>"],
            eosTokenIds: [7]
        )

        let resolved = config.resolved(
            modelDirectory: URL(filePath: "/tmp/model"),
            tokenizerDirectory: URL(filePath: "/tmp/tokenizer")
        )

        XCTAssertEqual(resolved.extraEOSTokens, ["<extra>"])
        XCTAssertEqual(resolved.stopStrings, ["<stop>"])
        XCTAssertEqual(resolved.eosTokenIds, [7])
    }

    func testStopStringSplitAcrossChunksStopsAndIsNotEmitted() {
        var filter = StopStringFilter(stopStrings: ["<stop>"])

        let first = filter.process("hel")
        let second = filter.process("lo<")
        let third = filter.process("stop")
        let fourth = filter.process(">hidden")
        let fifth = filter.process("tail")

        XCTAssertEqual(first.text, "hel")
        XCTAssertFalse(first.stopped)
        XCTAssertEqual(second.text, "lo")
        XCTAssertFalse(second.stopped)
        XCTAssertNil(third.text)
        XCTAssertFalse(third.stopped)
        XCTAssertNil(fourth.text)
        XCTAssertTrue(fourth.stopped)
        XCTAssertNil(fifth.text)
        XCTAssertTrue(fifth.stopped)
    }

    func testStopStringFilterFlushesBufferedSuffixWhenNoStopArrives() {
        var filter = StopStringFilter(stopStrings: ["<stop>"])

        let first = filter.process("hello<")
        let second = filter.process("world")

        XCTAssertEqual(first.text, "hello")
        XCTAssertFalse(first.stopped)
        XCTAssertEqual(second.text, "<world")
        XCTAssertFalse(second.stopped)
        XCTAssertNil(filter.finish())
    }

    func testEmptyStopStringsAreIgnored() {
        var filter = StopStringFilter(stopStrings: [""])

        let result = filter.process("visible")

        XCTAssertEqual(result.text, "visible")
        XCTAssertFalse(result.stopped)
    }

    func testKnownRegistryEntriesExposeFamilyStopDefaults() {
        assertStops(LLMRegistry.gemma3_1B_qat_4bit, "<end_of_turn>")
        assertStops(LLMRegistry.gemma3n_E4B_it_lm_4bit, "<end_of_turn>")
        assertStops(LLMRegistry.gemma4_e2b_it_4bit, "<turn|>")
        assertStops(LLMRegistry.qwen3_0_6b_4bit, "<|im_end|>")
        assertStops(LLMRegistry.qwen3_5_2b_4bit, "<|im_end|>")
        assertStops(LLMRegistry.phi3_5_4bit, "<|end|>")
        assertStops(LLMRegistry.phi3_5MoE, "<|end|>")
        assertStops(LLMRegistry.llama3_8B_4bit, "<|eot_id|>")
        assertStops(LLMRegistry.llama3_2_3B_4bit, "<|eot_id|>")

        assertStops(VLMRegistry.gemma3_4B_qat_4bit, "<end_of_turn>")
        assertStops(VLMRegistry.gemma4_E2B_it_4bit, "<end_of_turn>")
        assertStops(VLMRegistry.qwen2VL2BInstruct4Bit, "<|im_end|>")
        assertStops(VLMRegistry.qwen3VL4BInstruct8Bit, "<|im_end|>")
        assertStops(VLMRegistry.qwen3_5_27B_4bit, "<|im_end|>")
    }

    private func assertStops(
        _ configuration: ModelConfiguration,
        _ token: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(configuration.extraEOSTokens.contains(token), file: file, line: line)
        XCTAssertTrue(configuration.stopStrings.contains(token), file: file, line: line)
    }
}
