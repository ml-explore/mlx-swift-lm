// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SparkTests: XCTestCase {
    private func configuration() throws -> SparkConfiguration {
        let json = """
            {
              "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
              "vocab_size": 32, "sliding_window": 4,
              "layer_types": ["sliding_attention", "full_attention"],
              "headwise_attn_output_gate": true, "tie_word_embeddings": true,
              "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
                "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.5}
              }
            }
            """
        return try JSONDecoder().decode(SparkConfiguration.self, from: Data(json.utf8))
    }

    func testCacheMatchesFullForwardAcrossWindow() throws {
        let model = SparkModel(try configuration())
        let tokens = MLXArray((0 ..< 12).map { Int32($0 % 32) }).reshaped(1, 12)
        let reference = model(tokens, cache: nil)
        let cache = try model.newCache()
        XCTAssertTrue(cache[0] is RotatingKVCache)
        XCTAssertFalse(cache[1] is RotatingKVCache)
        var results: [MLXArray] = []
        for i in 0 ..< 12 {
            results.append(model(tokens[0..., i ..< i + 1], cache: cache))
        }
        let incremental = concatenated(results, axis: 1)
        let error = MLX.max(MLX.abs(reference - incremental)).item(Float.self)
        XCTAssertLessThan(error, 0.0001)
    }

    func testChunkedCacheWithSmallerLimit() throws {
        for keep in 0 ... 1 {
            try checkChunkedCache(keep: keep)
        }
    }

    private func checkChunkedCache(keep: Int) throws {
        let model = SparkModel(try configuration())
        let parameters = GenerateParameters(
            kvCache: KVCacheConfiguration(
                capacity: try .init(maxTokens: 2, preservedPrefixTokens: keep)))
        let tokens = MLXArray((0 ..< 10).map { Int32($0) }).reshaped(1, 10)
        let singleCache = try model.newCache(parameters: parameters)
        let chunkCache = try model.newCache(parameters: parameters)
        var single: [MLXArray] = []
        var chunked: [MLXArray] = []
        for i in 0 ..< 10 {
            single.append(model(tokens[0..., i ..< i + 1], cache: singleCache))
        }
        for i in stride(from: 0, to: 10, by: 2) {
            chunked.append(model(tokens[0..., i ..< i + 2], cache: chunkCache))
        }
        let error = MLX.max(MLX.abs(concatenated(single, axis: 1) - concatenated(chunked, axis: 1)))
            .item(Float.self)
        XCTAssertLessThan(error, 0.0001)
    }

    func testPrefillPreparesAndFinalizesCaches() throws {
        let model = SparkModel(try configuration())
        let caches = [RecordingSparkCache(), RecordingSparkCache()]
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5])
        let input = LMInput(text: .init(tokens: tokens, mask: MLXArray([Int32(1), 1, 1, 1, 1])))
        _ = try model.prepare(input, cache: caches, prefill: .init(stepSize: 2))
        for cache in caches {
            XCTAssertEqual(cache.lengths, [5])
            XCTAssertTrue(cache.finalized)
            XCTAssertEqual(cache.offset, 4)
        }
    }

    func testSparkToolCalls() throws {
        let parser = SparkToolCallParser()
        let empty = try XCTUnwrap(parser.parse(content: "<tool_call>clock</tool_call>", tools: nil))
        XCTAssertEqual(empty.function.name, "clock")
        XCTAssertTrue(empty.function.arguments.isEmpty)
        let call = try XCTUnwrap(
            parser.parse(
                content:
                    "<tool_call>weather<arg_key>city</arg_key><arg_value>Dubai</arg_value></tool_call>",
                tools: nil))
        XCTAssertEqual(call.function.name, "weather")
        XCTAssertEqual(call.function.arguments["city"], .string("Dubai"))
    }

    func testToolArgumentsAndMalformedCalls() throws {
        let parser = SparkToolCallParser()
        let properties: [String: any Sendable] = [
            "text": ["type": "string"], "count": ["type": "integer"],
        ]
        let parameters: [String: any Sendable] = ["properties": properties]
        let function: [String: any Sendable] = ["name": "run", "parameters": parameters]
        let tools: [[String: any Sendable]] = [["function": function]]
        let call = try XCTUnwrap(
            parser.parse(
                content:
                    "<tool_call>run<arg_key>text</arg_key><arg_value>  1  </arg_value><arg_key>count</arg_key><arg_value>1</arg_value></tool_call>",
                tools: tools))
        XCTAssertEqual(call.function.arguments["text"], .string("  1  "))
        XCTAssertEqual(call.function.arguments["count"], .int(1))
        XCTAssertNil(
            parser.parse(
                content: "<tool_call>run<arg_key>x</arg_key><arg_value>unfinished</tool_call>",
                tools: nil))
        XCTAssertNil(
            parser.parse(
                content:
                    "<tool_call>run<arg_key>x</arg_key><arg_value>1</arg_value>garbage</tool_call>",
                tools: nil))
        XCTAssertNil(
            parser.parse(
                content:
                    "<tool_call>run<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>",
                tools: nil))
    }

    func testRegistryAndToolFormat() async throws {
        let registered = await LLMTypeRegistry.shared.contains("spark2_5")
        XCTAssertTrue(registered)
        XCTAssertEqual(SparkModel(try configuration()).toolCallFormat, .spark)
    }
}

private final class RecordingSparkCache: BaseKVCache {
    private let inner = KVCacheSimple()
    var lengths: [Int]?
    var finalized = false

    override func prepare(lengths: [Int]?) { self.lengths = lengths }
    override func finalize() { finalized = true }
    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = inner.update(keys: keys, values: values)
        offset = inner.offset
        return result
    }
    override func innerState() -> [MLXArray] { inner.state }
}
