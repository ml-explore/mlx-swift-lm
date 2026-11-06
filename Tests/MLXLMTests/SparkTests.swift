// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
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

    // Spark-MLX-LLM 079a6a3, with deterministic float32 weights.
    func testLogitsMatchSparkReferenceAcrossWindow() throws {
        let model = SparkModel(try configuration())
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { name, value in
                let values = (0 ..< value.size).map { Float($0 % 17 - 8) * 0.05 }
                return (name, MLXArray(values, value.shape))
            })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7]).reshaped(1, 7)
        let expected: [Float] = [
            0.3873586, 0.3319290, 0.0817104, -0.0460823, -0.1685149, -0.3181255,
            -0.3532947, -0.3234146, -0.1604977, -0.0786889, -0.0371127, -0.0272825,
            -0.0413007, 0.0452750, 0.1627202, 0.2397714, 0.3055498, 0.3873586,
            0.3319290, 0.0817104, -0.0460823, -0.1685149, -0.3181255, -0.3532947,
            -0.3234146, -0.1604977, -0.0786889, -0.0371127, -0.0272825, -0.0413007,
            0.0452750, 0.1627202,
        ]
        let logits = model(tokens, cache: nil)[0, -1, 0...]
        XCTAssertLessThan(MLX.max(MLX.abs(logits - MLXArray(expected))).item(Float.self), 1e-5)
    }

    func testSanitizeConvertedCheckpointNames() throws {
        let model = SparkModel(try configuration())
        let tensor = MLXArray([Float(1)])
        let quantization = BaseConfiguration.Quantization(groupSize: 32, bits: 8)
        let cleaned = try model.prepareCheckpoint(
            ModelCheckpoint(
                weights: [
                    "model.embed_tokens.weight": tensor,
                    "model.embed_tokens.scales": tensor,
                    "model.layers.0.self_attn.rotary_emb.inv_freq": tensor,
                    "lm_head.weight": tensor,
                ],
                perLayerQuantization: .init(perLayerQuantization: [
                    "model.embed_tokens": .quantize(quantization)
                ])))
        XCTAssertEqual(
            Set(cleaned.weights.keys), ["model.embedding.weight", "model.embedding.scales"])
        XCTAssertEqual(
            cleaned.perLayerQuantization?.quantization(layer: "model.embedding"), quantization)
        XCTAssertThrowsError(
            try model.prepareCheckpoint(
                ModelCheckpoint(weights: [
                    "model.embedding.weight": tensor, "model.embed_tokens.weight": tensor,
                ])))
    }

    func testConvertedCheckpointLoadsThroughSharedLoader() throws {
        let config = try configuration()
        let source = SparkModel(config)
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["model.embed_tokens.weight"] = weights.removeValue(forKey: "model.embedding.weight")
        weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = MLXArray([Float(1)])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        let loaded = SparkModel(config)
        try loadWeights(modelDirectory: directory, model: loaded)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let difference = MLX.max(MLX.abs(source(tokens, cache: nil) - loaded(tokens, cache: nil)))
        XCTAssertLessThan(difference.item(Float.self), 1e-5)
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
