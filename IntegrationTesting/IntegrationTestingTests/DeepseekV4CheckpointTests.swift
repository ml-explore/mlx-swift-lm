// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class DeepseekV4CheckpointTests: XCTestCase {

    private func checkpoint() throws -> (URL, DeepseekV4Configuration, BaseConfiguration) {
        guard let path = ProcessInfo.processInfo.environment["MLX_DEEPSEEK_V4_CHECKPOINT"] else {
            throw XCTSkip(
                "Set MLX_DEEPSEEK_V4_CHECKPOINT to an MXFP4 MLX conversion of DeepSeek-V4-Flash")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        try config.validateModelConfiguration()
        return (directory, config, try JSONDecoder().decode(BaseConfiguration.self, from: data))
    }

    /// The safetensors headers against the module tree after sanitize and per-layer
    /// quantization; no weight is read.
    func testCheckpointLayoutMatchesModuleTree() throws {
        let (directory, config, base) = try checkpoint()
        try Device.withDefaultDevice(.cpu) {
            let dtypes: [String: DType] = [
                "BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32,
                "I64": .int64, "I32": .int32,
            ]
            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(atPath: directory.path())
                .filter { $0.hasSuffix(".safetensors") }
            XCTAssertFalse(files.isEmpty)
            for file in files {
                let handle = try FileHandle(
                    forReadingFrom: directory.appending(component: file))
                defer { try? handle.close() }
                let size = try XCTUnwrap(handle.read(upToCount: 8)).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt64.self)
                }
                let header = try XCTUnwrap(handle.read(upToCount: Int(size)))
                let entries = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: header) as? [String: Any])
                for (name, value) in entries where name != "__metadata__" {
                    let entry = try XCTUnwrap(value as? [String: Any])
                    let shape = try XCTUnwrap(entry["shape"] as? [Int])
                    let dtype = try XCTUnwrap(dtypes[try XCTUnwrap(entry["dtype"] as? String)])
                    weights[name] = MLXArray.zeros(shape, dtype: dtype)
                }
            }
            let model = DeepseekV4Model(config)
            let sanitized = model.sanitize(weights: weights)
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? base.perLayerQuantization?.quantization(layer: path)?.asTuple : nil
            }
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            let modes = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    key, module -> (String, QuantizationMode)? in
                    (module as? any Quantized).map { (key, $0.mode) }
                })
            XCTAssertEqual(modes["layers.2.ffn.switch_mlp.gate_proj"], .mxfp4)
            XCTAssertEqual(modes["layers.2.ffn.shared_experts.w1"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.wo_a"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.indexer.wq_b"], .mxfp8)
            XCTAssertNil(modes["layers.2.attn.compressor.wkv"])
            XCTAssertNil(modes["head"])
        }
    }

    /// The loaded checkpoint prefills 512 tokens in chunks and decodes 8 greedy steps with
    /// finite logits.
    func testCheckpointPrefillAndDecode() throws {
        let (directory, config, base) = try checkpoint()
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let count = 512
        let tokens = MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % model.vocabularySize) })
            .reshaped(1, count)
        let cache = try model.newCache(parameters: nil)
        var logits = MLXArray()
        for offset in stride(from: 0, to: count, by: 256) {
            logits = model(tokens[0..., offset ..< (offset + 256)], cache: cache)[0..., (-1)...]
            eval(logits)
            eval(cache)
        }
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self), "prefill")
        for _ in 0 ..< 8 {
            logits = model(argMax(logits[0, -1]).reshaped(1, 1), cache: cache)
            eval(logits)
            XCTAssertTrue(all(isFinite(logits)).item(Bool.self), "decode")
        }
    }
}
