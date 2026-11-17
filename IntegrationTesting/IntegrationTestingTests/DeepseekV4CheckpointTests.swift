// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class DeepseekV4CheckpointTests: XCTestCase {

    /// A real checkpoint's safetensors headers (names, shapes, quantization modes) against the
    /// module tree after sanitize and per-layer quantization; no weight is read.
    func testCheckpointLayoutMatchesModuleTree() throws {
        guard let path = ProcessInfo.processInfo.environment["DEEPSEEK_V4_CHECKPOINT_DIR"] else {
            throw XCTSkip(
                "Set DEEPSEEK_V4_CHECKPOINT_DIR to an MXFP4 MLX conversion of DeepSeek-V4-Flash")
        }
        try Device.withDefaultDevice(.cpu) {
            let directory = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let dtypes: [String: DType] = [
                "BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32,
                "I64": .int64, "I32": .int32,
            ]
            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(atPath: path)
                .filter { $0.hasSuffix(".safetensors") }
            XCTAssertFalse(files.isEmpty)
            for file in files {
                let handle = try FileHandle(
                    forReadingFrom: directory.appendingPathComponent(file))
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
}
