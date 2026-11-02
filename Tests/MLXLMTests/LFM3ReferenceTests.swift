// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFM3ReferenceTests: XCTestCase {
    private func resource(_ name: String, _ extensionName: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: extensionName))
    }

    func testLongContextChunking() throws {
        guard ProcessInfo.processInfo.environment["MLX_LFM3_LONG_CONTEXT"] == "1" else {
            throw XCTSkip("Set MLX_LFM3_LONG_CONTEXT=1 for 4K, 16K and 32K cache checks")
        }
        let config = try JSONDecoder().decode(
            LFM3Configuration.self,
            from: Data(contentsOf: resource("lfm3-reference-config", "json")))
        let model = LFM3Model(config)
        let weights = try loadArrays(url: resource("lfm3-reference-weights", "safetensors"))
        try model.update(
            parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)),
            verify: [.all])
        for length in [4096, 16384, 32768] {
            let tokens = MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % 16) }).reshaped(
                1, length)
            var results = [MLXArray]()
            for chunkSize in [257, 512] {
                let cache = try model.newCache(parameters: nil)
                for start in stride(from: 0, to: length, by: chunkSize) {
                    let hidden = model.hiddenStates(
                        tokens[0..., start ..< min(start + chunkSize, length)], cache: cache)
                    eval(hidden)
                }
                let output = model(MLXArray([Int32(1)]).reshaped(1, 1), cache: cache)
                eval(output)
                results.append(output)
                for (i, layerCache) in cache.enumerated() {
                    if config.layerTypes[i] == .conv {
                        XCTAssertEqual(
                            (layerCache as? MambaCache)?[0]?.dim(1), config.convLCache - 1)
                    } else {
                        XCTAssertEqual(layerCache.offset, length + 1)
                        if config.layerTypes[i] == .slidingAttention {
                            XCTAssertLessThanOrEqual(
                                layerCache.state[0].dim(2), config.slidingWindow)
                        }
                    }
                }
            }
            let error = abs(results[0] - results[1]).max().item(Float.self)
            print("LFM3 context \(length): final-logit max_abs=\(error)")
            XCTAssertLessThan(error, MatmulPrecision.splitTolerance)
            XCTAssertTrue(all(isFinite(results[0])).item(Bool.self))
        }
    }

    func testOriginalHuggingFaceFloat32Parity() throws {
        // CPU pins IEEE FP32, including on devices that use TF32 for GPU matmuls.
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(
                LFM3Configuration.self,
                from: Data(contentsOf: resource("lfm3-reference-config", "json")))
            let model = LFM3Model(config)
            let weights = try loadArrays(url: resource("lfm3-reference-weights", "safetensors"))
            try model.update(
                parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)),
                verify: [.all])
            let reference = try loadArrays(url: resource("lfm3-reference-outputs", "safetensors"))
            let tokens = try XCTUnwrap(reference["tokens"])

            func compare(_ actual: MLXArray, _ key: String) throws {
                let expected = try XCTUnwrap(reference[key], key)
                XCTAssertEqual(actual.shape, expected.shape, key)
                let maximum = abs(actual - expected).max().item(Float.self)
                let relative = maximum / max(abs(expected).max().item(Float.self), 1e-8)
                print("LFM3 FP32 \(key): max_abs=\(maximum), normalized_max=\(relative)")
                XCTAssertLessThan(maximum, 2e-5, key)
            }

            var h = model.model.embed(tokens)
            try compare(h, "model.embed_tokens")
            for (i, layer) in model.model.layers.enumerated() {
                let mask = makeAttentionMask(
                    n: tokens.dim(1), cache: nil,
                    windowSize: layer.layerType == .slidingAttention ? config.slidingWindow : nil)
                let normalized = layer.operatorNorm(h)
                if let conv = layer.conv {
                    try compare(conv(normalized, cache: nil), "model.layers.\(i).conv")
                } else if let attention = layer.attention {
                    try compare(
                        attention(normalized, mask: mask, cache: nil), "model.layers.\(i).self_attn"
                    )
                }
                h = layer(h, mask: mask, cache: nil)
                try compare(h, "model.layers.\(i)")
            }
            let hidden = model.hiddenStates(tokens, cache: nil)
            try compare(hidden, "model.embedding_norm")
            let logits = model.projectLogits(hidden)
            try compare(logits, "logits")
            XCTAssertEqual(
                argMax(logits, axis: -1).asArray(Int.self),
                argMax(try XCTUnwrap(reference["logits"]), axis: -1).asArray(Int.self))

            let cache = try model.newCache(parameters: nil)
            var outputs = [MLXArray]()
            for (start, end) in [(0, 2), (2, 7), (7, 18), (18, 19)] {
                outputs.append(model(tokens[0..., start ..< end], cache: cache))
                eval(outputs.last!)
            }
            try compare(concatenated(outputs, axis: 1), "cached_logits")
        }
    }
}
