// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFM3CheckpointTests: XCTestCase {
    private func loadModel() throws -> LFM3Model {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_LFM3_CHECKPOINT to a local published LFM3 checkpoint")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM3Configuration.self, from: data)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let start = Date.timeIntervalSinceReferenceDate
        let model = LFM3Model(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        eval(model)
        print("LFM3 checkpoint load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
        return model
    }

    func testPublishedCheckpointPrefillAndDecode() throws {
        let model = try loadModel()
        Memory.peakMemory = 0
        let count = 1024
        let tokens = MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % model.vocabularySize) })
            .reshaped(1, count)
        let cache = try model.newCache(parameters: nil)
        let start = Date.timeIntervalSinceReferenceDate
        var hidden: MLXArray?
        for offset in stride(from: 0, to: count, by: 256) {
            hidden = model.hiddenStates(tokens[0..., offset ..< offset + 256], cache: cache)
            eval(hidden!)
            eval(cache)
        }
        var logits = model.projectLogits(try XCTUnwrap(hidden)[0..., (-1)...])
        eval(logits)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - start
        let decodeStart = Date.timeIntervalSinceReferenceDate
        for _ in 0 ..< 16 {
            let next = argMax(logits[0, -1]).reshaped(1, 1)
            logits = model(next, cache: cache)
            eval(logits)
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        print(
            "LFM3 checkpoint prefill_tok_s=\(Double(count) / prefillSeconds), decode_tok_s=\(16 / decodeSeconds), peak_bytes=\(Memory.peakMemory)"
        )
    }

    func testPublishedCheckpointReferenceParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_REFERENCE_FILE"] else {
            throw XCTSkip("Set MLX_LFM3_REFERENCE_FILE to the matching FP32 reference outputs")
        }
        guard !MatmulPrecision.isReducedFloat32 else {
            throw XCTSkip("Set MLX_ENABLE_TF32=0 to compare the published model in IEEE FP32")
        }
        let reference = try loadArrays(url: URL(filePath: path))
        let model = try loadModel()
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0 }
        for length in [19, 257, 1024] {
            let tokens = try XCTUnwrap(reference["tokens_\(length)"])
            let cache = try model.newCache(parameters: nil)
            var hidden: MLXArray?
            for start in stride(from: 0, to: length, by: 127) {
                hidden = model.hiddenStates(
                    tokens[0..., start ..< min(start + 127, length)], cache: cache)
                eval(hidden!)
                eval(cache)
            }
            let final = try XCTUnwrap(hidden)[0..., (-1)...]
            let logits = model.projectLogits(final)
            let expected = try XCTUnwrap(reference["logits_\(length)"])
            let error = abs(logits - expected).max().item(Float.self)
            let hiddenError = abs(final - (try XCTUnwrap(reference["hidden_\(length)"]))).max()
                .item(Float.self)
            print(
                "LFM3 published FP32 \(length): logits_max_abs=\(error), hidden_max_abs=\(hiddenError)"
            )
            XCTAssertLessThan(error, 2e-4)
            XCTAssertLessThan(hiddenError, 2e-4)
            XCTAssertEqual(
                argMax(logits, axis: -1).item(Int.self), argMax(expected, axis: -1).item(Int.self))
        }
    }
}
