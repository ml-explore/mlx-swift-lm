import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class RicursiveCheckpointTests: XCTestCase {
    private func loadModel() throws -> RicursiveModel {
        guard let path = ProcessInfo.processInfo.environment["MLX_RICURSIVE_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_RICURSIVE_CHECKPOINT to a local published Ricursive checkpoint")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(RicursiveConfiguration.self, from: data)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let start = Date.timeIntervalSinceReferenceDate
        let model = RicursiveModel(config)
        // Rejects checkpoint keys the model lacks and model parameters the checkpoint lacks.
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        model.train(false)
        eval(model)
        print("Ricursive checkpoint load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
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
        var output: MLXArray?
        for offset in stride(from: 0, to: count, by: 256) {
            output =
                model.nextTokenLogits(
                    .init(tokens: tokens[0..., offset ..< offset + 256]), cache: cache, state: nil
                ).logits
            eval(output!)
            eval(cache)
        }
        var logits = try XCTUnwrap(output)[0..., (-1)...]
        eval(logits)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - start
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        let decodeStart = Date.timeIntervalSinceReferenceDate
        for _ in 0 ..< 16 {
            let next = argMax(logits[0, -1]).reshaped(1, 1)
            logits = model(next, cache: cache)
            eval(logits)
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        print(
            "Ricursive checkpoint prefill_tok_s=\(Double(count) / prefillSeconds), decode_tok_s=\(16 / decodeSeconds), peak_bytes=\(Memory.peakMemory)"
        )
    }
}
