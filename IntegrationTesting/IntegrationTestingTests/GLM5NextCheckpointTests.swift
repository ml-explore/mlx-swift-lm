import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class GLM5NextCheckpointTests: XCTestCase {
    private func loadModel() async throws -> any LanguageModel {
        guard let path = ProcessInfo.processInfo.environment["MLX_GLM5NEXT_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_GLM5NEXT_CHECKPOINT to a local GLM-5-Next checkpoint")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let start = Date.timeIntervalSinceReferenceDate
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: baseConfig.modelType)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        eval(model)
        print("GLM5Next checkpoint load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
        return model
    }

    /// Prefill 1,024 tokens in chunks of 256, then 16 greedy decode steps. The weights are
    /// held wired, as a caller of `generate` does with a wired memory ticket: unwired, a
    /// checkpoint this size is paged on every decode step.
    func testCheckpointPrefillAndDecode() async throws {
        let model = try await loadModel()
        let weightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        let ticket = MLX.WiredSumPolicy().ticket(size: weightBytes, kind: .active)
        try await ticket.withWiredLimit {
            Memory.peakMemory = 0
            let count = 1024
            // Small ids, inside any GLM-5-Next vocabulary.
            let tokens = MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % 1000) })
                .reshaped(1, count)
            let cache = try model.newCache(parameters: nil)
            let start = Date.timeIntervalSinceReferenceDate
            var output: MLXArray?
            for offset in stride(from: 0, to: count, by: 256) {
                output = model(tokens[0..., offset ..< offset + 256], cache: cache)
                eval(output!)
                eval(cache)
            }
            var logits = try XCTUnwrap(output)[0..., (-1)...]
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
                "GLM5Next checkpoint prefill_tok_s=\(Double(count) / prefillSeconds), decode_tok_s=\(16 / decodeSeconds), peak_bytes=\(Memory.peakMemory)"
            )
        }
    }
}
