import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Parity with the model author's implementation, transformers' `glm5_next`, on the tiny
/// checkpoint described in `tools/glm5-next-reference.py`.
final class GLM5NextReferenceTests: XCTestCase {

    /// The generator's `weight_values`: a hash of the flat element index, the tensor's index
    /// in sorted checkpoint-name order and the seed, then one float32 division.
    private func weightValues(_ name: String, index: Int, shape: [Int], seed: Int) -> MLXArray {
        let (numeratorShift, denominator): (Int, Int) =
            if name.hasSuffix("norm.weight") {
                (196608, 196608)
            } else if name.hasSuffix("hc_attn_scale") || name.hasSuffix("hc_ffn_scale") {
                (98304, 196608)
            } else if name.hasSuffix("e_score_correction_bias") {
                (0, 196608)
            } else {
                (0, 131072)
            }
        let count = shape.reduce(1, *)
        let values = (0 ..< count).map { j -> Float in
            var h = (j * 7919 + index * 104729 + seed * 1_299_709 + 12345) % 65521
            h = (h * h + 7) % 65521
            h = (h * h + 13) % 65521
            return Float(h - 32760 + numeratorShift) / Float(denominator)
        }
        return MLXArray(values, shape)
    }

    /// Writes the reference's config.json and the formula weights under the checkpoint names
    /// `save_pretrained` writes, vision tower included.
    private func writeCheckpoint(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(GLM5NextReferenceValues.configuration.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        var weights = [String: MLXArray]()
        // Match the Python reference independently of the order of the literal.
        let tensors = GLM5NextReferenceValues.checkpointTensors.sorted { $0.0 < $1.0 }
        for (index, (name, shape)) in tensors.enumerated() {
            weights[name] = weightValues(
                name, index: index, shape: shape, seed: GLM5NextReferenceValues.weightSeed)
        }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    }

    /// Through the model registry and `loadWeights`, so configuration decoding (the
    /// `indexed_attention` layer type), validation and `sanitize` all run on the checkpoint
    /// layout transformers saves.
    private func loadReferenceModel() async throws -> any LanguageModel {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm5next-reference-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(to: directory)

        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        XCTAssertEqual(base.modelType, "glm5_next")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    /// Prefill `tokens` in chunks ending at `prefillEnds`, then one token per step;
    /// returns the float32 log-softmax at every position.
    private func teacherForced(
        _ model: any LanguageModel, tokens: [Int], prefillEnds: [Int]
    ) throws -> [[Float]] {
        let cache = try model.newCache(parameters: nil)
        let ends =
            prefillEnds + Array(stride(from: prefillEnds.last! + 1, through: tokens.count, by: 1))
        var rows = [[Float]]()
        var start = 0
        for end in ends {
            let chunk = MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, end - start)
            let logits = model(chunk, cache: cache)[0].asType(.float32)
            let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
            eval(logprobs)
            for row in 0 ..< (end - start) {
                rows.append(logprobs[row].asArray(Float.self))
            }
            start = end
        }
        return rows
    }

    /// Every prefill plan of the generator, teacher-forced on the reference's greedy tokens,
    /// against the reference's one-call log-probabilities (its own plans agree with those
    /// within a tenth of the tolerance), and greedy tokens equal.
    func testTinyCheckpointMatchesTransformersReference() async throws {
        let model = try await loadReferenceModel()
        let prompt = GLM5NextReferenceValues.prompt
        let greedy = GLM5NextReferenceValues.greedy
        let tokens = prompt + greedy.dropLast()
        let vocabulary = GLM5NextReferenceValues.logprobs.count / tokens.count
        XCTAssertEqual(vocabulary * tokens.count, GLM5NextReferenceValues.logprobs.count)
        let expected = (0 ..< tokens.count).map {
            GLM5NextReferenceValues.logprobs[($0 * vocabulary) ..< (($0 + 1) * vocabulary)]
        }
        let plans: [(String, [Int])] = [
            ("one_call", [32]),
            ("chunks_5_13_22_32", [5, 13, 22, 32]),
            ("chunks_of_7", [7, 14, 21, 28, 32]),
            ("token_by_token", Array(1 ... 32)),
        ]
        let tolerance = GLM5NextReferenceValues.tolerance

        // CPU pins IEEE FP32. MLX's compile cache matches entries on its global default
        // stream, not on `withDefaultDevice`'s task-local device, so compilation is off.
        compile(enable: false)
        defer { compile(enable: true) }
        try Device.withDefaultDevice(.cpu) {
            var worst: Float = 0
            for (name, prefillEnds) in plans {
                let rows = try teacherForced(model, tokens: tokens, prefillEnds: prefillEnds)
                XCTAssertEqual(rows.count, expected.count, name)
                var difference: Float = 0
                for (row, reference) in zip(rows, expected) {
                    XCTAssertEqual(row.count, reference.count, name)
                    for (a, b) in zip(row, reference) {
                        difference = max(difference, abs(a - b))
                    }
                }
                worst = max(worst, difference)
                print("glm5_next parity \(name): max |dlogprob| = \(difference)")
                XCTAssertLessThanOrEqual(difference, tolerance, name)

                // The reference's argmax margins exceed its threshold, so greedy agrees exactly.
                let tokens = rows[(prompt.count - 1)...].map { row in
                    row.indices.max { row[$0] < row[$1] }!
                }
                XCTAssertEqual(tokens, greedy, name)
            }
            print("glm5_next parity max |dlogprob| = \(worst) tolerance = \(tolerance)")
        }
    }
}
