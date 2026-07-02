import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Mamba2Tests: XCTestCase {

    private func makeConfig() throws -> Mamba2Configuration {
        let json = """
            {
                "model_type": "mamba2",
                "num_heads": 4,
                "head_dim": 4,
                "vocab_size": 32,
                "hidden_size": 16,
                "state_size": 8,
                "num_hidden_layers": 2,
                "layer_norm_epsilon": 1e-5,
                "conv_kernel": 4,
                "n_groups": 1,
                "use_bias": false,
                "use_conv_bias": true,
                "tie_word_embeddings": false,
                "time_step_limit": [0.0, 100.0]
            }
            """
        return try JSONDecoder().decode(Mamba2Configuration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = Mamba2Model(try makeConfig())
        let inputs = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let logits = model(inputs, cache: model.newCache(parameters: nil))
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 5, 32])
    }

    func testNewCacheIsMambaCachePerLayer() throws {
        let config = try makeConfig()
        let model = Mamba2Model(config)
        let cache = model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, config.numHiddenLayers)
        XCTAssertTrue(cache.allSatisfy { $0 is MambaCache })
    }

    func testSanitizeSwapsConv1dWeightAxes() throws {
        let model = Mamba2Model(try makeConfig())

        // HF stores depthwise conv1d weight as [channels, 1, kernel]; the Swift
        // Conv1d wants [channels, kernel, 1]. sanitize swaps axes 1 and 2.
        let hf = MLXArray.zeros([20, 1, 4])
        let out = model.sanitize(weights: ["backbone.layers.0.mixer.conv1d.weight": hf])
        XCTAssertEqual(out["backbone.layers.0.mixer.conv1d.weight"]?.shape, [20, 4, 1])

        // Already-correct ([channels, kernel, 1] with last dim == 1) is left alone.
        let ok = MLXArray.zeros([20, 4, 1])
        let out2 = model.sanitize(weights: ["backbone.layers.0.mixer.conv1d.weight": ok])
        XCTAssertEqual(out2["backbone.layers.0.mixer.conv1d.weight"]?.shape, [20, 4, 1])
    }
}
