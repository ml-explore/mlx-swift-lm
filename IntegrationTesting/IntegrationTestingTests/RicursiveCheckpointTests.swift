import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class RicursiveCheckpointTests: XCTestCase {
    /// Checks that a published checkpoint's weight names are exactly the ones `RicursiveModel`
    /// needs. Set `MLX_RICURSIVE_CHECKPOINTS` to a colon-separated list of directories holding
    /// `config.json` and `model.safetensors.index.json`; no weights are read. The model is
    /// built at a tiny width because the names depend only on the layer stack.
    func testCheckpointWeightNames() throws {
        guard let list = ProcessInfo.processInfo.environment["MLX_RICURSIVE_CHECKPOINTS"] else {
            throw XCTSkip(
                "Set MLX_RICURSIVE_CHECKPOINTS to colon-separated local Ricursive checkpoints")
        }
        for path in list.split(separator: ":") {
            let directory = URL(filePath: String(path))
            let configData = try Data(contentsOf: directory.appending(component: "config.json"))
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: configData) as? [String: Any])
            let tiny: [String: Any] = [
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "sliding_window": 8,
            ]
            json.merge(tiny) { $1 }
            let config = try JSONDecoder().decode(
                RicursiveConfiguration.self,
                from: JSONSerialization.data(withJSONObject: json))
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

            let index = try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: directory.appending(
                            component: "model.safetensors.index.json"))) as? [String: Any])
            let map = try XCTUnwrap(index["weight_map"] as? [String: Any])
            let names = Set(map.keys)

            let model = RicursiveModel(config)
            if let perLayer = base.perLayerQuantization {
                quantize(model: model) { path, _ in
                    guard names.contains("\(path).scales"),
                        let option = perLayer.quantization(layer: path)
                    else { return nil }
                    // Parameter names do not depend on the group size.
                    return (32, option.bits, option.mode)
                }
            }
            let expected = Set(model.parameters().flattened().map(\.0))
            XCTAssertEqual(
                expected.subtracting(names).sorted(), [], "\(path): missing from the checkpoint")
            XCTAssertEqual(
                names.subtracting(expected).sorted(), [], "\(path): not used by the model")
            print("ricursive names \(directory.lastPathComponent): \(names.count) keys match")
        }
    }
}
