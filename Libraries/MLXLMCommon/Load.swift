// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return Set(index.weightMap.values)
            .sorted()
            .map { modelDirectory.appendingPathComponent($0) }
    }

    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    return enumerator.compactMap { item -> URL? in
        guard let url = item as? URL, url.pathExtension == "safetensors" else {
            return nil
        }
        return url
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for url in try safetensorWeightURLs(in: modelDirectory) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
