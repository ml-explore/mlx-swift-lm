// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
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
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // handle pre-quantized weights (models where safetensors already contain
    // quantized weight + scales + biases) vs. float-weight models that need
    // load-time quantization
    if hasPreQuantizedWeights(weights) {
        try loadPreQuantizedWeights(model: model, weights: &weights, quantization: quantization)
    } else if quantization != nil || perLayerQuantization != nil {
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

/// Check if the loaded weights contain pre-quantized layers.
private func hasPreQuantizedWeights(_ weights: [String: MLXArray]) -> Bool {
    weights.keys.contains { $0.hasSuffix(".scales") }
}

/// Load pre-quantized weights directly into QuantizedLinear layers.
///
/// For each layer path that has `.scales` and `.biases` in the weights dict:
/// - If the path corresponds to a `Linear` child in the model, replace it with
///   a `QuantizedLinear`.
/// - Otherwise (e.g. `Embedding`), dequantize the weights in-place so they can
///   be applied via the normal parameter update path.
private func loadPreQuantizedWeights(
    model: BaseLanguageModel, weights: inout [String: MLXArray],
    quantization: BaseConfiguration.Quantization? = nil
) throws {
    // Use provided quantization config, or default to MXFP8 (groupSize=32, bits=8)
    // which is the format used by models like MiniCPM-V 4.6.
    let groupSize = quantization?.groupSize ?? 32
    let bits = quantization?.bits ?? 8
    let mode = quantization?.mode ?? .mxfp8
    // Collect all quantized layer paths that exist as Linear children in the model.
    let linearPaths = Set(model.leafModules().flattened().filter { $0.1 is Linear }.map { $0.0 })
    var quantizedPaths = Set<String>()
    var nonLinearQuantizedPaths = Set<String>()
    for key in weights.keys {
        if key.hasSuffix(".scales") {
            let prefix = String(key.dropLast(".scales".count))
            if linearPaths.contains(prefix) {
                quantizedPaths.insert(prefix)
            } else {
                nonLinearQuantizedPaths.insert(prefix)
            }
        }
    }

    // Build module replacements: for each quantized path, create a QuantizedLinear
    // with the pre-quantized values and replace the existing Linear.
    var moduleReplacements = [String: Module]()
    for path in quantizedPaths {
        guard let weight = weights["\(path).weight"],
              let scales = weights["\(path).scales"]
        else {
            continue
        }
        let biases = weights["\(path).biases"]

        let quantized = QuantizedLinear(
            weight: weight, bias: nil,
            scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode
        )
        quantized.freeze()
        moduleReplacements[path] = quantized
    }

    // Apply module replacements (swap Linear → QuantizedLinear)
    if !moduleReplacements.isEmpty {
        let children = ModuleChildren.unflattened(moduleReplacements)
        model.update(modules: children)
    }

    // For non-Linear quantized paths (e.g. Embedding), dequantize the weights
    // in-place so they match the expected shape for the normal parameter update.
    for path in nonLinearQuantizedPaths {
        guard let weight = weights["\(path).weight"],
              let scales = weights["\(path).scales"]
        else {
            continue
        }
        let biases = weights["\(path).biases"]
        weights["\(path).weight"] = dequantized(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode
        )
        // Remove quantization metadata — no longer needed.
        weights["\(path).scales"] = nil
        weights["\(path).biases"] = nil
    }

    // Remove VLM-only keys from the weights dict.
    // Quantized path keys (weight/scales/biases) are kept because Swift
    // Mirror exposes them as parameters on QuantizedLinear; filtering them
    // out would cause .allModelKeysSet validation to fail.
    // sanitize transforms merger.* → language_model.merger.* (and similar)
    let excludePrefixes = [
        "vision_tower", "merger", "vit_merger", "vpm",
        "language_model.merger", "language_model.vit_merger", "language_model.vpm",
    ]
    weights = weights.filter { key, _ in
        for prefix in excludePrefixes {
            if key.hasPrefix(prefix) || key.hasPrefix("\(prefix).") {
                return false
            }
        }
        return true
    }
}
