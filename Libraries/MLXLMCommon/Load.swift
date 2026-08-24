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

/// How the safetensors files holding a model's weights are chosen.
///
/// ## See Also
/// - ``ModelConfiguration/weightFileSelection``
public enum WeightFileSelection: Sendable, Equatable {
    /// Use `model.safetensors.index.json` when it names files that exist, otherwise the
    /// conventional `model*.safetensors` (then `weight*.safetensors`) names.
    ///
    /// This is what a well-packaged checkpoint wants and it is the default.
    case automatic

    /// Load every safetensors file in the model directory.
    ///
    /// This is an escape hatch for a checkpoint whose index is known to be wrong in a way
    /// ``automatic`` cannot detect -- an index that names files that all exist but omits
    /// weights the model needs. It loads files that may not belong to this model, and a
    /// stray tensor whose name collides with one the model's `sanitize(weights:)` rewrites
    /// is loaded silently rather than reported, so prefer a model that declares its own
    /// extra files (see ``AdditionalWeightFilesProviding``) where that is possible.
    case allFilesPresent
}

/// The safetensors files in `modelDirectory` that hold the model's weights.
///
/// Only the top level of the directory is considered. Checkpoints keep auxiliary weights that
/// belong to a different module in subdirectories (for example `mlx-community/Qwen3.5-4B-OptiQ-4bit`
/// and its `optiq/mtp.safetensors`), and a nested Hugging Face snapshot cache under a local
/// checkpoint directory would otherwise be pulled in as well.
///
/// With ``WeightFileSelection/automatic`` the files are chosen in this order:
///
/// 1. The files named by `model.safetensors.index.json`, when it exists and every file it names
///    exists. The index is precise about which of several weight files belong to the model, which
///    matters for a repo that ships both a consolidated file and shards.
/// 2. The conventional `model*.safetensors` names, matching `mlx_lm.utils.load_model`. Uploads
///    regularly ship an index carried over from an unquantized source repo that names shards the
///    repo does not contain, and the convention is what those repos actually follow.
/// 3. `weight*.safetensors`, then every safetensors file present, so a directory that follows no
///    convention at all still loads.
///
/// `additionalFiles` names files the model requires that no rule above selects, for example the
/// Jina reranker's `projector.safetensors`. They are appended, so a file the index already names
/// is not loaded twice, and names that are not present are ignored.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - selection: how to choose the files, see ``WeightFileSelection``
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     selected ones. See ``AdditionalWeightFilesProviding/additionalWeightFiles``.
package func safetensorWeightURLs(
    in modelDirectory: URL,
    selection: WeightFileSelection = .automatic,
    additionalFiles: [String] = []
) throws -> [URL] {
    let present = topLevelSafetensorURLs(in: modelDirectory)

    let selected: [URL]
    switch selection {
    case .allFilesPresent:
        selected = present
    case .automatic:
        selected = try indexedWeightURLs(in: modelDirectory) ?? conventionalWeightURLs(in: present)
    }

    var seen = Set(selected.map(\.standardizedFileURL.path))
    var urls = selected
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }
    return urls
}

/// The files named by `model.safetensors.index.json`, or `nil` when there is no index or it
/// names a file the directory does not contain.
///
/// Existence is checked against the file system rather than the top-level listing: an index may
/// legitimately map weights into a subdirectory, and that is a deliberate statement about where
/// this model's weights live rather than an unrelated file that happens to be nearby.
private func indexedWeightURLs(in modelDirectory: URL) throws -> [URL]? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return nil
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let urls = Set(index.weightMap.values)
        .sorted()
        .map { modelDirectory.appendingPathComponent($0) }

    guard !urls.isEmpty,
        urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return nil
    }
    return urls
}

/// The conventionally named weight files among `present`, matching `mlx_lm.utils.load_model`'s
/// `model*.safetensors` glob, with `weight*.safetensors` and then everything as fallbacks.
private func conventionalWeightURLs(in present: [URL]) -> [URL] {
    for prefix in ["model", "weight"] {
        let matches = present.filter { $0.lastPathComponent.hasPrefix(prefix) }
        if !matches.isEmpty {
            return matches
        }
    }
    return present
}

private func topLevelSafetensorURLs(in modelDirectory: URL) -> [URL] {
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    return
        contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
///
/// The weight files are chosen from `model.safetensors.index.json` when it names files that
/// exist, and otherwise by the conventional `model*.safetensors` names. A model can name extra
/// files it needs by conforming to ``AdditionalWeightFilesProviding``, and a caller can override
/// the choice with ``ModelConfiguration/weightFileSelection``.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let additionalFiles = (model as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
    for url in try safetensorWeightURLs(
        in: modelDirectory,
        selection: weightFileSelection,
        additionalFiles: additionalFiles ?? [])
    {
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
