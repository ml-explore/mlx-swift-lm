// Copyright © 2024 Apple Inc.

import Foundation
import HuggingFace
import MLX
import MLXNN
import Tokenizers

public enum ModelLoadError: LocalizedError {
    case cacheNotConfigured

    public var errorDescription: String? {
        switch self {
        case .cacheNotConfigured:
            return "Hub cache is not configured"
        }
    }
}

/// Download the model from the Hugging Face Hub.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// By default, returns cached model files without network calls if the model has been
/// previously downloaded. Pass `useLatest: true` to check the Hub for updates.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubClient instance
///   - configuration: the model identifier
///   - useLatest: when true, always checks the Hub for the latest version
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    from hub: HubClient, configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    switch configuration.id {
    case .id(let id, let revision):
        guard let cache = hub.cache else {
            throw ModelLoadError.cacheNotConfigured
        }
        let repoID = Repo.ID(stringLiteral: id)
        let destination = cache.repoDirectory(repo: repoID, kind: .model)

        // Fast path: if this revision has been downloaded before, return the repo
        // directory without any network calls.
        if !useLatest,
            cachedSnapshotDirectory(cache: cache, repo: repoID, revision: revision) != nil
        {
            return destination
        }

        do {
            return try await hub.downloadSnapshot(
                of: repoID,
                to: destination,
                revision: revision,
                matching: ["*.safetensors", "*.json"],
                progressHandler: progressHandler
            )
        } catch {
            // Fall back to local cache if download fails (offline, unauthorized, etc.)
            if let cached = cachedSnapshotDirectory(
                cache: cache, repo: repoID, revision: revision)
            {
                return cached
            }
            throw error
        }
    case .directory(let directory):
        return directory
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
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

/// Look up a cached snapshot directory for a given repo and revision.
///
/// Returns the snapshot directory URL if it exists, nil otherwise.
public func cachedSnapshotDirectory(
    cache: HubCache, repo: Repo.ID, revision: String
) -> URL? {
    let commitHash: String
    if revision.count == 40, revision.allSatisfy(\.isHexDigit) {
        commitHash = revision
    } else if let resolved = cache.resolveRevision(repo: repo, kind: .model, ref: revision) {
        commitHash = resolved
    } else {
        return nil
    }
    let snapshotDir = cache.snapshotsDirectory(repo: repo, kind: .model)
        .appendingPathComponent(commitHash)
    if FileManager.default.fileExists(atPath: snapshotDir.path) {
        return snapshotDir
    }
    return nil
}
