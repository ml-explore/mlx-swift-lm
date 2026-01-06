// Copyright © 2024 Apple Inc.

import Foundation
@preconcurrency import Hub
import MLX
import MLXNN
import Tokenizers

public enum EmbedderError: LocalizedError {
    case unsupportedModelType(String)
    case configurationFileError(String, String, Error)
    case configurationDecodingError(String, String, DecodingError)
    case missingTokenizerConfig

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
        case .missingTokenizerConfig:
            return "Missing tokenizer configuration"
        }
    }

    private func extractDecodingErrorDetail(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = (context.codingPath + [key]).map { $0.stringValue }.joined(separator: ".")
            return "Missing field '\(path)'"
        case .typeMismatch(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Type mismatch at '\(path)'"
        case .valueNotFound(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Missing value at '\(path)'"
        case .dataCorrupted(let context):
            if context.codingPath.isEmpty {
                return "Invalid JSON"
            } else {
                let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
                return "Invalid data at '\(path)'"
            }
        @unknown default:
            return error.localizedDescription
        }
    }
}

func prepareModelDirectory(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "config.json", "*/config.json"]
            return try await hub.snapshot(
                from: repo, matching: modelFiles, progressHandler: progressHandler)

        case .directory(let directory):
            return directory
        }
    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)
    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load and return the model and tokenizer
public func load(
    hub: HubApi = HubApi(), configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> (EmbeddingModel, Tokenizer) {
    let modelDirectory = try await prepareModelDirectory(
        hub: hub, configuration: configuration, progressHandler: progressHandler)

    // Load tokenizer and model in parallel using async let.
    async let tokenizerTask = loadTokenizer(configuration: configuration, hub: hub)
    let model = try loadSynchronous(modelDirectory: modelDirectory, modelName: configuration.name)
    let tokenizer = try await tokenizerTask

    return (model, tokenizer)
}

func loadSynchronous(modelDirectory: URL, modelName: String) throws -> EmbeddingModel {
    // Load config.json once and decode for both base config and model-specific config
    let configurationURL = modelDirectory.appending(component: "config.json")
    let configData: Data
    do {
        configData = try Data(contentsOf: configurationURL)
    } catch {
        throw EmbedderError.configurationFileError(
            configurationURL.lastPathComponent, modelName, error)
    }
    let baseConfig: BaseConfiguration
    do {
        baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
    } catch let error as DecodingError {
        throw EmbedderError.configurationDecodingError(
            configurationURL.lastPathComponent, modelName, error)
    }

    let modelType = ModelType(rawValue: baseConfig.modelType)
    let model: EmbeddingModel
    do {
        model = try modelType.createModel(configuration: configData)
    } catch let error as DecodingError {
        throw EmbedderError.configurationDecodingError(
            configurationURL.lastPathComponent, modelName, error)
    }

    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if let perLayerQuantization = baseConfig.perLayerQuantization {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                return perLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)

    return model
}

/// Load and return the model and tokenizer wrapped in a ``ModelContainer`` (provides
/// thread safe access).
public func loadModelContainer(
    hub: HubApi = HubApi(), configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> ModelContainer {
    let modelDirectory = try await prepareModelDirectory(
        hub: hub, configuration: configuration, progressHandler: progressHandler)
    return try await ModelContainer(
        hub: hub, modelDirectory: modelDirectory, configuration: configuration)
}
