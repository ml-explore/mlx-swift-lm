// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM

/// Loads supported decision checkpoints behind one architecture-neutral container.
///
/// The factory reads the checkpoint's `config.json` and picks the matching
/// implementation. Callers do not provide prompts, answer tokens, or other model details.
public final class DecisionModelFactory: Sendable {
    /// Shared factory with the standard MLX model registries.
    public static let shared = DecisionModelFactory()

    /// Create a factory.
    public init() {}

    /// Download and load a decision model by its model identifier.
    ///
    /// - Parameters:
    ///   - downloader: Provider used to download model and tokenizer files.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - id: Model identifier, such as `LiquidAI/d1-3B`.
    ///   - revision: Model revision to download.
    ///   - useLatest: Whether to check for a newer cached revision.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral decision container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        id: String,
        revision: String = "main",
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DecisionContainer {
        try await loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: ModelConfiguration(id: id, revision: revision),
            useLatest: useLatest,
            progressHandler: progressHandler)
    }

    /// Download and load a decision model.
    ///
    /// - Parameters:
    ///   - downloader: Provider used to download model and tokenizer files.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - configuration: Model and tokenizer source configuration.
    ///   - useLatest: Whether to check for a newer cached revision.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral decision container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DecisionContainer {
        let resolved = try await resolve(
            configuration: configuration,
            from: downloader,
            useLatest: useLatest,
            progressHandler: progressHandler)
        return try await loadContainer(
            resolved: resolved, modelID: configuration.name, using: tokenizerLoader)
    }

    /// Load a decision model from a local model directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model configuration, tokenizer, and weights.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    /// - Returns: An architecture-neutral decision container.
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> DecisionContainer {
        let resolved = ResolvedModelConfiguration(directory: directory)
        return try await loadContainer(
            resolved: resolved, modelID: resolved.name, using: tokenizerLoader)
    }

    private func loadContainer(
        resolved: ResolvedModelConfiguration,
        modelID: String,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> DecisionContainer {
        let descriptor = try loadDescriptor(resolved)
        switch descriptor.architecture {
        case .d1:
            let (model, tokenizer) = try await loadD1(
                resolved: resolved, modelID: modelID, tokenizerLoader: tokenizerLoader)
            return DecisionModelContainer(model: model, tokenizer: tokenizer)
                .decisionContainer(modelID: modelID)
        case nil:
            throw DecisionError.unsupportedModel(descriptor.unsupportedMessage)
        }
    }

    /// d1-3B: the LFM2-VL model and processor settings that `VLMModelFactory` loads.
    func loadD1(
        resolved: ResolvedModelConfiguration, modelID: String, tokenizerLoader: any TokenizerLoader
    ) async throws -> (D1, any Tokenizer) {
        var resolved = resolved
        // Keeps the processor unwrapped.
        resolved.messageGenerator = nil
        let context = try await VLMModelFactory.shared._load(
            configuration: resolved, tokenizerLoader: tokenizerLoader)
        guard let base = context.model as? LFM2VL,
            let processor = context.processor as? LFM2VLProcessor
        else {
            throw DecisionError.unsupportedModel("\(modelID) did not load as an LFM2-VL model.")
        }
        return (D1(base, imageSettings: processor.imageSettings), context.tokenizer)
    }

    private func loadDescriptor(_ resolved: ResolvedModelConfiguration) throws
        -> DecisionDescriptor
    {
        let url = resolved.modelDirectory.appending(component: "config.json")
        do {
            return try JSONDecoder.json5().decode(
                DecisionDescriptor.self, from: Data(contentsOf: url))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                url.lastPathComponent, resolved.name, error)
        } catch {
            throw ModelFactoryError.configurationFileError(
                url.lastPathComponent, resolved.name, error)
        }
    }
}

enum DecisionArchitecture: Sendable, Equatable {
    /// LiquidAI d1: LFM2-VL weights with an answer-slot readout.
    case d1
}

/// The `config.json` fields that select a decision architecture.
struct DecisionDescriptor: Decodable, Sendable {
    var modelType: String
    var autoModel: String?

    /// The architecture to load, as mlx-vlm's `get_model_and_args` routes it.
    var architecture: DecisionArchitecture? {
        switch modelType {
        case "lfm2_vl", "lfm2-vl":
            autoModel?.hasSuffix("D1Model") == true ? .d1 : nil
        default:
            nil
        }
    }

    var unsupportedMessage: String {
        let autoModel = autoModel.map { " with AutoModel '\($0)'" } ?? ""
        return "Unsupported decision model type '\(modelType)'\(autoModel)."
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case autoMap = "auto_map"
    }

    private enum AutoMapKeys: String, CodingKey {
        case autoModel = "AutoModel"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        // auto_map entries can be lists; only a string AutoModel names a class.
        let autoMap = try? container.nestedContainer(keyedBy: AutoMapKeys.self, forKey: .autoMap)
        autoModel = try? autoMap?.decodeIfPresent(String.self, forKey: .autoModel)
    }
}
