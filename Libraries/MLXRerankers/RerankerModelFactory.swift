// Copyright © 2026 Apple Inc.

import Foundation
import MLXEmbedders
import MLXLLM
import MLXLMCommon

/// Loads supported reranker checkpoints behind one architecture-neutral container.
///
/// The factory inspects `config.json` and selects the appropriate encoder, causal, or
/// listwise implementation. Callers do not provide prompt templates, token IDs, padding
/// IDs, classifier policies, or model architecture details.
public final class RerankerModelFactory: Sendable {
    /// Shared factory with the standard MLX model registries.
    public static let shared = RerankerModelFactory()

    public init() {}

    /// Download and load a reranker by its model identifier.
    ///
    /// The checkpoint's `config.json` determines whether the factory loads an encoder,
    /// causal, or listwise reranker.
    ///
    /// - Parameters:
    ///   - downloader: Provider used to download model and tokenizer files.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - id: Model identifier, such as `mlx-community/Qwen3-Reranker-0.6B-4bit`.
    ///   - revision: Model revision to download.
    ///   - useLatest: Whether to check for a newer cached revision.
    ///   - allowUnverifiedModel: Allow a custom checkpoint whose identifier does not declare
    ///     that it is a reranker. Keep this `false` for downloaded third-party models.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        id: String,
        revision: String = "main",
        useLatest: Bool = false,
        allowUnverifiedModel: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> RerankerContainer {
        try await loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: ModelConfiguration(id: id, revision: revision),
            useLatest: useLatest,
            allowUnverifiedModel: allowUnverifiedModel,
            progressHandler: progressHandler)
    }

    /// Download and load a reranker model.
    ///
    /// - Parameters:
    ///   - downloader: Provider used to download model and tokenizer files.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - configuration: Model and tokenizer source configuration.
    ///   - useLatest: Whether to check for a newer cached revision.
    ///   - allowUnverifiedModel: Allow a custom checkpoint whose identifier does not declare
    ///     that it is a reranker. Keep this `false` for downloaded third-party models.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        allowUnverifiedModel: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> RerankerContainer {
        let resolved = try await resolve(
            configuration: configuration,
            from: downloader,
            useLatest: useLatest,
            progressHandler: progressHandler)
        return try await loadContainer(
            resolved: resolved,
            modelID: configuration.name,
            allowUnverifiedModel: allowUnverifiedModel,
            using: tokenizerLoader)
    }

    /// Load a reranker from a local model directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model configuration, tokenizer, and weights.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - allowUnverifiedModel: Allow a custom checkpoint whose directory name does not
    ///     declare that it is a reranker.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        allowUnverifiedModel: Bool = false
    ) async throws -> RerankerContainer {
        let resolved = ResolvedModelConfiguration(directory: directory)
        return try await loadContainer(
            resolved: resolved,
            modelID: resolved.name,
            allowUnverifiedModel: allowUnverifiedModel,
            using: tokenizerLoader)
    }

    private func loadContainer(
        resolved: ResolvedModelConfiguration,
        modelID: String,
        allowUnverifiedModel: Bool,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> RerankerContainer {
        let descriptor = try loadDescriptor(from: resolved.modelDirectory)
        let architecture = try descriptor.architecture(
            modelID: modelID, allowUnverifiedModel: allowUnverifiedModel)

        switch architecture {
        case .jina:
            let context = try await LLMModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: tokenizerLoader)
            let container = ModelContainer(context: context)
            return RerankerContainer(
                modelID: modelID,
                scoreKind: .cosineSimilarity
            ) { query, documents, instruction, options in
                try await container.listwiseRerankerScores(
                    query: query,
                    documents: documents,
                    instruction: instruction,
                    maxInputTokens: descriptor.maxPositionEmbeddings ?? 131_072,
                    maximumDocuments: 64,
                    options: options)
            }
        case .encoder:
            let context = try await EmbedderModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: tokenizerLoader)
            let container = EmbedderModelContainer(context: context)
            guard let scoreKind = await container.rerankerScoreKind else {
                throw RerankerError.unsupportedModel(
                    "\(modelID) did not load an encoder reranker head.")
            }
            return RerankerContainer(modelID: modelID, scoreKind: scoreKind) {
                query, documents, _, options in
                try await container.rerankerScores(
                    query: query, documents: documents, options: options)
            }
        case .qwen3:
            let context = try await LLMModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: tokenizerLoader)
            let container = ModelContainer(context: context)
            return RerankerContainer(modelID: modelID, scoreKind: .normalizedRelevance) {
                query, documents, instruction, options in
                try await container.causalRerankerScores(
                    query: query,
                    documents: documents,
                    instruction: instruction,
                    maxInputTokens: min(descriptor.maxPositionEmbeddings ?? 8_192, 8_192),
                    options: options)
            }
        case nil:
            throw RerankerError.unsupportedModel(
                "Unsupported reranker model type '\(descriptor.modelType)' with architectures \(descriptor.architectures)."
            )
        }
    }

    private func loadDescriptor(from directory: URL) throws -> RerankerDescriptor {
        let url = directory.appending(component: "config.json")
        do {
            return try JSONDecoder.json5().decode(
                RerankerDescriptor.self, from: Data(contentsOf: url))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                url.lastPathComponent,
                directory.lastPathComponent,
                error)
        } catch {
            throw ModelFactoryError.configurationFileError(
                url.lastPathComponent,
                directory.lastPathComponent,
                error)
        }
    }
}

package enum RerankerArchitecture: Sendable, Equatable {
    case encoder
    case qwen3
    case jina
}

package struct RerankerDescriptor: Decodable, Sendable {
    var modelType: String
    var architectures: [String]
    var maxPositionEmbeddings: Int?

    package func architecture(
        modelID: String,
        allowUnverifiedModel: Bool = false
    ) throws -> RerankerArchitecture? {
        if architectures.contains("JinaForRanking") {
            return .jina
        }
        let declaresReranker = modelID.localizedCaseInsensitiveContains("rerank")
        let verified = allowUnverifiedModel || declaresReranker
        if ["bert", "roberta", "xlm-roberta"].contains(modelType),
            architectures.contains(where: { $0.contains("ForSequenceClassification") })
        {
            guard verified else {
                throw RerankerError.unsupportedModel(
                    "Sequence-classification checkpoint '\(modelID)' is not identified as a reranker. Set allowUnverifiedModel only for a trusted custom reranker."
                )
            }
            return .encoder
        }
        if modelType == "qwen3", architectures.contains("Qwen3ForCausalLM") {
            guard verified else {
                throw RerankerError.unsupportedModel(
                    "Qwen3 checkpoint '\(modelID)' is not identified as a reranker. Set allowUnverifiedModel only for a trusted custom reranker."
                )
            }
            return .qwen3
        }
        return nil
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
    }
}
