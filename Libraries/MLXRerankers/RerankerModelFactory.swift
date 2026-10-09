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
    ///   - family: The reranking protocol of a renamed or fine-tuned checkpoint. When `nil`, the
    ///     factory identifies the family from the checkpoint.
    ///   - allowUnverifiedModel: Assert that a trusted custom checkpoint uses the official
    ///     Qwen3-Reranker protocol, or the Jina protocol its configuration declares. This does
    ///     not bypass incompatible scoring metadata. Keep this `false` for third-party models.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        id: String,
        revision: String = "main",
        useLatest: Bool = false,
        family: RerankerFamily? = nil,
        allowUnverifiedModel: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> RerankerContainer {
        try await loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: ModelConfiguration(id: id, revision: revision),
            useLatest: useLatest,
            family: family,
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
    ///   - family: The reranking protocol of a renamed or fine-tuned checkpoint. When `nil`, the
    ///     factory identifies the family from the checkpoint.
    ///   - allowUnverifiedModel: Assert that a trusted custom checkpoint uses the official
    ///     Qwen3-Reranker protocol, or the Jina protocol its configuration declares. This does
    ///     not bypass incompatible scoring metadata. Keep this `false` for third-party models.
    ///   - progressHandler: Callback that receives model download progress.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        family: RerankerFamily? = nil,
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
            family: family,
            allowUnverifiedModel: allowUnverifiedModel,
            using: tokenizerLoader)
    }

    /// Load a reranker from a local model directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model configuration, tokenizer, and weights.
    ///   - tokenizerLoader: Loader used to construct the model tokenizer.
    ///   - family: The reranking protocol of a checkpoint with an opaque directory name.
    ///   - allowUnverifiedModel: Assert the official Qwen3-Reranker protocol, or the Jina
    ///     protocol its configuration declares, for a trusted custom checkpoint.
    /// - Returns: An architecture-neutral reranker container.
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        family: RerankerFamily? = nil,
        allowUnverifiedModel: Bool = false
    ) async throws -> RerankerContainer {
        let resolved = ResolvedModelConfiguration(directory: directory)
        return try await loadContainer(
            resolved: resolved,
            modelID: resolved.name,
            family: family,
            allowUnverifiedModel: allowUnverifiedModel,
            using: tokenizerLoader)
    }

    private func loadContainer(
        resolved: ResolvedModelConfiguration,
        modelID: String,
        family: RerankerFamily?,
        allowUnverifiedModel: Bool,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> RerankerContainer {
        let descriptor = try loadDescriptor(from: resolved.modelDirectory)
        let architecture = try descriptor.architecture(
            modelID: modelID, family: family, allowUnverifiedModel: allowUnverifiedModel)

        switch architecture {
        case .jina(let family):
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
                    family: family,
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
        case .causal(let family):
            let metadata = try loadCausalMetadata(from: resolved.modelDirectory)
            // Reject contradictory metadata before loading weights.
            try family.declaredScorePolicy(metadata)?.validate(
                vocabularySize: descriptor.vocabularySize)
            let context = try await LLMModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: tokenizerLoader)
            let container = ModelContainer(context: context)
            let scorePolicy = try family.scorePolicy(
                metadata: metadata, tokenizer: context.tokenizer,
                vocabularySize: descriptor.vocabularySize)
            let maxInputTokens = min(
                descriptor.maxPositionEmbeddings ?? family.maxInputTokens, family.maxInputTokens)
            return RerankerContainer(modelID: modelID, scoreKind: family.scoreKind) {
                query, documents, instruction, options in
                try await container.causalRerankerScores(
                    query: query,
                    documents: documents,
                    instruction: instruction,
                    maxInputTokens: maxInputTokens,
                    family: family,
                    scorePolicy: scorePolicy,
                    options: options)
            }
        case nil:
            throw RerankerError.unsupportedModel(
                "Unsupported reranker model type '\(descriptor.modelType)' with architectures \(descriptor.architectures)."
            )
        }
    }

    private func loadCausalMetadata(from directory: URL) throws -> CausalRerankerMetadata? {
        let file = "1_LogitScore/config.json"
        guard FileManager.default.fileExists(atPath: directory.appending(path: file).path) else {
            return nil
        }
        return try decode(CausalRerankerMetadata.self, file: file, from: directory)
    }

    private func loadDescriptor(from directory: URL) throws -> RerankerDescriptor {
        try decode(RerankerDescriptor.self, file: "config.json", from: directory)
    }

    private func decode<T: Decodable>(_ type: T.Type, file: String, from directory: URL) throws -> T
    {
        let url = directory.appending(path: file)
        do {
            return try JSONDecoder.json5().decode(
                type, from: Data(contentsOf: url))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                file,
                directory.lastPathComponent,
                error)
        } catch {
            throw ModelFactoryError.configurationFileError(
                file,
                directory.lastPathComponent,
                error)
        }
    }
}

package enum RerankerArchitecture: Sendable, Equatable {
    case encoder
    case causal(CausalRerankerFamily)
    case jina(JinaRerankerFamily)
}

package struct RerankerDescriptor: Decodable, Sendable {
    var modelType: String
    var architectures: [String]
    var maxPositionEmbeddings: Int?
    var vocabularySize: Int?
    var usesSlidingWindow: Bool

    package func architecture(
        modelID: String,
        family: RerankerFamily? = nil,
        allowUnverifiedModel: Bool = false
    ) throws -> RerankerArchitecture? {
        let isQwen3 = modelType == "qwen3"
        let isJina = isQwen3 && architectures.contains("JinaForRanking")
        let isCausal = isQwen3 && architectures.contains("Qwen3ForCausalLM")
        let isEncoder =
            ["bert", "roberta", "xlm-roberta"].contains(modelType)
            && architectures.contains(where: { $0.contains("ForSequenceClassification") })

        if let family {
            switch family.architecture {
            case .jina where isJina, .causal where isCausal:
                return family.architecture
            default:
                throw RerankerError.unsupportedModel(
                    "\(family) does not match '\(modelType)' with architectures \(architectures)."
                )
            }
        }

        let identified = RerankerFamily(checkpointName: modelID)
        if isJina {
            if let identified, case .jina = identified.architecture {
                return identified.architecture
            }
            guard allowUnverifiedModel else {
                throw RerankerError.unsupportedModel(
                    "Unknown Jina reranking protocol for '\(modelID)'. Pass its family to opt in.")
            }
            // Only v3.5 interleaves sliding-window layers.
            return .jina(usesSlidingWindow ? .v35 : .v3)
        }
        if isEncoder {
            guard allowUnverifiedModel || modelID.localizedCaseInsensitiveContains("rerank") else {
                throw RerankerError.unsupportedModel(
                    "Sequence-classification checkpoint '\(modelID)' is not identified as a reranker. Set allowUnverifiedModel only for a trusted custom reranker."
                )
            }
            return .encoder
        }
        if isCausal {
            if let identified, case .causal = identified.architecture {
                return identified.architecture
            }
            guard allowUnverifiedModel else {
                throw RerankerError.unsupportedModel(
                    "Unknown causal reranking protocol for '\(modelID)'. Pass its family to opt in, or set allowUnverifiedModel for a custom checkpoint using the official Qwen3-Reranker protocol."
                )
            }
            return .causal(.qwen3)
        }
        return nil
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case maxPositionEmbeddings = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case useSlidingWindow = "use_sliding_window"
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
        usesSlidingWindow =
            try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
    }
}
