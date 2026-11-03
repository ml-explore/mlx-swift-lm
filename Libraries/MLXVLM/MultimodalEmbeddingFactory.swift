// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Loads an embedding from resolved checkpoint and tokenizer directories.
public typealias MultimodalEmbeddingLoader =
    @Sendable (ResolvedModelConfiguration, any TokenizerLoader) async throws ->
    any MultimodalEmbedding

/// Registry of checkpoint `model_type` values to multimodal embedding loaders.
public enum MultimodalEmbeddingTypeRegistry {

    public static let shared = ModelTypeRegistry<MultimodalEmbeddingLoader>(creators: [
        "embedding_gemma2": { _ in embeddingGemma2 },
        "embedding_gemma2_text": { _ in embeddingGemma2 },
    ])

    private static let embeddingGemma2: MultimodalEmbeddingLoader = {
        try await EmbeddingGemma2Embedding(configuration: $0, tokenizerLoader: $1)
    }
}

/// Registry of known multimodal embedding checkpoints.
public class MultimodalEmbeddingRegistry: AbstractModelRegistry, @unchecked Sendable {

    public static let shared = MultimodalEmbeddingRegistry(modelConfigurations: [embeddinggemma2])

    /// EmbeddingGemma 2 with text, vision, and audio weights in bfloat16.
    public static let embeddinggemma2 = ModelConfiguration(
        id: "mlx-community/embeddinggemma-2-bf16")
}

/// Loads a ``MultimodalEmbedding`` by its checkpoint's `model_type`.
///
/// Use the shared instance or supply registries for application-defined models.
/// Media encoders load on first use, so checkpoint files must stay readable.
///
/// ```swift
/// let embedding = try await MultimodalEmbeddingFactory.shared.loadContainer(
///     from: directory, using: loader)
/// let vectors = try await embedding.embed(
///     [.init(text: "Aurora over Balestrino", images: [.url(photo)])], task: .document)
/// ```
public final class MultimodalEmbeddingFactory: GenericModelFactory {

    public typealias ContextType = any MultimodalEmbedding
    public typealias ContainerType = any MultimodalEmbedding

    public static let shared = MultimodalEmbeddingFactory(
        typeRegistry: MultimodalEmbeddingTypeRegistry.shared,
        modelRegistry: MultimodalEmbeddingRegistry.shared)

    public let typeRegistry: ModelTypeRegistry<MultimodalEmbeddingLoader>
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry<MultimodalEmbeddingLoader>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> any MultimodalEmbedding {
        let url = configuration.modelDirectory.appending(component: "config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ModelFactoryError.configurationFileError(
                url.lastPathComponent, configuration.name, error)
        }

        let loader: MultimodalEmbeddingLoader
        do {
            let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
            loader = try await typeRegistry.createModel(
                configuration: data, modelType: base.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                url.lastPathComponent, configuration.name, error)
        }
        return try await loader(configuration, tokenizerLoader)
    }

    public func _wrap(_ context: any MultimodalEmbedding) -> any MultimodalEmbedding {
        // The embedding already provides Sendable access to its model.
        context
    }

    /// Loads a local checkpoint using the shared registries.
    public static func load(
        modelDirectory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> any MultimodalEmbedding {
        try await shared.load(from: modelDirectory, using: tokenizerLoader)
    }
}
