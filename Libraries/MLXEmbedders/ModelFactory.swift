// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXNN

private func create<C: Decodable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'bert', to functions that can instantiate the model from configuration.
public enum EmbedderTypeRegistry {

    public static let shared: ModelTypeRegistry<EmbeddingModel> = .init(creators: [
        "bert": create(BertConfiguration.self) { BertModel($0) },
        "roberta": create(BertConfiguration.self) { BertModel($0) },
        "xlm-roberta": create(BertConfiguration.self) { BertModel($0) },
        "distilbert": create(BertConfiguration.self) { BertModel($0) },

        "nomic_bert": create(NomicBertConfiguration.self) { NomicBertModel($0, pooler: false) },
        "qwen3": create(Qwen3Configuration.self) { Qwen3Model($0) },

        "gemma3": create(Gemma3Configuration.self) { EmbeddingGemma($0) },
        "gemma3_text": create(Gemma3Configuration.self) { EmbeddingGemma($0) },
        "gemma3n": create(Gemma3Configuration.self) { EmbeddingGemma($0) },
    ])

}

/// Registry of known embedder model configurations.
public class EmbedderRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = EmbedderRegistry(modelConfigurations: all())

    /// BGE Micro v2 (TaylorAI) - optimized for extremely low latency.
    public static let bge_micro = ModelConfiguration(id: "TaylorAI/bge-micro-v2")
    /// GTE Tiny - a small, efficient embedding model.
    public static let gte_tiny = ModelConfiguration(id: "TaylorAI/gte-tiny")
    /// MiniLM-L6 - the industry-standard small embedding model.
    public static let minilm_l6 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L6-v2")
    /// Snowflake Arctic Embed XS.
    public static let snowflake_xs = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-xs")
    /// MiniLM-L12 - a more accurate version of MiniLM.
    public static let minilm_l12 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L12-v2")
    /// BGE Small en v1.5.
    public static let bge_small = ModelConfiguration(id: "BAAI/bge-small-en-v1.5")
    /// Multilingual E5 Small - supports over 100 languages.
    public static let multilingual_e5_small = ModelConfiguration(
        id: "intfloat/multilingual-e5-small")
    /// BGE Base en v1.5.
    public static let bge_base = ModelConfiguration(id: "BAAI/bge-base-en-v1.5")
    /// Nomic Embed Text v1.
    public static let nomic_text_v1 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1")
    /// Nomic Embed Text v1.5 - supports Matryoshka embeddings.
    public static let nomic_text_v1_5 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1.5")
    /// BGE Large en v1.5.
    public static let bge_large = ModelConfiguration(id: "BAAI/bge-large-en-v1.5")
    /// Snowflake Arctic Embed L.
    public static let snowflake_lg = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-l")
    /// BGE-M3 - Multi-lingual, Multi-functional, Multi-granularity.
    public static let bge_m3 = ModelConfiguration(id: "BAAI/bge-m3")
    /// Mixedbread AI Large v1.
    public static let mixedbread_large = ModelConfiguration(
        id: "mixedbread-ai/mxbai-embed-large-v1")
    /// Qwen3 Embedding 0.6B - 4-bit quantized version.
    public static let qwen3_embedding = ModelConfiguration(
        id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

    private static func all() -> [ModelConfiguration] {
        [
            bge_micro,
            gte_tiny,
            minilm_l6,
            snowflake_xs,
            minilm_l12,
            bge_small,
            multilingual_e5_small,
            bge_base,
            nomic_text_v1,
            nomic_text_v1_5,
            bge_large,
            snowflake_lg,
            bge_m3,
            mixedbread_large,
            qwen3_embedding,
        ]
    }
}

/// Context of values that work together to provide an ``EmbeddingModel``.
///
/// This is created using a ``EmbedderModelFactory`` and often used
/// inside a ``EmbedderModelContainer``.
public struct EmbedderModelContext {
    public var configuration: ModelConfiguration
    public var model: any EmbeddingModel
    public var tokenizer: any Tokenizer
    public let pooling: Pooling

    public init(
        configuration: ModelConfiguration, model: any EmbeddingModel,
        tokenizer: any Tokenizer, pooling: Pooling
    ) {
        self.configuration = configuration
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
    }
}

/// Factory for creating new Embedder models.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let downloader: any Downloader
/// let tokenizerLoader: any TokenizerLoader
/// let modelId = "mlx-community/gemma-3-1b-it-qat-4bit"
/// let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
///     from: downloader, using: tokenizerLoader, configuration: .init(id: modelId),
///     progressHandler: logProgress(modelId)
/// )
/// ```
public final class EmbedderModelFactory: GenericModelFactory {

    public typealias ContextType = EmbedderModelContext
    public typealias ContainerType = EmbedderModelContainer

    public init(
        typeRegistry: ModelTypeRegistry<EmbeddingModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = EmbedderModelFactory(
        typeRegistry: EmbedderTypeRegistry.shared, modelRegistry: EmbedderRegistry.shared)

    /// registry of model type, e.g. configuration value `gemma3` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry<EmbeddingModel>

    /// registry of model id to configuration, e.g. `sentence-transformers/all-MiniLM-L6-v2`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> EmbedderModelContext {
        let modelDirectory = configuration.modelDirectory

        // Load config.json once and decode for both base config and model-specific config
        let configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: EmbeddingModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // Load tokenizer and weights in parallel
        async let tokenizerTask = tokenizerLoader.load(
            from: configuration.tokenizerDirectory)

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let tokenizer = try await tokenizerTask

        // Build a ModelConfiguration for the ModelContext
        let tokenizerSource: TokenizerSource? =
            configuration.tokenizerDirectory == modelDirectory
            ? nil
            : .directory(configuration.tokenizerDirectory)
        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: tokenizerSource,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: configuration.extraEOSTokens,
            eosTokenIds: configuration.eosTokenIds,
            toolCallFormat: configuration.toolCallFormat)

        let pooling = loadPooling(modelDirectory: modelDirectory, model: model)

        return .init(
            configuration: modelConfig, model: model,
            tokenizer: tokenizer, pooling: pooling
        )
    }

    public func _wrap(_ context: EmbedderModelContext) -> EmbedderModelContainer {
        .init(context: context)
    }
}
