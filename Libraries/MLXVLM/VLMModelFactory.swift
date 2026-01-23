// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public enum VLMError: LocalizedError, Equatable {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case singleVideoAllowed
    case singleMediaTypeAllowed
    case imageProcessingFailure(String)
    case processing(String)
    case noVideoTrackFound
    case videoNotDecodable

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return String(localized: "An image is required for this operation.")
        case .maskRequired:
            return String(localized: "An image mask is required for this operation.")
        case .singleImageAllowed:
            return String(localized: "Only a single image is allowed for this operation.")
        case .singleVideoAllowed:
            return String(localized: "Only a single video is allowed for this operation.")
        case .singleMediaTypeAllowed:
            return String(
                localized:
                    "Only a single media type (image or video) is allowed for this operation.")
        case .imageProcessingFailure(let details):
            return String(localized: "Failed to process the image: \(details)")
        case .processing(let details):
            return String(localized: "Processing error: \(details)")
        case .noVideoTrackFound:
            return String(localized: "Video file has no video tracks.")
        case .videoNotDecodable:
            return String(localized: "Video file not decodable.")
        }
    }
}

public struct BaseProcessorConfiguration: Codable, Sendable {
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

private func create<C: Codable, P>(
    _ configurationType: C.Type,
    _ processorInit:
        @escaping (
            C,
            any Tokenizer
        ) -> P
) -> (Data, any Tokenizer) throws -> P {
    { data, tokenizer in
        let configuration = try JSONDecoder().decode(C.self, from: data)
        return processorInit(configuration, tokenizer)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
public enum VLMTypeRegistry {

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(creators: [
        "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
        "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
        "qwen2_5_vl": create(Qwen25VLConfiguration.self, Qwen25VL.init),
        "qwen3_vl": create(Qwen3VLConfiguration.self, Qwen3VL.init),
        "idefics3": create(Idefics3Configuration.self, Idefics3.init),
        "gemma3": create(Gemma3Configuration.self, Gemma3.init),
        "smolvlm": create(SmolVLM2Configuration.self, SmolVLM2.init),
        // TODO: see if we can make it work with fastvlm rather than llava_qwen2
        "fastvlm": create(FastVLMConfiguration.self, FastVLM.init),
        "llava_qwen2": create(FastVLMConfiguration.self, FastVLM.init),
        "pixtral": create(PixtralConfiguration.self, PixtralVLM.init),
        "mistral3": create(Mistral3VLMConfiguration.self, Mistral3VLM.init),
        "lfm2_vl": create(LFM2VLConfiguration.self, LFM2VL.init),
        "lfm2-vl": create(LFM2VLConfiguration.self, LFM2VL.init),
    ])
}

public enum VLMProcessorTypeRegistry {

    /// Shared instance with default processor types.
    public static let shared: ProcessorTypeRegistry = .init(creators: [
        "PaliGemmaProcessor": create(
            PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
        "Qwen2VLProcessor": create(
            Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
        "Qwen2_5_VLProcessor": create(
            Qwen25VLProcessorConfiguration.self, Qwen25VLProcessor.init),
        "Qwen3VLProcessor": create(
            Qwen3VLProcessorConfiguration.self, Qwen3VLProcessor.init),
        "Idefics3Processor": create(
            Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
        "Gemma3Processor": create(
            Gemma3ProcessorConfiguration.self, Gemma3Processor.init),
        "SmolVLMProcessor": create(
            SmolVLMProcessorConfiguration.self, SmolVLMProcessor.init),
        "FastVLMProcessor": create(
            FastVLMProcessorConfiguration.self, FastVLMProcessor.init),
        "PixtralProcessor": create(
            PixtralProcessorConfiguration.self, PixtralProcessor.init),
        "Mistral3Processor": create(
            Mistral3VLMProcessorConfiguration.self, Mistral3VLMProcessor.init),
        "Lfm2VlProcessor": create(
            LFM2VLProcessorConfiguration.self, LFM2VLProcessor.init),
    ])
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class VLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared: VLMRegistry = .init(modelConfigurations: all())

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2_5VL3BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct4Bit = ModelConfiguration(
        id: "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct8Bit = ModelConfiguration(
        id: "mlx-community/Qwen3-VL-4B-Instruct-8bit",
        defaultPrompt: "Write a haiku about Swift programming"
    )

    static public let smolvlminstruct4bit = ModelConfiguration(
        id: "mlx-community/SmolVLM-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let lfm2_5_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2.5-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let lfm2_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let mistral3_3B_Instruct_4bit = ModelConfiguration(
        id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        defaultPrompt: ""
    )

    static public let gemma3_4B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-4b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_12B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-12b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_27B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-27b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let smolvlm = ModelConfiguration(
        id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
        defaultPrompt:
            "What is the main action or notable event happening in this segment? Describe it in one brief sentence."
    )

    static public let fastvlm = ModelConfiguration(
        id: "mlx-community/FastVLM-0.5B-bf16",
        defaultPrompt: "Describe this image in detail."
    )

    static public func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
            qwen2_5VL3BInstruct4Bit,
            qwen3VL4BInstruct4Bit,
            qwen3VL4BInstruct8Bit,
            smolvlminstruct4bit,
            gemma3_4B_qat_4bit,
            gemma3_12B_qat_4bit,
            gemma3_27B_qat_4bit,
            smolvlm,
            fastvlm,
        ]
    }

}

@available(*, deprecated, renamed: "VLMRegistry", message: "Please use VLMRegistry directly.")
public typealias ModelRegistry = VLMRegistry

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await VLMModelFactory.shared.loadContainer(
///     configuration: VLMRegistry.paligemma3bMix4488bit)
/// ```
public final class VLMModelFactory: ModelFactory {

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = VLMModelFactory(
        typeRegistry: VLMTypeRegistry.shared, processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry: ProcessorTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

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
            baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // Load EOS token IDs from config.json, with optional override from generation_config.json
        var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        if let generationData = try? Data(contentsOf: generationConfigURL),
            let generationConfig = try? JSONDecoder().decode(
                GenerationConfigFile.self, from: generationData),
            let genEosIds = generationConfig.eosTokenIds?.values
        {
            eosTokenIds = Set(genEosIds)  // Override per Python mlx-lm behavior
        }

        // Create mutable configuration with loaded EOS token IDs
        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds

        // Load tokenizer, processor config, and weights in parallel using async let.
        // Note: loadProcessorConfig does synchronous I/O but is marked async to enable
        // parallel scheduling. This may briefly block a cooperative thread pool thread,
        // but the config file is small and model loading is not a high-concurrency path.
        async let tokenizerTask = loadTokenizer(configuration: configuration, hub: hub)
        async let processorConfigTask = loadProcessorConfig(from: modelDirectory)

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let tokenizer = try await tokenizerTask
        let processorConfigData: Data
        let baseProcessorConfig: BaseProcessorConfiguration
        do {
            (processorConfigData, baseProcessorConfig) = try await processorConfigTask
        } catch let error as ProcessorConfigError {
            if let decodingError = error.underlying as? DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    error.filename, configuration.name, decodingError)
            }
            throw ModelFactoryError.configurationFileError(
                error.filename, configuration.name, error.underlying)
        }

        // Override processor type based on model type for models that need special handling
        // Mistral3 models ship with "PixtralProcessor" in their config but need Mistral3Processor
        // to handle spatial merging correctly
        let processorTypeOverrides: [String: String] = [
            "mistral3": "Mistral3Processor"
        ]
        let processorType =
            processorTypeOverrides[baseConfig.modelType] ?? baseProcessorConfig.processorClass

        let processor = try await processorRegistry.createModel(
            configuration: processorConfigData,
            processorType: processorType, tokenizer: tokenizer)

        return .init(
            configuration: mutableConfiguration, model: model, processor: processor,
            tokenizer: tokenizer)
    }

}

/// Error wrapper that includes the filename for better error messages.
private struct ProcessorConfigError: Error {
    let filename: String
    let underlying: Error
}

/// Loads processor configuration, preferring preprocessor_config.json over processor_config.json.
/// Marked async to enable parallel scheduling via async let, though the underlying I/O is synchronous.
/// Throws ProcessorConfigError wrapping any underlying error with the filename.
private func loadProcessorConfig(from modelDirectory: URL) async throws -> (
    Data, BaseProcessorConfiguration
) {
    let processorConfigURL = modelDirectory.appending(component: "processor_config.json")
    let preprocessorConfigURL = modelDirectory.appending(component: "preprocessor_config.json")
    let url =
        FileManager.default.fileExists(atPath: preprocessorConfigURL.path)
        ? preprocessorConfigURL
        : processorConfigURL
    do {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(BaseProcessorConfiguration.self, from: data)
        return (data, config)
    } catch {
        throw ProcessorConfigError(filename: url.lastPathComponent, underlying: error)
    }
}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
        VLMModelFactory.shared
    }
}
