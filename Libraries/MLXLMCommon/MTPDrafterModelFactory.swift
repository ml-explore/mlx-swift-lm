// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Registry of `model_type` strings (e.g. `"gemma4_assistant"`) to creator
/// closures that instantiate ``MTPDrafterModel`` instances from `config.json`
/// data.
///
/// Empty at bootstrap because drafter implementations live in MLXVLM —
/// importing them here would create a circular dependency. Downstream
/// modules call ``ModelTypeRegistry/registerModelType(_:creator:)`` on the
/// shared registry at app/import time. See `Gemma4AssistantRegistration`
/// in MLXVLM.
///
/// Note: ``ModelTypeRegistry`` is an `actor`; registration is `async`.
public enum MTPDrafterTypeRegistry {
    /// Shared registry. Empty until a downstream module registers a drafter
    /// type via `await MTPDrafterTypeRegistry.shared.registerModelType(...)`.
    public static let shared: ModelTypeRegistry<any TrainableMTPDrafterModel> = .init()
}

/// Registry of model id (e.g. `"mlx-community/gemma-4-31B-it-assistant-bf16"`)
/// to ``ModelConfiguration``. Drafters don't have prompts, so the
/// configurations omit `defaultPrompt`.
public class MTPDrafterRegistry: AbstractModelRegistry, @unchecked Sendable {
    public static let shared = MTPDrafterRegistry(modelConfigurations: all())

    public static let gemma4_26B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    )
    public static let gemma4_31B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-31B-it-assistant-bf16"
    )

    private static func all() -> [ModelConfiguration] {
        [gemma4_26B_assistant_bf16, gemma4_31B_assistant_bf16]
    }
}

/// Loader for ``MTPDrafterModel`` checkpoints. Mirrors `LLMModelFactory`
/// in shape, but produces an ``MTPDrafterContext`` (no tokenizer, no user
/// input processor, no message generator — drafters borrow their target's
/// tokenizer).
public final class MTPDrafterModelFactory: GenericModelFactory {
    public typealias ContextType = MTPDrafterContext

    public static let shared = MTPDrafterModelFactory(
        typeRegistry: MTPDrafterTypeRegistry.shared,
        modelRegistry: MTPDrafterRegistry.shared
    )

    public let typeRegistry: ModelTypeRegistry<any TrainableMTPDrafterModel>
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry<any TrainableMTPDrafterModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> MTPDrafterContext {
        let modelDirectory = configuration.modelDirectory
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

        let model: any TrainableMTPDrafterModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization
        )

        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: nil,
            defaultPrompt: ""
        )
        return MTPDrafterContext(configuration: modelConfig, model: model)
    }
}
