// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Registry of `model_type` strings to creators of ``DFlash2DrafterModel``
/// instances. Empty at bootstrap: the drafter lives in MLXLLM, which
/// registers it through `DFlash2Registration.register()`.
public enum DFlash2DrafterTypeRegistry {
    public static let shared: ModelTypeRegistry<any DFlash2DrafterModel> = .init()
}

/// Registry of DFlash2 drafter ids to ``ModelConfiguration``.
public class DFlash2DrafterRegistry: AbstractModelRegistry, @unchecked Sendable {
    public static let shared = DFlash2DrafterRegistry(modelConfigurations: all())

    public static let qwen38_27B_dflash2 = ModelConfiguration(id: "z-lab/Qwen3.8-27B-DFlash2")

    private static func all() -> [ModelConfiguration] {
        [qwen38_27B_dflash2]
    }
}

/// Context for a loaded DFlash2 drafter. Drafters have no tokenizer, input
/// processor or chat template; they borrow the target's.
///
/// Not `Sendable`; cross-domain access goes through ``DFlash2DrafterContainer``.
public struct DFlash2DrafterContext {
    public var configuration: ModelConfiguration
    public var model: any DFlash2DrafterModel

    public init(configuration: ModelConfiguration, model: any DFlash2DrafterModel) {
        self.configuration = configuration
        self.model = model
    }
}

/// Sendable container for a ``DFlash2DrafterContext``, in the shape of
/// ``ModelContainer``.
public final class DFlash2DrafterContainer: Sendable {
    private let context: SerialAccessContainer<DFlash2DrafterContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public init(context: consuming DFlash2DrafterContext) {
        self.context = .init(context)
    }

    /// Perform an action on the context. Callers must eval any `MLXArray`
    /// before returning, as `MLXArray` is not `Sendable`.
    public func perform<R: Sendable>(
        _ action: @Sendable (DFlash2DrafterContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }
}

/// Loader for DFlash2 drafter checkpoints, in the shape of `LLMModelFactory`.
public final class DFlash2DrafterModelFactory: GenericModelFactory {
    public typealias ContextType = DFlash2DrafterContext
    public typealias ContainerType = DFlash2DrafterContainer

    public static let shared = DFlash2DrafterModelFactory(
        typeRegistry: DFlash2DrafterTypeRegistry.shared,
        modelRegistry: DFlash2DrafterRegistry.shared
    )

    public let typeRegistry: ModelTypeRegistry<any DFlash2DrafterModel>
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry<any DFlash2DrafterModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> DFlash2DrafterContext {
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

        let model: any DFlash2DrafterModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization,
            weightFileSelection: configuration.weightFileSelection
        )

        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: nil,
            defaultPrompt: ""
        )
        return DFlash2DrafterContext(configuration: modelConfig, model: model)
    }

    public func _wrap(_ context: DFlash2DrafterContext) -> DFlash2DrafterContainer {
        .init(context: context)
    }
}
