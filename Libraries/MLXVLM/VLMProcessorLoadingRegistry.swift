// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// The model metadata available while resolving how to load a VLM processor.
///
/// External model packages can decode ``configurationData`` into their own model
/// configuration type without introducing a dependency from MLXVLM back to that package.
public struct VLMProcessorLoadingContext: Sendable {
    public let modelId: String
    public let modelType: String
    public let configurationData: Data

    public init(modelId: String, modelType: String, configurationData: Data) {
        self.modelId = modelId
        self.modelType = modelType
        self.configurationData = configurationData
    }
}

/// Fully resolved input for constructing a VLM processor.
public struct VLMProcessorConfiguration: Sendable {
    public let data: Data
    public let processorType: String

    public init(data: Data, processorType: String) {
        self.data = data
        self.processorType = processorType
    }
}

/// Supplies model-specific processor loading policy that is not present in a checkpoint.
///
/// Both hooks are optional. A fallback is consulted only when neither
/// `preprocessor_config.json` nor `processor_config.json` exists. Processor type resolvers are
/// consulted after a configuration has been selected, so they can supply missing metadata or
/// correct metadata shipped by a checkpoint.
public protocol VLMProcessorLoadingResolver: Sendable {
    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration?

    /// The processor type to use for this model, or `nil` to defer to another resolver or
    /// the checkpoint declaration. `declaredProcessorType` is `nil` when the selected
    /// processor configuration does not contain `processor_class`.
    func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String?
}

extension VLMProcessorLoadingResolver {
    public func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? { nil }

    public func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? { nil }
}

/// Ordered registry of processor-loading resolvers used by ``VLMModelFactory``.
///
/// Most recently registered wins for each hook independently. This lets a downstream
/// package choose one value while deferring the other to an earlier resolver.
public final class VLMProcessorLoadingRegistry: @unchecked Sendable {

    /// Shared instance, preloaded with this package's built-in loading rules.
    public static let shared = VLMProcessorLoadingRegistry(resolvers: [
        Qwen35ProcessorLoadingResolver(),
        ModelTypeProcessorResolver(processorTypes: [
            "mistral3": "Mistral3Processor",
            "gemma4_unified": "Gemma4UnifiedProcessor",
        ]),
    ])

    private let lock = NSLock()
    private var resolvers: [any VLMProcessorLoadingResolver]

    public init(resolvers: [any VLMProcessorLoadingResolver] = []) {
        self.resolvers = resolvers
    }

    /// Add a resolver. It takes precedence over resolvers registered before it.
    public func register(_ resolver: any VLMProcessorLoadingResolver) {
        lock.withLock { resolvers.append(resolver) }
    }

    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        for resolver in snapshot() {
            if let configuration = try resolver.fallbackProcessorConfiguration(for: context) {
                return configuration
            }
        }
        return nil
    }

    func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? {
        for resolver in snapshot() {
            if let processorType = try resolver.processorType(
                for: context, declaredProcessorType: declaredProcessorType)
            {
                return processorType
            }
        }
        return nil
    }

    /// Copy under the lock, then execute external code outside it. A resolver may be slow
    /// or may re-enter this registry, neither of which should block registration.
    private func snapshot() -> [any VLMProcessorLoadingResolver] {
        lock.withLock { resolvers.reversed() }
    }
}

/// A reusable processor-type rule keyed by `model_type` from `config.json`.
public struct ModelTypeProcessorResolver: VLMProcessorLoadingResolver {
    public let processorTypes: [String: String]

    public init(processorTypes: [String: String]) {
        self.processorTypes = processorTypes
    }

    public func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? {
        processorTypes[context.modelType]
    }
}

/// Reconstructs processor metadata omitted by Qwen3.5 checkpoints from their vision config.
public struct Qwen35ProcessorLoadingResolver: VLMProcessorLoadingResolver {
    public init() {}

    public func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard context.modelType == "qwen3_5" || context.modelType == "qwen3_5_moe" else {
            return nil
        }

        let modelConfiguration = try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: context.configurationData)
        let processorConfiguration = Qwen3VLProcessorConfiguration(
            qwen35VisionConfiguration: modelConfiguration.visionConfiguration)
        return VLMProcessorConfiguration(
            data: try JSONEncoder().encode(processorConfiguration),
            processorType: "Qwen3VLProcessor")
    }
}
