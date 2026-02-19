// Copyright © 2024 Apple Inc.

import Foundation
import HuggingFace
import Tokenizers

public enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
    case configurationFileError(String, String, Error)
    case configurationDecodingError(String, String, DecodingError)
    case noModelFactoryAvailable

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type):
            return "Unsupported processor type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .noModelFactoryAvailable:
            return "No model factory available via ModelFactoryRegistry"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
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

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is created by ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This contains the following:
///
/// - ``ModelConfiguration`` -- identifier for the model
/// - ``LanguageModel`` -- the model itself, see ``generate(input:cache:parameters:context:)``
/// - ``UserInputProcessor`` -- can convert ``UserInput`` into ``LMInput``
/// - `Tokenizer` -- the tokenizer used by ``UserInputProcessor``
///
/// See also ``ModelFactory/loadContainer(from:configuration:progressHandler:)`` and
/// ``ModelContainer``.
public struct ModelContext {
    public var configuration: ModelConfiguration
    public var model: any LanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer

    public init(
        configuration: ModelConfiguration, model: any LanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}

/// Protocol for code that can load models.
///
/// ## See Also
/// - ``loadModel(from:id:progressHandler:)``
/// - ``loadModel(from:)-ModelContext``
/// - ``loadModelContainer(from:id:progressHandler:)``
/// - ``loadModelContainer(from:)-ModelContainer``
public protocol ModelFactory: Sendable {

    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        from hub: HubClient, configuration: ModelConfiguration,
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext

    func _loadContainer(
        from hub: HubClient, configuration: ModelConfiguration,
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer

}

extension ModelFactory {

    /// Resolve a model identifier, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit", into
    /// a ``ModelConfiguration``.
    ///
    /// This will either create a new (mostly unconfigured) ``ModelConfiguration`` or
    /// return a registered instance that matches the id.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    public func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    /// Returns true if ``modelRegistry`` contains a model with the id. Otherwise, false.
    public func contains(id: String) -> Bool {
        modelRegistry.contains(id: id)
    }

}

/// Default HubClient instance for downloading models.
public let defaultHubClient: HubClient = .default

extension ModelFactory {

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContext``.
    ///
    /// This method returns a ``ModelContext``. See also
    /// ``loadContainer(from:configuration:progressHandler:)`` for a method that
    /// returns a ``ModelContainer``.
    ///
    /// ## See Also
    /// - ``loadModel(from:id:progressHandler:)``
    /// - ``loadModelContainer(from:id:progressHandler:)``
    public func load(
        from hub: HubClient = defaultHubClient, configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        try await _load(
            from: hub, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContainer``.
    public func loadContainer(
        from hub: HubClient = defaultHubClient, configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        try await _loadContainer(
            from: hub, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }

    public func _loadContainer(
        from hub: HubClient = defaultHubClient, configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await _load(
            from: hub, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
        return ModelContainer(context: context)
    }

}

/// Load a model given a ``ModelConfiguration``.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenzier without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - hub: optional HubClient -- by default uses ``defaultHubClient``
///   - configuration: a ``ModelConfiguration``
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from hub: HubClient = defaultHubClient, configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: hub, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }
}

/// Load a model given a ``ModelConfiguration``.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - hub: optional HubClient -- by default uses ``defaultHubClient``
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the Hub for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from hub: HubClient = defaultHubClient, configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: hub, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }
}

/// Load a model given a Hugging Face identifier.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenzier without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - hub: optional HubClient -- by default uses ``defaultHubClient``
///   - id: Hugging Face model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from hub: HubClient = defaultHubClient, id: String, revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: hub, configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a Hugging Face identifier.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - hub: optional HubClient -- by default uses ``defaultHubClient``
///   - id: Hugging Face model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - useLatest: when true, always checks the Hub for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from hub: HubClient = defaultHubClient, id: String, revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: hub, configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a directory of configuration and weights.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenizer without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - directory: directory of configuration and weights
/// - Returns: a ``ModelContext``
public func loadModel(
    from directory: URL
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: .default, configuration: .init(directory: directory),
            progressHandler: { _ in })
    }
}

/// Load a model given a directory of configuration and weights.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - directory: directory of configuration and weights
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from directory: URL
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: .default, configuration: .init(directory: directory),
            progressHandler: { _ in })
    }
}

private func load<R>(loader: (ModelFactory) async throws -> sending R) async throws -> sending R {
    let factories = ModelFactoryRegistry.shared.modelFactories()
    var lastError: Error?
    for factory in factories {
        do {
            let model = try await loader(factory)
            return model
        } catch {
            lastError = error
        }
    }

    if let lastError {
        throw lastError
    } else {
        throw ModelFactoryError.noModelFactoryAvailable
    }
}

/// Protocol for types that can provide ModelFactory instances.
///
/// Not used directly.
///
/// This is used internally to provide dynamic lookup of a trampoline -- this lets
/// API in MLXLMCommon use code present in MLXLLM:
///
/// ```swift
/// public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
///     public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
///         LLMModelFactory.shared
///     }
/// }
/// ```
///
/// That is looked up dynamically with:
///
/// ```swift
/// {
///     (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
///         .modelFactory()
/// }
/// ```
///
/// ## See Also
/// - ``ModelFactoryRegistry``
public protocol ModelFactoryTrampoline {
    static func modelFactory() -> ModelFactory?
}

/// Registry of ``ModelFactory`` trampolines.
///
/// This allows ``loadModel(from:id:progressHandler:)`` to use any ``ModelFactory`` instances
/// available but be defined in the `LLMCommon` layer.  This is not typically used directly -- it is
/// called via ``loadModel(from:id:progressHandler:)``:
///
/// ```swift
/// let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
/// ```
///
/// ## See Also
/// - ``loadModel(from:id:progressHandler:)``
/// - ``loadModel(from:)-ModelContext``
/// - ``loadModelContainer(from:id:progressHandler:)``
/// - ``loadModelContainer(from:)-ModelContainer``
final public class ModelFactoryRegistry: @unchecked Sendable {
    public static let shared = ModelFactoryRegistry()

    private let lock = NSLock()
    private var trampolines: [() -> ModelFactory?]

    private init() {
        self.trampolines = [
            {
                (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
            {
                (NSClassFromString("MLXLLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
        ]
    }

    public func addTrampoline(_ trampoline: @escaping () -> ModelFactory?) {
        lock.withLock {
            trampolines.append(trampoline)
        }
    }

    public func modelFactories() -> [ModelFactory] {
        lock.withLock {
            trampolines.compactMap { $0() }
        }
    }
}
