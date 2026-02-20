// Copyright © 2024 Apple Inc.

import Foundation
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
        configuration: ResolvedModelConfiguration
    ) async throws -> ModelContext

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

extension ModelFactory {

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContext``.
    ///
    /// This resolves the configuration (downloading remote sources via the downloader)
    /// and then loads the model from local files.
    ///
    /// ## See Also
    /// - ``loadModel(from:configuration:useLatest:progressHandler:)``
    /// - ``loadModelContainer(from:configuration:useLatest:progressHandler:)``
    public func load(
        from downloader: any Downloader, configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        return try await _load(configuration: resolved)
    }

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContainer``.
    public func loadContainer(
        from downloader: any Downloader, configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        let context = try await _load(configuration: resolved)
        return ModelContainer(context: context)
    }

    /// Load a model from a local directory, producing a ``ModelContext``.
    ///
    /// No downloader is needed — the model and tokenizer are loaded from
    /// the given directory.
    public func load(
        from directory: URL
    ) async throws -> sending ModelContext {
        try await _load(configuration: .init(directory: directory))
    }

    /// Load a model from a local directory, producing a ``ModelContainer``.
    public func loadContainer(
        from directory: URL
    ) async throws -> ModelContainer {
        let context = try await _load(configuration: .init(directory: directory))
        return ModelContainer(context: context)
    }

}

/// Resolve a ``ModelConfiguration`` into a ``ResolvedModelConfiguration`` by
/// downloading remote sources via a ``Downloader``.
///
/// This handles the `.id` vs `.directory` switch for the model source and
/// resolves ``TokenizerSource`` for the tokenizer.
public func resolve(
    configuration: ModelConfiguration,
    from downloader: any Downloader,
    useLatest: Bool,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> ResolvedModelConfiguration {
    let modelDirectory: URL
    switch configuration.id {
    case .id(let id, let revision):
        modelDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: ["*.safetensors", "*.json"],
            useLatest: useLatest,
            progressHandler: progressHandler)
    case .directory(let directory):
        modelDirectory = directory
    }

    let tokenizerDirectory: URL
    switch configuration.tokenizerSource {
    case .id(let id):
        tokenizerDirectory = try await downloader.download(
            id: id, revision: nil,
            matching: ["*.json"],
            useLatest: useLatest,
            progressHandler: { _ in })
    case .directory(let directory):
        tokenizerDirectory = directory
    case nil:
        tokenizerDirectory = modelDirectory
    }

    return configuration.resolved(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory)
}

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader, configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }
}

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from downloader: any Downloader, configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, configuration: configuration, useLatest: useLatest,
            progressHandler: progressHandler)
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader, id: String, revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from downloader: any Downloader, id: String, revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - directory: directory of configuration and weights
/// - Returns: a ``ModelContext``
public func loadModel(
    from directory: URL
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(from: directory)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - directory: directory of configuration and weights
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from directory: URL
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(from: directory)
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
