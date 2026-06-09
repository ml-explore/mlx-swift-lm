// Copyright © 2024 Apple Inc.

import Foundation
import MLXNN

/// File patterns required to resolve a tokenizer without downloading model weights.
package let tokenizerDownloadPatterns = ["*.json", "*.jinja"]
package let modelDownloadPatterns = ["*.safetensors"] + tokenizerDownloadPatterns

public enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
    case configurationFileError(String, String, Error)
    case configurationDecodingError(String, String, DecodingError)
    case invalidConfiguration(String)
    case noModelFactoryAvailable

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type):
            return "Unsupported processor type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .invalidConfiguration(let message):
            return "Invalid model configuration: \(message)"
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

public protocol ModelConfigurationValidating {
    func validateModelConfiguration() throws
}

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is `Sendable` and is created by ``loadModel(from:using:configuration:useLatest:progressHandler:)``
/// or ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This contains the following:
///
/// - ``ModelConfiguration``: identifier for the model
/// - ``LanguageModel``: the model itself (wrapped in a `MaterializedModule`), see ``generate(input:cache:parameters:context:wiredMemoryTicket:tools:)``
/// - ``UserInputProcessor``: can convert ``UserInput`` into ``LMInput``
/// - `Tokenizer` -- the tokenizer used by ``UserInputProcessor``
///
/// See also the deprecated ``GenericModelFactory/loadContainer(from:using:configuration:useLatest:progressHandler:)`` and
/// ``ModelContainer``.
public struct ModelContext: Sendable {
    public var configuration: ModelConfiguration
    public var model: any LanguageModel & Sendable
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer

    public init(
        configuration: ModelConfiguration, model: some TrainableLanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = MaterializedModule(model)
        self.processor = processor
        self.tokenizer = tokenizer
    }

    public init(_ context: consuming TrainableModelContext) {
        self.configuration = context.configuration
        func makeModel<T: TrainableLanguageModel>(_ m: consuming T) -> any LanguageModel & Sendable
        {
            MaterializedModule(m)
        }
        self.model = makeModel(context.model)
        self.processor = context.processor
        self.tokenizer = context.tokenizer
    }
}

/// A version of ``ModelContext`` that holds a ``TrainableLanguageModel``.
///
/// This context contains the same properties as a `ModelContext` but the `model` is
/// mutable and thus the context is _not_ `Sendable`.
///
/// Produced by ``loadTrainable(from:using:configuration:useLatest:progressHandler:)`` or the
/// equivalent methods on `LLMModelFactory` and `VLMModelFactory`.
public struct TrainableModelContext {
    public var configuration: ModelConfiguration
    public var model: any TrainableLanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer

    public init(
        configuration: ModelConfiguration, model: some TrainableLanguageModel,
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
/// See concrete implementations in:
///
/// - `LLMModelFactory`
/// - `VLMModelFactory`
/// - `EmbedderModelFactory`
///
/// or, if loading LLM/VLMs, use the free functions:
///
/// - ``loadModel(from:using:configuration:useLatest:progressHandler:)``
/// - ``loadModelContainer(from:using:configuration:useLatest:progressHandler:)``
///
/// or variants.
public protocol GenericModelFactory<ContextType, ContainerType>: Sendable {

    associatedtype ContextType: Sendable
    associatedtype ContainerType: Sendable

    var modelRegistry: AbstractModelRegistry { get }

    /// load level load of a ``ResolvedModelConfiguration`` (urls) into a
    /// ``ContextType``.  This is typically a `Sendable` `struct` that holds the
    /// values needed to run inference in the model (for ``ModelContext`` the
    /// model is wrapped in a `MaterializedModule`).
    func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ContextType

    /// Wrap a ``ContextType`` in a ``ContainerType``.
    ///
    /// The `ContainerType` is a `Sendable` container for managing the model contained
    /// in the `ContextType`.
    func _wrap(_ context: ContextType) -> ContainerType
}

extension GenericModelFactory {

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

extension GenericModelFactory {

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContext``.
    ///
    /// This resolves the configuration (downloading remote sources via the downloader)
    /// and then loads the model from local files.
    ///
    /// ## See Also
    /// - ``loadModel(from:using:configuration:useLatest:progressHandler:)``
    /// - ``loadModelContainer(from:using:configuration:useLatest:progressHandler:)``
    public func load(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ContextType {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        return try await _load(configuration: resolved, tokenizerLoader: tokenizerLoader)
    }

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContainer``.
    ///
    /// Note: `ModelContext` is now `Sendable` and is preferred over `ModelContainer`.
    @available(*, deprecated, message: "use load instead")
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ContainerType {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        let context = try await _load(configuration: resolved, tokenizerLoader: tokenizerLoader)
        return _wrap(context)
    }

    /// Load a model from a local directory, producing a ``ModelContext``.
    ///
    /// No downloader is needed — the model and tokenizer are loaded from
    /// the given directory.
    public func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> sending ContextType {
        try await _load(
            configuration: .init(directory: directory), tokenizerLoader: tokenizerLoader)
    }

    /// Load a model from a local directory, producing a ``ModelContainer``.
    ///
    /// Note: `ModelContext` is now `Sendable` and is preferred over `ModelContainer`.
    @available(*, deprecated, message: "use load instead")
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> ContainerType {
        let context = try await _load(
            configuration: .init(directory: directory), tokenizerLoader: tokenizerLoader)
        return _wrap(context)
    }

}

extension GenericModelFactory
where ContextType == ModelContext, ContainerType == ModelContainerConstraint {

    public func _wrap(_ context: ModelContext) -> ModelContainerConstraint {
        .init(context: context)
    }

}

public protocol TrainableModelContextLoader {

    func loadTrainable(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending TrainableModelContext
}

extension TrainableModelContextLoader {

    public func loadTrainable(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending TrainableModelContext {
        try await loadTrainable(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }

}

/// For backward compatibility: `ModelFactory` refers to an LLM/VLM model factory.
public typealias ModelFactory = GenericModelFactory<ModelContext, ModelContainerConstraint>
    & TrainableModelContextLoader

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
            matching: modelDownloadPatterns,
            useLatest: useLatest,
            progressHandler: progressHandler)
    case .directory(let directory):
        modelDirectory = directory
    }

    let tokenizerDirectory: URL
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        tokenizerDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: tokenizerDownloadPatterns,
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

// MARK: - LLM Model Loading Free Functions -- implied ModelFactory

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
@available(*, deprecated, message: "use loadModel instead")
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    id: String,
    revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, using: tokenizerLoader,
            configuration: .init(id: id, revision: revision),
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
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
@available(*, deprecated, message: "use loadModel instead")
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    id: String,
    revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContext`` holding the model and tokenizer
/// in a Sendable context.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
/// - Returns: a ``ModelContext``
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(from: directory, using: tokenizerLoader)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``TrainableModelContext`` holding the model and tokenizer
/// in a non-Sendable context (suitable for training).
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
/// - Returns: a ``TrainableModelContext``
public func loadTrainable(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending TrainableModelContext {
    try await load {
        try await $0.loadTrainable(
            from: LocalDownloader(url: directory),
            using: tokenizerLoader,
            configuration: .init(directory: directory))
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``TrainableModelContext`` holding the model and tokenizer
/// in a non-Sendable context (suitable for training).
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``TrainableModelContext``
public func loadTrainable(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending TrainableModelContext {
    try await load {
        try await $0.loadTrainable(
            from: downloader, using: tokenizerLoader,
            configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
/// - Returns: a ``ModelContainer``
@available(*, deprecated, message: "use loadModel instead")
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(from: directory, using: tokenizerLoader)
    }
}

private func load<R>(loader: (any ModelFactory) async throws -> sending R) async throws -> sending R
{
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
    static func modelFactory() -> (any ModelFactory)?
}

/// Registry of ``ModelFactory`` trampolines.
///
/// This allows ``loadModel(from:using:id:revision:useLatest:progressHandler:)`` to use any ``ModelFactory`` instances
/// available but be defined in the `LLMCommon` layer.  This is not typically used directly -- it is
/// called via ``loadModel(from:using:id:revision:useLatest:progressHandler:)``:
///
/// ```swift
/// let model = try await loadModel(from: downloader, using: tokenizerLoader, id: "mlx-community/Qwen3-4B-4bit")
/// ```
///
/// ## See Also
/// - ``loadModel(from:using:id:revision:useLatest:progressHandler:)``
/// - ``loadModel(from:using:)``
/// - ``loadModelContainer(from:using:id:revision:useLatest:progressHandler:)``
/// - ``loadModelContainer(from:using:)``
final public class ModelFactoryRegistry: @unchecked Sendable {
    public static let shared = ModelFactoryRegistry()

    private let lock = NSLock()
    private var trampolines: [() -> (any ModelFactory)?]

    private init() {
        self.trampolines = [
            {
                (NSClassFromString("MLXVLM.TrampolineModelFactory")
                    as? any ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
            {
                (NSClassFromString("MLXLLM.TrampolineModelFactory")
                    as? any ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
        ]
    }

    public func addTrampoline(_ trampoline: @escaping () -> (any ModelFactory)?) {
        lock.withLock {
            trampolines.append(trampoline)
        }
    }

    public func modelFactories() -> [any ModelFactory] {
        lock.withLock {
            trampolines.compactMap { $0() }
        }
    }
}
