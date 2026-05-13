import Foundation
import MLXLMCommon

/// Wrap a `HubClient` as a `Downloader`.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
///
/// let model = try await loadModelContainer(
///     from: #hubDownloader(HubClient()),
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration
/// )
/// ```
@freestanding(expression)
public macro hubDownloader(_ hub: Any) -> MLXLMCommon.Downloader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "DownloaderMacro")

/// Provide a default `HubClient` as a `Downloader`.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
///
/// let model = try await loadModelContainer(
///     from: #hubDownloader(),
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration
/// )
/// ```
@freestanding(expression)
public macro hubDownloader() -> MLXLMCommon.Downloader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "DownloaderMacro")

/// Wrap a `Tokenizers.Tokenizer` in `Tokenizer`.
///
/// This is used internally by ``huggingFaceTokenizerLoader()`` -- typically not used directly.
///
/// ```swift
/// import MLXHuggingFace
/// import Tokenizers
///
/// let t: Tokenizers.Tokenizer
///
/// let tokenizer = #adaptHuggingFaceTokenizer(t)
/// ```
@freestanding(expression)
public macro adaptHuggingFaceTokenizer(_ upstream: Any) -> MLXLMCommon.Tokenizer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "TokenizerAdaptorMacro")

/// Provide a `TokenizerLoader` from `Tokenizers.AutoTokenizer`.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
///
/// let model = try await loadModelContainer(
///     from: #hubDownloader(),
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration
/// )
/// ```
@freestanding(expression)
public macro huggingFaceTokenizerLoader() -> MLXLMCommon.TokenizerLoader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "TokenizerLoaderMacro")

/// Load a `ModelContainer` using default hub client and tokenizer loader.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
/// import Tokenizers
///
/// let model = try await huggingFaceLoadModelContainer(
///     configuration: modelConfiguration
/// )
/// ```
@freestanding(expression)
public macro huggingFaceLoadModelContainer(
    configuration: ModelConfiguration
) -> ModelContainer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContainerMacro")

/// Load a `ModelContainer` using default hub client and tokenizer loader with progress.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
/// import Tokenizers
///
/// let model = try await huggingFaceLoadModelContainer(
///     configuration: modelConfiguration
/// ) { progres in ... }
/// ```
@freestanding(expression)
public macro huggingFaceLoadModelContainer(
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) -> ModelContainer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContainerMacro")

/// Load a `ModelContext` using default hub client and tokenizer loader.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
/// import Tokenizers
///
/// let modelContext = try await huggingFaceLoadModel(
///     configuration: modelConfiguration
/// )
/// ```
@freestanding(expression)
public macro huggingFaceLoadModel(
    configuration: ModelConfiguration
) -> ModelContext =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContextMacro")

/// Load a `ModelContext` using default hub client and tokenizer loader with progress.
///
/// ```swift
/// import MLXHuggingFace
/// import HuggingFace
/// import Tokenizers
///
/// let modelContext = try await huggingFaceLoadModel(
///     configuration: modelConfiguration
/// ) { progres in ... }
/// ```
@freestanding(expression)
public macro huggingFaceLoadModel(
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) -> ModelContext =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContextMacro")

public enum HuggingFaceDownloaderError: LocalizedError {
    case invalidRepositoryID(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            return "Invalid Hugging Face repository ID: '\(id)'. Expected format 'namespace/name'."
        }
    }
}
