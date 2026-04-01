import Foundation
import MLXLMCommon

@freestanding(expression)
public macro hubDownloader(_ hub: Any) -> MLXLMCommon.Downloader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "DownloaderMacro")

@freestanding(expression)
public macro hubDownloader() -> MLXLMCommon.Downloader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "DownloaderMacro")

@freestanding(expression)
public macro adaptHuggingFaceTokenizer(_ upstream: Any) -> MLXLMCommon.Tokenizer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "TokenizerAdaptorMacro")

@freestanding(expression)
public macro huggingFaceTokenizerLoader() -> MLXLMCommon.TokenizerLoader =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "TokenizerLoaderMacro")

@freestanding(expression)
public macro huggingFaceLoadModelContainer(
    configuration: ModelConfiguration
) -> ModelContainer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContainerMacro")

@freestanding(expression)
public macro huggingFaceLoadModelContainer(
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) -> ModelContainer =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContainerMacro")

@freestanding(expression)
public macro huggingFaceLoadModel(
    configuration: ModelConfiguration
) -> ModelContext =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LoadContextMacro")

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
