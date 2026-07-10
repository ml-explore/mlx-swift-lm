// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import MLXFoundationModels
import MLXLMCommon

/// Builds an `MLXLanguageModel` backed by HuggingFace downloading and
/// tokenizer loading, so a configuration is all the caller provides.
///
/// The macro synthesizes the `weightsLocation:` and `load:` arguments; the
/// caller supplies only `configuration` (plus optional `capabilities` and
/// `configurationResolver`). A caller needing a custom weights location or
/// loader should call the `MLXLanguageModel` initializer directly.
///
/// The expansion references symbols the caller must have in scope:
/// ```swift
/// import Foundation          // URL, Progress (via #hubDownloader)
/// import MLXHuggingFace       // this macro + #hubDownloader / #huggingFaceTokenizerLoader
/// import MLXFoundationModels  // MLXLanguageModel
/// import MLXLMCommon          // ModelConfiguration, loadModelContainer
/// import HuggingFace          // HubClient (via #hubDownloader) + HubCache (synthesized weightsLocation)
/// import Tokenizers           // AutoTokenizer (via #huggingFaceTokenizerLoader)
/// ```
///
/// ```swift
/// let model = #huggingFaceLanguageModel(configuration: LLMRegistry.gemma3_1B_qat_4bit)
/// let session = LanguageModelSession(model: model)
/// ```
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@freestanding(expression)
public macro huggingFaceLanguageModel(
    configuration: ModelConfiguration,
    // The `capabilities` / `configurationResolver` defaults mirror
    // `MLXLanguageModel.init(configuration:capabilities:configurationResolver:weightsLocation:load:)`.
    // The expansion forwards each argument only when the caller supplies it, so
    // an omitted argument falls through to the initializer's own default rather
    // than the value written here — keep the two in sync so this signature does
    // not advertise a default the expansion never applies.
    capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration],
    configurationResolver: any ModelConfigurationResolver = DefaultConfigurationResolver()
) -> MLXLanguageModel =
    #externalMacro(module: "MLXHuggingFaceMacros", type: "LanguageModelMacro")

#endif
