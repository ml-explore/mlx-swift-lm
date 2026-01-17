// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

/// Asynchronously loads and initializes a pretrained tokenizer.
///
/// This function serves as the primary entry point for preparing a tokenizer. It fetches
/// configuration and vocabulary data—either from the Hugging Face Hub or a local
/// directory—and initializes a `PreTrainedTokenizer`.
///
/// - Parameters:
///   - configuration: The `ModelConfiguration` containing the model ID or directory path.
///   - hub: An instance of `HubApi` used to manage downloads and file access.
/// - Returns: An initialized `Tokenizer` ready for encoding and decoding text.
/// - Throws: `EmbedderError.missingTokenizerConfig` if the configuration files cannot be found,
///   or standard network/parsing errors.
public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
        configuration: configuration, hub: hub)

    return try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

/// Retrieves the raw configuration and data files required to build a tokenizer.
///
/// This internal helper handles the logic of determining where to fetch files from.
/// It includes a robust fallback: if a network request fails due to lack of internet
/// connectivity, it attempts to load the files from the local model directory.
///
/// - Parameters:
///   - configuration: The model configuration providing the `tokenizerId` or `modelDirectory`.
///   - hub: The `HubApi` interface for remote or local file resolution.
/// - Returns: A tuple containing the `tokenizerConfig` and `tokenizerData` configurations.
/// - Throws: `NSURLError` for network issues (other than offline status).
/// - Throws: `EmbedderError.missingTokenizerConfig` if the configuration files are
///   successfully accessed but do not contain a valid `tokenizerConfig` payload.
///   This typically occurs when the model repository or directory is missing a
///   `tokenizer_config.json` file.
func loadTokenizerConfig(
    configuration: ModelConfiguration,
    hub: HubApi
) async throws -> (Config, Config) {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id):
        do {
            // Attempt to load from the remote Hub or Hub cache
            let loaded = LanguageModelConfigurationFromHub(
                modelName: configuration.tokenizerId ?? id, hubApi: hub)

            // Trigger an async fetch to verify the config exists
            _ = try await loaded.tokenizerConfig
            config = loaded
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain
                && nserror.code == NSURLErrorNotConnectedToInternet
            {
                // Fallback: Internet connection is offline, load from the local model directory
                config = LanguageModelConfigurationFromHub(
                    modelFolder: configuration.modelDirectory(hub: hub), hubApi: hub)
            } else {
                // Re-throw if it's a critical error (e.g., 404, parsing error)
                throw error
            }
        }
    case .directory(let directory):
        // Load directly from a specified local directory
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard let tokenizerConfig = try await config.tokenizerConfig else {
        throw EmbedderError.missingTokenizerConfig
    }
    let tokenizerData = try await config.tokenizerData
    return (tokenizerConfig, tokenizerData)
}
