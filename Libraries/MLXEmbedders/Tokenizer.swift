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
    switch configuration.id {
    case .id(let id, let revision):
        return try await AutoTokenizer.from(
            pretrained: configuration.tokenizerId ?? id,
            hubApi: hub,
            revision: revision
        )
    case .directory(let directory):
        return try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)
    }
}
