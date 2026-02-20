import Foundation
import HuggingFace
import MLXEmbedders
import MLXLMCommon
import MLXLMHuggingFace
import Tokenizers

// MARK: - Convenience overloads defaulting to HubClient.default

/// Load an embedding model using the default Hugging Face Hub client.
///
/// This is equivalent to calling ``MLXEmbedders/load(from:configuration:useLatest:progressHandler:)``
/// with `HubClient.default` as the downloader.
public func load(
    from hub: HubClient = .default,
    configuration: MLXEmbedders.ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> (EmbeddingModel, Tokenizer) {
    try await MLXEmbedders.load(
        from: hub, configuration: configuration, useLatest: useLatest,
        progressHandler: progressHandler)
}

/// Load an embedding model container using the default Hugging Face Hub client.
///
/// This is equivalent to calling ``MLXEmbedders/loadModelContainer(from:configuration:useLatest:progressHandler:)``
/// with `HubClient.default` as the downloader.
public func loadModelContainer(
    from hub: HubClient = .default,
    configuration: MLXEmbedders.ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> MLXEmbedders.ModelContainer {
    try await MLXEmbedders.loadModelContainer(
        from: hub, configuration: configuration, useLatest: useLatest,
        progressHandler: progressHandler)
}
