// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The progress callbacks of loading a model, grouped in one value.
///
/// Loading a model is a two phase affair, and each phase has its own progress:
///
/// 1. _downloading_ the model, when it comes from a provider rather than a local
///    directory;
/// 2. _reading the weights_ from disk, which dominates the load once the files are
///    local. This phase is byte precise: a model is frequently split into several
///    `safetensors` shards, read concurrently, and the reported `Progress` sums the
///    bytes read across all of them.
///
/// Pass it to the loading entry points, e.g.:
///
/// ```swift
/// let progress = LoadProgressHandlers.weights { progress in
///     print("reading weights \(progress.fractionCompleted)")
/// }
/// let container = try await LLMModelFactory.shared.loadContainer(
///     from: directory, using: tokenizerLoader, progress: progress)
/// ```
///
/// The handlers are called from background threads -- the download from the
/// downloader, the weights from MLX worker threads -- and are on the critical path
/// of the load, so they should be cheap and must not call back into loading. Hop to
/// the main actor if the progress drives UI:
///
/// ```swift
/// @MainActor
/// final class LoadingModel {
///     var fraction: Double = 0
///
///     var handlers: LoadProgressHandlers {
///         .weights { [weak self] progress in
///             Task { @MainActor in self?.fraction = progress.fractionCompleted }
///         }
///     }
/// }
/// ```
///
/// - Note: the former `progressHandler` loading overloads are deprecated. Use
///   ``download`` for download progress.
public struct LoadProgressHandlers: Sendable {

    /// Progress of downloading the model from a provider.
    ///
    /// Not called when loading from a local directory.
    public var download: (@Sendable (Progress) -> Void)?

    /// Progress of reading the weights from disk, reported in bytes.
    ///
    /// Loading is lazy, so this is reported while the model is being evaluated, and
    /// reaches completion when the load returns. Weights dropped by
    /// ``BaseLanguageModel/sanitize(weights:metadata:)`` are never read.
    public var weights: (@Sendable (Progress) -> Void)?

    public init(
        download: (@Sendable (Progress) -> Void)? = nil,
        weights: (@Sendable (Progress) -> Void)? = nil
    ) {
        self.download = download
        self.weights = weights
    }

    /// Only report the progress of reading the weights.
    public static func weights(_ handler: @escaping @Sendable (Progress) -> Void) -> Self {
        .init(weights: handler)
    }

    /// Only report the progress of the download.
    public static func download(_ handler: @escaping @Sendable (Progress) -> Void) -> Self {
        .init(download: handler)
    }

}

extension GenericModelFactory {

    /// ``_load(configuration:tokenizerLoader:)`` reporting the progress of the load --
    /// the reading of the weights -- to `progress.weights`.
    ///
    /// Loading is lazy: the weights are read while the model is evaluated at the end of
    /// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``, so the
    /// progress is collected around the whole load.
    func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader,
        progress: LoadProgressHandlers
    ) async throws -> sending ContextType {
        guard let weightsHandler = progress.weights else {
            return try await _load(configuration: configuration, tokenizerLoader: tokenizerLoader)
        }

        let reporter = ModelLoadProgressReporter(
            totalUnitCount: safetensorsByteCount(in: configuration.modelDirectory),
            handler: weightsHandler)

        let context = try await MLX.withLoadProgressHandler(
            { reporter.update($0) },
            {
                try await _load(configuration: configuration, tokenizerLoader: tokenizerLoader)
            }
        )
        reporter.finish()
        return context
    }
}

/// The number of bytes of `safetensors` weights in `modelDirectory`.
///
/// This traverses the directory the same way
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)`` does, so the
/// result is the number of bytes that loading the model will read.
func safetensorsByteCount(in modelDirectory: URL) -> Int64 {
    guard FileManager.default.fileExists(atPath: modelDirectory.path) else { return 0 }

    let urls = (try? safetensorWeightURLs(in: modelDirectory)) ?? []
    return urls.reduce(into: 0) { total, url in
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        total += Int64(size)
    }
}

/// Aggregates the per file byte progress reported by MLX while the weights of a model are
/// read into a single `Progress` for the whole model.
///
/// A model is frequently split into several `safetensors` shards, and MLX reports progress
/// per file, from several threads. This sums the most recent progress of each file against
/// the total size of the weights.
final class ModelLoadProgressReporter: @unchecked Sendable {

    /// Publish at most this many updates over the course of the load.
    ///
    /// MLX reports progress in fairly small chunks -- roughly one per 4MB -- and the handler
    /// typically hops to the main actor to update a progress bar, so coalesce the updates.
    private static let updateCount: Int64 = 1000

    private let progress: Progress
    private let handler: @Sendable (Progress) -> Void

    private let lock = NSLock()
    private var completedByFile = [URL: Int64]()
    private var lastPublished: Int64 = 0
    private var didPublish = false

    init(totalUnitCount: Int64, handler: @escaping @Sendable (Progress) -> Void) {
        self.progress = Progress.discreteProgress(totalUnitCount: max(totalUnitCount, 0))
        self.handler = handler
    }

    /// Record the progress of a single file and publish the aggregate.
    func update(_ update: MLX.LoadProgress) {
        lock.withLock {
            completedByFile[update.url] = update.completedUnitCount
            let completed = completedByFile.values.reduce(0, +)

            let step = max(progress.totalUnitCount / Self.updateCount, 1)
            guard !didPublish || completed >= lastPublished + step else { return }

            publish(completed)
        }
    }

    /// Publish completion.
    ///
    /// Call this once the model has loaded. The weights dropped by
    /// ``BaseLanguageModel/sanitize(weights:metadata:)`` are never evaluated, and therefore
    /// never read, so the aggregate legitimately stops short of the size of the files.
    func finish() {
        lock.withLock {
            publish(progress.totalUnitCount)
        }
    }

    private func publish(_ completedUnitCount: Int64) {
        lastPublished = completedUnitCount
        didPublish = true
        progress.completedUnitCount = min(completedUnitCount, progress.totalUnitCount)
        handler(progress)
    }
}
