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
///    local. The reported `Progress` sums bytes read across the model's
///    `safetensors` readers. Headers can be read more than once.
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
    /// The progress reaches completion when the load returns. Some checkpoint tensors
    /// can be excluded from loading, so the reported byte count may stop short until then.
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
    /// Collect progress around the whole load, including checkpoint reads and evaluation.
    func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader,
        progress: LoadProgressHandlers
    ) async throws -> sending ContextType {
        guard let weightsHandler = progress.weights else {
            return try await _load(configuration: configuration, tokenizerLoader: tokenizerLoader)
        }

        let reporter = ModelLoadProgressReporter(
            totalUnitCount: safetensorsByteCount(
                in: configuration.modelDirectory, selection: configuration.weightFileSelection),
            handler: weightsHandler)

        let context = try await ModelLoadProgressScope.$reporter.withValue(reporter) {
            try await MLX.withLoadProgressHandler(
                { reporter.update($0) },
                {
                    try await _load(configuration: configuration, tokenizerLoader: tokenizerLoader)
                }
            )
        }
        reporter.finish()
        return context
    }
}

enum ModelLoadProgressScope {
    @TaskLocal static var reporter: ModelLoadProgressReporter?
}

/// The number of bytes of `safetensors` weights in `modelDirectory`.
///
/// The model may select additional files or skip some checkpoint tensors.
func safetensorsByteCount(
    in modelDirectory: URL, selection: WeightFileSelection = .automatic
) -> Int64 {
    guard FileManager.default.fileExists(atPath: modelDirectory.path) else { return 0 }

    let urls = (try? safetensorWeightURLs(in: modelDirectory, selection: selection)) ?? []
    return urls.reduce(into: 0) { total, url in
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        total += Int64(size)
    }
}

/// Aggregates the byte progress of the model's readers.
final class ModelLoadProgressReporter: @unchecked Sendable {

    /// Publish at most this many updates over the course of the load.
    ///
    /// MLX reports progress in fairly small chunks -- roughly one per 4MB -- and the handler
    /// typically hops to the main actor to update a progress bar, so coalesce the updates.
    private static let updateCount: Int64 = 1000

    private let totalUnitCount: Int64
    private let handler: @Sendable (Progress) -> Void

    private let lock = NSLock()
    private enum Reader: Hashable {
        case file(URL)
        case worker(UUID)
    }
    private var completedByReader = [Reader: Int64]()
    private var lastPublished: Int64 = 0
    private var didPublish = false

    init(totalUnitCount: Int64, handler: @escaping @Sendable (Progress) -> Void) {
        self.totalUnitCount = max(totalUnitCount, 0)
        self.handler = handler
    }

    /// Record the progress of a single file and publish the aggregate.
    func update(_ update: MLX.LoadProgress, worker: UUID? = nil) {
        lock.withLock {
            let reader = worker.map(Reader.worker) ?? .file(update.url)
            completedByReader[reader] = update.completedUnitCount
            let completed = completedByReader.values.reduce(0, +)

            let step = max(totalUnitCount / Self.updateCount, 1)
            guard !didPublish || completed >= lastPublished + step else { return }

            publish(completed)
        }
    }

    /// Publish completion.
    ///
    /// Call this once the model has loaded. Some checkpoint tensors may not be read.
    func finish() {
        lock.withLock {
            if totalUnitCount == 0 {
                let progress = Progress.discreteProgress(totalUnitCount: 1)
                progress.completedUnitCount = 1
                handler(progress)
            } else {
                publish(totalUnitCount)
            }
        }
    }

    private func publish(_ completedUnitCount: Int64) {
        lastPublished = completedUnitCount
        didPublish = true
        let progress = Progress.discreteProgress(totalUnitCount: totalUnitCount)
        progress.completedUnitCount = min(completedUnitCount, totalUnitCount)
        handler(progress)
    }
}
