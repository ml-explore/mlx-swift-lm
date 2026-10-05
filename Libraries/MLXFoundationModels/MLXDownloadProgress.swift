// Copyright © 2026 Apple Inc.

// Gated identically to the MLXLanguageModel adapter. This observable's only
// producer is the adapter's download path (MLXLanguageModel reports into it),
// so it lives and dies with the adapter rather than surviving as an orphan
// when the trait or the 27.0 SDK is absent.
#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// One reported change to a model's load.
///
/// Carries only `Sendable` primitives, so a producer on any thread builds one
/// without an isolation check.
enum DownloadProgressUpdate: Sendable {
    case started(modelID: String, loadID: UUID)
    case progress(modelID: String, loadID: UUID, fraction: Double, completed: Int64, total: Int64)
    case ended(modelID: String, loadID: UUID)
}

/// The shared instance's inbox.
///
/// A producer yields synchronously from whatever thread it is on, and
/// `AsyncStream` delivers in yield order. Order matters for one model's own
/// updates: an end applied before its start would leave the entry stranded. The
/// buffering policy is stated rather than defaulted, because a policy that drops
/// elements would break both the order and the start-to-end pairing.
private let sharedProgressInbox = AsyncStream<DownloadProgressUpdate>.makeStream(
    bufferingPolicy: .unbounded)

/// Reads `MLXDownloadProgress.shared` once, which is what starts the drain.
/// Nothing else does, so without this the inbox buffers every update for the
/// life of the process. A global initializer runs lazily and exactly once.
private let startProgressDrainOnce: Void = {
    Task { @MainActor in _ = MLXDownloadProgress.shared }
}()

/// Observable download progress for MLX model loading.
///
/// Tracks one entry per model the adapter is downloading, and aggregates them.
/// Shared singleton so any view in the app can observe download state.
///
/// Usage:
/// ```swift
/// struct MyView: View {
///     var downloadProgress = MLXDownloadProgress.shared
///
///     var body: some View {
///         ForEach(downloadProgress.activeDownloads) { download in
///             ProgressView(value: download.fractionCompleted) {
///                 Text(download.modelID)
///             }
///         }
///     }
/// }
/// ```
@MainActor
@Observable
public final class MLXDownloadProgress {

    /// One model's download progress.
    ///
    /// A value, so a consumer reads it once and holds it across an actor hop.
    public struct Download: Sendable, Identifiable {

        /// The model identifier, as `ModelConfiguration.name` gives it. A
        /// repository ID or a path fragment, never a display name.
        public let modelID: String

        /// The model identifier, which is what distinguishes one download here.
        public var id: String { modelID }

        /// Progress from 0.0 to 1.0.
        public let fractionCompleted: Double

        /// Bytes downloaded so far, from `Progress.completedUnitCount`.
        public let completedBytes: Int64

        /// Total bytes, from `Progress.totalUnitCount`.
        public let totalBytes: Int64

        /// When the load now running first reported progress for this model.
        public let startedAt: Date

        /// Rolling average throughput over the last five seconds of samples. nil
        /// until two samples span more than a tenth of a second.
        public let throughputBytesPerSecond: Double?
    }

    /// Shared singleton instance, and the only instance that drains
    /// `sharedProgressInbox`.
    public static let shared: MLXDownloadProgress = {
        let progress = MLXDownloadProgress()
        progress.consume(sharedProgressInbox.stream)
        return progress
    }()

    /// One entry per model being loaded. An entry whose `download` is nil has
    /// started and has not reported yet, so it is not a download.
    private var states: [String: State] = [:]

    private struct State {
        /// Which load owns this entry. Eviction cancels a load and forgets it at
        /// once, so a replacement load can start while the cancelled one is still
        /// unwinding a network read. Every update the cancelled load still sends
        /// carries its own identifier, and this is what rejects it.
        let loadID: UUID
        var download: Download?
        var samples: [(time: Date, bytes: Int64)] = []
    }

    /// Width of the throughput window. Short enough that a stall shows within a
    /// few seconds, long enough to smooth the jitter in chunk arrivals.
    private static let throughputWindow: TimeInterval = 5

    /// Internal, so a test builds an isolated instance and calls ``apply(_:)`` on
    /// it. An instance other than ``shared`` drains no inbox, so nothing reaches
    /// it except a direct ``apply(_:)`` call.
    init() {}

    // MARK: - Reading

    /// Unsorted, for the aggregates. ``activeDownloads`` is the published order.
    private var reportedDownloads: [Download] {
        states.values.compactMap(\.download)
    }

    /// Every model downloading now, earliest first. Empty when none is.
    public var activeDownloads: [Download] {
        reportedDownloads.sorted { ($0.startedAt, $0.modelID) < ($1.startedAt, $1.modelID) }
    }

    /// This model's download, or nil when it is not downloading or has not
    /// reported progress yet.
    public func download(forModelID modelID: String) -> Download? {
        states[modelID]?.download
    }

    /// Combined progress across every downloading model, weighted by bytes. 0
    /// before the first report carries a total.
    public var fractionCompleted: Double {
        let total = totalBytes
        guard total > 0 else { return 0 }
        return Double(completedBytes) / Double(total)
    }

    /// Bytes downloaded so far, across every downloading model.
    public var completedBytes: Int64 {
        reportedDownloads.reduce(0) { $0 + $1.completedBytes }
    }

    /// Total bytes, across every downloading model.
    public var totalBytes: Int64 {
        reportedDownloads.reduce(0) { $0 + $1.totalBytes }
    }

    /// When the earliest download still running first reported, and nil when none
    /// is running. Consumers can compute elapsed time as
    /// `Date.now.timeIntervalSince(startedAt)`.
    public var startedAt: Date? {
        reportedDownloads.map(\.startedAt).min()
    }

    /// Combined throughput across every downloading model, and nil when no model
    /// has enough samples yet.
    ///
    /// Rolling (not cumulative) so a stall shows up immediately as the number
    /// dropping toward 0 -- consumers can show "still moving" vs "stuck" without
    /// needing a separate indicator.
    public var throughputBytesPerSecond: Double? {
        let rates = reportedDownloads.compactMap(\.throughputBytesPerSecond)
        return rates.isEmpty ? nil : rates.reduce(0, +)
    }

    // MARK: - Reporting

    /// Reports that a load for `modelID` started. `loadID` identifies this load
    /// attempt, and every later call for it must carry the same value.
    ///
    /// Pair every call with ``reportEnded(modelID:loadID:)``, or that model's
    /// progress never clears.
    nonisolated static func reportStarted(modelID: String, loadID: UUID) {
        _ = startProgressDrainOnce
        sharedProgressInbox.continuation.yield(.started(modelID: modelID, loadID: loadID))
    }

    /// Reports one progress change for the load `loadID` identifies.
    ///
    /// Nonisolated so a producer inside a sendable closure, for example the cache
    /// loader's `progressHandler`, reaches the inbox without a hop to the main
    /// actor. A hop here would lose the order the reports were made in.
    nonisolated static func report(progress: Progress, modelID: String, loadID: UUID) {
        _ = startProgressDrainOnce
        sharedProgressInbox.continuation.yield(
            .progress(
                modelID: modelID,
                loadID: loadID,
                fraction: progress.fractionCompleted,
                completed: progress.completedUnitCount,
                total: progress.totalUnitCount))
    }

    /// Reports that the load `loadID` identifies ended, whether it succeeded,
    /// failed, or was cancelled.
    nonisolated static func reportEnded(modelID: String, loadID: UUID) {
        _ = startProgressDrainOnce
        sharedProgressInbox.continuation.yield(.ended(modelID: modelID, loadID: loadID))
    }

    // MARK: - Applying

    /// Drains `updates` for the life of this instance. Only ``shared`` calls it.
    /// A second consumer would take updates the shared instance needs.
    private func consume(_ updates: AsyncStream<DownloadProgressUpdate>) {
        Task { @MainActor in
            for await update in updates {
                self.apply(update)
            }
        }
    }

    /// Folds one update into the published state, in the order it was yielded.
    ///
    /// An update from a load that no longer owns its model's entry is dropped, so
    /// a load that was cancelled cannot disturb the load that replaced it.
    func apply(_ update: DownloadProgressUpdate) {
        switch update {
        case .started(let modelID, let loadID):
            // Replaces any earlier entry, so the newest load owns the model and
            // brings its own start time and samples. Nothing is published for it
            // until it reports.
            states[modelID] = State(loadID: loadID)

        case .progress(let modelID, let loadID, let fraction, let completed, let total):
            // A model already on disk reports 1.0 at once. Publishing it would
            // flash a full progress bar for one frame. A finishing download's
            // final 1.0 is dropped for the same reason, and its entry goes away
            // when its load ends.
            guard fraction < 1 else { return }
            guard var state = states[modelID], state.loadID == loadID else { return }

            let now = Date.now
            state.samples.append((time: now, bytes: completed))
            let cutoff = now.addingTimeInterval(-Self.throughputWindow)
            state.samples.removeAll { $0.time < cutoff }
            state.download = Download(
                modelID: modelID,
                fractionCompleted: fraction,
                completedBytes: completed,
                totalBytes: total,
                startedAt: state.download?.startedAt ?? now,
                throughputBytesPerSecond: Self.throughput(across: state.samples))
            states[modelID] = state

        case .ended(let modelID, let loadID):
            guard states[modelID]?.loadID == loadID else { return }
            states[modelID] = nil
        }
    }

    /// Rate across the retained samples. nil below two samples, and nil when they
    /// span a tenth of a second or less, because a shorter span gives a
    /// meaningless rate.
    private static func throughput(across samples: [(time: Date, bytes: Int64)]) -> Double? {
        guard samples.count >= 2, let oldest = samples.first, let newest = samples.last
        else { return nil }
        let seconds = newest.time.timeIntervalSince(oldest.time)
        guard seconds > 0.1 else { return nil }
        return Double(newest.bytes - oldest.bytes) / seconds
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
