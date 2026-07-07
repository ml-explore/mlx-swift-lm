// Copyright © 2026 Apple Inc.

// Gated identically to the MLXLanguageModel adapter. This observable's only
// producer is the adapter's download path (MLXLanguageModel reports into it),
// so it lives and dies with the adapter rather than surviving as an orphan
// when the trait or the 27.0 SDK is absent.
#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// Observable download progress for MLX model loading.
///
/// Tracks whether a model is being downloaded/loaded and reports progress.
/// Shared singleton so any view in the app can observe download state.
///
/// Usage:
/// ```swift
/// struct MyView: View {
///     var downloadProgress = MLXDownloadProgress.shared
///
///     var body: some View {
///         if downloadProgress.isActive {
///             ProgressView(value: downloadProgress.fractionCompleted)
///         }
///     }
/// }
/// ```
@MainActor
@Observable
public final class MLXDownloadProgress {

    /// Shared singleton instance.
    public static let shared = MLXDownloadProgress()

    /// Whether a model is currently being downloaded or loaded.
    public private(set) var isActive = false

    /// Download progress from 0.0 to 1.0.
    public private(set) var fractionCompleted: Double = 0

    /// The model identifier being downloaded, if any.
    public private(set) var modelName: String?

    /// When the current download started. nil when inactive.
    /// Consumers can compute elapsed time as `Date.now.timeIntervalSince(startedAt)`.
    public private(set) var startedAt: Date?

    /// Bytes downloaded so far for the current download. Derived from the
    /// underlying `Progress.completedUnitCount`.
    public private(set) var completedBytes: Int64 = 0

    /// Total bytes for the current download. Derived from the underlying
    /// `Progress.totalUnitCount`. May be 0 before the first progress report.
    public private(set) var totalBytes: Int64 = 0

    /// Rolling average throughput in bytes per second, computed over the
    /// most recent ~5 seconds of progress samples. nil until we have at
    /// least two samples spanning a meaningful window.
    ///
    /// Rolling (not cumulative) so a stall shows up immediately as the
    /// number dropping toward 0 -- consumers can show "still moving" vs
    /// "stuck" without needing a separate indicator.
    public private(set) var throughputBytesPerSec: Double?

    /// Width of the throughput rolling window. Short enough that stalls
    /// are visible within a few seconds; long enough to smooth out the
    /// natural jitter in HF chunk arrivals.
    private let throughputWindow: TimeInterval = 5.0

    /// Samples used to compute rolling throughput. Pruned to
    /// `throughputWindow` on every `reportProgress` call.
    private var samples: [(time: Date, bytes: Int64)] = []

    private init() {}

    /// Nonisolated entry point for `reportProgress` so callers from sendable
    /// closures (e.g. the cache loader's `progressHandler`) don't have to
    /// hop to the main actor just to read `.shared`. The instance method is
    /// already nonisolated; this shim only forwards.
    nonisolated public static func report(progress: Progress, modelID: String) {
        Task { @MainActor in
            shared.reportProgress(progress, modelID: modelID)
        }
    }

    /// Nonisolated entry point for `reportCompleted`. Same rationale as
    /// ``report(progress:modelID:)``.
    nonisolated public static func reportCompleted() {
        Task { @MainActor in
            shared.reportCompleted()
        }
    }

    nonisolated func reportProgress(_ progress: Progress, modelID: String) {
        let fraction = progress.fractionCompleted
        // Don't show the progress UI for already-cached models (immediate 100%)
        guard fraction < 1.0 else { return }
        let completed = progress.completedUnitCount
        let total = progress.totalUnitCount
        Task { @MainActor in
            if self.startedAt == nil {
                self.startedAt = Date()
                self.samples.removeAll()
            }
            self.isActive = true
            self.fractionCompleted = fraction
            self.modelName = modelID
            self.completedBytes = completed
            self.totalBytes = total
            self.appendSampleAndRecompute(bytes: completed)
        }
    }

    nonisolated func reportCompleted() {
        Task { @MainActor in
            self.isActive = false
            self.fractionCompleted = 1.0
            self.modelName = nil
            self.startedAt = nil
            self.completedBytes = 0
            self.totalBytes = 0
            self.throughputBytesPerSec = nil
            self.samples.removeAll()
        }
    }

    /// Append the latest byte count, prune samples outside the rolling
    /// window, and recompute throughput. Requires at least 2 samples
    /// spanning a non-trivial time interval to produce a meaningful rate.
    private func appendSampleAndRecompute(bytes: Int64) {
        let now = Date()
        samples.append((time: now, bytes: bytes))
        let cutoff = now.addingTimeInterval(-throughputWindow)
        samples.removeAll { $0.time < cutoff }

        guard let oldest = samples.first,
            let newest = samples.last,
            samples.count >= 2
        else {
            throughputBytesPerSec = nil
            return
        }
        let dt = newest.time.timeIntervalSince(oldest.time)
        guard dt > 0.1 else {
            throughputBytesPerSec = nil
            return
        }
        let db = newest.bytes - oldest.bytes
        throughputBytesPerSec = Double(db) / dt
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
