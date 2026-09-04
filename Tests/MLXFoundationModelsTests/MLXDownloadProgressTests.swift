// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Unit tests for the main-actor fold. Each builds its own instance, so none of
/// them reads or writes `MLXDownloadProgress.shared`, and the suite needs no
/// serialization against its neighbors.
@Suite("MLXDownloadProgress state")
struct MLXDownloadProgressStateTests {

    @Test @MainActor
    func aReportForAModelThatNeverStartedIsIgnored() {
        let progress = MLXDownloadProgress()
        progress.apply(
            .progress(modelID: "org/a", loadID: UUID(), fraction: 0.5, completed: 5, total: 10))

        #expect(progress.activeDownloads.isEmpty)
    }

    @Test @MainActor
    func aStartedModelAppearsOnlyAfterItsFirstReport() {
        let progress = MLXDownloadProgress()
        let load = UUID()
        progress.apply(.started(modelID: "org/a", loadID: load))

        #expect(progress.activeDownloads.isEmpty)

        progress.apply(
            .progress(modelID: "org/a", loadID: load, fraction: 0.5, completed: 5, total: 10))

        #expect(progress.activeDownloads.map(\.modelID) == ["org/a"])
        #expect(progress.download(forModelID: "org/a")?.completedBytes == 5)
        #expect(progress.download(forModelID: "org/a")?.totalBytes == 10)
    }

    @Test @MainActor
    func aModelAlreadyOnDiskNeverAppears() {
        let progress = MLXDownloadProgress()
        let load = UUID()
        progress.apply(.started(modelID: "org/a", loadID: load))
        progress.apply(
            .progress(modelID: "org/a", loadID: load, fraction: 1, completed: 10, total: 10))

        #expect(progress.activeDownloads.isEmpty)
    }

    @Test @MainActor
    func twoModelsReportIndependentlyAndTheAggregatesSumThem() {
        let progress = MLXDownloadProgress()
        let loadA = UUID()
        let loadB = UUID()
        progress.apply(.started(modelID: "org/a", loadID: loadA))
        progress.apply(
            .progress(modelID: "org/a", loadID: loadA, fraction: 0.5, completed: 5, total: 10))
        progress.apply(.started(modelID: "org/b", loadID: loadB))
        progress.apply(
            .progress(modelID: "org/b", loadID: loadB, fraction: 0.25, completed: 30, total: 120))

        #expect(progress.activeDownloads.map(\.modelID) == ["org/a", "org/b"])
        #expect(progress.download(forModelID: "org/a")?.fractionCompleted == 0.5)
        #expect(progress.download(forModelID: "org/b")?.fractionCompleted == 0.25)
        #expect(progress.completedBytes == 35)
        #expect(progress.totalBytes == 130)
        #expect(progress.fractionCompleted == 35.0 / 130.0)
    }

    @Test @MainActor
    func endingOneModelLeavesTheOtherDownloading() {
        let progress = MLXDownloadProgress()
        let loadA = UUID()
        let loadB = UUID()
        progress.apply(.started(modelID: "org/a", loadID: loadA))
        progress.apply(
            .progress(modelID: "org/a", loadID: loadA, fraction: 0.5, completed: 5, total: 10))
        progress.apply(.started(modelID: "org/b", loadID: loadB))
        progress.apply(
            .progress(modelID: "org/b", loadID: loadB, fraction: 0.25, completed: 30, total: 120))
        progress.apply(.ended(modelID: "org/a", loadID: loadA))

        #expect(progress.activeDownloads.map(\.modelID) == ["org/b"])
        #expect(progress.download(forModelID: "org/a") == nil)
        #expect(progress.completedBytes == 30)
    }

    @Test @MainActor
    func aReportThatArrivesAfterTheEndIsIgnored() {
        let progress = MLXDownloadProgress()
        let load = UUID()
        progress.apply(.started(modelID: "org/a", loadID: load))
        progress.apply(
            .progress(modelID: "org/a", loadID: load, fraction: 0.5, completed: 5, total: 10))
        progress.apply(.ended(modelID: "org/a", loadID: load))
        progress.apply(
            .progress(modelID: "org/a", loadID: load, fraction: 0.75, completed: 7, total: 10))

        #expect(progress.activeDownloads.isEmpty)
    }

    @Test @MainActor
    func aCancelledLoadDoesNotEndTheLoadThatReplacedIt() {
        let progress = MLXDownloadProgress()
        let cancelled = UUID()
        let replacement = UUID()
        progress.apply(.started(modelID: "org/a", loadID: cancelled))
        progress.apply(
            .progress(modelID: "org/a", loadID: cancelled, fraction: 0.4, completed: 4, total: 10))
        // An eviction cancels the first load and forgets it at once, so the
        // replacement starts while the first is still unwinding and reports its
        // end afterwards.
        progress.apply(.started(modelID: "org/a", loadID: replacement))
        progress.apply(.ended(modelID: "org/a", loadID: cancelled))
        progress.apply(
            .progress(
                modelID: "org/a", loadID: replacement, fraction: 0.1, completed: 1, total: 10))

        #expect(progress.download(forModelID: "org/a")?.fractionCompleted == 0.1)

        progress.apply(.ended(modelID: "org/a", loadID: replacement))

        #expect(progress.activeDownloads.isEmpty)
    }

    @Test @MainActor
    func aReportFromACancelledLoadCannotMoveTheReplacementBackward() {
        let progress = MLXDownloadProgress()
        let cancelled = UUID()
        let replacement = UUID()
        progress.apply(.started(modelID: "org/a", loadID: cancelled))
        progress.apply(
            .progress(modelID: "org/a", loadID: cancelled, fraction: 0.4, completed: 4, total: 10))
        progress.apply(.started(modelID: "org/a", loadID: replacement))
        progress.apply(
            .progress(
                modelID: "org/a", loadID: replacement, fraction: 0.1, completed: 1, total: 10))
        let replacementStart = progress.download(forModelID: "org/a")?.startedAt
        // The cancelled load's last report, arriving after its replacement began.
        progress.apply(
            .progress(modelID: "org/a", loadID: cancelled, fraction: 0.4, completed: 4, total: 10))

        let download = progress.download(forModelID: "org/a")
        #expect(download?.fractionCompleted == 0.1)
        #expect(download?.completedBytes == 1)
        #expect(download?.startedAt == replacementStart)
        // The cancelled load's byte count never enters the replacement's samples,
        // so no rate is computed across two loads.
        #expect(download?.throughputBytesPerSecond == nil)
    }

    @Test @MainActor
    func aReplacementLoadPublishesNothingUntilItReports() {
        let progress = MLXDownloadProgress()
        let cancelled = UUID()
        let replacement = UUID()
        progress.apply(.started(modelID: "org/a", loadID: cancelled))
        progress.apply(
            .progress(modelID: "org/a", loadID: cancelled, fraction: 0.4, completed: 4, total: 10))
        progress.apply(.started(modelID: "org/a", loadID: replacement))

        #expect(progress.activeDownloads.isEmpty)
        #expect(progress.download(forModelID: "org/a") == nil)
    }

    @Test @MainActor
    func nothingDownloadingReportsEmptyAggregates() {
        let progress = MLXDownloadProgress()

        #expect(progress.activeDownloads.isEmpty)
        #expect(progress.fractionCompleted == 0)
        #expect(progress.completedBytes == 0)
        #expect(progress.totalBytes == 0)
        #expect(progress.startedAt == nil)
        #expect(progress.throughputBytesPerSecond == nil)
    }
}

/// Waits until the shared progress state satisfies `condition`.
///
/// A producer only hands its update to the inbox, so a test cannot read the state
/// on the line after a load returns. Each turn waits for the next change to what
/// `condition` reads, and never sleeps. A condition that never holds hangs here,
/// which the caller's time limit turns into a failure.
///
/// Runs on the main actor and does not suspend between testing the condition and
/// installing the tracking, so a change cannot land in that gap and leave this
/// waiting for a change that already happened.
@MainActor
private func waitForSharedProgress(toSatisfy condition: @MainActor () -> Bool) async {
    while !condition() {
        await withCheckedContinuation { continuation in
            withObservationTracking {
                _ = condition()
            } onChange: {
                continuation.resume()
            }
        }
    }
}

/// Satisfies `LanguageModel`'s requirements and nothing more. A test that only
/// loads a container never generates a token, so these bodies never run. Do not
/// treat this as a model. It has no parameters, and `callAsFunction` returns a
/// fixed zero array whose dimensions mean nothing.
private final class StubLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

/// Satisfies `UserInputProcessor`'s one requirement. A test that only loads a
/// container prepares no input, so this body never runs.
private struct StubInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        LMInput(tokens: MLXArray([0]))
    }
}

/// Holds a loader open, so a test observes the progress a load reported before
/// that load ends.
private actor ProgressLoadGate {
    private var reportedAlready = false
    private var reportedContinuation: CheckedContinuation<Void, Never>?
    private var releasedAlready = false
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func signalReported() {
        reportedAlready = true
        reportedContinuation?.resume()
        reportedContinuation = nil
    }

    func waitUntilReported() async {
        if reportedAlready { return }
        await withCheckedContinuation { reportedContinuation = $0 }
    }

    func waitForRelease() async {
        if releasedAlready { return }
        await withCheckedContinuation { releaseContinuation = $0 }
    }

    func release() {
        releasedAlready = true
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

/// Thrown when a parked loader resumes without being cancelled. Only the
/// cancellation handler releases it, so this error means the test's own gate
/// misbehaved rather than the cache.
private struct ParkedLoaderResumedWithoutCancellation: Error {}

/// Nested under the serialized `FoundationModelsCacheTests` parent, declared in
/// ModelCacheEvictionTests.swift, because these tests drive a real
/// `MLXLanguageModel` load and read the process-global
/// `MLXDownloadProgress.shared`.
extension FoundationModelsCacheTests {

    @Suite("MLXDownloadProgress load lifecycle")
    struct DownloadProgressLifecycle {

        struct LoadFailure: Error {}

        @Test(
            "a failed load clears the progress state for its model",
            .timeLimit(.minutes(1)))
        func failedLoadClearsProgress() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/throwing-loader-\(UUID().uuidString)"
            let gate = ProgressLoadGate()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: id),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
                load: { _, progressHandler in
                    let fakeProgress = Progress(totalUnitCount: 10)
                    fakeProgress.completedUnitCount = 5
                    progressHandler(fakeProgress)
                    await gate.signalReported()
                    await gate.waitForRelease()
                    throw LoadFailure()
                })

            let loadTask = Task { try await model.loadContainer() }
            await gate.waitUntilReported()
            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) != nil
            }

            await gate.release()
            await #expect(throws: LoadFailure.self) {
                try await loadTask.value
            }

            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) == nil
            }
        }

        @Test(
            "a cancelled load clears the progress state for its model",
            .timeLimit(.minutes(1)))
        func cancelledLoadClearsProgress() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/cancelled-loader-\(UUID().uuidString)"
            let gate = ProgressLoadGate()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: id),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
                load: { _, progressHandler in
                    let fakeProgress = Progress(totalUnitCount: 10)
                    fakeProgress.completedUnitCount = 5
                    progressHandler(fakeProgress)
                    await withTaskCancellationHandler {
                        await gate.waitForRelease()
                    } onCancel: {
                        // A cancellation handler cannot await, so hop onto the
                        // gate to release the parked loader.
                        Task { await gate.release() }
                    }
                    try Task.checkCancellation()
                    throw ParkedLoaderResumedWithoutCancellation()
                })

            let loadTask = Task { try await model.loadContainer() }
            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) != nil
            }

            await model.evict()
            await #expect(throws: CancellationError.self) {
                try await loadTask.value
            }

            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) == nil
            }
        }

        @Test(
            "a successful load clears the progress state for its model",
            .timeLimit(.minutes(1)))
        func successfulLoadClearsProgress() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/succeeding-loader-\(UUID().uuidString)"
            let gate = ProgressLoadGate()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: id),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
                load: { configuration, progressHandler in
                    let fakeProgress = Progress(totalUnitCount: 10)
                    fakeProgress.completedUnitCount = 5
                    progressHandler(fakeProgress)
                    await gate.signalReported()
                    await gate.waitForRelease()
                    return ModelContainer(
                        context: ModelContext(
                            configuration: configuration,
                            model: StubLanguageModel(),
                            processor: StubInputProcessor(),
                            tokenizer: ByteTokenizer()))
                })

            let loadTask = Task { try await model.loadContainer() }
            await gate.waitUntilReported()
            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) != nil
            }

            await gate.release()
            _ = try await loadTask.value

            await waitForSharedProgress {
                MLXDownloadProgress.shared.download(forModelID: id) == nil
            }

            // A successful load caches the container under `id`. Drop it so it
            // does not outlive this test in the process-global cache.
            await model.evict()
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
