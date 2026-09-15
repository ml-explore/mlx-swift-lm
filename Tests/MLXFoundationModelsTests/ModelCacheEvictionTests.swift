// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

// Single registration for both suites nested under `FoundationModelsCacheTests`
// (this file + AvailabilityTests.swift): register the real factory so
// loadModelContainer reaches the injected stub downloader instead of throwing
// .noModelFactoryAvailable before the in-flight gate can fire. This target links
// MLXLLM but references no MLXLLM symbol, so the linker can dead-strip its
// TrampolineModelFactory; ModelFactoryRegistry seeds itself purely via
// NSClassFromString("MLXLLM.TrampolineModelFactory"), which then resolves to nil —
// an empty registry. Registering explicitly (which also hard-references
// LLMModelFactory, defeating the dead-strip) guarantees the load path reaches the
// injected stub downloader.
let registerModelFactoryOnce: Void = {
    ModelFactoryRegistry.shared.addTrampoline { LLMModelFactory.shared }
}()

// Serialized parent so the cache-touching suites below never run concurrently.
// `MLXLanguageModel` holds one process-global `static let cache`; `evictAll()` is
// key-agnostic, so an eviction in one suite would wipe the parked-load windows the
// sibling availability suite asserts against. `.serialized` on a single suite only
// orders that suite's own tests — it does NOT order two top-level suites against
// each other — so both cache-touching suites are nested under this one serialized
// parent. AvailabilityTests extends this same type from its own file.
@Suite(.serialized)
struct FoundationModelsCacheTests {}

extension FoundationModelsCacheTests {

    @Suite("MLXLanguageModel cache eviction")
    struct CacheEviction {

        init() { _ = registerModelFactoryOnce }

        @Test("evictAll() clears a failed load's cached lastError")
        func evictAllClearsLastError() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/evictall-\(UUID().uuidString)"
            let gate = LoadGate()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: id),
                capabilities: [],
                weightsLocation: { _ in
                    URL(fileURLWithPath: "/no/such/path/\(UUID().uuidString)")
                },
                load: { configuration, progress in
                    try await loadModelContainer(
                        from: BlockingDownloader(gate: gate), using: EvictStubTokenizerLoader(),
                        configuration: configuration, progressHandler: progress)
                })

            // Drive a load that parks, then fails — populating lastErrors[id].
            let loadTask = Task { try await model.preload() }
            await gate.waitUntilStarted()
            await gate.release()
            await #expect(throws: BlockingDownloaderReleased.self) {
                try await loadTask.value
            }

            let before = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
            #expect(before != nil, "a failed load should record a cached lastError")

            await MLXLanguageModel.evictAll()

            let after = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
            #expect(after == nil, "evictAll() must clear the cached lastError")
        }

        @Test("evict() clears only this model's state, leaving other models cached")
        func evictIsPerModel() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            func failedLoad(_ id: String) async -> MLXLanguageModel {
                let gate = LoadGate()
                let model = MLXLanguageModel(
                    configuration: ModelConfiguration(id: id),
                    capabilities: [],
                    weightsLocation: { _ in
                        URL(fileURLWithPath: "/no/such/path/\(UUID().uuidString)")
                    },
                    load: { configuration, progress in
                        try await loadModelContainer(
                            from: BlockingDownloader(gate: gate),
                            using: EvictStubTokenizerLoader(),
                            configuration: configuration, progressHandler: progress)
                    })
                let task = Task { try await model.preload() }
                await gate.waitUntilStarted()
                await gate.release()
                await #expect(throws: BlockingDownloaderReleased.self) {
                    try await task.value
                }
                return model
            }

            let idA = "org/per-model-a-\(UUID().uuidString)"
            let idB = "org/per-model-b-\(UUID().uuidString)"
            let modelA = await failedLoad(idA)
            _ = await failedLoad(idB)

            // Both models have a cached lastError.
            #expect(await MLXLanguageModel.lastLoadErrorInCache(modelID: idA) != nil)
            #expect(await MLXLanguageModel.lastLoadErrorInCache(modelID: idB) != nil)

            await modelA.evict()

            #expect(
                await MLXLanguageModel.lastLoadErrorInCache(modelID: idA) == nil,
                "evict() must clear this model's cached state")
            #expect(
                await MLXLanguageModel.lastLoadErrorInCache(modelID: idB) != nil,
                "evict() must NOT clear other models' cached state")
        }

        @Test("a load evicted mid-flight does not re-populate the cache on completion")
        func evictedInFlightLoadDoesNotRepopulate() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/superseded-\(UUID().uuidString)"
            let gate = LoadGate()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: id),
                capabilities: [],
                weightsLocation: { _ in
                    URL(fileURLWithPath: "/no/such/path/\(UUID().uuidString)")
                },
                load: { configuration, progress in
                    try await loadModelContainer(
                        from: BlockingDownloader(gate: gate), using: EvictStubTokenizerLoader(),
                        configuration: configuration, progressHandler: progress)
                })

            // Park a genuine (non-warmup) load in flight.
            let loadTask = Task { try await model.preload() }
            await gate.waitUntilStarted()

            // In-flight load is registered and reported.
            let downloadingDuring = await MLXLanguageModel.isDownloadingInCache(modelID: id)
            #expect(downloadingDuring, "a parked load should report as downloading")

            // Evict while the load is suspended — removes loadingTasks[id].
            await MLXLanguageModel.evictAll()

            let downloadingAfterEvict = await MLXLanguageModel.isDownloadingInCache(modelID: id)
            #expect(
                !downloadingAfterEvict, "evictAll() must drop the in-flight load registration")

            // Let the parked load fail. The catch-path guard must NOT re-add lastError
            // for the now-superseded task.
            await gate.release()
            await #expect(throws: BlockingDownloaderReleased.self) {
                try await loadTask.value
            }

            let lastError = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
            #expect(lastError == nil, "a superseded load must not re-populate cache state")
        }

        @Test(
            "evict() cancels the model's registered load task",
            .timeLimit(.minutes(1)))
        func evictCancelsRegisteredLoad() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let probe = CancellationProbe()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: "org/cancel-evict-\(UUID().uuidString)"),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                load: makeCancellationProbingLoader(probe: probe))

            let loadTask = Task { try await model.preload() }
            await probe.waitUntilParked()

            await model.evict()

            await #expect(throws: CancellationError.self) {
                try await loadTask.value
            }
            #expect(await probe.wasCancelled)
        }

        @Test(
            "evictAll() cancels every registered load task",
            .timeLimit(.minutes(1)))
        func evictAllCancelsRegisteredLoads() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let probe = CancellationProbe()
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: "org/cancel-evictall-\(UUID().uuidString)"),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                load: makeCancellationProbingLoader(probe: probe))

            let loadTask = Task { try await model.preload() }
            await probe.waitUntilParked()

            await MLXLanguageModel.evictAll()

            await #expect(throws: CancellationError.self) {
                try await loadTask.value
            }
            #expect(await probe.wasCancelled)
        }
    }
}

// MARK: - Shared fixtures
//
// Hoisted to file scope so the eviction tests above share one definition. Kept
// file-`private` so they don't collide with AvailabilityTests.swift's own stubs.

/// Builds a container loader that parks until the cache cancels its load task.
/// The loader never produces a container: the test awaits its failure, so the
/// probe reads back without any timing assumption.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeCancellationProbingLoader(
    probe: CancellationProbe
) -> MLXLanguageModel.ContainerLoader {
    { _, _ in
        await withTaskCancellationHandler {
            await probe.parkUntilCancelled()
        } onCancel: {
            // A cancellation handler cannot await, so hop onto the probe actor to
            // record the cancellation and resume the parked continuation.
            Task { await probe.recordCancellation() }
        }
        try Task.checkCancellation()
        throw ParkedLoaderResumedWithoutCancellation()
    }
}

/// Records the cancellation of a registered load task, and parks the loader until
/// that cancellation arrives.
private actor CancellationProbe {
    private(set) var wasCancelled = false
    private var isParked = false
    private var parkContinuation: CheckedContinuation<Void, Never>?
    private var parkedWaiter: CheckedContinuation<Void, Never>?

    /// Suspends the loader until ``recordCancellation()`` runs.
    func parkUntilCancelled() async {
        if wasCancelled { return }
        isParked = true
        parkedWaiter?.resume()
        parkedWaiter = nil
        await withCheckedContinuation { parkContinuation = $0 }
    }

    /// Suspends until the loader parks, so a test evicts only after the cache has
    /// registered the load task.
    func waitUntilParked() async {
        if isParked { return }
        await withCheckedContinuation { parkedWaiter = $0 }
    }

    func recordCancellation() {
        wasCancelled = true
        parkContinuation?.resume()
        parkContinuation = nil
    }
}

/// Thrown if the parked loader resumes without a cancellation. Only
/// `CancellationProbe.recordCancellation()` resumes it, so this error means the
/// probe misbehaved rather than the cache.
private struct ParkedLoaderResumedWithoutCancellation: Error {}

/// Coordinates the in-flight window: the downloader signals when a load has entered
/// (so the load task is registered), then parks until the test releases it.
private actor LoadGate {
    private var startedAlready = false
    private var startedContinuation: CheckedContinuation<Void, Never>?
    private var releasedAlready = false
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func signalStarted() {
        startedAlready = true
        startedContinuation?.resume()
        startedContinuation = nil
    }
    func waitUntilStarted() async {
        if startedAlready { return }
        await withCheckedContinuation { startedContinuation = $0 }
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

private struct BlockingDownloaderReleased: Error {}

/// A `Downloader` that parks inside `download` until the gate is released, so a load
/// stays deterministically in flight, then fails the load on release.
private struct BlockingDownloader: Downloader {
    let gate: LoadGate
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        await gate.signalStarted()
        await gate.waitForRelease()
        throw BlockingDownloaderReleased()
    }
}

private struct EvictStubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { EvictStubTokenizer() }
}

private struct EvictStubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
