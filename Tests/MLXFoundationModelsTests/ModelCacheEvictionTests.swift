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
                    modelID: id,
                    capabilities: LanguageModelCapabilities(capabilities: []),
                    from: BlockingDownloader(gate: gate),
                    using: EvictStubTokenizerLoader(),
                    locatedBy: { _ in URL(fileURLWithPath: "/no/such/path/\(UUID().uuidString)") }
                )

                // Drive a load that parks, then fails — populating lastErrors[id].
                let loadTask = Task { try? await model.preload() }
                await gate.waitUntilStarted()
                await gate.release()
                _ = await loadTask.value

                let before = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
                #expect(before != nil, "a failed load should record a cached lastError")

                await MLXLanguageModel.evictAll()

                let after = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
                #expect(after == nil, "evictAll() must clear the cached lastError")
            }

            @Test("a load evicted mid-flight does not re-populate the cache on completion")
            func evictedInFlightLoadDoesNotRepopulate() async throws {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let id = "org/superseded-\(UUID().uuidString)"
                let gate = LoadGate()
                let model = MLXLanguageModel(
                    modelID: id,
                    capabilities: LanguageModelCapabilities(capabilities: []),
                    from: BlockingDownloader(gate: gate),
                    using: EvictStubTokenizerLoader(),
                    locatedBy: { _ in URL(fileURLWithPath: "/no/such/path/\(UUID().uuidString)") }
                )

                // Park a genuine (non-warmup) load in flight.
                let loadTask = Task { try? await model.preload() }
                await gate.waitUntilStarted()

                // In-flight load is registered and reported.
                let downloadingDuring = await MLXLanguageModel.isDownloadingInCache(modelID: id)
                #expect(downloadingDuring, "a parked load should report as downloading")

                // Evict while the load is suspended — removes loadingTasks[id].
                await MLXLanguageModel.evictAll()

                let downloadingAfterEvict = await MLXLanguageModel.isDownloadingInCache(modelID: id)
                #expect(!downloadingAfterEvict, "evictAll() must drop the in-flight load registration")

                // Let the parked load fail. The catch-path guard must NOT re-add lastError
                // for the now-superseded task.
                await gate.release()
                _ = await loadTask.value

                let lastError = await MLXLanguageModel.lastLoadErrorInCache(modelID: id)
                #expect(lastError == nil, "a superseded load must not re-populate cache state")
            }
        }
    }

    // MARK: - Shared fixtures
    //
    // Hoisted to file scope so the eviction tests above (and Tasks 2-3's tests, which
    // land in the same `CacheEviction` suite in this file) share one definition. Kept
    // file-`private` so they don't collide with AvailabilityTests.swift's own stubs.

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
    private final class BlockingDownloader: Downloader, @unchecked Sendable {
        private let gate: LoadGate
        init(gate: LoadGate) { self.gate = gate }
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

    private final class EvictStubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
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
