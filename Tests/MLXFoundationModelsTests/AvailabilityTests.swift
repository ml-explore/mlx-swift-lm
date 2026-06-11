// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    // This target links MLXLLM but references no MLXLLM symbol, so the linker can
// dead-strip its TrampolineModelFactory. ModelFactoryRegistry seeds itself purely
// via NSClassFromString("MLXLLM.TrampolineModelFactory"), which then resolves to
// nil — an empty registry. With no factory, loadModelContainer throws
// .noModelFactoryAvailable *before* reaching the downloader, so the in-flight
// gate these tests await never fires and the suite deadlocks. Registering the
// factory explicitly (which also hard-references LLMModelFactory, defeating the
// dead-strip) guarantees the load path reaches the injected stub downloader.
private let registerModelFactoryOnce: Void = {
    ModelFactoryRegistry.shared.addTrampoline { LLMModelFactory.shared }
}()

@Suite("MLXLanguageModel availability")
struct AvailabilityTests {

    init() { _ = registerModelFactoryOnce }

    @Test("returns .unavailable(.modelNotDownloaded) when the configured weights path is missing")
    func missingOnDisk() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = MLXLanguageModel(
            modelIdentifier: "org/repo",
            capabilities: LanguageModelCapabilities(capabilities: []),
            from: StubAvailabilityDownloader(),
            using: StubAvailabilityTokenizerLoader(),
            locatedBy: { _ in URL(fileURLWithPath: "/definitely/not/a/real/path") }
        )

        let availability = await model.availability
        if case .unavailable(.modelNotDownloaded) = availability {
            // expected
        } else if case .unavailable(.deviceNotCapable) = availability {
            // Acceptable when the test runs on a host with no Metal device
            // (e.g. a non-Apple-silicon CI worker). The plain
            // `.modelNotDownloaded` path is unreachable on those hosts.
        } else {
            Issue.record(
                "expected .unavailable(.modelNotDownloaded) or .deviceNotCapable, got \(availability)"
            )
        }
    }

    // MARK: - Prewarm `.downloading` suppression
    //
    // These exercise the availability state machine deterministically on the
    // host: a blocking downloader parks a load in flight (the load task and its
    // suppression tag are registered synchronously before the await), and we
    // read `availability` during that window. The contrast between `warmUp()`
    // (suppressed) and `preload()` (not suppressed) on the *same* already-present
    // model isolates the suppress flag as the only varying input.

    @Test("warmUp of an already-present model does NOT flip availability to .downloading")
    func warmupOfPresentModelStaysAvailable() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Suppression lives past the device-capability gate; skip where there's
        // no Metal device (availability short-circuits to .deviceNotCapable).
        guard MLXLanguageModel.isDeviceCapable else { return }

        let dir = try makePresentModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gate = LoadGate()
        let model = MLXLanguageModel(
            modelIdentifier: "org/warmup-present-\(UUID().uuidString)",
            capabilities: LanguageModelCapabilities(capabilities: []),
            from: BlockingDownloader(gate: gate),
            using: StubAvailabilityTokenizerLoader(),
            locatedBy: { _ in dir }
        )

        let warmTask = Task { try? await model.warmUp() }
        await gate.waitUntilStarted()

        let availability = await model.availability
        await gate.release()
        _ = await warmTask.value

        #expect(
            availability == .available,
            "A warmup of an already-present model must stay .available, got \(availability)")
    }

    @Test("a genuine (non-warmup) load of a present-but-unloaded model DOES report .downloading")
    func genuineLoadOfPresentModelReportsDownloading() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        guard MLXLanguageModel.isDeviceCapable else { return }

        let dir = try makePresentModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gate = LoadGate()
        let model = MLXLanguageModel(
            modelIdentifier: "org/genuine-present-\(UUID().uuidString)",
            capabilities: LanguageModelCapabilities(capabilities: []),
            from: BlockingDownloader(gate: gate),
            using: StubAvailabilityTokenizerLoader(),
            locatedBy: { _ in dir }
        )

        // preload() is NOT a warmup, so its in-flight load is not suppressed —
        // proving the suppression is what differs (and that the real
        // `.downloading` signal is not regressed).
        let loadTask = Task { try? await model.preload() }
        await gate.waitUntilStarted()

        let availability = await model.availability
        await gate.release()
        _ = await loadTask.value

        #expect(
            availability == .downloading,
            "A genuine in-flight load must report .downloading, got \(availability)")
    }

    @Test("warmUp of a not-yet-downloaded model still reports the genuine fetch as .downloading")
    func warmupOfAbsentModelReportsDownloading() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        guard MLXLanguageModel.isDeviceCapable else { return }

        // Absent on disk: warmUp's suppress condition (warmup AND on-disk) is
        // false, so the genuine fetch is reported.
        let missing = URL(fileURLWithPath: "/definitely/not/a/real/path/\(UUID().uuidString)")
        let gate = LoadGate()
        let model = MLXLanguageModel(
            modelIdentifier: "org/warmup-absent-\(UUID().uuidString)",
            capabilities: LanguageModelCapabilities(capabilities: []),
            from: BlockingDownloader(gate: gate),
            using: StubAvailabilityTokenizerLoader(),
            locatedBy: { _ in missing }
        )

        let warmTask = Task { try? await model.warmUp() }
        await gate.waitUntilStarted()

        let availability = await model.availability
        await gate.release()
        _ = await warmTask.value

        #expect(
            availability == .downloading,
            "A warmup that triggers a genuine fetch must report .downloading, got \(availability)")
    }

    // MARK: - Helpers

    /// A temp directory containing a `config.json`, so `modelExistsOnDisk()`
    /// reports the model as present (independent of the downloader path).
    private func makePresentModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "present-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appending(path: "config.json"))
        return dir
    }
}

// MARK: - Test Stubs

private final class StubAvailabilityDownloader: Downloader, @unchecked Sendable {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        URL(fileURLWithPath: "/tmp/\(id)")
    }
}

private final class StubAvailabilityTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any Tokenizer {
        StubAvailabilityTokenizer()
    }
}

private struct StubAvailabilityTokenizer: Tokenizer {
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
    ) throws -> [Int] {
        []
    }
}

/// Coordinates the in-flight window for the suppression tests: the downloader
/// signals when a load has entered (so the load task + suppression tag are
/// registered), and parks until the test releases it.
private actor LoadGate {
    private var startedAlready = false
    private var startedContinuation: CheckedContinuation<Void, Never>?
    private var releasedAlready = false
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    /// Called by the downloader on entry — the load is now in flight.
    func signalStarted() {
        startedAlready = true
        startedContinuation?.resume()
        startedContinuation = nil
    }

    /// Awaited by the test until the load is in flight.
    func waitUntilStarted() async {
        if startedAlready { return }
        await withCheckedContinuation { startedContinuation = $0 }
    }

    /// Awaited by the downloader until the test releases it.
    func waitForRelease() async {
        if releasedAlready { return }
        await withCheckedContinuation { releaseContinuation = $0 }
    }

    /// Called by the test to unblock the parked download (which then fails the
    /// load — the tests only assert on the in-flight window, not completion).
    func release() {
        releasedAlready = true
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private struct BlockingDownloaderReleased: Error {}

/// A `Downloader` that parks inside `download` until the gate is released, so a
/// load stays deterministically in flight while the test reads `availability`.
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
        // We never reach real weights; fail the load now that the test has read
        // the in-flight state.
        throw BlockingDownloaderReleased()
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
