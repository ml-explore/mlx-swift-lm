// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// A genuine cold-fetch of a small model through the production `HubClient`
/// path, asserting availability moves absent -> `.available` across the download.
///
/// This is the counterpart to `HuggingFaceLanguageModelMacroSmoke`, which is
/// deliberately network-free. Here we DO hit the network, so the suite is
/// opt-in behind the `MLX_RUN_COLD_FETCH` environment flag and never runs in a
/// default sweep. Enable it explicitly by setting `MLX_RUN_COLD_FETCH=1` (on
/// device, `TEST_RUNNER_MLX_RUN_COLD_FETCH=1`, which Xcode forwards to the test
/// runner process).
///
/// Forcing "cold" correctly means controlling the loader's cache, not the
/// availability observer:
///
/// - The DOWNLOADER's cache decides cold-vs-warm. We hand `#hubDownloader(_:)`
///   a `HubClient` backed by a throwaway `HubCache` rooted in a fresh temp
///   directory, so nothing is present and the fetch must go to the network.
/// - `weightsLocation:` is only the availability observer. It MUST resolve
///   against that SAME throwaway cache; otherwise download and availability
///   look at different directories — precisely the mismatch the production
///   `HubCache`-based resolver was introduced to fix. Pointing it at a bogus or
///   default path would fake "not downloaded" without forcing a fetch, so it is
///   the wrong lever.
/// - The process-global in-memory `ModelCache` is the other confound: if some
///   other test already loaded this model, `loadContainer()` would return the
///   cached instance and never touch our throwaway cache. `evictAll()` before
///   the load, plus `.serialized` and isolated invocation, keeps the fetch cold.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_COLD_FETCH"] == "1"))
struct ColdHubClientFetchTests {

    /// The smallest model in the fixtures, to keep the real download cheap.
    private static let modelID = TestFixtures.gemmaModelID

    @Test("a cold HubClient fetch moves availability from not-downloaded to available")
    func coldFetchMakesModelAvailable() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // A throwaway HubClient cache: empty, so the fetch cannot be a cache hit.
        let cacheRoot = FileManager.default.temporaryDirectory
            .appending(path: "cold-hub-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cache = HuggingFace.HubCache(cacheDirectory: cacheRoot)
        defer { try? FileManager.default.removeItem(at: cacheRoot) }

        // `weightsLocation:` observes the SAME throwaway cache the downloader
        // writes into (mirrors the production `HubCache` resolver), so download
        // and availability agree.
        let weightsLocation: @Sendable (String) -> URL = { id in
            guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                return cache.cacheDirectory
            }
            if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                let snapshot = try? cache.snapshotPath(
                    repo: repo, kind: .model, commitHash: commit)
            {
                return snapshot
            }
            return cache.repoDirectory(repo: repo, kind: .model)
        }

        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: Self.modelID),
            capabilities: [],
            weightsLocation: weightsLocation,
            load: { configuration, progress in
                try await loadModelContainer(
                    from: #hubDownloader(HuggingFace.HubClient(cache: cache)),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: configuration, progressHandler: progress)
            })

        // Drop any in-memory entry so `loadContainer()` runs the load closure
        // (and therefore the download) instead of returning a cached instance.
        await MLXLanguageModel.evictAll()
        defer { Task { await MLXLanguageModel.evictAll() } }

        // Before: nothing on disk in the throwaway cache.
        let before = await model.availability
        if before == .unavailable(.deviceNotCapable) {
            // No Metal device (e.g. a non-Apple-silicon CI worker); the download
            // assertion is meaningless here.
            return
        }
        #expect(
            before == .unavailable(.modelNotDownloaded),
            "an empty throwaway cache must report .modelNotDownloaded, got \(before)")

        // Trigger the genuine fetch. Because the suite is opt-in, a network /
        // Hub outage surfaces as a recorded issue rather than a hang: if you
        // asked for the cold fetch and it could not reach the Hub, that is worth
        // seeing rather than a silent pass.
        do {
            try await model.preload()
        } catch {
            Issue.record(
                "cold HubClient fetch of \(Self.modelID) could not complete (network?): \(error)")
            return
        }

        // After: the download populated the throwaway cache, so the same
        // `weightsLocation` resolver now sees `config.json`.
        let after = await model.availability
        #expect(
            after == .available,
            "after a successful cold download the model must be .available, got \(after)")
    }
}

#endif  // FoundationModelsIntegration
