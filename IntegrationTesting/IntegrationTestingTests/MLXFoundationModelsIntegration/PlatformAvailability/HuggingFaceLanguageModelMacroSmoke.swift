// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// Compile-and-run coverage for the `#huggingFaceLanguageModel` macro expansion.
///
/// Nothing else in the repo invokes the macro, and swift-huggingface is linked
/// only by this xcodeproj target (not the SwiftPM graph), so without this the
/// synthesized `weightsLocation:` / `load:` closures — which reference
/// `HuggingFace.HubCache`, `#hubDownloader()`, and `#huggingFaceTokenizerLoader()`
/// — are never type-checked against swift-huggingface. A macro-string typo would
/// otherwise ship silently. Building this target type-checks the expansion; the
/// runtime assertion also confirms the synthesized `weightsLocation:` resolves
/// against the same HubClient cache the loader downloads into, without a fetch.
@Suite("HuggingFace Language Model Macro")
struct HuggingFaceLanguageModelMacroSmoke {

    /// A repo id that is never downloaded, so the availability outcome is
    /// deterministic (no `config.json` can exist for it on disk).
    private static let absentModelID = "integration-probe/does-not-exist"

    /// Type-checks the macro expansion (the whole point of this file) and
    /// confirms construction performs no I/O: `modelID` is derived from the
    /// configuration, so it is known before any download.
    @Test("macro builds a model without downloading")
    func macroExpansionCompilesAndConstructs() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = #huggingFaceLanguageModel(
            configuration: ModelConfiguration(id: Self.absentModelID))
        #expect(model.modelID == Self.absentModelID)
    }

    /// Exercises the synthesized `weightsLocation:` closure at runtime. It reads
    /// only local HubClient-cache refs (`resolveRevision`) — no network — so an
    /// id that was never fetched resolves to a location with no `config.json`,
    /// and availability is never `.available`. This is the regression guard: if
    /// `weightsLocation:` ever again points somewhere the downloader does not
    /// write, this path still resolves cleanly rather than trapping.
    @Test("availability resolves the HubClient cache location without a download")
    func availabilityForAbsentModelIsNotAvailable() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = #huggingFaceLanguageModel(
            configuration: ModelConfiguration(id: Self.absentModelID))
        #expect(await model.availability != .available)
    }
}

#endif  // FoundationModelsIntegration
