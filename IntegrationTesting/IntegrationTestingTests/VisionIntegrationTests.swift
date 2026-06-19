// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration

    /// Opt-in end-to-end VLM test: drives a real `gemma-4-e2b-it-4bit` through
    /// the FoundationModels adapter with a labeled image attachment and `.vision`
    /// declared, proving the labeled-attachment path reaches the already
    /// multimodal MLX pipeline.
    ///
    /// Skipped unless `MLX_RUN_VLM_INTEGRATION=1`, so default CI never downloads
    /// multi-GB weights; run on Apple silicon on demand.
    ///
    /// The OS gate is an in-body `guard #available` rather than an `@available`
    /// on the suite: the swift-testing `@Suite`/`@Test` macros reject an
    /// availability-annotated declaration here, so this mirrors the runtime gate
    /// every other suite in this target uses (e.g. `IntegrationTests`).
    @Suite(
        .serialized,
        .timeLimit(.minutes(10)),
        .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_VLM_INTEGRATION"] == "1"))
    struct VisionIntegrationTests {

        @Test
        func describesAndReadsScoreboard() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                "mlx-community/gemma-4-e2b-it-4bit",
                capabilities: LanguageModelCapabilities(capabilities: [.vision]))
            let session = LanguageModelSession(model: model, tools: [], instructions: nil)
            let cgImage = try Self.loadScoreboardCGImage()

            // Turn 1: attach the image and ask for a description.
            let described = try await session.respond {
                "Describe this photo in one or two sentences."
                Attachment(cgImage).label("photo")
            }
            #expect(!described.content.isEmpty)

            // Turn 2: text-only follow-up. The turn-1 image rides forward via
            // the transcript history, which TranscriptConverter re-extracts, so
            // no re-attachment is needed. OCR legibility is bounded by Gemma4's
            // internal ~800x800 resize, so the assertion stays loose.
            let quarter = try await session.respond {
                "According to the scoreboard, what quarter is the football game in? Answer with just the quarter."
            }
            #expect(!quarter.content.isEmpty)
        }

        // MARK: - Fixture

        private enum FixtureError: Error { case unreadable(URL) }

        /// Resolves `Tests/MLXFoundationModelsTests/Fixtures/scoreboard.jpg`
        /// relative to this source file (mirrors the CXGrammarTests `#filePath`
        /// convention; no `Bundle.module`). The opt-in test only runs from an
        /// intact source checkout, so walking to the package root is safe.
        private static func scoreboardURL() -> URL {
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()  // IntegrationTestingTests
                .deletingLastPathComponent()  // IntegrationTesting
                .deletingLastPathComponent()  // <package root>
                .appendingPathComponent("Tests", isDirectory: true)
                .appendingPathComponent("MLXFoundationModelsTests", isDirectory: true)
                .appendingPathComponent("Fixtures", isDirectory: true)
                .appendingPathComponent("scoreboard.jpg")
        }

        private static func loadScoreboardCGImage() throws -> CGImage {
            let url = scoreboardURL()
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                throw FixtureError.unreadable(url)
            }
            return image
        }
    }

#endif  // FoundationModelsIntegration
