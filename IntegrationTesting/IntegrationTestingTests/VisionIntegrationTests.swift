// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import FoundationModels
import IntegrationTestHelpers
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration

    /// Opt-in end-to-end VLM test: drives a real `gemma-4-e2b-it-4bit` through
    /// the FoundationModels adapter with a labeled image attachment and `.vision`
    /// declared, proving the labeled-attachment path reaches the already
    /// multimodal MLX pipeline.
    ///
    /// The input is a synthetic solid-color square built in-memory (no binary
    /// fixture); the test is parameterized over two colors and asserts the model
    /// names the matching color as a whole word. Two colors give an implicit
    /// negative control — a model that always answers "red" fails the blue case —
    /// and word-level matching keeps "colored"/"coloured" from satisfying a color
    /// name. This keeps the adapter end-to-end coverage while removing the
    /// photographic fixture.
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

        /// Colors exercised by ``namesImageColor(color:)``. `Sendable` with a plain
        /// `String` raw value so it's a valid parameterized-test argument (`CIColor`
        /// is not `Sendable`); `ciColor` feeds the image builder and `rawValue` is the
        /// word the response must contain.
        enum TestColor: String, CaseIterable, Sendable {
            case red, blue

            var ciColor: CIColor { self == .red ? .red : .blue }
        }

        @Test(arguments: TestColor.allCases)
        func namesImageColor(color: TestColor) async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                "mlx-community/gemma-4-e2b-it-4bit",
                capabilities: [.vision])
            let session = LanguageModelSession(model: model, tools: [], instructions: nil)
            let image = VisionTestImages.solidColor(color.ciColor)
            let response = try await session.respond {
                "What color is this image? Reply with just the color name."
                Attachment(image).label("color")
            }
            // Whole-word match: split on non-letters so trailing punctuation ("red.")
            // still counts, while "colored"/"coloured" cannot satisfy a color name.
            let words = Set(
                response.content.lowercased()
                    .split(whereSeparator: { !$0.isLetter })
                    .map(String.init))
            #expect(
                words.contains(color.rawValue),
                "expected the model to name the color \(color.rawValue); got: \(response.content)")
        }
    }

#endif  // FoundationModelsIntegration
