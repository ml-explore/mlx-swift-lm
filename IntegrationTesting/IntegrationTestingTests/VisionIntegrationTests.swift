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
    /// The input is a synthetic solid-red square built in-memory (no binary
    /// fixture), and the test asserts the model names the color "red". This
    /// keeps the adapter end-to-end coverage while removing the photographic
    /// fixture.
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
        func namesImageColor() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                "mlx-community/gemma-4-e2b-it-4bit",
                capabilities: LanguageModelCapabilities(capabilities: [.vision]))
            let session = LanguageModelSession(model: model, tools: [], instructions: nil)
            let redImage = VisionTestImages.solidColor(.red)
            let response = try await session.respond {
                "What color is this image? Reply with just the color name."
                Attachment(redImage).label("color")
            }
            #expect(response.content.lowercased().contains("red"))
        }
    }

#endif  // FoundationModelsIntegration
