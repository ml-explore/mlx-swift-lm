// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import FoundationModels
import IntegrationTestHelpers
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// The two vision models this coverage runs against. Both were used to choose
/// the label rendering, so both are worth holding to it.
///
/// Known issue: `mlx-community/Qwen3-VL-4B-Instruct-4bit` currently fails to
/// load, and the failure is in the published repo, not here. Its
/// `model.safetensors.index.json` references `model-00001-of-00002.safetensors`
/// and `model-00002-of-00002.safetensors`, neither of which the repo ships,
/// and declares a total size of about 8.9 GB while the repo actually contains
/// a single 3.1 GB `model.safetensors`. The 8.9 GB figure is the unquantized
/// size, so the index looks like it was carried over from the source repo and
/// never regenerated for the quantized upload. The error surfaced through
/// `loadModelContainer`'s factory fallback chain reads as
/// `.unsupportedModelType("qwen3_vl")`, which is a red herring: the real error
/// is a file-not-found on the missing shard, masked because the LLM factory
/// (which has no qwen3_vl entry) is what reports last. This also affects the
/// pre-existing color-naming test above, so the condition predates these two
/// vision-labeling tests.
let labeledVisionModels = [
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
]

/// Response type for ``VisionIntegrationTests/generatedImageReferenceResolvesBackToTheAttachment``.
/// The `ImageReference` field is the round trip under test: the model names an
/// input image and the app resolves that name back to the pixels.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct ColorReport {
    var image: ImageReference
    var color: String
}

/// Opt-in end-to-end VLM test: drives a real `Qwen3-VL-4B-Instruct-4bit` through
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
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
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

    @Test(arguments: labeledVisionModels)
    func namesTheLabelOfTheRequestedImage(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(modelID, capabilities: [.vision])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        // Two labeled images, asked about by color rather than by position, so a
        // model that always answers with the first label it saw fails.
        let response = try await session.respond {
            "Which label goes with the blue image? Reply with only the label."
            Attachment(VisionTestImages.solidColor(.red)).label("Photo_A1B2C3")
            Attachment(VisionTestImages.solidColor(.blue)).label("Photo_D4E5F6")
        }
        let text = response.content
        #expect(
            text.contains("D4E5F6"),
            "expected the blue image's label; got: \(text)")
        #expect(
            !text.contains("A1B2C3"),
            "expected only the blue image's label; got: \(text)")
    }

    @Test(arguments: labeledVisionModels)
    func generatedImageReferenceResolvesBackToTheAttachment(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(
            modelID, capabilities: [.vision, .guidedGeneration])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        let response = try await session.respond(generating: ColorReport.self) {
            "Report which label goes with the blue image, and name its color."
            Attachment(VisionTestImages.solidColor(.red)).label("Photo_A1B2C3")
            Attachment(VisionTestImages.solidColor(.blue)).label("Photo_D4E5F6")
        }

        // The schema pins the label to the transcript's labels, so this must be
        // one of them and must resolve. Without that constraint the model is
        // free to paraphrase and the lookup returns nil.
        #expect(response.content.image.attachmentLabel == "Photo_D4E5F6")
        let resolved = response.content.image.resolved(in: session.transcript)
        #expect(resolved != nil, "expected the reference to resolve to an attachment")
    }
}

#endif  // FoundationModelsIntegration
