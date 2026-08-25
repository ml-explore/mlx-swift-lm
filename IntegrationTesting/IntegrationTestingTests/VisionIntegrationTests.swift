// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import FoundationModels
import IntegrationTestHelpers
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// How well a model repeats an image's name back depends on the model. Measured
/// 2026-08-24: Qwen2.5-VL 32B repeats `Photo_D4E5F6` exactly, the 7B and 3B shorten
/// or ignore it, and Qwen3-VL-2B answered differently on all six runs, so it is not
/// in this list. None of the three layouts we tried changed that, which is why the
/// test below names its images with one word: both models here repeat that exactly.
let labeledVisionModels = [
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
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

/// Opt-in end-to-end VLM tests: drive real vision models through the
/// FoundationModels adapter with labeled image attachments and `.vision`
/// declared, proving the labeled-attachment path reaches the already multimodal
/// MLX pipeline. Three tests live here: one names the color of a single image,
/// one picks which of two labeled images is a given color, and one takes a
/// generated reference to an input image and resolves it back to the attachment.
///
/// Inputs are synthetic solid-color squares built in-memory, no binary fixtures.
/// The color test is parameterized over two colors and asserts the model names
/// the matching color as a whole word. Two colors give an implicit negative
/// control, since a model that always answers "red" fails the blue case, and
/// word-level matching keeps "colored" and "coloured" from satisfying a color
/// name.
///
/// Skipped unless `MLX_RUN_VLM_INTEGRATION=1`, so default CI never downloads
/// multi-GB weights; run on Apple silicon on demand. Under `xcodebuild` the
/// variable needs a `TEST_RUNNER_` prefix, or every test here reports as skipped.
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
            "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            capabilities: [.vision])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)
        let image = VisionTestImages.solidColor(color.ciColor)
        // Greedy, matching the two tests below, so a sampled answer cannot make
        // this fail intermittently.
        let response = try await session.respond(
            options: GenerationOptions(samplingMode: .greedy)
        ) {
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
        //
        // Greedy sampling is required, not a preference. Measured on
        // Qwen2.5-VL-7B under default sampling this prompt answers correctly
        // only about four times in six: it otherwise names the red image or
        // returns a fragment of its label. Greedy answered identically six times
        // out of six on both models. Apple's own guidance says to use greedy
        // when you want the most likely option rather than something close to it.
        let response = try await session.respond(
            options: GenerationOptions(samplingMode: .greedy)
        ) {
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
    func namesTheLabelOfTheMiddleOfThreeImages(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(modelID, capabilities: [.vision])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        // The middle image, so neither "answer with the first label" nor "answer
        // with the last label" passes.
        let response = try await session.respond(
            options: GenerationOptions(samplingMode: .greedy)
        ) {
            "Which label goes with the blue image? Reply with only the label."
            Attachment(VisionTestImages.solidColor(.red)).label("Photo_A1B2C3")
            Attachment(VisionTestImages.solidColor(.blue)).label("Photo_D4E5F6")
            Attachment(VisionTestImages.solidColor(.green)).label("Photo_G7H8I9")
        }
        let text = response.content
        #expect(
            text.contains("D4E5F6"),
            "expected the blue image's label; got: \(text)")
        #expect(
            !text.contains("A1B2C3") && !text.contains("G7H8I9"),
            "expected only the blue image's label; got: \(text)")
    }

    @Test(arguments: labeledVisionModels)
    func generatedImageReferenceResolvesBackToTheAttachment(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(
            modelID, capabilities: [.vision, .guidedGeneration])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        // Greedy, because which name the model picks is a choice, and sampling
        // makes that choice unstable.
        // Nothing constrains the label to the two present ones, so this only
        // rules out sampling noise, not the model inventing or misspelling a name.
        let response = try await session.respond(
            generating: ColorReport.self,
            options: GenerationOptions(samplingMode: .greedy)
        ) {
            "Report which label goes with the blue image, and name its color."
            Attachment(VisionTestImages.solidColor(.red)).label("Aurora")
            Attachment(VisionTestImages.solidColor(.blue)).label("Beacon")
        }

        // The app needs the reference to find the picture. A model may write
        // `[Beacon]`, which the SDK's bracket fallback resolves, so do not require
        // an exact string.
        let label = response.content.image.attachmentLabel
        #expect(
            label.contains("Beacon"),
            "expected the blue image's name; got: \(label)")
        let resolved = response.content.image.resolved(in: session.transcript)
        #expect(resolved != nil, "expected the reference to resolve to an attachment")
    }
}

#endif  // FoundationModelsIntegration
