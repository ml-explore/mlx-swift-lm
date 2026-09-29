// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import FoundationModels
import IntegrationTestHelpers
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

let labeledVisionModels = [
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
]

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct ColorReport {
    var image: ImageReference
    var color: String
}

/// Skipped unless `MLX_RUN_VLM_INTEGRATION=1`, so default CI never downloads
/// multi-GB weights; run on Apple silicon on demand.
/// If you run `xcodebuild`, prefix the variable with `TEST_RUNNER_`, or every test is skipped.
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

        let response = try await session.respond(
            generating: ColorReport.self,
            options: GenerationOptions(samplingMode: .greedy)
        ) {
            "Report which label goes with the blue image, and name its color."
            Attachment(VisionTestImages.solidColor(.red)).label("Aurora")
            Attachment(VisionTestImages.solidColor(.blue)).label("Beacon")
        }

        // A model may write `[Beacon]`, and the SDK resolves that form too, so do not
        // require an exact match.
        let label = response.content.image.attachmentLabel
        #expect(
            label.contains("Beacon"),
            "expected the blue image's name; got: \(label)")
        let resolved = response.content.image.resolved(in: session.transcript)
        #expect(resolved != nil, "expected the reference to resolve to an attachment")
    }
}

#endif  // FoundationModelsIntegration
