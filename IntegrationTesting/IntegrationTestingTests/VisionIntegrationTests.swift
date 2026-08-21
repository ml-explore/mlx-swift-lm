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
/// Qwen3-VL is deliberately not in this list. `mlx-community/Qwen3-VL-4B-Instruct-4bit`
/// cannot load: its `model.safetensors.index.json` names
/// `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`,
/// neither of which the repo ships, and declares about 8.9 GB while the repo
/// contains a single 3.1 GB `model.safetensors`. The 8.9 GB figure is the
/// unquantized size, so the index was carried over from the source repo and
/// never regenerated for the quantized upload. The same is true of
/// `Qwen3-VL-8B-Instruct-4bit` (names four shards, ships two) and
/// `Qwen3-VL-4B-Instruct-8bit` (names two, ships one), so it is the
/// quantization batch rather than one repo. `Qwen3-VL-2B-Instruct-4bit` is
/// packaged correctly if a Qwen3-VL model is ever wanted here.
///
/// Two things to know if you chase this again. The error reported through
/// `loadModelContainer`'s factory fallback reads as
/// `.unsupportedModelType("qwen3_vl")`, which is misleading: the real failure is
/// a file-not-found on the missing shard, and the LLM factory, which has no
/// qwen3_vl entry, is what reports last. And the weights themselves are fine:
/// pointing the loader at a copy of the directory with the index removed loads
/// the model and answers correctly, because `safetensorWeightURLs` only falls
/// back to globbing `*.safetensors` when no index is present.
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
/// multi-GB weights; run on Apple silicon on demand.
///
/// Setting that variable in your shell is not enough under `xcodebuild`, which
/// does not forward the parent environment into the test process. It has to be
/// prefixed, and `xcodebuild` strips the prefix on the way in:
///
/// ```
/// TEST_RUNNER_MLX_RUN_VLM_INTEGRATION=1 xcodebuild test \
///   -project IntegrationTesting/IntegrationTesting.xcodeproj \
///   -scheme IntegrationTesting -destination 'platform=macOS,name=My Mac' \
///   -skipPackagePluginValidation -parallel-testing-enabled NO \
///   -only-testing:IntegrationTestingTests/VisionIntegrationTests
/// ```
///
/// Without the prefix every test here reports as skipped and the run still ends
/// in `** TEST SUCCEEDED **`, which reads exactly like a pass.
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
    func generatedImageReferenceResolvesBackToTheAttachment(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(
            modelID, capabilities: [.vision, .guidedGeneration])
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        // Greedy for the same reason as the test above: which of two labels the
        // model picks is a choice, and sampling makes that choice unstable.
        // Nothing constrains the label to the two present ones, so this only
        // rules out sampling noise, not the model inventing or misspelling a name.
        let response = try await session.respond(
            generating: ColorReport.self,
            options: GenerationOptions(samplingMode: .greedy)
        ) {
            "Report which label goes with the blue image, and name its color."
            Attachment(VisionTestImages.solidColor(.red)).label("Photo_A1B2C3")
            Attachment(VisionTestImages.solidColor(.blue)).label("Photo_D4E5F6")
        }

        // The prompt names each image as `[label]` (see
        // AttachmentLabelRenderer), matching what FoundationModels itself emits,
        // and `ImageReference.resolved(in:)` matches that label exactly before
        // falling back to extracting a bracketed substring. Nothing constrains
        // what the model writes here: `attachmentLabel` is an unconstrained
        // `String` in FoundationModels, so a model that paraphrased the label
        // instead of echoing it would fail the assertion below rather than being
        // prevented from paraphrasing.
        #expect(response.content.image.attachmentLabel == "Photo_D4E5F6")
        let resolved = response.content.image.resolved(in: session.transcript)
        #expect(resolved != nil, "expected the reference to resolve to an attachment")
    }
}

#endif  // FoundationModelsIntegration
