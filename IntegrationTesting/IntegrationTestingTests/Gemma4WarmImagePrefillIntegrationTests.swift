// Copyright © 2026 Apple Inc.

import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// With `use_bidirectional_attention: "vision"` (31B, 26B-A4B, 12B unified), the soft tokens of
// one image attend to each other in both directions. A prefill that continues a warm cache with
// an image must keep that. E2B does not set the flag and is the control.
//
// Cache-gated: each test runs only when its checkpoint is already in the HF cache.

private let bidirectionalTarget = "mlx-community/gemma-4-31b-it-4bit"
private let unifiedTarget = "mlx-community/gemma-4-12B-it-4bit"
private let controlTarget = "mlx-community/gemma-4-e2b-it-4bit"

extension ModelLoadingSuite {
    @Suite(.serialized) struct Gemma4WarmImagePrefillIntegrationTests {}
}

extension ModelLoadingSuite.Gemma4WarmImagePrefillIntegrationTests {

    @Test(.enabled(if: hfSnapshotDir(modelId: bidirectionalTarget) != nil))
    func bidirectionalWarmImageSuffixMatchesCold() async throws {
        try await expectWarmImageSuffixMatchesCold(modelId: bidirectionalTarget)
    }

    @Test(.enabled(if: hfSnapshotDir(modelId: unifiedTarget) != nil))
    func unifiedWarmImageSuffixMatchesCold() async throws {
        try await expectWarmImageSuffixMatchesCold(modelId: unifiedTarget)
    }

    @Test(.enabled(if: hfSnapshotDir(modelId: controlTarget) != nil))
    func controlWarmImageSuffixMatchesCold() async throws {
        try await expectWarmImageSuffixMatchesCold(modelId: controlTarget)
    }

    /// Prefill a text prefix cold, then the image and the rest warm, and compare the
    /// last-position logits with one cold prefill. A text-only prompt split the same way
    /// sets the bound.
    private func expectWarmImageSuffixMatchesCold(modelId: String) async throws {
        let dir = try #require(hfSnapshotDir(modelId: modelId))
        let context = try await TestModelManager.shared.load("vlm-context:\(dir.path)") {
            try await VLMModelFactory.shared.load(
                from: dir, using: #huggingFaceTokenizerLoader())
        }
        let model = context.model
        let splitter = try #require(model as? PreparedInputSplitting)

        let image = CIFilter.checkerboardGenerator()
        image.color0 = .red
        image.color1 = .blue
        image.width = 40
        let picture = try #require(image.outputImage)
            .cropped(to: CGRect(x: 0, y: 0, width: 320, height: 240))
        let history: [Chat.Message] = [
            .user("Name a primary color."),
            .assistant("Red."),
        ]
        let question = "Describe the pattern in this image in one sentence."

        let full = try await context.processor.prepare(
            input: UserInput(chat: history + [.user(question, images: [.ciImage(picture)])]))
        let text = try await context.processor.prepare(
            input: UserInput(chat: history + [.user(question)]))

        let fullIds = full.text.tokens.asArray(Int.self)
        let textIds = text.text.tokens.asArray(Int.self)
        let sharedPrefix = zip(fullIds, textIds).prefix { $0 == $1 }.count
        // Cut a few tokens into the last turn, before the image block opens.
        let cut = sharedPrefix - 4
        try #require(cut > 0)

        let textDrift = try warmDrift(
            model: model, full: text, warmPrefix: LMInput(tokens: text.text.tokens[0..., ..<cut]),
            warmSuffix: LMInput(tokens: text.text.tokens[0..., cut...]))

        let suffix = try #require(splitter.splitPreparedInput(full, droppingFirst: cut))
        let imageDrift = try warmDrift(
            model: model, full: full,
            warmPrefix: LMInput(tokens: full.text.tokens[0..., ..<cut]), warmSuffix: suffix)

        // The text-only drift of the same split, with headroom. The floor covers a text-only
        // run that is exact by chance.
        let bound = max(1.25 * textDrift, 0.1)
        #expect(
            imageDrift <= bound,
            "\(modelId): image drift \(imageDrift) exceeds \(bound) (text-only \(textDrift))")
    }

    private func warmDrift(
        model: any LanguageModel, full: LMInput, warmPrefix: LMInput, warmSuffix: LMInput
    ) throws -> Float {
        let cold = try lastLogits(
            model.prepare(
                full, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 512)))

        let cache = try model.newCache(parameters: nil)
        _ = try model.prepare(warmPrefix, cache: cache, state: nil, prefill: .init(stepSize: 512))
        let warm = try lastLogits(
            model.prepare(warmSuffix, cache: cache, state: nil, prefill: .init(stepSize: 512)))

        return abs(warm - cold).max().item(Float.self)
    }

    private func lastLogits(_ result: PrepareResult) throws -> MLXArray {
        guard case .logits(let output) = result else {
            throw Gemma4WarmImagePrefillError.expectedLogits
        }
        let logits = output.logits[0..., -1, 0...].asType(.float32)
        eval(logits)
        return logits
    }
}

private enum Gemma4WarmImagePrefillError: Error {
    case expectedLogits
}
