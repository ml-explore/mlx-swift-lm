// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

private let templateDownloader: any Downloader = #hubDownloader()
private let templateTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

/// One model family's rendered-prompt contract for labeled images.
///
/// `placeholder` is the text the family's chat template emits for one image
/// part, before a processor expands it into per-patch tokens.
struct LabeledPromptFamily: Sendable, CustomStringConvertible {
    let family: String
    let modelID: String
    let revision: String
    let generator: any MessageGenerator
    let placeholder: String

    var description: String { family }
}

/// Ten families at pinned revisions. Six generator types serve all ten, because
/// `Qwen2VLMessageGenerator` is shared by Qwen25VL, Gemma3, LFM2VL and
/// MuseGlimmer. SmolVLM2 also shares it and is deliberately absent: its template
/// concatenates content as a string, so an array reaches it as a dumped
/// dictionary list whenever an image is present, which is broken independently
/// of labels and filed separately.
let labeledPromptFamilies: [LabeledPromptFamily] = [
    .init(
        family: "Qwen2VL", modelID: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        revision: "01af461cdb9574acc09084a0ef94e216e142b085",
        generator: Qwen2VLMessageGenerator(),
        placeholder: "<|vision_start|><|image_pad|><|vision_end|>"),
    .init(
        family: "Qwen25VL", modelID: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        revision: "46d4cf06a06ffc1a766c214174f9cbed2f45bcab",
        generator: Qwen2VLMessageGenerator(),
        placeholder: "<|vision_start|><|image_pad|><|vision_end|>"),
    .init(
        family: "Qwen3VL", modelID: "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        revision: "9c4f5209e57b31f4b9dfba735de3fb983739c9cc",
        generator: Qwen3VLMessageGenerator(),
        placeholder: "<|vision_start|><|image_pad|><|vision_end|>"),
    .init(
        family: "Gemma4", modelID: "mlx-community/gemma-4-e4b-it-4bit",
        revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
        generator: Gemma4MessageGenerator(), placeholder: "<|image|>"),
    .init(
        family: "Gemma3", modelID: "mlx-community/gemma-3-4b-it-qat-4bit",
        revision: "3d9ef289111449933c22761961f16a5df237ce2a",
        generator: Qwen2VLMessageGenerator(), placeholder: "<start_of_image>"),
    .init(
        family: "GlmOcr", modelID: "mlx-community/GLM-OCR-4bit",
        revision: "97f587506984cc92fa69b2694b4128e53db6b081",
        generator: GlmOcrMessageGenerator(),
        placeholder: "<|begin_of_image|><|image|><|end_of_image|>"),
    .init(
        family: "Mistral3",
        modelID: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        revision: "46135ef3c556bfed61013d8789bd26af02e416c4",
        generator: Mistral3MessageGenerator(), placeholder: "[IMG]"),
    .init(
        family: "FastVLM", modelID: "mlx-community/FastVLM-0.5B-bf16",
        revision: "81ffe929046666c43de53691147b1669ba0f3a4c",
        generator: FastVLMMessageGenerator(), placeholder: "<image>"),
    .init(
        family: "LFM2VL", modelID: "mlx-community/LFM2-VL-1.6B-4bit",
        revision: "be587c055231b0905e846f2c99803dd6d7f33a7a",
        generator: Qwen2VLMessageGenerator(), placeholder: "<image>"),
    .init(
        family: "MuseGlimmer", modelID: "mlx-community/Muse-Glimmer-30B-4bit",
        revision: "3e7677d7a40d348a3daba263a2b1c0aa41910710",
        generator: Qwen2VLMessageGenerator(), placeholder: "<|patch|>"),
]

/// Applies each family's real chat template to a two-image labeled message and
/// asserts on the rendered prompt. Weights are never downloaded: a template
/// needs the tokenizer only, so this costs a few megabytes of JSON per family.
@Suite(.serialized)
struct AttachmentLabelTemplateIntegrationTests {

    static let prose = "which one is blue?"

    private func tokenizer(for family: LabeledPromptFamily) async throws -> any MLXLMCommon
        .Tokenizer
    {
        let directory = try await templateDownloader.download(
            id: family.modelID,
            revision: family.revision,
            matching: ["*.json", "*.jinja"],
            useLatest: false,
            progressHandler: { _ in })
        return try await templateTokenizerLoader.load(from: directory)
    }

    private func rendered(
        _ family: LabeledPromptFamily, content: [[String: String]]
    ) async throws -> String {
        let tokenizer = try await tokenizer(for: family)
        let messages: [MLXLMCommon.Message] = [["role": "user", "content": content]]
        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        return tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    }

    @Test(arguments: labeledPromptFamilies)
    func labelLandsBesideItsImage(family: LabeledPromptFamily) async throws {
        let message = Chat.Message.user(
            Self.prose,
            images: [
                .url(URL(fileURLWithPath: "/tmp/a.png"), label: "A"),
                .url(URL(fileURLWithPath: "/tmp/b.png"), label: "B"),
            ])
        let parts = try #require(
            family.generator.generate(message: message)["content"] as? [[String: String]])
        let prompt = try await rendered(family, content: parts)

        #expect(
            prompt.contains("[A]" + family.placeholder),
            "\(family): first label must precede its own image placeholder; got: \(prompt)")
        #expect(
            prompt.contains("[B]" + family.placeholder),
            "\(family): second label must precede its own image placeholder; got: \(prompt)")
        #expect(
            prompt.contains(Self.prose),
            "\(family): the caller's text must survive; got: \(prompt)")
        #expect(
            prompt.components(separatedBy: family.placeholder).count - 1 == 2,
            "\(family): expected one placeholder per image; got: \(prompt)")
    }

    @Test(arguments: labeledPromptFamilies)
    func noSeparatorReachesThePrompt(family: LabeledPromptFamily) async throws {
        let message = Chat.Message.user(
            Self.prose,
            images: [
                .url(URL(fileURLWithPath: "/tmp/a.png"), label: "A"),
                .url(URL(fileURLWithPath: "/tmp/b.png"), label: "B"),
            ])
        let parts = try #require(
            family.generator.generate(message: message)["content"] as? [[String: String]])
        let prompt = try await rendered(family, content: parts)

        #expect(
            !prompt.contains("\u{2060}"),
            "\(family): a word joiner reached the prompt; got: \(prompt)")
        #expect(
            !prompt.contains("\u{200B}"),
            "\(family): a zero width space reached the prompt; got: \(prompt)")
    }

    /// A label emitted after its image would still pass a check for `[A]` alone.
    @Test(arguments: labeledPromptFamilies)
    func noLabelTrailsItsImage(family: LabeledPromptFamily) async throws {
        let message = Chat.Message.user(
            Self.prose,
            images: [
                .url(URL(fileURLWithPath: "/tmp/a.png"), label: "A"),
                .url(URL(fileURLWithPath: "/tmp/b.png"), label: "B"),
            ])
        let parts = try #require(
            family.generator.generate(message: message)["content"] as? [[String: String]])
        let prompt = try await rendered(family, content: parts)

        #expect(
            !prompt.contains(family.placeholder + "[A]"),
            "\(family): the first label must not follow a placeholder; got: \(prompt)")
    }
}
