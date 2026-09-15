// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

/// Each generator names a labeled image immediately before that image, and keeps
/// the arrangement it had: GlmOcr puts its text first, and three of the six emit
/// video parts.
@Suite("Labeled image generators")
struct LabeledImageGeneratorTests {

    private func message(_ labels: [String?]) -> Chat.Message {
        .user(
            "which one is blue?",
            images: labels.map { .url(URL(fileURLWithPath: "/tmp/example.png"), label: $0) })
    }

    private func parts(_ raw: MLXLMCommon.Message) throws -> [[String: String]] {
        try #require(raw["content"] as? [[String: String]])
    }

    /// Images first, then the caller's text, with nothing between the parts.
    private var imagesThenText: [[String: String]] {
        [
            ["type": "text", "text": "[A]"],
            ["type": "image"],
            ["type": "text", "text": "[B]"],
            ["type": "image"],
            ["type": "text", "text": "which one is blue?"],
        ]
    }

    @Test("Qwen2VL names each image before itself")
    func qwen2VL() throws {
        let raw = Qwen2VLMessageGenerator().generate(message: message(["A", "B"]))
        let content = try parts(raw)
        #expect(content == imagesThenText)
        #expect(raw["role"] as? String == "user")
    }

    @Test("Qwen3VL names each image before itself")
    func qwen3VL() throws {
        let content = try parts(Qwen3VLMessageGenerator().generate(message: message(["A", "B"])))
        #expect(content == imagesThenText)
    }

    @Test("Gemma4 names each image before itself")
    func gemma4() throws {
        let content = try parts(Gemma4MessageGenerator().generate(message: message(["A", "B"])))
        #expect(content == imagesThenText)
    }

    @Test("Mistral3 names each image before itself")
    func mistral3() throws {
        let content = try parts(Mistral3MessageGenerator().generate(message: message(["A", "B"])))
        #expect(content == imagesThenText)
    }

    @Test("FastVLM names each image before itself")
    func fastVLM() throws {
        let content = try parts(FastVLMMessageGenerator().generate(message: message(["A", "B"])))
        #expect(content == imagesThenText)
    }

    @Test("GlmOcr keeps its text first and still names each image")
    func glmOcr() throws {
        let content = try parts(GlmOcrMessageGenerator().generate(message: message(["A", "B"])))
        #expect(
            content == [
                ["type": "text", "text": "which one is blue?"],
                ["type": "text", "text": "[A]"],
                ["type": "image"],
                ["type": "text", "text": "[B]"],
                ["type": "image"],
            ])
    }

    @Test("Gemma4 keeps a system message as a plain string")
    func gemma4SystemMessage() throws {
        let raw = Gemma4MessageGenerator().generate(message: .system("You are useful."))
        #expect(raw["content"] as? String == "You are useful.")
    }

    @Test("An unlabeled message keeps the array it had before labels")
    func unlabeledMessagesAreUnchanged() throws {
        let unlabeled = message([nil, nil])
        let qwen = try parts(Qwen2VLMessageGenerator().generate(message: unlabeled))
        #expect(
            qwen == [
                ["type": "image"],
                ["type": "image"],
                ["type": "text", "text": "which one is blue?"],
            ])
        let glm = try parts(GlmOcrMessageGenerator().generate(message: unlabeled))
        #expect(
            glm == [
                ["type": "text", "text": "which one is blue?"],
                ["type": "image"],
                ["type": "image"],
            ])
    }
}
