// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

extension ModelLoadingSuite {
    @Suite struct EmptyToolCallContentIntegrationTests {}
}

extension ModelLoadingSuite.EmptyToolCallContentIntegrationTests {
    private func tokenizer(id: String, revision: String) async throws -> any MLXLMCommon.Tokenizer {
        let downloader: any Downloader = #hubDownloader()
        let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let directory = try await downloader.download(
            id: id, revision: revision, matching: ["*.json", "*.jinja"],
            useLatest: false, progressHandler: { _ in })
        return try await loader.load(from: directory)
    }

    @Test func ministral3AcceptsEmptyMessagesAndToolCalls() async throws {
        let tokenizer = try await tokenizer(
            id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            revision: "a962dcb09eee4169c890e544c9eb938f1113fdee")
        let generator = Mistral3MessageGenerator()
        let user = generator.generate(message: .user("Hello"))
        for role in ["user", "assistant"] {
            let empty: MLXLMCommon.Message = ["role": role, "content": [[String: String]]()]
            #expect(throws: (any Error).self) {
                try tokenizer.applyChatTemplate(messages: role == "user" ? [empty] : [user, empty])
            }
        }
        for assistant: Chat.Message in [.assistant(""), .assistant("", toolCalls: [])] {
            let messages = generator.generate(messages: [.user(""), assistant])
            let tokens = try tokenizer.applyChatTemplate(messages: messages)
            let prompt = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            #expect(prompt.contains("[INST][/INST]"))
        }
        let call = ToolCall(function: .init(name: "get_weather", arguments: [:]))
        let messages = generator.generate(messages: [
            .user("Weather in Rome?"), .assistant("", toolCalls: [call]),
        ])
        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        let prompt = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        #expect(prompt.contains("[TOOL_CALLS]get_weather[ARGS]{}"))
    }

    @Test func qwen3VLToolCallRendersLikeEmptyStringContent() async throws {
        let tokenizer = try await tokenizer(
            id: "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            revision: "9c4f5209e57b31f4b9dfba735de3fb983739c9cc")
        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": .string("Rome")]))
        let generator = Qwen3VLMessageGenerator()
        let user = generator.generate(message: .user("Weather in Rome?"))
        let assistant = generator.generate(message: .assistant("", toolCalls: [call]))
        let tokens = try tokenizer.applyChatTemplate(messages: [user, assistant])
        var stringAssistant = assistant
        stringAssistant["content"] = ""
        let reference = try tokenizer.applyChatTemplate(messages: [user, stringAssistant])
        #expect(tokens == reference)
        let prompt = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        #expect(prompt.contains("\n<tool_call>"))
        #expect(!prompt.contains("\n\n<tool_call>"))
        var legacyAssistant = assistant
        legacyAssistant["content"] = [["type": "text", "text": ""]]
        let legacy = try tokenizer.applyChatTemplate(messages: [user, legacyAssistant])
        #expect(legacy != reference)
    }
}
