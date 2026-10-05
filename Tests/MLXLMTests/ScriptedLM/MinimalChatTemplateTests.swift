// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM
import Testing

struct MinimalChatTemplateTests {

    static let generator = DefaultMessageGenerator()

    static func messages(_ chat: [Chat.Message]) -> [[String: any Sendable]] {
        generator.generate(messages: chat)
    }

    @Test func `renders roles in order with end markers and a generation prompt`() throws {
        let tokenizer = ScriptedTokenizer(corpus: [])
        let segments = try tokenizer.render(
            messages: Self.messages([
                .system("be brief"), .user("hi"), .assistant("hello"), .tool("42"),
            ]))

        #expect(
            segments == [
                "<|system|>", "be brief", "<|end|>",
                "<|user|>", "hi", "<|end|>",
                "<|assistant|>", "hello", "<|end|>",
                "<|tool|>", "42", "<|end|>",
                "<|assistant|>",
            ])
    }

    /// The property `ChatSession` prompt-cache reuse relies on: one turn's prompt plus
    /// its generated tokens (ending in EOS) is a prefix of the next turn's rendering.
    @Test(arguments: [Set<String>(), ["<|assistant|>", "<|user|>"]])
    func `a turn's prompt and reply prefix the next turn`(nonAtomic: Set<String>) throws {
        let reply = "fine, you?"
        let tokenizer = ScriptedTokenizer(
            corpus: ["how are you?", reply, "good"],
            template: MinimalChatTemplate(nonAtomicMarkers: nonAtomic))
        let eos = try #require(tokenizer.eosTokenId)

        let turn1: [Chat.Message] = [.user("how are you?")]
        let prompt1 = try tokenizer.applyChatTemplate(messages: Self.messages(turn1))
        let generated = tokenizer.encode(text: reply, addSpecialTokens: false) + [eos]

        let turn2 = turn1 + [.assistant(reply), .user("good")]
        let prompt2 = try tokenizer.applyChatTemplate(messages: Self.messages(turn2))

        #expect(prompt2.starts(with: prompt1 + generated))
    }

    @Test func `tools and tool calls render as canonical JSON`() throws {
        let tokenizer = ScriptedTokenizer(corpus: [])
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": ["name": "get_weather", "description": "Weather"]
                    as [String: any Sendable],
            ]
        ]
        let call = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: ["location": "Tokyo", "days": 2] as [String: any Sendable]))
        let messages = Self.messages([.assistant("", toolCalls: [call])])

        let segments = try tokenizer.render(messages: messages, tools: tools)
        let again = try tokenizer.render(messages: messages, tools: tools)
        #expect(segments == again)

        #expect(segments.first == "<|tools|>")
        #expect(
            segments[1]
                == #"[{"function":{"description":"Weather","name":"get_weather"},"type":"function"}]"#
        )
        #expect(segments.contains("<|tool_call|>"))
        #expect(
            segments.contains(#"{"arguments":{"days":2,"location":"Tokyo"},"name":"get_weather"}"#))
    }

    @Test func `add_generation_prompt false omits the trailing assistant marker`() throws {
        let tokenizer = ScriptedTokenizer(corpus: [])
        let segments = try tokenizer.render(
            messages: Self.messages([.user("hi")]),
            additionalContext: ["add_generation_prompt": false])

        #expect(segments.last == "<|end|>")
    }

    @Test func `content parts are concatenated`() throws {
        let tokenizer = ScriptedTokenizer(corpus: [])
        let parts: [[String: any Sendable]] = [
            ["type": "text", "text": "a"], ["type": "image"], ["type": "text", "text": "b"],
        ]
        let segments = try tokenizer.render(messages: [["role": "user", "content": parts]])

        #expect(segments.contains("ab"))
    }

    @Test func `unknown roles throw`() {
        let tokenizer = ScriptedTokenizer(corpus: [])
        #expect(throws: ScriptedTemplateError.unknownRole("narrator")) {
            try tokenizer.render(messages: [["role": "narrator", "content": "once"]])
        }
    }
}
