// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM
import Testing

struct ChatMLTemplateTests {

    static func messages(_ chat: [Chat.Message]) -> [[String: any Sendable]] {
        DefaultMessageGenerator().generate(messages: chat)
    }

    @Test func `renders ChatML turns with role names as text`() throws {
        let tokenizer = ScriptedTokenizer(corpus: [], template: ChatMLTemplate())
        let segments = try tokenizer.render(
            messages: Self.messages([.system("be brief"), .user("hi"), .assistant("hello")]))

        #expect(
            segments == [
                "<|im_start|>", "system\n", "be brief",
                "<|im_end|>", "\n",
                "<|im_start|>", "user\n", "hi",
                "<|im_end|>", "\n",
                "<|im_start|>", "assistant\n", "hello",
                "<|im_end|>", "\n",
                "<|im_start|>", "assistant\n",
            ])
        #expect(
            segments.joined()
                == "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
                + "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n")
    }

    @Test func `the template supplies its own tokens`() {
        // The scenario lists only its own text; the template adds its markers and role names.
        let tokenizer = ScriptedTokenizer(corpus: ["hi"], template: ChatMLTemplate())

        #expect(tokenizer.eosToken == "<|im_end|>")
        #expect(tokenizer.eosTokenId == tokenizer.convertTokenToId("<|im_end|>"))
        #expect(tokenizer.vocabulary.isSpecial(tokenizer.convertTokenToId("<|im_start|>")!))
        let assistant = tokenizer.encode(text: "assistant\n", addSpecialTokens: false)
        #expect(assistant.contains { tokenizer.vocabulary.entry($0)?.kind == .piece })
    }

    @Test func `a turn's prompt and reply prefix the next turn`() throws {
        let reply = "fine, you?"
        let tokenizer = ScriptedTokenizer(
            corpus: ["how are you?", reply, "good"], template: ChatMLTemplate())
        let eos = try #require(tokenizer.eosTokenId)

        let turn1: [Chat.Message] = [.user("how are you?")]
        let prompt1 = try tokenizer.applyChatTemplate(messages: Self.messages(turn1))
        let generated = tokenizer.encode(text: reply, addSpecialTokens: false) + [eos]

        let turn2 = turn1 + [.assistant(reply), .user("good")]
        let prompt2 = try tokenizer.applyChatTemplate(messages: Self.messages(turn2))

        #expect(prompt2.starts(with: prompt1 + generated))
    }

    @Test func `tools and tool calls are unsupported`() {
        let tokenizer = ScriptedTokenizer(corpus: [], template: ChatMLTemplate())
        let call = ToolCall(
            function: .init(name: "f", arguments: [:] as [String: any Sendable]))

        #expect(throws: ScriptedTemplateError.unsupported("tools")) {
            try tokenizer.render(
                messages: Self.messages([.user("hi")]), tools: [["type": "function"]])
        }
        #expect(throws: ScriptedTemplateError.unsupported("tool_calls")) {
            try tokenizer.render(messages: Self.messages([.assistant("", toolCalls: [call])]))
        }
    }

    @Test func `a message without a role throws`() {
        let tokenizer = ScriptedTokenizer(corpus: [], template: ChatMLTemplate())
        #expect(throws: ScriptedTemplateError.unknownRole("")) {
            try tokenizer.render(messages: [["content": "orphan"]])
        }
    }
}
