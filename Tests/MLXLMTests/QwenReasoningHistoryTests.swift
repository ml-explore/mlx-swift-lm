// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class QwenReasoningHistoryTests: XCTestCase {
    private func primedAssistant(_ content: String, delimiter: String = "<think>") -> Chat.Message {
        var message = Chat.Message.assistant(content)
        message.prefilledReasoningStartDelimiter = delimiter
        return message
    }

    private func text(in message: MLXLMCommon.Message) throws -> String {
        let parts = try XCTUnwrap(message["content"] as? [[String: String]])
        return parts.filter { $0["type"] == "text" }.compactMap { $0["text"] }.joined()
    }

    func testCompletedReasoningIsSeparatedWithoutTrimming() throws {
        for message in [
            primedAssistant("  thought\n</think>\n\n answer "),
            .assistant("<think>  thought\n</think>\n\n answer "),
        ] {
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertEqual(raw["reasoning_content"] as? String, "  thought\n")
            XCTAssertEqual(try text(in: raw), "\n\n answer ")
        }
    }

    func testEmptyCompletedReasoningIsPresent() throws {
        for message in [primedAssistant("</think>answer"), .assistant("<think></think>answer")] {
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertEqual(raw["reasoning_content"] as? String, "")
            XCTAssertEqual(try text(in: raw), "answer")
        }
    }

    func testPrimedReasoningPreservesAGeneratedOpeningDelimiter() throws {
        let raw = Qwen3VLMessageGenerator().generate(
            message: primedAssistant("<think>nested opening</think>answer"))
        XCTAssertEqual(raw["reasoning_content"] as? String, "<think>nested opening")
        XCTAssertEqual(try text(in: raw), "answer")
    }

    func testUnmarkedLiteralAndIncompleteContentRemainUnchanged() throws {
        let messages: [Chat.Message] = [
            .assistant("The closing tag is </think>, followed by an answer."),
            .assistant("An example: <think>thought</think>answer"),
            .assistant(" <think>thought</think>answer"),
            .assistant("```xml\n<think>thought</think>\n```"),
            .assistant("thought</think>answer"),
            .assistant("<think>unfinished"),
            primedAssistant("unfinished"),
            primedAssistant("unfinished</thi"),
            primedAssistant("thought<tool_call>"),
            primedAssistant("thought</think>answer", delimiter: "<reason>"),
        ]
        for message in messages {
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertNil(raw["reasoning_content"], message.content)
            XCTAssertEqual(try text(in: raw), message.content)
        }
    }

    func testImplicitToolBoundaryDoesNotInventAReasoningClose() throws {
        let call = ToolCall(
            function: .init(name: "read_file", arguments: ["path": .string("README.md")]))
        var message = Chat.Message.assistant("unfinished thought", toolCalls: [call])
        message.prefilledReasoningStartDelimiter = "<think>"
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        XCTAssertNil(raw["reasoning_content"])
        XCTAssertEqual(try text(in: raw), message.content)
        let calls = try XCTUnwrap(raw["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
    }

    func testOnlyAssistantMessagesAreNormalized() throws {
        for role in [Chat.Message.Role.user, .system, .tool] {
            var message = Chat.Message(role: role, content: "<think>thought</think>answer")
            message.prefilledReasoningStartDelimiter = "<think>"
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertEqual(raw["role"] as? String, role.rawValue)
            XCTAssertNil(raw["reasoning_content"])
            XCTAssertEqual(try text(in: raw), message.content)
        }
    }

    func testNormalizationPreservesMediaAndToolMetadata() throws {
        let call = ToolCall(
            function: .init(
                name: "read_file", arguments: ["path": .string("README.md"), "offset": .int(1)]),
            id: "call_1")
        var message = Chat.Message.assistant(
            "thought</think>answer",
            images: [.url(URL(fileURLWithPath: "/tmp/reasoning-image.png"))],
            videos: [.url(URL(fileURLWithPath: "/tmp/reasoning-video.mp4"))],
            toolCalls: [call])
        message.prefilledReasoningStartDelimiter = "<think>"
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        let parts = try XCTUnwrap(raw["content"] as? [[String: String]])
        XCTAssertEqual(parts.map { $0["type"] }, ["image", "video", "text"])
        XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
        XCTAssertEqual(try text(in: raw), "answer")
        let calls = try XCTUnwrap(raw["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0]["id"] as? String, "call_1")
        let function = try XCTUnwrap(calls[0]["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "read_file")
        let arguments = try XCTUnwrap(function["arguments"] as? [String: any Sendable])
        XCTAssertEqual(arguments["path"] as? String, "README.md")
        XCTAssertEqual(arguments["offset"] as? Int, 1)

        let tool = Chat.Message.tool("<think>literal</think>", id: "call_1", name: "read_file")
        let rawTool = Qwen3VLMessageGenerator().generate(message: tool)
        XCTAssertEqual(rawTool["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(rawTool["name"] as? String, "read_file")
        XCTAssertEqual(try text(in: rawTool), tool.content)
        XCTAssertNil(rawTool["reasoning_content"])
    }

    func testOtherMessageGeneratorsKeepTheirExistingContent() throws {
        let message = primedAssistant("thought</think>answer")
        let standard = DefaultMessageGenerator().generate(message: message)
        XCTAssertEqual(standard["content"] as? String, message.content)
        XCTAssertNil(standard["reasoning_content"])
        let qwen2 = Qwen2VLMessageGenerator().generate(message: message)
        XCTAssertEqual(try text(in: qwen2), message.content)
        XCTAssertNil(qwen2["reasoning_content"])
    }

    func testSessionRetainsPromptPrimingAndKeepsReasoningBesideTheAnswer() async throws {
        let output = "thought</think>\n\nanswer"
        for whitespace in [
            "\n \t", String(repeating: " ", count: 61), String(repeating: " ", count: 130),
        ] {
            // The stream delivers reasoning apart from the answer, so the session
            // returns and retains the answer alone and keeps the thought beside it.
            let retained = try await retainedAssistant(
                promptTail: "<think>" + whitespace, output: output, visibleOutput: "answer")
            XCTAssertEqual(retained.content, "answer")
            XCTAssertEqual(retained.reasoning, "thought")
            XCTAssertEqual(
                retained.prefilledReasoningStartDelimiter, "<think>",
                "Trailing whitespace length: \(whitespace.count)")
            let raw = Qwen3VLMessageGenerator().generate(message: retained)
            XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
            XCTAssertEqual(try text(in: raw), "answer")
        }
    }

    func testSessionKeepsGeneratedReasoningBesideTheAnswer() async throws {
        // Not primed: the model opens the thought itself.
        let retained = try await retainedAssistant(
            promptTail: "", output: "<think>thought</think>answer", visibleOutput: "answer")
        XCTAssertEqual(retained.content, "answer")
        XCTAssertEqual(retained.reasoning, "thought")
        XCTAssertNil(retained.prefilledReasoningStartDelimiter)
        let raw = Qwen3VLMessageGenerator().generate(message: retained)
        XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
        XCTAssertEqual(try text(in: raw), "answer")

        // Other generators replay it too; templates that do not read the key ignore it.
        let other = DefaultMessageGenerator().generate(message: retained)
        XCTAssertEqual(other["content"] as? String, "answer")
        XCTAssertEqual(other["reasoning_content"] as? String, "thought")
    }

    func testRecordedReasoningWinsOverTagsInTheAnswer() throws {
        var message = Chat.Message.assistant("write <think>x</think> to think")
        message.reasoning = "thought"
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
        XCTAssertEqual(try text(in: raw), message.content)
    }

    func testSessionDoesNotInferPrimingFromEarlierOrClosedTags() async throws {
        for tail in ["", "<think>\n\n</think>\n\n", "<think>quoted example\nassistant\n"] {
            let retained = try await retainedAssistant(promptTail: tail, output: "literal</think>")
            XCTAssertNil(retained.prefilledReasoningStartDelimiter, tail)
            let raw = Qwen3VLMessageGenerator().generate(message: retained)
            XCTAssertNil(raw["reasoning_content"])
            XCTAssertEqual(try text(in: raw), retained.content)
        }
    }

    func testSessionRequiresTheConfiguredDelimiter() async throws {
        let unconfigured = try await retainedAssistant(
            promptTail: "<think>\n", output: "thought</think>", reasoning: nil)
        XCTAssertNil(unconfigured.prefilledReasoningStartDelimiter)

        let custom = ReasoningConfig(
            startDelimiter: "<reason>", endDelimiter: "</reason>", promptStrategy: .alwaysOn)
        let mismatched = try await retainedAssistant(
            promptTail: "<think>\n", output: "thought</think>", reasoning: custom)
        XCTAssertNil(mismatched.prefilledReasoningStartDelimiter)

        // Primed with the custom delimiter and never closed: all of it is reasoning.
        let matching = try await retainedAssistant(
            promptTail: "<reason>\n", output: "literal</think>", visibleOutput: "",
            reasoning: custom)
        XCTAssertEqual(matching.prefilledReasoningStartDelimiter, "<reason>")
        XCTAssertNil(matching.reasoning)
        let raw = Qwen3VLMessageGenerator().generate(message: matching)
        XCTAssertNil(raw["reasoning_content"])
        XCTAssertEqual(try text(in: raw), matching.content)
    }

    func testLengthLimitedOpenReasoningIsNotCompletedByTheGenerator() async throws {
        // A thought cut off by the length limit is streamed as reasoning but not
        // retained: replaying it would present it as a finished thought.
        let retained = try await retainedAssistant(
            promptTail: "<think>\n", output: "unfinished", visibleOutput: "", maxTokens: 4)
        XCTAssertEqual(retained.content, "")
        XCTAssertNil(retained.reasoning)
        XCTAssertEqual(retained.prefilledReasoningStartDelimiter, "<think>")
        let raw = Qwen3VLMessageGenerator().generate(message: retained)
        XCTAssertNil(raw["reasoning_content"])
        XCTAssertEqual(try text(in: raw), "")
    }

    /// A regenerated prompt can extend the cache by no more than its prefill, as
    /// when thinking is switched on after a thinking-off turn. Priming must be
    /// read from the whole prompt, not from the suffix the cache leaves to prefill.
    func testPrimingIsReadFromTheWholePromptNotTheUncachedSuffix() async throws {
        let tokenizer = ByteTokenizer()
        let output = "thought</think>answer"
        let outputTokens = tokenizer.encode(text: output, addSpecialTokens: false) + [256]
        let processor = TailProcessor(
            tails: ["<think>\n\n</think>\n\n", "<think>\n "], tokenizer: tokenizer)
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "reasoning-priming-test", eosTokenIds: [256],
                reasoningConfig: QwenReasoningProtocol.tagged),
            model: ScriptedModel(output: outputTokens), processor: processor, tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: outputTokens.count, temperature: 0))

        // Thinking off: the prompt closed its block, so the stray close is answer text.
        let first = try await session.respond(to: "first")
        XCTAssertEqual(first, output)

        // Thinking on: the prompt shares everything up to `<think>\n` with the cache
        // and adds one token, so only that token is prefilled. It is still primed.
        let second = try await session.respond(to: "second")
        XCTAssertEqual(second, "answer")
    }

    /// Renders a generation prompt alone, with a different tail on each call.
    private struct TailProcessor: UserInputProcessor {
        let tails: [String]
        let tokenizer: ByteTokenizer
        let calls = CallCounter()

        func prepare(input: UserInput) throws -> LMInput {
            let tail = tails[min(calls.next(), tails.count - 1)]
            return LMInput(
                tokens: MLXArray(
                    tokenizer.encode(text: "assistant:\n" + tail, addSpecialTokens: false)))
        }
    }

    private final class CallCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func next() -> Int {
            lock.lock()
            defer { lock.unlock() }
            let value = count
            count += 1
            return value
        }
    }

    private struct RecordedMessage: Sendable {
        let role: Chat.Message.Role
        let content: String
        let prefilledReasoningStartDelimiter: String?
        let reasoning: String?
    }

    private struct RecordingProcessor: UserInputProcessor {
        let promptTail: String
        let tokenizer: ByteTokenizer
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func prepare(input: UserInput) throws -> LMInput {
            guard case .chat(let messages) = input.prompt else {
                throw TokenizerError.missingChatTemplate
            }
            continuation.yield(
                messages.map {
                    RecordedMessage(
                        role: $0.role, content: $0.content,
                        prefilledReasoningStartDelimiter: $0.prefilledReasoningStartDelimiter,
                        reasoning: $0.reasoning)
                })
            let transcript = messages.map { "\($0.role.rawValue): \($0.content)\n" }.joined()
            let prompt = transcript + "assistant:\n" + promptTail
            return LMInput(
                tokens: MLXArray(tokenizer.encode(text: prompt, addSpecialTokens: false)))
        }
    }

    private struct ByteTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        var eosTokenId: Int? { 256 }
        var unknownTokenId: Int? { 257 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? {
            token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            UInt8(exactly: id).map { String(decoding: [$0], as: UTF8.self) }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    private final class ScriptedModel: Module, LanguageModel, KVCacheDimensionProvider {
        let output: [Int]
        private var index = 0
        var kvHeads: [Int] { [1] }

        init(output: [Int]) {
            self.output = output
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            index = 0
            return .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let length = inputs.size
            if let cache = cache?.first {
                let entries = MLXArray.zeros([1, 1, length, 1])
                _ = cache.update(keys: entries, values: entries)
            }
            let token = output[min(index, output.count - 1)]
            index += 1
            var logits = Array(repeating: Float(-100), count: length * 258)
            for position in 0 ..< length { logits[position * 258 + token] = 100 }
            return MLXArray(logits, [1, length, 258])
        }
    }

    private func retainedAssistant(
        promptTail: String, output: String, visibleOutput: String? = nil,
        reasoning: ReasoningConfig? = QwenReasoningProtocol.tagged,
        maxTokens: Int? = nil
    ) async throws -> Chat.Message {
        let (observations, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let tokenizer = ByteTokenizer()
        let outputTokens = tokenizer.encode(text: output, addSpecialTokens: false) + [256]
        let processor = RecordingProcessor(
            promptTail: promptTail, tokenizer: tokenizer, continuation: continuation)
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "reasoning-history-test", eosTokenIds: [256], reasoningConfig: reasoning),
            model: ScriptedModel(output: outputTokens), processor: processor, tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(
                maxTokens: maxTokens ?? outputTokens.count, temperature: 0))
        let firstOutput = try await session.respond(to: "first")
        XCTAssertEqual(
            firstOutput, visibleOutput ?? String(output.prefix(maxTokens ?? output.count)))
        _ = try await session.respond(to: "second")
        continuation.finish()
        var calls: [[RecordedMessage]] = []
        for await messages in observations { calls.append(messages) }
        XCTAssertEqual(calls.count, 2)
        let assistant = try XCTUnwrap(calls.last?.first { $0.role == .assistant })
        var message = Chat.Message.assistant(assistant.content)
        message.prefilledReasoningStartDelimiter = assistant.prefilledReasoningStartDelimiter
        message.reasoning = assistant.reasoning
        return message
    }
}
