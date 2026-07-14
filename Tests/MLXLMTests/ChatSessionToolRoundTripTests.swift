// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Regression test for the tool-call round trip: after dispatching tool calls,
/// the restarted generation must see BOTH the assistant message that made the
/// calls and the role:"tool" results. Chat templates that render tool results
/// by forward-scanning from an assistant message carrying `tool_calls`
/// (e.g. Gemma 4) otherwise drop the results from the rendered prompt.
public class ChatSessionToolRoundTripTests: XCTestCase {

    /// Ignores token ids and decodes a scripted string instead: the first
    /// generation pass yields a JSON tool call, later passes yield plain text.
    /// Stateful across passes, hence a class; calls are serialized by the
    /// session's generation task.
    private final class ScriptedToolCallTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private let toolCallScript =
            #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
        private let plainScript = "It is sunny in Paris today."
        private var pass = 0

        let vocabularySize = 100
        private let _eosTokenId = 101
        private let _unknownTokenId = 102

        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { _eosTokenId }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { _unknownTokenId }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            pass += 1
            return encode(text: "")
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            (0 ..< 8).map { _ in Int.random(in: 1 ..< vocabularySize) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let script = pass <= 1 ? toolCallScript : plainScript
            return String(script.prefix(min(tokenIds.count * 4, script.count)))
        }

        func convertTokenToId(_ token: String) -> Int? { 1 }
        func convertIdToToken(_ id: Int) -> String? { "x" }
    }

    private final class MessageLog: @unchecked Sendable {
        private let lock = NSLock()
        private var passes: [[Chat.Message]] = []

        func record(_ messages: [Chat.Message]) {
            lock.lock()
            passes.append(messages)
            lock.unlock()
        }

        var all: [[Chat.Message]] {
            lock.lock()
            defer { lock.unlock() }
            return passes
        }
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let log: MessageLog

        func generate(messages: [Chat.Message]) -> [Message] {
            log.record(messages)
            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    private final class DispatchLog: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [ToolCall] = []

        func record(_ call: ToolCall) {
            lock.lock()
            calls.append(call)
            lock.unlock()
        }

        var all: [ToolCall] {
            lock.lock()
            defer { lock.unlock() }
            return calls
        }
    }

    private static func makeContext(
        tokenizer: any Tokenizer, messageGenerator: MessageGenerator
    ) -> ModelContext {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test", toolCallFormat: .json),
            messageGenerator: messageGenerator)
        return .init(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)
    }

    func testRestartedGenerationSeesAssistantToolCallsBeforeResults() async throws {
        let log = MessageLog()
        let dispatched = DispatchLog()
        let tokenizer = ScriptedToolCallTokenizer()
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))

        let weatherTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 24),
            tools: [weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast": "sunny"}"#
            })

        let reply = try await session.respond(to: "What's the weather in Paris?")
        XCTAssertFalse(reply.isEmpty)

        // The scripted tool call was parsed and dispatched once.
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(dispatched.all.first?.function.name, "get_weather")
        XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))

        // A restart happened, and its rendered transcript pairs the tool
        // result with the assistant message that made the call (the shape
        // asserted by ToolCallIdTests).
        let passes = log.all
        XCTAssertGreaterThanOrEqual(passes.count, 2)
        let restart = try XCTUnwrap(passes.last)

        let toolIndex = try XCTUnwrap(
            restart.lastIndex { $0.role == .tool },
            "restarted generation must include the tool result")
        XCTAssertEqual(restart[toolIndex].content, #"{"forecast": "sunny"}"#)

        guard toolIndex > 0 else {
            XCTFail("tool result was rendered with no preceding assistant tool_calls message")
            return
        }
        let assistant = restart[toolIndex - 1]
        XCTAssertEqual(assistant.role, .assistant)

        let rendered = DefaultMessageGenerator().generate(message: assistant)
        let toolCalls = try XCTUnwrap(
            rendered["tool_calls"] as? [[String: any Sendable]],
            "assistant message before the tool result must carry tool_calls")
        XCTAssertEqual(toolCalls.count, 1)
        let function = try XCTUnwrap(toolCalls.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "get_weather")
    }
}
