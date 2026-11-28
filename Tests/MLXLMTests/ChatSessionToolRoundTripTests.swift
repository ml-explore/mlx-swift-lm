// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
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
        private struct MissingUserQuery: Error {}
        private struct ConsecutiveToolAndUser: Error {}

        private let requiresUserQuery: Bool
        private let toolCallPrefix: String
        private let toolCallSuffix: String
        private let rejectUserAfterTool: Bool
        private let toolCallPasses: Int
        private let onToolCallsWritten: (@Sendable () -> Void)?
        private let plainScript = "It is sunny in Paris today."
        private var pass = 0
        private var reportedToolCallsWritten = false

        init(
            requiresUserQuery: Bool = true, toolCallPrefix: String = "",
            toolCallSuffix: String = "",
            rejectUserAfterTool: Bool = false, toolCallPasses: Int = 1,
            onToolCallsWritten: (@Sendable () -> Void)? = nil
        ) {
            self.requiresUserQuery = requiresUserQuery
            self.toolCallPrefix = toolCallPrefix
            self.toolCallSuffix = toolCallSuffix
            self.rejectUserAfterTool = rejectUserAfterTool
            self.toolCallPasses = toolCallPasses
            self.onToolCallsWritten = onToolCallsWritten
        }

        private var toolCallScript: String {
            toolCallPrefix
                + #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
                + toolCallSuffix
        }

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
            guard
                !requiresUserQuery
                    || messages.contains(where: { $0["role"] as? String == "user" })
            else {
                throw MissingUserQuery()
            }
            if rejectUserAfterTool {
                for (previous, current) in zip(messages, messages.dropFirst()) {
                    if previous["role"] as? String == "tool",
                        current["role"] as? String == "user"
                    {
                        throw ConsecutiveToolAndUser()
                    }
                }
            }
            pass += 1
            return encode(text: "")
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            (0 ..< 8).map { _ in Int.random(in: 1 ..< vocabularySize) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let script = pass <= toolCallPasses ? toolCallScript : plainScript
            let text = String(script.prefix(min(tokenIds.count * 4, script.count)))
            // Runs on the generation thread, before the parser sees the last call.
            if script == toolCallScript, text == script, !reportedToolCallsWritten {
                reportedToolCallsWritten = true
                onToolCallsWritten?()
            }
            return text
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

    private actor DispatchGate {
        private var released = false
        private var waiter: CheckedContinuation<Void, Never>?

        func wait() async {
            if !released { await withCheckedContinuation { waiter = $0 } }
        }

        func release() {
            released = true
            waiter?.resume()
            waiter = nil
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

    private final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        func set() { lock.withLock { value = true } }
        var isSet: Bool { lock.withLock { value } }
    }

    private static func makeContext(
        tokenizer: any Tokenizer, messageGenerator: MessageGenerator
    ) -> ModelContext {
        let config = Gemma3TextConfiguration.smallConfiguration(
            vocabularySize: 100)
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

    private static let weatherTool: ToolSpec = [
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

    func testRestartedGenerationSeesAssistantToolCallsBeforeResults() async throws {
        let log = MessageLog()
        let dispatched = DispatchLog()
        let tokenizer = ScriptedToolCallTokenizer()
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))

        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 24),
            tools: [Self.weatherTool],
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
        XCTAssertTrue(
            restart.contains { $0.role == .user },
            "conversation-aware templates must retain the user query on tool continuation")

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

    func testWhenWrittenConcurrentDispatchOverlapsGenerationAndKeepsCallOrder() async throws {
        let log = MessageLog()
        let romeStarted = DispatchSemaphore(value: 0)
        let parisStarted = DispatchGate()
        let overlappedGeneration = Flag()
        let romeCall =
            #"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#
        // Holds the step open once both calls are written. Only a call that
        // started while the model generated can release it.
        let tokenizer = ScriptedToolCallTokenizer(
            toolCallPrefix: romeCall,
            onToolCallsWritten: {
                if romeStarted.wait(timeout: .now() + 10) == .success {
                    overlappedGeneration.set()
                }
            })
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))

        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(
                maxTokens: 48, toolCallPolicy: .init(dispatch: [.whenWritten, .concurrent])),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                guard case .string(let city) = call.function.arguments["city"] else {
                    return "unknown"
                }
                if city == "Rome" {
                    romeStarted.signal()
                    // Rome ends only after Paris starts, so the calls must overlap.
                    await parisStarted.wait()
                } else {
                    await parisStarted.release()
                }
                return "\(city) is sunny"
            })

        _ = try await session.respond(to: "What's the weather in Rome and Paris?")

        XCTAssertTrue(overlappedGeneration.isSet, "Rome must start while the step generates")
        let restart = try XCTUnwrap(log.all.last)
        XCTAssertEqual(
            restart.filter { $0.role == .tool }.map(\.content),
            ["Rome is sunny", "Paris is sunny"])
    }

    func testRejectedCallCancelsCallsStartedWhenWritten() async throws {
        let romeStarted = DispatchSemaphore(value: 0)
        let romeCancelled = expectation(description: "Rome is cancelled")
        let parisStarted = expectation(description: "Paris never starts")
        parisStarted.isInverted = true
        // The undeclared call is parsed last, after Rome has started.
        let tokenizer = ScriptedToolCallTokenizer(
            toolCallPrefix:
                #"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#,
            toolCallSuffix: #"<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"#,
            onToolCallsWritten: { _ = romeStarted.wait(timeout: .now() + 10) })
        let context = Self.makeContext(
            tokenizer: tokenizer, messageGenerator: DefaultMessageGenerator())

        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 64),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                guard call.function.arguments["city"] == .string("Rome") else {
                    parisStarted.fulfill()
                    return "Paris is sunny"
                }
                romeStarted.signal()
                do {
                    try await Task.sleep(for: .seconds(60))
                } catch {
                    romeCancelled.fulfill()
                    throw error
                }
                return "Rome is sunny"
            })
        session.generateParameters.toolCallPolicy.dispatch = [.whenWritten]

        do {
            _ = try await session.respond(to: "What's the weather in Rome and Paris?")
            XCTFail("expected the undeclared call to end the response")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.toolName, "get_time")
        }
        await fulfillment(of: [romeCancelled, parisStarted], timeout: 1)
    }

    /// Renders each message to fixed tokens, so a longer transcript extends a
    /// shorter one exactly. Steps without a tool result write two calls.
    private final class FixedTemplateTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private let lock = NSLock()
        private var answersTools = false
        private var rendered: [[Int]] = []
        private let calls =
            #"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#
            + #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
        private let answer = "Both are sunny."

        var renders: [[Int]] { lock.withLock { rendered } }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            let roles = ["system": 1, "user": 2, "assistant": 3, "tool": 4]
            var tokens: [Int] = []
            for message in messages {
                let text =
                    (message["content"] as? String ?? "")
                    + String(repeating: "#", count: (message["tool_calls"] as? [Any])?.count ?? 0)
                tokens.append(roles[message["role"] as? String ?? ""] ?? 5)
                tokens += text.unicodeScalars.map { Int($0.value % 80) + 10 }
                tokens.append(95)
            }
            tokens.append(3)
            lock.withLock {
                answersTools = messages.contains { $0["role"] as? String == "tool" }
                rendered.append(tokens)
            }
            return tokens
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let script = lock.withLock { answersTools } ? answer : calls
            return String(script.prefix(min(tokenIds.count * 4, script.count)))
        }

        func convertTokenToId(_ token: String) -> Int? { 1 }
        func convertIdToToken(_ id: Int) -> String? { "x" }
    }

    /// Runs one response with two calls and returns the last step's report.
    private func respondWithTwoCalls(
        _ options: ToolDispatchOptions, tokenizer: FixedTemplateTokenizer
    ) async throws -> (info: GenerateCompletionInfo, results: [String]) {
        let log = MessageLog()
        let romeReturned = DispatchGate()
        let session = ChatSession(
            Self.makeContext(
                tokenizer: tokenizer, messageGenerator: RecordingMessageGenerator(log: log)),
            generateParameters: GenerateParameters(
                maxTokens: 48, toolCallPolicy: .init(dispatch: options)),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                guard call.function.arguments["city"] == .string("Paris") else {
                    await romeReturned.release()
                    return "Rome is sunny"
                }
                // Paris still runs while the session handles Rome's result.
                await romeReturned.wait()
                try await Task.sleep(for: .milliseconds(50))
                return "Paris is sunny"
            })

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "Weather in Rome and Paris?", images: [], videos: [])
        {
            if case .info(let completion) = generation { info = completion }
        }
        let results = try XCTUnwrap(log.all.last).filter { $0.role == .tool }.map(\.content)
        return (try XCTUnwrap(info), results)
    }

    func testPrefillResultLeavesOnlyTheLastResultToTheNextStep() async throws {
        let tokenizer = FixedTemplateTokenizer()
        let (info, results) = try await respondWithTwoCalls(
            [.concurrent, .prefillResult], tokenizer: tokenizer)
        XCTAssertEqual(results, ["Rome is sunny", "Paris is sunny"])

        // Renders: the first step, Rome's result with and without the next call,
        // then the step that answers both results.
        let renders = tokenizer.renders
        guard renders.count == 4 else {
            return XCTFail("expected 4 renders, got \(renders.count)")
        }
        let throughRome = renders[1].dropLast()
        XCTAssertEqual(Array(renders[2].prefix(throughRome.count)), Array(throughRome))
        // The step that answers prefills only Paris's result.
        XCTAssertEqual(info.cachedPromptTokenCount, throughRome.count)
        XCTAssertEqual(info.promptTokenCount, renders[3].count - throughRome.count)

        // Without the option the next step prefills both results.
        let baseline = try await respondWithTwoCalls(
            [.concurrent], tokenizer: FixedTemplateTokenizer())
        XCTAssertEqual(baseline.results, results)
        XCTAssertLessThan(baseline.info.cachedPromptTokenCount, throughRome.count)
    }

    func testApplicationPrefillsResultsWhileItsLaterToolsRun() async throws {
        let tokenizer = FixedTemplateTokenizer()
        let session = ChatSession(
            Self.makeContext(tokenizer: tokenizer, messageGenerator: DefaultMessageGenerator()),
            generateParameters: GenerateParameters(maxTokens: 48),
            tools: [Self.weatherTool])

        var calls: [ToolCall] = []
        for try await generation in session.streamDetails(
            to: "Weather in Rome and Paris?", images: [], videos: [])
        {
            if case .toolCall(let call) = generation { calls.append(call) }
        }
        guard calls.count == 2 else { return XCTFail("expected 2 calls, got \(calls.count)") }
        let rome = Chat.Message.tool("Rome is sunny", id: calls[0].id, name: "get_weather")
        let paris = Chat.Message.tool("Paris is sunny", id: calls[1].id, name: "get_weather")

        // The application runs Paris itself; Rome has returned.
        let prefilled = try await session.prefill(toolResults: [rome], before: calls[1])

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(to: [rome, paris]) {
            if case .info(let completion) = generation { info = completion }
        }
        let renders = tokenizer.renders
        guard renders.count == 4 else {
            return XCTFail("expected 4 renders, got \(renders.count)")
        }
        let throughRome = renders[1].count - 1
        XCTAssertGreaterThan(prefilled, 0)
        XCTAssertEqual(info?.cachedPromptTokenCount, throughRome)
        XCTAssertEqual(info?.promptTokenCount, renders[3].count - throughRome)
    }

    func testImmediateSteeringAfterToolWaitsForAssistant() async throws {
        try await checkSteeringAfterTool(policy: .nextSafeBoundary)
    }

    func testQueuedSteeringAfterToolWaitsForAssistant() async throws {
        try await checkSteeringAfterTool(policy: .nextStepBoundary)
    }

    func testSteeringWaitsThroughConsecutiveToolCalls() async throws {
        try await checkSteeringAfterTool(policy: .nextSafeBoundary, toolCallPasses: 2)
    }

    private func checkSteeringAfterTool(
        policy: SteeringPolicy, toolCallPasses: Int = 1
    ) async throws {
        let log = MessageLog()
        let gate = DispatchGate()
        let (dispatchStarted, started) = AsyncStream<Void>.makeStream()
        let context = Self.makeContext(
            tokenizer: ScriptedToolCallTokenizer(
                rejectUserAfterTool: true, toolCallPasses: toolCallPasses),
            messageGenerator: RecordingMessageGenerator(log: log))
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 24),
            tools: [Self.weatherTool],
            toolDispatch: { _ in
                started.yield(())
                await gate.wait()
                return "sunny"
            })
        let stream = session.streamDetails(to: "Weather?")
        var iterator = dispatchStarted.makeAsyncIterator()
        _ = await iterator.next()
        let id = try session.steer("Be concise", policy: policy)
        await gate.release()

        var applied: [UUID] = []
        for try await event in stream {
            if case .steering(.applied(let ids)) = event { applied += ids }
        }
        XCTAssertEqual(applied, [id])
        let passes = log.all
        XCTAssertEqual(passes.count, toolCallPasses + 2)
        guard passes.count == toolCallPasses + 2 else { return }
        for index in 1 ... toolCallPasses {
            XCTAssertEqual(passes[index].last?.role, .tool)
            XCTAssertEqual(passes[index].last?.content, "sunny")
        }
        let expectedRoles: [Chat.Message.Role] =
            [.user] + Array(repeating: [.assistant, .tool], count: toolCallPasses).flatMap { $0 }
            + [.assistant, .user]
        XCTAssertEqual(passes.last?.map(\.role), expectedRoles)
        XCTAssertEqual(passes.last?.last?.content, "Be concise")
    }

    func testTokenObservationOrdersEachAutomaticToolPassBeforeItsCompletion() async throws {
        let log = MessageLog()
        let dispatched = DispatchLog()
        let context = Self.makeContext(
            tokenizer: ScriptedToolCallTokenizer(),
            messageGenerator: RecordingMessageGenerator(log: log))
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 24, temperature: 0),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast": "sunny"}"#
            })
        var passTokens = 0
        var completions = 0
        var reply = ""
        for try await event in session.observe(
            to: "What's the weather in Paris?",
            using: .logProbabilities(topCount: 2))
        {
            if event.logProbabilities != nil { passTokens += 1 }
            if let generation = event.generation {
                if let chunk = generation.chunk { reply += chunk }
                if let info = generation.info {
                    XCTAssertEqual(passTokens, info.generationTokenCount)
                    XCTAssertGreaterThan(passTokens, 0)
                    passTokens = 0
                    completions += 1
                }
            }
        }
        XCTAssertFalse(reply.isEmpty)
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(completions, 2)
        XCTAssertEqual(passTokens, 0)
    }

    func testCustomTransformerGetsFreshHandlerForEachAutomaticToolPass() async throws {
        let dispatched = DispatchLog()
        let context = Self.makeContext(
            tokenizer: ScriptedToolCallTokenizer(), messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 24, temperature: 0),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast": "sunny"}"#
            })
        let factories = TransformerFactoryCount()
        var tokens = 0
        var completions = 0
        var reply = ""
        for try await event in session.observe(
            to: "What's the weather in Paris?",
            using: TokenCountingTransformer(factories: factories))
        {
            switch event {
            case .token(let index):
                tokens += 1
                XCTAssertEqual(index, tokens)
            case .output(let generation, let count):
                XCTAssertNil(generation.toolCall)
                if let chunk = generation.chunk { reply += chunk }
                if let info = generation.info {
                    XCTAssertEqual(tokens, count)
                    XCTAssertEqual(tokens, info.generationTokenCount)
                    XCTAssertGreaterThan(tokens, 0)
                    tokens = 0
                    completions += 1
                }
            }
        }
        XCTAssertFalse(reply.isEmpty)
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(factories.value, 2)
        XCTAssertEqual(completions, 2)
        XCTAssertEqual(tokens, 0)
    }

    func testGenerationPolicyControlsValidationBeforeAutomaticDispatch() async throws {
        let restrictedTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string", "enum": ["Rome"]] as [String: any Sendable]
                    ],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        for validation in ToolCallValidationPolicy.allCases {
            let dispatched = DispatchLog()
            let context = Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(),
                messageGenerator: DefaultMessageGenerator())
            var policy = ToolCallPolicy(recovery: .disabled)
            if validation == .strict { policy.validation = .strict }
            let session = ChatSession(
                context,
                generateParameters: .init(maxTokens: 24, toolCallPolicy: policy),
                tools: [restrictedTool],
                toolDispatch: { call in
                    dispatched.record(call)
                    return "sunny"
                })
            do {
                _ = try await session.respond(to: "Weather in Paris?")
                XCTAssertEqual(validation, .permissive)
                XCTAssertEqual(dispatched.all.count, 1)
                XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))
            } catch let error as RejectedToolCallError {
                XCTAssertEqual(validation, .strict)
                XCTAssertEqual(error.rejection.reason, .invalidArguments)
                XCTAssertTrue(dispatched.all.isEmpty)
            }
        }
    }

    func testRawCacheToolRestartPreservesLegacyFragmentShape() async throws {
        let log = MessageLog()
        let tokenizer = ScriptedToolCallTokenizer(
            requiresUserQuery: false,
            toolCallPrefix: "Checking the tool. ")
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))
        let parameters = GenerateParameters(maxTokens: 40)
        let rawCache = try context.model.newCache(parameters: parameters)
        let session = ChatSession(
            context,
            cache: rawCache,
            generateParameters: parameters,
            tools: [Self.weatherTool],
            toolDispatch: { _ in #"{"forecast": "sunny"}"# })

        _ = try await session.respond(to: "What's the weather in Paris?")

        let passes = log.all
        XCTAssertGreaterThanOrEqual(passes.count, 2)
        let restart = try XCTUnwrap(passes.last)
        XCTAssertFalse(
            restart.contains { $0.role == .user },
            "raw KV caches cannot reconstruct and replay their original transcript")

        let toolIndex = try XCTUnwrap(restart.lastIndex { $0.role == .tool })
        guard toolIndex > 0 else {
            XCTFail("tool result was rendered with no preceding assistant message")
            return
        }
        XCTAssertEqual(restart[toolIndex - 1].role, .assistant)
        XCTAssertEqual(
            restart[toolIndex - 1].content,
            "",
            "generated text is already represented by the raw cache and must not be replayed")
    }
}
