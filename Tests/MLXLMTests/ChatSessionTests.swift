// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import XCTest
import os

@testable import MLXLLM
@testable import MLXLMCommon

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase {

    private struct RecordedMessage: Equatable, Sendable {
        var role: Chat.Message.Role
        var content: String
    }

    private struct RecordedTemplate: Equatable, Sendable {
        var messages: [RecordedMessage]
        var toolCount: Int?
    }

    private final class PreparedTokenRecorder: Sendable {
        private let values = OSAllocatedUnfairLock(initialState: [[Int]]())
        private let shapes = OSAllocatedUnfairLock(initialState: [[Int]]())

        func append(_ value: [Int], shape: [Int]) {
            values.withLock {
                $0.append(value)
            }
            shapes.withLock {
                $0.append(shape)
            }
        }

        var snapshot: [[Int]] {
            values.withLock { $0 }
        }

        var shapeSnapshot: [[Int]] {
            shapes.withLock { $0 }
        }
    }

    private final class RecordingLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
        let recorder: PreparedTokenRecorder
        let kvHeads: [Int]

        init(recorder: PreparedTokenRecorder, kvHeadCount: Int = 1) {
            self.recorder = recorder
            self.kvHeads = Array(repeating: 1, count: kvHeadCount)
            super.init()
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            recorder.append(
                input.text.tokens.asArray(Int.self),
                shape: input.text.tokens.shape)
            return .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokenCount = max(1, inputs.asArray(Int.self).count)
            return MLXArray.zeros([1, tokenCount, 8])
        }
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func generate(messages: [Chat.Message]) -> [Message] {
            continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    private struct RecordingTokenizer: Tokenizer {
        let continuation: AsyncStream<RecordedTemplate>.Continuation
        let base = TestTokenizer()

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            base.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            base.convertTokenToId(token)
        }

        func convertIdToToken(_ id: Int) -> String? {
            base.convertIdToToken(id)
        }

        var bosToken: String? { base.bosToken }
        var eosToken: String? { base.eosToken }
        var unknownToken: String? { base.unknownToken }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            let recordedMessages = messages.map { message in
                RecordedMessage(
                    role: Chat.Message.Role(rawValue: message["role"] as? String ?? "")!,
                    content: message["content"] as? String ?? "")
            }
            continuation.yield(.init(messages: recordedMessages, toolCount: tools?.count))
            return base.encode(text: "")
        }
    }

    private struct StaticPrefixTokenizer: Tokenizer {
        let unsafeDynamicTemplate: Bool
        var staticOnlyAssistantHeader = false

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            []
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            String(id)
        }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            let dynamicTokens: [Int] = messages.flatMap { message -> [Int] in
                switch message["content"] as? String {
                case "first":
                    return [101]
                case "second":
                    return [102]
                case "tool result":
                    return [103]
                default:
                    return []
                }
            }

            let hasSystem = messages.contains { $0["role"] as? String == "system" }
            if hasSystem || tools != nil {
                let systemContent =
                    messages.first { $0["role"] as? String == "system" }?[
                        "content"] as? String
                let staticTokens = systemContent == "Changed prefix." ? [21, 22] : [11, 12]
                return staticTokens + dynamicTokens
                    + (addGenerationPrompt && staticOnlyAssistantHeader ? [200] : [])
            }

            return (unsafeDynamicTemplate ? [99] : []) + dynamicTokens
        }
    }

    private struct UserHeaderPrefixTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            switch text {
            case "first": [101]
            case "second": [102]
            default: []
            }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            String(id)
        }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            let dynamicTokens = messages.flatMap { message -> [Int] in
                switch message["content"] as? String {
                case "first": [101]
                case "second": [102]
                default: []
                }
            }
            let hasStaticPrompt =
                messages.contains { $0["role"] as? String == "system" }
                || tools != nil
            let staticTokens = hasStaticPrompt ? [11, 12, 50] : []
            return staticTokens + dynamicTokens + (addGenerationPrompt ? [200] : [])
        }
    }

    private struct StaticPrefixInputProcessor: PrefixUserInputProcessor {
        let tokenizer: Tokenizer
        let configuration: ModelConfiguration
        let messageGenerator: MessageGenerator

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)

            return LMInput(tokens: MLXArray(promptTokens))
        }

        func preparePrefix(
            input: UserInput,
            fullInput: LMInput
        ) throws -> LMInput? {
            guard case .chat(let messages) = input.prompt else {
                return nil
            }
            let chat = Array(messages.prefix { $0.role == .system })
            let rawMessages = messageGenerator.generate(messages: chat)
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: rawMessages,
                tools: input.tools,
                additionalContext: input.additionalContext,
                addGenerationPrompt: false)

            return LMInput(tokens: MLXArray(promptTokens))
        }
    }

    private struct TwoDimensionalStaticPrefixProcessor: PrefixUserInputProcessor {
        let configuration = ModelConfiguration(id: "test")

        func prepare(input: UserInput) throws -> LMInput {
            let messages = DefaultMessageGenerator().generate(from: input)
            let dynamicTokens: [Int] = messages.flatMap { message -> [Int] in
                switch message["content"] as? String {
                case "first":
                    return [101]
                case "second":
                    return [102]
                default:
                    return []
                }
            }
            let hasSystem = messages.contains { $0["role"] as? String == "system" }
            let tokens = (hasSystem || input.tools != nil ? [11, 12] : []) + dynamicTokens
            return LMInput(tokens: MLXArray(tokens).reshaped(1, tokens.count))
        }

        func preparePrefix(
            input: UserInput,
            fullInput: LMInput
        ) throws -> LMInput? {
            guard case .chat(let messages) = input.prompt else {
                return nil
            }
            let chat = Array(messages.prefix { $0.role == .system })
            return try prepare(
                input: UserInput(
                    chat: chat,
                    processing: input.processing,
                    tools: input.tools,
                    additionalContext: input.additionalContext))
        }
    }

    private struct UnexpectedDraftModelLoadError: Error {}

    private actor DraftModelLoadCounter {
        private var count = 0

        func increment() {
            count += 1
        }

        var value: Int {
            count
        }
    }

    private static func makeModel(processor: TestInputProcessor = TestInputProcessor())
        -> ModelContext
    {
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

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        return .init(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
    }

    private func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        Self.makeModel(processor: processor)
    }

    private func recordingModel(
        tokenizer: Tokenizer,
        recorder: PreparedTokenRecorder,
        kvHeadCount: Int = 1
    ) -> ModelContext {
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        return recordingModel(
            processor: processor,
            tokenizer: tokenizer,
            recorder: recorder,
            kvHeadCount: kvHeadCount)
    }

    private func recordingModel(
        processor: any UserInputProcessor,
        tokenizer: Tokenizer,
        recorder: PreparedTokenRecorder,
        kvHeadCount: Int = 1
    ) -> ModelContext {
        return .init(
            configuration: ModelConfiguration(id: "test"),
            model: RecordingLanguageModel(recorder: recorder, kvHeadCount: kvHeadCount),
            processor: processor,
            tokenizer: tokenizer)
    }

    private var lookupTools: [ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
    }

    private func staticPrefixSession(
        recorder: PreparedTokenRecorder,
        unsafeDynamicTemplate: Bool = false,
        staticOnlyAssistantHeader: Bool = false,
        maxKVSize: Int? = nil,
        repetitionPenalty: Float? = nil
    ) -> ChatSession {
        let tokenizer = StaticPrefixTokenizer(
            unsafeDynamicTemplate: unsafeDynamicTemplate,
            staticOnlyAssistantHeader: staticOnlyAssistantHeader)
        let processor = StaticPrefixInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        return ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(
                maxTokens: 1,
                maxKVSize: maxKVSize,
                repetitionPenalty: repetitionPenalty),
            tools: lookupTools)
    }

    private let generationParameters = GenerateParameters(maxTokens: 50)

    private let targetLength = 1

    func testChatSessionSync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        let result1 = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result1.count, targetLength, result1)
        let result2 = try await session.respond(to: "hello again")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        var result1 = ""
        for try await part in session.streamResponse(to: "hello") {
            result1 += part
        }
        XCTAssertGreaterThan(result1.count, targetLength, result1)

        var result2 = ""
        for try await part in session.streamResponse(to: "hello again") {
            result2 += part
        }
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionRespondToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        let result = try await session.respond(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ])
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChatSessionStreamResponseToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        var result = ""
        for try await part in session.streamResponse(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ]) {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testStructuredContinuationAvoidsReplayingHistoryAcrossToolTurns() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let history: [Chat.Message] = (0 ..< 8).flatMap { index in
            [
                .user("question \(index)"),
                .assistant("answer \(index)"),
            ]
        }
        let continuations: [[Chat.Message]] = [
            [.tool("first tool result")],
            [.tool("second tool result")],
            [.user("final answer")],
        ]
        let session = ChatSession(
            model(processor: processor),
            history: history,
            generateParameters: GenerateParameters(maxTokens: 1))

        for messages in continuations {
            _ = try await session.respond(to: messages)
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.map(\.count), [history.count + 1, 1, 1])
        XCTAssertEqual(calls[0].map(\.role), history.map(\.role) + [.tool])
        XCTAssertEqual(calls[1].map(\.role), [.tool])
        XCTAssertEqual(calls[2].map(\.role), [.user])

        let actualPreparedMessageCount = calls.reduce(0) { $0 + $1.count }
        let replayedHistoryPreparedMessageCount = continuations.indices.reduce(0) {
            $0 + history.count + $1 + 1
        }
        XCTAssertLessThan(actualPreparedMessageCount, replayedHistoryPreparedMessageCount)
    }

    func testStaticInstructionsAndPromptToolsUseExactCachedPrefix() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")
        _ = try await session.respond(to: [.tool("tool result")])

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
                [103],
            ])
    }

    func testGenericProcessorDoesNotInferReusableStaticPrefix() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ChatSession(
            recordingModel(
                tokenizer: StaticPrefixTokenizer(unsafeDynamicTemplate: false),
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testLLMProcessorEnablesStaticPrefixOptimizationForUsers() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = StaticPrefixTokenizer(unsafeDynamicTemplate: false)
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
            ])
    }

    func testLLMProcessorRendersReusablePrefixWithoutGenerationPrompt() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = StaticPrefixTokenizer(
            unsafeDynamicTemplate: false,
            staticOnlyAssistantHeader: true)
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101, 200],
                [102, 200],
            ])
    }

    func testLLMProcessorRejectsPrefixEndingWithUserTurnHeader() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = UserHeaderPrefixTokenizer()
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 50, 101, 200],
                [11, 12, 50, 102, 200],
            ])
    }

    func testStaticPrefixOptimizationDisabledForBoundedKVCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder, maxKVSize: 2)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testStaticPrefixOptimizationDisabledForRepetitionPenalty() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder, repetitionPenalty: 1.1)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testStaticPrefixOptimizationPreservesTwoDimensionalTokens() async throws {
        let recorder = PreparedTokenRecorder()
        let processor = TwoDimensionalStaticPrefixProcessor()
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: TestTokenizer(),
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
        XCTAssertEqual(recorder.shapeSnapshot, [[1, 3], [1, 3]])
    }

    func testStaticPrefixOptimizationUsesFullPromptSuffixNotDynamicTemplate() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder, unsafeDynamicTemplate: true)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
            ])
    }

    func testParserOnlyToolConfigurationChangeDoesNotInvalidatePromptCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        session.toolConfiguration.parser = [
            [
                "type": "function",
                "function": [
                    "name": "new_parser_only_tool",
                    "description": "Used only for parsing",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
            ])
    }

    func testStaticPrefixMismatchThrowsInsteadOfCorruptingCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        session.instructions = "Changed prefix."

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
    }

    func testPromptConfigurationChangeThrowsAfterReusableStaticPrefix() async throws {
        let recorder = PreparedTokenRecorder()
        let session = staticPrefixSession(
            recorder: recorder,
            staticOnlyAssistantHeader: true)

        _ = try await session.respond(to: "first")
        session.instructions = "Changed prefix."

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101, 200]])
        }
    }

    func testRestoredCacheRendersConfiguredInstructionsAndPromptTools() async throws {
        let (recordedTemplates, continuation) = AsyncStream<RecordedTemplate>.makeStream()
        let tokenizer = RecordingTokenizer(continuation: continuation)
        let recorder = PreparedTokenRecorder()
        let restoredCache = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 1, 1])
        let values = MLXArray.zeros([1, 1, 1, 1])
        _ = restoredCache.update(keys: keys, values: values)
        let session = ChatSession(
            recordingModel(tokenizer: tokenizer, recorder: recorder),
            instructions: "Restored instructions.",
            cache: [restoredCache],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        continuation.finish()

        var calls: [RecordedTemplate] = []
        for await call in recordedTemplates {
            calls.append(call)
        }

        XCTAssertEqual(calls.map { $0.messages.map(\.role) }, [[.system, .user]])
        XCTAssertEqual(calls.map(\.toolCount), [1])
    }

    func testClearRebuildsStaticInstructionsAndPromptTools() async throws {
        let (recordedTemplates, continuation) = AsyncStream<RecordedTemplate>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: RecordingTokenizer(continuation: continuation),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "lookup",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model(processor: processor),
            instructions: "Static prefix.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: tools)

        _ = try await session.respond(to: "first")
        await session.clear()
        _ = try await session.respond(to: "after clear")
        continuation.finish()

        var calls: [RecordedTemplate] = []
        for await call in recordedTemplates {
            calls.append(call)
        }

        XCTAssertEqual(
            calls.map { $0.messages.map(\.role) },
            [
                [.system, .user],
                [.system, .user],
            ])
        XCTAssertEqual(calls.map(\.toolCount), [1, 1])
    }

    func testParserOnlyToolsAreNotRenderedIntoPrompt() async throws {
        let (recordedTemplates, continuation) = AsyncStream<RecordedTemplate>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: RecordingTokenizer(continuation: continuation),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "lookup",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 1))
        session.toolConfiguration = .init(prompt: nil, parser: tools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")
        continuation.finish()

        var calls: [RecordedTemplate] = []
        for await call in recordedTemplates {
            calls.append(call)
        }

        XCTAssertEqual(
            calls.map { $0.messages.map(\.role) },
            [
                [.user],
                [.user],
            ])
        XCTAssertEqual(calls.map(\.toolCount), [nil, nil])
    }

    func testChatSessionAsyncInterrupt() async throws {
        // interrupt the streamResponse and continue with another request
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        for _ in 0 ..< 10 {
            var result1 = ""
            for try await part in session.streamResponse(to: "hello") {
                result1 += part
                break
            }

            // at this point the performStreaming/generate code may still be running.
            // the next call can corrupt the state if not thread safe

            var result2 = ""
            for try await part in session.streamResponse(to: "hello again") {
                result2 += part
                if result2.count > 100 {
                    break
                }
            }
        }

        // since we are interrupting we need to wait for everything to finish
        // (avoids shutdown issues if this is the last/only test). because the
        // streaming task is not a synchronous shutdown
        await session.synchronize()
    }

    func testChatSessionWithTools() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        let result = try await session.respond(to: "What is the weather in SF?")
        XCTAssertGreaterThan(result.count, targetLength, result)

        // second turn to verify tools persist through cache
        let result2 = try await session.respond(to: "How about NYC?")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionWithToolsStreaming() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        var result = ""
        for try await part in session.streamResponse(to: "hello") {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testSpeculativeDecodingMemoryPolicyFallbackUsesDefaultGeneration() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
    }

    func testSpeculativeDecodingMemoryPolicyFailThrows() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFallbackDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFailDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch is UnexpectedDraftModelLoadError {
            XCTFail("draft model loader should not be called")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingLoadsDraftModelOnceAcrossTurns() async throws {
        let loadCounter = DraftModelLoadCounter()
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                await loadCounter.increment()
                return ModelContainer(context: Self.makeModel())
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        _ = try await session.respond(to: "hello")
        _ = try await session.respond(to: "again")

        let loadCount = await loadCounter.value
        XCTAssertEqual(loadCount, 1)
    }

    // MARK: - KV Cache

    func testCurrentCacheNilBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheAfterGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testInitWithKVCache() async throws {
        // build a cache from an initial session
        let container = ModelContainer(context: model())
        let initial = ChatSession(container, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        try await initial.withCache { [targetLength, generationParameters] cache in
            XCTAssertNotNil(cache)

            if let cache {
                // restore the cache into a new session and verify generation continues
                let restored = ChatSession(
                    container,
                    cache: cache.map { $0.copy() },
                    generateParameters: generationParameters)
                let result = try await restored.respond(to: "hello again")
                XCTAssertGreaterThan(result.count, targetLength, result)
            }
        }
    }

    func testSaveCacheThrowsBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        do {
            try await session.saveCache(to: url)
            XCTFail("expected ChatSessionError.noCacheAvailable")
        } catch ChatSessionError.noCacheAvailable {
            // expected
        }
    }

    func testSaveAndRestoreCache() async throws {
        let ctx = model()
        let initial = ChatSession(ctx, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try await initial.saveCache(to: url)

        let (loadedCache, _) = try loadPromptCache(url: url)
        let restored = ChatSession(
            ctx, cache: loadedCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testCurrentCacheNilForHistorySessionBeforeGeneration() async throws {
        // .history state should behave like .empty: no cache until first generation
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(), history: history, generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheNonNilForHistorySessionAfterGeneration() async throws {
        // after generation from .history state, cache transitions to .kvcache
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(),
            history: history,
            generateParameters: generationParameters)
        _ = try await session.respond(to: "hello again")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testCurrentCacheNilAfterClear() async throws {
        // clear() resets to .empty; currentCache() should return nil again
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
        await session.clear()
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    /// something that looks like a view model
    @MainActor class ChatModel {
        let session: ChatSession

        public var messages = [Chat.Message]()

        private var task: Task<Void, Error>?
        public var isBusy: Bool {
            task != nil
        }

        init(model: ModelContext) {
            self.session = ChatSession(model)
        }

        public func cancel() {
            task?.cancel()
        }

        public func respond(_ message: String) {
            guard task == nil else { return }

            self.messages.append(.init(role: .user, content: message))
            self.messages.append(.init(role: .assistant, content: "..."))
            let lastIndex = self.messages.count - 1

            self.task = Task {
                var first = true
                for try await item in session.streamResponse(to: message) {
                    if first {
                        self.messages[lastIndex].content = item
                        first = false
                    } else {
                        self.messages[lastIndex].content += item
                    }
                }
                self.task = nil
            }
        }
    }

    @MainActor
    func testViewModel() async throws {
        let model = ChatModel(model: model())

        // start producing a response but interrupt it
        // triggers https://github.com/ml-explore/mlx-swift/pull/323
        model.respond("message1")
        try await Task.sleep(for: .milliseconds(50))
        model.cancel()

        // wait for it to finish
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }

        // try another message, wait for full completion (but cap the length)
        model.session.generateParameters = self.generationParameters
        model.respond("message2")
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
