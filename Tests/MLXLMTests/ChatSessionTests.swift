// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXOptimizers
import XCTest

@testable import MLXLMCommon

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase {

    private struct RecordedMessage: Equatable, Sendable {
        var role: Chat.Message.Role
        var content: String
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func generate(messages: [Chat.Message]) -> [Message] {
            continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    /// Produces a transcript token stream where a rendered follow-up is an
    /// exact extension of the prompt and generated tokens from the prior turn.
    private struct PrefixPreservingTokenizer: Tokenizer {
        let renderedLengthContinuation: AsyncStream<Int>.Continuation
        var rewritesPrefixOnContinuation = false
        var rewritesCachedTailOnContinuation = false

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil
        var eosTokenId: Int? { 101 }
        var unknownTokenId: Int? { 102 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.unicodeScalars.map { scalar in
                if (0xE000 ..< 0xE064).contains(Int(scalar.value)) {
                    return Int(scalar.value) - 0xE000
                }
                return Int(scalar.value) % 80 + 1
            }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(
                String.UnicodeScalarView(
                    tokenIds.compactMap { UnicodeScalar(0xE000 + $0) }))
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? {
            UnicodeScalar(0xE000 + id).map(String.init)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens: [Int] = []
            for message in messages {
                let role = message["role"] as? String
                let content = message["content"] as? String ?? ""
                switch role {
                case "system":
                    tokens.append(96)
                case "user":
                    tokens.append(97)
                case "tool":
                    tokens.append(98)
                case "assistant":
                    tokens.append(94)
                default:
                    tokens.append(99)
                }
                tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
                if role != "assistant" {
                    tokens.append(95)
                }
            }
            tokens.append(94)  // assistant generation marker
            if rewritesPrefixOnContinuation {
                tokens.insert(messages.count > 1 ? 93 : 92, at: 0)
            }
            if rewritesCachedTailOnContinuation, let marker = tokens.firstIndex(of: 94) {
                tokens[marker] = messages.count > 1 ? 93 : 92
            }

            renderedLengthContinuation.yield(tokens.count)
            return tokens
        }
    }

    private struct MediaAwareInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration = ModelConfiguration(id: "test")
        let messageGenerator = DefaultMessageGenerator()

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext)
            let image =
                input.images.isEmpty
                ? nil : LMInput.ProcessedImage(pixels: MLXArray([Float(0)]))
            return LMInput(text: .init(tokens: MLXArray(tokens)), image: image)
        }
    }

    private struct MaskedInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration = ModelConfiguration(id: "test")
        let messageGenerator = DefaultMessageGenerator()

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext)
            let mask = MLXArray(Array(repeating: 1, count: tokens.count))
                .reshaped(1, tokens.count)
            return LMInput(text: .init(tokens: MLXArray(tokens), mask: mask))
        }
    }

    /// The synthetic text model does not consume attention masks itself.
    /// Strip the mask only after `ChatSession` has made its cache-reuse
    /// decision so the regression test can exercise that decision with a
    /// valid masked processor input.
    private final class MaskTolerantLanguageModel: Module, LanguageModel {
        let base: any LanguageModel

        init(_ base: any LanguageModel) {
            self.base = base
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(
                LMInput(tokens: input.text.tokens),
                cache: cache,
                state: state,
                prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters)
        }
    }

    /// A model that opts into append-only media cache reuse.
    ///
    /// The synthetic text model underneath cannot consume a media payload, so the
    /// split keeps the token suffix and the payload is stripped in `prepare` — the
    /// same shape as `MaskTolerantLanguageModel`. What these tests measure is
    /// `ChatSession`'s decision, which is made before the model is called.
    private final class SplittingLanguageModel: Module, LanguageModel, PreparedInputSplitting {
        let base: any LanguageModel
        let declinesToSplit: Bool

        init(_ base: any LanguageModel, declinesToSplit: Bool = false) {
            self.base = base
            self.declinesToSplit = declinesToSplit
            super.init()
        }

        func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
            -> LMInput?
        {
            guard !declinesToSplit else { return nil }
            let ids = input.text.tokens.asArray(Int.self)
            guard prefixTokenCount > 0, prefixTokenCount < ids.count else { return nil }
            return LMInput(
                text: .init(tokens: MLXArray(Array(ids[prefixTokenCount...]))),
                image: input.image)
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(
                LMInput(tokens: input.text.tokens),
                cache: cache,
                state: state,
                prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            base.newCache(parameters: parameters)
        }
    }

    private struct EmptyChatTemplateTokenizer: Tokenizer {
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil
        var eosTokenId: Int? { 101 }
        var unknownTokenId: Int? { 102 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            []
        }
    }

    private struct UnexpectedDraftModelLoadError: Error {}

    private struct PreparedImageInputProcessor: UserInputProcessor {
        private let base = TestInputProcessor()

        func prepare(input: UserInput) throws -> LMInput {
            let text = try base.prepare(input: input).text
            return LMInput(
                text: text,
                image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
        }
    }

    /// Attaches media to the first prepared turn only. That turn refuses speculation and drops
    /// the draft cache; later text-only turns are what should be able to pick it back up.
    ///
    /// `@unchecked Sendable` because the counter is serialised by the lock.
    private final class FirstTurnImageInputProcessor: UserInputProcessor, @unchecked Sendable {
        private let base: TestInputProcessor
        private let lock = NSLock()
        private var prepared = 0

        init(base: TestInputProcessor) {
            self.base = base
        }

        private func consumeFirst() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            prepared += 1
            return prepared == 1
        }

        func prepare(input: UserInput) async throws -> LMInput {
            let text = try await base.prepare(input: input).text
            guard consumeFirst() else { return LMInput(text: text) }
            return LMInput(
                text: text,
                image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
        }
    }

    /// A model that hands back model state on every prefill, the way the Qwen
    /// vision families and GLM-OCR hand back an M-RoPE anchor. The state is
    /// what a session has to carry across turns and persist with its cache;
    /// this stands in for a VLM without needing image inputs.
    private final class StateProducingModel: Module, LanguageModel, KVCacheDimensionProvider {
        static let anchorKey = LMOutput.Key<MLXArray>("test.ropeDeltas")

        @ModuleInfo var inner: Gemma3TextModel

        var kvHeads: [Int] { (inner as? KVCacheDimensionProvider)?.kvHeads ?? [] }

        init(_ inner: Gemma3TextModel) {
            self.inner = inner
        }

        private func batched(_ tokens: MLXArray) -> MLXArray {
            tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            let logits = inner(batched(input.text.tokens), cache: cache.isEmpty ? nil : cache)
            // Carry a seeded anchor forward, or start one at zero — the shape
            // of every wired model's cold prefill.
            var produced = LMOutput.State()
            produced[Self.anchorKey] = state?[Self.anchorKey] ?? MLXArray([Int32(0)])
            let total = input.text.tokens.dim(-1)
            prefill.progress?(total, total)
            return .logits(LMOutput(logits: logits, state: produced))
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            LMOutput(logits: inner(batched(input.tokens), cache: cache), state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try inner.newCache(parameters: parameters)
        }
    }

    private static func makeStateProducingModel() -> ModelContext {
        let base = makeModel()
        guard let inner = base.model as? Gemma3TextModel else {
            fatalError("expected the test model to be a Gemma3TextModel")
        }
        return .init(
            configuration: base.configuration,
            model: StateProducingModel(inner),
            processor: base.processor,
            tokenizer: base.tokenizer)
    }

    private actor DraftModelLoadCounter {
        private var count = 0

        func increment() {
            count += 1
        }

        var value: Int {
            count
        }
    }

    private static func makeModel(
        processor: any UserInputProcessor,
        configuration: ModelConfiguration,
        tokenizer: any Tokenizer
    )
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
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)
    }

    private static func makeModel(processor: TestInputProcessor = TestInputProcessor())
        -> ModelContext
    {
        makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    private func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        Self.makeModel(processor: processor)
    }

    private func model(processor: MediaAwareInputProcessor) -> ModelContext {
        Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    private func model(
        processor: MediaAwareInputProcessor, splitting: Bool, declinesToSplit: Bool = false
    ) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        if splitting {
            context.model = SplittingLanguageModel(
                context.model, declinesToSplit: declinesToSplit)
        }
        return context
    }

    private func model(processor: MaskedInputProcessor) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        context.model = MaskTolerantLanguageModel(context.model)
        return context
    }

    private func collectGeneration(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var completionInfo: GenerateCompletionInfo?
        for try await item in stream {
            if let chunk = item.chunk {
                text += chunk
            }
            if let info = item.info {
                completionInfo = info
            }
        }
        return (text, try XCTUnwrap(completionInfo))
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

    func testChangingInstructionsUpdatesRetainedConversation() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            instructions: "first instruction",
            generateParameters: GenerateParameters(maxTokens: 1))

        _ = try await session.respond(to: "first question")
        session.instructions = "replacement instruction"
        _ = try await session.respond(to: "second question")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.system, .user])
        XCTAssertEqual(calls[0].first?.content, "first instruction")
        XCTAssertEqual(calls[1].map(\.role), [.system, .user, .assistant, .user])
        XCTAssertEqual(calls[1].first?.content, "replacement instruction")
        XCTAssertFalse(calls[1].contains { $0.content == "first instruction" })
    }

    func testEmptyGenerationRollsBackIncompleteTurn() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 0))

        let firstResponse = try await session.respond(to: "first question")
        let secondResponse = try await session.respond(to: "second question")
        XCTAssertEqual(firstResponse, "")
        XCTAssertEqual(secondResponse, "")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.user])
        XCTAssertEqual(calls[1].map(\.role), [.user])
        XCTAssertEqual(calls[1].first?.content, "second question")
    }

    func testInterruptedGenerationRollsBackIncompleteTurn() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 50))

        for try await _ in session.streamResponse(to: "first question") {
            break
        }
        await session.synchronize()

        session.generateParameters = GenerateParameters(maxTokens: 1)
        _ = try await session.respond(to: "second question")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.user])
        XCTAssertEqual(calls[1].map(\.role), [.user])
        XCTAssertEqual(calls[1].first?.content, "second question")
    }

    func testEmptyPreparedInputThrowsClearError() async throws {
        let tokenizer = EmptyChatTemplateTokenizer()
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 1))

        do {
            _ = try await session.respond(to: "ignored")
            XCTFail("expected ChatSessionError.emptyPreparedInput")
        } catch ChatSessionError.emptyPreparedInput {
            // expected
        }
    }

    func testStructuredContinuationRendersCompleteTranscriptAcrossToolTurns() async throws {
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

        XCTAssertEqual(
            calls.map(\.count), [history.count + 1, history.count + 3, history.count + 5])
        XCTAssertEqual(calls[0].map(\.role), history.map(\.role) + [.tool])
        XCTAssertEqual(
            calls[1].map(\.role),
            history.map(\.role) + [.tool, .assistant, .tool])
        XCTAssertEqual(
            calls[2].map(\.role),
            history.map(\.role) + [.tool, .assistant, .tool, .assistant, .user])
    }

    func testLegacyStringContinuationPrefillsOnlyUncachedSuffix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let info = try XCTUnwrap(completionInfo)
        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertLessThan(info.promptTokenCount, fullSecondPromptLength)
        XCTAssertEqual(
            info.promptTokenCount,
            fullSecondPromptLength - firstPromptLength - 3)
    }

    func testStructuredContinuationPrefillsOnlyUncachedSuffix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: [.user("first")])
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: [.user("second")]) {
            if let info = item.info {
                completionInfo = info
            }
        }

        let info = try XCTUnwrap(completionInfo)
        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertLessThan(info.promptTokenCount, fullSecondPromptLength)
        XCTAssertEqual(
            info.promptTokenCount,
            fullSecondPromptLength - firstPromptLength - 3)
    }

    func testExactPrefixReuseRebuildsWhenPreparedInputHasMask() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MaskedInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        _ = await lengthIterator.next()

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        XCTAssertEqual(second.info.promptTokenCount, secondRenderedLength)
    }

    func testContinuationRebuildsCacheWhenTemplateRewritesPrefix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesPrefixOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
    }

    func testContinuationTrimsCacheToLongestCommonPrefix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        let expectedCommonPrefixLength = firstRenderedLength - 1
        XCTAssertEqual(
            completionInfo?.promptTokenCount,
            fullSecondPromptLength - expectedCommonPrefixLength)
        XCTAssertLessThan(completionInfo?.promptTokenCount ?? .max, fullSecondPromptLength)
    }

    func testHistoricalMediaReusesSuffixButNewMediaRebuildsCache() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var textFollowUpInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                textFollowUpInfo = info
            }
        }
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        XCTAssertEqual(
            textFollowUpInfo?.promptTokenCount,
            secondRenderedLength - firstRenderedLength - 3)

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)
        XCTAssertEqual(newMediaInfo?.promptTokenCount, thirdRenderedLength)
    }

    func testLongestCommonPrefixTrimmingFallsBackForHistoricalMedia() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
    }

    /// Processor that masks every logit except one, forcing that token to be sampled.
    private struct ForceTokenProcessor: LogitProcessor {
        let token: Int32

        func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            let indices = MLXArray(0 ..< Int32(logits.dim(-1)))
            return MLX.where(indices .== MLXArray(token), logits, MLXArray(-Float.infinity))
        }

        func didSample(token: MLXArray) {}
    }

    private struct LiteralTokenizer: Tokenizer {
        let tokenByCharacter: [Character: Int]
        let characterByToken: [Int: Character]

        init(output: String) {
            let characters = Array(Set(output)).sorted { String($0) < String($1) }
            tokenByCharacter = Dictionary(
                uniqueKeysWithValues: characters.enumerated().map { ($0.element, $0.offset + 1) })
            characterByToken = Dictionary(
                uniqueKeysWithValues: tokenByCharacter.map { ($0.value, $0.key) })
        }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        var eosTokenId: Int? { 99 }
        var unknownTokenId: Int? { 98 }

        func tokens(for output: String) -> [Int] {
            output.compactMap { tokenByCharacter[$0] }
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [97] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(tokenIds.compactMap { characterByToken[$0] })
        }

        func convertTokenToId(_ token: String) -> Int? {
            token.count == 1 ? token.first.flatMap { tokenByCharacter[$0] } : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            characterByToken[id].map(String.init)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            [97]
        }
    }

    private final class TokenSequenceState: @unchecked Sendable {
        private let lock = NSLock()
        private let tokens: [Int32]
        private var index = 0

        init(tokens: [Int]) {
            self.tokens = tokens.map(Int32.init)
        }

        var current: Int32 {
            lock.withLock { tokens[min(index, tokens.count - 1)] }
        }

        func advance() {
            lock.withLock { index = min(index + 1, tokens.count - 1) }
        }
    }

    private struct ForceTokenSequenceProcessor: LogitProcessor {
        let state: TokenSequenceState

        func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            let indices = MLXArray(0 ..< Int32(logits.dim(-1)))
            return MLX.where(
                indices .== MLXArray(state.current), logits, MLXArray(-Float.infinity))
        }

        func didSample(token: MLXArray) {
            state.advance()
        }
    }

    private func rejectionSession(
        output: String,
        messageGenerator: any MessageGenerator = DefaultMessageGenerator(),
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) -> ChatSession {
        let tokenizer = LiteralTokenizer(output: output)
        let configuration = ModelConfiguration(
            id: "rejected-tool-call-test",
            eosTokenIds: [99],
            toolCallFormat: .json)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: messageGenerator)
        let outputTokens = tokenizer.tokens(for: output) + [99]
        let components = GenerationComponents {
            ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: outputTokens))
        }
        return ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(
                maxTokens: outputTokens.count + 1, temperature: 0),
            components: components,
            tools: tools,
            toolDispatch: toolDispatch)
    }

    func testStreamDetailsEmitsRejectedToolCallAndCompletionCount() async throws {
        let session = rejectionSession(output: #"<tool_call>{"#)
        var rejection: RejectedToolCall?
        var info: GenerateCompletionInfo?
        var chunks: [String] = []

        for try await event in session.streamDetails(to: "trigger") {
            if let value = event.rejectedToolCall { rejection = value }
            if let value = event.info { info = value }
            if let value = event.chunk { chunks.append(value) }
        }

        XCTAssertEqual(rejection?.reason, .incompleteOutput)
        XCTAssertEqual(info?.rejectedToolCallCount, 1)
        XCTAssertTrue(chunks.isEmpty)
    }

    func testTextStreamThrowsRejectedToolCallError() async throws {
        let session = rejectionSession(output: #"<tool_call>{"#)

        do {
            for try await _ in session.streamResponse(to: "trigger") {}
            XCTFail("Expected RejectedToolCallError")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.reason, .incompleteOutput)
        }
    }

    func testRejectedGenerationIsNotCommittedToConversation() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let session = rejectionSession(
            output: #"<tool_call>{"#,
            messageGenerator: RecordingMessageGenerator(continuation: continuation))

        for prompt in ["first", "second"] {
            do {
                for try await _ in session.streamResponse(to: prompt) {}
                XCTFail("Expected RejectedToolCallError")
            } catch is RejectedToolCallError {
                // Expected. The next request must start from a clean transcript.
            }
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages { calls.append(call) }
        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0], [.init(role: .user, content: "first")])
        XCTAssertEqual(calls[1], [.init(role: .user, content: "second")])
    }

    func testRejectedGenerationPreventsPartialToolDispatch() async throws {
        let dispatchCount = CallCounter()
        let output =
            #"<tool_call>{"name":"allowed","arguments":{}}</tool_call><tool_call>{"#
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": ["name": "allowed"] as [String: any Sendable],
            ]
        ]
        let session = rejectionSession(
            output: output,
            tools: tools,
            toolDispatch: { _ in
                dispatchCount.increment()
                return "should not execute"
            })

        do {
            for try await _ in session.streamDetails(to: "trigger") {}
            XCTFail("Expected RejectedToolCallError")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.reason, .incompleteOutput)
        }
        XCTAssertEqual(dispatchCount.value, 0)
    }

    /// Thread-safe counter for asserting how many times a `@Sendable` factory runs.
    private final class CallCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func increment() {
            lock.withLock { count += 1 }
        }

        var value: Int {
            lock.withLock { count }
        }
    }

    /// A custom ``LogitProcessor`` injected via ``GenerationComponents`` must ride
    /// the real ``ChatSession`` generation path -- the reason this API exists.
    func testChatSessionUsesCustomLogitProcessor() async throws {
        let forcedToken = 7
        let inputProcessor = TestInputProcessor()
        let model = model(processor: inputProcessor)

        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var components = GenerationComponents()
        components.logitProcessorFactory = {
            ForceTokenProcessor(token: Int32(forcedToken))
        }

        let session = ChatSession(model, generateParameters: parameters, components: components)
        let result = try await session.respond(to: "hello")

        // with all other logits masked, every generated token must be the forced token
        let expectedWord = try XCTUnwrap(inputProcessor.tokenizer.convertIdToToken(forcedToken))
        let words = result.split(separator: " ").map(String.init)
        XCTAssertFalse(words.isEmpty, result)
        XCTAssertTrue(words.allSatisfy { $0 == expectedWord }, result)
    }

    /// A single ``ChatSession`` reuses one ``GenerationComponents`` across turns.
    /// The factory MUST be invoked fresh for each generation so a stateful
    /// processor never leaks state between turns.
    func testChatSessionInvokesLogitProcessorFactoryPerGeneration() async throws {
        let counter = CallCounter()
        let model = model()

        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var components = GenerationComponents()
        components.logitProcessorFactory = {
            counter.increment()
            return ForceTokenProcessor(token: 7)
        }

        let session = ChatSession(model, generateParameters: parameters, components: components)
        _ = try await session.respond(to: "hello")
        _ = try await session.respond(to: "hello again")

        // two turns -> two fresh processor instances
        XCTAssertEqual(counter.value, 2)
    }

    /// Parameter-dependent components must fail through the real session API
    /// before prompt prefill starts.
    func testChatSessionRunsGenerationComponentValidation() async throws {
        let inputProcessor = TestInputProcessor()
        let components = try GenerationComponents().applyingThinkingBudget(
            ThinkingBudgetConfiguration(
                maximumTokenCount: 100,
                transitionOverride: .immediate),
            reasoning: .alwaysOnThinking,
            tokenizer: inputProcessor.tokenizer)
        let session = ChatSession(
            model(processor: inputProcessor),
            generateParameters: GenerateParameters(maxTokens: 1),
            components: components)

        do {
            _ = try await session.respond(to: "hello")
            XCTFail("Expected generation-component validation to reject maxTokens")
        } catch let error as ThinkingBudgetError {
            guard case .insufficientGenerationTokenLimit = error else {
                XCTFail("Unexpected thinking-budget error: \(error)")
                return
            }
        }
    }

    /// Passing an empty ``GenerationComponents()`` must be non-breaking: the
    /// session still generates normally, exactly as when no components are
    /// supplied. (The exact processor-equivalence guarantee is proven
    /// deterministically by `testEmptyGenerationComponentsMatchesParametersProcessor`;
    /// full token-sequence equality is not asserted here because argmax over
    /// randomly-initialized weights is not bitwise-reproducible on GPU.)
    func testChatSessionEmptyComponentsMatchesDefault() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 50),
            components: GenerationComponents())
        let result = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    /// The append-only counterpart of
    /// `testHistoricalMediaReusesSuffixButNewMediaRebuildsCache`: same transcript,
    /// same turns, but a model that can split its own prepared input. The new-image
    /// turn now prefills only the uncached suffix instead of the whole prompt.
    func testAppendOnlyMediaReusesCacheWhenModelSplitsPreparedInput() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        for try await _ in session.streamDetails(to: "describe it") {}
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)

        XCTAssertEqual(
            newMediaInfo?.promptTokenCount,
            thirdRenderedLength - secondRenderedLength - 3,
            "an append-only media turn should prefill only the uncached suffix")
    }

    /// A model that declines the split keeps the pre-existing behavior, which is
    /// also what every non-conforming model gets.
    func testAppendOnlyMediaRebuildsWhenModelDeclinesToSplit() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true, declinesToSplit: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        for try await _ in session.streamDetails(to: "describe it") {}
        _ = await lengthIterator.next()

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)

        XCTAssertEqual(newMediaInfo?.promptTokenCount, thirdRenderedLength)
    }

    /// Splitting is offered only where the transcript is *extended*. When the
    /// template rewrites an already-cached tail the turn needs a rewind, and media
    /// still forces a rebuild there — the guard PR #472 added is untouched.
    func testLongestCommonPrefixTrimmingStillFallsBackForMediaWhenModelCanSplit()
        async throws
    {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                completionInfo = info
            }
        }
        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
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

    func testSpeculativeDecodingMemoryPolicyFallbackReusesMainCacheAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault)),
            generateParameters: GenerateParameters(maxTokens: 3, temperature: 0))

        _ = try await session.respond(to: "first")
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        XCTAssertEqual(
            completionInfo?.promptTokenCount,
            secondRenderedLength - firstRenderedLength - 3)
        XCTAssertNil(completionInfo?.speculativeDecodingTelemetry)
    }

    func testActiveSpeculativeDecodingSafelyRebuildsAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesPrefixOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let container = ModelContainer(context: model(processor: processor))
        let speculativeDecoding = SpeculativeDecodingConfig(
            draftModel: container,
            numDraftTokens: 2)
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let session = ChatSession(
            container,
            speculativeDecoding: speculativeDecoding,
            generateParameters: parameters)

        let first = try await collectGeneration(session.streamDetails(to: "first"))
        _ = await lengthIterator.next()
        let firstTelemetry = try XCTUnwrap(first.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(firstTelemetry.roundCount, 0)
        XCTAssertGreaterThan(firstTelemetry.draftTokenCount, 0)
        XCTAssertEqual(firstTelemetry.emittedTokenCount, first.info.generationTokenCount)

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        let secondTelemetry = try XCTUnwrap(second.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(secondTelemetry.roundCount, 0)
        XCTAssertGreaterThan(secondTelemetry.draftTokenCount, 0)
        XCTAssertEqual(secondTelemetry.emittedTokenCount, second.info.generationTokenCount)
        XCTAssertEqual(second.info.promptTokenCount, secondRenderedLength)

        let cleanSession = ChatSession(
            container,
            history: [.user("first"), .assistant(first.text)],
            speculativeDecoding: speculativeDecoding,
            generateParameters: parameters)
        let cleanSecond = try await collectGeneration(
            cleanSession.streamDetails(to: "second"))
        let cleanRenderedLengthValue = await lengthIterator.next()
        let cleanRenderedLength = try XCTUnwrap(cleanRenderedLengthValue)

        let cleanTelemetry = try XCTUnwrap(cleanSecond.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(cleanTelemetry.draftTokenCount, 0)
        XCTAssertEqual(cleanTelemetry.emittedTokenCount, cleanSecond.info.generationTokenCount)
        XCTAssertEqual(cleanSecond.info.promptTokenCount, cleanRenderedLength)
        XCTAssertEqual(second.text, cleanSecond.text)
    }

    func testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let session = ChatSession(
            model(processor: processor),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2),
            generateParameters: parameters)

        _ = try await collectGeneration(session.streamDetails(to: "first"))
        let firstRenderedLengthValue = await lengthIterator.next()
        _ = try XCTUnwrap(firstRenderedLengthValue)
        let optionalFirstProgress = await session.cacheProgress()
        let firstProgress = try XCTUnwrap(optionalFirstProgress)
        XCTAssertEqual(firstProgress.main, firstProgress.draft)

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        XCTAssertNotNil(second.info.speculativeDecodingTelemetry)
        XCTAssertEqual(
            second.info.promptTokenCount,
            secondRenderedLength - firstProgress.main)
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

    func testSpeculativeDecodingFallsBackForPreparedMedia() async throws {
        var context = model()
        context.processor = PreparedImageInputProcessor()
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
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

    /// Carried model state does not disqualify a turn from speculation. The
    /// anchors these models carry position from the cache offset, and a
    /// rejected proposal rewinds that offset along with the KV rows.
    func testSpeculativeDecodingRunsWithCarriedModelState() async throws {
        let context = model()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let cache = try context.model.newCache(parameters: parameters)
        let stateKey = LMOutput.Key<MLXArray>("test.carriedState")
        var state = LMOutput.State()
        state[stateKey] = MLXArray([Int32(1)])
        let session = ChatSession(
            context,
            cache: cache,
            state: state,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: parameters
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
        XCTAssertNotNil(completionInfo.speculativeDecodingTelemetry)
    }

    /// Speculation refused on one turn must not disable it for the rest of the session.
    /// The media turn drops the draft cache; the next text turn extends the cached prefix, which
    /// sets `reusedMainCacheWithoutDraft` so both caches are rebuilt from the full input and
    /// speculation resumes. Without that, the gate stays false forever once the main cache is
    /// warm and the draft cache is nil.
    func testSpeculativeDecodingResumesAfterAMediaTurn() async throws {
        // The prefix-preserving tokenizer makes turn 2 render as an exact extension of turn 1, so
        // the session takes the suffix-reuse path and keeps a warm main cache with a valid ledger.
        // That is what sets `reusedMainCacheWithoutDraft`; with an ordinary tokenizer turn 2
        // rebuilds from cold instead and never exercises this path.
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var context = model()
        context.processor = FirstTurnImageInputProcessor(
            base: TestInputProcessor(
                tokenizer: PrefixPreservingTokenizer(renderedLengthContinuation: continuation),
                configuration: ModelConfiguration(id: "test"),
                messageGenerator: DefaultMessageGenerator()))
        _ = renderedLengths
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        func telemetryForTurn(_ prompt: String) async throws -> SpeculativeDecodingTelemetry? {
            var info: GenerateCompletionInfo?
            for try await generation in session.streamDetails(
                to: prompt,
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {
                if let generationInfo = generation.info {
                    info = generationInfo
                }
            }
            return try XCTUnwrap(info).speculativeDecodingTelemetry
        }

        let mediaTurn = try await telemetryForTurn("describe this")
        XCTAssertNil(
            mediaTurn, "a media turn cannot speculate — the draft would have to prefill it")

        let textTurn = try await telemetryForTurn("and now in one word")
        XCTAssertNotNil(
            textTurn,
            "speculation must resume once a text turn can rebuild both caches from the transcript")
    }

    func testSpeculativeDecodingFallsBackForPrebuiltCacheWithoutDraftCache() async throws {
        let context = model()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let cache = try context.model.newCache(parameters: parameters)
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("cached prefix")]))
        _ = try TokenIterator(
            input: input,
            model: context.model,
            cache: cache,
            parameters: parameters)
        XCTAssertTrue(cache.contains { $0.offset > 0 })

        let session = ChatSession(
            context,
            cache: cache,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: parameters
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

        let promptCache = try loadPromptCacheSnapshot(url: url)
        let restored = ChatSession(
            ctx, promptCache: promptCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testSaveCachePreservesRestoredState() async throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let stateKey = LMOutput.Key<MLXArray>("test.chatSessionState")
        let stateValue = MLXArray([Int32(2), 4, 6])
        var state = LMOutput.State()
        state[stateKey] = stateValue
        let promptCache = PromptCacheSnapshot(cache: [cache], metadata: [:], state: state)
        let session = ChatSession(
            model(), promptCache: promptCache, generateParameters: generationParameters)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try await session.saveCache(to: url)
        let restored = try loadPromptCacheSnapshot(url: url)
        let restoredState = try XCTUnwrap(restored.state?[stateKey])

        XCTAssertTrue(allClose(restoredState, stateValue, rtol: 0, atol: 0).item(Bool.self))
    }

    // MARK: - Carrying model state

    /// Stateful continuation models cannot safely rewind arbitrary model state
    /// when speculative proposals are rejected, so the session must use normal
    /// generation without loading a draft model.
    /// A model that hands back an anchor on every prefill still speculates, and
    /// the anchor survives the speculative turn: the iterator seeds it, threads
    /// it through verification, and hands it back for the next turn to persist.
    func testSpeculativeDecodingCarriesModelStateProducedByItsTurns() async throws {
        let context = Self.makeStateProducingModel()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: parameters
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(to: "hello") {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }
        XCTAssertNotNil(try XCTUnwrap(info).speculativeDecodingTelemetry)

        // A second turn must succeed with the anchor the speculative turn carried.
        _ = try await session.respond(to: "hello again")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try await session.saveCache(to: url)

        let restored = try loadPromptCacheSnapshot(url: url)
        XCTAssertNotNil(
            restored.state?[StateProducingModel.anchorKey],
            "the session dropped the model state produced by its turns")
    }

    /// The end-to-end path the snapshot API exists for: a session that carries
    /// model state saves it with its cache, and a new session restored from
    /// that snapshot keeps generating.
    func testStateProducingSessionSurvivesSaveAndRestore() async throws {
        let context = Self.makeStateProducingModel()
        let session = ChatSession(context, generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try await session.saveCache(to: url)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        XCTAssertNotNil(
            snapshot.state?[StateProducingModel.anchorKey],
            "saveCache dropped the model state")

        let restored = ChatSession(
            Self.makeStateProducingModel(), promptCache: snapshot,
            generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    /// A snapshot can be built in process, not only read from disk, so a caller
    /// holding a cache and its state has a way to keep the two together.
    func testSessionAcceptsInProcessPromptCacheSnapshot() async throws {
        let context = Self.makeStateProducingModel()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)

        // Warm a cache directly, the way a caller building a prefix cache in
        // process would, and pair it with its state without going through disk.
        let cache = try context.model.newCache(parameters: parameters)
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("cached prefix")]))
        let iterator = try TokenIterator(
            input: input, model: context.model, cache: cache, parameters: parameters)
        XCTAssertTrue(cache.contains { $0.offset > 0 })

        let snapshot = PromptCacheSnapshot(cache: cache, state: iterator.state)
        XCTAssertNotNil(snapshot.state?[StateProducingModel.anchorKey])

        let restored = ChatSession(
            context, promptCache: snapshot, generateParameters: generationParameters)
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

    func testCacheStatusPreservesTheRealizedRequestWhenParametersChange() async throws {
        let initialConfiguration = KVCacheConfiguration(
            strategy: .turboQuant(.qualityFirst))
        let session = ChatSession(
            model(),
            cache: [KVCacheSimple()],
            generateParameters: GenerateParameters(kvCache: initialConfiguration))

        session.generateParameters.kvCache = KVCacheConfiguration(strategy: .fullPrecision)

        let status = try await session.cacheStatus()
        XCTAssertEqual(status.phase, .realized)
        XCTAssertEqual(status.requestSource, .typed)
        XCTAssertEqual(status.requestedConfiguration, initialConfiguration)
    }

    func testSessionRetainsCacheReplacementTriggeredDuringDecode() async throws {
        let (_, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let affine = try AffineKVCacheConfiguration(
            bits: 4, groupSize: 64, compressionStart: 8)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(
                maxTokens: 3,
                kvCache: KVCacheConfiguration(
                    strategy: .affine(affine), compatibility: .allowPartial),
                temperature: 0))

        _ = try await session.respond(to: "hello")
        let observedFirstOffset = try await session.withCache { cache in
            let cache = try XCTUnwrap(cache)
            let quantized = try XCTUnwrap(
                cache.first { $0 is QuantizedKVCache } as? QuantizedKVCache)
            return quantized.offset
        }
        let firstOffset = try XCTUnwrap(observedFirstOffset)
        let firstStatus = try await session.cacheStatus()
        XCTAssertEqual(firstStatus.processedTokenCount, firstOffset)
        XCTAssertEqual(firstStatus.phase, .realized)
        XCTAssertEqual(firstStatus.requestSource, .typed)
        XCTAssertEqual(firstStatus.requestedStrategy, .affine)
        XCTAssertGreaterThan(firstStatus.compressedLayerCount, 0)
        XCTAssertTrue(
            firstStatus.layers.contains {
                $0.state == .active && $0.resolvedStrategy == .affine
            })

        _ = try await session.respond(to: "hello again")
        try await session.withCache { cache in
            let cache = try XCTUnwrap(cache)
            let quantized = try XCTUnwrap(
                cache.first { $0 is QuantizedKVCache } as? QuantizedKVCache)
            XCTAssertGreaterThan(quantized.offset, firstOffset)
        }
    }

    func testModelContainerReportsCappedSlidingAndFullAttentionCaches() async throws {
        let container = ModelContainer(context: model())
        let parameters = GenerateParameters(maxTokens: 1, maxKVSize: 64)

        let status = try await container.cacheStatus(parameters: parameters)

        XCTAssertEqual(status.requestSource, .legacy)
        XCTAssertEqual(status.requestedConfiguration?.capacity?.maxTokens, 64)
        XCTAssertEqual(status.capacityDisposition, .fullyApplied)
        XCTAssertEqual(status.layers.count, 8)
        XCTAssertTrue(status.attentionMaxSizes.allSatisfy { $0 == 64 })
    }

    func testModelContainerPreservesNativeSlidingWindowForTypedCapacity() async throws {
        let container = ModelContainer(context: model())
        let capacity = try KVCacheConfiguration.Capacity(
            maxTokens: 64, preservedPrefixTokens: 3)
        let parameters = GenerateParameters(
            maxTokens: 1,
            kvCache: KVCacheConfiguration(capacity: capacity))

        let status = try await container.cacheStatus(parameters: parameters)

        XCTAssertEqual(status.requestSource, .typed)
        XCTAssertEqual(status.requestedConfiguration?.capacity, capacity)
        XCTAssertEqual(status.capacityDisposition, .fullyApplied)
        XCTAssertEqual(status.capacityAppliedLayerCount, 1)
        XCTAssertEqual(status.layers.count, 8)
        XCTAssertEqual(
            status.attentionMaxSizes.compactMap { $0 },
            [512, 512, 512, 512, 512, 64, 512, 512])
    }

    func testSessionReportsRealizedRawCacheInsteadOfPlannedLayout() async throws {
        let rawCache: [KVCache] = Array(repeating: 0, count: 8).map { _ in KVCacheSimple() }
        let session = ChatSession(
            model(),
            cache: rawCache,
            generateParameters: GenerateParameters(maxTokens: 1, maxKVSize: 64))

        let status = try await session.cacheStatus()

        XCTAssertEqual(status.phase, .realized)
        XCTAssertEqual(status.requestSource, .legacy)
        XCTAssertEqual(status.requestedConfiguration?.capacity?.maxTokens, 64)
        XCTAssertEqual(status.capacityDisposition, .ignored)
        XCTAssertTrue(status.attentionMaxSizes.allSatisfy { $0 == nil })
    }

    func testSessionRebuildsStructuredCacheWhenMaxKVSizeChanges() async throws {
        let session = ChatSession(
            model(), generateParameters: GenerateParameters(maxTokens: 1))

        _ = try await session.respond(to: "first")
        let initial = try await session.cacheStatus()
        XCTAssertEqual(initial.capacityDisposition, .notRequested)
        XCTAssertTrue(initial.attentionMaxSizes.contains(512))
        XCTAssertTrue(initial.attentionMaxSizes.contains(nil))

        session.generateParameters = GenerateParameters(maxTokens: 1, maxKVSize: 64)
        let stale = try await session.cacheStatus()
        XCTAssertEqual(stale.phase, .realized)
        XCTAssertNil(stale.requestedConfiguration)

        _ = try await session.respond(to: "second")
        let limited = try await session.cacheStatus()
        XCTAssertEqual(limited.capacityDisposition, .fullyApplied)
        XCTAssertEqual(limited.requestSource, .legacy)
        XCTAssertTrue(limited.attentionMaxSizes.allSatisfy { $0 == 64 })
        await session.withCache { cache in
            XCTAssertEqual(cache?.count, 8)
            XCTAssertTrue(cache?.allSatisfy { $0.maxSize == 64 } == true)
        }

        session.generateParameters = GenerateParameters(maxTokens: 1)
        _ = try await session.respond(to: "third")
        let restored = try await session.cacheStatus()
        XCTAssertEqual(restored.capacityDisposition, .notRequested)
        XCTAssertTrue(restored.attentionMaxSizes.contains(512))
        XCTAssertTrue(restored.attentionMaxSizes.contains(nil))
    }

    func testSessionRebuildsWhenTypedRequestRestoresNativeWindows() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 1, maxKVSize: 64))

        _ = try await session.respond(to: "legacy")
        let legacy = try await session.cacheStatus()
        XCTAssertEqual(legacy.capacityDisposition, .fullyApplied)
        XCTAssertEqual(legacy.requestSource, .legacy)
        XCTAssertTrue(legacy.attentionMaxSizes.allSatisfy { $0 == 64 })

        let capacity = try KVCacheConfiguration.Capacity(maxTokens: 64)
        session.generateParameters = GenerateParameters(
            maxTokens: 1,
            kvCache: KVCacheConfiguration(
                capacity: capacity,
                compatibility: .allowPartial))

        _ = try await session.respond(to: "typed")
        let typed = try await session.cacheStatus()
        XCTAssertEqual(typed.requestSource, .typed)
        XCTAssertEqual(typed.capacityDisposition, .fullyApplied)
        XCTAssertEqual(typed.capacityAppliedLayerCount, 1)
        XCTAssertEqual(
            typed.attentionMaxSizes.compactMap { $0 },
            [512, 512, 512, 512, 512, 64, 512, 512])
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
