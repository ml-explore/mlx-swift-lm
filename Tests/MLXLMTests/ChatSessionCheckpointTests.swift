// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

/// A template that drops a reply's reasoning cannot be followed by a trim once a sliding window
/// wraps or recurrent state moves on. These turns resume from the checkpoint the session took
/// before the reply.
final class ChatSessionCheckpointTests: XCTestCase {

    /// ChatML that drops past reasoning, as Mellum 2.1 and Qwen3 render history, and prefills an
    /// empty reasoning block when thinking is off.
    private struct ReasoningDroppingTemplate: ScriptedChatTemplate {
        var specialTokens: [String] {
            [ChatMLTemplate.imStart, ChatMLTemplate.imEnd, "<think>", "</think>"]
        }
        var corpus: [String] { ["user\n", "assistant\n", "\n\n", "word ", "reason", "answer"] }
        var endOfTurnToken: String? { ChatMLTemplate.imEnd }

        func render(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [String] {
            var segments: [String] = []
            let lastUser = messages.lastIndex { $0["role"] as? String == "user" } ?? -1
            for (index, message) in messages.enumerated() {
                let role = message["role"] as? String ?? ""
                var content = message["content"] as? String ?? ""
                if role == "assistant", index < lastUser, let end = content.range(of: "</think>") {
                    content = String(content[end.upperBound...].drop(while: \.isNewline))
                }
                segments += [ChatMLTemplate.imStart, role + "\n"]
                segments += content.isEmpty ? [] : [content]
                segments += [ChatMLTemplate.imEnd, "\n"]
            }
            segments += [ChatMLTemplate.imStart, "assistant\n"]
            if additionalContext?["enable_thinking"] as? Bool == false {
                segments += ["<think>", "\n\n", "</think>", "\n\n"]
            }
            return segments
        }
    }

    /// Serves `base`'s attention caches beside recurrent state that sums every token it has
    /// seen, and records each forward with the sum it started from.
    private final class RecurrentStateModel: Module, LanguageModel {
        static let carriedSum = LMOutput.Key<Int>("checkpoint.carriedSum")
        let base: any LanguageModel
        private(set) var forwards: [(tokens: [Int], stateBefore: Int, carriedBefore: Int)] = []

        init(_ base: any LanguageModel) {
            self.base = base
            super.init()
        }

        func clearForwards() { forwards.removeAll() }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            let recurrent = cache?.last as! MambaCache
            let tokens = input.tokens.asArray(Int.self)
            let before = recurrent[0]?.item(Int.self) ?? 0
            forwards.append((tokens, before, state?[Self.carriedSum] ?? 0))
            recurrent[0] = MLXArray(before + tokens.reduce(0, +))
            recurrent.offset += tokens.count
            let output = base(input, cache: cache.map { Array($0.dropLast()) }, state: nil)
            var carried = LMOutput.State()
            carried[Self.carriedSum] = (state?[Self.carriedSum] ?? 0) + tokens.reduce(0, +)
            return LMOutput(logits: output.logits, state: carried)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters) + [MambaCache()]
        }
    }

    /// Forces each pass to emit `tokens`.
    private struct ForcedReply: LogitProcessor {
        let tokens: [Int]
        var emitted = 0

        mutating func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            var row = [Float](repeating: -1e9, count: logits.dim(-1))
            row[tokens[min(emitted, tokens.count - 1)]] = 0
            return MLXArray(row).reshaped(logits.shape).asType(logits.dtype)
        }

        mutating func didSample(token: MLXArray) { emitted += 1 }
    }

    private let tokenizer = ScriptedTokenizer(
        corpus: [], template: ReasoningDroppingTemplate())
    private let firstQuestion = String(repeating: "word ", count: 40)

    private func makeSession(
        reply: String, reasoning: ReasoningConfig? = nil,
        additionalContext: [String: any Sendable]? = nil,
        prefill: PrefillParameters = .init(), kvCache: KVCacheConfiguration? = nil,
        headDim: Int = 16
    ) throws -> (ChatSession, RecurrentStateModel) {
        let eos = try XCTUnwrap(tokenizer.eosTokenId)
        let configuration = ModelConfiguration(
            id: "checkpoint-test", eosTokenIds: [eos], reasoningConfig: reasoning)
        let base = Gemma3TextModel(
            Gemma3TextConfiguration(
                modelType: "text", hiddenSize: 32, hiddenLayers: 2, intermediateSize: 32,
                attentionHeads: 2, headDim: headDim, rmsNormEps: 0.00001,
                vocabularySize: tokenizer.vocabulary.size, kvHeads: 1,
                ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
                queryPreAttnScalar: 16, slidingWindow: 16, slidingWindowPattern: 2,
                maxPositionEmbeddings: 4096))
        eval(base)
        let model = RecurrentStateModel(base)
        let replyTokens = tokenizer.encode(text: reply, addSpecialTokens: false) + [eos]
        let session = ChatSession(
            ModelContext(
                configuration: configuration, model: model,
                processor: TestInputProcessor(
                    tokenizer: tokenizer, configuration: configuration,
                    messageGenerator: DefaultMessageGenerator(),
                    vocabularySize: tokenizer.vocabulary.size),
                tokenizer: tokenizer),
            generateParameters: GenerateParameters(
                maxTokens: replyTokens.count + 1, kvCache: kvCache, temperature: 0, prefill: prefill
            ),
            additionalContext: additionalContext)
        session.components = GenerationComponents(logitProcessorFactory: {
            ForcedReply(tokens: replyTokens)
        })
        return (session, model)
    }

    private func respond(_ session: ChatSession, to prompt: String, role: Chat.Message.Role = .user)
        async throws -> (
            text: String, info: GenerateCompletionInfo
        )
    {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: prompt, role: role) {
            text += item.chunk ?? ""
            info = item.info ?? info
        }
        return (text, try XCTUnwrap(info))
    }

    private func render(
        _ messages: [(role: String, content: String)],
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: messages.map { ["role": $0.role, "content": $0.content] }, tools: nil,
            additionalContext: additionalContext)
    }

    /// Asserts the second turn prefilled exactly the prompt past `prefix`, starting from the
    /// recurrent state of the tokens before it.
    private func assertResumes(
        _ model: RecurrentStateModel, from prefix: Int, of prompt: [Int],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let prefilled = model.forwards.flatMap { $0.tokens }.prefix(prompt.count - prefix)
        XCTAssertEqual(Array(prefilled), Array(prompt.dropFirst(prefix)), file: file, line: line)
        XCTAssertEqual(
            model.forwards.first?.stateBefore, prompt.prefix(prefix).reduce(0, +),
            file: file, line: line)
        XCTAssertEqual(
            model.forwards.first?.carriedBefore, prompt.prefix(prefix).reduce(0, +),
            file: file, line: line)
    }

    func testCacheStatusReportsBoundedCheckpointMemorySeparately() async throws {
        let (session, _) = try makeSession(reply: "<think>\nreason\n</think>\n\nanswer")
        let planned = try await session.cacheStatus()
        XCTAssertEqual(planned.phase, .planned)
        XCTAssertNil(planned.checkpoint)

        let first = try await respond(session, to: firstQuestion)
        let status = try await session.cacheStatus()
        // Global history adds no checkpoint storage; only the window and recurrent state do.
        let checkpointBytes = 2 * 16 * 16 * MemoryLayout<Float>.size + MemoryLayout<Int32>.size
        XCTAssertEqual(status.phase, .realized)
        XCTAssertEqual(status.checkpoint?.memory, checkpointBytes)
        let checkpoint = try XCTUnwrap(status.checkpoint)
        let prefix = try render([("user", firstQuestion)]).count - 1
        XCTAssertEqual(checkpoint.tokens, prefix)
        XCTAssertLessThan(checkpoint.tokens, try XCTUnwrap(status.processedTokenCount))
        let liveBytes = await session.withCache { KVCacheStatus(cache: $0 ?? []).memoryBytes }
        XCTAssertEqual(status.memoryBytes, liveBytes)

        _ = try await respond(session, to: firstQuestion)
        let updated = try await session.cacheStatus()
        XCTAssertEqual(updated.checkpoint?.memory, checkpointBytes)
        let prompt = try render([
            ("user", firstQuestion), ("assistant", first.text), ("user", firstQuestion),
        ])
        XCTAssertEqual(updated.checkpoint?.tokens, prompt.count - 1)

        await session.clear()
        let cleared = try await session.cacheStatus()
        XCTAssertEqual(cleared.phase, .planned)
        XCTAssertEqual(cleared.memoryBytes, 0)
        XCTAssertNil(cleared.checkpoint)
    }

    func testForkReportsItsRetainedCheckpointAfterTheSourceIsCleared() async throws {
        let (session, _) = try makeSession(reply: "<think>\nreason\n</think>\n\nanswer")
        _ = try await respond(session, to: firstQuestion)
        let original = try await session.cacheStatus()
        let fork = await session.fork()
        let forked = try await fork.cacheStatus()
        let checkpoint = try XCTUnwrap(original.checkpoint)
        XCTAssertGreaterThan(checkpoint.memory, 0)
        XCTAssertEqual(forked.checkpoint, checkpoint)

        await session.clear()
        let retained = try await fork.cacheStatus()
        XCTAssertEqual(retained.checkpoint, forked.checkpoint)
        let reply = try await respond(fork, to: "word")
        XCTAssertGreaterThan(reply.info.cachedPromptTokenCount, 0)
    }

    func testReferencePrefillSchedulesReportNoCheckpointMemory() async throws {
        for chunking in [PrefillParameters.Chunking.unchunked, .remainder] {
            let (session, _) = try makeSession(reply: "answer", prefill: .init(chunking: chunking))
            _ = try await respond(session, to: firstQuestion)
            let status = try await session.cacheStatus()
            XCTAssertGreaterThan(status.memoryBytes, 0)
            XCTAssertNil(status.checkpoint)
        }
    }

    func testAppendOnlyTurnsKeepTheCheckpointForALaterRewrite() async throws {
        let (session, model) = try makeSession(reply: "answer")
        let first = try await respond(session, to: firstQuestion)
        let original = try await session.cacheStatus()
        let prefix = try XCTUnwrap(original.checkpoint).tokens
        let replyTokens =
            tokenizer.encode(
                text: "<think>\nreason\n</think>\n\nanswer", addSpecialTokens: false
            ) + [try XCTUnwrap(tokenizer.eosTokenId)]
        session.generateParameters.maxTokens = replyTokens.count + 1
        session.components = GenerationComponents(logitProcessorFactory: {
            ForcedReply(tokens: replyTokens)
        })

        let second = try await respond(session, to: "word")
        let extended = try await session.cacheStatus()
        XCTAssertGreaterThan(second.info.cachedPromptTokenCount, 0)
        XCTAssertEqual(extended.checkpoint, original.checkpoint)

        model.clearForwards()
        let third = try await respond(session, to: "answer")
        let prompt = try render([
            ("user", firstQuestion), ("assistant", first.text), ("user", "word"),
            ("assistant", second.text), ("user", "answer"),
        ])
        XCTAssertEqual(third.info.cachedPromptTokenCount, prefix)
        assertResumes(model, from: prefix, of: prompt)
    }

    func testDroppedReasoningResumesBeforeTheLastPromptToken() async throws {
        let (session, model) = try makeSession(reply: "<think>\nreason\n</think>\n\nanswer")
        let first = try await respond(session, to: firstQuestion)
        let firstPrompt = try render([("user", firstQuestion)])
        XCTAssertGreaterThan(firstPrompt.count, 16, "the first prompt must wrap the window")

        model.clearForwards()
        let second = try await respond(session, to: "word")
        let secondPrompt = try render([
            ("user", firstQuestion), ("assistant", first.text), ("user", "word"),
        ])

        XCTAssertEqual(second.info.cachedPromptTokenCount, firstPrompt.count - 1)
        XCTAssertEqual(
            second.info.promptTokenCount, secondPrompt.count - (firstPrompt.count - 1))
        assertResumes(model, from: firstPrompt.count - 1, of: secondPrompt)

        model.clearForwards()
        let third = try await respond(session, to: "answer")
        XCTAssertEqual(third.info.cachedPromptTokenCount, secondPrompt.count - 1)
    }

    func testDroppedPrefilledBlockResumesBeforeIt() async throws {
        let thinkingOff: [String: any Sendable] = ["enable_thinking": false]
        let (session, model) = try makeSession(
            reply: "answer", reasoning: .thinkTagsWithEnableThinking,
            additionalContext: thinkingOff)
        let first = try await respond(session, to: firstQuestion)
        let firstPrompt = try render(
            [("user", firstQuestion)], additionalContext: thinkingOff)
        let block = tokenizer.encode(
            text: "<think>\n\n</think>\n\n", addSpecialTokens: false
        ).count

        model.clearForwards()
        let second = try await respond(session, to: "word")
        let secondPrompt = try render(
            [("user", firstQuestion), ("assistant", first.text), ("user", "word")],
            additionalContext: thinkingOff)

        XCTAssertEqual(second.info.cachedPromptTokenCount, firstPrompt.count - block)
        assertResumes(model, from: firstPrompt.count - block, of: secondPrompt)
    }

    func testUnchunkedSessionPrefillsTheWholePromptInOneForward() async throws {
        let (session, model) = try makeSession(
            reply: "<think>\nreason\n</think>\n\nanswer", prefill: .init(chunking: .unchunked))
        _ = try await respond(session, to: firstQuestion)
        let prompt = try render([("user", firstQuestion)])
        XCTAssertEqual(model.forwards.first?.tokens, prompt)
        model.clearForwards()
        let second = try await respond(session, to: "word")
        XCTAssertEqual(second.info.cachedPromptTokenCount, 0)
        XCTAssertEqual(model.forwards.first?.stateBefore, 0)
        XCTAssertEqual(model.forwards.first?.carriedBefore, 0)
    }

    func testForkRestoresIndependentlyAndRejectsADifferentPrefix() async throws {
        let (session, _) = try makeSession(reply: "<think>\nreason\n</think>\n\nanswer")
        _ = try await respond(session, to: firstQuestion)
        let fork = await session.fork()
        let prefix = try render([("user", firstQuestion)]).count - 1
        let branch = try await respond(fork, to: "word")
        let original = try await respond(session, to: "answer")
        XCTAssertEqual(branch.info.cachedPromptTokenCount, prefix)
        XCTAssertEqual(original.info.cachedPromptTokenCount, prefix)

        let unrelated = await session.fork(history: [])
        let cold = try await respond(unrelated, to: "different question")
        XCTAssertEqual(cold.info.cachedPromptTokenCount, 0)
    }

    func testToolContinuationKeepsTheCheckpointBeforeTheFirstReply() async throws {
        let (session, model) = try makeSession(reply: "<think>\nreason\n</think>\n\nanswer")
        let first = try await respond(session, to: firstQuestion)
        let tool = try await respond(session, to: "result", role: .tool)
        let prefix = try render([("user", firstQuestion)]).count - 1
        XCTAssertGreaterThan(tool.info.cachedPromptTokenCount, 0)

        model.clearForwards()
        let next = try await respond(session, to: "word")
        let prompt = try render([
            ("user", firstQuestion), ("assistant", first.text), ("tool", "result"),
            ("assistant", tool.text), ("user", "word"),
        ])
        XCTAssertEqual(next.info.cachedPromptTokenCount, prefix)
        assertResumes(model, from: prefix, of: prompt)
    }

    func testTileCompressionIsRealizedBeforeSavingTheFirstCheckpoint() async throws {
        let configuration = KVCacheConfiguration(
            strategy: .varianceNormalized(try .init(tileSize: 32, sinkhornIterations: 2)),
            compatibility: .allowPartial)
        let (session, model) = try makeSession(
            reply: "<think>\nreason\n</think>\n\nanswer", kvCache: configuration, headDim: 32)
        let first = try await respond(session, to: firstQuestion)
        let prefix = try render([("user", firstQuestion)]).count - 1
        let status = try await session.cacheStatus()
        let slidingBytes = 2 * 16 * 32 * MemoryLayout<Float>.size
        let tailBytes = 2 * (prefix % 32) * 32 * MemoryLayout<Float>.size
        XCTAssertEqual(status.checkpoint?.tokens, prefix)
        XCTAssertEqual(
            status.checkpoint?.memory, slidingBytes + tailBytes + MemoryLayout<Int32>.size)
        model.clearForwards()
        let second = try await respond(session, to: "word")
        let prompt = try render([
            ("user", firstQuestion), ("assistant", first.text), ("user", "word"),
        ])
        XCTAssertEqual(second.info.cachedPromptTokenCount, prefix)
        assertResumes(model, from: prefix, of: prompt)
    }
}
