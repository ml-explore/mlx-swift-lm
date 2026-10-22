// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMCommon
import MLXNN
import MLXScriptedLM
import Testing

/// A small random-weight Gemma3 for scripted tests. Its layers mix sliding-window and
/// global attention.
func scriptedTestInnerModel(vocabularySize: Int) -> any LanguageModel {
    let config = Gemma3TextConfiguration.smallConfiguration(
        vocabularySize: vocabularySize)
    let model = Gemma3TextModel(config)
    eval(model)
    return model
}

/// Lowers one token's logit by a fixed amount.
private struct PenalizeToken: LogitProcessor {
    let token: Int
    let amount: Float

    mutating func prompt(_ prompt: MLXArray) {}
    mutating func didSample(token: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        let ids = MLXArray(Array(Int32(0) ..< Int32(vocabularySize)))
        return which(ids .== Int32(token), logits - amount, logits)
    }
}

@Suite(.serialized)
struct ScriptedLogitsModelTests {

    static let transcript = Transcript([
        .user("hello"),
        .assistant("hi there, how are you?"),
        .user("fine, and you?"),
        .assistant("doing well, thanks for asking."),
    ])

    static func scenario(margin: ScriptedLogitsModel.Margin = .dominant) throws
        -> ScriptedScenario
    {
        try ScriptedScenario(transcript: transcript, margin: margin) {
            scriptedTestInnerModel(vocabularySize: $0)
        }
    }

    @Test func testScriptedModel() async throws {
        // this can be used as a mock model that
        // works inside a ChatSession (or any other
        // way you would like to evaluate it)
        let scenario = try Self.scenario()
        let chat = ChatSession(scenario.context)

        let r1 = try await chat.respond(to: "hello")
        #expect(r1 == "hi there, how are you?")
        let r2 = try await chat.respond(to: "fine, and you?")
        #expect(r2 == "doing well, thanks for asking.")
    }

    @Test func testGoingOffScript() async throws {
        // The model will go by turns if you go off script
        let scenario = try Self.scenario()
        let chat = ChatSession(scenario.context)

        let r1 = try await chat.respond(to: "howdy doody!")
        #expect(r1 == "hi there, how are you?")
        let r2 = try await chat.respond(
            to: "i am doing prtty good but I wish it would stop raining.")
        #expect(r2 == "doing well, thanks for asking.")

        // and if you keep chatting past the end of the script it
        // produces empty responses
        let r3 = try await chat.respond(to: "what are you doing this weekend?")
        #expect(r3 == "")
    }

    @Test func testScripedKVCache() async throws {
        // the model produces a normal looking KVCache from
        // the inner model (a very small gemma3).  the
        // scripted model adds its own cache at the end
        let scenario = try Self.scenario()
        let kvCache = try scenario.model.newCache(parameters: nil)
        #expect(kvCache.count == (8 + 1))
    }

    // MARK: -

    /// Generates from the first user turn with a token iterator.
    static func firstTurnTokens(
        _ scenario: ScriptedScenario, parameters: GenerateParameters,
        processor: LogitProcessor? = nil
    ) async throws -> [Int] {
        let input = try await scenario.context.processor.prepare(
            input: UserInput(chat: [.user(transcript.userMessages[0])]))
        let script = scenario.model.scripts.scripts[0]
        var iterator: TokenIterator
        if let processor {
            iterator = try TokenIterator(
                input: input, model: scenario.model, processor: processor,
                sampler: ArgMaxSampler(), maxTokens: script.count)
        } else {
            var parameters = parameters
            parameters.maxTokens = script.count
            iterator = try TokenIterator(
                input: input, model: scenario.model, parameters: parameters)
        }
        return Array(IteratorSequence(iterator))
    }

    @Test func `greedy decoding emits the script`() async throws {
        let scenario = try Self.scenario()
        let tokens = try await Self.firstTurnTokens(
            scenario, parameters: GenerateParameters(temperature: 0))
        #expect(tokens == scenario.model.scripts.scripts[0])
        #expect(scenario.model.callLog.inputMismatches.isEmpty)
    }

    @Test func `seeded temperature sampling emits the script with a dominant margin`() async throws
    {
        let scenario = try Self.scenario()
        let tokens = try await Self.firstTurnTokens(
            scenario, parameters: GenerateParameters(temperature: 0.9, topP: 0.95, seed: 7))
        #expect(tokens == scenario.model.scripts.scripts[0])
    }

    @Test func `a processor overrides a plausible margin but not a dominant one`() async throws {
        let dominant = try Self.scenario(margin: .dominant)
        let first = dominant.model.scripts.scripts[0][0]
        let penalty = PenalizeToken(token: first, amount: 10)

        let kept = try await Self.firstTurnTokens(dominant, parameters: .init(), processor: penalty)
        #expect(kept.first == first)

        let plausible = try Self.scenario(margin: .plausible)
        let overridden = try await Self.firstTurnTokens(
            plausible, parameters: .init(), processor: penalty)
        #expect(overridden.first != first)
    }

    @Test func `the cache transcript holds the prompt and the emitted tokens`() async throws {
        let scenario = try Self.scenario()
        let input = try await scenario.context.processor.prepare(
            input: UserInput(chat: [.user(Self.transcript.userMessages[0])]))
        let prompt = input.text.tokens.asArray(Int32.self).map(Int.init)
        let script = scenario.model.scripts.scripts[0]

        let cache = try scenario.model.newCache(parameters: nil)
        let iterator = try TokenIterator(
            input: input, model: scenario.model, cache: cache,
            parameters: GenerateParameters(maxTokens: script.count, temperature: 0))
        let tokens = Array(IteratorSequence(iterator))

        // The iterator feeds each token to the model as it returns it.
        #expect(ScriptedLogitsModel.transcript(in: cache) == prompt + tokens)
    }

    @Test func `ChatSession follows the script across turns and reuses the cache`() async throws {
        let scenario = try Self.scenario()
        let session = ChatSession(
            scenario.context, generateParameters: GenerateParameters(maxTokens: 64, temperature: 0))

        let replies = [
            try await session.respond(to: Self.transcript.userMessages[0]),
            try await session.respond(to: Self.transcript.userMessages[1]),
        ]
        #expect(replies == Self.transcript.assistantTurns.map(\.text))

        let log = scenario.model.callLog
        #expect(log.inputMismatches.isEmpty)

        let prepares = log.calls.filter { $0.kind == .prepare }
        #expect(prepares.count == 2)
        #expect(prepares[0].cacheOffsetBefore == 0)
        #expect(prepares[1].cacheOffsetBefore > 0, "turn 2 should reuse turn 1's cache")
        #expect(log.calls.last?.cursor?.turn == 1)
    }

    // MARK: - Dropped turns

    static func session(_ transcript: Transcript) throws -> (ScriptedScenario, ChatSession) {
        let scenario = try ScriptedScenario(transcript: transcript) {
            scriptedTestInnerModel(vocabularySize: $0)
        }
        let session = ChatSession(
            scenario.context, generateParameters: GenerateParameters(maxTokens: 64, temperature: 0))
        return (scenario, session)
    }

    @Test func `an empty turn does not replay on the next turn`() async throws {
        let (_, session) = try Self.session(
            Transcript([
                .user("hello"), .assistant("hi there"),
                .user("say nothing"), .assistant(""),
                .user("and now?"), .assistant("back again"),
            ]))

        #expect(try await session.respond(to: "hello") == "hi there")
        #expect(try await session.respond(to: "say nothing") == "")
        #expect(try await session.respond(to: "and now?") == "back again")
    }

    @Test func `a cancelled turn does not replay on the next turn`() async throws {
        let (_, session) = try Self.session(
            Transcript([
                .user("hello"), .assistant("hi there"),
                .user("count"), .assistant("one two three four five"),
                .user("and now?"), .assistant("back again"),
            ]))

        #expect(try await session.respond(to: "hello") == "hi there")
        for try await _ in session.streamResponse(to: "count") {
            break
        }
        await session.synchronize()
        #expect(try await session.respond(to: "and now?") == "back again")
    }

    @Test func `a turn past the transcript ends at once`() async throws {
        let scenario = try Self.scenario()
        let session = ChatSession(scenario.context)

        for message in Self.transcript.userMessages {
            _ = try await session.respond(to: message)
        }
        #expect(try await session.respond(to: "one more?") == "")
    }

    // MARK: - Unscripted tokens

    @Test func `configured stop tokens are blocked after the script`() throws {
        let configuration = ModelConfiguration(id: "scripted", extraEOSTokens: ["<|user|>"])
        let scenario = try ScriptedScenario(
            transcript: Self.transcript, configuration: configuration
        ) {
            scriptedTestInnerModel(vocabularySize: $0)
        }
        let user = try #require(scenario.tokenizer.convertTokenToId("<|user|>"))
        #expect(scenario.model.scripts.blockedUnscriptedTokens.contains(user))
    }

    @Test func `unscripted tokens after a none terminator never end the turn`() async throws {
        let transcript = Transcript([.user("hello"), .assistant(.init("hi", terminator: .none))])
        let scenario = try ScriptedScenario(transcript: transcript) {
            scriptedTestInnerModel(vocabularySize: $0)
        }
        let input = try await scenario.context.processor.prepare(
            input: UserInput(chat: [.user("hello")]))
        let script = scenario.model.scripts.scripts[0]
        let maxTokens = script.count + 40

        let iterator = try TokenIterator(
            input: input, model: scenario.model,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 1, seed: 3))
        let tokens = Array(IteratorSequence(iterator))

        #expect(tokens.count == maxTokens)
        #expect(Array(tokens.prefix(script.count)) == script)
        let blocked = scenario.model.scripts.blockedUnscriptedTokens
        #expect(tokens.dropFirst(script.count).allSatisfy { !blocked.contains($0) })
    }

    // MARK: - Wrapper

    @Test func `the wrapper exposes the wrapped model's parameters under their own keys`() throws {
        let scenario = try Self.scenario()
        let model = scenario.model
        let inner = model.wrapped

        let keys = Set(model.parameters().flattened().map(\.0))
        #expect(keys == Set(inner.parameters().flattened().map(\.0)))
        #expect(!keys.contains { $0.hasPrefix("wrapped.") })

        // Loading through the wrapper updates the wrapped model.
        let (key, value) = try #require(inner.parameters().flattened().first)
        try model.update(
            parameters: ModuleParameters.unflattened([key: MLXArray.zeros(like: value)]),
            verify: [.noUnusedKeys, .shapeMismatch])
        let updated = try #require(inner.parameters().flattened().first { $0.0 == key })
        #expect(updated.1.abs().max().item(Float.self) == 0)
    }

    @Test func `cache status matches the wrapped model's`() throws {
        let model = try Self.scenario().model
        let parameters = GenerateParameters(maxKVSize: 256)
        #expect(
            try model.cacheStatus(parameters: parameters)
                == model.wrapped.cacheStatus(parameters: parameters))
    }
}
