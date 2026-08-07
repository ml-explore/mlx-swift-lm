// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Two-pass ChatSession test for the Harmony tool loop.
///
/// Guards the reported failure: Harmony control frames and analysis must never
/// land in `Chat.Message.content`, because the GPT-OSS chat template raises:
///
/// > You have passed a message containing <|channel|> tags in the content field.
///
/// This package does not ship a Jinja engine, so the template's four
/// `raise_exception` conditions are enforced by ``HarmonyTemplateContract``
/// inside the scripted tokenizer's `applyChatTemplate` — the same call the
/// real session uses on every tool restart.
final class HarmonyChatSessionRoundTripTests: XCTestCase {

    private enum Harmony {
        static let start = 0
        static let channel = 1
        static let message = 2
        static let end = 3
        static let call = 4
        static let `return` = 5
        static let constrain = 6
        static let tags: [String: Int] = [
            "<|start|>": start,
            "<|channel|>": channel,
            "<|message|>": message,
            "<|end|>": end,
            "<|call|>": call,
            "<|return|>": `return`,
            "<|constrain|>": constrain,
        ]
        static let firstText = 16
    }

    private enum TemplateContract: Error, Equatable {
        case channelTagsInContent
        case channelTagsInThinking
        case contentAndThinkingOnToolCall
        case toolResultWithoutPrecedingCall

        static let markers = [
            "<|channel|>analysis<|message|>",
            "<|channel|>final<|message|>",
        ]

        static func validate(_ messages: [[String: any Sendable]]) throws {
            var lastToolCallName: String?
            for message in messages {
                let role = message["role"] as? String
                if role == "assistant" {
                    let content = message["content"] as? String ?? ""
                    let thinking = message["thinking"] as? String ?? ""
                    if markers.contains(where: { content.contains($0) }) {
                        throw TemplateContract.channelTagsInContent
                    }
                    if markers.contains(where: { thinking.contains($0) }) {
                        throw TemplateContract.channelTagsInThinking
                    }
                    if let toolCalls = message["tool_calls"] as? [[String: any Sendable]] {
                        if !content.isEmpty && !thinking.isEmpty {
                            throw TemplateContract.contentAndThinkingOnToolCall
                        }
                        let function = toolCalls.first?["function"] as? [String: any Sendable]
                        lastToolCallName = function?["name"] as? String
                    } else {
                        lastToolCallName = nil
                    }
                }
                if role == "tool", lastToolCallName == nil {
                    throw TemplateContract.toolResultWithoutPrecedingCall
                }
            }
        }
    }

    private final class RenderLog: @unchecked Sendable {
        private let lock = NSLock()
        private var passes: [[[String: any Sendable]]] = []
        func record(_ messages: [[String: any Sendable]]) {
            lock.lock()
            passes.append(messages)
            lock.unlock()
        }
        var all: [[[String: any Sendable]]] {
            lock.lock()
            defer { lock.unlock() }
            return passes
        }
    }

    private final class HarmonyTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        let fragments: [String]
        let rendered: RenderLog
        init(fragments: [String], rendered: RenderLog) {
            self.fragments = fragments
            self.rendered = rendered
        }
        var vocabularySize: Int { Harmony.firstText + fragments.count }
        var bosToken: String? { nil }
        var eosToken: String? { "<|return|>" }
        var eosTokenId: Int? { Harmony.return }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try TemplateContract.validate(messages)
            rendered.record(messages)
            if messages.contains(where: { $0["tool_calls"] != nil }) {
                // Cold template omits private analysis. A correct Harmony tool
                // restart keeps the live cache (through the committed `<|call|>`)
                // and appends only the tool-result suffix after that boundary.
                // Multi-turn transcripts may contain earlier call tokens; the
                // session must splice at the *last* one.
                var tokens = [Harmony.start, Harmony.call]
                if messages.contains(where: { ($0["role"] as? String) == "tool" }) {
                    tokens.append(contentsOf: [
                        Harmony.start, Harmony.message, Harmony.firstText, Harmony.end,
                    ])
                }
                return tokens
            }
            return [Harmony.start]
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if let id = Harmony.tags[text] { return [id] }
            if let i = fragments.firstIndex(of: text) { return [Harmony.firstText + i] }
            return [Harmony.firstText]
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { id -> String in
                if let tag = Harmony.tags.first(where: { $0.value == id })?.key {
                    return skipSpecialTokens ? "" : tag
                }
                let i = id - Harmony.firstText
                guard i >= 0, i < fragments.count else { return "" }
                return fragments[i]
            }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? { Harmony.tags[token] }
        func convertIdToToken(_ id: Int) -> String? {
            Harmony.tags.first(where: { $0.value == id })?.key
        }
    }

    private final class ScriptedModel: Module, LLMModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        let kvHeads: [Int] = [1]
        let vocabularySize: Int
        private let scripts: [[Int]]
        private var pass = 0
        private var cursor = 0
        private(set) var prefillOffsets: [Int] = []
        private(set) var prefillTokenCounts: [Int] = []

        init(scripts: [[Int]], vocabularySize: Int) {
            self.scripts = scripts
            self.vocabularySize = vocabularySize
            super.init()
        }

        func beginPass() {
            if cursor > 0 {
                pass += 1
                cursor = 0
            }
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let steps = inputs.dim(-1)
            if cursor == 0 {
                prefillOffsets.append(cache?.first?.offset ?? 0)
                prefillTokenCounts.append(steps)
            }
            let cacheEntry = MLXArray.zeros([1, 1, steps, 1])
            _ = cache?.first?.update(keys: cacheEntry, values: cacheEntry)
            // Only the final position is sampled by TokenIterator; advance the
            // scripted generation stream once per forward pass so multi-token
            // tool-result prefills do not burn the response script.
            let script = scripts[min(pass, scripts.count - 1)]
            let token = cursor < script.count ? script[cursor] : Harmony.return
            cursor += 1
            var row = [Float](repeating: -30, count: vocabularySize)
            row[token] = 30
            let logits = MLXArray(row, [1, 1, vocabularySize])
            if steps == 1 {
                return logits
            }
            return concatenated(Array(repeating: logits, count: steps), axis: 1)
        }

        public var toolCallFormat: ToolCallFormat? { .gptOSS }
        var loraLayers: [Module] { [] }
    }

    /// Position-addressed model used to make a speculative round deterministic.
    /// With three drafts, every four generated tokens form one verifier round;
    /// the twelfth token below is therefore returned as the uncommitted final
    /// sample of the third round.
    private final class TimelineModel: Module, LLMModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        struct Call {
            var offset: Int
            var tokenCount: Int
        }

        let kvHeads: [Int] = [1]
        let vocabularySize: Int
        private let timeline: [Int]
        private(set) var calls: [Call] = []

        init(timeline: [Int], vocabularySize: Int) {
            self.timeline = timeline
            self.vocabularySize = vocabularySize
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let offset = cache?.first?.offset ?? 0
            let tokenCount = inputs.dim(-1)
            calls.append(Call(offset: offset, tokenCount: tokenCount))

            let entry = MLXArray.zeros([1, 1, tokenCount, 1])
            _ = cache?.first?.update(keys: entry, values: entry)

            var logits = [Float]()
            logits.reserveCapacity(tokenCount * vocabularySize)
            for position in 0 ..< tokenCount {
                let nextPosition = offset + position + 1
                let nextToken =
                    nextPosition < timeline.count ? timeline[nextPosition] : Harmony.return
                var row = [Float](repeating: -30, count: vocabularySize)
                row[nextToken] = 30
                logits.append(contentsOf: row)
            }
            return MLXArray(logits, [1, tokenCount, vocabularySize])
        }

        public var toolCallFormat: ToolCallFormat? { .gptOSS }
        var loraLayers: [Module] { [] }
    }

    func testHarmonyToolRoundTripContentNeverContainsChannelTags() async throws {
        let fragments = [
            "assistant",
            "analysis",
            "commentary to=functions.get_weather",
            "final",
            "Need weather.",
            #"{"city":"Paris"}"#,
            "Sunny in Paris.",
        ]
        func t(_ s: String) -> Int { Harmony.firstText + fragments.firstIndex(of: s)! }

        let scripts: [[Int]] = [
            [
                Harmony.channel, t("analysis"), Harmony.message,
                t("Need weather."), Harmony.end,
                Harmony.start, t("assistant"), Harmony.channel,
                t("commentary to=functions.get_weather"), Harmony.message,
                t(#"{"city":"Paris"}"#), Harmony.call,
            ],
            [
                Harmony.channel, t("final"), Harmony.message,
                t("Sunny in Paris."), Harmony.return,
            ],
        ]

        let renderLog = RenderLog()
        let tokenizer = HarmonyTokenizer(fragments: fragments, rendered: renderLog)
        let model = ScriptedModel(scripts: scripts, vocabularySize: tokenizer.vocabularySize)

        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "gpt-oss-test", toolCallFormat: .gptOSS),
            messageGenerator: GPTOSSMessageGenerator())
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)

        let dispatched = DispatchBox()
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 32, temperature: 0),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                model.beginPass()
                return #"{"forecast":"sunny"}"#
            })

        let reply = try await session.respond(to: "Weather in Paris?")

        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(dispatched.all.first?.function.name, "get_weather")
        XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))

        // Public reply is final-channel text only.
        XCTAssertTrue(reply.contains("Sunny in Paris."))
        XCTAssertFalse(reply.contains("<|channel|>"))
        XCTAssertFalse(reply.contains("Need weather"))

        // Every template render — including the tool restart — was legal.
        XCTAssertGreaterThanOrEqual(renderLog.all.count, 2)
        // Live cache retained through the generated call commit: prefill offset
        // is far past the cold template's short prefix, and only the tool-result
        // suffix (not a re-feed of `<|call|>`) is appended.
        XCTAssertGreaterThan(model.prefillOffsets[1], 1)
        XCTAssertEqual(model.prefillTokenCounts[1], 4)
        let restart = try XCTUnwrap(renderLog.all.last)
        try TemplateContract.validate(restart)

        let assistantWithCall = try XCTUnwrap(restart.first { $0["tool_calls"] != nil })
        let content = assistantWithCall["content"] as? String ?? ""
        XCTAssertFalse(content.contains("<|channel|>"))
        // Analysis is not persisted into the cold transcript.
        XCTAssertNil(assistantWithCall["thinking"])
        XCTAssertTrue(restart.contains { ($0["role"] as? String) == "tool" })
    }

    func testSpeculativeFinalCallTokenResumesFromTheLiveMainCache() async throws {
        let fragments = [
            "assistant",
            "analysis",
            "commentary to=functions.get_weather",
            "final",
            "Need weather.",
            #"{"city":"Paris"}"#,
            "Sunny in Paris.",
        ]
        func t(_ text: String) -> Int { Harmony.firstText + fragments.firstIndex(of: text)! }

        let firstGeneration = [
            Harmony.channel, t("analysis"), Harmony.message,
            t("Need weather."), Harmony.end,
            Harmony.start, t("assistant"), Harmony.channel,
            t("commentary to=functions.get_weather"), Harmony.message,
            t(#"{"city":"Paris"}"#), Harmony.call,
        ]
        XCTAssertEqual(firstGeneration.count, 12)

        // The cold restart prompt is [start, call, start, message, tool-result, end].
        // The live main cache ends immediately before `call`, so it must receive
        // the final five tokens as one main-only prefill.
        let timeline =
            [Harmony.start]
            + firstGeneration
            + [Harmony.start, Harmony.message, Harmony.firstText, Harmony.end]
            + [
                Harmony.channel, t("final"), Harmony.message,
                t("Sunny in Paris."), Harmony.return,
            ]

        let renderLog = RenderLog()
        let tokenizer = HarmonyTokenizer(fragments: fragments, rendered: renderLog)
        let configuration = ModelConfiguration(id: "gpt-oss-test", toolCallFormat: .gptOSS)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: GPTOSSMessageGenerator())
        let mainModel = TimelineModel(
            timeline: timeline, vocabularySize: tokenizer.vocabularySize)
        let draftModel = TimelineModel(
            timeline: timeline, vocabularySize: tokenizer.vocabularySize)
        let mainContext = ModelContext(
            configuration: configuration,
            model: mainModel,
            processor: processor,
            tokenizer: tokenizer)
        let draftContext = ModelContext(
            configuration: configuration,
            model: draftModel,
            processor: processor,
            tokenizer: tokenizer)

        let dispatched = DispatchBox()
        let session = ChatSession(
            mainContext,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: draftContext), numDraftTokens: 3),
            generateParameters: GenerateParameters(maxTokens: 32, temperature: 0),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast":"sunny"}"#
            })

        let reply = try await session.respond(to: "Weather in Paris?")

        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertTrue(reply.contains("Sunny in Paris."))
        XCTAssertFalse(reply.contains("Need weather"))
        XCTAssertTrue(
            mainModel.calls.contains { $0.offset == 12 && $0.tokenCount == 5 },
            "The main cache should retain private analysis and prefill call + tool result")
        XCTAssertFalse(
            draftModel.calls.contains { $0.offset >= 12 },
            "A lagging draft cache must not be reused for the Harmony continuation")
    }

    func testRawGenerationDoesNotApplyHarmonySemanticStopTokens() async throws {
        let fragments = ["raw token after call"]
        let tokenizer = HarmonyTokenizer(fragments: fragments, rendered: RenderLog())
        let configuration = ModelConfiguration(id: "gpt-oss-test", toolCallFormat: .gptOSS)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: GPTOSSMessageGenerator())
        let model = TimelineModel(
            timeline: [
                Harmony.start,
                Harmony.call,
                Harmony.firstText,
                Harmony.return,
            ],
            vocabularySize: tokenizer.vocabularySize)
        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)

        var tokens: [Int] = []
        let stream = try generateTokens(
            input: LMInput(tokens: MLXArray([Harmony.start])),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            context: context,
            includeStopToken: true)
        for await output in stream {
            if let token = output.token {
                tokens.append(token)
            }
        }

        XCTAssertEqual(tokens, [Harmony.call, Harmony.firstText, Harmony.return])
    }

    func testMessageGeneratorUsesStructuredToolMetadataOnly() throws {
        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": "Paris"]),
            id: "call_1")
        let messages = GPTOSSMessageGenerator().generate(messages: [
            .assistant("", toolCalls: [call]),
            .tool(#"{"forecast":"sunny"}"#, id: "call_1"),
        ])
        try TemplateContract.validate(messages)
        let toolCalls = try XCTUnwrap(messages[0]["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(toolCalls.count, 1)

        // JSON-shaped content is never re-interpreted as a tool call.
        let jsonish = #"{"name":"get_weather","arguments":{"city":"X"}}"#
        let plain = GPTOSSMessageGenerator().generate(messages: [.assistant(jsonish)])
        XCTAssertNil(plain[0]["tool_calls"])
        XCTAssertEqual(plain[0]["content"] as? String, jsonish)
    }

    private final class DispatchBox: @unchecked Sendable {
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

    private static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "weather",
            "parameters": [
                "type": "object",
                "properties": ["city": ["type": "string"] as [String: any Sendable]]
                    as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
}
