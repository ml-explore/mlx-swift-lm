// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon

/// Reasoning streamed apart from the answer must still reach the next prompt, or a
/// tool loop re-renders history without it and the prompt cache rewinds.
///
/// Based on repro cases from the PR 563 review.
@Suite("Reasoning replay", .serialized)
struct ReasoningReplayTests {

    // MARK: - Message generators

    private func assistantCall(reasoning: String?) -> Chat.Message {
        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": .string("Paris")]))
        var message = Chat.Message.assistant("", toolCalls: [call])
        message.reasoning = reasoning
        return message
    }

    @Test("The default generator replays recorded reasoning as reasoning_content")
    func defaultGeneratorReplaysReasoning() {
        let generator = DefaultMessageGenerator()
        let raw = generator.generate(message: assistantCall(reasoning: "thought"))
        #expect(raw["reasoning_content"] as? String == "thought")
        #expect(
            generator.generate(message: assistantCall(reasoning: nil))["reasoning_content"] == nil)
    }

    @Test("Qwen3 (MLXLLM) replays recorded reasoning")
    func qwen3LLMGeneratorReplaysReasoning() throws {
        let raw = try qwen3Generator().generate(message: assistantCall(reasoning: "thought"))
        #expect(raw["reasoning_content"] as? String == "thought")
    }

    @Test("Gemma 4 (MLXVLM) replays recorded reasoning")
    func gemma4VLMGeneratorReplaysReasoning() {
        let raw = Gemma4MessageGenerator().generate(message: assistantCall(reasoning: "thought"))
        #expect(raw["reasoning_content"] as? String == "thought")
        let none = Gemma4MessageGenerator().generate(message: assistantCall(reasoning: nil))
        #expect(none["reasoning_content"] == nil)
    }

    private func qwen3Generator() throws -> any MessageGenerator {
        let configuration = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "model_type": "qwen3", "vocab_size": 128, "hidden_size": 8,
                  "num_hidden_layers": 1, "intermediate_size": 16,
                  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                  "rms_norm_eps": 1e-6, "tie_word_embeddings": true
                }
                """.utf8))
        return Qwen3Model(configuration).messageGenerator(tokenizer: ByteTokenizer())
    }

    // MARK: - Tool loops keep the prompt cache

    private let qwenFirstOutput =
        bytes(
            #"<think>thought</think><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#
        ) + [256]

    @Test("A Qwen3 (MLXLLM) tool loop re-renders its thought and keeps the cache")
    func qwen3LLMToolLoopKeepsCache() async throws {
        let (prompts, infos) = try await runToolLoop(
            generator: try qwen3Generator(), firstOutput: qwenFirstOutput,
            secondOutput: bytes("Sunny.") + [256])

        #expect(prompts.count == 2)
        #expect(text(prompts[1]).contains("<think>thought</think><tool_call>"))
        #expect(infos.last?.cachedPromptTokenCount == prompts[0].count + qwenFirstOutput.count)
    }

    @Test("A Qwen3-VL tool loop re-renders its thought and keeps the cache")
    func qwen3VLToolLoopKeepsCache() async throws {
        let (prompts, infos) = try await runToolLoop(
            generator: Qwen3VLMessageGenerator(), firstOutput: qwenFirstOutput,
            secondOutput: bytes("Sunny.") + [256])

        #expect(prompts.count == 2)
        #expect(prompts[1].starts(with: prompts[0] + qwenFirstOutput))
        #expect(infos.last?.cachedPromptTokenCount == prompts[0].count + qwenFirstOutput.count)
    }

    @Test("A Gemma 4 (MLXLLM) tool loop re-renders its thought channel and keeps the cache")
    func gemma4LLMToolLoopKeepsCache() async throws {
        // MLXLLM's Gemma 4 models use the default generator.
        let firstOutput =
            bytes(
                "<|channel>thought\nI need the weather\n<channel|>"
                    + #"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#
            ) + [256]
        let log = EventLog()
        let (prompts, infos) = try await runToolLoop(
            processor: GemmaLikeProcessor(generator: DefaultMessageGenerator(), log: log),
            log: log, format: .gemma4, reasoning: .gemma4,
            firstOutput: firstOutput, secondOutput: bytes("Sunny.") + [256])

        #expect(prompts.count == 2)
        #expect(
            text(prompts[1]).contains(
                "<|channel>thought\nI need the weather\n<channel|><|tool_call>"))
        #expect(infos.last?.cachedPromptTokenCount == prompts[0].count + firstOutput.count)
    }

    // MARK: - The prompt is read before the prefill is queued

    @Test("generate() reads the prompt before the first forward pass")
    func generateReadsPromptBeforePrefill() async throws {
        let log = EventLog()
        let prompt = bytes("user:\nhi\nassistant:\n")
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "replay", eosTokenIds: [256], reasoningConfig: QwenReasoningProtocol.qwen3),
            model: ScriptedModel(scripts: [bytes("ok") + [256]], log: log),
            processor: QwenLikeProcessor(generator: DefaultMessageGenerator(), log: log),
            tokenizer: ByteTokenizer(log: log))

        let stream = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(prompt)),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0), context: context)
        for await _ in stream {}

        let events = log.events
        let firstForward = try #require(events.firstIndex(of: .forward))
        #expect(events[..<firstForward].contains(.decode(prompt)))
        #expect(!events[firstForward...].contains(.decode(prompt)))
    }

    @Test("Speculative generate() reads the prompt before the first forward pass")
    func speculativeGenerateReadsPromptBeforePrefill() async throws {
        let log = EventLog()
        let prompt = bytes("user:\nhi\nassistant:\n")
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "replay", eosTokenIds: [256], reasoningConfig: QwenReasoningProtocol.qwen3),
            model: ScriptedModel(scripts: [bytes("ok") + [256]], log: log),
            processor: QwenLikeProcessor(generator: DefaultMessageGenerator(), log: log),
            tokenizer: ByteTokenizer(log: log))

        let stream = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(prompt)),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0), context: context,
            draftModel: ScriptedModel(scripts: [bytes("ok") + [256]], log: log),
            numDraftTokens: 2)
        for await _ in stream {}

        let events = log.events
        let firstForward = try #require(events.firstIndex(of: .forward))
        #expect(events[..<firstForward].contains(.decode(prompt)))
        #expect(!events[firstForward...].contains(.decode(prompt)))
    }

    @Test("ChatSession decodes the prompt tail once per turn, before the prefill")
    func chatSessionDecodesPromptTailOnce() async throws {
        let log = EventLog()
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "replay", eosTokenIds: [256], reasoningConfig: QwenReasoningProtocol.qwen3),
            model: ScriptedModel(scripts: [bytes("ok") + [256]], log: log),
            processor: QwenLikeProcessor(generator: DefaultMessageGenerator(), log: log),
            tokenizer: ByteTokenizer(log: log))
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 4, temperature: 0))

        _ = try await session.respond(to: "hi")

        let prompt = try #require(log.prompts.first)
        let events = log.events
        let firstForward = try #require(events.firstIndex(of: .forward))
        #expect(events.filter { $0 == .decode(prompt) }.count == 1)
        #expect(events[..<firstForward].contains(.decode(prompt)))
    }

    // MARK: - Gemma 4 tool call inside an unclosed thought channel

    @Test("A Gemma 4 tool call inside an unclosed thought channel ends the thought")
    func gemma4ToolCallInsideUnclosedChannel() {
        let result = runDecoder([
            "<|channel>", "thought\n", "I need the weather",
            #"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
        ])

        #expect(result.toolCalls.map(\.function.name) == ["get_weather"])
        #expect(result.reasoning == "I need the weather")
        #expect(result.response.isEmpty)
    }

    @Test("A Gemma 4 tool call after the channel closes is parsed")
    func gemma4ToolCallAfterClosedChannel() {
        let result = runDecoder([
            "<|channel>", "thought\n", "I need the weather", "<channel|>",
            #"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
        ])

        #expect(result.toolCalls.map(\.function.name) == ["get_weather"])
        #expect(result.reasoning == "I need the weather")
    }

    @Test("A Gemma 4 tool call token inside an unclosed thought channel ends the thought")
    func gemma4ToolCallTokenInsideUnclosedChannel() {
        // Every delimiter is its own token here, so all are matched by id.
        let result = runDecoder([
            "<|channel>", "thought\n", "I need the weather", "<|tool_call>",
            #"call:get_weather{city:<|"|>Paris<|"|>}"#, "<tool_call|>", "<channel|>",
        ])

        #expect(result.toolCalls.map(\.function.name) == ["get_weather"])
        #expect(result.reasoning == "I need the weather")
    }
}

// MARK: - Helpers

private func bytes(_ text: String) -> [Int] { text.utf8.map(Int.init) }

private func text(_ tokens: [Int]) -> String {
    String(decoding: tokens.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
}

private func runToolLoop(
    generator: any MessageGenerator, firstOutput: [Int], secondOutput: [Int]
) async throws -> (prompts: [[Int]], infos: [GenerateCompletionInfo]) {
    let log = EventLog()
    return try await runToolLoop(
        processor: QwenLikeProcessor(generator: generator, log: log), log: log,
        format: .json, reasoning: QwenReasoningProtocol.qwen3,
        firstOutput: firstOutput, secondOutput: secondOutput)
}

private func runToolLoop(
    processor: any UserInputProcessor, log: EventLog, format: ToolCallFormat,
    reasoning: ReasoningConfig, firstOutput: [Int], secondOutput: [Int]
) async throws -> (prompts: [[Int]], infos: [GenerateCompletionInfo]) {
    let context = ModelContext(
        configuration: ModelConfiguration(
            id: "repro", eosTokenIds: [256], toolCallFormat: format,
            reasoningConfig: reasoning),
        model: ScriptedModel(scripts: [firstOutput, secondOutput], log: nil),
        processor: processor,
        tokenizer: ByteTokenizer())
    let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "parameters": [
                "type": "object",
                "properties": ["city": ["type": "string"] as [String: any Sendable]]
                    as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
    let session = ChatSession(
        context, generateParameters: GenerateParameters(maxTokens: 200, temperature: 0),
        tools: [weatherTool], toolDispatch: { _ in "sunny" })

    var infos: [GenerateCompletionInfo] = []
    for try await item in session.streamDetails(to: "Weather in Paris?") {
        if let info = item.info { infos.append(info) }
    }
    return (log.prompts, infos)
}

private struct DecoderResult {
    var reasoning = ""
    var response = ""
    var toolCalls: [ToolCall] = []
}

/// Drives `StandardTokenStreamDecoder` with Gemma 4's protocol, one fragment per token.
private func runDecoder(_ fragments: [String]) -> DecoderResult {
    let decoding = Dictionary(
        uniqueKeysWithValues: fragments.enumerated().map { ($0.offset + 1, $0.element) })
    var decoder = StandardTokenStreamDecoder(
        tokenizer: FragmentTokenizer(decoding: decoding), format: .gemma4, tools: nil,
        stopStrings: [], reasoning: (config: .gemma4, primedInside: false))
    var result = DecoderResult()
    func consume(_ event: TokenStreamEvent) -> Bool {
        switch event {
        case .reasoning(let text): result.reasoning += text
        case .response(let text): result.response += text
        case .toolCall(let call): result.toolCalls.append(call)
        case .rejectedToolCall, .protocolError, .stop: break
        }
        return true
    }
    for id in 1 ... fragments.count {
        if !decoder.push(id, emit: consume) { break }
    }
    _ = decoder.finish(emit: consume)
    return result
}

private enum Event: Equatable {
    case forward
    case decode([Int])
}

private final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [Event] = []
    private var _prompts: [[Int]] = []

    func record(_ event: Event) {
        lock.withLock { _events.append(event) }
    }

    func record(prompt: [Int]) {
        lock.withLock { _prompts.append(prompt) }
    }

    var events: [Event] { lock.withLock { _events } }
    var prompts: [[Int]] { lock.withLock { _prompts } }
}

/// A Qwen-like template: replays `reasoning_content` as `<think>...</think>` before the
/// assistant content, and ends each assistant turn with the EOS token (256).
private struct QwenLikeProcessor: UserInputProcessor {
    let generator: any MessageGenerator
    let log: EventLog

    func prepare(input: UserInput) throws -> LMInput {
        var tokens: [Int] = []
        for raw in generator.generate(from: input) {
            let role = raw["role"] as? String ?? ""
            var rendered = "\(role):\n"
            if role == "assistant", let reasoning = raw["reasoning_content"] as? String {
                rendered += "<think>\(reasoning)</think>"
            }
            if let content = raw["content"] as? String {
                rendered += content
            } else if let parts = raw["content"] as? [[String: String]] {
                rendered += parts.compactMap { $0["text"] }.joined()
            }
            for call in raw["tool_calls"] as? [[String: any Sendable]] ?? [] {
                let function = call["function"] as? [String: any Sendable] ?? [:]
                let name = function["name"] as? String ?? ""
                let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
                let json = try JSONSerialization.data(
                    withJSONObject: arguments, options: [.sortedKeys, .withoutEscapingSlashes])
                rendered +=
                    #"<tool_call>{"name":"\#(name)","arguments":"#
                    + String(decoding: json, as: UTF8.self) + "}</tool_call>"
            }
            tokens += bytes(rendered)
            if role == "assistant" { tokens.append(256) }
            tokens += bytes("\n")
        }
        tokens += bytes("assistant:\n")
        log.record(prompt: tokens)
        return LMInput(tokens: MLXArray(tokens))
    }
}

/// A Gemma-4-like template: replays `reasoning` or `reasoning_content` as a thought
/// channel on assistant tool-call turns, using the template's exact replay string.
private struct GemmaLikeProcessor: UserInputProcessor {
    let generator: any MessageGenerator
    let log: EventLog

    func prepare(input: UserInput) throws -> LMInput {
        var tokens: [Int] = []
        for raw in generator.generate(from: input) {
            let role = raw["role"] as? String ?? ""
            let calls = raw["tool_calls"] as? [[String: any Sendable]] ?? []
            var rendered = "\(role):\n"
            if role == "assistant", !calls.isEmpty,
                let reasoning = (raw["reasoning"] ?? raw["reasoning_content"]) as? String
            {
                rendered += "<|channel>thought\n" + reasoning + "\n<channel|>"
            }
            if let content = raw["content"] as? String {
                rendered += content
            } else if let parts = raw["content"] as? [[String: String]] {
                rendered += parts.compactMap { $0["text"] }.joined()
            }
            for call in calls {
                let function = call["function"] as? [String: any Sendable] ?? [:]
                let name = function["name"] as? String ?? ""
                let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
                let body = arguments.keys.sorted().map {
                    "\($0):<|\"|>\(arguments[$0] as? String ?? "")<|\"|>"
                }.joined(separator: ",")
                rendered += "<|tool_call>call:\(name){\(body)}<tool_call|>"
            }
            tokens += bytes(rendered)
            if role == "assistant" { tokens.append(256) }
            tokens += bytes("\n")
        }
        tokens += bytes("assistant:\n")
        log.record(prompt: tokens)
        return LMInput(tokens: MLXArray(tokens))
    }
}

/// One token per UTF-8 byte; 256 is EOS.
private struct ByteTokenizer: Tokenizer {
    var log: EventLog? = nil

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var eosTokenId: Int? { 256 }
    var unknownTokenId: Int? { 257 }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { bytes(text) }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        log?.record(.decode(tokenIds))
        return text(tokenIds)
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

/// Decodes each id to a fixed fragment; a fragment that is a whole delimiter is one token.
private struct FragmentTokenizer: Tokenizer {
    let decoding: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        decoding.first { $0.value == text }.map { [$0.key] } ?? []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { decoding[$0] ?? "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { decoding.first { $0.value == token }?.key }
    func convertIdToToken(_ id: Int) -> String? { decoding[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Emits one scripted token sequence per `prepare` call (one per generation pass).
private final class ScriptedModel: Module, LanguageModel, KVCacheDimensionProvider {
    let scripts: [[Int]]
    let log: EventLog?
    private var pass = -1
    private var index = 0
    var kvHeads: [Int] { [1] }

    init(scripts: [[Int]], log: EventLog?) {
        self.scripts = scripts
        self.log = log
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        pass += 1
        index = 0
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        log?.record(.forward)
        let length = inputs.size
        if let cache = cache?.first {
            let entries = MLXArray.zeros([1, 1, length, 1])
            _ = cache.update(keys: entries, values: entries)
        }
        let script = scripts[min(pass, scripts.count - 1)]
        let token = script[min(index, script.count - 1)]
        index += 1
        var logits = Array(repeating: Float(-100), count: length * 258)
        for position in 0 ..< length { logits[position * 258 + token] = 100 }
        return MLXArray(logits, [1, length, 258])
    }
}
