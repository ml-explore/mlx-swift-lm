// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXOptimizers
import XCTest
import os

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
        let kvHeadDim: Int
        let nextToken: Int

        init(
            recorder: PreparedTokenRecorder,
            kvHeadCount: Int = 1,
            kvHeadDim: Int = 1,
            nextToken: Int = 0
        ) {
            self.recorder = recorder
            self.kvHeads = Array(repeating: 1, count: kvHeadCount)
            self.kvHeadDim = kvHeadDim
            self.nextToken = nextToken
            super.init()
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            recorder.append(
                input.text.tokens.asArray(Int.self),
                shape: input.text.tokens.shape)
            let tokenCount = input.text.tokens.size
            update(cache: cache, tokenCount: tokenCount)
            return .logits(.init(logits: logits(tokenCount: tokenCount), state: nil))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokenCount = max(1, inputs.asArray(Int.self).count)
            if let cache {
                update(cache: cache, tokenCount: tokenCount)
            }
            return logits(tokenCount: tokenCount)
        }

        private func update(cache: [KVCache], tokenCount: Int) {
            for (index, layerCache) in cache.enumerated() {
                let heads = kvHeads[min(index, kvHeads.count - 1)]
                let keys = MLXArray.zeros([1, heads, tokenCount, kvHeadDim])
                let values = MLXArray.zeros([1, heads, tokenCount, kvHeadDim])
                if let quantizedCache = layerCache as? QuantizedKVCacheProtocol {
                    _ = quantizedCache.updateQuantized(keys: keys, values: values)
                } else {
                    _ = layerCache.update(keys: keys, values: values)
                }
            }
        }

        private func logits(tokenCount: Int) -> MLXArray {
            let row = (0 ..< 8).map { $0 == nextToken ? Float(1_000) : Float(-1_000) }
            return MLXArray(Array(repeating: row, count: tokenCount).flatMap { $0 })
                .reshaped(1, tokenCount, 8)
        }
    }

    private final class SequenceLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
        let recorder: PreparedTokenRecorder
        let kvHeads = [1]
        let tokens: [Int]
        private let index = OSAllocatedUnfairLock(initialState: 0)

        init(recorder: PreparedTokenRecorder, tokens: [Int]) {
            self.recorder = recorder
            self.tokens = tokens
            super.init()
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            recorder.append(
                input.text.tokens.asArray(Int.self),
                shape: input.text.tokens.shape)
            update(cache: cache, tokenCount: input.text.tokens.size)
            return .logits(.init(logits: logits(tokenCount: input.text.tokens.size), state: nil))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokenCount = max(1, inputs.asArray(Int.self).count)
            if let cache {
                update(cache: cache, tokenCount: tokenCount)
            }
            return logits(tokenCount: tokenCount)
        }

        private func update(cache: [KVCache], tokenCount: Int) {
            for layerCache in cache {
                let keys = MLXArray.zeros([1, 1, tokenCount, 1])
                let values = MLXArray.zeros([1, 1, tokenCount, 1])
                _ = layerCache.update(keys: keys, values: values)
            }
        }

        private func logits(tokenCount: Int) -> MLXArray {
            let tokens = self.tokens
            let token = index.withLock { index in
                let token = tokens[min(index, tokens.count - 1)]
                index += 1
                return token
            }
            let vocabularySize = (tokens.max() ?? token) + 1
            let row = (0 ..< vocabularySize).map { $0 == token ? Float(1_000) : Float(-1_000) }
            return MLXArray(Array(repeating: row, count: tokenCount).flatMap { $0 })
                .reshaped(1, tokenCount, vocabularySize)
        }
    }

    private final class GoldenLogitRecorder: Sendable {
        private let values = OSAllocatedUnfairLock(initialState: [[Float]]())

        func append(_ value: MLXArray) {
            let value = value.asArray(Float.self)
            values.withLock {
                $0.append(value)
            }
        }

        var snapshot: [[Float]] {
            values.withLock { $0 }
        }
    }

    private final class GoldenKVCache: BaseKVCache {
        private(set) var tokenHistory: [Int] = []

        override var supportsStaticPrefixReuse: Bool { true }

        func append(tokens: [Int]) {
            tokenHistory.append(contentsOf: tokens)
            offset = tokenHistory.count
        }

        override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            offset += keys.dim(2)
            return (keys, values)
        }

        override func copy() -> any KVCache {
            let new = GoldenKVCache()
            new.tokenHistory = tokenHistory
            new.offset = offset
            return new
        }
    }

    private final class GoldenLogitLanguageModel: Module, LanguageModel {
        let vocabularySize = 128
        let recorder: GoldenLogitRecorder?

        init(recorder: GoldenLogitRecorder? = nil) {
            self.recorder = recorder
            super.init()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [GoldenKVCache()]
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            let tokens = input.text.tokens.asArray(Int.self)
            let state = append(tokens: tokens, to: cache)
            let logits = logits(for: state, tokenCount: max(1, tokens.count))
            recorder?.append(logits[0, -1, 0...])
            return .logits(.init(logits: logits, state: nil))
        }

        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
            -> LMOutput
        {
            let tokens = input.tokens.asArray(Int.self)
            let state = append(tokens: tokens, to: cache ?? [])
            return .init(logits: logits(for: state, tokenCount: max(1, tokens.count)))
        }

        private func append(tokens: [Int], to cache: [KVCache]) -> [Int] {
            if let cache = cache.first as? GoldenKVCache {
                cache.append(tokens: tokens)
                return cache.tokenHistory
            }
            return tokens
        }

        private func logits(for tokens: [Int], tokenCount: Int) -> MLXArray {
            var hash = 17
            for token in tokens {
                hash = (hash &* 31 &+ token &+ 1) % 1_000_003
            }
            let target = 10 + (abs(hash) % (vocabularySize - 10))
            let row = (0 ..< vocabularySize).map { index -> Float in
                index == target ? 1_000 : Float(-abs(index - target))
            }
            return MLXArray(Array(repeating: row, count: tokenCount).flatMap { $0 })
                .reshaped(1, tokenCount, vocabularySize)
        }
    }

    private struct GoldenTokenizer: Tokenizer {
        let eosTokenId = 2

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            var tokens: [Int] = []
            var index = text.startIndex
            while index < text.endIndex {
                if text[index...].hasPrefix("<eos>") {
                    tokens.append(eosTokenId)
                    index = text.index(index, offsetBy: 5)
                    continue
                }
                if text[index...].hasPrefix("<g") {
                    var cursor = text.index(index, offsetBy: 2)
                    var digits = ""
                    while cursor < text.endIndex, text[cursor].isNumber {
                        digits.append(text[cursor])
                        cursor = text.index(after: cursor)
                    }
                    if cursor < text.endIndex, text[cursor] == ">", let token = Int(digits) {
                        tokens.append(token)
                        index = text.index(after: cursor)
                        continue
                    }
                }
                let scalarValue = Int(String(text[index]).unicodeScalars.first?.value ?? 0)
                tokens.append(20 + (scalarValue % 90))
                index = text.index(after: index)
            }
            return tokens
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { "<g\($0)>" }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            token == "<eos>" ? 2 : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            "<g\(id)>"
        }

        var bosToken: String? { nil }
        var eosToken: String? { "<eos>" }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens = [3]
            if tools != nil {
                tokens += [4, 5]
            }
            for message in messages {
                switch message["role"] as? String {
                case Chat.Message.Role.system.rawValue:
                    tokens.append(11)
                case Chat.Message.Role.user.rawValue:
                    tokens.append(12)
                case Chat.Message.Role.assistant.rawValue:
                    tokens.append(17)
                    tokens += encode(
                        text: message["content"] as? String ?? "",
                        addSpecialTokens: false)
                    continue
                case Chat.Message.Role.tool.rawValue:
                    tokens.append(14)
                default:
                    tokens.append(15)
                }
                tokens += encode(text: message["content"] as? String ?? "", addSpecialTokens: false)
                tokens.append(16)
            }
            tokens.append(17)
            return tokens
        }
    }

    private struct GoldenInputProcessor: UserInputProcessor {
        let tokenizer = GoldenTokenizer()

        func prepare(input: UserInput) throws -> LMInput {
            let messages = DefaultMessageGenerator().generate(from: input)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(tokens))
        }
    }

    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var value = state
            value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
            value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
            return value ^ (value >> 31)
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

    private struct LedgerTokenizer: Tokenizer {
        let unsafeDynamicTemplate: Bool
        var staticOnlyAssistantHeader = false
        var eosTokenId: Int? = nil

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            []
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            if let eosToken, token == eosToken {
                return eosTokenId
            }
            return nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            String(id)
        }

        var bosToken: String? { nil }
        var eosToken: String? { eosTokenId == nil ? nil : "<eos>" }
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
                let role = message["role"] as? String
                let content = message["content"] as? String
                if role == Chat.Message.Role.assistant.rawValue,
                    content?.isEmpty == true,
                    let eosTokenId
                {
                    return [eosTokenId]
                }

                switch content {
                case "first":
                    return [101]
                case "second":
                    return [102]
                case "tool result":
                    return [103]
                case "First continuation":
                    return [104]
                case "Second continuation":
                    return [105]
                case "0":
                    return [0]
                case "7":
                    return [7]
                case "7 7":
                    return [7, 7]
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

    private struct ToolCallLedgerTokenizer: Tokenizer {
        static let prose = "I'll inspect the file.\n"
        static let toolCall =
            "<tool_call>{\"id\":\"call_lookup\",\"name\":\"lookup\",\"arguments\":{}}</tool_call>"

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            switch text {
            case Self.prose:
                return [201]
            case Self.toolCall:
                return [202]
            case Self.prose + Self.toolCall:
                return [201, 202]
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

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map {
                skipSpecialTokens && $0 == 202 ? "" : convertIdToToken($0) ?? ""
            }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            switch id {
            case 201:
                return Self.prose
            case 202:
                return Self.toolCall
            default:
                return nil
            }
        }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens = tools == nil ? [] : [11, 12]
            for message in messages {
                switch message["role"] as? String {
                case Chat.Message.Role.user.rawValue:
                    tokens += encode(
                        text: message["content"] as? String ?? "",
                        addSpecialTokens: false)
                case Chat.Message.Role.assistant.rawValue:
                    let content = message["content"] as? String ?? ""
                    if content == Self.prose + Self.toolCall {
                        tokens += [201]
                    } else {
                        tokens += encode(text: content, addSpecialTokens: false)
                    }
                    if let toolCalls = message["tool_calls"] as? [[String: any Sendable]] {
                        tokens += toolCalls.map { toolCall in
                            let function = toolCall["function"] as? [String: any Sendable]
                            if toolCall["id"] as? String == "call_lookup",
                                function?["name"] as? String == "lookup"
                            {
                                return 202
                            }
                            return 999
                        }
                    }
                case Chat.Message.Role.tool.rawValue:
                    guard message["tool_call_id"] as? String == "call_lookup" else {
                        tokens += [999]
                        continue
                    }
                    tokens += encode(
                        text: message["content"] as? String ?? "",
                        addSpecialTokens: false)
                default:
                    break
                }
            }
            return tokens
        }
    }

    private struct UserHeaderPrefixTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            switch text {
            case "first": [101]
            case "second": [102]
            case "0": [0]
            case "7": [7]
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

    private struct LiteralQwenStyleTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            switch text {
            case "first": [101]
            case "second": [102]
            case "0": [0]
            case "7": [7]
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
            var tokens: [Int] = []
            if messages.contains(where: { $0["role"] as? String == "system" }) || tools != nil {
                tokens += [151_644, 9_125]
                if tools != nil {
                    tokens += [27_091, 25_791]
                }
                tokens += [151_645]
            }
            for message in messages {
                switch message["role"] as? String {
                case "user":
                    tokens += [151_644, 872]
                    tokens += encode(
                        text: message["content"] as? String ?? "",
                        addSpecialTokens: false)
                    tokens += [151_645]
                case "assistant":
                    tokens += [151_644, 77_091]
                    tokens += encode(
                        text: message["content"] as? String ?? "",
                        addSpecialTokens: false)
                default:
                    break
                }
            }
            if addGenerationPrompt {
                tokens += [151_644, 77091]
            }
            return tokens
        }
    }

    private struct LedgerInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration: ModelConfiguration
        let messageGenerator: MessageGenerator

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)

            return LMInput(tokens: MLXArray(promptTokens))
        }
    }

    private struct MultimodalLedgerInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration: ModelConfiguration
        let messageGenerator: MessageGenerator
        let recorder: PreparedTokenRecorder

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            recorder.append(promptTokens, shape: [promptTokens.count])

            let image =
                if input.images.isEmpty {
                    nil as LMInput.ProcessedImage?
                } else {
                    LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 1, 3]))
                }

            return LMInput(
                text: .init(tokens: MLXArray(promptTokens)),
                image: image)
        }
    }

    private struct TwoDimensionalInputProcessor: UserInputProcessor {
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
    }

    private struct FullHistoryRejectingProcessor: UserInputProcessor {
        struct FullHistoryError: Error {}

        let configuration = ModelConfiguration(id: "test")
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let recorder: PreparedTokenRecorder
        var throwsCancellationOnFullHistory = false

        func prepare(input: UserInput) throws -> LMInput {
            let messages = DefaultMessageGenerator().generate(from: input)
            if messages.count > 2 {
                if throwsCancellationOnFullHistory {
                    throw CancellationError()
                }
                throw FullHistoryError()
            }

            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext)
            recorder.append(promptTokens, shape: [promptTokens.count])
            return LMInput(tokens: MLXArray(promptTokens))
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
        kvHeadCount: Int = 1,
        kvHeadDim: Int = 1,
        nextToken: Int = 0
    ) -> ModelContext {
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        return recordingModel(
            processor: processor,
            tokenizer: tokenizer,
            recorder: recorder,
            kvHeadCount: kvHeadCount,
            kvHeadDim: kvHeadDim,
            nextToken: nextToken)
    }

    private func recordingModel(
        processor: any UserInputProcessor,
        tokenizer: Tokenizer,
        recorder: PreparedTokenRecorder,
        kvHeadCount: Int = 1,
        kvHeadDim: Int = 1,
        nextToken: Int = 0
    ) -> ModelContext {
        return .init(
            configuration: ModelConfiguration(id: "test"),
            model: RecordingLanguageModel(
                recorder: recorder,
                kvHeadCount: kvHeadCount,
                kvHeadDim: kvHeadDim,
                nextToken: nextToken),
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

    private func ledgerSession(
        recorder: PreparedTokenRecorder,
        unsafeDynamicTemplate: Bool = false,
        staticOnlyAssistantHeader: Bool = false,
        maxKVSize: Int? = nil,
        repetitionPenalty: Float? = nil
    ) -> ChatSession {
        let tokenizer = LedgerTokenizer(
            unsafeDynamicTemplate: unsafeDynamicTemplate,
            staticOnlyAssistantHeader: staticOnlyAssistantHeader)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        return ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(
                maxTokens: 1,
                maxKVSize: maxKVSize,
                repetitionPenalty: repetitionPenalty),
            tools: lookupTools)
    }

    private func goldenFullRenderLogits(
        conversation: [Chat.Message],
        instructions: String,
        tools: [ToolSpec]
    ) throws -> [Float] {
        let processor = GoldenInputProcessor()
        let model = GoldenLogitLanguageModel()
        let input = try processor.prepare(
            input: UserInput(
                chat: [.system(instructions)] + conversation,
                tools: tools))
        let result = try model.prepare(
            input,
            cache: model.newCache(parameters: nil),
            windowSize: nil)
        guard case .logits(let output) = result else {
            XCTFail("Expected logits from golden model")
            return []
        }
        return output.logits[0, -1, 0...].asArray(Float.self)
    }

    private func randomGoldenMessage(using rng: inout SeededGenerator) -> Chat.Message {
        let corpus = [
            "",
            " ",
            "\n\t  ",
            "hello",
            "こんにちは",
            "emoji 👩‍💻",
            "<eos>",
            "<|endoftext|>",
            "<s>[INST] not a real special token [/INST]",
            "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
            "{\"tool_call_id\":\"abc\",\"output\":\"ok\"}",
            " leading and trailing whitespace ",
            "line\nbreak",
        ]
        let roles: [Chat.Message.Role] = [.user, .user, .tool, .assistant]
        let role = roles[Int.random(in: 0 ..< roles.count, using: &rng)]
        let first = corpus[Int.random(in: 0 ..< corpus.count, using: &rng)]
        let second = corpus[Int.random(in: 0 ..< corpus.count, using: &rng)]
        let content = Bool.random(using: &rng) ? first : first + second
        return Chat.Message(role: role, content: content)
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
        let session = ledgerSession(recorder: recorder)

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

    func testGenericProcessorReusesExactTokenLedger() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ChatSession(
            recordingModel(
                tokenizer: LedgerTokenizer(unsafeDynamicTemplate: false),
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
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

    func testPlainTextSessionUsesExactTranscriptLedger() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0))

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [101],
                [102],
            ])
    }

    func testMultimodalTurnSkipsLedgerValidationBeforePrepare() async throws {
        let modelRecorder = PreparedTokenRecorder()
        let processorRecorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = MultimodalLedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator(),
            recorder: processorRecorder)
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: modelRecorder,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(
            to: "second",
            images: [.array(MLXArray.zeros([1, 1, 3]))],
            videos: [],
            audios: [])

        XCTAssertEqual(
            processorRecorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
        XCTAssertEqual(
            modelRecorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testGoldenLogitsMatchFullRenderAcrossRandomCachedTranscripts() async throws {
        let instructions = "System prompt with Unicode 🧪 and <eos> marker."
        var rng = SeededGenerator(state: 0xC0FFEE)

        for transcriptIndex in 0 ..< 40 {
            let processor = GoldenInputProcessor()
            let recorder = GoldenLogitRecorder()
            let sessionModel = GoldenLogitLanguageModel(recorder: recorder)
            let context = ModelContext(
                configuration: ModelConfiguration(id: "golden"),
                model: sessionModel,
                processor: processor,
                tokenizer: processor.tokenizer)
            let session = ChatSession(
                context,
                instructions: instructions,
                generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
                tools: lookupTools)
            var conversation: [Chat.Message] = []
            let turnCount = Int.random(in: 3 ... 7, using: &rng)

            for turnIndex in 0 ..< turnCount {
                let message = randomGoldenMessage(using: &rng)
                conversation.append(message)

                let expected = try goldenFullRenderLogits(
                    conversation: conversation,
                    instructions: instructions,
                    tools: lookupTools)
                let previousPrepareCount = recorder.snapshot.count
                _ = try await session.respond(to: [message])

                let logits = recorder.snapshot
                XCTAssertEqual(
                    logits.count,
                    previousPrepareCount + 1,
                    "transcript \(transcriptIndex), turn \(turnIndex)")
                let actual = try XCTUnwrap(
                    logits.last,
                    "transcript \(transcriptIndex), turn \(turnIndex)")
                XCTAssertEqual(
                    actual,
                    expected,
                    "transcript \(transcriptIndex), turn \(turnIndex), message \(message)")

                let generatedToken = actual.indices.max { actual[$0] < actual[$1] }!
                let assistantText = processor.tokenizer.decode(
                    tokenIds: [generatedToken],
                    skipSpecialTokens: false)
                conversation.append(.assistant(assistantText))
            }
        }
    }

    func testMultiTokenAssistantTranscriptUsesExactLedger() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)
        session.generateParameters.maxTokens = 2

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
            ])
    }

    func testExactLedgerFallsBackWhenGenerationPromptIsNotTranscriptPrefix() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(
            unsafeDynamicTemplate: false,
            staticOnlyAssistantHeader: true)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101, 200],
                [11, 12, 102, 200],
            ])
    }

    func testExactLedgerRejectsPrefixEndingWithUserTurnHeader() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = UserHeaderPrefixTokenizer()
        let processor = LedgerInputProcessor(
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

    func testExactLedgerDoesNotInferPrefixForEmptyUserTurnHeader() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = UserHeaderPrefixTokenizer()
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 50, 200],
                [11, 12, 50, 102, 200],
            ])
    }

    func testLiteralQwenStyleToolPrefixCanReusePromptTools() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LiteralQwenStyleTokenizer()
        let processor = LedgerInputProcessor(
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
                [
                    151_644, 9_125, 27_091, 25_791, 151_645, 151_644, 872, 101,
                    151_645, 151_644, 77_091,
                ],
                [151_644, 872, 102, 151_645, 151_644, 77_091],
            ])
    }

    func testExactLedgerReuseDisabledForBoundedKVCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder, maxKVSize: 2)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testExactLedgerReuseDisabledForDynamicKVQuantization() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                kvHeadDim: 32,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(
                maxTokens: 2,
                kvBits: 4,
                quantizedKVStart: 0),
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

    func testExactLedgerReuseDisabledForRepetitionPenalty() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder, repetitionPenalty: 1.1)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testExactLedgerReuseDisabledForPresencePenalty() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)
        session.generateParameters.presencePenalty = 0.5

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testExactLedgerReuseDisabledForFrequencyPenalty() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)
        session.generateParameters.frequencyPenalty = 0.5

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testFullHistoryRenderFailureFallsBackToIncrementalPrompt() async throws {
        let recorder = PreparedTokenRecorder()
        let processor = FullHistoryRejectingProcessor(recorder: recorder)
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: PreparedTokenRecorder(),
                nextToken: 7),
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

    func testFullHistoryRenderCancellationPropagates() async throws {
        let recorder = PreparedTokenRecorder()
        let processor = FullHistoryRejectingProcessor(
            recorder: recorder,
            throwsCancellationOnFullHistory: true)
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: PreparedTokenRecorder(),
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected CancellationError")
        } catch is CancellationError {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
    }

    func testStreamCancellationInvalidatesExactLedger() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 20),
            tools: lookupTools)

        for try await _ in session.streamResponse(to: "first") {
            break
        }
        _ = try await session.respond(to: "second")
        await session.synchronize()

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [11, 12, 102],
            ])
    }

    func testHistorySessionReusesExactLedgerAfterInitialHydration() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            history: [
                .system("Historical system prompt"),
                .user("Old question"),
                .assistant("Old answer"),
            ],
            generateParameters: GenerateParameters(maxTokens: 1))

        _ = try await session.respond(to: "First continuation")
        _ = try await session.respond(to: "Second continuation")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 104],
                [105],
            ])
    }

    func testEOSTerminatedGenerationKeepsLedgerAlignedWithCache() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false, eosTokenId: 7)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            instructions: "You are concise.",
            generateParameters: GenerateParameters(maxTokens: 5),
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

    func testHistorySessionInstructionChangeThrowsAfterHydration() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "Initial prefix.",
            history: [.user("first"), .assistant("7")],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "second")
        session.instructions = "Changed prefix."

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101, 7, 102]])
        }
    }

    func testHistorySessionPromptToolChangeThrowsAfterHydration() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "Initial prefix.",
            history: [.user("first"), .assistant("7")],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "second")
        session.tools = [
            [
                "type": "function",
                "function": [
                    "name": "changed_lookup",
                    "description": "Changed prompt-rendered tool",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101, 7, 102]])
        }
    }

    func testHistorySessionAdditionalContextChangeThrowsAfterHydration() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "Initial prefix.",
            history: [.user("first"), .assistant("7")],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "second")
        session.additionalContext = ["chat_template_kwargs": ["enable_thinking": false]]

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101, 7, 102]])
        }
    }

    func testToolCallLedgerUsesRawGeneratedTextNotDisplayChunks() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = ToolCallLedgerTokenizer()
        let configuration = ModelConfiguration(id: "test", toolCallFormat: .json)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: configuration,
            model: SequenceLanguageModel(recorder: recorder, tokens: [201, 202]),
            processor: processor,
            tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 2),
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

    func testParsedToolCallAndToolResultContinuationUsesExactLedger() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = ToolCallLedgerTokenizer()
        let configuration = ModelConfiguration(id: "test", toolCallFormat: .json)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: configuration,
            model: SequenceLanguageModel(
                recorder: recorder,
                tokens: [201, 202, 201, 201, 201]),
            processor: processor,
            tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 2),
            tools: lookupTools
        ) { toolCall in
            XCTAssertEqual(toolCall.function.name, "lookup")
            return "tool result"
        }

        _ = try await session.respond(to: "first")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [103],
            ])
    }

    func testHistoryLedgerPreservesStructuredToolMetadata() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = ToolCallLedgerTokenizer()
        let configuration = ModelConfiguration(id: "test", toolCallFormat: .json)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let toolCall = ToolCall(
            function: .init(name: "lookup", arguments: [:]),
            id: "call_lookup")
        let context = ModelContext(
            configuration: configuration,
            model: SequenceLanguageModel(recorder: recorder, tokens: [201, 201]),
            processor: processor,
            tokenizer: tokenizer)
        let session = ChatSession(
            context,
            history: [
                .user("first"),
                .assistant(
                    ToolCallLedgerTokenizer.prose + ToolCallLedgerTokenizer.toolCall,
                    toolCalls: [toolCall]),
                .tool("tool result", id: "call_lookup"),
            ],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "second")
        _ = try await session.respond(to: "first")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101, 201, 202, 103, 102],
                [101],
            ])
    }

    func testExactLedgerReuseDisabledForSpeculativeDecoding() async throws {
        let recorder = PreparedTokenRecorder()
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let processor = LedgerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let draft = ModelContainer(
            context: recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: PreparedTokenRecorder()))
        let session = ChatSession(
            recordingModel(
                processor: processor,
                tokenizer: tokenizer,
                recorder: recorder),
            instructions: "You are concise.",
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault)),
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

    func testExactLedgerReuseSkipsTwoDimensionalTokens() async throws {
        let recorder = PreparedTokenRecorder()
        let processor = TwoDimensionalInputProcessor()
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

    func testExactLedgerReuseUsesRenderedTranscriptSuffix() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder, unsafeDynamicTemplate: true)

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
        let session = ledgerSession(recorder: recorder)

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

    func testPromptConfigurationChangeThrowsInsteadOfCorruptingCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        session.instructions = "Changed prefix."

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
    }

    func testPromptToolChangeThrowsInsteadOfCorruptingCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        session.tools = [
            [
                "type": "function",
                "function": [
                    "name": "changed_lookup",
                    "description": "Changed prompt-rendered tool",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
    }

    func testAdditionalContextChangeThrowsInsteadOfCorruptingCache() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        session.additionalContext = ["chat_template_kwargs": ["enable_thinking": false]]

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
    }

    func testPromptConfigurationChangeThrowsAfterCachedPrompt() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(
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

    func testRestoredCachePromptConfigurationChangeThrowsAfterFirstGeneration() async throws {
        let recorder = PreparedTokenRecorder()
        let restoredCache = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 1, 1])
        let values = MLXArray.zeros([1, 1, 1, 1])
        _ = restoredCache.update(keys: keys, values: values)
        let tokenizer = LedgerTokenizer(unsafeDynamicTemplate: false)
        let session = ChatSession(
            recordingModel(
                tokenizer: tokenizer,
                recorder: recorder,
                nextToken: 7),
            instructions: "Initial prefix.",
            cache: [restoredCache],
            generateParameters: GenerateParameters(maxTokens: 1),
            tools: lookupTools)

        _ = try await session.respond(to: "first")
        session.instructions = "Changed prefix."

        do {
            _ = try await session.respond(to: "second")
            XCTFail("Expected promptCacheMismatch")
        } catch ChatSessionError.promptCacheMismatch {
            XCTAssertEqual(recorder.snapshot, [[11, 12, 101]])
        }
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

    func testClearErasesExactLedgerReuse() async throws {
        let recorder = PreparedTokenRecorder()
        let session = ledgerSession(recorder: recorder)

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")
        await session.clear()
        _ = try await session.respond(to: "first")

        XCTAssertEqual(
            recorder.snapshot,
            [
                [11, 12, 101],
                [102],
                [11, 12, 101],
            ])
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
