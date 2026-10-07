// Copyright © 2026 Apple Inc.
//
// Resuming a hybrid model's cache across turns from a checkpoint.
//
// A hybrid's recurrent layers (`MambaCache`) cannot be trimmed, so when a
// follow-up prompt does not extend the cache, `RewindToCommonPrefixRule`
// rebuilds it. A template that renders a past reply differently from how it
// was generated -- Qwen3.5 drops the `<think></think>` it was generated after
// -- makes that every turn. `ChatSession` now checkpoints the end of each
// prompt's last message, which the next prompt does repeat, and puts the
// cache back there instead.
//
// The models are tiny random-weight Qwen3.5s, the VLM and the LLM class, so
// these run without downloads.

import Foundation
import MLX
import MLXLLM
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class PromptCheckpointTests: XCTestCase {

    // MARK: - Where a prompt is checkpointed

    func testTheCheckpointIsTheEndOfTheLastMessage() {
        // system... <end> \n user... <end> \n generation prompt
        let tokens = [20, 21, 2, 3, 22, 23, 2, 3, 1, 8, 3]
        XCTAssertEqual(
            promptCheckpointPosition(in: tokens, stopTokenIds: [2], newline: 3), 8)
        XCTAssertEqual(
            promptCheckpointPosition(in: tokens, stopTokenIds: [2], newline: nil), 7)
    }

    func testAPromptEndingOnAMessageEndIsCheckpointedAtTheOneBefore() {
        // Nothing would be left to read after a checkpoint at the very end.
        let tokens = [20, 2, 3, 21, 2, 3]
        XCTAssertEqual(
            promptCheckpointPosition(in: tokens, stopTokenIds: [2], newline: 3), 3)
    }

    func testAPromptWithNoMessageEndHasNoCheckpoint() {
        XCTAssertNil(promptCheckpointPosition(in: [20, 21, 22], stopTokenIds: [2], newline: 3))
        XCTAssertNil(promptCheckpointPosition(in: [20, 21, 2], stopTokenIds: [2], newline: 3))
    }

    // MARK: - The rule

    private let checkpoint = [20, 21, 2, 3, 22, 2, 3]
    /// What the cache holds after the first turn: the prompt, whose generation prompt the
    /// template will not render again, then the reply.
    private var cached: [Int] { checkpoint + [1, 8, 3, 4, 3, 3, 5, 3, 3, 40, 41] }
    /// The next prompt: the reply rendered without its think block, then a new question.
    private var nextPrompt: [Int] { checkpoint + [1, 8, 3, 40, 41, 2, 3, 23, 2, 3, 1, 8, 3] }

    private func decide(
        prompt: [Int]? = nil, trimmable: Bool = false, checkpointTokens: [Int]?,
        media: Bool = false, mask: Bool = false, speculative: Bool = false
    ) -> PromptCacheReuseDecision {
        PromptCacheReusePolicy().decide(
            turn: PromptCacheTurn(
                promptTokens: prompt ?? nextPrompt, carriesNewMedia: media,
                carriesPreparedMedia: media, carriesAttentionMask: mask,
                usesSpeculativeDecoding: speculative),
            cache: PromptCacheState(
                cachedTokens: cached, processedTokenCount: cached.count,
                mainCacheIsAligned: true, isTrimmable: trimmable,
                checkpointTokens: checkpointTokens))
    }

    func testAnUntrimmableCacheIsPutBackToItsCheckpoint() {
        XCTAssertEqual(
            decide(checkpointTokens: checkpoint),
            .restoreCheckpoint(position: checkpoint.count))
    }

    func testWithoutACheckpointAnUntrimmableCacheIsRebuilt() {
        XCTAssertEqual(decide(checkpointTokens: nil), .rebuild)
    }

    /// Trimming reaches the common prefix, which is never shorter than a checkpoint.
    func testATrimmableCacheIsTrimmedInstead() {
        guard case .trimToCommonPrefix = decide(trimmable: true, checkpointTokens: checkpoint)
        else { return XCTFail("a trimmable cache was not trimmed") }
    }

    func testAPromptThatExtendsTheCacheIsStillAppended() {
        let extending = cached + [2, 3, 1, 7, 3, 24]
        guard case .appendSuffix = decide(prompt: extending, checkpointTokens: checkpoint)
        else { return XCTFail("an extending prompt was not appended") }
    }

    func testAPromptTheCheckpointDoesNotBeginIsRebuilt() {
        var changed = nextPrompt
        changed[0] = 30
        XCTAssertEqual(decide(prompt: changed, checkpointTokens: checkpoint), .rebuild)
    }

    func testAPromptWithNothingPastTheCheckpointIsRebuilt() {
        XCTAssertEqual(decide(prompt: checkpoint, checkpointTokens: checkpoint), .rebuild)
    }

    func testMediaMasksAndSpeculationAreRebuilt() {
        XCTAssertEqual(decide(checkpointTokens: checkpoint, media: true), .rebuild)
        XCTAssertEqual(decide(checkpointTokens: checkpoint, mask: true), .rebuild)
        XCTAssertEqual(decide(checkpointTokens: checkpoint, speculative: true), .rebuild)
    }

    // MARK: - KVCacheStorage

    private func attention(_ count: Int) -> MLXArray {
        MLXRandom.normal([1, 2, count, 8])
    }

    /// Two positions of attention and a recurrent state, advanced together.
    private func advance(_ storage: KVCacheStorage, by count: Int) {
        for entry in storage.cache {
            if let recurrent = entry as? MambaCache {
                recurrent[0] = MLXRandom.normal([1, 3, 4])
                recurrent[1] = MLXRandom.normal([1, 2, 4, 4])
            } else {
                _ = entry.update(keys: attention(count), values: attention(count))
            }
        }
        storage.commitProcessedTokens(count)
    }

    func testACheckpointPutsEveryEntryAndTheTimelineBack() throws {
        let recurrent = MambaCache()
        let storage = KVCacheStorage([KVCacheSimple(), recurrent], plan: .disabled)
        advance(storage, by: 5)
        let saved = (recurrent[0], recurrent[1])
        let checkpoint = try XCTUnwrap(storage.checkpoint())

        advance(storage, by: 7)
        XCTAssertEqual(storage.processedTokenCount, 12)
        XCTAssertTrue(storage.restore(checkpoint))

        XCTAssertEqual(storage.processedTokenCount, 5)
        XCTAssertEqual(storage.cache[0].offset, 5)
        XCTAssertTrue(recurrent[0] === saved.0 && recurrent[1] === saved.1)
        XCTAssertTrue(storage.nativeAttentionOffsetsAreAligned)
    }

    func testARestoreRefusesAReplacedEntryAndTouchesNothing() throws {
        let storage = KVCacheStorage([KVCacheSimple(), MambaCache()], plan: .disabled)
        advance(storage, by: 5)
        let checkpoint = try XCTUnwrap(storage.checkpoint())
        advance(storage, by: 7)

        storage.replace(with: [storage.cache[0], MambaCache()])
        XCTAssertFalse(storage.restore(checkpoint))
        XCTAssertEqual(storage.processedTokenCount, 12)
        XCTAssertEqual(storage.cache[0].offset, 12)
    }

    func testACacheListCannotBeCheckpointed() {
        let storage = KVCacheStorage(
            [CacheList(MambaCache(), KVCacheSimple())], plan: .disabled)
        XCTAssertNil(storage.checkpoint())
    }

    // MARK: - The models

    private static func qwen35VLM() throws -> any LanguageModel {
        let json = """
            {
                "model_type": "qwen3_5_vl",
                "image_token_id": 500, "video_token_id": 501,
                "vision_start_token_id": 502, "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                    "vocab_size": 512, "full_attention_interval": 2,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                    "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
                    "rope_parameters": {
                        "type": "default", "mrope_section": [8, 4, 4],
                        "rope_theta": 100000.0, "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 64
                }
            }
            """
        let config = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 1)) { MLXVLM.Qwen35(config) }
    }

    private static func qwen35LLM() throws -> any LanguageModel {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "vocab_size": 512,
                "full_attention_interval": 2
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen35TextModel(config) }
    }

    private static let models: [(name: String, make: @Sendable () throws -> any LanguageModel)] = [
        ("Qwen3.5 VLM", { try qwen35VLM() }),
        ("Qwen3.5 LLM", { try qwen35LLM() }),
    ]

    private static func context(_ model: any LanguageModel) -> ModelContext {
        let tokenizer = ThinkDroppingTokenizer()
        return ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"), model: model,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: ModelConfiguration(id: "tiny-qwen35"),
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
    }

    // MARK: - TokenIterator

    /// Checkpoint partway through a prompt, generate past it, put the cache back and read the
    /// rest again: the second generation is the first one, token for token, because the
    /// checkpoint is exactly the state the first one continued from.
    func testAnIteratorCheckpointResumesExactly() throws {
        let prompt = [20, 21, 22, 2, 3, 23, 24, 25, 26, 2, 3, 1, 8, 3, 4, 3, 3, 5, 3, 3]
        let position = 11
        let parameters = GenerateParameters(maxTokens: 6, temperature: 0)
        for (name, make) in Self.models {
            let model = try make()
            let storage = KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)

            var first = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model, cacheStorage: storage,
                parameters: parameters, checkpointAt: position)
            let checkpoint = try XCTUnwrap(first.promptCheckpoint, name)
            XCTAssertEqual(checkpoint.storage.processedTokenCount, position, name)
            var firstTokens: [Int] = []
            while let token = first.next() { firstTokens.append(token) }

            XCTAssertTrue(storage.restore(checkpoint.storage), name)
            var second = try TokenIterator(
                input: LMInput(tokens: MLXArray(Array(prompt[position...]))), model: model,
                cacheStorage: storage, state: checkpoint.state, parameters: parameters)
            var secondTokens: [Int] = []
            while let token = second.next() { secondTokens.append(token) }

            XCTAssertEqual(firstTokens.count, 6, name)
            XCTAssertEqual(secondTokens, firstTokens, name)
        }
    }

    // MARK: - ChatSession

    private struct Turn {
        let reply: String
        let info: GenerateCompletionInfo
    }

    private func respond(_ session: ChatSession, to question: String) async throws -> Turn {
        var reply = ""
        var info: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: question) {
            if let chunk = item.chunk { reply += chunk }
            if let completion = item.info { info = completion }
        }
        return Turn(reply: reply, info: try XCTUnwrap(info))
    }

    private let instructions = "you answer questions about the notes"
    private let questions = [
        "here are my notes : the red house has four windows and a green door",
        "which house has a green door",
        "how many windows does it have",
    ]

    /// Every turn after the first resumes from the end of the previous question, where
    /// before it rebuilt; and it answers what a session reading the whole conversation
    /// answers.
    func testAChatSessionResumesAHybridFromTheLastQuestion() async throws {
        let parameters = GenerateParameters(maxTokens: 6, temperature: 0)
        for (name, make) in Self.models {
            let context = Self.context(try make())
            let session = ChatSession(
                context, instructions: instructions, generateParameters: parameters)
            var history: [Chat.Message] = []
            var previousPromptLength = 0

            for (index, question) in questions.enumerated() {
                let turn = try await respond(session, to: question)
                let label = "\(name), turn \(index + 1)"

                if index == 0 {
                    XCTAssertEqual(turn.info.cachedPromptTokenCount, 0, label)
                } else {
                    XCTAssertEqual(
                        turn.info.cachedPromptTokenCount,
                        previousPromptLength - ThinkDroppingTokenizer.generationPrompt.count,
                        "\(label) did not resume from the end of the last question")

                    // The same conversation read from nothing.
                    let cold = ChatSession(
                        Self.context(try make()), instructions: instructions,
                        history: history, generateParameters: parameters)
                    let control = try await respond(cold, to: question)
                    XCTAssertEqual(control.info.cachedPromptTokenCount, 0, label)
                    XCTAssertEqual(
                        control.info.totalPromptTokenCount, turn.info.totalPromptTokenCount,
                        label)
                    XCTAssertEqual(turn.reply, control.reply, label)
                }
                XCTAssertFalse(turn.reply.isEmpty, label)
                history += [.user(question), .assistant(turn.reply)]
                previousPromptLength = turn.info.totalPromptTokenCount
            }
        }
    }

    /// A model with no recurrent layers takes no checkpoint and trims as it did.
    func testAnAttentionOnlyModelStillTrims() async throws {
        let json = """
            {
                "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
                "vocab_size": 512
            }
            """
        let model = try await withRandomState(MLXRandom.RandomState(seed: 1)) {
            try await LLMTypeRegistry.shared.createModel(
                configuration: Data(json.utf8), modelType: "qwen3")
        }
        let session = ChatSession(
            Self.context(model), instructions: instructions,
            generateParameters: GenerateParameters(maxTokens: 6, temperature: 0))
        let first = try await respond(session, to: questions[0])
        let second = try await respond(session, to: questions[1])

        // The common prefix runs three tokens into the generation prompt,
        // `<|im_start|>assistant\n`, past where a checkpoint would sit.
        XCTAssertEqual(
            second.info.cachedPromptTokenCount,
            first.info.totalPromptTokenCount - ThinkDroppingTokenizer.generationPrompt.count + 3)
    }
}

/// One token per word, and a ChatML template that renders a past reply the way Qwen3.5's does:
/// without the `<think>\n\n</think>\n\n` it was generated after.
private struct ThinkDroppingTokenizer: Tokenizer {
    static let special = [
        "<|im_start|>": 1, "<|im_end|>": 2, "\n": 3, "<think>": 4, "</think>": 5,
        "system": 6, "user": 7, "assistant": 8,
    ]
    /// Clear of the specials and of the tiny VLM's image and video tokens (500...503).
    static let words = 16 ..< 496
    /// The start of a reply with thinking off.
    static let generationPrompt = [1, 8, 3, 4, 3, 3, 5, 3, 3]

    var bosToken: String? { nil }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }

    static func id(of word: Substring) -> Int {
        if word.first == "w", let id = Int(word.dropFirst()), words.contains(id) { return id }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in word.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3 }
        return words.lowerBound + Int(hash % UInt64(words.count))
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let id = Self.special[text] { return [id] }
        return text.split(separator: " ").map { Self.id(of: $0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { " w\($0)" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { Self.special[token] }
    func convertIdToToken(_ id: Int) -> String? { "w\(id)" }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            tokens += [1, Self.special[role] ?? 7, 3]
            tokens += encode(text: content, addSpecialTokens: false)
            tokens += [2, 3]
        }
        return tokens + Self.generationPrompt
    }
}
