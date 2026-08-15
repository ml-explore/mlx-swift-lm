// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite(.serialized)
struct ChatSessionMTPEarlyTerminationTests {
    @Test("early EOS finalizes MTP lookahead before the next turn")
    func earlyEOSFinalizesLookaheadBeforeNextTurn() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))

        #expect(first.text == "x")
        #expect(first.info.stopReason == .stop)
        #expect(first.info.generationTokenCount == 1)
        #expect(first.info.proposedDraftTokens == 2)
        #expect(first.info.acceptedDraftTokens == 2)

        await session.synchronize()

        let stoppedStatus = try await session.cacheStatus()
        #expect(stoppedStatus.phase == .realized)
        // Four prompt tokens plus the emitted bonus and terminating EOS are
        // represented. The other accepted EOS draft was lookahead and must
        // have been removed by generation finalization.
        #expect(stoppedStatus.processedTokenCount == 6)

        let second = try await collect(session.streamDetails(to: "b"))

        #expect(second.text == "xxxx")
        // user("a"), assistant("x"), user("b") each render to three
        // tokens; the generation marker contributes the tenth token.
        #expect(second.info.promptTokenCount == 10)
        #expect(second.info.stopReason == .length)
        #expect(second.info.proposedDraftTokens == 2)
        #expect(second.info.acceptedDraftTokens == 2)

        let continuedStatus = try await session.cacheStatus()
        #expect(continuedStatus.phase == .realized)
        // The ten-token transcript plus three generated tokens represented
        // in K/V; the final sampled token remains the next decode input.
        #expect(continuedStatus.processedTokenCount == 13)
    }

    private func makeModelContext() -> ModelContext {
        let tokenizer = EarlyTerminationMTPTokenizer()
        let processor = EarlyTerminationMTPInputProcessor(tokenizer: tokenizer)
        return ModelContext(
            configuration: processor.configuration,
            model: EarlyTerminationMTPTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
    }

    private func makeMTPConfiguration() throws -> SpeculativeDecodingConfig {
        let container = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(id: "mtp-early-termination-test"),
                model: EarlyTerminationMTPDrafter()
            )
        )
        return try SpeculativeDecodingConfig(mtpDrafter: container, blockSize: 4)
    }

    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let completion = event.info {
                info = completion
            }
        }
        return (text, try #require(info))
    }
}

private struct EarlyTerminationMTPTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        Array(repeating: 7, count: text.unicodeScalars.count)
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(repeating: "x", count: tokenIds.filter { $0 != 15 }.count)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            switch message["role"] as? String {
            case "user":
                tokens.append(1)
            case "assistant":
                tokens.append(2)
            case "system":
                tokens.append(3)
            case "tool":
                tokens.append(5)
            default:
                tokens.append(6)
            }
            let content = message["content"] as? String ?? ""
            tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
            tokens.append(8)
        }
        tokens.append(9)
        return tokens
    }
}

private struct EarlyTerminationMTPInputProcessor: UserInputProcessor {
    let tokenizer: EarlyTerminationMTPTokenizer
    let configuration = ModelConfiguration(id: "mtp-early-termination-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}

/// Attention-only target whose logits accept the next token already present
/// in a verification block. The last position always predicts ordinary token
/// 4, which is the prepare-time and all-accepted bonus.
private final class EarlyTerminationMTPTarget: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }

    func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        output(for: inputs, cache: cache, emitMTPState: false).logits
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        output(for: input.tokens, cache: cache, emitMTPState: state?[mtpEmitFlagKey] == true)
    }

    private func output(
        for tokens: MLXArray,
        cache: [KVCache]?,
        emitMTPState: Bool
    ) -> LMOutput {
        let positions = tokens.dim(-1)
        let newKV = MLXArray.zeros([1, 1, positions, 1])
        let sharedKV = cache?.first?.update(keys: newKV, values: newKV) ?? (newKV, newKV)

        let tokenValues = tokens.asArray(Int.self)
        let vocabularySize = 16
        var values = [Float](repeating: -100, count: positions * vocabularySize)
        for position in 0 ..< positions {
            let nextToken = position + 1 < tokenValues.count ? tokenValues[position + 1] : 4
            values[position * vocabularySize + nextToken] = 100
        }
        let logits = MLXArray(values, [1, positions, vocabularySize])

        guard emitMTPState else {
            return LMOutput(logits: logits)
        }

        var outputState = LMOutput.State()
        outputState[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        outputState[mtpSharedKVStatesKey] = ["full_attention": sharedKV]
        outputState[mtpSharedKVSourceIndicesKey] = ["full_attention": 0]
        outputState[mtpSharedKVOffsetsKey] = [
            "full_attention": cache?.first?.offset ?? positions
        ]
        return LMOutput(logits: logits, state: outputState)
    }
}

/// The first four-token prompt proposes EOS for every available draft position;
/// later full-transcript prompts propose ordinary token 4. Query offset makes
/// this behavior per-stream and deterministic without mutable model state.
private final class EarlyTerminationMTPDrafter: Module, StatefulMTPDrafterModel {
    var maximumBlockSize: Int? { 4 }
    var requiresSharedTargetKV: Bool { false }
    var requiresPromptPrefill: Bool { true }
    var requiresGreedySampling: Bool { true }

    func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [])
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(
            batchSize: lastToken.dim(0), queryOffset: queryOffset, blockSize: blockSize)
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(
            batchSize: lastToken.dim(0), queryOffset: queryOffset, blockSize: blockSize)
    }

    private func draftedTokens(
        batchSize: Int,
        queryOffset: Int,
        blockSize: Int
    ) -> MLXArray {
        let token = queryOffset == 4 ? Int32(15) : Int32(4)
        return MLXArray(
            Array(repeating: token, count: batchSize * (blockSize - 1)),
            [batchSize, blockSize - 1]
        )
    }
}
