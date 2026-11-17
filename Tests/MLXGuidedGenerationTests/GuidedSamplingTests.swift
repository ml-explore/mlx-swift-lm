// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Byte-level tokenizer: token id = byte value, 255 = EOS.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != 255 }.map(UInt8.init), encoding: .utf8) ?? ""
    }
    func convertTokenToId(_ token: String) -> Int? {
        if token == "<eos>" { return 255 }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        if id == 255 { return "<eos>" }
        return id >= 0 && id < 255 ? String(UnicodeScalar(UInt8(id))) : nil
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}

/// Prefers EOS, then lower byte values, so greedy decoding picks "A" over "B".
private final class PreferLowBytesModel: Module, LanguageModel {
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let count = input.tokens.size
        var row = (0 ..< 256).map { -Float($0) / 1000 }
        row[255] = 20
        let rows = Array(repeating: row, count: count).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, count, 256))
    }
}

/// Picks the best token other than `avoid`.
private struct AvoidingSampler: LogitSampler {
    let avoid: Int
    func sample(logits: MLXArray) -> MLXArray {
        let positions = MLXArray(Array(Int32(0) ..< Int32(logits.dim(-1))))
        let blocked = positions .== MLXArray(Int32(avoid))
        return argMax(which(blocked, MLXArray(-Float.infinity), logits), axis: -1)
    }
}

private final class History {
    var prompt: [Int] = []
    var sampled: [Int] = []
    var emitted: [Int] = []
    var finalized = 0
}

/// Records its lifecycle and, optionally, shifts one token's logit.
private struct RecordingProcessor: LogitProcessor {
    let history: History
    var token: Int?
    var shift: Float = -1000

    mutating func prompt(_ prompt: MLXArray) {
        history.prompt = prompt.asArray(Int32.self).map(Int.init)
    }
    func process(logits: MLXArray) -> MLXArray {
        guard let token else { return logits }
        let positions = MLXArray(Array(Int32(0) ..< Int32(logits.dim(-1))))
        return logits
            + which(positions .== MLXArray(Int32(token)), MLXArray(shift), MLXArray(Float(0)))
    }
    mutating func didSample(token: MLXArray) { history.sampled.append(token.item(Int.self)) }
    mutating func didEmit(token: Int) { history.emitted.append(token) }
    mutating func finalizeGeneration() { history.finalized += 1 }
}

/// Always returns the same token.
private struct ConstantSampler: LogitSampler {
    let token: Int
    func sample(logits: MLXArray) -> MLXArray { MLXArray([UInt32(token)]) }
}

@Suite(.serialized)
struct GuidedSamplingTests {

    private func generate(
        grammar: String = #"root ::= "{\"key\":\"" ("A" | "B") "\"}""#,
        fastForward: Bool = true,
        processor: (any LogitProcessor)? = nil,
        sampler: (any LogitSampler)? = nil
    ) throws -> String {
        let tokenizer = ByteTokenizer()
        let vocabulary = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let grammarTokenizer = try GrammarTokenizer(
            vocab: vocabulary, vocabType: .byteFallback, eosTokenId: 255)
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, grammar: grammar,
            fastForward: fastForward, hostTokenizer: tokenizer)
        let context = ModelContext(
            configuration: ModelConfiguration(id: "prefer-low-bytes"),
            model: PreferLowBytesModel(), processor: UnusedProcessor(), tokenizer: tokenizer)
        var output = ""
        _ = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), context: context,
            constraint: constraint, maxTokens: 64, vocabSize: 256,
            logitProcessor: processor, logitSampler: sampler
        ) {
            output += $0
            return true
        }
        return output
    }

    @Test func greedyByDefault() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let output = try generate()
            #expect(output == #"{"key":"A"}"#)
        }
    }

    @Test func injectedSamplerDecides() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let output = try generate(sampler: AvoidingSampler(avoid: 65))
            #expect(output == #"{"key":"B"}"#)
        }
    }

    @Test func injectedProcessorDecides() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let processor = RecordingProcessor(history: History(), token: 65)
            let output = try generate(processor: processor)
            #expect(output == #"{"key":"B"}"#)
        }
    }

    /// The grammar mask is applied after the processor, so a processor cannot
    /// make an excluded token selectable.
    @Test func processorCannotUndoTheGrammarMask() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let processor = RecordingProcessor(history: History(), token: 90, shift: 1000)
            let output = try generate(processor: processor)
            #expect(output == #"{"key":"A"}"#)
        }
    }

    /// The processor sees the prompt, every output token through `didSample`
    /// and `didEmit`, including fast-forward tokens, and one finalization.
    @Test(arguments: [false, true])
    func processorSeesEveryOutputToken(fastForward: Bool) throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let history = History()
            let output = try generate(
                fastForward: fastForward, processor: RecordingProcessor(history: history))
            #expect(history.prompt == [1, 2, 3])
            #expect(history.emitted == output.utf8.map(Int.init))
            #expect(history.sampled.filter { $0 != 255 } == output.utf8.map(Int.init))
            #expect(history.finalized == 1)
        }
    }

    /// As in `TokenIterator`, a sampled stop token reaches `didSample` but not
    /// `didEmit`.
    @Test func stopTokenIsSampledButNotEmitted() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let history = History()
            let output = try generate(
                grammar: #"root ::= "A" "B"?"#, processor: RecordingProcessor(history: history))
            #expect(output == "A")
            #expect(history.sampled == [65, 255])
            #expect(history.emitted == [65])
            #expect(history.finalized == 1)
        }
    }

    /// A sampler that ignores the mask is an error, not a successful stop.
    @Test func samplerChoosingAnExcludedTokenThrows() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            #expect(throws: GrammarError.self) {
                _ = try generate(sampler: ConstantSampler(token: 255))
            }
        }
    }
}
