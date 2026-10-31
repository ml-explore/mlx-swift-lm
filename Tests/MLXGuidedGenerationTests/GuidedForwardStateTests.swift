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

/// Records every token it is fed. Its logits prefer EOS, so the grammar
/// decides the output.
private final class RecordingModel: Module, LanguageModel {
    var forwarded: [Int] = []

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let ids = input.tokens.flattened().asArray(Int32.self).map(Int.init)
        forwarded += ids
        var row = (0 ..< 256).map { -Float($0) / 1000 }
        row[255] = 20
        let rows = Array(repeating: row, count: ids.count).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, ids.count, 256))
    }
}

@Suite(.serialized)
struct GuidedForwardStateTests {

    /// The model must see exactly the prompt followed by every emitted token,
    /// including a sampled token that the grammar follows with forced tokens.
    @Test(arguments: [false, true])
    func modelSeesEveryEmittedToken(fastForward: Bool) throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let tokenizer = ByteTokenizer()
            let vocabulary = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary, vocabType: .byteFallback, eosTokenId: 255)
            let grammar = #"root ::= "{\"key\":\"" ("A" | "B") "\"}""#
            let constraint = try GrammarConstraint(
                tokenizer: grammarTokenizer, grammar: grammar, fastForward: fastForward,
                hostTokenizer: tokenizer)
            let model = RecordingModel()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "recording"), model: model,
                processor: UnusedProcessor(), tokenizer: tokenizer)
            let prompt: [Int32] = [1, 2, 3]
            let sink = GuidedGenerationDiagnosticSink()
            var output = ""
            _ = try GuidedGenerationDiagnosticSink.$current.withValue(sink) {
                try GuidedGenerationLoop.run(
                    input: LMInput(tokens: MLXArray(prompt)), context: context,
                    constraint: constraint, maxTokens: 64, vocabSize: 256
                ) {
                    output += $0
                    return true
                }
            }

            #expect(output == #"{"key":"A"}"#)
            #expect(sink.grammarTerminated)
            #expect(sink.fastForwardTokenIDs.isEmpty != fastForward)
            #expect(model.forwarded == prompt.map(Int.init) + output.utf8.map(Int.init))
        }
    }
}
