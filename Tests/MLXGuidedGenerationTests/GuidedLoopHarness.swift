// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXGuidedGeneration

/// Byte tokenizer for loop tests: ids 0...254 are bytes, 255 is EOS.
struct HarnessTokenizer: MLXLMCommon.Tokenizer {
    static let eosTokenId = 255

    /// A byte id to report as the unknown token, or nil for none.
    var unknownByte: Int?

    /// Number of token ids, and the default width of a logit row.
    var vocabularySize: Int { 256 }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let bytes = tokenIds.filter { $0 != Self.eosTokenId }.map { UInt8(truncatingIfNeeded: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == "<eos>" { return Self.eosTokenId }
        guard token.unicodeScalars.count == 1, let scalar = token.unicodeScalars.first,
            scalar.value < UInt32(Self.eosTokenId)
        else { return nil }
        return Int(scalar.value)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eosTokenId { return "<eos>" }
        guard id >= 0, id < Self.eosTokenId else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { unknownByte.flatMap { convertIdToToken($0) } }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }

    /// The vocabulary as xgrammar reads it: one `<0xNN>` piece per id.
    var grammarVocabulary: [String] { (0 ..< 256).map { String(format: "<0x%02X>", $0) } }

    func makeGrammarTokenizer(stopTokenIds: [Int] = []) throws -> GrammarTokenizer {
        try GrammarTokenizer(
            vocab: grammarVocabulary, vocabType: .byteFallback,
            eosTokenId: Int32(Self.eosTokenId), stopTokenIds: stopTokenIds)
    }
}

/// Logits as a function of the tokens the model received after the prompt.
///
/// Every row starts at `-id / 1000`, so with no push greedy decoding takes the
/// lowest allowed id.
indirect enum LogitsPolicy {
    /// Pushes the next byte of the script; EOS once the script is done or the
    /// output has left it.
    case script(String)
    /// The first id gets `+magnitude`, the second `+magnitude - 1`, and so on.
    case prefer([UInt8])
    /// `.script(prefix)` while the output is a strict prefix of `prefix`, then
    /// the second policy, which sees the whole output.
    case then(prefix: String, LogitsPolicy)

    func row(emitted: [Int], vocabularySize: Int, magnitude: Float) -> [Float] {
        switch self {
        case .script(let script):
            var row = Self.baseRow(vocabularySize)
            if let next = Self.nextByte(of: script, after: emitted) {
                row[Int(next)] += magnitude
            } else {
                row[HarnessTokenizer.eosTokenId] += magnitude
            }
            return row
        case .prefer(let ids):
            var row = Self.baseRow(vocabularySize)
            for (rank, id) in ids.enumerated() {
                row[Int(id)] += magnitude - Float(rank)
            }
            return row
        case .then(let prefix, let next):
            if Self.nextByte(of: prefix, after: emitted) != nil {
                return LogitsPolicy.script(prefix).row(
                    emitted: emitted, vocabularySize: vocabularySize, magnitude: magnitude)
            }
            return next.row(emitted: emitted, vocabularySize: vocabularySize, magnitude: magnitude)
        }
    }

    private static func baseRow(_ count: Int) -> [Float] {
        (0 ..< count).map { -Float($0) / 1000 }
    }

    /// The byte of `script` after `emitted`, or nil when `emitted` is not a
    /// strict prefix of it.
    private static func nextByte(of script: String, after emitted: [Int]) -> UInt8? {
        let bytes = Array(script.utf8)
        guard emitted.count < bytes.count,
            emitted.elementsEqual(bytes[..<emitted.count].map(Int.init))
        else { return nil }
        return bytes[emitted.count]
    }
}

/// What a `PolicyModel` received.
final class PolicyHistory {
    let promptLength: Int
    var received = 0
    var fed: [Int] = []
    var calls = 0

    init(promptLength: Int) {
        self.promptLength = promptLength
    }
}

/// A model whose last logit row follows a `LogitsPolicy` over the tokens fed
/// after the prompt.
final class PolicyModel: Module, LanguageModel {
    let policy: LogitsPolicy
    let magnitude: Float
    let vocabularySize: Int
    let history: PolicyHistory

    init(policy: LogitsPolicy, magnitude: Float, vocabularySize: Int, history: PolicyHistory) {
        self.policy = policy
        self.magnitude = magnitude
        self.vocabularySize = vocabularySize
        self.history = history
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        history.calls += 1
        let tokens = input.tokens.asArray(Int32.self).map(Int.init)
        for token in tokens {
            if history.received >= history.promptLength {
                history.fed.append(token)
            }
            history.received += 1
        }
        let row = policy.row(
            emitted: history.fed, vocabularySize: vocabularySize, magnitude: magnitude)
        let rows = Array(repeating: row, count: tokens.count).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, tokens.count, vocabularySize))
    }
}

/// The grammar of a harness run.
enum HarnessSource {
    case jsonSchema(String)
    case grammar(String)
}

/// The outcome of a harness run.
struct GuidedRun {
    var text = ""
    /// Output tokens, as `didEmit` saw them.
    var tokenIDs: [Int] = []
    /// What `run` returned, or nil when it threw.
    var returnedCount: Int?
    var error: (any Error)?
    /// Tokens fed to the model after the prompt.
    var fed: [Int] = []
    /// Model calls, the prompt included.
    var modelCalls = 0
    /// Calls to the logit processor's `process`.
    var processCalls = 0
}

private final class EmittedTokens {
    var ids: [Int] = []
    var processCalls = 0
}

/// Records every token the loop emits.
private struct EmitRecorder: LogitProcessor {
    let emitted: EmittedTokens

    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        emitted.processCalls += 1
        return logits
    }
    mutating func didSample(token: MLXArray) {}
    mutating func didEmit(token: Int) { emitted.ids.append(token) }
}

private struct HarnessInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the harness supplies the prepared input")
    }
}

/// Runs `GuidedGenerationLoop.run` with a `PolicyModel` and the prompt `[1, 2, 3]`.
///
/// `logitDim` sets the model's logit width; it defaults to the vocabulary size.
/// `rawMaskHook` changes every mask xgrammar fills, before anything reads it.
/// `grammarTokenizer` builds the grammar's tokenizer from the loop's stops, to
/// give the grammar other stops than the loop.
///
/// By default the grammar gets the same stop tokens as the loop. Setup errors throw; an
/// error from the loop is returned in `GuidedRun.error`.
func runLoop(
    source: HarnessSource,
    tokenizer: HarnessTokenizer = HarnessTokenizer(),
    policy: LogitsPolicy,
    maxTokens: Int,
    magnitude: Float = 20,
    logitDim: Int? = nil,
    eosTokenIds: Set<Int> = [],
    fastForward: Bool = true,
    completionReserve: Int = 64,
    hardReserve: Int = 0,
    closingBias: MLXArray? = nil,
    sampler: (any LogitSampler)? = nil,
    rawMaskHook: (@Sendable (MaskResult) -> MaskResult)? = nil,
    grammarTokenizer makeGrammar: ((_ loopStops: [Int]) throws -> GrammarTokenizer)? = nil
) throws -> GuidedRun {
    let configuration = ModelConfiguration(id: "harness", eosTokenIds: eosTokenIds)
    let stopTokenIDs = GuidedGenerationLoop.stopTokenIDs(
        tokenizer: tokenizer, configuration: configuration)
    let grammarTokenizer =
        try makeGrammar?(stopTokenIDs.sorted())
        ?? tokenizer.makeGrammarTokenizer(stopTokenIds: stopTokenIDs.sorted())
    let constraint: GrammarConstraint
    switch source {
    case .jsonSchema(let schema):
        constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, jsonSchema: schema,
            fastForward: fastForward, hostTokenizer: tokenizer)
    case .grammar(let grammar):
        constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, grammar: grammar,
            fastForward: fastForward, hostTokenizer: tokenizer)
    }

    constraint.rawMaskHookForTesting = rawMaskHook

    let prompt: [Int32] = [1, 2, 3]
    let history = PolicyHistory(promptLength: prompt.count)
    let model = PolicyModel(
        policy: policy, magnitude: magnitude,
        vocabularySize: logitDim ?? tokenizer.vocabularySize, history: history)
    let context = ModelContext(
        configuration: configuration,
        model: model, processor: HarnessInputProcessor(), tokenizer: tokenizer)
    let emitted = EmittedTokens()

    var run = GuidedRun()
    var text = ""
    do {
        run.returnedCount = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray(prompt)), context: context,
            constraint: constraint, maxTokens: maxTokens,
            vocabSize: grammarTokenizer.vocabSize,
            completionReserve: completionReserve, hardReserve: hardReserve,
            closingBias: closingBias,
            logitProcessor: EmitRecorder(emitted: emitted), logitSampler: sampler
        ) { delta in
            text += delta
            return true
        }
    } catch {
        run.error = error
    }
    run.text = text
    run.tokenIDs = emitted.ids
    run.fed = history.fed
    run.modelCalls = history.calls
    run.processCalls = emitted.processCalls
    return run
}
