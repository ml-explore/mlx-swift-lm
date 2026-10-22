// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// A model that emits scripted tokens by boosting them in a wrapped model's logits.
///
/// The wrapped model (usually small with random weights) does the real work: cache
/// updates, prefill chunking, and state. This model only raises the scripted token above
/// each logits row's maximum, so samplers, processors, and masks still see a realistic
/// distribution.
///
/// It appends a ``TranscriptKVCache`` to the wrapped model's caches, and finds the turn
/// and position from it (see ``TurnScripts``). Generation trims and copies that entry with
/// the rest of the cache, so the cursor stays right across speculative rejection, prompt
/// cache reuse, and `ChatSession` rebuilds.
///
/// Inside a turn where the script has no token, it blocks
/// ``TurnScripts/blockedUnscriptedTokens``, so unscripted output cannot end the turn or
/// start the next one.
///
/// Reading the input tokens forces evaluation on every call. That is fine for tests but
/// removes the iterator's overlap of compute and sampling.
public final class ScriptedLogitsModel: ForwardingLanguageModel, @unchecked Sendable {

    /// How far above a row's maximum the scripted token goes.
    public enum Margin: Sendable, Equatable {
        /// Large enough that samplers and penalties do not override the script.
        case dominant
        /// Small: the script wins under greedy decoding, but masks and penalties can beat it.
        case plausible
        case custom(Float)

        public var value: Float {
            switch self {
            case .dominant: 1000
            case .plausible: 1
            case .custom(let value): value
            }
        }
    }

    public let scripts: TurnScripts
    public let margin: Margin
    public let callLog = ForwardCallLog()

    /// - Parameters:
    ///   - wrapped: the model that computes the background logits. Its vocabulary must be
    ///     at least as large as the largest scripted token id.
    ///   - scripts: the tokens to emit on each assistant turn.
    ///   - margin: how strongly to boost the scripted token.
    public init(
        wrapping wrapped: any LanguageModel, scripts: TurnScripts, margin: Margin = .dominant
    ) {
        self.scripts = scripts
        self.margin = margin
        super.init(wrapping: wrapped)
    }

    /// The tokens recorded in `cache`, or `nil` if it has no ``TranscriptKVCache``.
    public static func transcript(in cache: [KVCache]) -> [Int]? {
        (cache.last as? TranscriptKVCache)?.tokens
    }

    public override func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try wrapped.newCache(parameters: parameters) + [TranscriptKVCache()]
    }

    /// The status of this model's own cache, which includes the transcript entry.
    public override func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        KVCacheStatus(
            cache: try newCache(parameters: parameters),
            plan: try parameters?.kvCachePlan() ?? .disabled,
            phase: .planned)
    }

    public override func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let (inner, transcript) = split(cache)
        let tokens = Self.tokens(input.text.tokens)
        let before = transcript?.tokens ?? []

        let result = try wrapped.prepare(input, cache: inner, state: state, prefill: prefill)

        switch result {
        case .tokens(let remainder):
            let processed = tokens.count - remainder.cacheSequenceLength
            transcript?.append(Array(tokens.prefix(processed)))
            record(
                .prepare, before: before, input: tokens, processed: processed,
                cursors: [], mask: input.text.mask, state: state)
            return result

        case .logits(let output):
            transcript?.append(tokens)
            let (logits, cursors) = boost(output.logits, history: before + tokens)
            record(
                .prepare, before: before, input: tokens, processed: tokens.count,
                cursors: cursors, mask: input.text.mask, state: state)
            return .logits(LMOutput(logits: logits, state: output.state))
        }
    }

    public override func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(.forward, input, cache: cache, state: state) { inner in
            wrapped.callAsFunction(input, cache: inner, state: state)
        }
    }

    public override func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(.nextTokenLogits, input, cache: cache, state: state) { inner in
            wrapped.nextTokenLogits(input, cache: inner, state: state)
        }
    }

    public override func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    // MARK: - Implementation

    private func forward(
        _ kind: ForwardCall.Kind, _ input: LMInput.Text, cache: [KVCache]?,
        state: LMOutput.State?, body: ([KVCache]?) -> LMOutput
    ) -> LMOutput {
        let (inner, transcript) = split(cache ?? [])
        let tokens = Self.tokens(input.tokens)
        let before = transcript?.tokens ?? []

        let output = body(cache == nil ? nil : inner)
        transcript?.append(tokens)

        let (logits, cursors) = boost(output.logits, history: before + tokens)
        record(
            kind, before: before, input: tokens, processed: tokens.count,
            cursors: cursors, mask: input.mask, state: state)
        return LMOutput(logits: logits, state: output.state)
    }

    /// Separates the transcript entry from the wrapped model's caches.
    private func split(_ cache: [KVCache]) -> ([KVCache], TranscriptKVCache?) {
        if let transcript = cache.last as? TranscriptKVCache {
            return (Array(cache.dropLast()), transcript)
        }
        precondition(
            cache.isEmpty,
            "ScriptedLogitsModel needs the cache from its own newCache(parameters:), which ends with a TranscriptKVCache"
        )
        return (cache, nil)
    }

    private static func tokens(_ array: MLXArray) -> [Int] {
        precondition(
            array.ndim <= 1 || array.dim(0) == 1, "ScriptedLogitsModel supports batch size 1")
        return array.asType(.int32).asArray(Int32.self).map(Int.init)
    }

    /// Boosts the scripted token in each row, and blocks EOS and the turn marker in rows
    /// inside a turn with no scripted token. Rows are the last positions of `history`.
    private func boost(_ logits: MLXArray, history: [Int]) -> (MLXArray, [TurnScripts.Cursor?]) {
        precondition(logits.dim(0) == 1, "ScriptedLogitsModel supports batch size 1")
        let rows = logits.dim(1)
        let vocabularySize = logits.dim(-1)
        let cursors = Array(scripts.cursors(for: history).suffix(rows))
        let targets = cursors.map { $0.flatMap(scripts.token(at:)) }
        let unscripted = zip(cursors, targets).map { $0 != nil && $1 == nil }

        guard targets.contains(where: { $0 != nil }) || unscripted.contains(true) else {
            return (logits, cursors)
        }
        for target in targets.compactMap({ $0 }) {
            precondition(
                target < vocabularySize,
                "scripted token \(target) is outside the wrapped model's vocabulary (\(vocabularySize))"
            )
        }

        let ids = MLXArray(Array(Int32(0) ..< Int32(vocabularySize))).reshaped(1, 1, vocabularySize)
        let targetArray = MLXArray(targets.map { Int32($0 ?? -1) }).reshaped(1, rows, 1)
        let boosted = logits.max(axis: -1, keepDims: true) + margin.value
        var result = which(ids .== targetArray, boosted, logits)

        if unscripted.contains(true) {
            let blockedTokens = scripts.blockedUnscriptedTokens
            let blocked = MLXArray((0 ..< vocabularySize).map { blockedTokens.contains($0) })
                .reshaped(1, 1, vocabularySize)
            let rowMask = MLXArray(unscripted).reshaped(1, rows, 1)
            result = which(
                blocked .&& rowMask, MLXArray(-Float.infinity).asType(logits.dtype), result)
        }
        return (result, cursors)
    }

    private func record(
        _ kind: ForwardCall.Kind, before: [Int], input: [Int], processed: Int,
        cursors: [TurnScripts.Cursor?], mask: MLXArray?, state: LMOutput.State?
    ) {
        var mismatch: ForwardCall.InputMismatch?
        if let actual = input.first, let previous = before.indices.last,
            let cursor = scripts.cursors(for: before)[previous],
            let expected = scripts.token(at: cursor), expected != actual
        {
            mismatch = .init(position: before.count, expected: expected, actual: actual)
        }

        callLog.append(
            ForwardCall(
                kind: kind,
                cacheOffsetBefore: before.count,
                inputTokens: input,
                processedTokenCount: processed,
                scriptedTokens: cursors.map { $0.flatMap(scripts.token(at:)) },
                cursor: cursors.last ?? nil,
                inputMismatch: mismatch,
                maskPresent: mask != nil,
                hasInboundState: state != nil))
    }
}
