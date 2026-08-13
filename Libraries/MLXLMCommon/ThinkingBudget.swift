// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Errors raised while configuring a reasoning-token budget.
public enum ThinkingBudgetError: Equatable, LocalizedError, Sendable {
    /// A token budget cannot be negative. Zero means close reasoning immediately.
    case invalidMaximumTokenCount(Int)

    /// The answer reserve cannot be negative.
    case invalidMinimumAnswerTokenCount(Int)

    /// The model declares reasoning, but neither it nor the caller supplied a
    /// safe budget transition.
    case unsupportedReasoningProtocol

    /// A protocol boundary could not be represented by the tokenizer.
    case unencodableBoundary(String)

    /// The tokenizer could not represent the model's forced transition.
    case unencodableTransition(String)

    /// The tokenizer produced an ID that cannot be represented by MLX token tensors.
    case invalidTokenID(Int)

    /// `maxTokens` cannot contain the budget, protocol headroom and answer reserve.
    case insufficientGenerationTokenLimit(required: Int, actual: Int)

    /// The configured budget and reserves cannot be represented by `Int`.
    case generationTokenRequirementOverflow

    public var errorDescription: String? {
        switch self {
        case .invalidMaximumTokenCount(let count):
            "Thinking budget maximumTokenCount must be non-negative; received \(count)."
        case .invalidMinimumAnswerTokenCount(let count):
            "Thinking budget minimumAnswerTokenCount must be non-negative; received \(count)."
        case .unsupportedReasoningProtocol:
            """
            This model's reasoning protocol does not declare a safe budget transition. \
            Supply ThinkingBudgetConfiguration.transitionOverride to opt in explicitly.
            """
        case .unencodableBoundary(let boundary):
            "The tokenizer could not encode reasoning boundary \(boundary.debugDescription)."
        case .unencodableTransition(let transition):
            "The tokenizer could not encode thinking-budget transition \(transition.debugDescription)."
        case .invalidTokenID(let tokenID):
            "The tokenizer produced invalid thinking-budget token ID \(tokenID)."
        case .insufficientGenerationTokenLimit(let required, let actual):
            """
            Thinking budget requires maxTokens >= \(required) to leave room for the budget, \
            Unicode/protocol headroom and answer reserve; received \(actual).
            """
        case .generationTokenRequirementOverflow:
            "Thinking budget, protocol headroom and answer reserve exceed the supported token count."
        }
    }
}

/// A model-protocol transition used when a reasoning budget is exhausted.
///
/// The value is model-agnostic: model-family adapters provide the actual text.
/// Models without a validated transition leave
/// ``ReasoningConfig/budgetTransition`` unset, while applications can opt in
/// through ``ThinkingBudgetConfiguration/transitionOverride``.
public struct ReasoningBudgetTransition: Sendable, Equatable {
    /// Text forced immediately before ``ReasoningConfig/endDelimiter``.
    public var beforeEndDelimiter: String

    /// Text forced immediately after ``ReasoningConfig/endDelimiter``.
    public var afterEndDelimiter: String

    public init(
        beforeEndDelimiter: String = "",
        afterEndDelimiter: String = ""
    ) {
        self.beforeEndDelimiter = beforeEndDelimiter
        self.afterEndDelimiter = afterEndDelimiter
    }

    /// Close the reasoning span with no additional model-specific text.
    public static let immediate = ReasoningBudgetTransition()
}

/// Diagnostics emitted when a hard reasoning budget cannot remain authoritative.
public enum ThinkingBudgetDiagnostic: Sendable, Equatable {
    /// Another processor or a tokenizer/model mismatch produced a token outside
    /// the constraint. Enforcement was disabled to avoid corrupting generation.
    case enforcementDisabled(expectedTokenIDs: [Int], sampledTokenID: Int)

    /// The tokenizer still produced incomplete Unicode after the maximum
    /// possible UTF-8 continuation length. Enforcement was disabled rather
    /// than injecting a protocol transition into malformed text.
    case unicodeBoundaryCompletionFailed(sampledTokenCount: Int)
}

/// A hard upper bound for model-selected reasoning tokens.
///
/// Protocol tokens used to enter or leave reasoning are not charged to the
/// budget. If a byte-fallback token ends mid-scalar, generation may consume the
/// minimum additional tokens needed to complete valid Unicode before forcing
/// the transition. Forced transition tokens still count toward
/// ``GenerateParameters/maxTokens``; callers must leave room for the transition
/// and the final answer in their overall generation limit.
public struct ThinkingBudgetConfiguration: Sendable, Equatable {
    /// Maximum number of model-selected tokens allowed inside a reasoning span.
    /// Zero closes a prompt-primed reasoning span before it generates content.
    public let maximumTokenCount: Int

    /// Tokens reserved for the final answer after the worst-case transition.
    ///
    /// Parameter-driven generation rejects a finite `maxTokens` that cannot fit
    /// the reasoning budget, transition and this reserve. The default guarantees
    /// at least one answer token; applications should choose a larger product-
    /// appropriate reserve.
    public let minimumAnswerTokenCount: Int

    /// Replaces the model's ``ReasoningConfig/budgetTransition``.
    ///
    /// Most reasoning protocols ship with no declared transition, because
    /// sharing `<think>`/`</think>` with Qwen3 does not mean a family was
    /// trained to answer from a truncated reasoning span. Supplying a
    /// transition here is how an application opts a model in anyway, after
    /// validating the behavior for its exact weights and tokenizer.
    ///
    /// ``ReasoningBudgetTransition/immediate`` is the conservative choice: it
    /// closes the span with the model's own end delimiter and adds nothing else.
    public let transitionOverride: ReasoningBudgetTransition?

    public init(
        maximumTokenCount: Int,
        minimumAnswerTokenCount: Int = 1,
        transitionOverride: ReasoningBudgetTransition? = nil
    ) throws {
        guard maximumTokenCount >= 0 else {
            throw ThinkingBudgetError.invalidMaximumTokenCount(maximumTokenCount)
        }
        guard minimumAnswerTokenCount >= 0 else {
            throw ThinkingBudgetError.invalidMinimumAnswerTokenCount(minimumAnswerTokenCount)
        }
        self.maximumTokenCount = maximumTokenCount
        self.minimumAnswerTokenCount = minimumAnswerTokenCount
        self.transitionOverride = transitionOverride
    }
}

/// A token-level, protocol-aware reasoning budget.
///
/// The processor observes compiled start and end token sequences instead of
/// repeatedly decoding generated text. At the budget boundary it either lets a
/// partially sampled natural end finish, or forces the model-specific
/// ``ReasoningBudgetTransition``. Every forced token passes through the normal
/// generation loop and therefore remains coherent with KV cache, history,
/// speculative verification, and the emitted stream.
///
/// ### Pipelining
///
/// `TokenIterator` samples a token, hands it to the processor, and only then
/// calls `asyncEval`, so that the *previous* token is what gets read back on
/// the CPU. Reading the newest token instead would block on the in-flight
/// forward pass and collapse that one-step lookahead — the same hazard
/// `TokenRing` avoids for the penalty processors. This processor therefore
/// keeps the freshest sampled token in flight and only drains it when a budget
/// decision is actually within reach. See ``didSample(token:)``.
public struct ThinkingBudgetProcessor: LogitProcessor {
    /// Once the budget token has supplied the first byte, a valid UTF-8 scalar
    /// needs at most three more one-byte fallback tokens.
    static let maximumUnicodeCompletionTokenCount = 3

    private let plan: ThinkingBudgetPlan

    private var boundaries: ReasoningBoundaryTracker
    private var detokenizer: NaiveStreamingDetokenizer
    private var reasoningTokenCount = 0
    private var state: State = .observing
    private let diagnosticHandler: (@Sendable (ThinkingBudgetDiagnostic) -> Void)?

    /// Sampled tokens whose IDs have not been read back from the GPU yet.
    ///
    /// Everything except the last element was async-evaluated during an earlier
    /// step, so reading it costs nothing.
    private var deferredTokens: [MLXArray] = []

    /// Whether the processor stood down because a sampled token did not match
    /// the constraint it had applied.
    ///
    /// That happens when another component overrides this one, or when the
    /// tokenizer disagrees with the model vocabulary. Standing down is the
    /// correct conservative response — the budget stops interfering and the
    /// model finishes under its own control — so this is a diagnostic, not an
    /// error. It is deliberately not a trap: a `precondition` here would crash a
    /// shipping generation loop, and an `assertionFailure` would crash debug
    /// builds of client apps.
    public private(set) var didStandDown = false

    /// Sampled tokens still in flight on the GPU. Pins the pipelining contract
    /// in tests; see ``didSample(token:)``.
    var deferredTokenCount: Int { deferredTokens.count }

    private enum State {
        case observing
        case completingUnicode(additionalTokenCount: Int)
        case forcing(index: Int)
        case finished
    }

    /// Creates a processor for direct use with a token iterator.
    ///
    /// Prefer
    /// ``GenerationComponents/applyingThinkingBudget(_:reasoning:tokenizer:diagnosticHandler:)``
    /// for `ChatSession` and the parameter-driven generation APIs. That path
    /// also validates that `maxTokens` leaves transition and answer headroom.
    public init(
        configuration: ThinkingBudgetConfiguration,
        reasoning: ReasoningConfig,
        tokenizer: any Tokenizer,
        diagnosticHandler: (@Sendable (ThinkingBudgetDiagnostic) -> Void)? = nil
    ) throws {
        self.init(
            plan: try ThinkingBudgetPlan(
                configuration: configuration, reasoning: reasoning, tokenizer: tokenizer),
            diagnosticHandler: diagnosticHandler)
    }

    init(
        plan: ThinkingBudgetPlan,
        diagnosticHandler: (@Sendable (ThinkingBudgetDiagnostic) -> Void)? = nil
    ) {
        self.plan = plan
        self.diagnosticHandler = diagnosticHandler
        self.boundaries = ReasoningBoundaryTracker(
            start: plan.startTokenIDs, ends: plan.endTokenSequences)
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: plan.tokenizer)
    }

    public mutating func prompt(_ prompt: MLXArray) {
        boundaries.reset(with: prompt.asArray(Int.self))
        detokenizer = NaiveStreamingDetokenizer(tokenizer: plan.tokenizer)
        deferredTokens.removeAll(keepingCapacity: true)
        reasoningTokenCount = 0
        state =
            boundaries.isInsideReasoning && plan.configuration.maximumTokenCount == 0
            ? .forcing(index: 0)
            : .observing
    }

    public func process(logits: MLXArray) -> MLXArray {
        switch state {
        case .forcing(let index):
            return constrain(logits, to: [plan.forcedTransitionTokenIDs[index]])

        case .observing:
            let remaining = plan.configuration.maximumTokenCount - reasoningTokenCount
            guard
                let continuation = boundaries.requiredEndContinuation(
                    remainingReasoningTokens: remaining)
            else { return logits }
            return constrain(logits, to: continuation, preservingAllowedLogits: true)

        case .completingUnicode, .finished:
            return logits
        }
    }

    /// Records a sampled token, deferring the CPU←GPU read when it is safe.
    ///
    /// The newest `MLXArray` has not been evaluated yet: `item(_:)` on it blocks
    /// until the current forward pass finishes, which idles the GPU for every
    /// subsequent graph build. Holding it back for one step makes boundary
    /// detection lag by one token — unobservable while no decision can be taken
    /// inside the lag window. Once the budget comes within
    /// `ThinkingBudgetPlan.exactTrackingWindow` tokens the processor drains
    /// completely, accepting one stall per token, and stays exact through the
    /// forced transition. After the reasoning span closes it stops reading
    /// tokens altogether.
    public mutating func didSample(token: MLXArray) {
        if case .finished = state { return }

        deferredTokens.append(token)

        while deferredTokens.count > (needsExactTracking ? 0 : 1) {
            consume(deferredTokens.removeFirst().item(Int.self))
            if case .finished = state {
                // Nothing will ever be observed again, so release whatever is
                // still in flight rather than pinning its buffers for the rest
                // of the generation.
                deferredTokens.removeAll(keepingCapacity: true)
                return
            }
        }
    }

    /// Whether the state machine must see every sampled token immediately.
    ///
    /// True whenever a transition is in progress, and once the remaining budget
    /// is small enough that a decision could fall inside the deferred window.
    private var needsExactTracking: Bool {
        guard case .observing = state else { return true }
        let remaining = plan.configuration.maximumTokenCount - reasoningTokenCount
        return remaining <= deferredTokens.count + plan.exactTrackingWindow
    }

    /// Whether the boundary detokenizer is being fed.
    ///
    /// Only the transition decision needs to know whether a token completed a
    /// Unicode scalar, and `NaiveStreamingDetokenizer` re-decodes its whole
    /// current line on every token. Warming it up
    /// `ThinkingBudgetPlan.exactTrackingWindow` tokens before the budget
    /// gives it far more than the four bytes a scalar can span, so at the
    /// boundary it answers exactly as a stream-long detokenizer would.
    private var tracksTextBoundaries: Bool {
        guard case .observing = state else { return true }
        return plan.configuration.maximumTokenCount - reasoningTokenCount
            <= plan.exactTrackingWindow
    }

    private mutating func consume(_ tokenID: Int) {
        switch state {
        case .finished:
            return

        case .forcing(let index):
            guard tokenID == plan.forcedTransitionTokenIDs[index] else {
                standDown(
                    expectedTokenIDs: [plan.forcedTransitionTokenIDs[index]],
                    sampledTokenID: tokenID)
                return
            }

            let nextIndex = index + 1
            state =
                nextIndex == plan.forcedTransitionTokenIDs.count
                ? .finished
                : .forcing(index: nextIndex)

        case .observing:
            if let required = boundaries.requiredEndContinuation(
                remainingReasoningTokens:
                    plan.configuration.maximumTokenCount - reasoningTokenCount),
                !required.contains(tokenID)
            {
                standDown(expectedTokenIDs: required, sampledTokenID: tokenID)
                return
            }

            let wasInsideReasoning = boundaries.isInsideReasoning
            let observation = boundaries.observe(tokenID)
            let completesText = trackTextBoundary(tokenID)

            guard !observation.endedReasoning else {
                state = .finished
                return
            }

            guard wasInsideReasoning else {
                if observation.startedReasoning
                    && plan.configuration.maximumTokenCount == 0
                {
                    state = .forcing(index: 0)
                }
                return
            }

            reasoningTokenCount += observation.committedReasoningTokenCount
            guard reasoningTokenCount >= plan.configuration.maximumTokenCount else {
                return
            }

            // A proper prefix of a natural end has reached the boundary. Leave
            // the processor observing: process(logits:) now constrains sampling
            // to valid continuations, avoiding malformed double-closes.
            guard !boundaries.hasPendingEndPrefix else { return }
            state =
                completesText
                ? .forcing(index: 0)
                : .completingUnicode(additionalTokenCount: 0)

        case .completingUnicode(let additionalTokenCount):
            let observation = boundaries.observe(tokenID)
            let completesText = trackTextBoundary(tokenID)
            guard !observation.endedReasoning else {
                state = .finished
                return
            }
            if completesText {
                state = boundaries.hasPendingEndPrefix ? .observing : .forcing(index: 0)
                return
            }

            let nextCount = additionalTokenCount + 1
            guard nextCount < Self.maximumUnicodeCompletionTokenCount else {
                standDown(.unicodeBoundaryCompletionFailed(sampledTokenCount: nextCount))
                return
            }
            state = .completingUnicode(additionalTokenCount: nextCount)
        }
    }

    /// Stop enforcing and stop observing, leaving generation untouched.
    ///
    /// The fail-safe: if the processor loses sync with the sampler, getting out
    /// of the way preserves valid model output, whereas trapping would take the
    /// host application down mid-generation. See ``didStandDown``.
    private mutating func standDown(expectedTokenIDs: [Int], sampledTokenID: Int) {
        standDown(
            .enforcementDisabled(
                expectedTokenIDs: expectedTokenIDs, sampledTokenID: sampledTokenID))
    }

    private mutating func standDown(_ diagnostic: ThinkingBudgetDiagnostic) {
        didStandDown = true
        state = .finished
        deferredTokens.removeAll(keepingCapacity: true)
        diagnosticHandler?(diagnostic)
    }

    private mutating func trackTextBoundary(_ tokenID: Int) -> Bool {
        guard tracksTextBoundaries else { return false }
        detokenizer.append(token: tokenID)
        return detokenizer.next() != nil
    }

    private func constrain(
        _ logits: MLXArray,
        to tokenIDs: [Int],
        preservingAllowedLogits: Bool = false
    ) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        guard !tokenIDs.isEmpty,
            tokenIDs.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            // The compiled boundary is not in this model's vocabulary, so the
            // tokenizer and the weights disagree. Masking to a token the model
            // cannot emit would leave the sampler nothing to pick, so leave the
            // logits alone; `consume(_:)` then stands the processor down when
            // the resulting token does not match the intended constraint.
            return logits
        }

        // Scatter rather than compare against a full vocabulary range: this is
        // O(vocabulary) once plus O(constraint), and matches how the penalty
        // processors address logits (`takeAlong` / `putAlong`).
        var indices = MLXArray(tokenIDs.map(Int32.init))
        if logits.ndim > 1 {
            indices = indices.reshaped(
                Array(repeating: 1, count: logits.ndim - 1) + [tokenIDs.count])
        }

        let masked = MLXArray.full(
            logits.shape, values: MLXArray.maskFill(for: logits.dtype), dtype: logits.dtype)

        // Forced transition tokens intentionally override preceding processors.
        // When more than one natural boundary continuation is possible, retain
        // the model's relative preference (for example `</think>` versus
        // `<tool_call>`) while masking every non-protocol token.
        let allowedValues =
            preservingAllowedLogits
            ? takeAlong(logits, indices, axis: -1)
            : MLXArray.zeros(indices.shape, dtype: logits.dtype)

        return putAlong(masked, indices, values: allowedValues, axis: -1)
    }
}

extension GenerationComponents {
    /// Return a copy that enforces a reasoning-token budget for every generation.
    ///
    /// The reasoning protocol must declare a ``ReasoningBudgetTransition``, or
    /// the caller must supply one via
    /// ``ThinkingBudgetConfiguration/transitionOverride``. Unknown protocols
    /// fail during setup rather than receiving a guessed `</think>` sequence.
    /// The configured generation limit must include enough headroom for a
    /// Unicode boundary completion, the longest possible protocol transition,
    /// and the answer.
    public func applyingThinkingBudget(
        _ configuration: ThinkingBudgetConfiguration,
        reasoning: ReasoningConfig,
        tokenizer: any Tokenizer,
        diagnosticHandler: (@Sendable (ThinkingBudgetDiagnostic) -> Void)? = nil
    ) throws -> Self {
        let plan = try ThinkingBudgetPlan(
            configuration: configuration, reasoning: reasoning, tokenizer: tokenizer)
        let protocolHeadroom = max(
            plan.forcedTransitionTokenIDs.count,
            (plan.endTokenSequences.map(\.count).max() ?? 1) - 1)
        let (budgetAndUnicode, unicodeOverflow) =
            configuration.maximumTokenCount.addingReportingOverflow(
                ThinkingBudgetProcessor.maximumUnicodeCompletionTokenCount)
        let (budgetAndProtocol, protocolOverflow) =
            budgetAndUnicode.addingReportingOverflow(protocolHeadroom)
        let (requiredTokenCount, answerOverflow) =
            budgetAndProtocol.addingReportingOverflow(configuration.minimumAnswerTokenCount)
        guard !unicodeOverflow, !protocolOverflow, !answerOverflow else {
            throw ThinkingBudgetError.generationTokenRequirementOverflow
        }

        return appendingParameterValidator { parameters in
            guard let maxTokens = parameters.maxTokens else { return }
            guard maxTokens >= requiredTokenCount else {
                throw ThinkingBudgetError.insufficientGenerationTokenLimit(
                    required: requiredTokenCount, actual: maxTokens)
            }
        }.appendingLogitProcessor {
            ThinkingBudgetProcessor(plan: plan, diagnosticHandler: diagnosticHandler)
        }
    }
}

struct ThinkingBudgetPlan: Sendable {
    let configuration: ThinkingBudgetConfiguration
    let tokenizer: any Tokenizer
    let startTokenIDs: [Int]
    let endTokenSequences: [[Int]]
    let forcedTransitionTokenIDs: [Int]

    /// Tokens of slack before the budget within which tracking becomes exact.
    ///
    /// Covers the longest end delimiter — a partial natural close may straddle
    /// the boundary — plus the four bytes a UTF-8 scalar can span.
    let exactTrackingWindow: Int

    init(
        configuration: ThinkingBudgetConfiguration,
        reasoning: ReasoningConfig,
        tokenizer: any Tokenizer
    ) throws {
        guard let transition = configuration.transitionOverride ?? reasoning.budgetTransition
        else {
            throw ThinkingBudgetError.unsupportedReasoningProtocol
        }

        let startTokenIDs = try Self.encodeBoundary(reasoning.startDelimiter, tokenizer: tokenizer)
        let endDelimiters = [reasoning.endDelimiter] + reasoning.implicitEndDelimiters
        var endTokenSequences: [[Int]] = []
        for delimiter in endDelimiters {
            let sequence = try Self.encodeBoundary(delimiter, tokenizer: tokenizer)
            if !endTokenSequences.contains(sequence) {
                endTokenSequences.append(sequence)
            }
        }

        let forcedTransition =
            transition.beforeEndDelimiter + reasoning.endDelimiter
            + transition.afterEndDelimiter
        let forcedTransitionTokenIDs = tokenizer.encode(
            text: forcedTransition, addSpecialTokens: false)
        guard !forcedTransitionTokenIDs.isEmpty else {
            throw ThinkingBudgetError.unencodableTransition(forcedTransition)
        }
        try Self.validate(forcedTransitionTokenIDs)

        self.configuration = configuration
        self.tokenizer = tokenizer
        self.startTokenIDs = startTokenIDs
        self.endTokenSequences = endTokenSequences
        self.forcedTransitionTokenIDs = forcedTransitionTokenIDs
        self.exactTrackingWindow = (endTokenSequences.map(\.count).max() ?? 1) + 4
    }

    private static func encodeBoundary(
        _ boundary: String, tokenizer: any Tokenizer
    ) throws -> [Int] {
        let tokenIDs = tokenizer.encode(text: boundary, addSpecialTokens: false)
        guard !tokenIDs.isEmpty else {
            throw ThinkingBudgetError.unencodableBoundary(boundary)
        }
        try validate(tokenIDs)
        return tokenIDs
    }

    private static func validate(_ tokenIDs: [Int]) throws {
        if let invalidTokenID = tokenIDs.first(where: { $0 < 0 || $0 > Int(Int32.max) }) {
            throw ThinkingBudgetError.invalidTokenID(invalidTokenID)
        }
    }
}

private struct ReasoningBoundaryTracker: Sendable {
    struct Observation {
        var startedReasoning = false
        var endedReasoning = false
        var committedReasoningTokenCount = 0
    }

    private let start: [Int]
    private let ends: [[Int]]
    private var pending: [Int] = []

    private(set) var isInsideReasoning = false

    init(start: [Int], ends: [[Int]]) {
        precondition(!start.isEmpty && !ends.isEmpty)
        self.start = start
        self.ends = ends
    }

    var hasPendingEndPrefix: Bool {
        isInsideReasoning && !pending.isEmpty
    }

    mutating func reset(with tokens: [Int]) {
        pending.removeAll(keepingCapacity: true)
        isInsideReasoning = false
        for token in tokens {
            _ = observe(token)
        }
    }

    mutating func observe(_ token: Int) -> Observation {
        var observation = Observation()
        pending.append(token)

        if isInsideReasoning {
            if ends.contains(pending) {
                pending.removeAll(keepingCapacity: true)
                isInsideReasoning = false
                observation.endedReasoning = true
                return observation
            }

            let held = longestSuffixThatIsProperPrefix(of: ends)
            observation.committedReasoningTokenCount = pending.count - held.count
            pending = held
        } else {
            if pending == start {
                pending.removeAll(keepingCapacity: true)
                isInsideReasoning = true
                observation.startedReasoning = true
                return observation
            }
            pending = longestSuffixThatIsProperPrefix(of: [start])
        }

        return observation
    }

    /// When a possible natural end consumes the remaining budget, constrain
    /// the next sample to a valid continuation instead of letting the prefix
    /// become reasoning text and then injecting a second closing sequence.
    func requiredEndContinuation(remainingReasoningTokens: Int) -> [Int]? {
        guard isInsideReasoning, !pending.isEmpty,
            pending.count >= max(0, remainingReasoningTokens)
        else { return nil }

        let next = ends.compactMap { sequence -> Int? in
            guard sequence.count > pending.count,
                sequence.starts(with: pending)
            else { return nil }
            return sequence[pending.count]
        }
        let unique = Array(Set(next)).sorted()
        return unique.isEmpty ? nil : unique
    }

    private func longestSuffixThatIsProperPrefix(of sequences: [[Int]]) -> [Int] {
        let maximumLength = sequences.map(\.count).max() ?? 0
        let upperBound = min(pending.count, max(0, maximumLength - 1))
        guard upperBound > 0 else { return [] }

        for length in stride(from: upperBound, through: 1, by: -1) {
            let suffix = Array(pending.suffix(length))
            if sequences.contains(where: { $0.count > length && $0.starts(with: suffix) }) {
                return suffix
            }
        }
        return []
    }
}
