// Copyright © 2026 Apple Inc.

import MLX
import XCTest

@testable import MLXLMCommon

final class ThinkingBudgetTests: XCTestCase {
    private let tokenizer = ByteTokenizer()
    private let reasoning = QwenReasoningProtocol.qwen3

    func testConfigurationRejectsNegativeBudgetAndAcceptsZero() throws {
        XCTAssertThrowsError(
            try ThinkingBudgetConfiguration(maximumTokenCount: -1)
        ) { error in
            XCTAssertEqual(error as? ThinkingBudgetError, .invalidMaximumTokenCount(-1))
        }
        XCTAssertEqual(
            try ThinkingBudgetConfiguration(maximumTokenCount: 0).maximumTokenCount, 0)
        XCTAssertThrowsError(
            try ThinkingBudgetConfiguration(
                maximumTokenCount: 1, minimumAnswerTokenCount: -1)
        ) { error in
            XCTAssertEqual(
                error as? ThinkingBudgetError, .invalidMinimumAnswerTokenCount(-1))
        }
    }

    func testForcesModelDeclaredTransitionAfterExactBudget() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 2),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("B")), ascii("B"))

        let transition = tokenizer.encode(
            text: qwenEarlyStopText, addSpecialTokens: false)
        let forced = transition.map { expected in
            let actual = sample(&processor, modelToken: ascii("Z"))
            XCTAssertEqual(actual, expected)
            return actual
        }
        XCTAssertEqual(forced, transition)

        // The budget is terminal after closing; answer generation is untouched.
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    func testNaturalCloseDisablesBudgetWithoutInjectingAnotherDelimiter() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 32),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        for token in tokenizer.encode(text: "short</think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }

        for _ in 0 ..< 40 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        }
    }

    func testPartialNaturalCloseAtBudgetBoundaryIsNotDoubleClosed() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 2),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        for token in tokenizer.encode(text: "</think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }

        // No forced newline or duplicate close remains after the natural close.
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    func testPotentialNaturalCloseIsCompletedInsteadOfDuplicated() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 2),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("<")), ascii("<"))

        // The `<` may be the final reasoning token or the beginning of the
        // natural close. At the boundary the processor safely completes the
        // latter, even when the model proposes a divergent token.
        for expected in tokenizer.encode(text: "/think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: ascii("X")), expected)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    func testImplicitToolCallBoundaryStopsBudgetEnforcement() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 32),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        for token in tokenizer.encode(text: "plan<tool_call>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }

        // Tool arguments must never be corrupted by an injected reasoning end.
        for token in tokenizer.encode(text: #"{"name":"search"}"#, addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }
    }

    func testPotentialToolCallAtBudgetBoundaryIsCompletedWithoutCorruptingArguments() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 2),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("<")), ascii("<"))

        for expected in tokenizer.encode(text: "tool_call>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: expected), expected)
        }
        for token in tokenizer.encode(text: #"{"name":"search"}"#, addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }
    }

    func testDoesNotBudgetOrdinaryAnswerWhenReasoningNeverStarts() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 1),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n"))

        for _ in 0 ..< 5 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        }
    }

    func testDetectsGeneratedOpeningDelimiterAcrossTokens() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 1),
            reasoning: reasoning,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n"))

        for token in tokenizer.encode(text: "<think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("\n"))
    }

    func testZeroBudgetClosesPromptPrimedReasoningImmediately() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 0),
            reasoning: ReasoningConfig(
                startDelimiter: "<think>", endDelimiter: "</think>",
                promptStrategy: .alwaysOn, budgetTransition: .immediate),
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        for expected in tokenizer.encode(text: "</think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), expected)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
    }

    func testGenerationComponentsComposesBudgetAfterExistingProcessor() throws {
        let components = try GenerationComponents()
            .appendingLogitProcessor { AdditiveProcessor(amount: 1) }
            .applyingThinkingBudget(
                ThinkingBudgetConfiguration(maximumTokenCount: 1),
                reasoning: reasoning,
                tokenizer: tokenizer)

        var processor = try XCTUnwrap(
            components.logitProcessor(parameters: GenerateParameters()))
        XCTAssertTrue(processor is ChainedLogitProcessor)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("\n"))
    }

    func testComponentsProduceFreshBudgetStateForEveryGeneration() throws {
        let components = try GenerationComponents().applyingThinkingBudget(
            ThinkingBudgetConfiguration(maximumTokenCount: 1),
            reasoning: reasoning,
            tokenizer: tokenizer)

        var first = try XCTUnwrap(components.logitProcessor(parameters: .init()))
        first.prompt(tokens("assistant\n<think>\n"))
        XCTAssertEqual(sample(&first, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&first, modelToken: ascii("Z")), ascii("\n"))

        var second = try XCTUnwrap(components.logitProcessor(parameters: .init()))
        second.prompt(tokens("assistant\n<think>\n"))
        XCTAssertEqual(sample(&second, modelToken: ascii("B")), ascii("B"))
    }

    func testComponentsRejectGenerationLimitWithoutTransitionAndAnswerHeadroom() throws {
        let configuration = try ThinkingBudgetConfiguration(
            maximumTokenCount: 8, minimumAnswerTokenCount: 4)
        let components = try GenerationComponents().applyingThinkingBudget(
            configuration, reasoning: reasoning, tokenizer: tokenizer)
        let transitionCount = tokenizer.encode(
            text: qwenEarlyStopText, addSpecialTokens: false
        ).count
        let maximumUnicodeCompletionTokenCount = 3
        let required = 8 + maximumUnicodeCompletionTokenCount + transitionCount + 4

        XCTAssertThrowsError(
            try components.validate(parameters: .init(maxTokens: required - 1))
        ) { error in
            XCTAssertEqual(
                error as? ThinkingBudgetError,
                .insufficientGenerationTokenLimit(required: required, actual: required - 1))
        }
        XCTAssertNoThrow(try components.validate(parameters: .init(maxTokens: required)))
        XCTAssertNoThrow(try components.validate(parameters: .init(maxTokens: nil)))
    }

    func testRejectsUnrepresentableGenerationTokenRequirement() throws {
        let configuration = try ThinkingBudgetConfiguration(
            maximumTokenCount: Int.max, minimumAnswerTokenCount: 1)

        XCTAssertThrowsError(
            try GenerationComponents().applyingThinkingBudget(
                configuration, reasoning: reasoning, tokenizer: tokenizer)
        ) { error in
            XCTAssertEqual(
                error as? ThinkingBudgetError, .generationTokenRequirementOverflow)
        }
    }

    func testRejectsProtocolWithoutDeclaredBudgetTransition() throws {
        let unsupported = ReasoningConfig(
            startDelimiter: "<reasoning>", endDelimiter: "</reasoning>",
            promptStrategy: .none)

        XCTAssertThrowsError(
            try GenerationComponents().applyingThinkingBudget(
                ThinkingBudgetConfiguration(maximumTokenCount: 1),
                reasoning: unsupported, tokenizer: tokenizer)
        ) { error in
            XCTAssertEqual(error as? ThinkingBudgetError, .unsupportedReasoningProtocol)
        }
    }

    func testRejectsUnencodableBoundary() throws {
        let noDelimiter = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "",
            promptStrategy: .alwaysOn,
            budgetTransition: .immediate)

        XCTAssertThrowsError(
            try GenerationComponents().applyingThinkingBudget(
                ThinkingBudgetConfiguration(maximumTokenCount: 1),
                reasoning: noDelimiter, tokenizer: tokenizer)
        ) { error in
            XCTAssertEqual(error as? ThinkingBudgetError, .unencodableBoundary(""))
        }
    }

    /// Qwen's published `early_stopping_text`, byte-for-byte.
    ///
    /// Source: Qwen3 quickstart, "Thinking Budget". The space after the two
    /// newlines is part of the published string and therefore part of its token
    /// sequence.
    private static let qwenCanonicalEarlyStoppingText =
        "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

    func testQwen3PresetUsesOfficialEarlyStopTransition() throws {
        let reasoning = QwenReasoningProtocol.qwen3
        let transition = try XCTUnwrap(reasoning.budgetTransition)

        XCTAssertEqual(
            transition.beforeEndDelimiter + reasoning.endDelimiter
                + transition.afterEndDelimiter,
            Self.qwenCanonicalEarlyStoppingText)
    }

    /// Guards the exact token sequence, not just the characters.
    ///
    /// Qwen3 encodes the canonical prefix as `[271, 55777]` (`"\n\n"`,
    /// `" Considering"`). `QwenLikeTokenizer` reproduces that distinction with
    /// the real IDs.
    func testQwen3TransitionEncodesToQwensCanonicalTokenSequence() throws {
        let tokenizer = QwenLikeTokenizer()
        let plan = try ThinkingBudgetPlan(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 8),
            reasoning: QwenReasoningProtocol.qwen3,
            tokenizer: tokenizer)

        XCTAssertEqual(plan.forcedTransitionTokenIDs.count, 24)
        XCTAssertEqual(plan.forcedTransitionTokenIDs.prefix(2), [271, 55777])
        XCTAssertFalse(plan.forcedTransitionTokenIDs.contains(82796))

        // `</think>` is a single special token, and it is inside the sequence.
        XCTAssertEqual(plan.endTokenSequences, [[151668], [151657]])
        XCTAssertTrue(plan.forcedTransitionTokenIDs.contains(151668))
    }

    // MARK: - Single-token delimiters
    //
    // `ByteTokenizer` makes every delimiter multi-token, which is the opposite
    // regime from real Qwen weights where `<think>` and `</think>` are single
    // special tokens. These pin that shape so the prefix/suffix machinery in
    // `ReasoningBoundaryTracker` is never the only path under test.

    func testSingleTokenDelimitersForceTransitionAtExactBudget() throws {
        let qwen = QwenLikeTokenizer()
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 3),
            reasoning: QwenReasoningProtocol.qwen3,
            tokenizer: qwen)

        // Qwen primes the opening delimiter in the rendered prompt.
        processor.prompt(qwenTokens("<think>\n", qwen))

        let filler = 279  // " the"
        for _ in 0 ..< 3 {
            XCTAssertEqual(qwenSample(&processor, modelToken: filler), filler)
        }

        let transition = qwen.encode(
            text: Self.qwenCanonicalEarlyStoppingText, addSpecialTokens: false)
        for expected in transition {
            XCTAssertEqual(qwenSample(&processor, modelToken: filler), expected)
        }

        // Terminal: the answer generates freely.
        XCTAssertEqual(qwenSample(&processor, modelToken: filler), filler)
        XCTAssertFalse(processor.didStandDown)
    }

    func testSingleTokenNaturalCloseSuppressesTheBudget() throws {
        let qwen = QwenLikeTokenizer()
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 3),
            reasoning: QwenReasoningProtocol.qwen3,
            tokenizer: qwen)
        processor.prompt(qwenTokens("<think>\n", qwen))

        XCTAssertEqual(qwenSample(&processor, modelToken: 279), 279)
        // `</think>` on its own token closes the span before the budget binds.
        XCTAssertEqual(qwenSample(&processor, modelToken: 151_668), 151_668)

        for _ in 0 ..< 10 {
            XCTAssertEqual(qwenSample(&processor, modelToken: 279), 279)
        }
    }

    // MARK: - Pipelining
    //
    // `TokenIterator` hands the processor a token that has not been evaluated
    // yet, then calls `asyncEval`. Reading that token back would stall the GPU
    // for every subsequent graph build, so the processor must keep it in flight
    // whenever no budget decision can depend on it.

    func testKeepsNewestSampledTokenInFlightAwayFromTheBudget() throws {
        var processor = try makeProcessor(budget: 64)
        processor.prompt(tokens("assistant\n<think>\n"))

        for _ in 0 ..< 20 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
            XCTAssertEqual(
                processor.deferredTokenCount, 1,
                "the freshest sampled token must not be read back mid-generation")
        }
    }

    func testStopsReadingTokensOnceTheReasoningSpanCloses() throws {
        var processor = try makeProcessor(budget: 64)
        processor.prompt(tokens("assistant\n<think>\n"))

        for token in tokenizer.encode(text: "done</think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: token), token)
        }

        // The answer is usually far longer than the reasoning span, so a stall
        // per token here would dominate. Nothing is retained and nothing is read.
        for _ in 0 ..< 20 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
            XCTAssertEqual(processor.deferredTokenCount, 0)
        }
    }

    /// Deferring must not move the boundary: a budget well beyond the exact
    /// tracking window still fires on exactly the configured token.
    func testDeferredTrackingStillFiresOnTheExactBudgetToken() throws {
        let budget = 40
        var processor = try makeProcessor(budget: budget)
        processor.prompt(tokens("assistant\n<think>\n"))

        for _ in 0 ..< budget {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        }

        let transition = tokenizer.encode(text: qwenEarlyStopText, addSpecialTokens: false)
        for expected in transition {
            XCTAssertEqual(sample(&processor, modelToken: ascii("A")), expected)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    func testCompletesUnicodeScalarBeforeForcingTransition() throws {
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(
                maximumTokenCount: 1, transitionOverride: .immediate),
            reasoning: .alwaysOnThinking,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        let emojiBytes = Array("😀".utf8).map(Int.init)
        XCTAssertEqual(sample(&processor, modelToken: emojiBytes[0]), emojiBytes[0])
        for continuation in emojiBytes.dropFirst() {
            XCTAssertEqual(sample(&processor, modelToken: continuation), continuation)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("<"))
        XCTAssertFalse(processor.didStandDown)
    }

    func testStandsDownWhenTokenizerCannotCompleteUnicodeBoundary() throws {
        let diagnostics = DiagnosticRecorder()
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(
                maximumTokenCount: 1, transitionOverride: .immediate),
            reasoning: .alwaysOnThinking,
            tokenizer: tokenizer,
            diagnosticHandler: diagnostics.record)
        processor.prompt(tokens("assistant\n<think>\n"))

        for _ in 0 ..< 4 {
            XCTAssertEqual(sample(&processor, modelToken: 0xF0), 0xF0)
        }

        XCTAssertTrue(processor.didStandDown)
        XCTAssertEqual(
            diagnostics.events,
            [.unicodeBoundaryCompletionFailed(sampledTokenCount: 3)])
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    // MARK: - Speculative decoding

    /// In `SpeculativeTokenIterator.speculateRound` the canonical processor only
    /// ever receives `didSample` for accepted tokens — a throwaway copy does the
    /// `process`/`didSample` pairing for the verify window. The budget must
    /// produce an identical stream either way, and must not stand down.
    func testMatchesAutoregressiveRunUnderSpeculativeAcceptance() throws {
        let steps = 40

        var reference = try makeProcessor(budget: 6)
        reference.prompt(tokens("assistant\n<think>\n"))
        var referenceOutput: [Int] = []
        for _ in 0 ..< steps {
            referenceOutput.append(sample(&reference, modelToken: ascii("A")))
        }

        var canonical = try makeProcessor(budget: 6)
        canonical.prompt(tokens("assistant\n<think>\n"))
        var speculativeOutput: [Int] = []
        while speculativeOutput.count < steps {
            var verify = canonical
            var window: [MLXArray] = []
            for _ in 0 ..< 4 {
                let logits = verify.process(logits: oneHotLogits(for: ascii("A")))
                let token = ArgMaxSampler().sample(logits: logits)
                verify.didSample(token: token)
                window.append(token)
            }
            // The iterator evaluates the whole verify window before comparing.
            MLX.eval(window)

            for token in window {
                canonical.didSample(token: token)
                speculativeOutput.append(token.item(Int.self))
            }
        }

        XCTAssertEqual(Array(speculativeOutput.prefix(steps)), referenceOutput)
        XCTAssertFalse(canonical.didStandDown)
    }

    /// Verification may run the processor several tokens ahead, then reject
    /// the first proposal. Only the correction token is committed to the
    /// canonical processor; discarded verify state must not consume budget.
    func testSpeculativeRejectionDoesNotAdvanceCanonicalBudgetState() throws {
        let steps = 20
        var reference = try makeProcessor(budget: 3)
        reference.prompt(tokens("assistant\n<think>\n"))
        let referenceOutput = (0 ..< steps).map { _ in
            sample(&reference, modelToken: ascii("A"))
        }

        var canonical = try makeProcessor(budget: 3)
        canonical.prompt(tokens("assistant\n<think>\n"))
        var speculativeOutput: [Int] = []
        for _ in 0 ..< steps {
            var verify = canonical
            var verified: [MLXArray] = []
            for _ in 0 ..< 4 {
                let logits = verify.process(logits: oneHotLogits(for: ascii("A")))
                let token = ArgMaxSampler().sample(logits: logits)
                verify.didSample(token: token)
                verified.append(token)
            }

            // Pretend the first draft proposal differed. The iterator discards
            // the speculative processor copy and commits only the correction.
            let correction = verified[0]
            canonical.didSample(token: correction)
            speculativeOutput.append(correction.item(Int.self))
        }

        XCTAssertEqual(speculativeOutput, referenceOutput)
        XCTAssertFalse(canonical.didStandDown)
    }

    // MARK: - Caller-supplied transitions

    func testTransitionOverrideEnablesBudgetOnProtocolWithoutOne() throws {
        // DeepSeek-R1 declares no transition; an application may still opt in.
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(
                maximumTokenCount: 2, transitionOverride: .immediate),
            reasoning: .alwaysOnThinking,
            tokenizer: tokenizer)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("B")), ascii("B"))

        // `.immediate` closes the span and adds nothing else.
        for expected in tokenizer.encode(text: "</think>", addSpecialTokens: false) {
            XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), expected)
        }
        XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
    }

    func testTransitionOverrideTakesPrecedenceOverModelDeclaration() throws {
        let plan = try ThinkingBudgetPlan(
            configuration: ThinkingBudgetConfiguration(
                maximumTokenCount: 4, transitionOverride: .immediate),
            reasoning: QwenReasoningProtocol.qwen3,
            tokenizer: tokenizer)

        XCTAssertEqual(
            plan.forcedTransitionTokenIDs,
            tokenizer.encode(text: "</think>", addSpecialTokens: false))
    }

    // MARK: - Fail-safe

    /// Losing sync with the sampler must never trap. `precondition` would take
    /// a shipping app down mid-generation and `assertionFailure` would take down
    /// debug builds of client apps, so the processor stands down instead and
    /// leaves the model's own output untouched.
    func testStandsDownInsteadOfTrappingWhenTheForcedTokenIsIgnored() throws {
        let diagnostics = DiagnosticRecorder()
        var processor = try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: 2),
            reasoning: reasoning,
            tokenizer: tokenizer,
            diagnosticHandler: diagnostics.record)
        processor.prompt(tokens("assistant\n<think>\n"))

        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("B")), ascii("B"))

        // A downstream component overrides the constraint.
        processor.didSample(token: MLXArray([Int32(ascii("Q"))]))

        XCTAssertTrue(processor.didStandDown)
        XCTAssertEqual(
            diagnostics.events,
            [.enforcementDisabled(expectedTokenIDs: [ascii("\n")], sampledTokenID: ascii("Q"))]
        )
        XCTAssertEqual(processor.deferredTokenCount, 0)
        for _ in 0 ..< 5 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
        }
    }

    /// A tokenizer that does not match the model's vocabulary cannot mask to a
    /// token the model is able to emit. Rather than leaving the sampler with an
    /// all-masked row, the constraint is skipped and the budget stands down.
    func testStandsDownWhenBoundaryTokensExceedTheModelVocabulary() throws {
        var processor = try makeProcessor(budget: 1)
        processor.prompt(tokens("assistant\n<think>\n"))
        XCTAssertEqual(sample(&processor, modelToken: ascii("A")), ascii("A"))

        // The transition opens with `\n` (10), outside this 8-token vocabulary.
        let processed = processor.process(logits: oneHotLogits(for: 5, vocabularySize: 8))
        XCTAssertEqual(
            ArgMaxSampler().sample(logits: processed).item(Int.self), 5,
            "an unsatisfiable constraint must leave the logits untouched")

        processor.didSample(token: MLXArray([Int32(5)]))
        XCTAssertTrue(processor.didStandDown)

        for _ in 0 ..< 5 {
            XCTAssertEqual(sample(&processor, modelToken: ascii("Z")), ascii("Z"))
        }
    }

    // MARK: - Helpers

    private func makeProcessor(budget: Int) throws -> ThinkingBudgetProcessor {
        try ThinkingBudgetProcessor(
            configuration: ThinkingBudgetConfiguration(maximumTokenCount: budget),
            reasoning: reasoning,
            tokenizer: tokenizer)
    }

    private func tokens(_ text: String) -> MLXArray {
        MLXArray(tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init))
    }

    private func qwenTokens(_ text: String, _ qwen: QwenLikeTokenizer) -> MLXArray {
        MLXArray(qwen.encode(text: text, addSpecialTokens: false).map(Int32.init))
    }

    private func qwenSample(
        _ processor: inout ThinkingBudgetProcessor, modelToken: Int
    ) -> Int {
        sample(&processor, modelToken: modelToken, vocabularySize: 152_064)
    }

    private func oneHotLogits(for token: Int, vocabularySize: Int = 256) -> MLXArray {
        var values = [Float](repeating: -10, count: vocabularySize)
        values[token] = 10
        return MLXArray(values)[.newAxis, 0...]
    }

    private func ascii(_ character: Character) -> Int {
        Int(character.asciiValue!)
    }

    private var qwenEarlyStopText: String {
        let transition = reasoning.budgetTransition!
        return transition.beforeEndDelimiter + reasoning.endDelimiter
            + transition.afterEndDelimiter
    }

    private func sample<P: LogitProcessor>(
        _ processor: inout P, modelToken: Int, vocabularySize: Int = 256
    ) -> Int {
        var values = [Float](repeating: -10, count: vocabularySize)
        values[modelToken] = 10
        let logits = MLXArray(values)[.newAxis, 0...]
        let token = ArgMaxSampler().sample(logits: processor.process(logits: logits))
        processor.didSample(token: token)
        return token.item(Int.self)
    }
}

private struct AdditiveProcessor: LogitProcessor {
    let amount: Float

    func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits + amount }
    func didSample(token: MLXArray) {}
}

private final class DiagnosticRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [ThinkingBudgetDiagnostic] = []

    func record(_ diagnostic: ThinkingBudgetDiagnostic) {
        lock.withLock { storage.append(diagnostic) }
    }

    var events: [ThinkingBudgetDiagnostic] {
        lock.withLock { storage }
    }
}

/// A tokenizer that reproduces Qwen3's real token IDs for the strings these
/// tests exercise.
///
/// `ByteTokenizer` makes every delimiter multi-token. Real Qwen weights are the
/// opposite: `<think>`, `</think>` and `<tool_call>` are single special tokens,
/// and the early-stop text is a specific 24-token BPE sequence. Both regimes
/// need coverage, so this fixture carries the actual IDs from
/// `Qwen/Qwen3-0.6B` for the pieces involved and falls back to a disjoint byte
/// range for anything else.
private struct QwenLikeTokenizer: Tokenizer {
    /// Disjoint from every ID in ``pieces`` so decoding stays unambiguous.
    private static let byteFallbackBase = 100_000

    /// Real `Qwen/Qwen3-0.6B` IDs. Matched longest-first, as BPE would.
    private static let pieces: [(text: String, id: Int)] = [
        ("<think>", 151_667),
        ("</think>", 151_668),
        ("<tool_call>", 151_657),
        ("\n\n", 271),
        ("Considering", 82_796),
        (" Considering", 55_777),
        (" the", 279),
        (" limited", 7_199),
        (" time", 882),
        (" by", 553),
        (" user", 1_196),
        (",", 11),
        (" I", 358),
        (" have", 614),
        (" to", 311),
        (" give", 2_968),
        (" solution", 6_291),
        (" based", 3_118),
        (" on", 389),
        (" thinking", 7_274),
        (" directly", 5_961),
        (" now", 1_431),
        (".\n", 624),
    ].sorted { $0.text.count > $1.text.count }

    private static let textByID = Dictionary(
        uniqueKeysWithValues: pieces.map { ($0.id, $0.text) })

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let piece = Self.pieces.first(where: { rest.hasPrefix($0.text) }) {
                ids.append(piece.id)
                rest = rest.dropFirst(piece.text.count)
                continue
            }
            for byte in String(rest.removeFirst()).utf8 {
                ids.append(Self.byteFallbackBase + Int(byte))
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var bytes: [UInt8] = []
        for id in tokenIds {
            if let text = Self.textByID[id] {
                bytes.append(contentsOf: Array(text.utf8))
            } else if (Self.byteFallbackBase ... Self.byteFallbackBase + 255).contains(id) {
                bytes.append(UInt8(id - Self.byteFallbackBase))
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        Self.pieces.first { $0.text == token }?.id
    }

    func convertIdToToken(_ id: Int) -> String? { Self.textByID[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

/// A deterministic tokenizer whose token IDs are UTF-8 bytes.
private struct ByteTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard token.utf8.count == 1 else { return nil }
        return token.utf8.first.map(Int.init)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard (0 ... 255).contains(id) else { return nil }
        return String(decoding: [UInt8(id)], as: UTF8.self)
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
