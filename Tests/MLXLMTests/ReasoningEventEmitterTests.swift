// Copyright © 2025 Apple Inc.

import Testing

@testable import MLXLMCommon

@Suite
struct ReasoningEventEmitterTests {

    // MARK: - Fixtures & helpers

    private static let thinkConfig = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)

    private typealias Segment = ReasoningEventEmitter.Segment

    /// Feeds all chunks through the emitter and appends `finalize()`.
    private func run(
        config: ReasoningConfig = thinkConfig, primedInside: Bool = false, _ chunks: [String]
    ) -> [Segment] {
        var emitter = ReasoningEventEmitter(config: config, primedInside: primedInside)
        var segments: [Segment] = []
        for chunk in chunks { segments += emitter.process(chunk) }
        segments += emitter.finalize()
        return segments
    }

    private func reasoningText(_ segments: [Segment]) -> String {
        segments.compactMap { if case .reasoning(let s) = $0 { return s } else { return nil } }
            .joined()
    }

    private func responseText(_ segments: [Segment]) -> String {
        segments.compactMap { if case .response(let s) = $0 { return s } else { return nil } }
            .joined()
    }

    private func leaksMarker(_ segments: [Segment]) -> Bool {
        segments.contains {
            let text: String
            switch $0 {
            case .reasoning(let s), .response(let s): text = s
            }
            return text.contains("<think>") || text.contains("</think>")
        }
    }

    // MARK: - Core routing

    @Test func cleanBlock() {
        let segments = run(["<think>abc</think>xyz"])
        #expect(segments == [.reasoning("abc"), .response("xyz")])
    }

    @Test func noDelimitersIsAllResponse() {
        let segments = run(["just a plain answer"])
        #expect(segments == [.response("just a plain answer")])
    }

    @Test func emptyBlockProducesNoReasoning() {
        let segments = run(["<think></think>hi"])
        #expect(segments == [.response("hi")])
    }

    @Test func multipleBlocksEachRoute() {
        let segments = run(["<think>a</think>mid<think>b</think>end"])
        #expect(
            segments == [
                .reasoning("a"), .response("mid"), .reasoning("b"), .response("end"),
            ])
    }

    // MARK: - Primed state

    @Test func primedClosesAndRoutesAnswer() {
        let segments = run(primedInside: true, ["reasoning</think>answer"])
        #expect(segments == [.reasoning("reasoning"), .response("answer")])
        #expect(!leaksMarker(segments))
    }

    @Test func primedNeverClosesFlushesAsReasoning() {
        let segments = run(primedInside: true, ["thinking forever, no close in sight"])
        #expect(segments == [.reasoning("thinking forever, no close in sight")])
        #expect(responseText(segments).isEmpty)
    }

    @Test func primedCloseSplitAcrossChunks() {
        let segments = run(primedInside: true, ["think</thi", "nk>ans"])
        #expect(reasoningText(segments) == "think")
        #expect(responseText(segments) == "ans")
        #expect(!leaksMarker(segments))
    }

    // MARK: - Split delimiters / chunk-boundary robustness

    @Test func openDelimiterSplitAcrossChunks() {
        let segments = run(["resp<thi", "nk>think</think>more"])
        #expect(reasoningText(segments) == "think")
        #expect(responseText(segments) == "respmore")
        #expect(!leaksMarker(segments))
    }

    @Test func bareLessThanSplit() {
        let segments = run(["ab", "<", "think>x</think>y"])
        #expect(reasoningText(segments) == "x")
        #expect(responseText(segments) == "aby")
        #expect(!leaksMarker(segments))
    }

    @Test func singleCharStress() {
        let chunks = "<think>hi</think>".map { String($0) }
        let segments = run(chunks)
        #expect(reasoningText(segments) == "hi")
        #expect(responseText(segments).isEmpty)
        #expect(!leaksMarker(segments))
    }

    @Test func almostMatchDoesNotTransition() {
        let segments = run(["<thinkers> unite"])
        #expect(segments == [.response("<thinkers> unite")])
    }

    // MARK: - Adversarial / locked behaviors

    @Test func nestedInnerThinkIsLiteralReasoning() {
        let segments = run(["<think>a<think>b</think>c"])
        #expect(segments == [.reasoning("a<think>b"), .response("c")])
    }

    @Test func strayCloseWhenNeverOpenedIsLiteralResponse() {
        let segments = run(["answer </think> more"])
        #expect(segments == [.response("answer </think> more")])
    }

    /// Documented v1 limitation: a literal `<think>` in answer text misroutes to
    /// reasoning. The deferred token-ID detection is the real fix.
    @Test func thinkInAnswerTextIsMisrouted_documentedLimitation() {
        let segments = run(["Use the <think> tag in HTML"])
        #expect(reasoningText(segments) == "tag in HTML")
        #expect(responseText(segments) == "Use the")
    }

    // MARK: - Whitespace trimming (mirrors unwrapToolCallMarkers)

    @Test func trimsTemplateWhitespaceAroundMarkers() {
        let segments = run(["<think>\nthought\n</think>\n\nAnswer"])
        #expect(segments == [.reasoning("thought"), .response("Answer")])
    }

    @Test func trimsResponseLeadingWhitespaceAcrossChunks() {
        let segments = run(["<think>t</think>", "\n\nAnswer"])
        #expect(reasoningText(segments) == "t")
        #expect(responseText(segments) == "Answer")
    }

    // MARK: - Custom delimiters (registry-extensible families)

    @Test func customDelimitersRouteIndependently() {
        let kimiStyle = ReasoningConfig(
            startDelimiter: "◁think▷", endDelimiter: "◁/think▷", promptStrategy: .alwaysOn)
        let segments = run(config: kimiStyle, ["◁think▷pondering◁/think▷done"])
        #expect(segments == [.reasoning("pondering"), .response("done")])
        // A standard <think> is inert text for this config.
        let segments2 = run(config: kimiStyle, ["<think>not reasoning</think>"])
        #expect(segments2 == [.response("<think>not reasoning</think>")])
    }

    // MARK: - Reasoning emission gating

    @Test func emitsReasoningOnlyWhenDelimited() {
        let withReasoning = run(["<think>x</think>y"])
        #expect(withReasoning.contains(.reasoning("x")))

        let without = run(["plain answer"])
        let hasReasoning = without.contains {
            if case .reasoning = $0 { return true } else { return false }
        }
        #expect(!hasReasoning)
    }

    // MARK: - promptEndsInsideReasoning (prefill seeding)

    /// The killer case: Qwen3/R1 templates append `<think>\n` — a strict
    /// `hasSuffix("<think>")` returns false here and misroutes 100% of reasoning.
    @Test func prefillWithTrailingNewlineIsDetected() {
        let tail = "<|im_start|>assistant\n<think>\n"
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: tail, config: Self.thinkConfig))
    }

    @Test func prefillWithoutTrailingWhitespaceIsDetected() {
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "assistant\n<think>", config: Self.thinkConfig))
    }

    @Test func noPrefillIsNotDetected() {
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<|im_start|>assistant\n", config: Self.thinkConfig))
    }

    @Test func closedBlockInPromptIsNotInside() {
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<think>cached</think>\nanswer", config: Self.thinkConfig))
    }

    @Test func prefillWithMultipleTrailingNewlinesAndSpaces() {
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<think>\n\n  ", config: Self.thinkConfig))
    }

    @Test func customDelimiterPrefillDetected() {
        let kimi = ReasoningConfig(
            startDelimiter: "◁think▷", endDelimiter: "◁/think▷", promptStrategy: .alwaysOn)
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "assistant\n◁think▷\n", config: kimi))
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "assistant\n", config: kimi))
    }

    // MARK: - hasClosedReasoning (latching close signal for token collectors)

    @Test func hasClosedReasoningLatchesOnClose() {
        var e = ReasoningEventEmitter(config: Self.thinkConfig, primedInside: false)
        #expect(!e.hasClosedReasoning)
        _ = e.process("<think>abc")
        #expect(!e.hasClosedReasoning)  // opened but not yet closed
        _ = e.process("</think>xyz")
        #expect(e.hasClosedReasoning)
    }

    /// The case ``isInsideReasoning`` cannot report: an empty block opens and
    /// closes within one `process` call, so `inside` reads false before and after.
    @Test func hasClosedReasoningDetectsEmptyBlockInOneChunk() {
        var e = ReasoningEventEmitter(config: Self.thinkConfig, primedInside: false)
        _ = e.process("<think></think>hi")
        #expect(e.hasClosedReasoning)
        #expect(!e.isInsideReasoning)
    }

    @Test func hasClosedReasoningFiresForPrimedClose() {
        var e = ReasoningEventEmitter(config: Self.thinkConfig, primedInside: true)
        #expect(!e.hasClosedReasoning)
        _ = e.process("thinking</think>answer")
        #expect(e.hasClosedReasoning)
    }
}
