// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

/// Host-testable unit coverage for ``ReasoningTokenCollector`` — the pure core of
/// think-then-call Phase 1. Uses a deterministic map tokenizer so the exact
/// token→text boundaries (and the `<think>`/`</think>` split/empty cases) are
/// pinned, with no model or device.
@Suite
struct ReasoningTokenCollectorTests {

    private typealias Segment = ReasoningEventEmitter.Segment

    private static let thinkConfig = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)

    /// Deterministic id→string tokenizer. `decode` concatenates the mapped
    /// strings with no separator, so callers control the decoded stream exactly.
    private struct MapTokenizer: Tokenizer {
        let map: [Int: String]
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { map[$0] ?? "" }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? {
            map.first { $0.value == token }?.key
        }
        func convertIdToToken(_ id: Int) -> String? { map[id] }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private static let vocab: [Int: String] = [
        1: "<think>", 2: "reason", 3: "ing", 4: "</think>",
        5: "ans", 6: "wer", 7: "Sure",
        12: "</thi", 13: "nk>",  // split closing delimiter
        20: "<think></think>",  // empty block in a single token
        30: "a", 31: "b", 32: "c",
        40: "\nthought\n",  // template-style newlines
    ]
    private static let tok = MapTokenizer(map: vocab)

    /// Feeds tokens until the collector signals stop (mirroring Phase 1's break),
    /// else consumes all and finalizes. Returns routed segments, whether it
    /// stopped, and the accumulated token IDs.
    private func drive(primedInside: Bool = false, _ tokens: [Int])
        -> (segments: [Segment], stopped: Bool, tokenIDs: [Int])
    {
        var collector = ReasoningTokenCollector(
            config: Self.thinkConfig, primedInside: primedInside, tokenizer: Self.tok)
        var segments: [Segment] = []
        var stopped = false
        for token in tokens {
            segments += collector.ingest(token)
            if collector.shouldStopAfterReasoning {
                stopped = true
                break
            }
        }
        if !stopped { segments += collector.finalize() }
        return (segments, stopped, collector.reasoningTokenIDs)
    }

    private func reasoningText(_ s: [Segment]) -> String {
        s.compactMap { if case .reasoning(let t) = $0 { return t } else { return nil } }.joined()
    }
    private func responseText(_ s: [Segment]) -> String {
        s.compactMap { if case .response(let t) = $0 { return t } else { return nil } }.joined()
    }
    private func leaksMarker(_ s: [Segment]) -> Bool {
        s.contains {
            let t: String
            switch $0 {
            case .reasoning(let x), .response(let x): t = x
            }
            return t.contains("<think>") || t.contains("</think>")
        }
    }

    // MARK: - Core hand-off

    /// Non-primed (Qwen3-style): the opening `<think>` is generated, so it is
    /// captured; accumulation ends at the closing `</think>` token, and the
    /// answer tokens after it are never ingested.
    @Test func nonPrimedCapturesOpeningThroughClose() {
        let (segs, stopped, ids) = drive([1, 2, 3, 4, 5, 6])
        #expect(stopped)
        #expect(ids == [1, 2, 3, 4])  // <think> reason ing </think> — no answer tokens
        #expect(reasoningText(segs) == "reasoning")
        #expect(responseText(segs).isEmpty)
        #expect(!leaksMarker(segs))
    }

    /// Primed (R1-style): the opening `<think>` lives in the prompt, so the first
    /// generated token is already reasoning; IDs run from there through `</think>`.
    @Test func primedAccumulatesFromFirstReasoningToken() {
        let (segs, stopped, ids) = drive(primedInside: true, [2, 3, 4, 5, 6])
        #expect(stopped)
        #expect(ids == [2, 3, 4])
        #expect(reasoningText(segs) == "reasoning")
        #expect(!leaksMarker(segs))
    }

    // MARK: - The case `isInsideReasoning` alone cannot catch

    /// Empty `<think></think>` resolving inside one decoded chunk: `isInsideReasoning`
    /// reads false before AND after, but `hasClosedReasoning` latches, so Phase 1
    /// correctly stops and hands off to the constrained phase.
    @Test func emptyBlockInOneChunkStillStops() {
        let (segs, stopped, ids) = drive([20, 5, 6])
        #expect(stopped)
        #expect(ids == [20])
        #expect(reasoningText(segs).isEmpty)  // no reasoning content
        #expect(!leaksMarker(segs))
    }

    // MARK: - Robustness

    /// A `</think>` split across two tokens closes only once the full delimiter
    /// arrives — the collector stops on the second token, not the first.
    @Test func splitClosingDelimiterStopsOnCompletion() {
        let (segs, stopped, ids) = drive([1, 2, 12, 13, 5])
        #expect(stopped)
        #expect(ids == [1, 2, 12, 13])  // stopped on 13, not 12
        #expect(reasoningText(segs) == "reason")
        #expect(!leaksMarker(segs))
    }

    /// Never-opened (a reasoning-capable model that just answers): no close ever
    /// fires, so Phase 1 does not stop early (the caller bounds it by maxTokens).
    @Test func neverOpenedDoesNotStop() {
        let (segs, stopped, ids) = drive([5, 6])
        #expect(!stopped)
        #expect(ids == [5, 6])
        #expect(reasoningText(segs).isEmpty)
        #expect(responseText(segs) == "answer")
    }

    /// The stop latches on the FIRST close; a second `<think>…</think>` block is
    /// never reached because the caller has already broken to Phase 2.
    @Test func stopsOnFirstCloseNotReopen() {
        let (segs, stopped, ids) = drive([1, 30, 4, 31, 1, 32, 4])
        #expect(stopped)
        #expect(ids == [1, 30, 4])  // stopped at first </think>; second block never ingested
        #expect(reasoningText(segs) == "a")
        #expect(!leaksMarker(segs))
    }

    /// Template-style newlines (`<think>\n…\n</think>`): the detokenizer's
    /// newline segmenting and the emitter's whitespace trimming compose without
    /// leaking markers; the close is still detected on the `</think>` token.
    @Test func handlesTemplateNewlines() {
        let (segs, stopped, ids) = drive([1, 40, 4, 5])
        #expect(stopped)
        #expect(ids == [1, 40, 4])
        #expect(reasoningText(segs).contains("thought"))
        #expect(!leaksMarker(segs))
    }
}
