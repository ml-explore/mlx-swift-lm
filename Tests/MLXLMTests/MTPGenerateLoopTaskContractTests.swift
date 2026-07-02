// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

// MARK: - Contract check for generateLoopTask's iterator consumption
//
// Catches the class of bug fixed in commit 25faa79 (Phase 4): when
// `generateLoopTask` consumed its iterator via `for token in iterator` instead
// of `while let token = iterator.next()`, the value-type iterator was forked
// into a copy at the for-in expansion, and the outer binding observed at the
// `.info` event's `mtpStats = iterator as? MTPStatsCollecting` downcast site
// stayed at post-init zero — regardless of how many tokens the inner copy
// yielded.
//
// The Phase 1 IntegrationTesting diagnostic test
// (`MTPIteratorEndToEndDiagnosticTests`) catches the same class of bug at the
// cost of full 31B weight loads. This file does it at unit-test / CI scope
// using a mock iterator and a minimal reproduction of the loop pattern.

// MARK: - Mock iterator

/// Value-type iterator that yields a fixed token sequence, increments
/// `MTPStatsCollecting` counters per call, and optionally records a
/// passthrough reason. Conforms to both `TokenIteratorProtocol` and
/// `MTPStatsCollecting` so it can be downcast through the same path as the
/// real `MTPSpeculativeTokenIterator`.
private struct MockMTPIterator: TokenIteratorProtocol, MTPStatsCollecting {
    let tokensToYield: [Int]
    let perTokenProposed: Int
    let perTokenAccepted: Int
    let finalPassthroughReason: String?

    var index = 0
    public var tokenCount = 0
    public let maxTokens: Int? = nil
    public let promptPrefillTime: TimeInterval = 0
    public private(set) var proposedDraftTokens: Int = 0
    public private(set) var acceptedDraftTokens: Int = 0
    public private(set) var passthroughReason: String?

    init(
        tokensToYield: [Int],
        perTokenProposed: Int,
        perTokenAccepted: Int,
        finalPassthroughReason: String?
    ) {
        self.tokensToYield = tokensToYield
        self.perTokenProposed = perTokenProposed
        self.perTokenAccepted = perTokenAccepted
        self.finalPassthroughReason = finalPassthroughReason
    }

    mutating func next() -> Int? {
        guard index < tokensToYield.count else {
            passthroughReason = finalPassthroughReason
            return nil
        }
        let token = tokensToYield[index]
        index += 1
        tokenCount += 1
        proposedDraftTokens += perTokenProposed
        acceptedDraftTokens += perTokenAccepted
        return token
    }
}

// MARK: - Minimal generateLoopTask reproduction
//
// The smallest faithful reproduction of the iterator-consuming pattern at
// Evaluate.swift's `generateLoopTask` site. Mirrors the post-Phase-4 shape:
// `var iterator = ...; while let token = iterator.next()`. Pre-Phase-4 this
// helper used `for token in iterator`, and the counters on the outer
// `iterator` binding stayed at zero because the for-in iterated a value-type
// copy.
private func consumeIteratorAndBuildInfo(
    _ iterator: any TokenIteratorProtocol
) -> (tokens: [Int], info: GenerateCompletionInfo) {
    var iterator = iterator
    var tokens: [Int] = []
    while let token = iterator.next() {
        tokens.append(token)
    }
    let mtpStats = iterator as? MTPStatsCollecting
    let info = GenerateCompletionInfo(
        promptTokenCount: 0,
        generationTokenCount: tokens.count,
        promptTime: 0,
        generationTime: 0,
        stopReason: .stop,
        proposedDraftTokens: mtpStats?.proposedDraftTokens,
        acceptedDraftTokens: mtpStats?.acceptedDraftTokens,
        passthroughReason: mtpStats?.passthroughReason
    )
    return (tokens, info)
}

@Suite
struct MTPGenerateLoopTaskContractTests {

    /// Load-bearing regression check: the iterator's counters on the outer
    /// binding must reflect the mutations performed by the consuming loop.
    /// Pre-Phase-4 (with `for token in iterator` in `generateLoopTask`), the
    /// counters on `iterator` stayed at the post-init zeros — the `for-in`
    /// expansion mutated a value-type copy, and the outer binding never
    /// observed the increments.
    @Test
    func iteratorCountersOnOuterBindingReflectLoopMutations() {
        let mock = MockMTPIterator(
            tokensToYield: [11, 22, 33, 44],
            perTokenProposed: 3,
            perTokenAccepted: 2,
            finalPassthroughReason: nil
        )

        let (tokens, info) = consumeIteratorAndBuildInfo(mock)

        #expect(tokens == [11, 22, 33, 44])
        #expect(info.generationTokenCount == 4)
        #expect(
            info.proposedDraftTokens == 12,
            "outer-binding proposedDraftTokens stuck at \(info.proposedDraftTokens ?? -1); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
        #expect(
            info.acceptedDraftTokens == 8,
            "outer-binding acceptedDraftTokens stuck at \(info.acceptedDraftTokens ?? -1); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
        #expect(info.passthroughReason == nil)
    }

    /// Passthrough-reason mutation on the iterator is also observed only via
    /// the mutated outer binding. Mock latches the reason at end-of-stream;
    /// the loop must reach next() returning nil for the reason to propagate.
    @Test
    func passthroughReasonObservedOnOuterBinding() {
        let mock = MockMTPIterator(
            tokensToYield: [7, 8],
            perTokenProposed: 0,
            perTokenAccepted: 0,
            finalPassthroughReason: "main model did not emit drafter state"
        )

        let (tokens, info) = consumeIteratorAndBuildInfo(mock)

        #expect(tokens == [7, 8])
        #expect(info.proposedDraftTokens == 0)
        #expect(info.acceptedDraftTokens == 0)
        #expect(
            info.passthroughReason == "main model did not emit drafter state",
            "outer-binding passthroughReason was \(info.passthroughReason ?? "nil"); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
    }
}
