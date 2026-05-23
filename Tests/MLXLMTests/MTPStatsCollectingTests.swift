// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

// MARK: - Surface check
//
// These tests cover the Phase 0 observability plumbing for MTP speculative
// decoding: the `MTPStatsCollecting` protocol, the three new optional fields
// on `GenerateCompletionInfo`, and the downcast pattern that
// `generateLoopTask` uses to thread iterator counters through the emitted
// `.info` event. End-to-end exercise of the plumbing against a real iterator
// lives in `MTPIteratorEndToEndDiagnosticTests` (IntegrationTesting,
// checkpoint-gated).

// MARK: - Mock conformer

private struct MockMTPStats: MTPStatsCollecting {
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
    let passthroughReason: String?
}

@Suite
struct MTPStatsCollectingTests {

    // (A) A mock `MTPStatsCollecting` conformer can be downcast from an
    // existential `Any` — the same pattern `generateLoopTask` uses to pull
    // stats off the iterator value — and its fields populate
    // `GenerateCompletionInfo` correctly when threaded through the init.
    @Test
    func mockConformerPopulatesInfoFields() {
        let mock = MockMTPStats(
            proposedDraftTokens: 21,
            acceptedDraftTokens: 7,
            passthroughReason: nil
        )

        // Mirror the downcast pattern at Evaluate.swift's info construction
        // site: `iterator as? MTPStatsCollecting`.
        let asAny: Any = mock
        let stats = asAny as? MTPStatsCollecting

        #expect(stats != nil)

        let info = GenerateCompletionInfo(
            promptTokenCount: 10,
            generationTokenCount: 32,
            promptTime: 0.5,
            generationTime: 1.5,
            stopReason: .length,
            proposedDraftTokens: stats?.proposedDraftTokens,
            acceptedDraftTokens: stats?.acceptedDraftTokens,
            passthroughReason: stats?.passthroughReason
        )

        #expect(info.proposedDraftTokens == 21)
        #expect(info.acceptedDraftTokens == 7)
        #expect(info.passthroughReason == nil)
    }

    // Passthrough engagement: a conformer that returns a non-nil reason
    // threads through cleanly. This is the "iterator latched into
    // passthrough" observation.
    @Test
    func passthroughReasonThreadsThrough() {
        let mock = MockMTPStats(
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: "main model did not emit drafter state"
        )

        let info = GenerateCompletionInfo(
            promptTokenCount: 4,
            generationTokenCount: 16,
            promptTime: 0.1,
            generationTime: 0.2,
            stopReason: .length,
            proposedDraftTokens: mock.proposedDraftTokens,
            acceptedDraftTokens: mock.acceptedDraftTokens,
            passthroughReason: mock.passthroughReason
        )

        #expect(info.proposedDraftTokens == 0)
        #expect(info.acceptedDraftTokens == 0)
        #expect(info.passthroughReason == "main model did not emit drafter state")
    }

    // (B) A non-MTP iterator does not conform to `MTPStatsCollecting`; the
    // downcast returns nil and the three new fields default to nil on the
    // emitted info.
    @Test
    func nonMTPIteratorDowncastReturnsNil() {
        let nonMTP: Any = "I am not an iterator"
        let stats = nonMTP as? MTPStatsCollecting

        #expect(stats == nil)

        let info = GenerateCompletionInfo(
            promptTokenCount: 10,
            generationTokenCount: 32,
            promptTime: 0.5,
            generationTime: 1.5,
            stopReason: .length,
            proposedDraftTokens: stats?.proposedDraftTokens,
            acceptedDraftTokens: stats?.acceptedDraftTokens,
            passthroughReason: stats?.passthroughReason
        )

        #expect(info.proposedDraftTokens == nil)
        #expect(info.acceptedDraftTokens == nil)
        #expect(info.passthroughReason == nil)
    }

    // (C) The pre-Phase-0 public initializer signature still compiles and
    // populates the three new fields with their nil defaults. This is the
    // source-compatibility assertion: existing call sites that don't supply
    // the new parameters continue to work unchanged.
    @Test
    func legacyInitializerIsSourceCompatible() {
        let info = GenerateCompletionInfo(
            promptTokenCount: 10,
            generationTokenCount: 32,
            promptTime: 0.5,
            generationTime: 1.5,
            stopReason: .stop
        )

        #expect(info.proposedDraftTokens == nil)
        #expect(info.acceptedDraftTokens == nil)
        #expect(info.passthroughReason == nil)
        #expect(info.stopReason == .stop)
    }

    // The `MTPSpeculativeTokenIterator` itself conforms to
    // `MTPStatsCollecting`; an instance can be downcast from `Any` and its
    // initial counters are zero / nil before any rounds run.
    @Test
    func iteratorConformsToProtocol() {
        // Type-only conformance check (no instance required): if the
        // iterator stops conforming, this fails to compile.
        let _: any MTPStatsCollecting.Type = MTPSpeculativeTokenIterator.self
    }
}
