// Copyright © 2025 Apple Inc.

import MLXLMCommon
import Testing

@Suite
struct WhitespaceRunTrackerTests {

    @Test
    func belowThresholdReturnsFalseAndIsNotActive() {
        let whitespaceIDs: Set<Int> = [1, 2, 3]
        var tracker = WhitespaceRunTracker(threshold: 3, whitespaceTokenIDs: whitespaceIDs)

        // Initially not active
        #expect(tracker.isActive == false)

        // Record 1 whitespace token (below threshold of 3)
        let result1 = tracker.record(tokenID: 1)
        #expect(result1 == false)
        #expect(tracker.isActive == false)

        // Record 2nd whitespace token (still below threshold)
        let result2 = tracker.record(tokenID: 2)
        #expect(result2 == false)
        #expect(tracker.isActive == false)
    }

    @Test
    func exactlyThresholdWhitespaceTokensActivatesSuppression() {
        let whitespaceIDs: Set<Int> = [10, 20, 30]
        var tracker = WhitespaceRunTracker(threshold: 3, whitespaceTokenIDs: whitespaceIDs)

        // Record 3 consecutive whitespace tokens (exactly threshold)
        _ = tracker.record(tokenID: 10)
        _ = tracker.record(tokenID: 20)
        let result3 = tracker.record(tokenID: 30)

        #expect(result3 == true)
        #expect(tracker.isActive == true)
    }

    @Test
    func latchesPermanentlyAfterActivation() {
        let whitespaceIDs: Set<Int> = [10, 20, 30]
        var tracker = WhitespaceRunTracker(threshold: 3, whitespaceTokenIDs: whitespaceIDs)

        // Build up to threshold
        _ = tracker.record(tokenID: 10)
        _ = tracker.record(tokenID: 20)
        _ = tracker.record(tokenID: 30)
        #expect(tracker.isActive == true)

        // Non-whitespace token resets consecutive counter but suppression stays latched
        let result = tracker.record(tokenID: 99)
        #expect(result == true)
        #expect(tracker.isActive == true)

        // Remains active even after many non-whitespace tokens
        _ = tracker.record(tokenID: 100)
        _ = tracker.record(tokenID: 101)
        #expect(tracker.isActive == true)
    }

    @Test
    func thresholdZeroIsActiveFromInitialization() {
        let whitespaceIDs: Set<Int> = [10]
        var tracker = WhitespaceRunTracker(threshold: 0, whitespaceTokenIDs: whitespaceIDs)

        // Active immediately (0 >= 0)
        #expect(tracker.isActive == true)

        // record returns true for whitespace token
        let wsResult = tracker.record(tokenID: 10)
        #expect(wsResult == true)

        // record returns true even for non-whitespace token
        let nonWsResult = tracker.record(tokenID: 99)
        #expect(nonWsResult == true)
        #expect(tracker.isActive == true)
    }

    @Test
    func thresholdOneActivatesAfterSingleWhitespaceToken() {
        let whitespaceIDs: Set<Int> = [5]
        var tracker = WhitespaceRunTracker(threshold: 1, whitespaceTokenIDs: whitespaceIDs)

        // Not active initially (0 < 1)
        #expect(tracker.isActive == false)

        // Single whitespace token activates
        let result = tracker.record(tokenID: 5)
        #expect(result == true)
        #expect(tracker.isActive == true)
    }

    @Test
    func consecutiveNonWhitespaceTokensKeepIsActiveFalse() {
        let whitespaceIDs: Set<Int> = [1, 2]
        var tracker = WhitespaceRunTracker(threshold: 2, whitespaceTokenIDs: whitespaceIDs)

        // Many non-whitespace tokens in a row
        for tokenID in [50, 51, 52, 53, 54] {
            let result = tracker.record(tokenID: tokenID)
            #expect(result == false)
            #expect(tracker.isActive == false)
        }
    }
}
