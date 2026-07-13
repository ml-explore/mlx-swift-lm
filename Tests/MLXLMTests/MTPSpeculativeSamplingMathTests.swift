// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Test
func testSpeculativeAcceptanceProbabilityBoundaries() {
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: Foundation.log(0.6),
            draftLogProbability: Foundation.log(0.3)) == 1)

    let ratio = speculativeAcceptanceProbability(
        targetLogProbability: Foundation.log(0.2),
        draftLogProbability: Foundation.log(0.5))
    #expect(abs(ratio - 0.4) < 1e-6)

    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: -.infinity,
            draftLogProbability: Foundation.log(0.5)) == 0)
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: Foundation.log(0.5),
            draftLogProbability: -.infinity) == 0)
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: .nan,
            draftLogProbability: Foundation.log(0.5)) == 0)
}
