// Copyright © 2026 Apple Inc.

import MLXLMCommon
import XCTest

/// These tests intentionally stay policy-math only.
///
/// Rationale:
/// - `mlx-swift` already covers manager/ticket event-stream behavior in depth.
/// - `mlx-swift-lm` should validate only the policy layer it adds on top.
/// - Keeping this target model-free avoids large downloads so tests run in CI.
final class WiredMemoryPolicyTests: XCTestCase {
    func testWiredSumPolicyCapAffectsLimitAndAdmission() {
        let policy = WiredSumPolicy(cap: 200)

        XCTAssertEqual(policy.limit(baseline: 100, activeSizes: [50, 100]), 200)
        XCTAssertTrue(policy.canAdmit(baseline: 100, activeSizes: [50], newSize: 50))
        XCTAssertFalse(policy.canAdmit(baseline: 100, activeSizes: [50], newSize: 51))
    }

    func testWiredMaxPolicyReturnsLargestDemandOrBaseline() {
        let policy = WiredMaxPolicy()

        XCTAssertEqual(policy.limit(baseline: 100, activeSizes: [20, 150, 60]), 150)
        XCTAssertEqual(policy.limit(baseline: 200, activeSizes: [20, 150, 60]), 200)
    }

    func testWiredFixedPolicyIgnoresActiveSizes() {
        let policy = WiredFixedPolicy(limit: 123)

        XCTAssertEqual(policy.limit(baseline: 0, activeSizes: []), 123)
        XCTAssertEqual(policy.limit(baseline: 500, activeSizes: [1, 2, 3]), 123)
    }

    func testWiredBudgetPolicyIdentityAndCapBehavior() {
        let sharedID = UUID()
        let first = WiredBudgetPolicy(baseBytes: 100, cap: 300, id: sharedID)
        let second = WiredBudgetPolicy(baseBytes: 999, cap: 999, id: sharedID)
        let third = WiredBudgetPolicy(baseBytes: 100, cap: 300, id: UUID())

        XCTAssertEqual(first, second)
        XCTAssertNotEqual(first, third)
        XCTAssertEqual(first.limit(baseline: 50, activeSizes: [75]), 225)
        XCTAssertTrue(first.canAdmit(baseline: 50, activeSizes: [75], newSize: 75))
        XCTAssertFalse(first.canAdmit(baseline: 50, activeSizes: [75], newSize: 76))
    }

    func testSpeculativeDecodingMemoryPolicyEvaluatesCombinedModelBudget() {
        let policy = SpeculativeDecodingMemoryPolicy(
            limitBytes: 1_000,
            additionalBytes: 100,
            action: .fallbackToDefault
        )

        let admitted = policy.evaluate(mainModelBytes: 600, draftModelBytes: 250)
        XCTAssertEqual(admitted.estimatedBytes, 950)
        XCTAssertTrue(admitted.isWithinBudget)
        XCTAssertTrue(admitted.shouldUseSpeculativeDecoding)

        let denied = policy.evaluate(mainModelBytes: 600, draftModelBytes: 350)
        XCTAssertEqual(denied.estimatedBytes, 1_050)
        XCTAssertFalse(denied.isWithinBudget)
        XCTAssertFalse(denied.shouldUseSpeculativeDecoding)
    }

    func testSpeculativeDecodingMemoryPolicyAllowOverridesBudget() {
        let policy = SpeculativeDecodingMemoryPolicy(limitBytes: 100, action: .allow)
        let evaluation = policy.evaluate(mainModelBytes: 100, draftModelBytes: 1)

        XCTAssertFalse(evaluation.isWithinBudget)
        XCTAssertTrue(evaluation.shouldUseSpeculativeDecoding)
    }
}
