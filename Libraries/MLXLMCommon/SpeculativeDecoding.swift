// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private func defaultSpeculativeDecodingMemoryLimit() -> Int? {
    guard let bytes = GPU.maxRecommendedWorkingSetBytes(), bytes > 0 else {
        return nil
    }
    return bytes
}

/// Runtime counters for a speculative decoding pass.
public struct SpeculativeDecodingTelemetry: Sendable, Equatable {
    /// Number of speculative decoding rounds.
    public private(set) var roundCount: Int

    /// Number of tokens proposed by the draft model.
    public private(set) var draftTokenCount: Int

    /// Number of draft tokens accepted by the target model.
    public private(set) var acceptedDraftTokenCount: Int

    /// Number of target-model verification calls.
    public private(set) var targetModelCallCount: Int

    /// Number of draft-model calls.
    public private(set) var draftModelCallCount: Int

    /// Number of token positions evaluated by the target model during verification.
    public private(set) var targetVerifiedTokenCount: Int

    /// Number of tokens emitted from speculative rounds, including correction and bonus tokens.
    public private(set) var emittedTokenCount: Int

    public init(
        roundCount: Int = 0,
        draftTokenCount: Int = 0,
        acceptedDraftTokenCount: Int = 0,
        targetModelCallCount: Int = 0,
        draftModelCallCount: Int = 0,
        targetVerifiedTokenCount: Int = 0,
        emittedTokenCount: Int = 0
    ) {
        self.roundCount = roundCount
        self.draftTokenCount = draftTokenCount
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetModelCallCount = targetModelCallCount
        self.draftModelCallCount = draftModelCallCount
        self.targetVerifiedTokenCount = targetVerifiedTokenCount
        self.emittedTokenCount = emittedTokenCount
    }

    /// Number of draft tokens rejected by the target model.
    public var rejectedDraftTokenCount: Int {
        max(0, draftTokenCount - acceptedDraftTokenCount)
    }

    /// Fraction of drafted tokens accepted by the target model.
    public var acceptanceRate: Double {
        guard draftTokenCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(draftTokenCount)
    }

    /// Mean accepted draft tokens per speculative round.
    public var meanAcceptedDraftTokensPerRound: Double {
        guard roundCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(roundCount)
    }

    /// Mean emitted tokens per target-model verification call.
    public var meanEmittedTokensPerTargetCall: Double {
        guard targetModelCallCount > 0 else { return 0 }
        return Double(emittedTokenCount) / Double(targetModelCallCount)
    }

    package mutating func recordRound(
        drafted: Int,
        accepted: Int,
        targetVerified: Int,
        draftModelCalls: Int? = nil
    ) {
        roundCount += 1
        draftTokenCount += drafted
        acceptedDraftTokenCount += accepted
        targetModelCallCount += 1
        draftModelCallCount += draftModelCalls ?? drafted
        targetVerifiedTokenCount += targetVerified
    }

    package mutating func recordGeneratedToken() {
        emittedTokenCount += 1
    }

    package mutating func discardGeneratedToken() {
        emittedTokenCount = max(0, emittedTokenCount - 1)
    }
}

/// Action to take when speculative decoding exceeds a memory budget.
public enum SpeculativeDecodingMemoryAction: Sendable, Hashable {
    /// Use speculative decoding even if the estimate exceeds the budget.
    case allow

    /// Fall back to regular generation.
    case fallbackToDefault

    /// Throw an error instead of silently falling back.
    case fail
}

/// Result of evaluating speculative decoding against a memory policy.
public struct SpeculativeDecodingMemoryEvaluation: Sendable, Equatable {
    /// Estimated main-model parameter bytes.
    public let mainModelBytes: Int

    /// Estimated draft-model parameter bytes.
    public let draftModelBytes: Int

    /// Additional caller-provided budget for KV cache, workspace, or other resident data.
    public let additionalBytes: Int

    /// Total estimated resident bytes for speculative decoding.
    public var estimatedBytes: Int {
        mainModelBytes + draftModelBytes + additionalBytes
    }

    /// Memory budget used for the decision. `nil` means no budget was available.
    public let limitBytes: Int?

    /// Action selected by the policy.
    public let action: SpeculativeDecodingMemoryAction

    /// Whether speculative decoding is within the available budget.
    public var isWithinBudget: Bool {
        guard let limitBytes else { return true }
        return estimatedBytes <= limitBytes
    }

    /// Whether speculative decoding should be used.
    public var shouldUseSpeculativeDecoding: Bool {
        isWithinBudget || action == .allow
    }
}

/// Policy for gating auxiliary-model speculative decoding by resident memory estimates.
public struct SpeculativeDecodingMemoryPolicy: Sendable, Hashable {
    /// Optional absolute budget in bytes. When nil, no budget is enforced.
    public let limitBytes: Int?

    /// Extra bytes to reserve for KV cache, workspace, or application memory.
    public let additionalBytes: Int

    /// Action to take when the estimate exceeds the budget.
    public let action: SpeculativeDecodingMemoryAction

    public init(
        limitBytes: Int? = nil,
        additionalBytes: Int = 0,
        action: SpeculativeDecodingMemoryAction = .fallbackToDefault
    ) {
        self.limitBytes = limitBytes
        self.additionalBytes = max(0, additionalBytes)
        self.action = action
    }

    /// Default policy using `GPU.maxRecommendedWorkingSetBytes()` when available.
    public static var recommendedWorkingSet: Self {
        Self(limitBytes: defaultSpeculativeDecodingMemoryLimit())
    }

    /// Evaluate explicit byte estimates. This is useful before loading a draft model.
    public func evaluate(
        mainModelBytes: Int,
        draftModelBytes: Int
    ) -> SpeculativeDecodingMemoryEvaluation {
        SpeculativeDecodingMemoryEvaluation(
            mainModelBytes: max(0, mainModelBytes),
            draftModelBytes: max(0, draftModelBytes),
            additionalBytes: additionalBytes,
            limitBytes: limitBytes,
            action: action
        )
    }

    package func evaluate(
        mainModel: any LanguageModel,
        draftModel: any LanguageModel
    ) -> SpeculativeDecodingMemoryEvaluation {
        evaluate(
            mainModelBytes: Self.modelWeightBytes(mainModel),
            draftModelBytes: Self.modelWeightBytes(draftModel)
        )
    }

    package static func modelWeightBytes(_ model: any LanguageModel) -> Int {
        model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }
}

public struct SpeculativeDecodingMemoryError: Error, Sendable {
    public let evaluation: SpeculativeDecodingMemoryEvaluation

    public init(evaluation: SpeculativeDecodingMemoryEvaluation) {
        self.evaluation = evaluation
    }
}
