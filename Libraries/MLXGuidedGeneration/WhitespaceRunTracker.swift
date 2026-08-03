// Copyright © 2026 Apple Inc.

/// Tracks consecutive whitespace-only sampled tokens and signals when
/// suppression should activate.
///
/// Once the consecutive whitespace count reaches `threshold`, suppression
/// latches on permanently for this generation run. A model that hits the
/// threshold has demonstrated pathological whitespace preference; resetting
/// would let it cycle between whitespace runs and forced structural tokens,
/// wasting the token budget.
public struct WhitespaceRunTracker {

    // MARK: - Private State

    private let threshold: Int
    private let whitespaceTokenIDs: Set<Int>
    private var consecutiveCount: Int = 0
    private var activated: Bool = false

    // MARK: - Public API

    /// Creates a tracker with the given threshold and whitespace token IDs.
    ///
    /// - Parameters:
    ///   - threshold: Number of consecutive whitespace tokens before suppression activates.
    ///   - whitespaceTokenIDs: Set of token IDs classified as whitespace-only.
    public init(threshold: Int = 3, whitespaceTokenIDs: Set<Int>) {
        self.threshold = threshold
        self.whitespaceTokenIDs = whitespaceTokenIDs
    }

    /// Whether suppression is currently active. Once activated, stays active
    /// for the remainder of the generation run (latch behavior).
    public var isActive: Bool { activated || consecutiveCount >= threshold }

    /// Records a sampled token and returns whether suppression should be active
    /// for the next sampling step.
    public mutating func record(tokenID: Int) -> Bool {
        if whitespaceTokenIDs.contains(tokenID) {
            consecutiveCount += 1
        } else {
            consecutiveCount = 0
        }
        if consecutiveCount >= threshold {
            activated = true
        }
        return isActive
    }
}
