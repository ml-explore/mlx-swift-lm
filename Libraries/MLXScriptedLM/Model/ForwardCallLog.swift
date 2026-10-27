// Copyright © 2026 Apple Inc.

import Foundation

/// One call into a ``ScriptedLogitsModel``.
public struct ForwardCall: Sendable, Equatable {

    public enum Kind: Sendable, Equatable {
        /// `prepare(_:cache:state:prefill:)`
        case prepare
        /// `callAsFunction(_:cache:state:)`
        case forward
        /// `nextTokenLogits(_:cache:state:)`
        case nextTokenLogits
    }

    /// Input 0 differed from the token the script called for at that position.
    public struct InputMismatch: Sendable, Equatable {
        public let position: Int
        public let expected: Int
        public let actual: Int
    }

    public let kind: Kind

    /// The absolute position of input 0: the transcript length before the call.
    public let cacheOffsetBefore: Int
    public let inputTokens: [Int]

    /// Tokens the call added to the cache. Less than `inputTokens.count` when `prepare`
    /// leaves a remainder for the iterator.
    public let processedTokenCount: Int

    /// The boosted token for each returned logits row, or `nil` where the row was left as is.
    public let scriptedTokens: [Int?]

    /// The cursor for the last returned row, if it is inside an assistant turn.
    public let cursor: TurnScripts.Cursor?
    public let inputMismatch: InputMismatch?
    public let maskPresent: Bool
    public let hasInboundState: Bool
}

/// The calls a ``ScriptedLogitsModel`` has received, safe to read from any thread.
public final class ForwardCallLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [ForwardCall] = []

    public init() {}

    public var calls: [ForwardCall] {
        lock.withLock { storage }
    }

    public var inputMismatches: [ForwardCall.InputMismatch] {
        calls.compactMap(\.inputMismatch)
    }

    public func removeAll() {
        lock.withLock { storage.removeAll() }
    }

    func append(_ call: ForwardCall) {
        lock.withLock { storage.append(call) }
    }
}
