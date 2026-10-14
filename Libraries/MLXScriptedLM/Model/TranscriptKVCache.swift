// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// The tokens a model has processed, kept as an entry in its KV cache.
///
/// A scripted model appends this entry to its inner model's caches and keeps it in step with
/// them. Generation trims, rewinds, copies, and saves the whole cache together, so `tokens`
/// always matches what the inner caches hold, across speculative rejections, prompt-cache
/// reuse, and `ChatSession` rebuilds.
///
/// It is an `AuxiliaryKVCache`, so capacity and compression settings ignore it, and it never
/// stores keys or values.
///
/// Note: this KVCache should not be used for any production use.
/// Its performance is O(n^2) or worse and is strictly meant as
/// a mechanism that a `ScriptedModel` could use.
public final class TranscriptKVCache: BaseKVCache, AuxiliaryKVCache {

    public override static var serializedClassName: String? { "MLXScriptedLM.TranscriptKVCache" }

    /// Registers the type for prompt cache files once, on first use.
    private static let registration: Void = {
        try! registerKVCacheType(TranscriptKVCache.self)
    }()

    /// Register the type for KVCache save and restore.
    /// This will happen automatically if a cache instance is
    /// created.
    public static func register() {
        _ = Self.registration
    }

    public private(set) var tokens: [Int]

    public init(tokens: [Int] = []) {
        _ = Self.registration
        self.tokens = tokens
        super.init()
        offset = tokens.count
    }

    /// Records tokens the inner model has just processed.
    public func append(_ newTokens: [Int]) {
        tokens += newTokens
        offset = tokens.count
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("TranscriptKVCache records tokens; it does not store keys and values")
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = Swift.min(Swift.max(n, 0), tokens.count)
        tokens.removeLast(trimmed)
        offset = tokens.count
        return trimmed
    }

    public override func copy() -> any KVCache {
        TranscriptKVCache(tokens: tokens)
    }

    // MARK: - Prompt cache files

    public override var state: [MLXArray] {
        get { tokens.isEmpty ? [] : [MLXArray(tokens.map(Int32.init))] }
        set {
            tokens = newValue.first?.asArray(Int32.self).map(Int.init) ?? []
            offset = tokens.count
        }
    }

    public override var metaState: [String] {
        get { [String(tokens.count)] }
        set {}
    }

    public override static func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        let cache = TranscriptKVCache()
        cache.state = state
        guard metaState == [String(cache.tokens.count)] else {
            throw TranscriptKVCacheError.countMismatch(
                stored: metaState.first ?? "", restored: cache.tokens.count)
        }
        return cache
    }
}

public enum TranscriptKVCacheError: LocalizedError, Equatable {
    case countMismatch(stored: String, restored: Int)

    public var errorDescription: String? {
        switch self {
        case .countMismatch(let stored, let restored):
            "TranscriptKVCache file says \(stored) tokens but holds \(restored)"
        }
    }
}
