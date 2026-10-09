// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A cache type defined outside MLXLMCommon that prompt cache files can save and restore.
///
/// ``savePromptCache(url:cache:metadata:state:)`` writes ``promptCacheClassName`` with the
/// cache's `state` and `metaState`. To load the file, register the type first with
/// ``registerPromptCacheType(_:)``.
public protocol PromptCacheRestorable: KVCache {
    /// The class name written to prompt cache files. It must not be a built-in name such as
    /// `KVCacheSimple`, and it should be unique, for example by including a module prefix.
    static var promptCacheClassName: String { get }

    /// Rebuild a cache from the `state` and `metaState` it saved.
    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache
}

/// Register a cache type so ``loadPromptCacheSnapshot(url:)`` can restore it.
///
/// Registering the same type again has no effect.
///
/// - Throws: if the name is a built-in cache name, or another type already uses it.
public func registerPromptCacheType(_ type: any PromptCacheRestorable.Type) throws {
    try PromptCacheTypeRegistry.shared.register(type)
}

final class PromptCacheTypeRegistry: @unchecked Sendable {
    static let shared = PromptCacheTypeRegistry()

    /// Names that `restoreCacheFromMetaState` handles itself.
    static let builtInNames: Set<String> = [
        "KVCache", "KVCacheSimple", "RotatingKVCache", "QuantizedKVCache",
        "VarianceNormalizedKVCache", "ChunkedKVCache", "MambaCache", "ArraysCache",
        "TurboQuantKVCache", "CacheList",
    ]

    private let lock = NSLock()
    private var types: [String: any PromptCacheRestorable.Type] = [:]

    func register(_ type: any PromptCacheRestorable.Type) throws {
        let name = type.promptCacheClassName
        guard !Self.builtInNames.contains(name) else {
            throw KVCacheError(message: "\(name) is a built-in prompt cache class name")
        }
        try lock.withLock {
            if let existing = types[name] {
                guard ObjectIdentifier(existing) == ObjectIdentifier(type) else {
                    throw KVCacheError(
                        message: "Prompt cache class name \(name) is already registered")
                }
                return
            }
            types[name] = type
        }
    }

    func type(named name: String) -> (any PromptCacheRestorable.Type)? {
        lock.withLock { types[name] }
    }
}
