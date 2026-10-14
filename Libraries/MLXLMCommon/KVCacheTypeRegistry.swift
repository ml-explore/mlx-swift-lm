// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Register a cache type so ``loadPromptCacheSnapshot(url:)`` can restore it.
///
/// ``savePromptCache(url:cache:metadata:state:)`` registers the types it writes, so a file saved
/// and loaded in one process works without this call. Call it before loading in a new process.
/// Registering the same type again has no effect.
///
/// - Throws: if the type has no ``KVCache/serializedClassName``, or another type already uses it.
public func registerKVCacheType<T: KVCache>(_ type: T.Type) throws {
    try KVCacheTypeRegistry.shared.register(type)
}

private let logger = MLXLogger(label: "KVCacheTypeRegistry")

/// Maps ``KVCache/serializedClassName`` to cache types, starting with the built-in caches.
final class KVCacheTypeRegistry: @unchecked Sendable {
    private struct Entry {
        let type: any KVCache.Type
        /// True for an instance of `type`, or of a subclass that inherits its name.
        let matches: (any KVCache) -> Bool
    }

    static let shared = KVCacheTypeRegistry()

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]

    private init() {
        try! register(KVCacheSimple.self, aliases: ["KVCacheSimple"])
        try! register(RotatingKVCache.self)
        try! register(QuantizedKVCache.self)
        try! register(VarianceNormalizedKVCache.self)
        try! register(ChunkedKVCache.self)
        try! register(ArraysCache.self)
        try! register(MambaCache.self)
        try! register(TurboQuantKVCache.self)
        try! register(CacheList.self)
    }

    func register<T: KVCache>(_ type: T.Type, aliases: [String] = []) throws {
        guard let name = type.serializedClassName else {
            throw KVCacheError(message: "\(type) has no serializedClassName")
        }
        let entry = Entry(type: type, matches: { $0 is T })
        try lock.withLock {
            for name in [name] + aliases {
                if let existing = entries[name] {
                    guard ObjectIdentifier(existing.type) == ObjectIdentifier(type) else {
                        throw KVCacheError(
                            message:
                                "Cache class name \(name) is already registered to \(existing.type)"
                        )
                    }
                } else {
                    entries[name] = entry
                }
            }
        }
    }

    func type(named name: String) -> (any KVCache.Type)? {
        lock.withLock { entries[name]?.type }
    }

    /// The name to save `cache` under, registering its type if needed.
    ///
    /// - Throws: if the cache, or a `CacheList` child, has no name or uses another type's name.
    func serializedClassName(for cache: any KVCache) throws -> String {
        let cacheType = Swift.type(of: cache)
        guard let name = cacheType.serializedClassName else {
            throw KVCacheError(
                message: "\(cacheType) has no serializedClassName, so it cannot be saved")
        }

        if let entry = lock.withLock({ entries[name] }) {
            guard entry.matches(cache) else {
                throw KVCacheError(
                    message:
                        "\(cacheType) uses cache class name \(name), which belongs to \(entry.type)"
                )
            }
        } else {
            try register(cacheType)
            logger.warning(
                "Registered \(String(describing: cacheType)) while saving a prompt cache. Call registerKVCacheType(_:) before loading it in a new process."
            )
        }

        if let list = cache as? CacheList {
            for child in list.children {
                _ = try serializedClassName(for: child)
            }
        }
        return name
    }
}
