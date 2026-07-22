// Copyright © 2026 Apple Inc.

/// Immutable, resolved cache behavior for one generation request.
///
/// A disabled plan is a first-class value, so generation paths do not need to
/// repeat optional checks around validation and dynamic compression.
package struct KVCachePlan: Sendable, Equatable {
    package static let disabled = KVCachePlan(configuration: nil)

    package let configuration: KVCacheConfiguration?

    package init(configuration: KVCacheConfiguration?) {
        self.configuration = configuration
    }

    package func validated(_ cache: [KVCache]) throws -> [KVCache] {
        if let configuration {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
        return cache
    }

    package func validated(_ storage: KVCacheStorage) throws -> KVCacheStorage {
        precondition(storage.plan == self, "KVCacheStorage used with a different plan")
        storage.cache = try validated(storage.cache)
        return storage
    }

    package func apply(to cache: inout [KVCache]) {
        guard let configuration else { return }
        _ = applyKVCacheConfigurationFast(cache: &cache, configuration: configuration)
    }

    /// Apply dynamic conversion to shared cache storage.
    ///
    /// Once every eligible layer has either converted or reached a terminal
    /// unsupported state, later decode steps only compare the completed plan.
    package func apply(to storage: KVCacheStorage) {
        precondition(storage.plan == self, "KVCacheStorage used with a different plan")
        guard !storage.isApplicationTerminal else { return }
        guard let configuration else {
            storage.isApplicationTerminal = true
            return
        }

        if applyKVCacheConfigurationFast(
            cache: &storage.cache, configuration: configuration)
        {
            storage.isApplicationTerminal = true
        }
    }

    @discardableResult
    package func applyAndValidate(
        to cache: inout [KVCache]
    ) throws -> KVCacheApplicationResult? {
        guard let configuration else { return nil }
        return try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
    }

    @discardableResult
    package func applyAndValidate(
        to storage: KVCacheStorage
    ) throws -> KVCacheApplicationResult? {
        precondition(storage.plan == self, "KVCacheStorage used with a different plan")
        guard !storage.isApplicationTerminal else { return nil }
        guard let configuration else {
            storage.isApplicationTerminal = true
            return nil
        }

        let application = try applyKVCacheConfigurationValidated(
            cache: &storage.cache, configuration: configuration)
        if application.isTerminal {
            storage.isApplicationTerminal = true
        }
        return application.result
    }

    package func report(for cache: [KVCache]) -> KVCacheRuntimeReport? {
        configuration.map { kvCacheRuntimeReport(cache: cache, configuration: $0) }
    }
}

/// Shared ownership for a realized cache and its dynamic application state.
///
/// Cache arrays have value semantics, but dynamic compression replaces array
/// elements. Sharing this storage keeps sessions and iterators on the same
/// realized array while the cache objects themselves remain reference types.
/// Access is externally serialized by generation/session ownership.
package final class KVCacheStorage: @unchecked Sendable {
    package var cache: [KVCache] {
        didSet { isApplicationTerminal = false }
    }
    package let plan: KVCachePlan
    package fileprivate(set) var isApplicationTerminal = false

    package init(_ cache: [KVCache], plan: KVCachePlan) {
        self.cache = cache
        self.plan = plan
    }

    package func replace(with cache: [KVCache]) {
        self.cache = cache
    }
}

extension KVCacheConfiguration.Capacity {
    /// Construct the bounded cache represented by this value.
    package func makeRotatingCache() -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxTokens, keep: preservedPrefixTokens)
        cache.capacityOrigin = .requested
        return cache
    }
}

extension GenerateParameters {
    package var effectiveKVCacheCapacity: KVCacheConfiguration.Capacity? {
        if let capacity = kvCache?.capacity {
            return capacity
        }
        guard let maxKVSize else { return nil }
        return .init(
            uncheckedMaxTokens: maxKVSize,
            preservedPrefixTokens: min(4, max(0, maxKVSize - 1)))
    }

    package func kvCachePlan() throws -> KVCachePlan {
        KVCachePlan(configuration: try resolvedKVCacheConfiguration())
    }

    package func resolvedKVCacheConfiguration() throws -> KVCacheConfiguration? {
        if let kvCache {
            guard !hasLegacyKVCacheOverrides else {
                throw KVCacheConfigurationError.conflictingLegacyConfiguration
            }
            return kvCache
        }

        let capacity = try legacyKVCacheCapacity()
        let strategy = try legacyKVCacheStrategy()
        guard capacity != nil || strategy != .fullPrecision else { return nil }
        return KVCacheConfiguration(
            capacity: capacity,
            strategy: strategy,
            compatibility: .allowPartial)
    }

    private var hasLegacyKVCacheOverrides: Bool {
        maxKVSize != nil || kvBits != nil || kvScheme != nil || kvGroupSize != 64
            || quantizedKVStart != 0
    }

    private func legacyKVCacheCapacity() throws -> KVCacheConfiguration.Capacity? {
        guard let maxKVSize else { return nil }
        return try .init(
            maxTokens: maxKVSize,
            preservedPrefixTokens: min(4, max(0, maxKVSize - 1)))
    }

    private func legacyKVCacheStrategy() throws -> KVCacheConfiguration.Strategy {
        if let kvScheme {
            return try strategy(forLegacyScheme: kvScheme)
        }
        guard let kvBits else { return .fullPrecision }
        return .affine(
            try .init(
                bits: kvBits,
                groupSize: kvGroupSize,
                compressionStart: quantizedKVStart))
    }

    private func strategy(
        forLegacyScheme scheme: String
    ) throws -> KVCacheConfiguration.Strategy {
        if let affine = resolveAffineScheme(scheme) {
            return .affine(
                try .init(
                    bits: affine.bits,
                    groupSize: affine.groupSize,
                    compressionStart: quantizedKVStart))
        }
        if let turbo = resolveTurboScheme(scheme),
            let keyPrecision = TurboQuantKVCacheConfiguration.KeyPrecision(
                legacyBitWidth: turbo.keyBits),
            let valuePrecision = TurboQuantKVCacheConfiguration.ValuePrecision(
                legacyBitWidth: turbo.valueBits)
        {
            return .turboQuant(
                try .init(
                    keyPrecision: keyPrecision,
                    valuePrecision: valuePrecision,
                    compressionStart: quantizedKVStart))
        }
        throw KVCacheConfigurationError.unsupportedLegacyScheme(scheme)
    }
}
