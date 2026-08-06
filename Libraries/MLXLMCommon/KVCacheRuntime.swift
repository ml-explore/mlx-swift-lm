// Copyright © 2026 Apple Inc.

/// Apply a typed cache strategy to every eligible cache layer.
///
/// This is the single runtime dispatch point used by autoregressive,
/// speculative, guided, and diagnostic generation paths.
public func applyKVCacheConfiguration(
    cache: inout [KVCache],
    configuration: KVCacheConfiguration
) throws -> KVCacheApplicationResult {
    try applyKVCacheConfigurationValidated(
        cache: &cache, configuration: configuration
    ).result
}

/// Validate before mutation, then produce the diagnostic application result.
/// This keeps throwing application transactional for all validation failures.
package func applyKVCacheConfigurationValidated(
    cache: inout [KVCache],
    configuration: KVCacheConfiguration
) throws -> (result: KVCacheApplicationResult, isTerminal: Bool) {
    try validateKVCacheCompatibility(cache, configuration: configuration)
    let before = KVCacheTree.leaves(in: cache)
    let isTerminal = applyKVCacheConfigurationFast(
        cache: &cache, configuration: configuration)

    let after = KVCacheTree.leaves(in: cache)
    let report = kvCacheRuntimeReport(cache: cache, configuration: configuration)
    let converted = zip(before, after).count { original, current in
        guard case .simple = original.kind else { return false }
        return current.isCompressed
    }
    let protectedPaths = protectedPaths(for: before, configuration: configuration)
    let alreadyCompatible = before.count { leaf in
        if case .simple = leaf.kind {
            return configuration.strategy.identifier == .fullPrecision
        } else {
            return leaf.supports(configuration, protectedPaths: protectedPaths) == true
        }
    }
    return (
        KVCacheApplicationResult(
            convertedLayerCount: converted,
            alreadyCompatibleLayerCount: alreadyCompatible,
            pendingLayerCount: report.pendingLayerCount,
            skipped: report.layers.filter { $0.state == .skipped }),
        isTerminal
    )
}

/// Minimal decode-loop application. Returns true once no simple layer can
/// become newly eligible on a later token.
@discardableResult
package func applyKVCacheConfigurationFast(
    cache: inout [KVCache],
    configuration: KVCacheConfiguration
) -> Bool {
    switch configuration.strategy.storage {
    case .fullPrecision:
        true
    case .affine(let affine):
        maybeAffineQuantizeKVCache(
            cache: &cache,
            bits: affine.bits,
            groupSize: affine.groupSize,
            compressionStart: affine.compressionStart)
    case .turboQuant(let turbo):
        maybeTurboQuantizeKVCache(
            cache: &cache,
            keyBits: turbo.keyPrecision.bitWidth,
            valueBits: turbo.valuePrecision.bitWidth,
            quantizedKVStart: turbo.compressionStart)
    }
}

/// Validate the requested compatibility policy against a concrete model cache.
///
/// Full-precision and non-attention state entries do not participate. Affine
/// caches are accepted for TurboQuant because aggressive configurations use
/// them as an intentional boundary-layer fallback.
public func validateKVCacheCompatibility(
    _ cache: [KVCache],
    configuration: KVCacheConfiguration
) throws {
    let leaves = KVCacheTree.leaves(in: cache)

    if let capacity = configuration.capacity {
        let incompatibleCapacityCount = leaves.count { leaf in
            switch leaf.kind {
            case .recurrent:
                return false
            case .rotating(let rotating) where rotating.capacityOrigin == .modelNative:
                // Architecture-defined sliding windows are independent of the
                // caller-configurable capacity used for global attention.
                return false
            case .rotating(let rotating):
                return rotating.maxSize != capacity.maxTokens
                    || rotating.preservedPrefixTokens != capacity.preservedPrefixTokens
            default:
                return true
            }
        }
        guard incompatibleCapacityCount == 0 else {
            throw KVCacheConfigurationError.incompatibleCapacity(
                expected: capacity.maxTokens, count: incompatibleCapacityCount)
        }
    }

    if configuration.strategy.identifier == .fullPrecision {
        let compressedCount = leaves.count { $0.isCompressed }
        guard compressedCount == 0 else {
            throw KVCacheConfigurationError.incompatibleLayers(
                strategy: .fullPrecision, count: compressedCount)
        }
        return
    }

    let protectedPaths = protectedPaths(for: leaves, configuration: configuration)
    let support = leaves.compactMap {
        $0.supports(configuration, protectedPaths: protectedPaths)
    }
    let compatibleCount = support.count { $0 }
    let incompatibleCount = support.count { !$0 }

    switch configuration.compatibility {
    case .allowPartial:
        return
    case .requireAtLeastOneLayer:
        guard compatibleCount > 0 else {
            throw KVCacheConfigurationError.noCompatibleLayers(
                strategy: configuration.strategy.identifier)
        }
    case .requireAllLayers:
        guard compatibleCount > 0 else {
            throw KVCacheConfigurationError.noCompatibleLayers(
                strategy: configuration.strategy.identifier)
        }
        guard incompatibleCount == 0 else {
            throw KVCacheConfigurationError.incompatibleLayers(
                strategy: configuration.strategy.identifier,
                count: incompatibleCount)
        }
    }
}

/// Describe the effective strategy of every cache entry without mutating it.
public func kvCacheRuntimeReport(
    cache: [KVCache],
    configuration: KVCacheConfiguration
) -> KVCacheRuntimeReport {
    let leaves = KVCacheTree.leaves(in: cache)
    let protectedPaths = protectedPaths(for: leaves, configuration: configuration)
    return KVCacheRuntimeReport(
        requestedConfiguration: configuration,
        layers: leaves.map {
            $0.runtimeLayer(
                configuration: configuration,
                protectedPaths: protectedPaths)
        })
}

private func protectedPaths(
    for leaves: [KVCacheLeaf],
    configuration: KVCacheConfiguration
) -> Set<[Int]> {
    guard case .turboQuant(let turbo) = configuration.strategy.storage else { return [] }
    return KVCacheTree.turboQuantProtectedPaths(
        in: leaves,
        keyBits: turbo.keyPrecision.bitWidth,
        valueBits: turbo.valuePrecision.bitWidth)
}

extension KVCacheLeaf {
    fileprivate func runtimeLayer(
        configuration: KVCacheConfiguration,
        protectedPaths: Set<[Int]>
    ) -> KVCacheRuntimeReport.Layer {
        let requested = configuration.strategy.identifier
        switch kind {
        case .recurrent:
            return .init(
                path: path, state: .notApplicable, resolvedStrategy: nil,
                reason: .nonAttentionState)
        case .turboQuant(let turbo):
            let matches = requested == .turboQuant
            return .init(
                path: path,
                state: matches ? (turbo.isCompressed ? .active : .pending) : .skipped,
                resolvedStrategy: .turboQuant,
                reason: matches
                    ? (turbo.isCompressed ? nil : .awaitingCompressionStart)
                    : .differentStrategy)
        case .affine:
            let isBoundaryProtection =
                requested == .turboQuant && protectedPaths.contains(path)
            let matches = requested == .affine || isBoundaryProtection
            return .init(
                path: path,
                state: matches ? .active : .skipped,
                resolvedStrategy: .affine,
                reason: isBoundaryProtection
                    ? .boundaryProtection : (matches ? nil : .differentStrategy))
        case .rotating:
            return .init(
                path: path,
                state: requested == .fullPrecision ? .active : .skipped,
                resolvedStrategy: .fullPrecision,
                reason: requested == .fullPrecision ? nil : .slidingWindow)
        case .simple where requested == .fullPrecision:
            return .init(
                path: path, state: .active, resolvedStrategy: .fullPrecision, reason: nil)
        case .simple:
            guard supports(configuration, protectedPaths: protectedPaths) == true else {
                return .init(
                    path: path,
                    state: .skipped,
                    resolvedStrategy: .fullPrecision,
                    reason: .unsupportedShape)
            }
            let compressionPending = cache.offset <= configuration.strategy.compressionStart
            return .init(
                path: path,
                state: compressionPending ? .pending : .skipped,
                resolvedStrategy: .fullPrecision,
                reason: compressionPending ? .awaitingCompressionStart : .unsupportedShape)
        case .unsupported:
            return .init(
                path: path, state: .skipped, resolvedStrategy: nil, reason: .unsupportedShape)
        }
    }
}
