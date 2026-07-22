// Copyright © 2026 Apple Inc.

/// A classified leaf in a potentially nested cache topology.
struct KVCacheLeaf {
    enum Kind {
        case recurrent
        case rotating(RotatingKVCache)
        case simple(KVCacheSimple)
        case affine(QuantizedKVCache)
        case turboQuant(TurboQuantKVCache)
        case unsupported
    }

    let path: [Int]
    let cache: KVCache

    var kind: Kind {
        switch cache {
        case is MambaCache, is ArraysCache: .recurrent
        case let cache as RotatingKVCache: .rotating(cache)
        case let cache as TurboQuantKVCache: .turboQuant(cache)
        case let cache as QuantizedKVCache: .affine(cache)
        case let cache as KVCacheSimple: .simple(cache)
        default: .unsupported
        }
    }

    var isAttentionCache: Bool {
        if case .recurrent = kind { return false }
        return true
    }

    var isCompressed: Bool {
        switch kind {
        case .affine, .turboQuant: true
        default: false
        }
    }

    /// Whether this leaf can satisfy a compression strategy. `nil` identifies
    /// state that is not an attention cache and does not participate.
    func supports(
        _ configuration: KVCacheConfiguration,
        protectedPaths: Set<[Int]>
    ) -> Bool? {
        let strategy = configuration.strategy.identifier
        switch kind {
        case .recurrent:
            return nil
        case .rotating, .unsupported:
            return strategy == .fullPrecision
        case .simple(let simple):
            let groupSize: Int
            switch configuration.strategy.storage {
            case .fullPrecision:
                return true
            case .affine(let affine):
                groupSize = affine.groupSize
            case .turboQuant(let turbo):
                guard protectedPaths.contains(path) || turbo.keyPrecision == .affineEightBit
                else { return true }
                groupSize = 64
            }

            let state = simple.innerState()
            guard state.count >= 2 else { return true }
            return resolvedKVQuantizationGroupSize(
                requested: groupSize,
                keyHeadDim: state[0].dim(3),
                valueHeadDim: state[1].dim(3)) != nil
        case .affine:
            return strategy == .affine
                || (strategy == .turboQuant && protectedPaths.contains(path))
        case .turboQuant:
            return strategy == .turboQuant
        }
    }
}

/// The sole traversal and mutation boundary for nested cache topologies.
enum KVCacheTree {
    static func leaves(in cache: [KVCache]) -> [KVCacheLeaf] {
        var leaves = [KVCacheLeaf]()
        for (index, entry) in cache.enumerated() {
            appendLeaves(from: entry, path: [index], to: &leaves)
        }
        return leaves
    }

    static func rewrite(
        _ cache: inout [KVCache],
        using transform: (KVCacheLeaf) -> KVCache
    ) {
        for index in cache.indices {
            if let list = cache[index] as? CacheList {
                list.rewriteLeaves(path: [index], using: transform)
            } else {
                let leaf = KVCacheLeaf(path: [index], cache: cache[index])
                cache[index] = transform(leaf)
            }
        }
    }

    static func turboQuantProtectedPaths(
        in leaves: [KVCacheLeaf],
        keyBits: Int,
        valueBits: Int
    ) -> Set<[Int]> {
        let fragile = (keyBits > 0 && keyBits < 8) || valueBits <= 2
        guard fragile else { return [] }

        let attentionPaths = leaves.filter(\.isAttentionCache).map(\.path)
        let boundaryCount = min(2, attentionPaths.count / 2)
        guard boundaryCount > 0 else { return [] }
        return Set(attentionPaths.prefix(boundaryCount))
            .union(attentionPaths.suffix(boundaryCount))
    }

    private static func appendLeaves(
        from cache: KVCache,
        path: [Int],
        to leaves: inout [KVCacheLeaf]
    ) {
        if let list = cache as? CacheList {
            for (index, child) in list.children.enumerated() {
                appendLeaves(from: child, path: path + [index], to: &leaves)
            }
        } else {
            leaves.append(KVCacheLeaf(path: path, cache: cache))
        }
    }
}
