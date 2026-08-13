// Copyright © 2026 Apple Inc.

/// Immutable, resolved cache behavior for one generation request.
///
/// A disabled plan is a first-class value, so generation paths do not need to
/// repeat optional checks around validation and dynamic compression.
package struct KVCachePlan: Sendable, Equatable {
    package static let disabled = KVCachePlan(configuration: nil)

    package let configuration: KVCacheConfiguration?
    package let requestSource: KVCacheStatus.RequestSource?

    package init(
        configuration: KVCacheConfiguration?,
        requestSource: KVCacheStatus.RequestSource? = nil
    ) {
        self.configuration = configuration
        self.requestSource = configuration == nil ? nil : requestSource ?? .typed
    }

    package func validated(_ cache: [KVCache]) throws -> [KVCache] {
        if let configuration {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
        return cache
    }

    package func validated(_ storage: KVCacheStorage) throws -> KVCacheStorage {
        precondition(storage.plan == self, "KVCacheStorage used with a different plan")
        precondition(!storage.roundIsOpen, "validated(_:) ran inside a staged round")
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
        precondition(!storage.roundIsOpen, "apply(to:) ran inside a staged round")
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
        precondition(!storage.roundIsOpen, "applyAndValidate(to:) ran inside a staged round")
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

    package func report(for storage: KVCacheStorage) -> KVCacheRuntimeReport? {
        configuration.map { configuration in
            let report = kvCacheRuntimeReport(
                cache: storage.cache, configuration: configuration)
            return KVCacheRuntimeReport(
                requestedConfiguration: report.requestedConfiguration,
                layers: report.layers,
                processedTokenCount: storage.processedTokenCount)
        }
    }
}

/// Shared ownership for a realized cache and its dynamic application state.
///
/// Cache arrays have value semantics, but dynamic compression replaces array
/// elements. Sharing this storage keeps sessions and iterators on the same
/// realized array while the cache objects themselves remain reference types.
/// Access is externally serialized by generation/session ownership. This type
/// is intentionally not `Sendable`: sharing it across isolation domains would
/// permit unsynchronized mutation of both the cache and its progress state.
package final class KVCacheStorage {
    package var cache: [KVCache] {
        didSet {
            precondition(
                !roundIsOpen,
                "the realized cache was replaced while a staged round was open")
            // A rewind record names slots, and a replacement changes what those slots hold --
            // so it invalidates the record rather than being restored into blindly.
            lastRound = nil
            isApplicationTerminal = false
        }
    }
    package let plan: KVCachePlan
    package fileprivate(set) var isApplicationTerminal = false

    private var openRound: KVCacheRound?

    /// How to take one leaf's share of a commit back, addressed by slot rather than by object.
    ///
    /// `point == nil` means a plain `trim(_:)` undoes it exactly, which is true for every leaf
    /// that had not wrapped when the commit ran.
    private struct LeafRewind {
        let slot: Int
        let point: (any KVCacheLeafRestorePoint)?
    }

    /// What it would take to undo the most recent commit, kept only until the next write.
    private struct CompletedRound {
        let committedPositions: Int
        let leaves: [LeafRewind]
    }
    private var lastRound: CompletedRound?

    /// Whether a staged round is presently writing provisionally against these entries.
    ///
    /// A round hands the model substitute caches and defers part of its writes, so the entries
    /// and the timeline are deliberately out of step until it commits. Operations that assume
    /// they are in step -- replacing entries, trimming, snapshotting -- are refused while one is
    /// open rather than silently reading a half-written position.
    package var roundIsOpen: Bool { openRound != nil }
    /// Authoritative logical position represented by this model cache.
    ///
    /// Individual cache entries retain their native offsets and metadata, but
    /// the model-wide timeline lives here exactly once. Generation updates this
    /// value after successful model calls and rolls it back with cache trims.
    package private(set) var processedTokenCount: Int

    package init(
        _ cache: [KVCache],
        plan: KVCachePlan,
        processedTokenCount: Int? = nil
    ) {
        self.cache = cache
        self.plan = plan
        self.processedTokenCount =
            processedTokenCount ?? Self.inferProcessedTokenCount(from: cache)
    }

    package func replace(with cache: [KVCache]) {
        self.cache = cache
    }

    /// Commit tokens after a successful model evaluation.
    @inline(__always)
    package func commitProcessedTokens(_ count: Int) {
        precondition(count >= 0, "Processed token count cannot move backwards")
        let (updated, overflow) = processedTokenCount.addingReportingOverflow(count)
        precondition(!overflow, "Processed token count overflow")
        processedTokenCount = updated
        lastRound = nil
    }

    /// Trim cache state and the shared logical timeline atomically.
    @discardableResult
    package func trim(_ count: Int) -> Int {
        precondition(count >= 0, "Trim count cannot be negative")
        precondition(!roundIsOpen, "cannot trim while a staged round is open")
        let trimmed = trimPromptCache(cache, numTokens: count)
        precondition(
            trimmed <= processedTokenCount,
            "Cache trimmed beyond its processed-token timeline")
        processedTokenCount -= trimmed
        lastRound = nil
        return trimmed
    }

    /// Rewind a speculative tail across mixed attention and recurrent caches.
    @discardableResult
    package func rewindSpeculative(_ count: Int) -> Int {
        precondition(count >= 0, "Rewind count cannot be negative")
        let rewound = rewindSpeculativePromptCache(cache, numTokens: count)
        precondition(
            rewound <= processedTokenCount,
            "Cache rewound beyond its processed-token timeline")
        processedTokenCount -= rewound
        return rewound
    }

    /// Open a staged round over these entries, or `nil` when one of them cannot take part.
    ///
    /// Every leaf is classified before anything is constructed, so a refusal leaves the entries
    /// exactly as they were and the caller can fall back to unstaged decoding. Nested
    /// ``CacheList`` topologies are refused too: they expand to more leaves than array slots, and
    /// the caches handed back have to line up with the live array positionally.
    ///
    /// - Parameter maximumPositions: the widest write the round will make. Append-only leaves are
    ///   admitted only if they can still be rewound after absorbing it.
    package func beginRound(maximumPositions: Int) -> KVCacheRound? {
        precondition(!roundIsOpen, "a staged round is already open on this storage")

        let leaves = KVCacheTree.leaves(in: cache)
        guard leaves.count == cache.count, leaves.allSatisfy({ $0.path.count == 1 }) else {
            return nil
        }

        var strategies = [any KVCacheRoundStrategy]()
        strategies.reserveCapacity(leaves.count)
        for leaf in leaves {
            guard
                let strategy = KVCacheRoundStrategyFactory.make(
                    for: leaf, slot: leaf.path[0], maximumPositions: maximumPositions)
            else { return nil }
            strategies.append(strategy)
        }

        lastRound = nil
        let round = KVCacheRound(strategies: strategies, maximumPositions: maximumPositions)
        openRound = round
        return round
    }

    /// Keep the first `retaining` positions the round wrote and drop the rest, advancing the
    /// timeline by exactly that much.
    ///
    /// Leaf commit and timeline advance are one operation on purpose. A commit that moved leaf
    /// K/V without moving the timeline -- or the reverse -- would recreate, one level up, the
    /// split brain this storage exists to prevent.
    @discardableResult
    package func commit(_ round: KVCacheRound, retaining: Int) -> KVCacheRoundCommit {
        precondition(openRound === round, "committing a round that is not open on this storage")
        let written = round.writtenPositions
        precondition(
            retaining >= 0 && retaining <= written,
            "cannot retain \(retaining) of \(written) written positions")

        // Built before the commit, because that is the last moment the pre-commit contents of a
        // wrapped ring still exist. Records name slots: the object a rotating strategy presents
        // is a round-scoped wrapper that outlives nothing, and its `trim(_:)` is a no-op.
        var rewinds = [LeafRewind]()
        rewinds.reserveCapacity(round.strategies.count)
        for strategy in round.strategies {
            rewinds.append(
                LeafRewind(
                    slot: strategy.slot,
                    point: strategy.makeRestorePoint(retaining: retaining)))
            strategy.commit(retaining: retaining)
        }
        processedTokenCount += retaining
        openRound = nil
        // Nothing to keep when every leaf can be undone by a trim: that is the path `trim(_:)`
        // already takes, and recording a round would only widen what a rewind means.
        lastRound =
            rewinds.contains { $0.point != nil }
            ? CompletedRound(committedPositions: retaining, leaves: rewinds)
            : nil

        assert(
            nativeAttentionOffsetsAreAligned,
            "committing a staged round left the leaves and the timeline out of step")

        return KVCacheRoundCommit(
            committedPositions: retaining,
            discardedPositions: written - retaining,
            emittedLengths: round.strategies.map(\.emittedLength))
    }

    /// Drop everything the round wrote.
    @discardableResult
    package func rollback(_ round: KVCacheRound) -> KVCacheRoundCommit {
        commit(round, retaining: 0)
    }

    /// Take back `count` positions the last commit kept.
    ///
    /// A generation can stop with positions committed but never emitted, and those have to leave
    /// the cache before anything reconciles a token ledger against it. `trim(_:)` handles that
    /// whenever every leaf can undo an append by bookkeeping; past a sliding window one cannot,
    /// so the commit left behind what it takes to put that leaf back and replay the part that was
    /// emitted. Either way this is exact, and it is the only rewind left on the staged path.
    ///
    /// Only the most recent commit can be taken back: any write after it invalidates the record,
    /// and so does replacing an entry, since the record names slots rather than objects.
    @discardableResult
    package func rewindLastRound(_ count: Int) -> Int {
        precondition(count >= 0, "Rewind count cannot be negative")
        precondition(!roundIsOpen, "cannot rewind while a staged round is open")
        guard count > 0 else { return 0 }

        guard let last = lastRound else { return trim(count) }
        let rewound = Swift.min(count, last.committedPositions)
        guard rewound > 0 else { return 0 }

        // Resolve and check every slot before touching one. A rewind that undid some leaves and
        // not others would leave the entries further out of step than doing nothing does, so a
        // record that cannot be applied in full falls back to the plain trim instead.
        let leaves = last.leaves.map { record -> (record: LeafRewind, leaf: KVCache)? in
            guard cache.indices.contains(record.slot) else { return nil }
            let leaf = cache[record.slot]
            if let point = record.point {
                return point.canRestore(into: leaf) ? (record, leaf) : nil
            }
            return leaf.isTrimmable ? (record, leaf) : nil
        }
        guard leaves.allSatisfy({ $0 != nil }) else { return trim(count) }

        for (record, leaf) in leaves.compactMap({ $0 }) {
            if let point = record.point {
                point.restore(into: leaf, retaining: last.committedPositions - rewound)
            } else {
                let trimmed = leaf.trim(rewound)
                precondition(
                    trimmed == rewound,
                    """
                    entry \(record.slot) took back \(trimmed) of \(rewound) positions; a commit \
                    admits a leaf only if a trim can undo it exactly
                    """)
            }
        }
        processedTokenCount -= rewound
        lastRound = nil

        assert(
            nativeAttentionOffsetsAreAligned,
            "rewinding a committed round left the leaves and the timeline out of step")
        return rewound
    }

    /// How much of the emitted sequence the given leaf can still describe, outside a round.
    ///
    /// The whole stream for a global layer; the trailing window for a sliding one. This is the
    /// same quantity ``KVCacheRoundCommit/emittedLengths`` reports, for callers that need it
    /// without a round in hand -- prefill, for instance.
    package func emittedLength(forLeaf index: Int) -> Int {
        guard cache.indices.contains(index) else { return processedTokenCount }
        return Swift.min(processedTokenCount, cache[index].maxSize ?? .max)
    }

    /// Create an independent snapshot while preserving plan and progress.
    package func copy() -> KVCacheStorage {
        precondition(!roundIsOpen, "cannot snapshot while a staged round is open")
        let copy = KVCacheStorage(
            cache.map { $0.copy() },
            plan: plan,
            processedTokenCount: processedTokenCount)
        copy.isApplicationTerminal = isApplicationTerminal
        return copy
    }

    /// Debug-only consistency audit for native attention offsets. Recurrent
    /// entries intentionally do not participate in the shared timeline.
    package var nativeAttentionOffsetsAreAligned: Bool {
        KVCacheTree.leaves(in: cache)
            .filter(\.isAttentionCache)
            .allSatisfy { $0.cache.offset == processedTokenCount }
    }

    /// One-time compatibility inference for caches created outside the shared
    /// container API. Hot generation and reuse paths never scan child caches.
    private static func inferProcessedTokenCount(from cache: [KVCache]) -> Int {
        let leaves = KVCacheTree.leaves(in: cache)
        let attentionOffsets = leaves.filter(\.isAttentionCache).map { $0.cache.offset }
        if let inferred = attentionOffsets.min() {
            assert(
                attentionOffsets.allSatisfy { $0 == inferred },
                "Attention cache offsets diverged before storage adoption")
            return inferred
        }

        // Compatibility for recurrent-only caches created by older call sites
        // that used the inherited offset as a model-wide progress counter.
        let legacyOffsets = leaves.map { $0.cache.offset }
        let inferred = legacyOffsets.min() ?? 0
        assert(
            legacyOffsets.allSatisfy { $0 == inferred },
            "Recurrent cache offsets diverged before storage adoption")
        return inferred
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
    /// Resolve and validate the complete request before exposing its capacity.
    ///
    /// Cache factories use this instead of constructing an unchecked legacy
    /// capacity so direct `newCache(parameters:)` calls fail with the same typed
    /// errors as generation entry points.
    package func effectiveKVCacheCapacity() throws -> KVCacheConfiguration.Capacity? {
        try resolvedKVCacheConfiguration()?.capacity
    }

    package func kvCachePlan() throws -> KVCachePlan {
        let configuration = try resolvedKVCacheConfiguration()
        let requestSource: KVCacheStatus.RequestSource? =
            if configuration == nil {
                nil
            } else if kvCache == nil {
                .legacy
            } else {
                .typed
            }
        return KVCachePlan(
            configuration: configuration,
            requestSource: requestSource)
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
