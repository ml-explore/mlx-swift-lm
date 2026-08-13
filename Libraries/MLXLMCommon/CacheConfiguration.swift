// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Layer kinds

/// How a single model layer retains state across generation steps.
///
/// This is the stable, descriptive counterpart to concrete ``KVCache`` instances.
/// Obtain it through ``KVCacheStatus`` rather than inspecting or
/// casting runtime cache objects.
public enum CacheLayerKind: Sendable, Hashable {
    /// Standard or implementation-defined attention KV.
    ///
    /// A `nil` `maxSize` is unbounded. A non-nil value describes a bounded
    /// custom cache that is not a ``RotatingKVCache``.
    case attention(maxSize: Int?)

    /// Sliding / rotating attention window (``RotatingKVCache``).
    ///
    /// Used both for architecture-imposed sliding-window attention and for
    /// caller-requested capacity limits on full-attention layers.
    case rotatingAttention(maxSize: Int, keep: Int)

    /// Fixed-size state-space / Mamba / GDN recurrent state (``MambaCache``).
    ///
    /// These caches are not token-windowed; a ``GenerateParameters/maxKVSize``
    /// policy does not apply to them.
    case stateSpace

    /// Composite per-layer cache (``CacheList``), e.g. Mamba + attention.
    case composite([CacheLayerKind])

    /// Whether this kind stores attention keys/values that can be size-limited.
    public var isAttention: Bool {
        switch self {
        case .attention, .rotatingAttention:
            return true
        case .composite(let children):
            return children.contains(where: \.isAttention)
        case .stateSpace:
            return false
        }
    }

    /// Whether this kind is a bounded attention window.
    public var isBoundedAttention: Bool {
        switch self {
        case .attention(let maxSize):
            return maxSize != nil
        case .rotatingAttention:
            return true
        case .composite(let children):
            return children.contains(where: \.isBoundedAttention)
        case .stateSpace:
            return false
        }
    }

    /// Effective attention token budget, if any.
    public var attentionMaxSize: Int? {
        switch self {
        case .attention(let maxSize):
            return maxSize
        case .rotatingAttention(let maxSize, _):
            return maxSize
        case .composite(let children):
            return children.compactMap(\.attentionMaxSize).max()
        case .stateSpace:
            return nil
        }
    }

    fileprivate var containsStateSpace: Bool {
        switch self {
        case .stateSpace:
            return true
        case .composite(let children):
            return children.contains(where: \.containsStateSpace)
        case .attention, .rotatingAttention:
            return false
        }
    }
}

// MARK: - Unified status

/// Authoritative planned or realized state of a model's KV / recurrent cache.
///
/// Obtain this through ``LanguageModel/cacheStatus(parameters:)``,
/// ``ModelContainer/cacheStatus(parameters:)``, or ``ChatSession/cacheStatus()``.
/// The same value describes topology, capacity, strategy application, and
/// model-wide progress for both legacy and typed configuration entry points.
public struct KVCacheStatus: Sendable, Hashable {
    public enum Phase: Sendable, Hashable {
        /// The cache layout has been constructed for inspection but has not run generation.
        case planned
        /// The status describes cache storage owned by an active session.
        case realized
    }

    public enum RequestSource: Sendable, Hashable {
        case legacy
        case typed
    }

    /// Relationship between requested capacity and the attention layers to
    /// which that request applies.
    public enum CapacityDisposition: Sendable, Hashable {
        case notRequested
        case fullyApplied
        case partiallyApplied
        case unsupported
        case ignored
    }

    public let phase: Phase
    /// `nil` when no capacity or compression behavior was requested.
    public let requestedConfiguration: KVCacheConfiguration?
    public let requestSource: RequestSource?
    /// Flattened leaf layers. ``KVCacheLayerStatus/path`` preserves composite topology.
    public let layers: [KVCacheLayerStatus]
    /// Authoritative model-wide position for realized session storage.
    public let processedTokenCount: Int?
    public let capacityDisposition: CapacityDisposition
    public let capacityAppliedLayerCount: Int
    public let capacityUnappliedLayerCount: Int

    /// `true` when the topology contains both attention and recurrent state.
    public var isHybrid: Bool {
        layers.contains { $0.kind.isAttention }
            && layers.contains { $0.kind.containsStateSpace }
    }

    /// Per-leaf attention token bounds (`nil` for unbounded or non-attention state).
    public var attentionMaxSizes: [Int?] {
        layers.map { $0.kind.attentionMaxSize }
    }

    public var requestedStrategy: KVCacheStrategyIdentifier? {
        requestedConfiguration?.strategy.identifier
    }

    public var compressedLayerCount: Int {
        layers.count {
            $0.state == .active && $0.resolvedStrategy != .fullPrecision
        }
    }

    public var pendingLayerCount: Int {
        layers.count { $0.state == .pending }
    }

    public var skippedLayerCount: Int {
        layers.count { $0.state == .skipped }
    }

    /// Inspect a raw cache array using a typed request.
    public init(
        cache: [KVCache],
        configuration: KVCacheConfiguration? = nil,
        phase: Phase = .realized,
        processedTokenCount: Int? = nil
    ) {
        self.init(
            cache: cache,
            configuration: configuration,
            requestSource: configuration == nil ? nil : .typed,
            phase: phase,
            processedTokenCount: processedTokenCount)
    }

    package init(
        cache: [KVCache],
        plan: KVCachePlan,
        phase: Phase,
        processedTokenCount: Int? = nil
    ) {
        self.init(
            cache: cache,
            configuration: plan.configuration,
            requestSource: plan.requestSource,
            phase: phase,
            processedTokenCount: processedTokenCount)
    }

    private init(
        cache: [KVCache],
        configuration: KVCacheConfiguration?,
        requestSource: RequestSource?,
        phase: Phase,
        processedTokenCount: Int?
    ) {
        let layers = kvCacheLayerStatuses(cache: cache, configuration: configuration)
        let capacity = Self.capacityOutcome(
            layers: layers,
            capacity: configuration?.capacity,
            requestSource: requestSource)

        self.phase = phase
        self.requestedConfiguration = configuration
        self.requestSource = requestSource
        self.layers = layers
        self.processedTokenCount = processedTokenCount
        self.capacityDisposition = capacity.disposition
        self.capacityAppliedLayerCount = capacity.applied
        self.capacityUnappliedLayerCount = capacity.unapplied
    }

    private static func capacityOutcome(
        layers: [KVCacheLayerStatus],
        capacity: KVCacheConfiguration.Capacity?,
        requestSource: RequestSource?
    ) -> (disposition: CapacityDisposition, applied: Int, unapplied: Int) {
        guard let capacity else { return (.notRequested, 0, 0) }

        var applied = 0
        var unapplied = 0
        for layer in layers where layer.kind.isAttention {
            switch requestSource {
            case .typed:
                if layer.capacitySource == .modelDefined {
                    continue
                }
                if case .rotatingAttention(let maxSize, let keep) = layer.kind,
                    layer.capacitySource == .requested,
                    maxSize == capacity.maxTokens,
                    keep == capacity.preservedPrefixTokens
                {
                    applied += 1
                } else {
                    unapplied += 1
                }
            case .legacy, nil:
                if let maxSize = layer.kind.attentionMaxSize, maxSize <= capacity.maxTokens {
                    applied += 1
                } else {
                    unapplied += 1
                }
            }
        }

        if applied + unapplied == 0 { return (.unsupported, 0, 0) }
        if unapplied == 0 { return (.fullyApplied, applied, 0) }
        if applied == 0 { return (.ignored, 0, unapplied) }
        return (.partiallyApplied, applied, unapplied)
    }
}

// MARK: - Kind inference from concrete caches

extension CacheLayerKind {
    /// Best-effort classification of a concrete cache instance.
    init(cache: KVCache) {
        switch cache {
        case let rotating as RotatingKVCache:
            self = .rotatingAttention(maxSize: rotating.maxSize ?? 0, keep: rotating.keepCount)
        case is MambaCache:
            self = .stateSpace
        case let list as CacheList:
            self = .composite(list.children.map { CacheLayerKind(cache: $0) })
        case let arrays as ArraysCache where !(arrays is MambaCache):
            // Generic slot cache used by some SSM variants.
            self = .stateSpace
        default:
            self = .attention(maxSize: cache.maxSize)
        }
    }
}

// MARK: - Construction helpers

/// Build a standard full-attention cache kind that honors the requested capacity.
///
/// The capacity is the unified view of ``GenerateParameters/maxKVSize`` and the
/// capacity carried by ``GenerateParameters/kvCache``.
///
/// - Throws: ``KVCacheConfigurationError`` when the request is invalid.
public func attentionCacheKind(parameters: GenerateParameters?) throws -> CacheLayerKind {
    guard let capacity = try parameters?.effectiveKVCacheCapacity() else {
        return .attention(maxSize: nil)
    }
    return .rotatingAttention(
        maxSize: capacity.maxTokens,
        keep: capacity.preservedPrefixTokens)
}

/// Build a standard full-attention ``KVCache`` that honors the requested capacity.
///
/// - Throws: ``KVCacheConfigurationError`` when the request is invalid.
public func makeAttentionKVCache(parameters: GenerateParameters?) throws -> KVCache {
    if let capacity = try parameters?.effectiveKVCacheCapacity() {
        return capacity.makeRotatingCache()
    }
    return KVCacheSimple()
}

/// Architecture-imposed sliding-window attention kind, capped by legacy `maxKVSize`.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func slidingWindowCacheKind(
    parameters: GenerateParameters?,
    window: Int
) throws -> CacheLayerKind {
    .rotatingAttention(
        maxSize: try effectiveSlidingWindow(parameters: parameters, window: window),
        keep: 0)
}

/// Architecture-imposed sliding-window attention cache, capped by legacy `maxKVSize`.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func makeSlidingWindowKVCache(parameters: GenerateParameters?, window: Int) throws
    -> KVCache
{
    RotatingKVCache(maxSize: try effectiveSlidingWindow(parameters: parameters, window: window))
}

/// Full-attention kind if the layer is not sliding-window; otherwise the
/// architecture sliding window capped by legacy `maxKVSize`.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func hybridAttentionCacheKind(
    parameters: GenerateParameters?,
    slidingWindow: Int?,
    usesSlidingWindow: Bool
) throws -> CacheLayerKind {
    if usesSlidingWindow, let slidingWindow {
        return try slidingWindowCacheKind(parameters: parameters, window: slidingWindow)
    }
    return try attentionCacheKind(parameters: parameters)
}

/// Full-attention cache if the layer is not sliding-window; otherwise the
/// architecture sliding window capped by legacy `maxKVSize`.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func makeHybridAttentionKVCache(
    parameters: GenerateParameters?,
    slidingWindow: Int?,
    usesSlidingWindow: Bool
) throws -> KVCache {
    if usesSlidingWindow, let slidingWindow {
        return try makeSlidingWindowKVCache(parameters: parameters, window: slidingWindow)
    }
    return try makeAttentionKVCache(parameters: parameters)
}

/// The smaller of the architecture window and legacy `maxKVSize`.
///
/// Typed cache capacity deliberately does not participate: model-native sliding
/// windows retain their architecture-defined window under `KVCacheConfiguration`.
private func effectiveSlidingWindow(parameters: GenerateParameters?, window: Int) throws -> Int {
    guard window > 0 else {
        throw KVCacheConfigurationError.invalidSlidingWindow(window)
    }
    // Validate the whole request, including conflicts and compression fields,
    // before a cache constructor can observe unchecked legacy values.
    _ = try parameters?.kvCachePlan()
    if let maxKVSize = parameters?.maxKVSize {
        return min(window, maxKVSize)
    }
    return window
}
