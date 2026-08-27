// Copyright © 2026 Apple Inc.

import Foundation

/// A complete request for key-value cache storage during generation.
///
/// This keeps cache capacity, compression, and compatibility semantics in one
/// value so every generation path forwards the same configuration.
public struct KVCacheConfiguration: Sendable, Hashable {
    public var capacity: Capacity?
    public var strategy: Strategy
    public var compatibility: CompatibilityPolicy

    public init(
        capacity: Capacity? = nil,
        strategy: Strategy = .fullPrecision,
        compatibility: CompatibilityPolicy = .requireAtLeastOneLayer
    ) {
        self.capacity = capacity
        self.strategy = strategy
        self.compatibility = compatibility
    }

    /// Maximum resident size for caches whose capacity is caller-configurable.
    ///
    /// Model-native sliding-window layers retain their architecture-defined
    /// window and prefix behavior. This value bounds the remaining attention
    /// layers created by the model's `newCache(parameters:)` implementation.
    public struct Capacity: Sendable, Hashable {
        public let maxTokens: Int
        public let preservedPrefixTokens: Int

        public init(maxTokens: Int, preservedPrefixTokens: Int = 4) throws {
            guard maxTokens > 0 else {
                throw KVCacheConfigurationError.invalidCapacity(maxTokens)
            }
            guard preservedPrefixTokens >= 0, preservedPrefixTokens < maxTokens else {
                throw KVCacheConfigurationError.invalidPreservedPrefix(
                    preservedPrefixTokens, capacity: maxTokens)
            }
            self.maxTokens = maxTokens
            self.preservedPrefixTokens = preservedPrefixTokens
        }

        package init(uncheckedMaxTokens maxTokens: Int, preservedPrefixTokens: Int = 4) {
            self.maxTokens = maxTokens
            self.preservedPrefixTokens = preservedPrefixTokens
        }
    }

    /// Behavior when only part of a model's cache topology supports the strategy.
    public enum CompatibilityPolicy: Sendable, Hashable {
        /// Apply the strategy where supported and retain other layers unchanged.
        case allowPartial
        /// Reject the request unless at least one attention layer can use it.
        case requireAtLeastOneLayer
        /// Reject the request unless every attention layer can use it.
        case requireAllLayers
    }

    /// Opaque cache strategy. New strategies can be added without introducing
    /// public enum cases that clients must exhaustively switch over.
    public struct Strategy: Sendable, Hashable {
        package enum Storage: Sendable, Hashable {
            case fullPrecision
            case affine(AffineKVCacheConfiguration)
            case turboQuant(TurboQuantKVCacheConfiguration)
            case varianceNormalized(VarianceNormalizedKVCacheConfiguration)
        }

        package let storage: Storage

        private init(storage: Storage) {
            self.storage = storage
        }

        public static let fullPrecision = Strategy(storage: .fullPrecision)

        public static func affine(_ configuration: AffineKVCacheConfiguration) -> Strategy {
            Strategy(storage: .affine(configuration))
        }

        public static func turboQuant(
            _ configuration: TurboQuantKVCacheConfiguration
        ) -> Strategy {
            Strategy(storage: .turboQuant(configuration))
        }

        public static func varianceNormalized(
            _ configuration: VarianceNormalizedKVCacheConfiguration
        ) -> Strategy {
            Strategy(storage: .varianceNormalized(configuration))
        }

        /// Stable algorithm identity for diagnostics and persistence adapters.
        public var identifier: KVCacheStrategyIdentifier {
            switch storage {
            case .fullPrecision: .fullPrecision
            case .affine: .affine
            case .turboQuant: .turboQuant
            case .varianceNormalized: .varianceNormalized
            }
        }

        package var compressionStart: Int {
            switch storage {
            case .fullPrecision: 0
            case .affine(let configuration): configuration.compressionStart
            case .turboQuant(let configuration): configuration.compressionStart
            case .varianceNormalized(let configuration): configuration.compressionStart
            }
        }
    }
}

/// Stable, open-ended identity for a resolved cache strategy.
public struct KVCacheStrategyIdentifier: Sendable, Hashable, CustomStringConvertible {
    public let rawValue: String

    private init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    public static let fullPrecision = KVCacheStrategyIdentifier("full-precision")
    public static let affine = KVCacheStrategyIdentifier("affine")
    public static let turboQuant = KVCacheStrategyIdentifier("turbo-quant")
    public static let varianceNormalized = KVCacheStrategyIdentifier("variance-normalized")

    public var description: String { rawValue }
}

/// Affine cache compression settings.
public struct AffineKVCacheConfiguration: Sendable, Hashable {
    private static let supportedBitWidths: Set<Int> = [2, 3, 4, 5, 6, 8]

    public let bits: Int
    public let groupSize: Int
    public let compressionStart: Int

    public init(bits: Int, groupSize: Int = 64, compressionStart: Int = 0) throws {
        guard Self.supportedBitWidths.contains(bits) else {
            throw KVCacheConfigurationError.invalidAffineBits(bits)
        }
        guard groupSize > 0 else {
            throw KVCacheConfigurationError.invalidGroupSize(groupSize)
        }
        guard compressionStart >= 0 else {
            throw KVCacheConfigurationError.invalidCompressionStart(compressionStart)
        }
        self.bits = bits
        self.groupSize = groupSize
        self.compressionStart = compressionStart
    }

    package init(uncheckedBits bits: Int, groupSize: Int, compressionStart: Int) {
        self.bits = bits
        self.groupSize = groupSize
        self.compressionStart = compressionStart
    }

    public static let fourBit = AffineKVCacheConfiguration(
        uncheckedBits: 4, groupSize: 64, compressionStart: 0)
    public static let eightBit = AffineKVCacheConfiguration(
        uncheckedBits: 8, groupSize: 64, compressionStart: 0)
}

/// TurboQuant cache compression settings.
public struct TurboQuantKVCacheConfiguration: Sendable, Hashable {
    /// Opaque key encoding precision. Additional encodings can be introduced
    /// without adding public enum cases that clients must exhaustively handle.
    public struct KeyPrecision: Sendable, Hashable {
        package let bitWidth: Int

        private init(bitWidth: Int) {
            self.bitWidth = bitWidth
        }

        /// Keep keys in FP16; only values are compressed.
        public static let fp16 = KeyPrecision(bitWidth: 0)
        /// TurboQuant-compressed keys.
        public static let twoBit = KeyPrecision(bitWidth: 2)
        public static let threeBit = KeyPrecision(bitWidth: 3)
        public static let fourBit = KeyPrecision(bitWidth: 4)
        /// Affine 8-bit keys with TurboQuant-compressed values.
        public static let affineEightBit = KeyPrecision(bitWidth: 8)

        package init?(legacyBitWidth: Int) {
            switch legacyBitWidth {
            case 0: self = .fp16
            case 2: self = .twoBit
            case 3: self = .threeBit
            case 4: self = .fourBit
            case 8: self = .affineEightBit
            default: return nil
            }
        }
    }

    /// Opaque value encoding precision.
    public struct ValuePrecision: Sendable, Hashable {
        package let bitWidth: Int

        private init(bitWidth: Int) {
            self.bitWidth = bitWidth
        }

        public static let twoBit = ValuePrecision(bitWidth: 2)
        public static let threeBit = ValuePrecision(bitWidth: 3)
        public static let fourBit = ValuePrecision(bitWidth: 4)

        package init?(legacyBitWidth: Int) {
            switch legacyBitWidth {
            case 2: self = .twoBit
            case 3: self = .threeBit
            case 4: self = .fourBit
            default: return nil
            }
        }
    }

    public let keyPrecision: KeyPrecision
    public let valuePrecision: ValuePrecision
    public let compressionStart: Int

    public init(
        keyPrecision: KeyPrecision,
        valuePrecision: ValuePrecision,
        compressionStart: Int = 0
    ) throws {
        guard compressionStart >= 0 else {
            throw KVCacheConfigurationError.invalidCompressionStart(compressionStart)
        }
        self.keyPrecision = keyPrecision
        self.valuePrecision = valuePrecision
        self.compressionStart = compressionStart
    }

    private init(
        uncheckedKeyPrecision keyPrecision: KeyPrecision,
        valuePrecision: ValuePrecision,
        compressionStart: Int
    ) {
        self.keyPrecision = keyPrecision
        self.valuePrecision = valuePrecision
        self.compressionStart = compressionStart
    }

    /// FP16 keys and 4-bit values; the conservative quality-first preset.
    public static let qualityFirst = TurboQuantKVCacheConfiguration(
        uncheckedKeyPrecision: .fp16, valuePrecision: .fourBit, compressionStart: 0)

    /// Affine 8-bit keys and 3-bit values; the recommended general preset.
    public static let balanced = TurboQuantKVCacheConfiguration(
        uncheckedKeyPrecision: .affineEightBit,
        valuePrecision: .threeBit,
        compressionStart: 0)

    /// Affine 8-bit keys and 2-bit values for memory-bound long contexts.
    public static let memoryFirst = TurboQuantKVCacheConfiguration(
        uncheckedKeyPrecision: .affineEightBit,
        valuePrecision: .twoBit,
        compressionStart: 0)
}

/// Variance-normalized (KVarN-inspired) cache compression settings.
///
/// Completed tiles are rotated, dual-axis variance-normalized, and quantized
/// asymmetrically (keys and values may use different bit widths). The intended
/// use case is memory-constrained long-context decoding rather than latency-
/// sensitive generation.
///
/// - Warning: This strategy is experimental. Models must route attention through
///   `attentionWithCacheUpdate`; direct `KVCache.update` call sites use a full-cache
///   materialization fallback that is not suitable for long-context decode.
public struct VarianceNormalizedKVCacheConfiguration: Sendable, Hashable {
    private static let supportedBitWidths: Set<Int> = [2, 3, 4, 5, 6, 8]
    private static let supportedTileSizes: Set<Int> = [32, 64, 128]

    public let keyBits: Int
    public let valueBits: Int
    public let tileSize: Int
    public let sinkhornIterations: Int
    public let compressionStart: Int

    public init(
        keyBits: Int = 4,
        valueBits: Int = 2,
        tileSize: Int = 128,
        sinkhornIterations: Int = 8,
        compressionStart: Int = 0
    ) throws {
        guard Self.supportedBitWidths.contains(keyBits) else {
            throw KVCacheConfigurationError.invalidAffineBits(keyBits)
        }
        guard Self.supportedBitWidths.contains(valueBits) else {
            throw KVCacheConfigurationError.invalidAffineBits(valueBits)
        }
        guard Self.supportedTileSizes.contains(tileSize) else {
            throw KVCacheConfigurationError.invalidTileSize(tileSize)
        }
        guard sinkhornIterations > 0 else {
            throw KVCacheConfigurationError.invalidSinkhornIterations(sinkhornIterations)
        }
        guard compressionStart >= 0 else {
            throw KVCacheConfigurationError.invalidCompressionStart(compressionStart)
        }
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.tileSize = tileSize
        self.sinkhornIterations = sinkhornIterations
        self.compressionStart = compressionStart
    }

    package init(
        uncheckedKeyBits keyBits: Int,
        valueBits: Int,
        tileSize: Int,
        sinkhornIterations: Int,
        compressionStart: Int
    ) {
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.tileSize = tileSize
        self.sinkhornIterations = sinkhornIterations
        self.compressionStart = compressionStart
    }

    /// 4-bit keys and 2-bit values with 128-token tiles; the paper-inspired default.
    public static let memoryFirst = VarianceNormalizedKVCacheConfiguration(
        uncheckedKeyBits: 4, valueBits: 2, tileSize: 128, sinkhornIterations: 8,
        compressionStart: 0)

    /// 4-bit keys and values for higher fidelity at still-reduced memory.
    public static let balanced = VarianceNormalizedKVCacheConfiguration(
        uncheckedKeyBits: 4, valueBits: 4, tileSize: 128, sinkhornIterations: 8,
        compressionStart: 0)
}

/// Validation failures for typed or legacy KV-cache configuration.
public enum KVCacheConfigurationError: Error, Sendable, Equatable, LocalizedError {
    case conflictingLegacyConfiguration
    case invalidCapacity(Int)
    case invalidSlidingWindow(Int)
    case invalidPreservedPrefix(Int, capacity: Int)
    case invalidAffineBits(Int)
    case invalidGroupSize(Int)
    case invalidTileSize(Int)
    case invalidSinkhornIterations(Int)
    case invalidCompressionStart(Int)
    case unsupportedLegacyScheme(String)
    case incompatibleCapacity(expected: Int, count: Int)
    case noCompatibleLayers(strategy: KVCacheStrategyIdentifier)
    case incompatibleLayers(strategy: KVCacheStrategyIdentifier, count: Int)

    public var errorDescription: String? {
        switch self {
        case .conflictingLegacyConfiguration:
            "Set either GenerateParameters.kvCache or the legacy KV-cache fields, not both."
        case .invalidCapacity(let value):
            "KV-cache capacity must be positive; received \(value)."
        case .invalidSlidingWindow(let value):
            "Model sliding-window size must be positive; received \(value)."
        case .invalidPreservedPrefix(let value, let capacity):
            "Preserved prefix \(value) must be non-negative and smaller than capacity \(capacity)."
        case .invalidAffineBits(let value):
            "Affine KV-cache bit width must be one of 2, 3, 4, 5, 6, or 8; received \(value)."
        case .invalidGroupSize(let value):
            "KV-cache group size must be positive; received \(value)."
        case .invalidTileSize(let value):
            "Variance-normalized KV-cache tile size must be 32, 64, or 128; received \(value)."
        case .invalidSinkhornIterations(let value):
            "Variance-normalized Sinkhorn iterations must be positive; received \(value)."
        case .invalidCompressionStart(let value):
            "KV-cache compression start must be non-negative; received \(value)."
        case .unsupportedLegacyScheme(let value):
            "Unsupported legacy KV-cache scheme: \(value)."
        case .incompatibleCapacity(let expected, let count):
            "KV-cache capacity \(expected) is not realized by \(count) attention layer(s)."
        case .noCompatibleLayers(let strategy):
            "KV-cache strategy \(strategy) is not supported by any attention layer."
        case .incompatibleLayers(let strategy, let count):
            "KV-cache strategy \(strategy) is unsupported by \(count) attention layer(s)."
        }
    }
}

/// Effective state of one leaf in a model's cache topology.
///
/// This is shared by the high-level ``KVCacheStatus`` API and the lower-level
/// ``KVCacheRuntimeReport`` compatibility view.
public struct KVCacheLayerStatus: Sendable, Hashable {
    public enum State: Sendable, Hashable {
        case active
        case pending
        case skipped
        case notApplicable
    }

    public enum Reason: Sendable, Hashable {
        case awaitingCompressionStart
        case boundaryProtection
        case slidingWindow
        case unsupportedShape
        case differentStrategy
        case nonAttentionState
    }

    /// Where an attention layer's resident-token bound originated.
    public enum CapacitySource: Sendable, Hashable {
        /// The attention cache grows without an explicit resident-token bound.
        case unbounded
        /// The model architecture defines the bound, such as sliding-window attention.
        case modelDefined
        /// The caller requested the bound through generation parameters.
        case requested
        /// A model-specific cache implementation defines the bound.
        case implementationDefined
    }

    /// Stable path through top-level and composite cache entries.
    public let path: [Int]
    public let kind: CacheLayerKind
    /// `nil` for recurrent or other non-attention state.
    public let capacitySource: CapacitySource?
    public let state: State
    public let resolvedStrategy: KVCacheStrategyIdentifier?
    public let reason: Reason?

    package init(
        path: [Int],
        kind: CacheLayerKind,
        capacitySource: CapacitySource?,
        state: State,
        resolvedStrategy: KVCacheStrategyIdentifier?,
        reason: Reason?
    ) {
        self.path = path
        self.kind = kind
        self.capacitySource = capacitySource
        self.state = state
        self.resolvedStrategy = resolvedStrategy
        self.reason = reason
    }
}

/// Effective per-layer state for a configured KV cache.
///
/// Prefer ``KVCacheStatus`` for application diagnostics. This report remains
/// useful when directly applying a configuration to a raw cache array.
public struct KVCacheRuntimeReport: Sendable, Hashable {
    public typealias Layer = KVCacheLayerStatus

    public let requestedConfiguration: KVCacheConfiguration
    public let layers: [Layer]
    /// Authoritative model-wide position when the report comes from shared
    /// cache storage. Raw array reports leave this value `nil`.
    public let processedTokenCount: Int?

    package init(
        requestedConfiguration: KVCacheConfiguration,
        layers: [Layer],
        processedTokenCount: Int? = nil
    ) {
        self.requestedConfiguration = requestedConfiguration
        self.layers = layers
        self.processedTokenCount = processedTokenCount
    }

    public var compressedLayerCount: Int {
        layers.filter {
            $0.state == .active && $0.resolvedStrategy != .fullPrecision
        }.count
    }

    public var pendingLayerCount: Int {
        layers.filter { $0.state == .pending }.count
    }

    public var skippedLayerCount: Int {
        layers.filter { $0.state == .skipped }.count
    }
}

/// The observed outcome of applying a cache configuration to a realized cache.
public struct KVCacheApplicationResult: Sendable, Hashable {
    public let convertedLayerCount: Int
    public let alreadyCompatibleLayerCount: Int
    public let pendingLayerCount: Int
    public let skipped: [KVCacheRuntimeReport.Layer]
}
