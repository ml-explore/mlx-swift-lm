// Copyright © 2026 Apple Inc.

import MLX

/// Estimates of what generation with a model will need, computed before it runs.
///
/// Obtain one through ``LanguageModel/estimator(parameters:)`` or
/// ``ModelContainer/estimator(parameters:)``. It follows the caches the model
/// builds for the given parameters and describes one sequence. Nothing comes from
/// a running conversation; for what a cache holds now, use ``KVCacheStatus``.
///
/// ```swift
/// let estimator = try await container.estimator(parameters: parameters)
/// let bytes = estimator.cache.bytes(tokens: 32_768)
/// let maxTokens = estimator.cache.maxTokens(fittingIn: budget)
/// let perToken = estimator.decode.bytes(tokens: 32_768, weightBytes: weights)
/// ```
public struct GenerationEstimator: Sendable, Hashable {
    /// The memory the cache will hold as the context grows.
    public let cache: Cache

    /// What one decoded token reads.
    public let decode: Decode

    /// Estimate from the layers a model would use.
    ///
    /// Combine the layers of a target model and its drafter to estimate both.
    public init(cache: Cache) {
        self.cache = cache
        self.decode = Decode(layers: cache.layers)
    }
}

extension GenerationEstimator {
    /// What one decoded token reads.
    ///
    /// A decode step reads the model's weights and the cache it attends to. The
    /// weights belong to the model: their bytes come from the caller, because a
    /// mixture-of-experts model reads only the experts it routes to, and a
    /// vision tower is not read while decoding text. The cache it attends to
    /// leaves out a rotating layer's rewind reserve, which is stored but not
    /// read.
    public struct Decode: Sendable, Hashable {
        /// The layers the step reads from.
        let layers: [Cache.Layer]

        /// Bytes read for one decoded token, weights included.
        ///
        /// - Parameters:
        ///   - tokens: The context length the step runs at.
        ///   - weightBytes: Bytes the model's weights contribute to one decoded
        ///     token.
        public func bytes(tokens: Int, weightBytes: Int) -> Int {
            weightBytes + layers.reduce(0) { $0 + $1.readBytes(tokens: tokens) }
        }
    }
}

extension GenerationEstimator {
    /// The memory a cache will hold as the context grows.
    ///
    /// Sliding windows, recurrent state, the requested capacity and rewind reserve,
    /// and the ``KVCacheConfiguration`` strategy are accounted for.
    public struct Cache: Sendable, Hashable {
        /// The memory one cache layer will hold.
        public struct Layer: Sendable, Hashable {
            /// The layer's position, as in ``KVCacheLayerStatus/path``.
            public let path: [Int]

            public let kind: CacheLayerKind

            /// How the layer stores keys and values. `nil` for recurrent state, custom
            /// caches, and layers that store nothing.
            public let strategy: KVCacheStrategyIdentifier?

            /// Bytes each retained token adds.
            public let bytesPerToken: Int

            /// Bytes that do not grow with the context, such as recurrent state.
            public let fixedBytes: Int

            /// The most tokens the layer retains, including any rewind reserve.
            /// `nil` when the layer is unbounded.
            public let capacity: Int?

            public init(
                path: [Int],
                kind: CacheLayerKind,
                strategy: KVCacheStrategyIdentifier?,
                bytesPerToken: Int,
                fixedBytes: Int,
                capacity: Int?
            ) {
                self.path = path
                self.kind = kind
                self.strategy = strategy
                self.bytesPerToken = bytesPerToken
                self.fixedBytes = fixedBytes
                self.capacity = capacity
            }

            /// Bytes the layer holds after `tokens` tokens.
            public func bytes(tokens: Int) -> Int {
                fixedBytes + bytesPerToken * min(max(tokens, 0), capacity ?? .max)
            }

            /// Bytes the layer reads for one decoded token.
            ///
            /// A rotating layer keeps a rewind reserve that a decode step does not
            /// attend to.
            func readBytes(tokens: Int) -> Int {
                let tokens = max(tokens, 0)
                let attended: Int
                switch kind {
                case .rotatingAttention(let maxSize, _):
                    attended = min(tokens, maxSize)
                case .attention, .stateSpace, .composite:
                    attended = min(tokens, capacity ?? .max)
                }
                return fixedBytes + bytesPerToken * attended
            }
        }

        public let layers: [Layer]

        /// Combine layers, for example a target model's and its drafter's.
        public init(layers: [Layer]) {
            self.layers = layers
        }

        /// Bytes the cache holds after `tokens` tokens.
        ///
        /// Caches allocate in steps, usually of 256 tokens, so a layer can hold up to one
        /// step more than this.
        public func bytes(tokens: Int) -> Int {
            layers.reduce(0) { $0 + $1.bytes(tokens: tokens) }
        }

        /// Bytes the cache grows by for each token of context.
        ///
        /// Layers that cap the context stop growing at their capacity.
        public var bytesPerToken: Int {
            layers.reduce(0) { $0 + $1.bytesPerToken }
        }

        /// The longest context whose cache fits in `bytes`.
        ///
        /// Returns `nil` when every context fits because no layer grows without bound,
        /// and `0` when not even an empty cache fits.
        public func maxTokens(fittingIn bytes: Int) -> Int? {
            let growth = layers.filter { $0.capacity == nil }.reduce(0) { $0 + $1.bytesPerToken }
            var high: Int
            if growth > 0 {
                // Unbounded layers alone exceed `bytes` past this length.
                high = max(bytes, 0) / growth
            } else {
                let saturation = layers.compactMap(\.capacity).max() ?? 0
                if let total = exactBytes(tokens: saturation), total <= bytes { return nil }
                high = saturation
            }

            var low = 0
            while low < high {
                let middle = low + (high - low + 1) / 2
                if let total = exactBytes(tokens: middle), total <= bytes {
                    low = middle
                } else {
                    high = middle - 1
                }
            }
            return low
        }

        /// ``bytes(tokens:)``, or `nil` when it overflows `Int`.
        private func exactBytes(tokens: Int) -> Int? {
            var total = 0
            for layer in layers {
                let retained = min(max(tokens, 0), layer.capacity ?? .max)
                let (product, overflow) = layer.bytesPerToken.multipliedReportingOverflow(
                    by: retained)
                guard !overflow, total <= .max - product - layer.fixedBytes else { return nil }
                total += product + layer.fixedBytes
            }
            return total
        }
    }
}

extension GenerationEstimator.Cache {
    /// Measure `cache` after a probe forward pass has written its shapes.
    ///
    /// Compressed layers are measured on synthetic keys and values written into
    /// `makeCache()`, so the model's own forward pass is never evaluated.
    init(
        probed cache: [KVCache],
        configuration: KVCacheConfiguration?,
        makeCache: () throws -> [KVCache]
    ) throws {
        let leaves = KVCacheTree.leaves(in: cache)
        let compressed =
            if let configuration, configuration.strategy.identifier != .fullPrecision {
                try Self.measureCompression(
                    leaves: leaves, configuration: configuration, makeCache: makeCache)
            } else {
                [[Int]: Measurement]()
            }
        self.init(layers: leaves.map { $0.estimatedLayer(compressed: compressed[$0.path]) })
    }

    struct Measurement {
        let strategy: KVCacheStrategyIdentifier?
        let bytesPerToken: Int
        let fixedBytes: Int
    }

    private struct Geometry: Hashable {
        let keyShape: [Int]
        let valueShape: [Int]
        let keyType: DType
        let valueType: DType
        let isProtected: Bool

        func keys(tokens: Int) -> MLXArray {
            sample(keyShape, keyType, tokens: tokens)
        }

        func values(tokens: Int) -> MLXArray {
            sample(valueShape, valueType, tokens: tokens)
        }

        private func sample(_ shape: [Int], _ type: DType, tokens: Int) -> MLXArray {
            var shape = shape
            shape[2] = tokens
            return MLXRandom.normal(shape, dtype: type, key: MLXRandom.key(0))
        }
    }

    /// Two token counts that are multiples of every allocation step and tile size.
    private static let sampleTokens = (256, 512)

    /// Write each distinct layer geometry at two lengths, apply the strategy as the
    /// decode loop does, and fit bytes per token from the difference.
    private static func measureCompression(
        leaves: [KVCacheLeaf],
        configuration: KVCacheConfiguration,
        makeCache: () throws -> [KVCache]
    ) throws -> [[Int]: Measurement] {
        let protectedPaths: Set<[Int]> =
            if case .turboQuant(let turbo) = configuration.strategy.storage {
                KVCacheTree.turboQuantProtectedPaths(
                    in: leaves,
                    keyBits: turbo.keyPrecision.bitWidth,
                    valueBits: turbo.valuePrecision.bitWidth)
            } else {
                []
            }

        var geometries = [[Int]: Geometry]()
        var representatives = [Geometry: [Int]]()
        for leaf in leaves {
            guard case .simple = leaf.kind else { continue }
            let state = leaf.cache.innerState()
            guard state.count == 2, state.allSatisfy({ $0.ndim == 4 }) else { continue }
            let geometry = Geometry(
                keyShape: state[0].shape, valueShape: state[1].shape,
                keyType: state[0].dtype, valueType: state[1].dtype,
                isProtected: protectedPaths.contains(leaf.path))
            geometries[leaf.path] = geometry
            if representatives[geometry] == nil {
                representatives[geometry] = leaf.path
            }
        }
        guard !representatives.isEmpty else { return [:] }

        struct Sample {
            let strategy: KVCacheStrategyIdentifier?
            let tokens: Int
            let bytes: Int
        }

        func sample(tokens: Int) throws -> [[Int]: Sample] {
            var cache = try makeCache()
            for leaf in KVCacheTree.leaves(in: cache) {
                guard let geometry = geometries[leaf.path],
                    representatives[geometry] == leaf.path
                else { continue }
                _ = leaf.cache.update(
                    keys: geometry.keys(tokens: tokens - 1),
                    values: geometry.values(tokens: tokens - 1))
            }
            applyKVCacheConfigurationFast(
                cache: &cache, configuration: configuration, compressionStart: 0)

            var samples = [[Int]: Sample]()
            for leaf in KVCacheTree.leaves(in: cache) {
                guard let geometry = geometries[leaf.path],
                    representatives[geometry] == leaf.path
                else { continue }
                var sampledTokens = tokens - 1
                // These caches compress on their first decode write.
                if leaf.cache is any SharedAttentionKVCache {
                    _ = AttentionKVState.update(
                        keys: geometry.keys(tokens: 1), values: geometry.values(tokens: 1),
                        cache: leaf.cache)
                    sampledTokens += 1
                }
                samples[leaf.path] = Sample(
                    strategy: leaf.strategy, tokens: sampledTokens,
                    bytes: leaf.cache.innerState().reduce(0) { $0 + $1.nbytes })
            }
            return samples
        }

        let (shortTokens, longTokens) = sampleTokens
        let short = try sample(tokens: shortTokens)
        let long = try sample(tokens: longTokens)

        var measurements = [[Int]: Measurement]()
        for (path, geometry) in geometries {
            guard let representative = representatives[geometry],
                let short = short[representative], let long = long[representative],
                short.strategy != .fullPrecision
            else { continue }
            let span = long.tokens - short.tokens
            let bytesPerToken = (long.bytes - short.bytes + span - 1) / span
            measurements[path] = Measurement(
                strategy: short.strategy,
                bytesPerToken: bytesPerToken,
                fixedBytes: max(0, short.bytes - bytesPerToken * short.tokens))
        }
        return measurements
    }
}

extension KVCacheLeaf {
    fileprivate var strategy: KVCacheStrategyIdentifier? {
        switch kind {
        case .simple, .rotating: .fullPrecision
        case .affine: .affine
        case .turboQuant: .turboQuant
        case .varianceNormalized: .varianceNormalized
        case .recurrent, .unsupported: nil
        }
    }

    fileprivate func estimatedLayer(compressed: GenerationEstimator.Cache.Measurement?)
        -> GenerationEstimator.Cache.Layer
    {
        let state = cache.innerState()
        let kind = CacheLayerKind(cache: cache)
        guard isAttentionCache else {
            return .init(
                path: path, kind: kind, strategy: nil, bytesPerToken: 0,
                fixedBytes: state.reduce(0) { $0 + $1.nbytes }, capacity: nil)
        }

        let capacity =
            if case .rotating(let rotating) = self.kind {
                rotating.maxSize.map { $0 + rotating.rewindCapacity }
            } else {
                cache.maxSize
            }
        if let compressed {
            return .init(
                path: path, kind: kind, strategy: compressed.strategy,
                bytesPerToken: compressed.bytesPerToken, fixedBytes: compressed.fixedBytes,
                capacity: capacity)
        }
        // Attention caches store tokens on axis 2 of `[batch, heads, tokens, dim]`.
        let bytesPerToken = state.reduce(0) { total, array in
            array.ndim == 4 && array.dim(2) > 0 ? total + array.nbytes / array.dim(2) : total
        }
        return .init(
            path: path, kind: kind, strategy: state.isEmpty ? nil : strategy,
            bytesPerToken: bytesPerToken, fixedBytes: 0, capacity: capacity)
    }
}
