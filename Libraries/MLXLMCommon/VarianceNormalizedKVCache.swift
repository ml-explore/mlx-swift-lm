// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct VarianceNormalizedKVTile {
    var keyWeight: MLXArray
    var keyScales: MLXArray
    var keyBiases: MLXArray
    var keyColumnScales: MLXArray

    var valueWeight: MLXArray
    var valueScales: MLXArray
    var valueBiases: MLXArray
    var valueColumnScales: MLXArray
}

private struct VarianceNormalizedKVSlab {
    var records: VarianceNormalizedKVTile
    let count: Int
}

private let varianceNormalizationEpsilon: Float = 1e-8
private let varianceNormalizationMinimumStandardDeviation: Float = 1e-3
private let varianceNormalizationMaximumStandardDeviation: Float = 1e3
private let varianceNormalizationMinimumLogScale: Float = -0.3
private let varianceNormalizationMaximumLogScale: Float = 10

private func isPowerOfTwo(_ value: Int) -> Bool {
    value > 0 && (value & (value - 1)) == 0
}

func isSupportedVarianceNormalizedDType(_ dtype: DType) -> Bool {
    dtype == .float16 || dtype == .bfloat16 || dtype == .float32
}

private func varianceNormalizedDTypeName(_ dtype: DType?) -> String {
    dtype.map { String(describing: $0) } ?? "none"
}

func varianceNormalizedDType(named name: String) -> DType? {
    guard name != "none" else { return nil }
    return DType.allCases.first { String(describing: $0) == name }
}

private func varianceNormalizedValueGroupSize(valueHeadDim: Int) -> Int? {
    resolvedKVQuantizationGroupSize(
        requested: min(128, max(32, valueHeadDim)),
        keyHeadDim: valueHeadDim,
        valueHeadDim: valueHeadDim
    )
}

func supportsVarianceNormalizedKVCache(
    keyHeadDim: Int,
    valueHeadDim: Int,
    tileSize: Int
) -> Bool {
    [32, 64, 128].contains(tileSize)
        && keyHeadDim > 0
        && valueHeadDim > 0
        && isPowerOfTwo(keyHeadDim)
        && isPowerOfTwo(valueHeadDim)
        && varianceNormalizedValueGroupSize(valueHeadDim: valueHeadDim) != nil
}

extension KVCacheSimple {
    /// Convert to a variance-normalized tile cache.
    public func toVarianceNormalized(
        tileSize: Int = 128,
        keyBits: Int = 4,
        valueBits: Int = 2,
        sinkhornIterations: Int = 8
    ) -> VarianceNormalizedKVCache {
        let cache = VarianceNormalizedKVCache(
            tileSize: tileSize,
            keyBits: keyBits,
            valueBits: valueBits,
            sinkhornIterations: sinkhornIterations)

        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            _ = cache.update(keys: currentKeys, values: currentValues)
        } else {
            cache.offset = self.offset
        }

        return cache
    }
}

/// Variance-normalized KV cache following KVarN's core tile-compression steps.
///
/// This is intentionally implemented as a regular ``KVCache`` so model attention code can
/// use it without changes. When called through
/// ``attentionWithCacheUpdate(queries:keys:values:cache:scale:mask:)``, completed tiles stay
/// quantized on the hot path: attention scores and value products are computed with `quantizedMM`
/// in the rotated domain, and only the final output is inverse-rotated. The tile normalization uses
/// clipped log-domain dual-axis scale balancing with best-imbalance tracking, matching the central
/// KVarN VarN algorithm more closely while remaining an MLX Swift implementation. The RTN affine
/// scale/bias are fused into the matching variance-normalization axis so each completed tile stores
/// compact K/V records: packed weights, fused affine scale, fused affine bias, and the remaining
/// variance scale.
///
/// - Warning: This cache remains experimental pending model-level long-context quality gates and
///   a fused MLX/Metal attention implementation. Use `attentionWithCacheUpdate` for decode;
///   `update(keys:values:)` is a compatibility slow path.
public class VarianceNormalizedKVCache: BaseKVCache, KVCacheAttentionProtocol,
    CustomDebugStringConvertible
{
    static let compactTileStateCount = 8
    static let legacyTileStateCount = 10
    static let tailStateCount = 2
    static let metadataVersion = 1
    static let compressionBatchTileCount = 32
    static let compressionBatchHeadCount = 4
    static let baseSlabTileCount = 4
    static let slabFanout = 8

    private var tileSlabs: [VarianceNormalizedKVSlab] = []
    private var pendingTiles: [VarianceNormalizedKVTile] = []
    private var tailKeys: MLXArray?
    private var tailValues: MLXArray?
    private var keyDType: DType?
    private var valueDType: DType?
    private var restoredTileCount: Int?
    private var restoredTailLength: Int?
    public let tileSize: Int
    public let keyBits: Int
    public let valueBits: Int
    public let sinkhornIterations: Int

    public init(
        tileSize: Int = 128,
        keyBits: Int = 4,
        valueBits: Int = 2,
        sinkhornIterations: Int = 8
    ) {
        precondition(
            (try? VarianceNormalizedKVCacheConfiguration(
                keyBits: keyBits,
                valueBits: valueBits,
                tileSize: tileSize,
                sinkhornIterations: sinkhornIterations)) != nil,
            "VarianceNormalizedKVCache requires supported bit widths and tile size, plus a positive iteration count"
        )
        self.tileSize = tileSize
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.sinkhornIterations = sinkhornIterations
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    private func rotate(_ x: MLXArray) -> MLXArray {
        let dimensions = x.dim(-1)
        guard x.size > 0, isPowerOfTwo(dimensions) else { return x }
        return hadamardTransform(x)
    }

    private func inverseRotate(_ x: MLXArray) -> MLXArray {
        // Sylvester Hadamard is symmetric and orthonormal, so the inverse is itself.
        rotate(x)
    }

    private func balanced(
        original: MLXArray,
        columnScales: MLXArray,
        rowScales: MLXArray
    ) -> MLXArray {
        original / columnScales / rowScales
    }

    private func clippedStd(_ x: MLXArray, axis: Int) -> MLXArray {
        clip(
            maximum(
                std(x, axis: axis, keepDims: true, ddof: 1),
                MLXArray(varianceNormalizationEpsilon)),
            min: MLXArray(varianceNormalizationMinimumStandardDeviation),
            max: MLXArray(varianceNormalizationMaximumStandardDeviation))
    }

    private func varianceImbalance(_ x: MLXArray) -> MLXArray {
        let columnStd = std(x, axis: -2, keepDims: true, ddof: 1)
        let rowStd = std(x, axis: -1, keepDims: true, ddof: 1)
        let columnSpread =
            columnStd.max(axis: -1, keepDims: true)
            / maximum(
                columnStd.min(axis: -1, keepDims: true),
                MLXArray(varianceNormalizationEpsilon))
        let rowSpread =
            rowStd.max(axis: -2, keepDims: true)
            / maximum(
                rowStd.min(axis: -2, keepDims: true),
                MLXArray(varianceNormalizationEpsilon))
        return columnSpread + rowSpread
    }

    private func varianceNormalize(_ tile: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let original = tile.asType(.float32)
        var logColumnScales = MLXArray.zeros(
            std(original, axis: -2, keepDims: true, ddof: 1).shape)
        var logRowScales = MLXArray.zeros(
            std(original, axis: -1, keepDims: true, ddof: 1).shape)
        var columnScales = exp(logColumnScales)
        var rowScales = exp(logRowScales)
        var balancedTile = original
        var bestColumnScales = columnScales
        var bestRowScales = rowScales
        var bestImbalance = varianceImbalance(balancedTile)

        for _ in 0 ..< sinkhornIterations {
            logColumnScales = clip(
                logColumnScales + log(clippedStd(balancedTile, axis: -2)),
                min: MLXArray(varianceNormalizationMinimumLogScale),
                max: MLXArray(varianceNormalizationMaximumLogScale))
            columnScales = exp(logColumnScales)
            balancedTile = balanced(
                original: original, columnScales: columnScales, rowScales: rowScales)

            logRowScales = clip(
                logRowScales + log(clippedStd(balancedTile, axis: -1)),
                min: MLXArray(varianceNormalizationMinimumLogScale),
                max: MLXArray(varianceNormalizationMaximumLogScale))
            rowScales = exp(logRowScales)
            balancedTile = balanced(
                original: original, columnScales: columnScales, rowScales: rowScales)

            let imbalance = varianceImbalance(balancedTile)
            let useCandidate = imbalance .<= bestImbalance
            bestColumnScales = MLX.where(useCandidate, columnScales, bestColumnScales)
            bestRowScales = MLX.where(useCandidate, rowScales, bestRowScales)
            bestImbalance = MLX.where(useCandidate, imbalance, bestImbalance)
        }

        return (
            balanced(original: original, columnScales: bestColumnScales, rowScales: bestRowScales),
            bestColumnScales,
            bestRowScales
        )
    }

    private func quantizeBalanced(_ balanced: MLXArray, groupSize: Int, bits: Int) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let quantized = quantized(balanced, groupSize: groupSize, bits: bits, mode: .affine)
        return (
            quantized.wq,
            quantized.scales,
            quantized.biases
                ?? MLXArray.zeros(quantized.scales.shape, dtype: quantized.scales.dtype)
        )
    }

    private func compactAuxiliary(_ array: MLXArray) -> MLXArray {
        // The reference cache stores all auxiliary scales in FP16. Clamp before narrowing so a
        // pathological BF16/FP32 activation cannot serialize an infinity into persistent state.
        clip(array, min: MLXArray(-65_504), max: MLXArray(65_504)).asType(.float16)
    }

    /// Compress one or more complete tiles for a bounded group of KV heads. The tile axis is
    /// retained at index 2 so a prefill can be installed directly as immutable slabs.
    private func compressedHeadBatch(
        rotatedKeys: MLXArray, rotatedValues: MLXArray
    ) -> VarianceNormalizedKVTile {
        let tileCount = rotatedKeys.dim(2) / tileSize
        precondition(tileCount > 0 && rotatedKeys.dim(2).isMultiple(of: tileSize))

        let keyTile = rotatedKeys.reshaped([
            rotatedKeys.dim(0), rotatedKeys.dim(1), tileCount, tileSize, rotatedKeys.dim(3),
        ]).transposed(0, 1, 2, 4, 3)
        let (balancedKeys, keyColumnScales, keyRowScales) = varianceNormalize(keyTile)
        let (keyWeight, keyScales, keyBiases) = quantizeBalanced(
            balancedKeys, groupSize: tileSize, bits: keyBits)
        let fusedKeyScales = compactAuxiliary(keyScales * keyRowScales)
        let fusedKeyBiases = compactAuxiliary(keyBiases * keyRowScales)

        let valueTiles = rotatedValues.reshaped([
            rotatedValues.dim(0), rotatedValues.dim(1), tileCount, tileSize,
            rotatedValues.dim(3),
        ])
        let valueGroupSize = validatedValueGroupSize(rotatedValues.dim(3))
        let (balancedValues, valueColumnScales, valueRowScales) = varianceNormalize(valueTiles)
        let (valueWeight, valueScales, valueBiases) = quantizeBalanced(
            balancedValues, groupSize: valueGroupSize, bits: valueBits)
        let fusedValueScales = compactAuxiliary(valueScales * valueRowScales)
        let fusedValueBiases = compactAuxiliary(valueBiases * valueRowScales)

        return VarianceNormalizedKVTile(
            keyWeight: keyWeight,
            keyScales: fusedKeyScales,
            keyBiases: fusedKeyBiases,
            keyColumnScales: compactAuxiliary(keyColumnScales),
            valueWeight: valueWeight,
            valueScales: fusedValueScales,
            valueBiases: fusedValueBiases,
            valueColumnScales: compactAuxiliary(valueColumnScales)
        )
    }

    /// Compress complete tiles while bounding the FP32 normalization workspace. Variance
    /// normalization is independent across KV heads, so realizing four heads at a time preserves
    /// the exact algorithm and 32-tile batching without making all model heads live at once.
    private func compressedBatch(
        rotatedKeys: MLXArray, rotatedValues: MLXArray
    ) -> VarianceNormalizedKVTile {
        precondition(rotatedKeys.dim(1) == rotatedValues.dim(1))
        guard rotatedKeys.dim(1) > Self.compressionBatchHeadCount else {
            return compressedHeadBatch(rotatedKeys: rotatedKeys, rotatedValues: rotatedValues)
        }

        var headRecords: [VarianceNormalizedKVTile] = []
        for start in stride(
            from: 0, to: rotatedKeys.dim(1), by: Self.compressionBatchHeadCount)
        {
            let end = min(start + Self.compressionBatchHeadCount, rotatedKeys.dim(1))
            let record = compressedHeadBatch(
                rotatedKeys: rotatedKeys[0..., start ..< end, 0..., 0...],
                rotatedValues: rotatedValues[0..., start ..< end, 0..., 0...])
            evaluate(record)
            headRecords.append(record)
        }

        let merged = VarianceNormalizedKVTile(
            keyWeight: contiguous(concatenated(headRecords.map(\.keyWeight), axis: 1)),
            keyScales: contiguous(concatenated(headRecords.map(\.keyScales), axis: 1)),
            keyBiases: contiguous(concatenated(headRecords.map(\.keyBiases), axis: 1)),
            keyColumnScales: contiguous(
                concatenated(headRecords.map(\.keyColumnScales), axis: 1)),
            valueWeight: contiguous(concatenated(headRecords.map(\.valueWeight), axis: 1)),
            valueScales: contiguous(concatenated(headRecords.map(\.valueScales), axis: 1)),
            valueBiases: contiguous(concatenated(headRecords.map(\.valueBiases), axis: 1)),
            valueColumnScales: contiguous(
                concatenated(headRecords.map(\.valueColumnScales), axis: 1))
        )
        evaluate(merged)
        return merged
    }

    private func tileView(
        _ records: VarianceNormalizedKVTile, index: Int
    ) -> VarianceNormalizedKVTile {
        return VarianceNormalizedKVTile(
            keyWeight: records.keyWeight[0..., 0..., index, 0..., 0...],
            keyScales: records.keyScales[0..., 0..., index, 0..., 0...],
            keyBiases: records.keyBiases[0..., 0..., index, 0..., 0...],
            keyColumnScales: records.keyColumnScales[0..., 0..., index, 0..., 0...],
            valueWeight: records.valueWeight[0..., 0..., index, 0..., 0...],
            valueScales: records.valueScales[0..., 0..., index, 0..., 0...],
            valueBiases: records.valueBiases[0..., 0..., index, 0..., 0...],
            valueColumnScales: records.valueColumnScales[0..., 0..., index, 0..., 0...]
        )
    }

    private func evaluate(_ tile: VarianceNormalizedKVTile) {
        eval([
            tile.keyWeight, tile.keyScales, tile.keyBiases, tile.keyColumnScales,
            tile.valueWeight, tile.valueScales, tile.valueBiases, tile.valueColumnScales,
        ])
    }

    private func byteCount(_ tile: VarianceNormalizedKVTile) -> Int {
        tile.keyWeight.nbytes + tile.keyScales.nbytes + tile.keyBiases.nbytes
            + tile.keyColumnScales.nbytes + tile.valueWeight.nbytes
            + tile.valueScales.nbytes + tile.valueBiases.nbytes
            + tile.valueColumnScales.nbytes
    }

    /// Physical compact-record bytes retained by this cache, excluding the raw tail. Slabs have
    /// no spare capacity, and pending views share an exactly-sized backing batch.
    var compactStorageByteCount: Int {
        tileSlabs.reduce(0) { $0 + byteCount($1.records) }
            + pendingTiles.reduce(0) { $0 + byteCount($1) }
    }

    var attentionPartitionCount: Int {
        tileSlabs.count + pendingTiles.count
    }

    private func stackedRecords(_ tiles: [VarianceNormalizedKVTile]) -> VarianceNormalizedKVTile {
        precondition(!tiles.isEmpty)
        return VarianceNormalizedKVTile(
            keyWeight: stacked(tiles.map(\.keyWeight), axis: 2),
            keyScales: stacked(tiles.map(\.keyScales), axis: 2),
            keyBiases: stacked(tiles.map(\.keyBiases), axis: 2),
            keyColumnScales: stacked(tiles.map(\.keyColumnScales), axis: 2),
            valueWeight: stacked(tiles.map(\.valueWeight), axis: 2),
            valueScales: stacked(tiles.map(\.valueScales), axis: 2),
            valueBiases: stacked(tiles.map(\.valueBiases), axis: 2),
            valueColumnScales: stacked(tiles.map(\.valueColumnScales), axis: 2)
        )
    }

    private var completedTileCount: Int {
        tileSlabs.reduce(pendingTiles.count) { $0 + $1.count }
    }

    private func appendSlab(_ records: VarianceNormalizedKVTile, count: Int) {
        tileSlabs.append(VarianceNormalizedKVSlab(records: records, count: count))

        // Immutable tiered compaction: eight equal adjacent slabs become one larger slab. Each
        // record is copied only at exponentially spaced boundaries (4, 32, 256 tiles...), while
        // attention sees at most seven slabs per level instead of one dispatch per page.
        while tileSlabs.count >= Self.slabFanout {
            let suffix = Array(tileSlabs.suffix(Self.slabFanout))
            guard let first = suffix.first, suffix.allSatisfy({ $0.count == first.count }) else {
                break
            }
            let records = suffix.map(\.records)
            let merged = VarianceNormalizedKVTile(
                keyWeight: contiguous(concatenated(records.map(\.keyWeight), axis: 2)),
                keyScales: contiguous(concatenated(records.map(\.keyScales), axis: 2)),
                keyBiases: contiguous(concatenated(records.map(\.keyBiases), axis: 2)),
                keyColumnScales: contiguous(
                    concatenated(records.map(\.keyColumnScales), axis: 2)),
                valueWeight: contiguous(concatenated(records.map(\.valueWeight), axis: 2)),
                valueScales: contiguous(concatenated(records.map(\.valueScales), axis: 2)),
                valueBiases: contiguous(concatenated(records.map(\.valueBiases), axis: 2)),
                valueColumnScales: contiguous(
                    concatenated(records.map(\.valueColumnScales), axis: 2))
            )
            evaluate(merged)
            tileSlabs.removeLast(Self.slabFanout)
            tileSlabs.append(
                VarianceNormalizedKVSlab(
                    records: merged, count: first.count * Self.slabFanout))
        }
    }

    private func appendCompressedBatch(_ records: VarianceNormalizedKVTile, count: Int) {
        precondition(count > 0 && records.keyWeight.dim(2) == count)
        var index = 0

        while !pendingTiles.isEmpty && index < count {
            pendingTiles.append(tileView(records, index: index))
            index += 1
            if pendingTiles.count == Self.baseSlabTileCount {
                let slabRecords = stackedRecords(pendingTiles)
                evaluate(slabRecords)
                appendSlab(slabRecords, count: Self.baseSlabTileCount)
                pendingTiles.removeAll(keepingCapacity: true)
            }
        }

        while index + Self.baseSlabTileCount <= count {
            let end = index + Self.baseSlabTileCount
            let slabRecords = contiguousRecords(records, range: index ..< end)
            evaluate(slabRecords)
            appendSlab(slabRecords, count: Self.baseSlabTileCount)
            index = end
        }

        if index < count {
            let pendingRecords = contiguousRecords(records, range: index ..< count)
            evaluate(pendingRecords)
            pendingTiles.append(
                contentsOf: (0 ..< count - index).map {
                    tileView(pendingRecords, index: $0)
                })
        }
    }

    private func contiguousRecords(
        _ records: VarianceNormalizedKVTile, range: Range<Int>
    ) -> VarianceNormalizedKVTile {
        VarianceNormalizedKVTile(
            keyWeight: contiguous(records.keyWeight[0..., 0..., range, 0..., 0...]),
            keyScales: contiguous(records.keyScales[0..., 0..., range, 0..., 0...]),
            keyBiases: contiguous(records.keyBiases[0..., 0..., range, 0..., 0...]),
            keyColumnScales: contiguous(
                records.keyColumnScales[0..., 0..., range, 0..., 0...]),
            valueWeight: contiguous(records.valueWeight[0..., 0..., range, 0..., 0...]),
            valueScales: contiguous(records.valueScales[0..., 0..., range, 0..., 0...]),
            valueBiases: contiguous(records.valueBiases[0..., 0..., range, 0..., 0...]),
            valueColumnScales: contiguous(
                records.valueColumnScales[0..., 0..., range, 0..., 0...])
        )
    }

    private func allTiles() -> [VarianceNormalizedKVTile] {
        tileSlabs.flatMap { slab in
            (0 ..< slab.count).map { tileView(slab.records, index: $0) }
        } + pendingTiles
    }

    private func installTiles(_ tiles: [VarianceNormalizedKVTile]) {
        tileSlabs.removeAll(keepingCapacity: true)
        pendingTiles.removeAll(keepingCapacity: true)
        var index = 0
        while index + Self.baseSlabTileCount <= tiles.count {
            let records = stackedRecords(Array(tiles[index ..< index + Self.baseSlabTileCount]))
            evaluate(records)
            appendSlab(records, count: Self.baseSlabTileCount)
            index += Self.baseSlabTileCount
        }
        pendingTiles.append(contentsOf: tiles[index...])
    }

    private func reconstructedRotated(
        _ tile: VarianceNormalizedKVTile
    ) -> (MLXArray, MLXArray) {
        let scaledKeys = dequantized(
            tile.keyWeight,
            scales: tile.keyScales.asType(.float32),
            biases: tile.keyBiases.asType(.float32),
            groupSize: tileSize, bits: keyBits, mode: .affine)
        let rotatedKeys = (scaledKeys * tile.keyColumnScales.asType(.float32))
            .transposed(0, 1, 3, 2)

        let valueHeadDim = tile.valueColumnScales.dim(-1)
        let valueGroupSize = validatedValueGroupSize(valueHeadDim)
        let scaledValues = dequantized(
            tile.valueWeight,
            scales: tile.valueScales.asType(.float32),
            biases: tile.valueBiases.asType(.float32),
            groupSize: valueGroupSize, bits: valueBits, mode: .affine)
        let rotatedValues = scaledValues * tile.valueColumnScales.asType(.float32)

        return (
            rotatedKeys.asType(keyDType ?? .float32),
            rotatedValues.asType(valueDType ?? .float32)
        )
    }

    private func reconstructed(_ tile: VarianceNormalizedKVTile) -> (MLXArray, MLXArray) {
        let (rotatedKeys, rotatedValues) = reconstructedRotated(tile)
        return (inverseRotate(rotatedKeys), inverseRotate(rotatedValues))
    }

    private func reconstructedParts() -> [(keys: MLXArray, values: MLXArray)] {
        var parts: [(keys: MLXArray, values: MLXArray)] = []
        for tile in allTiles() {
            let (keys, values) = reconstructed(tile)
            parts.append((keys, values))
        }

        if let tailKeys, let tailValues {
            parts.append((inverseRotate(tailKeys), inverseRotate(tailValues)))
        }
        return parts
    }

    private func materializedState() -> (MLXArray, MLXArray)? {
        let parts = reconstructedParts()
        let keyParts = parts.map { $0.keys }
        let valueParts = parts.map { $0.values }

        guard !keyParts.isEmpty else { return nil }
        return (concatenated(keyParts, axis: 2), concatenated(valueParts, axis: 2))
    }

    private func validatedValueGroupSize(_ valueHeadDim: Int) -> Int {
        guard let groupSize = varianceNormalizedValueGroupSize(valueHeadDim: valueHeadDim) else {
            fatalError(
                "VarianceNormalizedKVCache requires value head dimension \(valueHeadDim) to be compatible with MLX quantization group sizes 32, 64, or 128."
            )
        }
        return groupSize
    }

    private func quantizedTileScores(
        rotatedQueries: MLXArray,
        tile: VarianceNormalizedKVTile,
        scale: Float
    ) -> MLXArray {
        let (batchSize, queryHeadCount, queryLength, headDim) = (
            rotatedQueries.dim(0), rotatedQueries.dim(1), rotatedQueries.dim(2),
            rotatedQueries.dim(3)
        )
        let kvHeadCount = tile.keyWeight.dim(1)
        let repeats = queryHeadCount / kvHeadCount

        if repeats > 1 {
            let groupedQueries = rotatedQueries.reshaped([
                batchSize, kvHeadCount, repeats, queryLength, headDim,
            ])
            let scores =
                quantizedMM(
                    groupedQueries,
                    expandedDimensions(tile.keyWeight, axis: -3),
                    scales: expandedDimensions(tile.keyScales, axis: -3),
                    biases: expandedDimensions(tile.keyBiases, axis: -3),
                    transpose: false,
                    groupSize: tileSize,
                    bits: keyBits,
                    mode: .affine
                ) * expandedDimensions(tile.keyColumnScales, axis: -3) * scale
            return scores.reshaped(batchSize, queryHeadCount, queryLength, tileSize)
        } else {
            return quantizedMM(
                rotatedQueries,
                tile.keyWeight,
                scales: tile.keyScales,
                biases: tile.keyBiases,
                transpose: false,
                groupSize: tileSize,
                bits: keyBits,
                mode: .affine
            ) * tile.keyColumnScales * scale
        }
    }

    private func quantizedTileValues(
        weights: MLXArray,
        tile: VarianceNormalizedKVTile,
        queryHeadCount: Int
    ) -> MLXArray {
        let (batchSize, _, queryLength, keyLength) = (
            weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(3)
        )
        let kvHeadCount = tile.valueWeight.dim(1)
        let repeats = queryHeadCount / kvHeadCount
        let groupSize = validatedValueGroupSize(tile.valueColumnScales.dim(-1))

        if repeats > 1 {
            let groupedWeights = weights.reshaped([
                batchSize, kvHeadCount, repeats, queryLength, keyLength,
            ])
            let output =
                quantizedMM(
                    groupedWeights,
                    expandedDimensions(tile.valueWeight, axis: -3),
                    scales: expandedDimensions(tile.valueScales, axis: -3),
                    biases: expandedDimensions(tile.valueBiases, axis: -3),
                    transpose: false,
                    groupSize: groupSize,
                    bits: valueBits,
                    mode: .affine
                ) * expandedDimensions(tile.valueColumnScales, axis: -3)
            return output.reshaped(
                batchSize, queryHeadCount, queryLength, tile.valueColumnScales.dim(-1))
        } else {
            return quantizedMM(
                weights,
                tile.valueWeight,
                scales: tile.valueScales,
                biases: tile.valueBiases,
                transpose: false,
                groupSize: groupSize,
                bits: valueBits,
                mode: .affine
            ) * tile.valueColumnScales
        }
    }

    private func stackedTileScores(
        rotatedQueries: MLXArray,
        records: VarianceNormalizedKVTile,
        tileCount: Int,
        scale: Float
    ) -> MLXArray {
        let (batchSize, queryHeadCount, queryLength, headDim) = (
            rotatedQueries.dim(0), rotatedQueries.dim(1), rotatedQueries.dim(2),
            rotatedQueries.dim(3)
        )
        let kvHeadCount = records.keyWeight.dim(1)
        let repeats = queryHeadCount / kvHeadCount
        let keyWeights = records.keyWeight
        let keyScales = records.keyScales
        let keyBiases = records.keyBiases
        let keyColumnScales = records.keyColumnScales

        if repeats > 1 {
            let groupedQueries = rotatedQueries.reshaped([
                batchSize, kvHeadCount, repeats, queryLength, headDim,
            ])
            let scores =
                quantizedMM(
                    expandedDimensions(groupedQueries, axis: 2),
                    expandedDimensions(keyWeights, axis: 3),
                    scales: expandedDimensions(keyScales, axis: 3),
                    biases: expandedDimensions(keyBiases, axis: 3),
                    transpose: false,
                    groupSize: tileSize,
                    bits: keyBits,
                    mode: .affine
                ) * expandedDimensions(keyColumnScales, axis: 3) * scale
            return
                scores
                .transposed(0, 1, 3, 4, 2, 5)
                .reshaped(batchSize, queryHeadCount, queryLength, tileCount * tileSize)
        } else {
            let scores =
                quantizedMM(
                    expandedDimensions(rotatedQueries, axis: 2),
                    keyWeights,
                    scales: keyScales,
                    biases: keyBiases,
                    transpose: false,
                    groupSize: tileSize,
                    bits: keyBits,
                    mode: .affine
                ) * keyColumnScales * scale
            return
                scores
                .transposed(0, 1, 3, 2, 4)
                .reshaped(batchSize, queryHeadCount, queryLength, tileCount * tileSize)
        }
    }

    private func stackedTileValues(
        weights: MLXArray,
        records: VarianceNormalizedKVTile,
        tileCount: Int,
        queryHeadCount: Int
    ) -> MLXArray {
        let (batchSize, _, queryLength, _) = (
            weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(3)
        )
        let kvHeadCount = records.valueWeight.dim(1)
        let repeats = queryHeadCount / kvHeadCount
        let valueHeadDim = records.valueColumnScales.dim(-1)
        let groupSize = validatedValueGroupSize(valueHeadDim)
        let valueWeights = records.valueWeight
        let valueScales = records.valueScales
        let valueBiases = records.valueBiases
        let valueColumnScales = records.valueColumnScales

        if repeats > 1 {
            let groupedWeights =
                weights
                .reshaped([batchSize, kvHeadCount, repeats, queryLength, tileCount, tileSize])
                .transposed(0, 1, 4, 2, 3, 5)
            let output =
                quantizedMM(
                    groupedWeights,
                    expandedDimensions(valueWeights, axis: 3),
                    scales: expandedDimensions(valueScales, axis: 3),
                    biases: expandedDimensions(valueBiases, axis: 3),
                    transpose: false,
                    groupSize: groupSize,
                    bits: valueBits,
                    mode: .affine
                ) * expandedDimensions(valueColumnScales, axis: 3)
            return
                output
                .sum(axis: 2)
                .reshaped(batchSize, queryHeadCount, queryLength, valueHeadDim)
        } else {
            let groupedWeights =
                weights
                .reshaped([batchSize, queryHeadCount, queryLength, tileCount, tileSize])
                .transposed(0, 1, 3, 2, 4)
            let output =
                quantizedMM(
                    groupedWeights,
                    valueWeights,
                    scales: valueScales,
                    biases: valueBiases,
                    transpose: false,
                    groupSize: groupSize,
                    bits: valueBits,
                    mode: .affine
                ) * valueColumnScales
            return output.sum(axis: 2)
        }
    }

    private func quantizedRotatedAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let rotatedQueries = rotate(queries)
        var scoreParts = tileSlabs.map { slab in
            stackedTileScores(
                rotatedQueries: rotatedQueries,
                records: slab.records,
                tileCount: slab.count,
                scale: scale)
        }
        scoreParts.append(
            contentsOf: pendingTiles.map { tile in
                quantizedTileScores(rotatedQueries: rotatedQueries, tile: tile, scale: scale)
            })
        if let tailKeys {
            scoreParts.append(
                attentionScores(queries: rotatedQueries, keys: tailKeys, scale: scale))
        }

        let scores = applyAttentionMask(scores: concatenated(scoreParts, axis: -1), mask: mask)
        // `precise` makes MLX compute the normalization in FP32 for half/bfloat inputs. This is
        // particularly important once the score vector spans many compressed slabs.
        let weights = softmax(scores, axis: -1, precise: true)

        var start = 0
        var rotatedOutputParts: [MLXArray] = []
        for slab in tileSlabs {
            let end = start + slab.count * tileSize
            rotatedOutputParts.append(
                stackedTileValues(
                    weights: weights[.ellipsis, start ..< end],
                    records: slab.records,
                    tileCount: slab.count,
                    queryHeadCount: queries.dim(1)))
            start = end
        }
        for tile in pendingTiles {
            let end = start + tileSize
            let tileWeights = weights[.ellipsis, start ..< end]
            rotatedOutputParts.append(
                quantizedTileValues(
                    weights: tileWeights,
                    tile: tile,
                    queryHeadCount: queries.dim(1)))
            start = end
        }

        if let tailValues {
            let end = start + tailValues.dim(2)
            let tailWeights = weights[.ellipsis, start ..< end]
            rotatedOutputParts.append(
                attentionValues(
                    weights: tailWeights,
                    values: tailValues,
                    queryHeadCount: queries.dim(1)))
        }

        let rotatedOutput = rotatedOutputParts.dropFirst().reduce(rotatedOutputParts[0]) {
            $0 + $1
        }
        return inverseRotate(rotatedOutput).asType(queries.dtype)
    }

    private func absorbTail() {
        guard var keys = tailKeys, var values = tailValues else { return }

        while keys.dim(2) >= tileSize {
            // Thirty-two tiles amortize both prefill and decode dispatch while bounding each
            // FP32 normalization batch to 4,096 tokens at the default tile size.
            let tileCount = min(Self.compressionBatchTileCount, keys.dim(2) / tileSize)
            let length = tileCount * tileSize
            let records = compressedBatch(
                rotatedKeys: keys[.ellipsis, ..<length, 0...],
                rotatedValues: values[.ellipsis, ..<length, 0...])
            appendCompressedBatch(records, count: tileCount)

            if keys.dim(2) == length {
                tailKeys = nil
                tailValues = nil
                return
            }

            keys = keys[.ellipsis, length..., 0...]
            values = values[.ellipsis, length..., 0...]
        }

        tailKeys = keys
        tailValues = values
    }

    private func append(keys: MLXArray, values: MLXArray) {
        precondition(
            isSupportedVarianceNormalizedDType(keys.dtype)
                && isSupportedVarianceNormalizedDType(values.dtype),
            "VarianceNormalizedKVCache supports float16, bfloat16, and float32 K/V tensors")
        if let keyDType {
            precondition(keyDType == keys.dtype, "VarianceNormalizedKVCache key dtype changed")
        } else {
            keyDType = keys.dtype
        }
        if let valueDType {
            precondition(
                valueDType == values.dtype, "VarianceNormalizedKVCache value dtype changed")
        } else {
            valueDType = values.dtype
        }

        let rotatedKeys = rotate(keys)
        let rotatedValues = rotate(values)

        if let currentKeys = tailKeys, let currentValues = tailValues {
            tailKeys = concatenated([currentKeys, rotatedKeys], axis: 2)
            tailValues = concatenated([currentValues, rotatedValues], axis: 2)
        } else {
            tailKeys = rotatedKeys
            tailValues = rotatedValues
        }

        offset += keys.dim(2)
        absorbTail()
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        append(keys: keys, values: values)

        // KVCache requires this materializing fallback for attention implementations that call
        // update directly. The standard attentionWithCacheUpdate path uses updateAndAttend below
        // and never reconstructs completed tiles during decode.
        guard let state = materializedState() else {
            return (keys, values)
        }
        return state
    }

    public func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        append(keys: keys, values: values)
        return quantizedRotatedAttention(queries: queries, scale: scale, mask: mask)
    }

    public override var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []
            for tile in allTiles() {
                arrays.append(contentsOf: [
                    tile.keyWeight, tile.keyScales, tile.keyBiases,
                    tile.keyColumnScales,
                    tile.valueWeight, tile.valueScales, tile.valueBiases,
                    tile.valueColumnScales,
                ])
            }
            if let tailKeys, let tailValues {
                arrays.append(tailKeys)
                arrays.append(tailValues)
            }
            return arrays
        }
        set {
            let tileCount =
                restoredTileCount
                ?? (newValue.count / Self.compactTileStateCount)
            let tailStateCount = (restoredTailLength ?? 0) > 0 ? Self.tailStateCount : 0
            let tileStateCount =
                if newValue.count - tailStateCount == tileCount * Self.legacyTileStateCount {
                    Self.legacyTileStateCount
                } else {
                    Self.compactTileStateCount
                }
            var index = 0
            var restored: [VarianceNormalizedKVTile] = []
            for _ in 0 ..< tileCount {
                guard index + tileStateCount - 1 < newValue.count else {
                    fatalError("VarianceNormalizedKVCache state is missing tile arrays")
                }
                if tileStateCount == Self.legacyTileStateCount {
                    restored.append(
                        VarianceNormalizedKVTile(
                            keyWeight: newValue[index],
                            keyScales: newValue[index + 1] * newValue[index + 4],
                            keyBiases: newValue[index + 2] * newValue[index + 4],
                            keyColumnScales: newValue[index + 3],
                            valueWeight: newValue[index + 5],
                            valueScales: newValue[index + 6] * newValue[index + 9],
                            valueBiases: newValue[index + 7] * newValue[index + 9],
                            valueColumnScales: newValue[index + 8]
                        ))
                } else {
                    restored.append(
                        VarianceNormalizedKVTile(
                            keyWeight: newValue[index],
                            keyScales: newValue[index + 1],
                            keyBiases: newValue[index + 2],
                            keyColumnScales: newValue[index + 3],
                            valueWeight: newValue[index + 4],
                            valueScales: newValue[index + 5],
                            valueBiases: newValue[index + 6],
                            valueColumnScales: newValue[index + 7]
                        ))
                }
                index += tileStateCount
            }
            installTiles(restored)

            if (restoredTailLength ?? 0) > 0 {
                guard index + 1 < newValue.count else {
                    fatalError("VarianceNormalizedKVCache state is missing tail arrays")
                }
                tailKeys = newValue[index]
                tailValues = newValue[index + 1]
                keyDType = tailKeys?.dtype
                valueDType = tailValues?.dtype
            } else {
                tailKeys = nil
                tailValues = nil
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(tileSize),
                String(offset),
                String(keyBits),
                String(valueBits),
                String(sinkhornIterations),
                String(completedTileCount),
                String(tailKeys?.dim(2) ?? 0),
                String(Self.metadataVersion),
                varianceNormalizedDTypeName(keyDType),
                varianceNormalizedDTypeName(valueDType),
            ]
        }
        set {
            guard newValue.count == 7 || newValue.count == 10 else {
                fatalError("VarianceNormalizedKVCache metaState must have 7 or 10 values")
            }
            guard
                let offset = Int(newValue[1]),
                let tileCount = Int(newValue[5]),
                let tailLength = Int(newValue[6])
            else {
                fatalError("Failed to parse VarianceNormalizedKVCache metaState")
            }
            self.offset = offset
            self.restoredTileCount = tileCount
            self.restoredTailLength = tailLength
            if newValue.count == 10 {
                guard Int(newValue[7]) == Self.metadataVersion else {
                    fatalError("Unsupported VarianceNormalizedKVCache metadata version")
                }
                let restoredKeyDType = varianceNormalizedDType(named: newValue[8])
                let restoredValueDType = varianceNormalizedDType(named: newValue[9])
                guard
                    newValue[8] == "none"
                        || restoredKeyDType.map(isSupportedVarianceNormalizedDType) == true,
                    newValue[9] == "none"
                        || restoredValueDType.map(isSupportedVarianceNormalizedDType) == true
                else {
                    fatalError("Invalid VarianceNormalizedKVCache dtype metadata")
                }
                keyDType = restoredKeyDType
                valueDType = restoredValueDType
            } else {
                keyDType = nil
                valueDType = nil
            }
        }
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        guard positions >= 0 else { return false }
        let tailLength = tailKeys?.dim(2) ?? 0
        // A staged round that crosses a tile boundary would quantize some provisional rows.
        // Trimming back into that tile reconstructs them approximately, so it is not the exact
        // rollback required by KVCacheRound. Refuse that round and let generation fall back to
        // the non-speculative path; rounds wholly inside the raw tail remain exactly rewindable.
        return tailLength + positions < tileSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let trimmed = min(offset, n)
        guard trimmed > 0 else { return 0 }

        let remaining = offset - trimmed
        let retainedTileCount = remaining / tileSize
        let retainedTailLength = remaining % tileSize

        if retainedTailLength == 0 {
            tailKeys = nil
            tailValues = nil
        } else if retainedTileCount < completedTileCount {
            let (rotatedKeys, rotatedValues) = reconstructedRotated(allTiles()[retainedTileCount])
            tailKeys = rotatedKeys[.ellipsis, ..<retainedTailLength, 0...]
            tailValues = rotatedValues[.ellipsis, ..<retainedTailLength, 0...]
        } else if let tailKeys, let tailValues {
            self.tailKeys = tailKeys[.ellipsis, ..<retainedTailLength, 0...]
            self.tailValues = tailValues[.ellipsis, ..<retainedTailLength, 0...]
        } else {
            preconditionFailure("VarianceNormalizedKVCache is missing its retained tail")
        }

        if retainedTileCount < completedTileCount {
            installTiles(Array(allTiles().prefix(retainedTileCount)))
        }
        restoredTileCount = nil
        restoredTailLength = nil
        offset = remaining
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = VarianceNormalizedKVCache(
            tileSize: tileSize, keyBits: keyBits, valueBits: valueBits,
            sinkhornIterations: sinkhornIterations)
        new.metaState = metaState
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), tileSize: \(tileSize), keyBits: \(keyBits), valueBits: \(valueBits), tiles: \(completedTileCount), slabs: \(tileSlabs.count), pending: \(pendingTiles.count), tail: \(tailKeys?.shape.description ?? "-")"
    }
}

/// Resolve a legacy `kvScheme` string for variance-normalized compression.
///
/// Supported schemes:
/// - `"varn"` / `"varn4v2"`: 4-bit keys, 2-bit values, 128-token tiles
/// - `"varn4"` / `"varn4v4"`: 4-bit keys and values
/// - `"varn2"` / `"varn2v2"`: 2-bit keys and values
/// - `"varn4v2t32"` / `"varn4v2t64"` / `"varn4v2t128"`: explicit tile size
public func resolveVarianceNormalizedScheme(
    _ scheme: String?
) -> (keyBits: Int, valueBits: Int, tileSize: Int, sinkhornIterations: Int)? {
    guard let scheme else { return nil }

    switch scheme {
    case "varn", "varn4v2":
        return (4, 2, 128, 8)
    case "varn4", "varn4v4":
        return (4, 4, 128, 8)
    case "varn2", "varn2v2":
        return (2, 2, 128, 8)
    case "varn4v2t32":
        return (4, 2, 32, 8)
    case "varn4v2t64":
        return (4, 2, 64, 8)
    case "varn4v2t128":
        return (4, 2, 128, 8)
    case "varn4v4t32":
        return (4, 4, 32, 8)
    default:
        return nil
    }
}
