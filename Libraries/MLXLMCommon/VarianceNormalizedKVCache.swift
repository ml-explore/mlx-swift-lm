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

private let varianceNormalizationEpsilon: Float = 1e-6
private let varianceNormalizationMinimumScale: Float = 1e-4
private let varianceNormalizationMaximumScale: Float = 1e4

private func isPowerOfTwo(_ value: Int) -> Bool {
    value > 0 && (value & (value - 1)) == 0
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
        && varianceNormalizedValueGroupSize(valueHeadDim: valueHeadDim) != nil
}

extension KVCacheSimple {
    /// Convert to a variance-normalized tile cache.
    public func toVarianceNormalized(
        tileSize: Int = 128,
        keyBits: Int = 4,
        valueBits: Int = 2,
        sinkhornIterations: Int = 4
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
/// use it without changes. When called through ``attentionWithCacheUpdate``, completed tiles stay
/// quantized on the hot path: attention scores and value products are computed with `quantizedMM`
/// in the rotated domain, and only the final output is inverse-rotated. The tile normalization uses
/// clipped log-domain dual-axis scale balancing with best-imbalance tracking, matching the central
/// KVarN VarN algorithm more closely while remaining an MLX Swift implementation. The RTN affine
/// scale/bias are fused into the matching variance-normalization axis so each completed tile stores
/// compact K/V records: packed weights, fused affine scale, fused affine bias, and the remaining
/// variance scale.
public class VarianceNormalizedKVCache: BaseKVCache, KVCacheAttentionProtocol,
    CustomDebugStringConvertible
{
    private static let compactTileStateCount = 8
    private static let legacyTileStateCount = 10

    private var tiles: [VarianceNormalizedKVTile] = []
    private var tailKeys: MLXArray?
    private var tailValues: MLXArray?
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
        sinkhornIterations: Int = 4
    ) {
        precondition(
            [32, 64, 128].contains(tileSize),
            "VarianceNormalizedKVCache tileSize must be one of MLX's supported quantization group sizes: 32, 64, or 128"
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
        guard isPowerOfTwo(dimensions) else { return x }

        let originalShape = x.shape
        var rotated = x.asType(.float32)
        var stride = 1

        while stride < dimensions {
            let prefixShape = Array(rotated.shape.dropLast())
            let paired = rotated.reshaped(prefixShape + [dimensions / (2 * stride), 2, stride])
            let halves = split(paired, parts: 2, axis: -2)
            let left = halves[0].squeezed(axis: -2)
            let right = halves[1].squeezed(axis: -2)
            rotated = concatenated([left + right, left - right], axis: -1).reshaped(originalShape)
            stride *= 2
        }

        return (rotated / sqrt(Float(dimensions))).asType(x.dtype)
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
            maximum(std(x, axis: axis, keepDims: true), MLXArray(varianceNormalizationEpsilon)),
            min: MLXArray(varianceNormalizationMinimumScale),
            max: MLXArray(varianceNormalizationMaximumScale))
    }

    private func varianceImbalance(_ x: MLXArray) -> MLXArray {
        let columnVariance = maximum(
            pow(std(x, axis: -2, keepDims: true), 2),
            MLXArray(varianceNormalizationEpsilon))
        let rowVariance = maximum(
            pow(std(x, axis: -1, keepDims: true), 2),
            MLXArray(varianceNormalizationEpsilon))
        let columnImbalance = abs(log(columnVariance)).max(axis: -1, keepDims: true)
        let rowImbalance = abs(log(rowVariance)).max(axis: -2, keepDims: true)
        return maximum(columnImbalance, rowImbalance)
    }

    private func varianceNormalize(_ tile: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let original = tile.asType(.float32)
        var logColumnScales = MLXArray.zeros(std(original, axis: -2, keepDims: true).shape)
        var logRowScales = MLXArray.zeros(std(original, axis: -1, keepDims: true).shape)
        var columnScales = exp(logColumnScales)
        var rowScales = exp(logRowScales)
        var balancedTile = original
        var bestColumnScales = columnScales
        var bestRowScales = rowScales
        var bestImbalance = varianceImbalance(balancedTile)

        for _ in 0 ..< sinkhornIterations {
            logColumnScales = logColumnScales + log(clippedStd(balancedTile, axis: -2))
            columnScales = exp(logColumnScales)
            balancedTile = balanced(
                original: original, columnScales: columnScales, rowScales: rowScales)

            logRowScales = logRowScales + log(clippedStd(balancedTile, axis: -1))
            rowScales = exp(logRowScales)
            balancedTile = balanced(
                original: original, columnScales: columnScales, rowScales: rowScales)

            let imbalance = varianceImbalance(balancedTile)
            let useCandidate = imbalance .< bestImbalance
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

    private func compress(rotatedKeys: MLXArray, rotatedValues: MLXArray) {
        let keyTile = rotatedKeys.transposed(0, 1, 3, 2)
        let (balancedKeys, keyColumnScales, keyRowScales) = varianceNormalize(keyTile)
        let (keyWeight, keyScales, keyBiases) = quantizeBalanced(
            balancedKeys, groupSize: tileSize, bits: keyBits)
        let fusedKeyScales = keyScales * keyRowScales
        let fusedKeyBiases = keyBiases * keyRowScales

        let valueGroupSize = validatedValueGroupSize(rotatedValues.dim(3))
        let (balancedValues, valueColumnScales, valueRowScales) = varianceNormalize(rotatedValues)
        let (valueWeight, valueScales, valueBiases) = quantizeBalanced(
            balancedValues, groupSize: valueGroupSize, bits: valueBits)
        let fusedValueScales = valueScales * valueRowScales
        let fusedValueBiases = valueBiases * valueRowScales

        tiles.append(
            VarianceNormalizedKVTile(
                keyWeight: keyWeight,
                keyScales: fusedKeyScales,
                keyBiases: fusedKeyBiases,
                keyColumnScales: keyColumnScales,
                valueWeight: valueWeight,
                valueScales: fusedValueScales,
                valueBiases: fusedValueBiases,
                valueColumnScales: valueColumnScales
            ))
    }

    private func reconstructed(_ tile: VarianceNormalizedKVTile) -> (MLXArray, MLXArray) {
        let scaledKeys = dequantized(
            tile.keyWeight, scales: tile.keyScales, biases: tile.keyBiases,
            groupSize: tileSize, bits: keyBits, mode: .affine)
        let rotatedKeys = (scaledKeys * tile.keyColumnScales)
            .transposed(0, 1, 3, 2)
        let keys = inverseRotate(rotatedKeys)

        let valueHeadDim = tile.valueColumnScales.dim(-1)
        let valueGroupSize = validatedValueGroupSize(valueHeadDim)
        let scaledValues = dequantized(
            tile.valueWeight, scales: tile.valueScales, biases: tile.valueBiases,
            groupSize: valueGroupSize, bits: valueBits, mode: .affine)
        let rotatedValues = scaledValues * tile.valueColumnScales
        let values = inverseRotate(rotatedValues)

        return (keys.asType(.float16), values.asType(.float16))
    }

    private func reconstructedParts() -> [(keys: MLXArray, values: MLXArray)] {
        var parts: [(keys: MLXArray, values: MLXArray)] = []
        for tile in tiles {
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

    private func quantizedRotatedAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let rotatedQueries = rotate(queries)
        var scoreParts = tiles.map { tile in
            quantizedTileScores(rotatedQueries: rotatedQueries, tile: tile, scale: scale)
        }
        if let tailKeys {
            scoreParts.append(
                attentionScores(queries: rotatedQueries, keys: tailKeys, scale: scale))
        }

        let scores = applyAttentionMask(scores: concatenated(scoreParts, axis: -1), mask: mask)
        let weights = softmax(scores, axis: -1)

        var start = 0
        var rotatedOutputParts: [MLXArray] = []
        for tile in tiles {
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
            compress(
                rotatedKeys: keys[.ellipsis, ..<tileSize, 0...],
                rotatedValues: values[.ellipsis, ..<tileSize, 0...])

            if keys.dim(2) == tileSize {
                tailKeys = nil
                tailValues = nil
                return
            }

            keys = keys[.ellipsis, tileSize..., 0...]
            values = values[.ellipsis, tileSize..., 0...]
        }

        tailKeys = keys
        tailValues = values
    }

    private func append(keys: MLXArray, values: MLXArray) {
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
            for tile in tiles {
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
            let tailStateCount = (restoredTailLength ?? 0) > 0 ? 2 : 0
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
            tiles = restored

            if (restoredTailLength ?? 0) > 0 {
                guard index + 1 < newValue.count else {
                    fatalError("VarianceNormalizedKVCache state is missing tail arrays")
                }
                tailKeys = newValue[index]
                tailValues = newValue[index + 1]
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
                String(tiles.count),
                String(tailKeys?.dim(2) ?? 0),
            ]
        }
        set {
            guard newValue.count == 7 else {
                fatalError("VarianceNormalizedKVCache metaState must have exactly 7 values")
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
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let trimmed = min(offset, n)
        guard trimmed > 0, let (keys, values) = materializedState() else {
            offset -= trimmed
            return trimmed
        }

        let remaining = max(0, offset - trimmed)
        tiles.removeAll()
        tailKeys = nil
        tailValues = nil
        offset = 0
        if remaining > 0 {
            _ = update(
                keys: keys[.ellipsis, ..<remaining, 0...],
                values: values[.ellipsis, ..<remaining, 0...])
        }
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
        "\(String(describing: Self.self)) offset: \(offset), tileSize: \(tileSize), keyBits: \(keyBits), valueBits: \(valueBits), tiles: \(tiles.count), tail: \(tailKeys?.shape.description ?? "-")"
    }
}
