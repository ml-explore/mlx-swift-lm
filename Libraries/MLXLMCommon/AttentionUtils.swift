import Foundation
import MLX

// MARK: - TurboKV Telemetry

enum TurboKVTelemetry {
    nonisolated(unsafe) private static var totalTokens: UInt64 = 0
    nonisolated(unsafe) private static var totalOrigBytes: UInt64 = 0
    nonisolated(unsafe) private static var totalPackedBytes: UInt64 = 0
    nonisolated(unsafe) private static var hasLogged = false

    static func logOnce(compressedOffset: Int, keys: MLXArray, values: MLXArray, headDim: Int) {
        let B = keys.dim(0)
        let nKVH = keys.dim(1)
        let newTokens = keys.dim(2)
        let packedBytes = UInt64(keys.nbytes + values.nbytes)
        let origBytes = UInt64(B * nKVH * newTokens * headDim * 2 * 2)
        record(tokens: newTokens, origBytes: Int(origBytes), packedBytes: Int(packedBytes))
    }

    static func record(tokens: Int, origBytes: Int, packedBytes: Int) {
        totalTokens += UInt64(tokens)
        totalOrigBytes += UInt64(origBytes)
        totalPackedBytes += UInt64(packedBytes)
        if !hasLogged && totalTokens > 0 {
            hasLogged = true
            let ratio = totalPackedBytes > 0 ? Double(totalOrigBytes) / Double(totalPackedBytes) : 0
            print("[TurboKV] \(totalTokens)t compressed, \(String(format: "%.1f", ratio))x ratio")
        }
    }
}

typealias TurboKVCacheTelemetry = TurboKVTelemetry

// MARK: - Attention Utilities

/// Automatic attention with cache update
///
/// Routes to quantized, TurboQuant, or standard attention based on cache type.
/// Handles cache updating, TurboQuant decode, and mask slicing transparently.
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)

        var fullKeys = cachedKeys
        var fullValues = cachedValues
        if let kvCache = cache as? KVCacheSimple,
           let pk = kvCache.polarKeys,
           let pv = kvCache.polarValues,
           kvCache.compressedOffset > 0 {
            let historyK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
            let historyV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
            var mergedK = historyK
            var mergedV = historyV
            if kvCache.turboSplitHeads {
                let B = historyK.dim(0), H2 = historyK.dim(1), T = historyK.dim(2)
                mergedK = historyK.reshaped(B, H2 / 2, T, 512)
                mergedV = historyV.reshaped(B, H2 / 2, T, 512)
            }
            fullKeys = concatenated([mergedK, cachedKeys], axis: 2)
            fullValues = concatenated([mergedV, cachedValues], axis: 2)
        }

        let targetS = fullKeys.dim(2)
        var safeMask = mask
        if case .array(let customMask) = mask {
            if customMask.dim(-1) != targetS && customMask.dim(-1) > targetS {
                let sliced: MLXArray
                if customMask.ndim == 2 {
                    sliced = customMask[0..., ..<targetS]
                } else if customMask.ndim == 4 {
                    sliced = customMask[0..., 0..., 0..., ..<targetS]
                } else {
                    fatalError("Unsupported mask dimensionality: \(customMask.ndim)")
                }
                safeMask = .array(sliced)
            }
        }

        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: fullKeys,
            values: fullValues,
            scale: scale,
            mask: safeMask
        )
    }
}
