import Foundation
import MLX
import MLXNN
#if canImport(MLXFast)
import MLXFast
#else
// Fallback for MLXFast when it's not available
public enum MLXFast {
    public enum ScaledDotProductAttentionMaskMode {
        case none
        case causal
        case array(MLXArray)
        @available(*, deprecated, message: "Use .array instead")
        case arrays([MLXArray])
    }

    // Fallback RoPE implementation using MLXNN.RoPE
    public static func RoPE(
        _ x: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: Int,
        freqs: MLXArray? = nil
    ) -> MLXArray {
        let rope = MLXNN.RoPE(
            dimensions: dimensions,
            traditional: traditional,
            base: base ?? 10000.0,
            scale: scale
        )
        return rope(x, offset: offset)
    }

    // Fallback RoPE with MLXArray offset
    public static func RoPE(
        _ x: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray? = nil
    ) -> MLXArray {
        let rope = MLXNN.RoPE(
            dimensions: dimensions,
            traditional: traditional,
            base: base ?? 10000.0,
            scale: scale
        )
        return rope(x, offset: offset)
    }

    // Fallback rmsNorm implementation
    public static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        // RMS norm: weight * x * rsqrt(mean(x^2) + eps)
        let meanSquare = mean(x * x, axis: -1, keepDims: true)
        return weight * x * rsqrt(meanSquare + eps)
    }

    // Fallback layerNorm implementation
    public static func layerNorm(_ x: MLXArray, weight: MLXArray?, bias: MLXArray?, eps: Float) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        var normalized = (x - mean) * rsqrt(variance + eps)
        if let weight {
            normalized = normalized * weight
        }
        if let bias {
            normalized = normalized + bias
        }
        return normalized
    }

    // Fallback scaledDotProductAttention implementation
    public static func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        // Handle GQA (Grouped Query Attention) where nHeads > nKVHeads
        let nHeads = queries.dim(1)
        let nKVHeads = keys.dim(1)

        var expandedKeys = keys
        var expandedValues = values

        if nHeads != nKVHeads {
            // Repeat KV heads to match query heads
            // e.g., if nHeads=32, nKVHeads=8, each KV head is repeated 4 times
            let repeats = nHeads / nKVHeads
            let B = keys.dim(0)
            let L = keys.dim(2)
            let D = keys.dim(3)

            // Expand and repeat: [B, nKVHeads, L, D] -> [B, nHeads, L, D]
            // Use repeated() free function which is the public API for tiling along an axis
            expandedKeys = repeated(
                keys.reshaped(B, nKVHeads, 1, L, D),
                count: repeats,
                axis: 2
            ).reshaped(B, nHeads, L, D)
            expandedValues = repeated(
                values.reshaped(B, nKVHeads, 1, L, D),
                count: repeats,
                axis: 2
            ).reshaped(B, nHeads, L, D)
        }

        var scores = (queries * scale).matmul(expandedKeys.transposed(0, 1, 3, 2))

        switch mask {
        case .none:
            break
        case .causal:
            let L = queries.dim(2)
            let S = keys.dim(2)
            let indices_q = MLXArray(0..<L)
            let indices_k = MLXArray(0..<S)
            let causalMask = indices_q.expandedDimensions(axis: 1) .>= (indices_k - MLXArray(S - L))
            let maskValues = MLXArray(Float(-1e9))
            scores = MLX.where(causalMask, scores, maskValues)
        case .array(let maskArray):
            if maskArray.dtype == .bool {
                let maskValues = MLXArray(Float(-1e9))
                scores = MLX.where(maskArray, scores, maskValues)
            } else {
                scores = scores + maskArray
            }
        case .arrays(let maskArrays):
            if let maskArray = maskArrays.first {
                if maskArray.dtype == .bool {
                    let maskValues = MLXArray(Float(-1e9))
                    scores = MLX.where(maskArray, scores, maskValues)
                } else {
                    scores = scores + maskArray
                }
            }
        }

        scores = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return matmul(scores, expandedValues)
    }
}
#endif

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
#if canImport(MLXFast)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
#else
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
#endif
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
#if canImport(MLXFast)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
#else
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
#endif
    }
}
