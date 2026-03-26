import Foundation
import MLX

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
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        return scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
    if let turboQuantCache = cache as? TurboQuantKVCache {
        precondition(sinks == nil, "TurboQuant attention does not support attention sinks.")
        return turboQuantCache.attention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized attention does not support attention sinks.")
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
        return scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}

public func updateCacheAndReturnArrays(
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?
) -> (MLXArray, MLXArray) {
    guard let cache else {
        return (keys, values)
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys,
            values: values
        )
        return (
            dequantized(
                quantizedKeys.0,
                scales: quantizedKeys.1,
                biases: quantizedKeys.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            ),
            dequantized(
                quantizedValues.0,
                scales: quantizedValues.1,
                biases: quantizedValues.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
        )
    }

    return cache.update(keys: keys, values: values)
}

public func dequantizedKVState(cache: KVCache) -> (MLXArray, MLXArray)? {
    if let turboQuantCache = cache as? TurboQuantKVCache {
        return turboQuantCache.dequantizedState()
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol,
        let (quantizedKeys, quantizedValues) = quantizedKVCache.getQuantizedState()
    {
        return (
            dequantized(
                quantizedKeys.0,
                scales: quantizedKeys.1,
                biases: quantizedKeys.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            ),
            dequantized(
                quantizedValues.0,
                scales: quantizedValues.1,
                biases: quantizedValues.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
        )
    }

    let state = cache.state
    guard state.count >= 2 else { return nil }
    return (state[0], state[1])
}

private func scaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    if let sinks {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }

    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask
    )
}
