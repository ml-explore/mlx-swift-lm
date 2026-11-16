import Foundation
import MLX

/// Cache-native attention that can reuse an owner's KV without appending it again.
package protocol SharedAttentionKVCache: KVCache {
    func updateForAttention(keys: MLXArray, values: MLXArray)
    func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// An owner's attention presentation, valid until its next cache update.
/// Keep this within one forward pass; serialization state is not an attention view.
package enum AttentionKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int, bits: Int, mode: QuantizationMode
    )
    case native(cache: any SharedAttentionKVCache, sequenceLength: Int)

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _): keys.dim(2)
        case .quantized(let keys, _, _, _, _): keys.0.dim(2)
        case .native(_, let sequenceLength): sequenceLength
        }
    }

    package static func update(keys: MLXArray, values: MLXArray, cache: KVCache?) -> Self {
        if let cache = cache as? SharedAttentionKVCache {
            cache.updateForAttention(keys: keys, values: values)
            return .native(cache: cache, sequenceLength: cache.offset)
        } else if let cache = cache as? QuantizedKVCacheProtocol {
            let (keys, values) = cache.updateQuantized(keys: keys, values: values)
            return .quantized(
                keys: keys, values: values,
                groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)
        } else if let cache {
            let (keys, values) = cache.update(keys: keys, values: values)
            return .regular(keys: keys, values: values)
        } else {
            return .regular(keys: keys, values: values)
        }
    }

    package func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch self {
        case .regular(let keys, let values):
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            return quantizedCacheAttention(
                queries: queries, quantizedKeys: keys, quantizedValues: values,
                scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            // Native storage must still describe the owner's presentation.
            precondition(cache.offset == sequenceLength)
            return cache.attend(queries: queries, scale: scale, mask: mask)
        }
    }
}

/// Whether attention can be split around a plain `KVCache.update` call.
///
/// Quantized and TurboQuant caches own their complete attention operation, so
/// compiled model segments must leave those cache routes on the general path.
package func usesPlainAttentionCacheRoute(_ cache: KVCache) -> Bool {
    !(cache is QuantizedKVCacheProtocol) && !(cache is TurboQuantKVCache)
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Detects cache-native attention implementations
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
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 && !turboCache.isCompressed {
            // Prefill (L>1) on a raw cache: plain update + standard SDPA, // zero overhead; compression is deferred to the first decode step.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask
            )
        }
        // Decode (L=1) or any call once the cache is compressed (speculative
        // verify chunks, multi-turn re-prefill): the compressed path. The raw
        // update() path is invalid after compression, its raw buffers are
        // gone. First decode call triggers compressRawCache().
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    } else if let attentionCache = cache as? KVCacheAttentionProtocol {
        return attentionCache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedCacheAttention(
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
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// Attention over a quantized cache's keys and values. Every quantized-cache route
/// goes through here so the fused-versus-quantized decision lives in one place.
package func quantizedCacheAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray {
    // Scales carry one entry per group, so they recover the value head dim for any mode.
    let valueHeadDim = quantizedValues.1.dim(-1) * groupSize
    if usesDequantizedPrefillAttention(queries: queries, valueHeadDim: valueHeadDim) {
        // Prefill over a quantized cache: dequantize once and run the fused kernel
        // instead of materializing [B, H, L, N] scores through quantized matmuls.
        let keys = dequantized(
            quantizedKeys.0, scales: quantizedKeys.1, biases: quantizedKeys.2,
            groupSize: groupSize, bits: bits, mode: mode, dtype: queries.dtype)
        let values = dequantized(
            quantizedValues.0, scales: quantizedValues.1, biases: quantizedValues.2,
            groupSize: groupSize, bits: bits, mode: mode, dtype: queries.dtype)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    return quantizedScaledDotProductAttention(
        queries: queries, quantizedKeys: quantizedKeys, quantizedValues: quantizedValues,
        scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
}

/// Query length from which a quantized cache attends through the fused SDPA kernel over
/// dequantized keys and values rather than through quantized matmuls.
///
/// This is a measured policy, not a kernel boundary. On M4 Pro (Qwen3-14B shapes, 4-bit
/// KV, 8k context) the quantized path is 1.45-1.7x slower from 64 queries up and
/// allocates [B, H, L, N] scores; below 32 the two are even, and at one query the
/// quantized path reads a quarter of the bytes. The crossover shifts with the GQA ratio,
/// head dim, cache length, bits, and GPU generation; retune it against new measurements.
package let dequantizedPrefillAttentionMinimumLength = 64

/// Head dims whose fused SDPA kernel MLX compiles for every GPU generation
/// (`mlx/backend/metal/scaled_dot_product_attention.cpp`). MLX routes 192 and 256 to
/// its unfused path, where the quantized matmuls are competitive, so they stay out.
package let dequantizedPrefillAttentionHeadDims: Set<Int> = [64, 72, 80, 96, 128]

/// Whether prefill attention over a quantized cache should dequantize and use the fused
/// kernel. MLA-style caches with narrower values than queries stay on the quantized route.
package func usesDequantizedPrefillAttention(queries: MLXArray, valueHeadDim: Int) -> Bool {
    let headDim = queries.dim(-1)
    return queries.dim(2) >= dequantizedPrefillAttentionMinimumLength
        && headDim == valueHeadDim
        && dequantizedPrefillAttentionHeadDims.contains(headDim)
}
