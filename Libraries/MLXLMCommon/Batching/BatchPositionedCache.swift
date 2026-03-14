// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache Protocol

/// Protocol for batch-aware KV caches that provide per-sequence positional offsets.
///
/// When applying rotary position embeddings (RoPE) in a batched context, each
/// sequence in the batch may be at a different position. This protocol provides
/// the per-sequence offsets as an `MLXArray` so that RoPE can be applied with
/// different offsets per batch element.
///
/// Conforming types expose `batchOffset: MLXArray` of shape `[B]` containing
/// the current position offset for each sequence in the batch.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence position offsets as an MLXArray of shape `[B]`.
    ///
    /// For a batch of sequences that have been prefilled to different lengths,
    /// this array contains the effective position index for each sequence,
    /// accounting for left-padding.
    var batchOffset: MLXArray { get }
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, dispatching to the appropriate offset type
/// based on the cache.
///
/// - For `BatchPositionedKVCache`: uses `ArrayOffsetLayer` with per-sequence
///   `MLXArray` offsets for batched inference.
/// - For single caches (non-batch): uses `OffsetLayer` with scalar `Int` offset.
/// - For `nil` cache: uses `OffsetLayer` with offset `0`.
///
/// This function enables models to use a single call site that transparently
/// supports both single-request and batched inference:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines offset type), or `nil` for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let batchCache = cache as? BatchPositionedKVCache {
        // Batch path: per-sequence MLXArray offsets
        return rope(x, offset: batchCache.batchOffset)
    } else {
        // Single path: scalar Int offset (or 0 for nil cache)
        return rope(x, offset: cache?.offset ?? 0)
    }
}

// MARK: - isBatchCompatible

/// Check whether a list of per-layer caches is compatible with batch KV cache
/// merge/extend operations.
///
/// Returns `false` for:
/// - `CacheList` (composite caches used by hybrid models like Jamba)
/// - `MambaCache` (SSM state-space caches, not key-value based)
/// - `QuantizedKVCache` (stores quantized tuples incompatible with batch merge/extend)
///
/// Returns `true` for:
/// - `KVCacheSimple` (standard transformer KV cache)
/// - `RotatingKVCache` (sliding-window attention cache)
/// - Empty cache arrays
///
/// - Parameter caches: The per-layer cache array to check.
/// - Returns: `true` if all caches support batch operations, `false` otherwise.
public func isBatchCompatible(_ caches: [KVCache]) -> Bool {
    for cache in caches {
        if cache is CacheList || cache is MambaCache || cache is QuantizedKVCache {
            return false
        }
    }
    return true
}
