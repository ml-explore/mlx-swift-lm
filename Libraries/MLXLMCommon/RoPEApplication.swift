// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

extension BatchPositionedKVCache {
    public var ropeOffset: RoPEOffset {
        .batch(batchOffset[.ellipsis])
    }
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines scalar or per-sequence offset), or `nil`
///     for offset 0.
/// - Returns: The input with rotary positional encoding applied.
@available(*, deprecated, message: "use applyRotaryPosition(_:to:offset:) instead")
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    applyRotaryPosition(rope, to: x, offset: cache?.ropeOffset)
}

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
///
/// ```swift
/// let offset = cache?.ropeOffset
/// queries = applyRotaryPosition(rope, to: queries, offset: offset)
/// keys = applyRotaryPosition(rope, to: keys, offset: offset)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - offset: the offset into the rotary positional encoding.  0 if nil.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, offset: RoPEOffset?)
    -> MLXArray
{
    switch offset {
    case nil:
        rope(x, offset: 0)
    case .scalar(let v):
        rope(x, offset: v)
    case .batch(let v):
        rope(x, offset: v)
    }
}
