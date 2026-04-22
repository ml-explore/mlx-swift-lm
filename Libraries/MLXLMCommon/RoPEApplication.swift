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

// MARK: - RoPEOffset

/// Positional offset for RoPE. Polymorphic so single-sequence and batched
/// inference share the same call site.
///
/// Snapshot once per forward pass via ``KVCache/ropeOffset`` before any
/// `cache.update` call, then reuse the value for every RoPE application in
/// the layer. Reading `cache.offset` at each RoPE call is unsafe because
/// `cache.update` advances the offset mid-pass.
public enum RoPEOffset {
    /// Single-sequence offset.
    case scalar(Int)
    /// Per-sequence offsets with shape `[B]`.
    case array(MLXArray)
}

extension KVCache {
    /// Snapshot of the current RoPE offset suitable for passing to
    /// ``applyRotaryPosition(_:to:offset:)``.
    ///
    /// Read once at the top of `callAsFunction` and reuse for every RoPE
    /// call in that layer. Do not re-read across a `cache.update` — the
    /// offset advances and a later read will not match an earlier one.
    ///
    /// The batched form defensively copies `batchOffset` (via `+ 0`) so the
    /// returned value is decoupled from any in-place mutation that
    /// `cache.update` may perform on the cache's stored offset array.
    public var ropeOffset: RoPEOffset {
        if let batched = self as? BatchPositionedKVCache {
            return .array(batched.batchOffset + 0)
        } else {
            return .scalar(offset)
        }
    }
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings at an explicit offset.
///
/// ```swift
/// let ropeOffset = cache?.ropeOffset ?? .scalar(0)
/// queries = applyRotaryPosition(rope, to: queries, offset: ropeOffset)
/// keys    = applyRotaryPosition(rope, to: keys,    offset: ropeOffset)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - offset: The positional offset, typically produced by ``KVCache/ropeOffset``.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(
    _ rope: R, to x: MLXArray, offset: RoPEOffset
) -> MLXArray {
    switch offset {
    case .scalar(let i): return rope(x, offset: i)
    case .array(let a): return rope(x, offset: a)
    }
}

/// Apply rotary position embeddings, reading the offset from the cache.
///
/// - Warning: This overload reads `cache.offset` at call time and is unsafe
///   when `cache.update` runs between RoPE calls in the same layer. Snapshot
///   the offset once via ``KVCache/ropeOffset`` and use
///   ``applyRotaryPosition(_:to:offset:)`` instead.
@available(
    *, deprecated,
    message:
        "Snapshot `cache?.ropeOffset ?? .scalar(0)` once before any `cache.update` and use `applyRotaryPosition(_:to:offset:)`."
)
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    applyRotaryPosition(rope, to: x, offset: cache?.ropeOffset ?? .scalar(0))
}
