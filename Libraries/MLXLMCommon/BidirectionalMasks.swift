// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Bidirectional attention mask: every query position attends to every kv
/// position equally (no causal restriction).
///
/// Used by the MTP drafter, whose queries sit at a single constant position
/// outside the target's KV cache and need full visibility into the target's
/// shared K/V pool. Returned as an additive mask (`0` for attend, `-inf` for
/// mask) compatible with `MLXFast.ScaledDotProductAttentionMaskMode.array`.
///
/// - Parameters:
///   - queryLen: number of query tokens (typically `1` for MTP drafting)
///   - kvLen: total kv positions in the shared pool
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` array of zeros.
public func createBidirectionalMask(
    queryLen: Int,
    kvLen: Int,
    dtype: DType
) -> MLXArray {
    MLXArray.zeros([queryLen, kvLen], dtype: dtype)
}

/// Bidirectional sliding-window attention mask: when `kvLen` exceeds
/// `windowSize`, each query attends to a contiguous window of the first
/// `windowSize` kv positions; the remaining `kvLen - windowSize` positions
/// are masked with `-inf`. When `windowSize >= kvLen` (the degenerate case
/// — and the production MTP path, where the sliding-attention KV cache is
/// capped at `windowSize` by `RotatingKVCache`), the helper early-exits to
/// an all-zeros mask (every query attends to every kv position).
///
/// Matches the fixture convention in `tools/fixtures/masks/` and HF
/// Transformers' `create_bidirectional_sliding_window_mask` (after the
/// kv-axis flip).
///
/// Contract for the non-degenerate path (`kvLen > windowSize`): the mask
/// is built from an absolute-position predicate (`kIdx < windowSize`), NOT
/// a distance-from-`queryOffset` predicate. Callers that need
/// distance-based windowing around a non-zero query offset (e.g. an mlx-vlm-
/// style `|q_idx - k_idx| < windowSize` mask) must build the mask inline
/// rather than calling this helper. MTP production callers don't hit this
/// path because the target's sliding-attention cache cap keeps `kvLen` at
/// or under `windowSize`; `Gemma4AssistantDraftModel.forwardHidden` enforces
/// the invariant with a precondition before calling here.
///
/// - Parameters:
///   - queryLen: number of query tokens
///   - kvLen: total kv positions
///   - windowSize: sliding window size
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` additive mask.
public func createBidirectionalSlidingWindowMask(
    queryLen: Int,
    kvLen: Int,
    windowSize: Int,
    dtype: DType
) -> MLXArray {
    if windowSize >= kvLen {
        return MLXArray.zeros([queryLen, kvLen], dtype: dtype)
    }
    let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
    let attend = kIdx .< Int32(windowSize)
    let row = MLX.where(
        attend,
        MLXArray(0, dtype: dtype),
        MLXArray(-Float.infinity, dtype: dtype)
    )
    // Broadcast the row across queryLen rows.
    return broadcast(row[.newAxis, 0...], to: [queryLen, kvLen])
}
