import MLX

// MARK: - Fused router top-k

/// One-kernel replacement for the decode router tail. Generic MLX router code
/// fully sorts all experts to name K winners, then gathers their score values
/// and may normalize them. For a single decode row, one threadgroup can do the
/// same work without the serial dispatch boundaries.
///
/// Bit-identical to the chain by construction: the sort is stable
/// (`sort.h`'s `LessThan` compares values only, ties keep input order), so
/// counting the elements ranked strictly above `i` — with the index packed
/// into the low bits of a monotone bit key as the tie-break — reproduces
/// each winner's slot. `±0.0` normalises to one bit pattern (they compare
/// equal but differ bitwise), NaN maps above `+inf` (all NaNs tie), and the
/// sum accumulates sequentially in the output dtype from zero, in slot
/// order, matching `reduce.metal`'s `thread_reduce`.
///
/// `selection` determines the winning experts; `values` supplies the scores
/// returned for those experts. Keeping them separate covers routers such as
/// Qwen 3, which selects on logits but weights experts with probabilities.
private let routerTopKSource = """
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;

    threadgroup ulong sk[E_];
    threadgroup float top_v[K_];

    float v = static_cast<float>(selection[row * E_ + t]);
    uint b = (v == 0.0f) ? 0u : as_type<uint>(v);
    uint mono = isnan(v) ? 0xFFFFFFFFu : (b ^ ((uint)(((int)b) >> 31) | 0x80000000u));
    uint tie = DESCENDING_ ? (0xFFFFFFFFu - t) : t;
    ulong key = (((ulong)mono) << 32) | (ulong)tie;
    sk[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int above = 0;
    for (uint j = 0; j < E_; ++j) {
        above += (sk[j] > key) ? 1 : 0;
    }
    if (above < K_) {
        uint slot = DESCENDING_ ? (uint)above : (uint)(K_ - 1 - above);
        top_v[slot] = static_cast<float>(values[row * E_ + t]);
        inds[row * K_ + slot] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        T acc = static_cast<T>(0);
        for (int q = 0; q < K_; ++q) {
            acc = static_cast<T>(top_v[q]) + acc;
        }
        for (int q = 0; q < K_; ++q) {
            T s = static_cast<T>(top_v[q]);
            scores[row * K_ + q] = NORMALIZE_ ? (s / acc) : s;
        }
    }
    """

private final class RouterTopKKernel: Sendable {
    static let shared = RouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "router_topk_norm",
            inputNames: ["selection", "values"],
            outputNames: ["inds", "scores"],
            source: routerTopKSource
        )
    }
}

/// Metal's threads-per-threadgroup ceiling. The fused router uses one thread
/// per expert, so larger routers must retain the generic MLX path.
package let maxFusedRouterExperts = 1024

/// Winner order produced by the generic router expression being replaced.
///
/// `argPartition(values, kth: E-K)[(E-K)...]` yields the selected values in
/// ascending order, while `argPartition(-values, kth: K-1)[..<K]` yields them
/// in descending order. Expert output reduction is order-sensitive, so the
/// fused path must preserve that distinction exactly.
package enum FusedRouterTopKOrder {
    case ascending
    case descending
}

/// Whether a router tensor can use the fused single-row Metal path.
package func supportsFusedRouterTopK(_ selection: MLXArray, k: Int) -> Bool {
    let e = selection.dim(-1)
    return selection.size == e && e <= maxFusedRouterExperts && k > 0 && k <= e
        && selection.dtype.isFloatingPoint
}

/// Fused top-k selection, selected-value gather, and optional normalization.
/// Callers are responsible for restricting production use to the single-row
/// decode shape with ``supportsFusedRouterTopK(_:k:)``.
package func fusedRouterTopK(
    selection: MLXArray,
    values: MLXArray,
    k: Int,
    normalize: Bool,
    order: FusedRouterTopKOrder
) -> (indices: MLXArray, scores: MLXArray) {
    precondition(selection.shape == values.shape, "router selection/value shapes must match")
    precondition(selection.dtype == values.dtype, "router selection/value dtypes must match")

    let e = selection.dim(-1)
    precondition(e <= maxFusedRouterExperts, "fused router exceeds Metal threadgroup limit")
    precondition(k > 0 && k <= e, "invalid fused router top-k")

    let rows = selection.size / e
    let shape = Array(selection.shape.dropLast()) + [k]
    let out = RouterTopKKernel.shared.kernel(
        [selection, values],
        template: [
            ("T", selection.dtype),
            ("E_", e),
            ("K_", k),
            ("NORMALIZE_", normalize ? 1 : 0),
            ("DESCENDING_", order == .descending ? 1 : 0),
        ],
        grid: (e, rows, 1),
        threadGroup: (e, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, values.dtype]
    )
    return (out[0], out[1])
}

/// Top-`k` + optional normalisation over the last axis in one dispatch:
/// `(indices, scores)` shaped `[..., k]`, bit-identical to
/// `chainRouterTopK`, `uint32` indices included. One threadgroup per row
/// with an `O(E²)` rank count. Internal so the bitwise test can reach it.
func fusedRouterTopK(
    _ gates: MLXArray, k: Int, normalize: Bool
) -> (indices: MLXArray, scores: MLXArray) {
    fusedRouterTopK(
        selection: gates, values: gates, k: k, normalize: normalize, order: .ascending
    )
}

/// The three-dispatch router tail the fused kernel replaces — the prefill
/// path, larger expert sets, and the bitwise test's reference.
func chainRouterTopK(
    _ gates: MLXArray, k: Int, normalize: Bool
) -> (indices: MLXArray, scores: MLXArray) {
    let kth = gates.dim(-1) - k
    let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
    var scores = MLX.takeAlong(gates, inds, axis: -1)
    if normalize {
        scores = scores / scores.sum(axis: -1, keepDims: true)
    }
    return (inds, scores)
}

/// Selects experts using the fused kernel for a single decode row and the
/// reference chain for prefill, batched decode, or unsupported expert counts.
package func moeRouterTopK(
    _ gates: MLXArray, k: Int, normalize: Bool
) -> (indices: MLXArray, scores: MLXArray) {
    if supportsFusedRouterTopK(gates, k: k) {
        return fusedRouterTopK(gates, k: k, normalize: normalize)
    }
    return chainRouterTopK(gates, k: k, normalize: normalize)
}
