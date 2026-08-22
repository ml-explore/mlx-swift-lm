import MLX

// MARK: - Fused router top-k

/// One-kernel replacement for the decode router tail: `chainRouterTopK`
/// fully sorts all `E` experts (`ArgPartition::eval_gpu` delegates to
/// `gpu_merge_sort`) just to name `K` — three serial dispatches, three
/// encoder-wide barriers, where barrier-bound decode needs one.
///
/// Bit-identical to the chain by construction: the sort is stable
/// (`sort.h`'s `LessThan` compares values only, ties keep input order), so
/// counting the elements ranked strictly above `i` — with the index packed
/// into the low bits of a monotone bit key as the tie-break — reproduces
/// each winner's slot. `±0.0` normalises to one bit pattern (they compare
/// equal but differ bitwise), NaN maps above `+inf` (all NaNs tie), and the
/// sum accumulates sequentially in the output dtype from zero, in slot
/// order, matching `reduce.metal`'s `thread_reduce`.
private let routerTopKSource = """
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;

    threadgroup ulong sk[E_];
    threadgroup float top_v[K_];

    float v = static_cast<float>(gates[row * E_ + t]);
    uint b = (v == 0.0f) ? 0u : as_type<uint>(v);
    uint mono = isnan(v) ? 0xFFFFFFFFu : (b ^ ((uint)(((int)b) >> 31) | 0x80000000u));
    ulong key = (((ulong)mono) << 32) | (ulong)t;
    sk[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int above = 0;
    for (uint j = 0; j < E_; ++j) {
        above += (sk[j] > key) ? 1 : 0;
    }
    if (above < K_) {
        top_v[K_ - 1 - above] = v;
        inds[row * K_ + (K_ - 1 - above)] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        T acc = static_cast<T>(0);
        for (int q = 0; q < K_; ++q) {
            acc = static_cast<T>(top_v[q]) + acc;
        }
        for (int q = 0; q < K_; ++q) {
            T s = static_cast<T>(top_v[q]);
            scores[row * K_ + q] = NORM_ ? (s / acc) : s;
        }
    }
    """

private final class RouterTopKKernel: Sendable {
    static let shared = RouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "router_topk_norm",
            inputNames: ["gates"],
            outputNames: ["inds", "scores"],
            source: routerTopKSource
        )
    }
}

/// Metal's threads-per-threadgroup ceiling; one thread per expert, so past
/// this the dispatch is invalid, not just slow.
private let maxFusedRouterExperts = 1024

/// Top-`k` + optional normalisation over the last axis in one dispatch:
/// `(indices, scores)` shaped `[..., k]`, bit-identical to
/// `chainRouterTopK`, `uint32` indices included. One threadgroup per row
/// with an `O(E²)` rank count. Internal so the bitwise test can reach it.
func fusedRouterTopK(
    _ gates: MLXArray, k: Int, normalize: Bool
) -> (indices: MLXArray, scores: MLXArray) {
    let e = gates.dim(-1)
    let rows = gates.size / e
    let shape = Array(gates.shape.dropLast()) + [k]
    let out = RouterTopKKernel.shared.kernel(
        [gates],
        template: [
            ("T", gates.dtype), ("E_", e), ("K_", k), ("NORM_", normalize ? 1 : 0),
        ],
        grid: (e, rows, 1),
        threadGroup: (e, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, gates.dtype]
    )
    return (out[0], out[1])
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
    let e = gates.dim(-1)
    if gates.size == e, e <= maxFusedRouterExperts {
        return fusedRouterTopK(gates, k: k, normalize: normalize)
    }
    return chainRouterTopK(gates, k: k, normalize: normalize)
}
