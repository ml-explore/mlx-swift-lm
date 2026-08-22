import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

// MARK: - Fused decode router top-k

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

/// One-kernel replacement for the decode router tail. Generic MLX router code
/// fully sorts all experts to name K winners, then gathers their score values
/// and may normalize them. For a single decode row, one threadgroup can do the
/// same work without the serial dispatch boundaries.
///
/// `selection` determines the winning experts; `values` supplies the scores
/// returned for those experts. Keeping them separate covers routers such as
/// Qwen 3, which selects on logits but weights experts with probabilities.
/// The kernel is deliberately decode-only at production call sites: its O(E²)
/// rank count loses to the generic block sort for many-row prefill.
private let fusedRouterTopKSource = """
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

private final class FusedRouterTopKKernel: Sendable {
    static let shared = FusedRouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "moe_router_topk",
            inputNames: ["selection", "values"],
            outputNames: ["inds", "scores"],
            source: fusedRouterTopKSource
        )
    }
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
) -> (MLXArray, MLXArray) {
    precondition(selection.shape == values.shape, "router selection/value shapes must match")
    precondition(selection.dtype == values.dtype, "router selection/value dtypes must match")

    let e = selection.dim(-1)
    precondition(e <= maxFusedRouterExperts, "fused router exceeds Metal threadgroup limit")
    precondition(k > 0 && k <= e, "invalid fused router top-k")

    let rows = selection.size / e
    let shape = Array(selection.shape.dropLast()) + [k]
    let out = FusedRouterTopKKernel.shared.kernel(
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

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        let activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
