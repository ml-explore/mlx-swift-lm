import Foundation
import MLX
import MLXNN

/// `SwitchGLU` with shared pairwise Givens rotations injected around the
/// expert projections — the PARO MoE expert block (mirrors upstream z-lab
/// `RotateSwitchGLU`, paroquant `modules.py`).
///
/// All experts share a single set of rotation parameters per projection
/// input: `gate_up_rot` rotates the gathered activations before
/// `gate_proj`/`up_proj` (which share their input), and `down_rot` rotates
/// the activated hidden state before `down_proj`. The projections themselves
/// stay stock `SwitchLinear`/`QuantizedSwitchLinear` — rotating the shared
/// activations *once* per token is what makes PARO MoE cheap: the rotation
/// cost is independent of the number of experts.
///
/// Checkpoint contract: the rotation parameters load under *nested* keys
/// (`switch_mlp.gate_up_rot.theta`, `switch_mlp.down_rot.pairs`, …) via the
/// two `PairwiseRotation` children. Upstream keeps them flat on the GLU
/// (`gate_up_rot_theta`); the nested layout is a deliberate divergence (#208)
/// so the rotation is a reusable Module rather than mixin state —
/// `remapSharedMoERotations` in the loader produces the nested keys.
///
/// Subclasses `SwitchGLU` so it satisfies the `switch_mlp: SwitchGLU`
/// property on MoE blocks (e.g. `Qwen35SparseMoeBlock`) through
/// `update(modules:)`, exactly like `RotateQuantizedLinear: QuantizedLinear`
/// on the dense path.
public class RotateSwitchGLU: SwitchGLU {

    @ModuleInfo(key: "gate_up_rot") var gateUpRot: PairwiseRotation
    @ModuleInfo(key: "down_rot") var downRot: PairwiseRotation

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        groupSize: Int,
        krot: Int,
        bias: Bool = false
    ) {
        self._gateUpRot.wrappedValue = PairwiseRotation(
            dims: inputDims, groupSize: groupSize, krot: krot)
        self._downRot.wrappedValue = PairwiseRotation(
            dims: hiddenDims, groupSize: groupSize, krot: krot)
        super.init(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts, bias: bias)
    }

    /// `SwitchGLU`'s dataflow (including the ≥64-indices gather/sort fast
    /// path) with the two shared rotations applied to the gathered
    /// activations. `PairwiseRotation.rotate` is shape-preserving over any
    /// leading shape and passes empty batches through, so both the sorted
    /// (flattened) and unsorted (broadcast) layouts go through unchanged.
    override public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        x = gateUpRot.rotate(x)

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        var activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        activated = downRot.rotate(activated)

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}
