import Foundation
import MLX
import MLXNN

/// `SwitchGLU` with shared pairwise Givens rotations injected around the
/// expert projections — the PARO MoE expert block (mirrors upstream z-lab
/// `RotateSwitchGLU`, paroquant `modules.py`).
///
/// All experts share a single set of rotation parameters per projection
/// input: `gate_up_rot` rotates the token activations ahead of the expert
/// gather (bitwise-equivalent to rotating the gathered rows, at 1/topK the
/// cost) for `gate_proj`/`up_proj` (which share their input), and `down_rot`
/// rotates the activated hidden state before `down_proj`. The projections themselves
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

    /// `gate_up_rot` runs *before* the gather/sort: the rotation kernel is
    /// row-independent (rows never mix, see the `PairwiseRotation` Metal
    /// source) and `gatherSort` only duplicates rows, so
    /// `rotate(gather(x)) ≡ gather(rotate(x))` bitwise — while rotating `L`
    /// rows instead of `L × topK`. `PairwiseRotation.rotate` is
    /// shape-preserving over any leading shape and passes empty batches
    /// through, so both the sorted (flattened) and unsorted (broadcast)
    /// layouts go through unchanged.
    override func transformInput(_ x: MLXArray) -> MLXArray {
        gateUpRot.rotate(x)
    }

    /// Rotate the activated hidden state ahead of the `down_proj` experts.
    override func transformHidden(_ x: MLXArray) -> MLXArray {
        downRot.rotate(x)
    }
}
