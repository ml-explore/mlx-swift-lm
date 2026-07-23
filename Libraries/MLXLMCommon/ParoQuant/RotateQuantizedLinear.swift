import Foundation
import MLX
import MLXNN

// The rotation machinery this class dispatches — Metal kernel source, kernel
// cache (`getRotationKernel`), and `packPairs` — lives in PairwiseRotation.swift,
// shared with the standalone `PairwiseRotation` module.

/// Pairwise Givens rotation + quantized matmul.
///
/// Subclasses `QuantizedLinear` so it can replace `Linear` in `@ModuleInfo` slots
/// via `update(modules:)`. Only overrides `callAsFunction` to insert the rotation
/// step before the standard quantized matmul.
///
/// Rotation is applied to activations at runtime via a Metal kernel, preserving
/// the quantization-friendly properties of the original weights.
open class RotateQuantizedLinear: QuantizedLinear {

    // Rotation parameters — discovered by Module reflection for update(parameters:).
    // `channelScales` uses @ParameterInfo so it can keep the snake_case checkpoint
    // key while having a Swift-idiomatic property name.
    let theta: MLXArray
    let pairs: MLXArray
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray

    // Rotation-derived state. Populated once by `prepareDerivedRotationState()`
    // after the checkpoint parameters are loaded (see ParoQuantLoader), and
    // never mutated afterwards. Underscore-prefixed private properties are
    // ignored by Module reflection — see Documentation.docc/porting.md
    // "Computed vs Loaded Parameters" — so they don't participate in weight
    // loading, which keeps the loader's strict `verify: [.allModelKeysSet]`
    // contract intact.
    //
    // Kept out of the forward pass's `eval` graph by materialising them
    // explicitly inside `prepareDerivedRotationState()`.
    private var _cosTheta: MLXArray
    private var _sinTheta: MLXArray
    private var _packedPairs: MLXArray
    private var _scalesFlat: MLXArray

    public init(
        inputDims: Int, outputDims: Int, hasBias: Bool,
        groupSize: Int, bits: Int, krot: Int
    ) {
        self.theta = MLXArray.zeros([krot, inputDims / 2])
        self.pairs = MLXArray.zeros([krot, inputDims], type: Int16.self)
        // Assign through `.wrappedValue` so the `@ParameterInfo(key:)` metadata
        // survives init. Replacing the wrapper with `.init(wrappedValue:)` drops
        // the `key: "channel_scales"` annotation — Module reflection then looks
        // up the parameter by the Swift property name `channelScales`, which
        // doesn't exist in the checkpoint, and `update(parameters:verify:)`
        // fails with `keyNotFound`. Pattern matches `LoRA+Layers.swift`.
        self._channelScales.wrappedValue = MLXArray.ones([1, inputDims])

        // Placeholder values — `prepareDerivedRotationState()` overwrites
        // these with real derived tensors after checkpoint load. Shapes are
        // correct so a forward pass before finalize would be degenerate
        // (identity-ish rotation) rather than crash.
        self._cosTheta = MLXArray.ones([krot, inputDims / 2])
        self._sinTheta = MLXArray.zeros([krot, inputDims / 2])
        self._packedPairs = MLXArray.zeros([krot, inputDims / 2], type: Int32.self)
        self._scalesFlat = MLXArray.ones([inputDims])

        super.init(
            weight: MLXArray.zeros([outputDims, inputDims * bits / 32], type: UInt32.self),
            bias: hasBias ? MLXArray.zeros([outputDims]) : nil,
            scales: MLXArray.zeros([outputDims, inputDims / groupSize]),
            biases: MLXArray.zeros([outputDims, inputDims / groupSize]),
            groupSize: groupSize,
            bits: bits
        )
    }

    /// Compute rotation-derived tensors from the loaded checkpoint parameters.
    ///
    /// Must be called once, after `update(parameters:)` populates
    /// `theta` / `pairs` / `channelScales`, and before any forward pass.
    /// Must not be called concurrently with forward passes — the loader
    /// owns this call, nothing else should.
    ///
    /// Each forward pass previously generated this state lazily on first
    /// call and cached it in a mutable `CachedRotation?` field. That pattern
    /// is unsafe under multi-threaded inference (issue #157 — a shared model
    /// container is driven by multiple tasks simultaneously), so derivation
    /// is now done explicitly at load time.
    ///
    /// The four derived arrays are `eval(...)`ed here because underscore-
    /// prefixed private fields are invisible to Module reflection — the
    /// loader's later `eval(model)` walks `@ParameterInfo` tensors only, so
    /// these would otherwise stay unmaterialised promises until the first
    /// forward pass, and materialisation would then become part of that
    /// pass's graph (exactly the eval-time state we're eliminating).
    public func prepareDerivedRotationState() {
        _cosTheta = MLX.cos(theta)
        _sinTheta = MLX.sin(theta)
        _packedPairs = packPairs(pairs, groupSize: groupSize)
        _scalesFlat = channelScales.reshaped(-1)
        eval(_cosTheta, _sinTheta, _packedPairs, _scalesFlat)
    }

    private func rotate(_ x: MLXArray) -> MLXArray {
        let dim = _scalesFlat.dim(0)
        let numGroups = dim / groupSize
        let krot = theta.dim(0)

        // The kernel assigns 2 of the group's 64 pair slots to each of its
        // 32 lanes; other group sizes would need a different lane mapping.
        precondition(groupSize == 128, "RotateQuantizedLinear: groupSize must be 128, got \(groupSize)")

        let batch = x.dim(0)
        let tile = batch <= 1 ? 1 : 4
        let gridX = ((batch + tile - 1) / tile) * 32
        let params = MLXArray([Int32(batch), Int32(dim), Int32(krot), Int32(groupSize)])

        return getRotationKernel(tile: tile, krot: krot, dtype: x.dtype)(
            [x, _packedPairs, _cosTheta, _sinTheta, _scalesFlat, params],
            grid: (gridX, numGroups, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
    }

    /// Forward pass: applies pairwise Givens rotation then quantized matmul.
    ///
    /// Computes `y = quantizedMM(rotate(x), W)` where `rotate(x)` fuses channel
    /// scaling and Givens rotations in a single Metal kernel. No mutable
    /// state is read or written by this method.
    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let rotated = rotate(x.reshaped(-1, _scalesFlat.dim(0)))

        var y = quantizedMM(
            rotated.reshaped(shape), weight,
            scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits
        )
        if let bias { y = y + bias }
        return y
    }
}
