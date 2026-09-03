import Foundation
import MLX
import MLXNN

// The rotation machinery this class dispatches — Metal kernel source and
// cache, `dispatchPairwiseRotation`, and `RotationDerivedState` — lives in
// PairwiseRotation.swift, shared with the standalone `PairwiseRotation`
// module.

/// Pairwise Givens rotation + quantized matmul.
///
/// Subclasses `QuantizedLinear` so it can replace `Linear` in `@ModuleInfo` slots
/// via `update(modules:)`. Only overrides `callAsFunction` to insert the rotation
/// step before the standard quantized matmul.
///
/// Rotation is applied to activations at runtime via a Metal kernel, preserving
/// the quantization-friendly properties of the original weights.
open class RotateQuantizedLinear: QuantizedLinear, RotationStatePreparing {

    // Rotation parameters — discovered by Module reflection for update(parameters:).
    // `channelScales` uses @ParameterInfo so it can keep the snake_case checkpoint
    // key while having a Swift-idiomatic property name.
    let theta: MLXArray
    let pairs: MLXArray
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray

    // Populated once by `prepareDerivedRotationState()` after the checkpoint
    // parameters are loaded (see ParoQuantLoader), and never mutated
    // afterwards. See `RotationDerivedState` for why the underscore prefix
    // keeps it out of weight loading.
    private var _rotation: RotationDerivedState

    /// - Precondition: `inputDims` is a positive multiple of an even
    ///   `groupSize`, and `krot >= 1` — see `rotationGeometryProblem`.
    public init(
        inputDims: Int, outputDims: Int, hasBias: Bool,
        groupSize: Int, bits: Int, krot: Int
    ) {
        assertRotationGeometry(dims: inputDims, groupSize: groupSize, krot: krot)
        self.theta = MLXArray.zeros([krot, inputDims / 2])
        self.pairs = MLXArray.zeros([krot, inputDims], type: Int16.self)
        // Assign through `.wrappedValue` so the `@ParameterInfo(key:)` metadata
        // survives init. Replacing the wrapper with `.init(wrappedValue:)` drops
        // the `key: "channel_scales"` annotation — Module reflection then looks
        // up the parameter by the Swift property name `channelScales`, which
        // doesn't exist in the checkpoint, and `update(parameters:verify:)`
        // fails with `keyNotFound`. Pattern matches `LoRA+Layers.swift`.
        self._channelScales.wrappedValue = MLXArray.ones([1, inputDims])
        self._rotation = RotationDerivedState(dims: inputDims, krot: krot)

        super.init(
            weight: MLXArray.zeros([outputDims, inputDims * bits / 32], type: UInt32.self),
            bias: hasBias ? MLXArray.zeros([outputDims]) : nil,
            scales: MLXArray.zeros([outputDims, inputDims / groupSize]),
            biases: MLXArray.zeros([outputDims, inputDims / groupSize]),
            groupSize: groupSize,
            bits: bits
        )
    }

    /// See `RotationStatePreparing` — loader-owned, results batched into a
    /// single load-time `eval` with every other rotation module's.
    @discardableResult
    public func prepareDerivedRotationState() -> [MLXArray] {
        _rotation.prepare(
            theta: theta, pairs: pairs, channelScales: channelScales, groupSize: groupSize)
        return _rotation.all
    }

    /// Forward pass: applies pairwise Givens rotation then quantized matmul.
    ///
    /// Computes `y = quantizedMM(rotate(x), W)` where the rotation fuses
    /// channel scaling and Givens rotations in a single Metal kernel. No
    /// mutable state is read or written by this method.
    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let rotated = dispatchPairwiseRotation(
            x.reshaped(-1, _rotation.scalesFlat.dim(0)),
            state: _rotation, groupSize: groupSize, krot: theta.dim(0)
        )

        var y = quantizedMM(
            rotated.reshaped(shape), weight,
            scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits
        )
        if let bias { y = y + bias }
        return y
    }
}
