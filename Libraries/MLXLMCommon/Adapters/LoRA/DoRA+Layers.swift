//
//  DoRA+Layers.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// Performs the forward pass for a DoRA linear layer.
private func forward(
    x: MLXArray, loraInput: MLXArray, y: MLXArray,
    weight: MLXArray, bias: MLXArray?,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let z = matmul(matmul(loraInput, loraA), loraB)
    var out = y + (scale * z).asType(x.dtype)

    let adapted = weight + matmul(scale * loraB.T, loraA.T)
    let denom = stopGradient(norm(adapted, axis: 1))
    out *= (magnitude / denom).asType(x.dtype)

    return if let bias {
        out + bias
    } else {
        out
    }
}

/// Fuses the base weights with the DoRA parameters.
private func fuse(
    weight: MLXArray,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let loraA = loraA.T.asType(weight.dtype)
    let loraB = (scale * loraB.T).asType(weight.dtype)

    var adapted = weight + matmul(loraB, loraA)
    let denom = norm(adapted, axis: 1)
    adapted *= (magnitude / denom).reshaped([-1, 1])

    return adapted
}

/// Filters out DoRA-specific parameters from a list of module keys.
private func filterFreezeKeys(from module: Module, keys: [String]?) -> [String] {
    return
        (keys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
        .flattened()
        .map { $0.0 })
        .filter { !["lora_a", "lora_b", "m"].contains($0) }
}

/// Implementation of DoRA `Linear` replacement layer.
///
/// This layer implements DoRA (Weight-Decomposed Low-Rank Adaptation) for `Linear` layers.
///
/// ``QDoRALinear`` is the equivalent class for `QuantizedLinear`.
public class DoRALinear: Linear, LoRALayer {

    let scale: Float
    let dropout: Dropout
    public var loraEnabled: Bool = true

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public convenience init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        self.init(linear: linear, rank: rank, scale: scale, dropout: 0.0)
    }

    public init(linear: Linear, rank: Int = 8, scale: Float = 20.0, dropout: Float) {
        precondition(type(of: linear) == Linear.self, "DoRALinear cannot adapt \(type(of: linear))")

        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self.dropout = Dropout(p: dropout)
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.weight, axis: 1)

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a DoRA adapter layer.
    ///
    /// An existing adapter layer is replaced: the new adapter wraps its base layer.
    ///
    /// - Precondition: the base layer is exactly `Linear` or `QuantizedLinear`. Subclasses such
    ///   as `HadamardQuantizedLinear` can carry state the adapter would drop.
    public static func from(
        linear: Linear, rank: Int = 8, scale: Float = 20.0, dropout: Float = 0.0
    ) -> LoRALayer {
        guard let result = adapter(for: linear, rank: rank, scale: scale, dropout: dropout) else {
            preconditionFailure("DoRA cannot adapt \(type(of: linear))")
        }
        return result
    }

    /// A DoRA adapter for `linear` in its training mode, or `nil` if ``LoRABase`` rejects it.
    static func adapter(
        for linear: Linear, rank: Int, scale: Float, dropout: Float
    ) -> LoRALayer? {
        let result: LoRALayer
        switch LoRABase(linear) {
        case .linear(let base):
            result = DoRALinear(linear: base, rank: rank, scale: scale, dropout: dropout)
        case .quantized(let base):
            result = QDoRALinear(linear: base, rank: rank, scale: scale, dropout: dropout)
        case nil:
            return nil
        }
        result.train(linear.training)
        return result
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    public func fused() -> Module {
        Linear(
            weight: fuse(
                weight: weight, loraA: loraA, loraB: loraB, scale: scale, magnitude: magnitude),
            bias: bias
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if !loraEnabled {
            return super.callAsFunction(x)
        }
        let y = matmul(x, weight.T)
        return forward(
            x: x, loraInput: dropout(x), y: y,
            weight: weight, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}

/// Implementation of DoRA `QuantizedLinear` replacement layer.
///
/// See ``DoRALinear`` (equivalent class for `Linear` layers) for more information.
///
/// ### See Also
/// - ``DoRALinear``
public class QDoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float
    let dropout: Dropout
    public var loraEnabled: Bool = true

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public convenience init(
        linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0
    ) {
        self.init(linear: linear, rank: rank, scale: scale, dropout: 0.0)
    }

    public init(
        linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0, dropout: Float
    ) {
        precondition(
            type(of: linear) == QuantizedLinear.self,
            "QDoRALinear cannot adapt \(type(of: linear))")

        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self.dropout = Dropout(p: dropout)
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.dequantizedWeight(), axis: 1)

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits,
            mode: linear.mode, globalScale: linear.globalScale
        )

        freeze()
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    public func fused() -> Module {
        QuantizedLinear(
            weight: fuse(
                weight: dequantizedWeight(), loraA: loraA, loraB: loraB, scale: scale,
                magnitude: magnitude),
            bias: bias, groupSize: groupSize, bits: bits, mode: mode, globalScale: globalScale
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if !loraEnabled {
            return super.callAsFunction(x)
        }
        // `quantizedMM` cannot apply a global scale, so use the dequantized weight.
        let y = matmul(x, dequantizedWeight(dtype: x.dtype).T)
        return forward(
            x: x, loraInput: dropout(x), y: y,
            weight: dequantizedWeight(), bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}
