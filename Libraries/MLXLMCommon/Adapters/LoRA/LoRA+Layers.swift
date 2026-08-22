// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities for `Linear` layers, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear`` / ``QLoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear` via ``LoRALinear/fused()``
/// - implementing the LoRA evaluation
/// - applying optional dropout to the LoRA branch during training
///
/// ``QLoRALinear`` is the equivalent class for `QuantizedLinear`.
///
/// This is not typically used directly -- `LoRATrain.convert(model:layers:)` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``QLoRALinear``
public class LoRALinear: Linear, LoRALayer {

    let scale: Float
    let dropout: Dropout
    public var loraEnabled: Bool = true

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public convenience init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: Linear
    ) {
        self.init(
            inputDimensions, outputDimensions, rank: rank, bias: bias, scale: scale,
            dropout: 0.0, linear: linear)
    }

    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, dropout: Float, linear: Linear
    ) {
        // Scale for low-rank update
        self.scale = scale
        self.dropout = Dropout(p: dropout)

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via `LoRATrain.convert(model:layers:)`.
    ///
    /// ### See Also
    /// - ``QLoRALinear/from(linear:rank:scale:dropout:)``
    public static func from(
        linear: Linear, rank: Int = 8, scale: Float = 20.0, dropout: Float = 0.0
    ) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            return QLoRALinear.from(
                linear: linear, rank: rank, scale: scale, dropout: dropout)
        }
        let (outputDimensions, inputDimensions) = linear.shape
        let result = LoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, dropout: dropout,
            linear: linear)
        result.train(linear.training)
        return result
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// ### See Also
    /// - ``QLoRALinear/fused()``
    public func fused() -> Module {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        if !loraEnabled { return y }
        let z = matmul(matmul(dropout(x), self.loraA), self.loraB)
        // Adapter checkpoints and base weights can use different 16-bit
        // formats (commonly fp16 LoRA over a bf16 model). MLX promotes
        // bf16 + fp16 to fp32, so adding `z` directly silently changes the
        // projection's public dtype and every downstream kernel
        // specialization. Keep the replacement layer's output contract
        // identical to the base layer, matching mlx-lm's LoRA path.
        return y + (scale * z).asType(y.dtype)
    }
}

/// Implementation of LoRA `QuantizedLinear` replacement layer.
///
/// See ``LoRALinear`` (equivalent class for `Linear` layers) for more information.
public class QLoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float
    let dropout: Dropout
    public var loraEnabled: Bool = true

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public convenience init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: QuantizedLinear
    ) {
        self.init(
            inputDimensions, outputDimensions, rank: rank, bias: bias, scale: scale,
            dropout: 0.0, linear: linear)
    }

    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, dropout: Float, linear: QuantizedLinear
    ) {

        // Scale for low-rank update
        self.scale = scale
        self.dropout = Dropout(p: dropout)

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode)

        // start frozen except for the lora keys
        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via `LoRATrain.convert(model:layers:)`.
    ///
    /// ### See Also
    /// - ``LoRALinear/from(linear:rank:scale:dropout:)``
    public static func from(
        linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0, dropout: Float = 0.0
    ) -> LoRALayer {
        let (outputDimensions, inputDimensions) = linear.shape
        let result = QLoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, dropout: dropout,
            linear: linear)
        result.train(linear.training)
        return result
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// ### See Also
    /// - ``LoRALinear/fused()``
    public func fused() -> Module {
        let weight = dequantizedWeight
        let dtype = dequantizedWeight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA),
            bias: bias,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        if !loraEnabled { return y }
        let z = matmul(matmul(dropout(x), self.loraA), self.loraB)
        return y + (scale * z).asType(y.dtype)
    }
}
