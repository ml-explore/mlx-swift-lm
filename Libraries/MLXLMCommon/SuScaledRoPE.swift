import Foundation
import MLX
import MLXNN

/// Su Scaled Rotary Position Embedding.
/// Switches between short and long factors based on sequence length.
public class SuScaledRoPE: Module, OffsetLayer {
    let dimensions: Int
    let originalMaxPositionEmbeddings: Int
    let _shortFreqs: MLXArray
    let _longFreqs: MLXArray
    let _shortScale: Float
    let _longScale: Float

    public init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131072,
        originalMaxPositionEmbeddings: Int = 4096,
        shortFactor: [Float] = [1.0],
        longFactor: [Float] = [1.0],
        shortMScale: Float? = nil,
        longMScale: Float? = nil
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let exponent =
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        let freqs = MLX.pow(MLXArray(base), exponent)
        self._shortFreqs = MLXArray(shortFactor).asType(.float32) * freqs
        self._longFreqs = MLXArray(longFactor).asType(.float32) * freqs

        func defaultScale(_ factor: Float) -> Float {
            sqrt(1 + log(factor) / log(Float(originalMaxPositionEmbeddings)))
        }

        let factor = Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        self._shortScale = shortMScale ?? (factor <= 1.0 ? 1.0 : defaultScale(factor))
        self._longScale = longMScale ?? (factor <= 1.0 ? 1.0 : defaultScale(factor))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = offset + x.dim(-2)
        let freqs: MLXArray
        let scale: Float
        if seqLen > originalMaxPositionEmbeddings {
            freqs = _longFreqs
            scale = _longScale
        } else {
            freqs = _shortFreqs
            scale = _shortScale
        }

        // Apply scaling only to the dimensions that will be rotated
        let scaledX = x
        scaledX[.ellipsis, 0 ..< dimensions] = scale * scaledX[.ellipsis, 0 ..< dimensions]

        return MLXFast.RoPE(
            scaledX,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}

/// Backward compatibility alias.
@available(*, deprecated, renamed: "SuScaledRoPE")
public typealias SuScaledRotaryEmbedding = SuScaledRoPE
