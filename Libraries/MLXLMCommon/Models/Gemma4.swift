import MLX

package enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            keys.0.dim(-2)
        }
    }
}
