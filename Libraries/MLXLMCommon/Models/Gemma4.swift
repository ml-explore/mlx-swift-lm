import MLX

/// Per-layer key/value state threaded between Gemma 4 decoder layers when KV sharing
/// is enabled.
///
/// `public` rather than `package` because `Gemma4DecoderLayer.callAsFunction` is exposed
/// at `@_spi(GemmaEncoder)` scope for encoder-style client taps, and it appears in that
/// signature — an SPI method cannot be called from outside the package if a type in its
/// signature is package-scoped. Clients that do not use KV sharing pass `nil` and ignore
/// the returned value.
public enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    public var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            keys.0.dim(-2)
        }
    }
}
