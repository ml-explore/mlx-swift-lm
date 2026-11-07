import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LFM3OmniAudioConfiguration: Decodable {
    let channels: Int
    let hiddenSize: Int
    let melBins: Int
    let window: Int
    let text: LFM2Configuration

    enum CodingKeys: String, CodingKey {
        case channels = "conv_channels"
        case hiddenSize = "hidden_size"
        case melBins = "num_mel_bins"
        case window = "sliding_window"
        case text = "text_config"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decode(Int.self, forKey: .channels)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        melBins = try c.decode(Int.self, forKey: .melBins)
        window = try c.decode(Int.self, forKey: .window)
        text = try c.decode(LFM2Configuration.self, forKey: .text)
        try text.validateModelConfiguration()
        guard channels > 0, melBins > 0, melBins.isMultiple(of: 8), window > 0,
            hiddenSize == text.hiddenSize, channels <= Int.max / melBins,
            text.normEps.isFinite, text.ropeTheta.isFinite,
            text.blockFFNDimMultiplier.isFinite
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid LFM3-Omni audio encoder configuration"))
        }
    }
}

final class LFM3OmniDownsampler: Module {
    @ModuleInfo(key: "conv_in") var input: Conv2d
    @ModuleInfo(key: "conv_mid") var middle: Conv2d
    @ModuleInfo(key: "lin_mid") var middleProjection: Conv2d
    @ModuleInfo(key: "conv_end") var end: Conv2d
    @ModuleInfo(key: "lin_end") var endProjection: Conv2d
    @ModuleInfo(key: "lin_out") var output: Linear

    init(_ config: LFM3OmniAudioConfiguration) {
        let channels = config.channels
        _input.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: channels,
            kernelSize: 3, stride: 2, padding: 1)
        _middle.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3, stride: 2, padding: 1, groups: channels)
        _middleProjection.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1)
        _end.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3, stride: 2, padding: 1, groups: channels)
        _endProjection.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1)
        _output.wrappedValue = Linear(config.melBins / 8 * channels, config.hiddenSize)
    }

    private func mask(_ x: MLXArray, lengths: MLXArray) -> MLXArray {
        let valid = MLXArray(0 ..< x.dim(2))[.newAxis, 0...] .< lengths[0..., .newAxis]
        return MLX.where(valid[0..., .newAxis, 0..., .newAxis], x, MLXArray.zeros(like: x))
    }

    private func activate(_ x: MLXArray) -> MLXArray {
        silu(x.asType(.float32)).asType(x.dtype)
    }

    func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> MLXArray {
        var x = mask(
            features.asType(input.weight.dtype).expandedDimensions(axis: -1), lengths: lengths)
        x = activate(input(x))
        var remaining = floorDivide(lengths + 1, 2)
        x = mask(x, lengths: remaining)
        x = activate(middleProjection(middle(x)))
        remaining = floorDivide(remaining + 1, 2)
        x = mask(x, lengths: remaining)
        x = activate(endProjection(end(x)))
        return output(x.transposed(0, 2, 3, 1).reshaped(x.dim(0), x.dim(2), -1))
    }
}

final class LFM3OmniAudioTrunk: Module {
    let layers: [LFM2DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    let window: Int

    init(_ config: LFM3OmniAudioConfiguration) {
        layers = (0 ..< config.text.hiddenLayers).map {
            LFM2DecoderLayer(config.text, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.text.normEps)
        window = config.window
    }

    func newCache() throws -> [KVCache] {
        try layers.map { layer in
            if layer.isAttentionLayer {
                return try makeHybridAttentionKVCache(
                    parameters: nil, slidingWindow: window, usesSlidingWindow: true)
            }
            return MambaCache()
        }
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let firstAttention = layers.firstIndex { $0.isAttentionLayer }
        let attentionCache = firstAttention.flatMap { cache?[$0] }
        let mask = createAttentionMask(h: embeddings, cache: attentionCache, windowSize: window)
        var hidden = embeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: mask, ssmMask: nil, cache: cache?[index])
        }
        return norm(hidden)
    }
}

final class LFM3OmniAudioEncoder: Module {
    let downsample: LFM3OmniDownsampler
    let lfm: LFM3OmniAudioTrunk
    let config: LFM3OmniAudioConfiguration

    init(_ config: LFM3OmniAudioConfiguration) {
        self.config = config
        downsample = LFM3OmniDownsampler(config)
        lfm = LFM3OmniAudioTrunk(config)
    }

    func newStreamingState() throws -> LFM3OmniAudioStreamingState {
        try LFM3OmniAudioStreamingState(owner: ObjectIdentifier(self), cache: lfm.newCache())
    }

    func stream(
        _ features: MLXArray, state: LFM3OmniAudioStreamingState,
        finalize: Bool = false
    ) throws -> MLXArray {
        guard features.ndim == 3, features.dim(0) > 0, features.dim(1) == config.melBins,
            features.dtype.isFloatingPoint, all(isFinite(features)).item(Bool.self),
            state.owner == ObjectIdentifier(self), !state.finished,
            state.batchSize == nil || state.batchSize == features.dim(0)
        else { throw VLMError.processing("Invalid LFM3-Omni encoder stream or feature chunk") }
        let combined = state.features.map { concatenated([$0, features], axis: -1) } ?? features
        let total = state.frameCount + features.dim(2)
        let tokenCount = (total + (finalize ? 7 : 0)) / 8
        let count = tokenCount - state.tokenCount
        let output: MLXArray
        if count > 0 {
            let start = state.tokenCount - state.offset / 8
            let lengths = MLXArray(Array(repeating: combined.dim(2), count: features.dim(0)))
            let embeddings = downsample(combined, lengths: lengths)[
                0..., start ..< start + count, 0...]
            output = lfm(embeddings, cache: state.cache)
            eval(output, state.cache)
        } else {
            output = MLXArray.zeros(
                [features.dim(0), 0, config.hiddenSize], dtype: downsample.input.weight.dtype)
        }
        let offset = max(0, (tokenCount - 1) * 8)
        state.features =
            finalize ? nil : contiguous(combined[0..., 0..., (offset - state.offset)...])
        state.offset = offset
        state.frameCount = total
        state.tokenCount = tokenCount
        state.batchSize = features.dim(0)
        state.finished = finalize
        return output
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let shapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        return weights.reduce(into: [:]) { result, entry in
            let (key, value) = entry
            if value.shape == shapes[key] {
                result[key] = value
            } else if key.hasPrefix("downsample."), key.hasSuffix(".weight"), value.ndim == 4 {
                result[key] = value.transposed(0, 2, 3, 1)
            } else if key == "downsample.lin_out.weight", value.ndim == 3 {
                result[key] = value.squeezed(axis: -1)
            } else if key.hasSuffix(".conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                result[key] = value.transposed(0, 2, 1)
            } else {
                result[key] = value
            }
        }
    }

    func callAsFunction(_ features: MLXArray, lengths: MLXArray) throws -> MLXArray {
        guard features.ndim == 3, features.dim(0) > 0, features.dim(1) == config.melBins,
            features.dim(2) > 0, lengths.shape == [features.dim(0)],
            features.dtype.isFloatingPoint, lengths.dtype == .int32 || lengths.dtype == .int64,
            all(isFinite(features)).item(Bool.self)
        else { throw VLMError.processing("Invalid LFM3-Omni audio features or lengths") }
        let counts = lengths.asArray(Int.self)
        guard counts.allSatisfy({ $0 > 0 && $0 <= features.dim(2) }) else {
            throw VLMError.processing("LFM3-Omni audio lengths exceed the feature sequence")
        }
        let hidden = lfm(downsample(features, lengths: lengths))
        return concatenated(
            counts.enumerated().map { index, count in
                hidden[index, ..<((count + 7) / 8), 0...]
            })
    }
}

final class LFM3OmniAudioProjector: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    init(inputSize: Int, outputSize: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: inputSize, eps: 1e-5)
        _first.wrappedValue = Linear(inputSize, outputSize)
        _second.wrappedValue = Linear(outputSize, outputSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = first(norm(x))
        return second(gelu(projected.asType(.float32)).asType(projected.dtype))
    }
}

final class LFM3OmniAudioStreamingState {
    fileprivate let owner: ObjectIdentifier
    fileprivate let cache: [KVCache]
    fileprivate var features: MLXArray?
    fileprivate var offset = 0
    fileprivate var frameCount = 0
    fileprivate var tokenCount = 0
    fileprivate var batchSize: Int?
    fileprivate var finished = false

    fileprivate init(owner: ObjectIdentifier, cache: [KVCache]) {
        self.owner = owner
        self.cache = cache
    }
}
