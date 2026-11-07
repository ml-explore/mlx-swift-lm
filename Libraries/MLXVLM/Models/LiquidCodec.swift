import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LiquidCodecConfiguration: Decodable {
    let hiddenSize: Int
    let channels: Int
    let fftSize: Int
    let hopSize: Int
    let windowSize: Int
    let frameStack: Int
    let quantizerSize: Int
    let levels: [Int]
    let codebooks: Int
    let decoderWindow: Int
    let sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "encoder_dim"
        case channels = "encoder_conv_dim"
        case fftSize = "stft_n_fft"
        case hopSize = "stft_hop_length"
        case windowSize = "stft_win_length"
        case frameStack = "stft_frame_stack"
        case quantizerSize = "quantizer_dim"
        case levels = "fsq_levels"
        case codebooks = "n_q"
        case decoderWindow = "decoder_window"
        case sampleRate = "sampling_rate"
    }

    func validate() throws {
        guard hiddenSize > 0, channels > 0, fftSize > 0, fftSize.isMultiple(of: 2),
            fftSize / 2 + 1 <= Int.max / 3,
            fftSize == windowSize, hopSize > 0, hopSize < windowSize,
            (windowSize - hopSize).isMultiple(of: 2), frameStack > 0,
            frameStack <= Int.max / hopSize, frameStack * hopSize > windowSize / 2,
            quantizerSize > 0, codebooks > 0, !levels.isEmpty,
            codebooks <= Int.max / levels.count,
            levels.allSatisfy({ $0 >= 2 }), decoderWindow > 0, sampleRate == 24_000
        else { throw VLMError.processing("Invalid LiquidCodec configuration") }
        var product = 1
        for level in levels {
            guard product <= Int(Int32.max) / level else {
                throw VLMError.processing("LiquidCodec codebook size overflows")
            }
            product *= level
        }
    }

    func backbone(window: Int? = nil) throws -> LFM3Configuration {
        let kind = window == nil ? "full_attention" : "sliding_attention"
        let config: [String: Any] = [
            "model_type": "lfm3", "vocab_size": 1, "hidden_size": hiddenSize,
            "intermediate_size": 2304, "num_hidden_layers": 8, "num_attention_heads": 8,
            "num_key_value_heads": 2, "head_dim": 64, "norm_eps": 1e-5,
            "conv_L_cache": 7, "conv_bias": false, "use_sandwich_norm": true,
            "layer_types": (0 ..< 8).map { [2, 4, 6].contains($0) ? kind : "conv" },
            "sliding_window": window ?? 256,
            "rope_parameters": [
                kind: [
                    "rope_type": "default", "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 1.0,
                ]
            ],
        ]
        return try JSONDecoder().decode(
            LFM3Configuration.self,
            from: JSONSerialization.data(withJSONObject: config))
    }
}

private final class LiquidCodecFrontEnd: Module {
    @ModuleInfo(key: "in_proj") var input: Conv1d
    let conv: Conv1d
    let out: Conv1d

    init(_ c: LiquidCodecConfiguration) {
        _input.wrappedValue = Conv1d(
            inputChannels: 3 * (c.fftSize / 2 + 1),
            outputChannels: c.channels, kernelSize: 1)
        conv = Conv1d(
            inputChannels: c.channels, outputChannels: c.channels,
            kernelSize: 3, padding: 2)
        out = Conv1d(
            inputChannels: c.channels, outputChannels: c.hiddenSize,
            kernelSize: c.frameStack, stride: c.frameStack)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let projected = input(features.asType(input.weight.dtype))
        let x = silu(projected.asType(.float32)).asType(projected.dtype)
        let convolved = conv(x)[0..., ..<x.dim(1), 0...]
        return out(silu(convolved.asType(.float32)).asType(convolved.dtype))
    }
}

private final class LiquidCodecEncoder: Module {
    let lfm: LFM3ModelInner
    init(_ config: LiquidCodecConfiguration) throws {
        lfm = LFM3ModelInner(try config.backbone(), includeTokenEmbedding: false)
    }
    func callAsFunction(_ embeddings: MLXArray) -> MLXArray { lfm(embeddings: embeddings) }
}

private final class LiquidCodecQuantizer: Module {
    @ModuleInfo(key: "project_in") var input: Linear?
    @ModuleInfo(key: "project_out") var output: Linear?
    let levelValues: [Int]
    let basisValues: [Int]
    var levels: MLXArray { MLXArray(levelValues) }
    var basis: MLXArray { MLXArray(basisValues) }
    let codebooks: Int
    let dimensions: Int

    init(_ c: LiquidCodecConfiguration) {
        codebooks = c.codebooks
        dimensions = c.levels.count
        levelValues = c.levels
        var product = 1
        basisValues = c.levels.map { level in
            defer { product *= level }
            return product
        }
        if c.quantizerSize != codebooks * dimensions {
            _input.wrappedValue = Linear(c.quantizerSize, codebooks * dimensions)
            _output.wrappedValue = Linear(codebooks * dimensions, c.quantizerSize)
        }
    }

    func encode(_ features: MLXArray) throws -> MLXArray {
        let x = (input?(features) ?? features).asType(.float32)
            .reshaped(features.dim(0), features.dim(1), codebooks, dimensions)
        guard all(isFinite(x)).item(Bool.self) else {
            throw VLMError.processing("LiquidCodec audio produces non-finite latents")
        }
        let half = (levels - 1).asType(.float32) * 1.001 / 2
        let offset = MLX.where((levels % 2) .== 0, MLXArray(Float(0.5)), MLXArray(Float(0)))
        let shift = atanh(offset / half)
        let bounded = tanh(x + shift) * half - offset
        let quantized = round(bounded) + floorDivide(levels, 2)
        return (quantized * basis).sum(axis: -1).asType(.int32).transposed(0, 2, 1)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let indices = codes.transposed(0, 2, 1).expandedDimensions(axis: -1)
        let half = floorDivide(levels, 2)
        let quantized =
            ((floorDivide(indices, basis) % levels) - half).asType(.float32)
            / half.asType(.float32)
        let x = quantized.reshaped(codes.dim(0), codes.dim(2), -1)
        return output?(x.asType(output!.weight.dtype)) ?? x
    }
}

private final class LiquidCodecSynthesis: Module {
    @ParameterInfo var window: MLXArray
    let config: LiquidCodecConfiguration

    init(_ c: LiquidCodecConfiguration) {
        config = c
        window = MLXArray(
            (0 ..< c.windowSize).map {
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(c.windowSize)))
            })
    }

    func callAsFunction(_ spectrum: MLXArray) -> (MLXArray, MLXArray) {
        let frames = irfft(spectrum, n: config.fftSize) * window.asType(.float32)
        let count = spectrum.dim(1)
        let size = (count - 1) * config.hopSize + config.windowSize
        let audio = MLXArray.zeros([spectrum.dim(0), size])
        let envelope = MLXArray.zeros([size])
        for index in 0 ..< count {
            let range = index * config.hopSize ..< index * config.hopSize + config.windowSize
            audio[0..., range] = audio[0..., range] + frames[0..., index, 0...]
            envelope[range] = envelope[range] + window.asType(.float32) * window.asType(.float32)
        }
        return (audio, envelope)
    }
}

private final class LiquidCodecDecoder: Module {
    let lfm: LFM3ModelInner
    let lin: Linear
    let istft: LiquidCodecSynthesis
    let config: LiquidCodecConfiguration

    init(_ c: LiquidCodecConfiguration) throws {
        config = c
        lfm = LFM3ModelInner(try c.backbone(window: c.decoderWindow), includeTokenEmbedding: false)
        lin = Linear(c.hiddenSize, 2 * (c.fftSize / 2 + 1))
        istft = LiquidCodecSynthesis(c)
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> (MLXArray, MLXArray) {
        let expanded = repeated(embeddings, count: config.frameStack, axis: 1)
        let hidden = lfm(embeddings: expanded, cache: cache)
        let parts = lin(hidden).split(parts: 2, axis: -1)
        let magnitude = exp(parts[0].asType(.float32))
        let angle = parts[1].asType(.float32)
        let spectrum =
            magnitude
            * (cos(angle).asType(.complex64)
                + sin(angle).asType(.complex64) * MLXArray(real: Float(0), imaginary: Float(1)))
        return istft(spectrum)
    }
}

final class LiquidCodec: Module {
    private let frontend: LiquidCodecFrontEnd
    private let encoder: LiquidCodecEncoder
    @ModuleInfo(key: "to_quantizer") var toQuantizer: Linear
    private let quantizer: LiquidCodecQuantizer
    private let decoder: LiquidCodecDecoder
    @ModuleInfo(key: "from_quantizer") var fromQuantizer: Linear
    let config: LiquidCodecConfiguration

    init(_ config: LiquidCodecConfiguration) throws {
        try config.validate()
        self.config = config
        frontend = LiquidCodecFrontEnd(config)
        encoder = try LiquidCodecEncoder(config)
        _toQuantizer.wrappedValue = Linear(config.hiddenSize, config.quantizerSize)
        quantizer = LiquidCodecQuantizer(config)
        decoder = try LiquidCodecDecoder(config)
        _fromQuantizer.wrappedValue = Linear(config.quantizerSize, config.hiddenSize)
    }

    func encode(_ samples: MLXArray) throws -> MLXArray {
        guard samples.ndim == 2, samples.dim(0) > 0, samples.dim(1) > 0,
            samples.dtype.isFloatingPoint,
            all(isFinite(samples)).item(Bool.self)
        else { throw VLMError.processing("LiquidCodec expects finite nonempty 24 kHz audio") }
        let frameSize = config.hopSize * config.frameStack
        let count = (samples.dim(1) + frameSize - 1) / frameSize * frameSize
        let padded = concatenated(
            [
                samples.asType(.float32),
                MLXArray.zeros([samples.dim(0), count - samples.dim(1)]),
            ], axis: 1)
        let left = (config.windowSize - config.hopSize) / 2
        let right = config.windowSize - config.hopSize - left
        let reflected = concatenated(
            [
                padded[0..., MLXArray(Array((1 ... left).reversed()))], padded,
                padded[0..., MLXArray((1 ... right).map { count - 1 - $0 })],
            ], axis: 1)
        let frames = asStrided(
            contiguous(reflected),
            [samples.dim(0), count / config.hopSize, config.fftSize],
            strides: [reflected.dim(1), config.hopSize, 1])
        let window = MLXArray(
            (0 ..< config.windowSize).map {
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(config.windowSize)))
            })
        let spectrum = rfft(frames * window)
        let magnitude = abs(spectrum)
        let denominator = maximum(magnitude, 1e-5)
        let features = concatenated(
            [
                log1p(magnitude), spectrum.realPart() / denominator,
                spectrum.imaginaryPart() / denominator,
            ], axis: -1)
        return try quantizer.encode(toQuantizer(encoder(frontend(features))))
    }

    func newStreamingState() throws -> LiquidCodecStreamingState {
        try LiquidCodecStreamingState(
            owner: ObjectIdentifier(self), cache: decoder.lfm.newCache(parameters: nil))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, entry in
            let (key, value) = entry
            if key.hasPrefix("frontend."), key.hasSuffix(".weight"), value.ndim == 3 {
                result[key] = value.transposed(0, 2, 1)
            } else if key.hasSuffix(".conv.weight"), value.ndim == 3 {
                result[key] = value.transposed(0, 2, 1)
            } else {
                result[key] = value
            }
        }
    }

    func decode(
        _ codes: MLXArray, state: LiquidCodecStreamingState? = nil,
        finalize: Bool = true
    ) throws -> MLXArray {
        let codebookSize = config.levels.reduce(1, *)
        guard codes.ndim == 3, codes.dim(0) > 0, codes.dim(1) == config.codebooks,
            codes.dtype == .int32 || codes.dtype == .int64,
            all((codes .>= 0) .&& (codes .< codebookSize)).item(Bool.self)
        else { throw VLMError.processing("Invalid LiquidCodec codebook indices") }
        if let state {
            guard state.owner == ObjectIdentifier(self), !state.finished,
                state.batchSize == nil || state.batchSize == codes.dim(0)
            else { throw VLMError.processing("Invalid or finalized LiquidCodec streaming state") }
        }
        let trim = (config.windowSize - config.hopSize) / 2
        let tail = 2 * trim
        guard codes.dim(2) > 0 || (finalize && state?.started == true) else {
            if finalize { state?.finished = true }
            return MLXArray.zeros([codes.dim(0), 0])
        }
        var audio: MLXArray
        var envelope: MLXArray
        if codes.dim(2) > 0 {
            let latents = quantizer.decode(codes).asType(fromQuantizer.weight.dtype)
            (audio, envelope) = decoder(fromQuantizer(latents), cache: state?.cache)
        } else {
            audio = MLXArray.zeros([codes.dim(0), tail])
            envelope = MLXArray.zeros([tail])
        }
        if let state {
            state.batchSize = codes.dim(0)
            if let previousAudio = state.audio, let previousEnvelope = state.envelope {
                audio[0..., ..<tail] = audio[0..., ..<tail] + previousAudio
                envelope[..<tail] = envelope[..<tail] + previousEnvelope
            }
            if !state.started {
                audio = audio[0..., trim...]
                envelope = envelope[trim...]
                state.started = true
            }
            if finalize {
                audio = audio[0..., ..<(audio.dim(1) - trim)]
                envelope = envelope[..<(envelope.size - trim)]
                state.audio = nil
                state.envelope = nil
                state.finished = true
            } else {
                let end = audio.dim(1) - tail
                state.audio = audio[0..., end...]
                state.envelope = envelope[end...]
                audio = audio[0..., ..<end]
                envelope = envelope[..<end]
            }
        } else {
            audio = audio[0..., trim ..< (audio.dim(1) - trim)]
            envelope = envelope[trim ..< (envelope.size - trim)]
        }
        let result = audio / envelope
        guard all(isFinite(result)).item(Bool.self) else {
            state?.finished = true
            throw VLMError.processing("LiquidCodec produced non-finite audio")
        }
        return result
    }
}

final class LiquidCodecStreamingState {
    fileprivate let owner: ObjectIdentifier
    fileprivate let cache: [KVCache]
    fileprivate var audio: MLXArray?
    fileprivate var envelope: MLXArray?
    fileprivate var batchSize: Int?
    fileprivate var started = false
    fileprivate var finished = false

    fileprivate init(owner: ObjectIdentifier, cache: [KVCache]) {
        self.owner = owner
        self.cache = cache
    }
}
