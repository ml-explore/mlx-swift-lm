// Copyright © 2026 Apple Inc.
// Port of mlx-vlm's d1-omni model (Blaizzy/mlx-vlm#2467), which follows LiquidAI's reference code.
// The Parakeet encoder and feature extractor come from mlx-vlm's nemotron_h_nano_omni/audio.py.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The `audio_config` of a d1-omni checkpoint. Missing keys take the Python defaults.
struct D1OmniAudioConfiguration: Codable, Sendable, Equatable {
    var featIn: Int
    var layerCount: Int
    var dModel: Int
    var subsamplingConvChannels: Int
    var ffExpansionFactor: Int
    var headCount: Int
    var convKernelSize: Int
    var residualWidth: Int

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case layerCount = "n_layers"
        case dModel = "d_model"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case ffExpansionFactor = "ff_expansion_factor"
        case headCount = "n_heads"
        case convKernelSize = "conv_kernel_size"
        case residualWidth = "residual_width"
    }

    init(
        featIn: Int = 128, layerCount: Int = 17, dModel: Int = 512,
        subsamplingConvChannels: Int = 256, ffExpansionFactor: Int = 4, headCount: Int = 8,
        convKernelSize: Int = 9, residualWidth: Int = 512
    ) {
        self.featIn = featIn
        self.layerCount = layerCount
        self.dModel = dModel
        self.subsamplingConvChannels = subsamplingConvChannels
        self.ffExpansionFactor = ffExpansionFactor
        self.headCount = headCount
        self.convKernelSize = convKernelSize
        self.residualWidth = residualWidth
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        func value(_ key: CodingKeys, _ fallback: Int) throws -> Int {
            try container.decodeIfPresent(Int.self, forKey: key) ?? fallback
        }
        self.init(
            featIn: try value(.featIn, defaults.featIn),
            layerCount: try value(.layerCount, defaults.layerCount),
            dModel: try value(.dModel, defaults.dModel),
            subsamplingConvChannels: try value(
                .subsamplingConvChannels, defaults.subsamplingConvChannels),
            ffExpansionFactor: try value(.ffExpansionFactor, defaults.ffExpansionFactor),
            headCount: try value(.headCount, defaults.headCount),
            convKernelSize: try value(.convKernelSize, defaults.convKernelSize),
            residualWidth: try value(.residualWidth, defaults.residualWidth))
    }
}

// MARK: - Waveform

/// Mono 16 kHz samples -> float32 `[N]`, cut to 30 s and zero padded to 0.5 s (Python `waveform`).
///
/// `int16` samples are divided by 32768; every other dtype is cast to float32 unscaled. NaN or
/// infinite samples throw.
func d1OmniWaveform(_ samples: MLXArray) throws -> MLXArray {
    guard samples.ndim == 1 else {
        throw DecisionError.invalidAudio("audio must be mono: a 1-D array of 16 kHz samples")
    }
    let minimum = D1OmniMelFrontend.minimumSamples
    let count = Swift.min(samples.dim(0), D1OmniMelFrontend.maximumSamples)
    guard count > 0 else {
        return MLXArray.zeros([minimum], type: Float.self)
    }
    // On the CPU, which also handles float64.
    return try Device.withDefaultDevice(.cpu) {
        var x = samples[..<count]
        x = samples.dtype == .int16 ? x.asType(.float32) / Float(32768) : x.asType(.float32)
        guard all(isFinite(x)).item(Bool.self) else {
            throw DecisionError.invalidAudio("the clip has NaN or infinite samples")
        }
        if count < minimum {
            x = padded(x, widths: [IntOrPair((0, minimum - count))])
        }
        return x
    }
}

/// A request's clip as a waveform. Files are decoded to 16 kHz mono by
/// `UserInput.Audio.asMLXArray`.
func d1OmniWaveform(_ audio: UserInput.Audio) async throws -> MLXArray {
    switch audio.source {
    case .array(let samples):
        return try d1OmniWaveform(samples)
    case .url(let url):
        var processing = UserInput.AudioProcessing()
        processing.sampleRate = Double(D1OmniMelFrontend.sampleRate)
        let samples: MLXArray
        do {
            // A new value, not the caller's, crosses into the concurrent decoder.
            samples = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        } catch let error as CancellationError {
            throw error
        } catch {
            throw DecisionError.invalidAudio(
                "\(url.lastPathComponent): \(error.localizedDescription)")
        }
        guard samples.size > 0 else {
            throw DecisionError.invalidAudio("\(url.lastPathComponent): the file has no samples")
        }
        return try d1OmniWaveform(samples)
    }
}

// MARK: - Mel front end

/// NeMo `FilterbankFeatures` in eval mode: preemphasis, centred STFT, power, Slaney mel, log and
/// per-bin normalisation over the valid frames. Always float32.
struct D1OmniMelFrontend {
    static let sampleRate = 16_000
    static let minimumSamples = 8_000
    static let maximumSamples = 480_000
    static let fftSize = 512
    static let windowLength = 400
    static let hopLength = 160
    static let preemphasis: Float = 0.97
    static let logGuard: Float = 0x1p-24
    static let normalizationEpsilon: Float = 1e-5

    /// `[fftSize]`: a symmetric 400-sample Hann window, centred.
    let window: MLXArray
    /// `[fftSize / 2 + 1, melBins]`
    let filterbank: MLXArray

    init(melBins: Int) {
        self.window = MLXArray(Self.window())
        self.filterbank = MLXArray(
            Self.slaneyFilterbank(melBins: melBins), [melBins, Self.fftSize / 2 + 1]
        ).transposed()
    }

    /// `torch.hann_window(400, periodic=False)` zero padded to 512 on both sides (56 + 400 + 56).
    static func window() -> [Float] {
        let left = (fftSize - windowLength) / 2
        var window = [Float](repeating: 0, count: fftSize)
        let denominator = Double(windowLength - 1)
        for n in 0 ..< windowLength {
            window[left + n] = Float(
                0.5 * (1 - Foundation.cos(2 * Double.pi * Double(n) / denominator)))
        }
        return window
    }

    /// librosa `filters.mel(sr, n_fft, n_mels, norm="slaney")`, row-major `[melBins, fftSize / 2 + 1]`.
    ///
    /// Computed in Double with librosa's float32 roundings, which reproduces it bit for bit.
    static func slaneyFilterbank(melBins: Int) -> [Float] {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = Foundation.log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + Foundation.log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * Foundation.exp(logStep * (m - minLogMel)) : fSp * m
        }

        // np.linspace: i * step + start, with the last point set to stop.
        let low = hzToMel(0)
        let high = hzToMel(Double(sampleRate) / 2)
        let points = melBins + 2
        let step = (high - low) / Double(points - 1)
        var mels = (0 ..< points).map { Double($0) * step + low }
        mels[points - 1] = high
        let hz = mels.map(melToHz)

        // np.fft.rfftfreq
        let binWidth = 1.0 / (Double(fftSize) * (1.0 / Double(sampleRate)))
        let bins = fftSize / 2 + 1
        var weights = [Float](repeating: 0, count: melBins * bins)
        for i in 0 ..< melBins {
            let norm = 2.0 / (hz[i + 2] - hz[i])
            for k in 0 ..< bins {
                let f = Double(k) * binWidth
                let lower = -(hz[i] - f) / (hz[i + 1] - hz[i])
                let upper = (hz[i + 2] - f) / (hz[i + 2] - hz[i + 1])
                let triangle = Float(Swift.max(0, Swift.min(lower, upper)))
                weights[i * bins + k] = Float(Double(triangle) * norm)
            }
        }
        return weights
    }

    /// A waveform `[N]` -> features `[1, N / 160 + 1, melBins]` and the number of valid frames.
    func callAsFunction(_ waveform: MLXArray) -> (features: MLXArray, validFrames: Int) {
        let count = waveform.dim(0)
        let validFrames = count / Self.hopLength
        let frameCount = validFrames + 1

        let x = waveform.asType(.float32)
        let emphasized = concatenated([x[..<1], x[1...] - Self.preemphasis * x[..<(count - 1)]])
        let half = Self.fftSize / 2
        let signal = padded(emphasized, widths: [IntOrPair((half, half))])
        let frames = asStrided(signal, [frameCount, Self.fftSize], strides: [Self.hopLength, 1])
        // The default stream, as in mlx-vlm. On a sine sweep, mlx-vlm's CPU and GPU runs give
        // prefixes 1e-2 apart.
        let power = abs(MLXFFT.rfft(frames * window, axis: -1)).square()
        let mel = log(matmul(power, filterbank) + Self.logGuard)

        let valid = mel[..<validFrames]
        let mean = valid.sum(axis: 0) / Float(validFrames)
        let variance =
            (valid - mean).square().sum(axis: 0) / Float(Swift.max(validFrames - 1, 1))
        let keep = (MLXArray(0 ..< frameCount) .< validFrames).asType(.float32)
        let features =
            (mel - mean) / (sqrt(variance) + Self.normalizationEpsilon)
            * keep[0..., .newAxis]
        return (features.expandedDimensions(axis: 0), validFrames)
    }
}

// MARK: - FastConformer

/// `rel_shift`: pads one zero column on the left and re-reads the scores so that column `j` of
/// query `i` holds relative position `T - 1 - i + j`. `[B, H, T, 2T - 1]` in and out.
func d1OmniRelShift(_ scores: MLXArray) -> MLXArray {
    let (b, h, t, p) = (scores.dim(0), scores.dim(1), scores.dim(2), scores.dim(3))
    let shifted = padded(
        scores, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))])
    return shifted.reshaped(b, h, p + 1, t)[0..., 0..., 1...].reshaped(b, h, t, p)
}

enum D1OmniConformer {
    /// Three stride-2 stages: 8x subsampling.
    static let subsamplingStages = 3

    /// Output length of a kernel-3, stride-2, padding-1 convolution: `ceil(n / 2)`.
    static func halved(_ n: Int) -> Int { (n + 1) / 2 }

    /// Interleaved sin/cos embeddings for relative positions `T - 1` down to `-(T - 1)`: `[2T - 1, D]`.
    static func relativePositions(count: Int, dimensions: Int) -> MLXArray {
        let positions = MLXArray((0 ..< (2 * count - 1)).map { Float(count - 1 - $0) })
        let inverse = MLXArray(
            stride(from: 0, to: dimensions, by: 2).map {
                Float(Foundation.exp(Double($0) * -(Foundation.log(10_000.0) / Double(dimensions))))
            })
        let angles = positions[0..., .newAxis] * inverse[.newAxis, 0...]
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(2 * count - 1, dimensions)
    }

    /// dw_striding subsampling: a conv, then depthwise-separable convs, all with time masks.
    final class Subsampling: Module {
        @ModuleInfo var layers: [UnaryLayer]
        @ModuleInfo var linear: Linear
        /// Stride of each layer, 0 for activations.
        let strides: [Int]

        init(_ config: D1OmniAudioConfiguration) {
            let channels = config.subsamplingConvChannels
            var layers: [UnaryLayer] = [
                Conv2d(
                    inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2,
                    padding: 1),
                ReLU(),
            ]
            var strides = [2, 0]
            for _ in 1 ..< D1OmniConformer.subsamplingStages {
                layers += [
                    Conv2d(
                        inputChannels: channels, outputChannels: channels, kernelSize: 3,
                        stride: 2, padding: 1, groups: channels),
                    Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1),
                    ReLU(),
                ]
                strides += [2, 1, 0]
            }
            var frequencies = config.featIn
            for _ in 0 ..< D1OmniConformer.subsamplingStages {
                frequencies = D1OmniConformer.halved(frequencies)
            }
            self._layers.wrappedValue = layers
            self._linear.wrappedValue = Linear(channels * frequencies, config.dModel)
            self.strides = strides
        }

        /// `[1, T, F]` features -> `[1, T', D]` and the valid length after subsampling.
        func callAsFunction(_ features: MLXArray, validFrames: Int) -> (MLXArray, Int) {
            var h = features[.ellipsis, .newAxis]
            var length = validFrames
            for (layer, stride) in zip(layers, strides) {
                h = layer(h)
                guard stride > 0 else { continue }
                if stride == 2 {
                    length = D1OmniConformer.halved(length)
                }
                let frames = h.dim(1)
                if length < frames {
                    let keep = (MLXArray(0 ..< frames) .< length).asType(h.dtype)
                    h = h * keep.reshaped(1, frames, 1, 1)
                }
            }
            let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
            h = h.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
            return (linear(h), length)
        }
    }

    final class FeedForward: Module, UnaryLayer {
        @ModuleInfo var linear1: Linear
        @ModuleInfo var linear2: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._linear1.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._linear2.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(silu(linear1(x)))
        }
    }

    /// Multi-head self-attention with Transformer-XL relative positions (`bias_u`, `bias_v`).
    final class Attention: Module {
        let headCount: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
        @ParameterInfo(key: "bias_u") var biasU: MLXArray
        @ParameterInfo(key: "bias_v") var biasV: MLXArray

        init(dimensions: Int, headCount: Int) {
            self.headCount = headCount
            self.headDimensions = dimensions / headCount
            self.scale = 1 / Float(headDimensions).squareRoot()
            self._qProj.wrappedValue = Linear(dimensions, dimensions)
            self._kProj.wrappedValue = Linear(dimensions, dimensions)
            self._vProj.wrappedValue = Linear(dimensions, dimensions)
            self._oProj.wrappedValue = Linear(dimensions, dimensions)
            self._relativeKProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
            self._biasU.wrappedValue = MLXArray.zeros([headCount, headDimensions])
            self._biasV.wrappedValue = MLXArray.zeros([headCount, headDimensions])
        }

        /// `valid` marks the real frames `[T]`; nil when every frame is real.
        func callAsFunction(_ x: MLXArray, positions: MLXArray, valid: MLXArray?) -> MLXArray {
            let (b, t) = (x.dim(0), x.dim(1))
            func heads(_ y: MLXArray) -> MLXArray {
                y.reshaped(b, t, headCount, headDimensions).transposed(0, 2, 1, 3)
            }
            let q = heads(qProj(x))
            let k = heads(kProj(x))
            let v = heads(vProj(x))
            let p = relativeKProj(positions).reshaped(1, -1, headCount, headDimensions)
                .transposed(0, 2, 3, 1)

            let u = biasU.reshaped(1, headCount, 1, headDimensions)
            let w = biasV.reshaped(1, headCount, 1, headDimensions)
            var bd = d1OmniRelShift(matmul(q + w, p))[.ellipsis, ..<t] * scale
            if let valid {
                // Mask keys only; padded query rows are zeroed below.
                bd = MLX.where(
                    valid.reshaped(1, 1, 1, t), bd, -bd.dtype.greatestFiniteMagnitudeArray)
            }
            var out = MLXFast.scaledDotProductAttention(
                queries: q + u, keys: k, values: v, scale: scale, mask: .array(bd))
            if let valid {
                out = out * valid.reshaped(1, 1, t, 1).asType(out.dtype)
            }
            return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
        }
    }

    /// Pointwise conv, GLU, depthwise conv, BatchNorm (running statistics in eval mode), SiLU, pointwise conv.
    final class ConvolutionModule: Module {
        @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
        @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
        @ModuleInfo var norm: BatchNorm
        @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

        init(dimensions: Int, kernelSize: Int) {
            self._pointwiseConv1.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: 2 * dimensions, kernelSize: 1)
            self._depthwiseConv.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: dimensions, kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2, groups: dimensions)
            self._norm.wrappedValue = BatchNorm(featureCount: dimensions)
            self._pointwiseConv2.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: dimensions, kernelSize: 1)
        }

        func callAsFunction(_ x: MLXArray, valid: MLXArray?) -> MLXArray {
            var h = glu(pointwiseConv1(x), axis: -1)
            if let valid {
                h = h * valid.reshaped(1, -1, 1).asType(h.dtype)
            }
            return pointwiseConv2(silu(norm(depthwiseConv(h))))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "feed_forward1") var feedForward1: FeedForward
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var conv: ConvolutionModule
        @ModuleInfo(key: "feed_forward2") var feedForward2: FeedForward
        @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
        @ModuleInfo(key: "norm_self_att") var normSelfAttention: LayerNorm
        @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
        @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
        @ModuleInfo(key: "norm_out") var normOut: LayerNorm

        init(_ config: D1OmniAudioConfiguration) {
            let d = config.dModel
            let hidden = d * config.ffExpansionFactor
            self._feedForward1.wrappedValue = FeedForward(dimensions: d, hiddenDimensions: hidden)
            self._selfAttention.wrappedValue = Attention(dimensions: d, headCount: config.headCount)
            self._conv.wrappedValue = ConvolutionModule(
                dimensions: d, kernelSize: config.convKernelSize)
            self._feedForward2.wrappedValue = FeedForward(dimensions: d, hiddenDimensions: hidden)
            self._normFeedForward1.wrappedValue = LayerNorm(dimensions: d)
            self._normSelfAttention.wrappedValue = LayerNorm(dimensions: d)
            self._normConv.wrappedValue = LayerNorm(dimensions: d)
            self._normFeedForward2.wrappedValue = LayerNorm(dimensions: d)
            self._normOut.wrappedValue = LayerNorm(dimensions: d)
        }

        func callAsFunction(_ x: MLXArray, positions: MLXArray, valid: MLXArray?) -> MLXArray {
            var h = x + 0.5 * feedForward1(normFeedForward1(x))
            h = h + selfAttention(normSelfAttention(h), positions: positions, valid: valid)
            h = h + conv(normConv(h), valid: valid)
            h = h + 0.5 * feedForward2(normFeedForward2(h))
            return normOut(h)
        }
    }

    final class Encoder: Module {
        let dModel: Int
        @ModuleInfo var subsampling: Subsampling
        @ModuleInfo var layers: [Block]

        init(_ config: D1OmniAudioConfiguration) {
            self.dModel = config.dModel
            self._subsampling.wrappedValue = Subsampling(config)
            self._layers.wrappedValue = (0 ..< config.layerCount).map { _ in Block(config) }
        }

        /// `[1, T, F]` features -> `[1, T_enc, D]` and the number of valid frames `P`.
        func callAsFunction(_ features: MLXArray, validFrames: Int) -> (MLXArray, Int) {
            var (h, length) = subsampling(features, validFrames: validFrames)
            let frames = h.dim(1)
            let positions = D1OmniConformer.relativePositions(count: frames, dimensions: dModel)
                .asType(h.dtype)
            let valid: MLXArray? = length < frames ? MLXArray(0 ..< frames) .< length : nil
            for layer in layers {
                h = layer(h, positions: positions, valid: valid)
            }
            return (h, length)
        }
    }
}

// MARK: - Adapter, residual, audio prefix

final class D1OmniAudioAdapter: Module, UnaryLayer {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._norm.wrappedValue = LayerNorm(dimensions: inputDimensions)
        self._linear1.wrappedValue = Linear(inputDimensions, outputDimensions)
        self._linear2.wrappedValue = Linear(outputDimensions, outputDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(norm(x))))
    }
}

final class D1OmniAudioResidual: Module, UnaryLayer {
    @ModuleInfo var ln: LayerNorm
    @ModuleInfo var down: Linear
    @ModuleInfo var up: Linear

    init(dimensions: Int, width: Int) {
        self._ln.wrappedValue = LayerNorm(dimensions: dimensions)
        self._down.wrappedValue = Linear(dimensions, width)
        self._up.wrappedValue = Linear(width, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + up(gelu(down(ln(x))))
    }
}

/// Speech -> prefix embeddings: log-mel front end, FastConformer, adapter and residual.
/// One embedding per 80 ms of audio.
final class D1OmniAudio: Module {
    let frontend: D1OmniMelFrontend
    @ModuleInfo var encoder: D1OmniConformer.Encoder
    @ModuleInfo var adapter: D1OmniAudioAdapter
    @ModuleInfo var residual: D1OmniAudioResidual

    init(_ config: D1OmniAudioConfiguration, outputDimensions: Int) {
        self.frontend = D1OmniMelFrontend(melBins: config.featIn)
        self._encoder.wrappedValue = D1OmniConformer.Encoder(config)
        self._adapter.wrappedValue = D1OmniAudioAdapter(
            inputDimensions: config.dModel, outputDimensions: outputDimensions)
        self._residual.wrappedValue = D1OmniAudioResidual(
            dimensions: outputDimensions, width: config.residualWidth)
        super.init()
        // Inference only: BatchNorm must read its running statistics.
        train(false)
    }

    /// A waveform from ``d1OmniWaveform(_:)`` -> prefix embeddings `[1, P, outputDimensions]`.
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        precondition(
            waveform.ndim == 1 && waveform.dim(0) >= D1OmniMelFrontend.minimumSamples,
            "expected a waveform from d1OmniWaveform")
        let (features, validFrames) = frontend(waveform)
        let dtype = adapter.norm.weight?.dtype ?? .float32
        let (hidden, length) = encoder(features.asType(dtype), validFrames: validFrames)
        return residual(adapter(hidden[0..., ..<length]))
    }
}

// MARK: - Weights

private let d1OmniAudioRenames = [
    ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
    ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
    ("self_attn.linear_q.", "self_attn.q_proj."),
    ("self_attn.linear_k.", "self_attn.k_proj."),
    ("self_attn.linear_v.", "self_attn.v_proj."),
    ("self_attn.linear_out.", "self_attn.o_proj."),
    ("self_attn.linear_pos.", "self_attn.relative_k_proj."),
    ("self_attn.pos_bias_u", "self_attn.bias_u"),
    ("self_attn.pos_bias_v", "self_attn.bias_v"),
    ("conv.batch_norm.", "conv.norm."),
]

/// Maps the checkpoint's NeMo `audio.*` names and torch conv layout to ``D1OmniAudio`` (Python
/// `sanitize_audio`). Other keys pass through. Idempotent, so converted checkpoints load too.
func d1OmniSanitizeAudio(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = [String: MLXArray]()
    for (key, value) in weights {
        guard key.hasPrefix("audio.") else {
            result[key] = value
            continue
        }
        if key.hasSuffix("num_batches_tracked") {
            continue
        }
        var key = key
        for (old, new) in d1OmniAudioRenames {
            key = key.replacingOccurrences(of: old, with: new)
        }
        // torch (out, in/groups, *kernel) -> MLX (out, *kernel, in/groups). 2-D kernels are
        // square; 1-D convs are pointwise (kernel 1) or depthwise (in/groups 1).
        var value = value
        if value.ndim == 4, value.dim(2) == value.dim(3) {
            value = value.transposed(0, 2, 3, 1)
        } else if value.ndim == 3, value.dim(key.contains("pointwise") ? -1 : 1) == 1 {
            value = value.transposed(0, 2, 1)
        }
        result[key] = value
    }
    return result
}
