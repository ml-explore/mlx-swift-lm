// Copyright © 2026 Apple Inc.

import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXDecisions

// Unit tests run on the CPU backend (see LFM2BidirectionalTests).

/// `audio_config` of the tiny d1_omni case in mlx-vlm's `extraction_cases.json`.
let tinyAudioConfigJSON = """
    {
      "feat_in": 16, "n_layers": 1, "d_model": 16, "subsampling_conv_channels": 4,
      "ff_expansion_factor": 2, "n_heads": 2, "conv_kernel_size": 3, "residual_width": 8
    }
    """

private func tinyAudioConfig() throws -> D1OmniAudioConfiguration {
    try JSONDecoder().decode(D1OmniAudioConfiguration.self, from: Data(tinyAudioConfigJSON.utf8))
}

/// A 32-bit integer hash of `0 ..< count`, the same as `hashed` in
/// `tools/d1_fixtures.py`, so Python can build the same weights and clips.
private func hashed(count: Int, seed: Int) -> [UInt32] {
    (0 ..< count).map { k in
        var h = UInt32(k) &* 0x9E37_79B1 &+ UInt32(seed) &* 0x85EB_CA77
        h ^= h >> 16
        h = h &* 0x7FEB_352D
        h ^= h >> 15
        h = h &* 0x846C_A68B
        h ^= h >> 16
        return h
    }
}

/// int16 noise in [-8192, 8191] (`pcm` in the fixture script).
private func hashedSamples(_ count: Int) -> [Int16] {
    hashed(count: count, seed: count).map { Int16(bitPattern: UInt16($0 >> 16)) >> 2 }
}

private func hashedPCM(_ count: Int) -> MLXArray {
    MLXArray(hashedSamples(count))
}

private func floats(_ pcm: [Int16]) -> [Float] {
    pcm.map { (sample: Int16) -> Float in Float(sample) / 32768 }
}

/// Runs `body` with a WAV file of `samples`, then removes it.
private func withWAV<R>(
    _ samples: [Int16], channels: Int = 1, sampleRate: Int = 16000,
    _ body: (URL) async throws -> R
) async throws -> R {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appending(component: "clip.wav")
    try wavData(samples, channels: channels, sampleRate: sampleRate).write(to: file)
    return try await body(file)
}

/// Writes `samples` as a 16 kHz mono float32 WAV file with AVAudioFile.
private func writeAudio(_ samples: [Float], to file: URL) throws {
    let format = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false))
    let buffer = try #require(
        AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
    buffer.frameLength = AVAudioFrameCount(samples.count)
    let channel = try #require(buffer.floatChannelData?[0])
    channel.update(from: samples, count: samples.count)
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true, AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false, AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1,
    ]
    let writer = try AVAudioFile(
        forWriting: file, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    try writer.write(from: buffer)
}

private func fileWaveform(_ file: URL) async throws -> [Float] {
    try await d1OmniWaveform(.url(file)).asArray(Float.self)
}

/// The tiny audio model with hashed weights in [-0.5, 0.5); running variances are in [0.5, 1].
private func tinyAudio(outputDimensions: Int = 64) throws -> D1OmniAudio {
    let model = D1OmniAudio(try tinyAudioConfig(), outputDimensions: outputDimensions)
    var weights = [String: MLXArray]()
    let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
    for (index, (key, value)) in parameters.enumerated() {
        let uniform = hashed(count: value.size, seed: index).map { Float($0 >> 8) / 0x1p24 - 0.5 }
        let w = MLXArray(uniform, value.shape)
        weights[key] = key.hasSuffix("running_var") ? abs(w) + 0.5 : w
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    return model
}

private func noise(_ count: Int, seed: UInt64 = 7) -> MLXArray {
    MLXRandom.normal([count], key: MLXRandom.key(seed)) * 0.1
}

@Suite("d1-omni audio")
struct D1OmniAudioTests {

    // MARK: Configuration

    @Test("audio_config decodes snake_case keys and falls back to the Python defaults")
    func configuration() throws {
        let tiny = try tinyAudioConfig()
        #expect(
            tiny
                == D1OmniAudioConfiguration(
                    featIn: 16, layerCount: 1, dModel: 16, subsamplingConvChannels: 4,
                    ffExpansionFactor: 2, headCount: 2, convKernelSize: 3, residualWidth: 8))
        let defaults = try JSONDecoder().decode(
            D1OmniAudioConfiguration.self, from: Data("{}".utf8))
        #expect(defaults == D1OmniAudioConfiguration())
        #expect(defaults.featIn == 128 && defaults.layerCount == 17 && defaults.dModel == 512)
        #expect(defaults.subsamplingConvChannels == 256 && defaults.ffExpansionFactor == 4)
        #expect(defaults.headCount == 8 && defaults.convKernelSize == 9)
        #expect(defaults.residualWidth == 512)
    }

    // MARK: Waveform

    @Test("int16 samples are scaled, other dtypes are cast, and short clips are padded")
    func waveformInputs() throws {
        let pcm: [Int16] = [-32768, -1, 0, 1, 32767]
        let samples = pcm.map { Float($0) / 32768 }
        try Device.withDefaultDevice(.cpu) {
            for input in [MLXArray(pcm), MLXArray(samples)] {
                let x = try d1OmniWaveform(input)
                #expect(x.dtype == .float32)
                #expect(x.shape == [8000])
                #expect(x[..<5].asArray(Float.self) == samples)
                #expect(!x[5...].any().item(Bool.self))
            }
            // Python casts every other dtype to float32 without scaling.
            let int32 = try d1OmniWaveform(MLXArray([Int32(3), -2]))
            #expect(int32[..<2].asArray(Float.self) == [3, -2])
            let half = try d1OmniWaveform(MLXArray([Float(0.25), -0.5]).asType(.float16))
            #expect(half.dtype == .float32)
            #expect(half[..<2].asArray(Float.self) == [0.25, -0.5])
        }
    }

    @Test("Clips are cut to 30 s, empty clips become 0.5 s of silence, stereo is rejected")
    func waveformLimits() throws {
        try Device.withDefaultDevice(.cpu) {
            #expect(try d1OmniWaveform(MLXArray.ones([31 * 16000])).shape == [480000])
            let full = try d1OmniWaveform(MLXArray.ones([480000]))
            #expect(full.shape == [480000])
            let empty = try d1OmniWaveform(MLXArray.zeros([0], type: Int16.self))
            #expect(empty.shape == [8000])
            #expect(empty.dtype == .float32)
            #expect(!empty.any().item(Bool.self))
        }
        let error = #expect(throws: DecisionError.self) {
            try d1OmniWaveform(MLXArray.zeros([8000, 2], type: Int16.self))
        }
        guard case .invalidAudio(let message) = error else {
            Issue.record("expected invalidAudio, got \(String(describing: error))")
            return
        }
        #expect(message.contains("mono"))
    }

    // MARK: Files

    @Test("A 16 kHz mono WAV decodes to its samples, and a long one to its first 30 s")
    func sixteenKilohertzFilesAreExact() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let pcm = hashedSamples(12_000)
            let short = try await withWAV(pcm, fileWaveform)
            #expect(short == floats(pcm))
            let long = hashedSamples(40 * 16_000)
            let cut = D1OmniMelFrontend.maximumSamples
            let longWaveform = try await withWAV(long, fileWaveform)
            #expect(longWaveform == floats(Array(long[..<cut])))
        }
    }

    @Test("Files at other rates and stereo files are converted to 16 kHz mono")
    func otherFilesAreConverted() async throws {
        try await Device.withDefaultDevice(.cpu) {
            for (rate, channels) in [(44_100, 1), (48_000, 2), (8_000, 1)] {
                let pcm = hashedSamples(2 * rate * channels)
                let waveform = try await withWAV(
                    pcm, channels: channels, sampleRate: rate, fileWaveform)
                // AVFoundation's converter can end a few samples short.
                #expect(abs(waveform.count - 32_000) <= 64, "\(rate) Hz, \(channels) channels")
            }
        }
    }

    @Test("NaN or infinite samples throw invalidAudio, in arrays and in files")
    func nonFiniteSamplesThrow() async throws {
        let error = DecisionError.invalidAudio("the clip has NaN or infinite samples")
        for value in [Float.nan, .infinity, -.infinity] {
            #expect(throws: error) {
                try d1OmniWaveform(MLXArray([0, value, 0.5]))
            }
        }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(component: "nan.wav")
        try writeAudio([0, .nan, 0.25, 0], to: file)
        await #expect(throws: error) {
            try await d1OmniWaveform(.url(file))
        }
    }

    // MARK: Lengths

    @Test("Frame counts match the reference table")
    func lengthsTable() throws {
        let table: [(samples: Int, mel: Int, valid: Int, encoder: Int, prefix: Int)] = [
            (8000, 51, 50, 7, 7),
            (16640, 105, 104, 14, 13),
            (176000, 1101, 1100, 138, 138),
            (480000, 3001, 3000, 376, 375),
        ]
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            // Shapes and lengths only; nothing is evaluated.
            for row in table {
                let (features, valid) = model.frontend(MLXArray.zeros([row.samples]))
                let (encoded, length) = model.encoder(features, validFrames: valid)
                #expect(features.dim(1) == row.mel)
                #expect(valid == row.valid)
                #expect(encoded.dim(1) == row.encoder)
                #expect(length == row.prefix)
            }
        }
    }

    // MARK: Front end

    @Test("The Slaney filterbank matches librosa's checksums")
    func filterbank() {
        let bins = 257
        let fb = D1OmniMelFrontend.slaneyFilterbank(melBins: 128)
        #expect(fb.count == 128 * bins)
        let sum = fb.reduce(0.0) { $0 + Double($1) }
        #expect(abs(sum - 4.09048748) < 1e-6)
        #expect(abs(fb[1] - 0.0283775423) < 1e-9)
        let peak = fb.indices.max { fb[$0] < fb[$1] }!
        #expect(peak == 3 * bins + 3)
        #expect(abs(fb[peak] - 0.04236608) < 1e-8)
        let lastRow = (0 ..< bins).filter { fb[127 * bins + $0] != 0 }
        #expect(lastRow == Array(244 ... 255))
        // Each filter is one contiguous triangle.
        for row in 0 ..< 128 {
            let columns = (0 ..< bins).filter { fb[row * bins + $0] != 0 }
            #expect(!columns.isEmpty)
            #expect(columns.last! - columns.first! + 1 == columns.count)
        }
    }

    @Test("The 400-sample Hann window is centred in 512 samples")
    func window() {
        let window = D1OmniMelFrontend.window()
        #expect(window.count == 512)
        #expect(window[..<56].allSatisfy { $0 == 0 })
        #expect(window[456...].allSatisfy { $0 == 0 })
        #expect(window[56] == 0)
        #expect(window[55 + 400] < 1e-7)
        for n in 0 ..< 400 {
            #expect(abs(window[56 + n] - window[455 - n]) < 2e-7)
        }
        #expect(abs(window.reduce(0.0) { $0 + Double($1) } - 199.5) < 1e-4)
        #expect(window[56 + 199] > 0.99998 && window[56 + 200] > 0.99998)
    }

    @Test("Features match a direct DFT log-mel and are normalised over the valid frames")
    func featuresMatchDirectComputation() {
        let count = 8000
        let x = Device.withDefaultDevice(.cpu) { noise(count).asArray(Float.self) }
        let (features, valid) = Device.withDefaultDevice(.cpu) {
            D1OmniMelFrontend(melBins: 128)(MLXArray(x))
        }
        #expect(valid == 50)
        #expect(features.shape == [1, 51, 128])
        let got = features.asArray(Float.self)

        // Direct computation in Double.
        let n = 512
        let bins = n / 2 + 1
        let window = D1OmniMelFrontend.window().map(Double.init)
        let fb = D1OmniMelFrontend.slaneyFilterbank(melBins: 128).map(Double.init)
        var emphasized = [Double](repeating: 0, count: count + n)
        for t in 0 ..< count {
            let value = Double(x[t]) - (t > 0 ? Double(Float(0.97)) * Double(x[t - 1]) : 0)
            emphasized[256 + t] = value
        }
        var cosTable = [Double](repeating: 0, count: n * bins)
        var sinTable = [Double](repeating: 0, count: n * bins)
        for k in 0 ..< bins {
            for j in 0 ..< n {
                let angle = 2 * Double.pi * Double(k * j % n) / Double(n)
                cosTable[k * n + j] = Foundation.cos(angle)
                sinTable[k * n + j] = Foundation.sin(angle)
            }
        }
        var mel = [[Double]](repeating: [Double](repeating: 0, count: 128), count: 51)
        for frame in 0 ..< 51 {
            var power = [Double](repeating: 0, count: bins)
            for k in 0 ..< bins {
                var re = 0.0
                var im = 0.0
                for j in 0 ..< n {
                    let s = emphasized[frame * 160 + j] * window[j]
                    re += s * cosTable[k * n + j]
                    im -= s * sinTable[k * n + j]
                }
                power[k] = re * re + im * im
            }
            for m in 0 ..< 128 {
                var sum = 0.0
                for k in 0 ..< bins { sum += fb[m * bins + k] * power[k] }
                mel[frame][m] = Foundation.log(sum + 0x1p-24)
            }
        }
        var worst = 0.0
        for m in 0 ..< 128 {
            let column = (0 ..< 50).map { mel[$0][m] }
            let mean = column.reduce(0, +) / 50
            let std = (column.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 49).squareRoot()
            for frame in 0 ..< 50 {
                let expected = (mel[frame][m] - mean) / (std + 1e-5)
                worst = Swift.max(worst, abs(expected - Double(got[frame * 128 + m])))
            }
            #expect(got[50 * 128 + m] == 0)
        }
        #expect(worst < 1e-3)
    }

    // MARK: Encoder

    @Test("rel_shift moves relative position T-1-i+j to column j")
    func relShift() {
        Device.withDefaultDevice(.cpu) {
            let (h, t) = (2, 5)
            let p = 2 * t - 1
            let scores = MLXRandom.normal([1, h, t, p], key: MLXRandom.key(3))
            let shifted = d1OmniRelShift(scores)
            #expect(shifted.shape == [1, h, t, p])
            let a = scores.asArray(Float.self)
            let b = shifted.asArray(Float.self)
            for head in 0 ..< h {
                for i in 0 ..< t {
                    for j in 0 ..< t {
                        let base = (head * t + i) * p
                        #expect(b[base + j] == a[base + t - 1 - i + j])
                    }
                }
            }
        }
    }

    @Test("Relative positions are interleaved sin/cos from T-1 down to -(T-1)")
    func relativePositions() {
        Device.withDefaultDevice(.cpu) {
            let table = D1OmniConformer.relativePositions(count: 3, dimensions: 4)
            #expect(table.shape == [5, 4])
            let values = table.asArray(Float.self)
            let inverse = [1.0, Foundation.exp(2 * -(Foundation.log(10_000.0) / 4))]
            for row in 0 ..< 5 {
                let position = Double(2 - row)
                for (i, frequency) in inverse.enumerated() {
                    let angle = position * frequency
                    #expect(abs(Double(values[row * 4 + 2 * i]) - Foundation.sin(angle)) < 1e-6)
                    #expect(
                        abs(Double(values[row * 4 + 2 * i + 1]) - Foundation.cos(angle)) < 1e-6)
                }
            }
        }
    }

    @Test(
        "The tiny model matches mlx-vlm on the same weights and clips",
        arguments: [12000, 12800])
    func tinyModelMatchesPython(sampleCount: Int) throws {
        // prefix[0, :, ::16] from `python tools/d1_fixtures.py tiny-audio` (mlx-vlm
        // 370f4ccf on the CPU).
        // 12800 samples give 80 valid mel frames, so the encoder has one padded frame.
        let golden: [Int: [Float]] = [
            12000: [
                0.553351, 0.654037, 0.782454, -1.19793,
                0.677955, 0.445838, 0.473903, -1.16988,
                0.636717, 0.47333, 0.561729, -1.16299,
                0.636602, 0.44792, 0.514891, -1.1337,
                0.649937, 0.465429, 0.521778, -1.16622,
                0.627132, 0.442336, 0.537478, -1.12583,
                0.678099, 0.434619, 0.457297, -1.15538,
                0.630927, 0.451946, 0.530861, -1.13406,
                0.691353, 0.42383, 0.431957, -1.15132,
                0.622487, 0.434195, 0.514618, -1.10298,
            ],
            12800: [
                0.495225, 0.775375, 0.93665, -1.18523,
                0.65333, 0.442835, 0.495558, -1.14395,
                0.684001, 0.425781, 0.455638, -1.14952,
                0.637645, 0.478433, 0.545303, -1.15697,
                0.615154, 0.493524, 0.607929, -1.1455,
                0.663478, 0.45473, 0.498319, -1.16056,
                0.624137, 0.488419, 0.57085, -1.15786,
                0.635638, 0.422992, 0.484012, -1.10441,
                0.645903, 0.446191, 0.516226, -1.13784,
                0.67236, 0.431182, 0.482538, -1.14544,
            ],
        ]
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            #expect(model.parameters().flattened().count == 63)
            let out = model(try d1OmniWaveform(hashedPCM(sampleCount)))
            #expect(out.shape == [1, 10, 64])
            #expect(out.dtype == .float32)
            let expected = MLXArray(golden[sampleCount]!, [10, 4])
            let error = maxAbsDiff(out[0, 0..., .stride(by: 16)], expected)
            #expect(error < 1e-4)
        }
    }

    @Test("The padding mel frame never reaches the encoder")
    func subsamplingMasksPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            // 56 valid mel frames: the encoder has 8 frames, the last one padding.
            let (features, valid) = model.frontend(noise(8960))
            #expect(valid == 56)
            let (clean, length) = model.encoder(features, validFrames: valid)
            #expect(clean.shape == [1, 8, 16])
            #expect(length == 7)
            let garbage = MLXRandom.normal([1, 1, 16], key: MLXRandom.key(11)) * 100
            let noisy = concatenated([features[0..., ..<valid], garbage], axis: 1)
            let (dirty, _) = model.encoder(noisy, validFrames: valid)
            #expect(maxAbsDiff(clean, dirty) == 0)
        }
    }

    @Test("Attention and the conv module ignore padded encoder frames")
    func blockMasksPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            let block = model.encoder.layers[0]
            let positions = D1OmniConformer.relativePositions(count: 8, dimensions: 16)
            let valid = MLXArray(0 ..< 8) .< 7
            let h = MLXRandom.normal([1, 8, 16], key: MLXRandom.key(12))
            let garbage = MLXRandom.normal([1, 1, 16], key: MLXRandom.key(13)) * 100
            let changed = concatenated([h[0..., ..<7], garbage], axis: 1)

            let clean = block(h, positions: positions, valid: valid)
            let dirty = block(changed, positions: positions, valid: valid)
            #expect(maxAbsDiff(clean[0..., ..<7], dirty[0..., ..<7]) < 1e-5)
            // Without the masks the padded frame leaks into the real ones.
            let unmasked = block(h, positions: positions, valid: nil)
            let leaked = block(changed, positions: positions, valid: nil)
            #expect(maxAbsDiff(unmasked[0..., ..<7], leaked[0..., ..<7]) > 1e-2)
        }
    }

    @Test("BatchNorm reads its running statistics, and training mode would change the prefix")
    func batchNormEvalMode() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            #expect(model.modules().allSatisfy { !$0.training })

            let norm = model.encoder.layers[0].conv.norm
            let stats = Dictionary(uniqueKeysWithValues: norm.parameters().flattened())
            let x = MLXRandom.normal([1, 9, 16], key: MLXRandom.key(5))
            let expected =
                (x - stats["running_mean"]!) * rsqrt(stats["running_var"]! + 1e-5)
                * stats["weight"]! + stats["bias"]!
            #expect(maxAbsDiff(norm(x), expected) < 1e-5)

            // A forward pass in training mode also overwrites the running statistics.
            let waveform = try d1OmniWaveform(noise(12000))
            let trained = try tinyAudio()
            trained.train(true)
            #expect(maxAbsDiff(model(waveform), trained(waveform)) > 1e-2)
        }
    }

    // MARK: Weights

    @Test("The full-size module tree has the checkpoint's 687 audio parameters")
    func fullSizeParameters() {
        Device.withDefaultDevice(.cpu) {
            let model = D1OmniAudio(D1OmniAudioConfiguration(), outputDimensions: 1024)
            let shapes = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(shapes.count == 687)
            #expect(shapes["encoder.subsampling.layers.0.weight"] == [256, 3, 3, 1])
            #expect(shapes["encoder.subsampling.layers.2.weight"] == [256, 3, 3, 1])
            #expect(shapes["encoder.subsampling.layers.3.weight"] == [256, 1, 1, 256])
            #expect(shapes["encoder.subsampling.linear.weight"] == [512, 4096])
            #expect(shapes["encoder.layers.16.self_attn.relative_k_proj.weight"] == [512, 512])
            #expect(shapes["encoder.layers.16.self_attn.relative_k_proj.bias"] == nil)
            #expect(shapes["encoder.layers.0.self_attn.bias_u"] == [8, 64])
            #expect(shapes["encoder.layers.0.conv.pointwise_conv1.weight"] == [1024, 1, 512])
            #expect(shapes["encoder.layers.0.conv.depthwise_conv.weight"] == [512, 9, 1])
            #expect(shapes["encoder.layers.0.conv.norm.running_var"] == [512])
            #expect(shapes["encoder.layers.0.feed_forward1.linear1.weight"] == [2048, 512])
            #expect(shapes["adapter.linear_1.weight"] == [1024, 512])
            #expect(shapes["residual.down.weight"] == [512, 1024])
            #expect(shapes["residual.up.weight"] == [1024, 512])
        }
    }
}

// MARK: - Parity with mlx-vlm (real weights)

/// The audio prefix against mlx-vlm 370f4ccf on clips cut from `jfk.wav`, around the front
/// end's edges (0.5 s padding, the 30 s cut, int16 input). Needs `MLX_D1_OMNI_MODEL` and
/// `MLX_D1_FIXTURES` (see README.md).
@Suite(
    "d1-omni audio parity",
    .enabled(if: D1Fixtures.omniModel != nil && D1Fixtures.directory != nil))
struct D1OmniAudioParityTests {

    private func loadModel() throws -> D1OmniAudio {
        let directory = try #require(D1Fixtures.omniModel)
        let all = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        var weights = [String: MLXArray]()
        for (key, value) in d1OmniSanitizeAudio(all) where key.hasPrefix("audio.") {
            weights[String(key.dropFirst("audio.".count))] = value
        }
        #expect(weights.count == 687)
        let model = D1OmniAudio(D1OmniAudioConfiguration(), outputDimensions: 1024)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }

    /// Max abs difference and worst-row cosine of two `[rows, D]` arrays.
    private func report(_ label: String, _ a: MLXArray, _ b: MLXArray) -> (
        maxAbs: Float, cosine: Float
    ) {
        let a = a.asType(.float32)
        let b = b.asType(.float32)
        let cosine =
            (a * b).sum(axis: -1)
            / (sqrt(a.square().sum(axis: -1)) * sqrt(b.square().sum(axis: -1)))
        let result = (maxAbsDiff(a, b), cosine.min().item(Float.self))
        print(
            "d1-omni audio parity \(label): maxAbs \(String(format: "%.3e", result.0))"
                + " worstRowCos \(String(format: "%.8f", result.1))")
        return result
    }

    @Test("The prefix matches mlx-vlm")
    func prefixes() async throws {
        let mlxvlm = try loadArrays(
            url: try D1Fixtures.url("d1-omni-600M-float32-audio.safetensors"))
        let model = try loadModel()
        let names = mlxvlm.keys.filter { $0.hasSuffix(".input") }
            .map { String($0.dropLast(".input".count)) }.sorted()
        try #require(!names.isEmpty)

        for name in names {
            let waveform = try d1OmniWaveform(mlxvlm["\(name).input"]!)
            let (mel, valid) = model.frontend(waveform)
            let (encoded, length) = model.encoder(mel, validFrames: valid)
            let prefix = model(waveform)
            #expect(valid == mlxvlm["\(name).mel_valid"]!.item(Int.self))
            #expect(length == mlxvlm["\(name).enc_valid"]!.item(Int.self))
            #expect(encoded.dim(1) == mlxvlm["\(name).enc"]!.dim(0))
            #expect(prefix.shape == [1, mlxvlm["\(name).prefix"]!.dim(0), 1024])
            #expect(!mel[0, valid...].any().item(Bool.self))

            let python = report("\(name) prefix vs mlx-vlm", prefix[0], mlxvlm["\(name).prefix"]!)
            #expect(python.maxAbs <= 5e-4, "\(name)")
            #expect(python.cosine >= 0.99999, "\(name)")
        }

        // The file decodes to exactly the samples mlx-vlm read.
        let decoded = try await d1OmniWaveform(.url(try D1Fixtures.url("jfk.wav")))
        #expect(decoded.shape == [176000])
        #expect(maxAbsDiff(decoded, mlxvlm["jfk11s.input"]!) == 0)
    }
}
