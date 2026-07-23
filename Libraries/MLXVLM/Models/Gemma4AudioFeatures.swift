// Copyright © 2026 Apple Inc.

//
// Log-mel feature extractor for Gemma 4 audio — a faithful port of HuggingFace
// Transformers `Gemma4AudioFeatureExtractor` (USM preprocessing), via
// Blaizzy/mlx-vlm `models/gemma4/audio_feature_extractor.py`. Raw 16 kHz mono
// waveform -> log-mel `[1, T, 128]` plus a per-frame validity mask `[1, T]`
// (true == valid). Everything runs in MLX (`rfft` is re-exported by `MLX`), so
// the result matches the reference's `mx.fft.rfft` pipeline numerically.
//
// Cross-checked against llama.cpp PR #21421 (`tools/mtmd/mtmd-audio.cpp`): same
// periodic Hann window, HTK mel scale, `sample_rate / n_fft` bin spacing,
// magnitude (not power) spectrum, and `log(mel + mel_floor)`.
//

import Foundation
import MLX

public struct Gemma4AudioFeatureExtractor {
    public let featureSize: Int
    public let sampleRate: Int
    public let frameLength: Int
    public let hopLength: Int
    public let fftLength: Int
    public let melFloor: Float
    public let maxSamples: Int
    public let padToMultipleOf: Int

    private let window: MLXArray  // [frameLength] periodic Hann
    private let melFilters: MLXArray  // [fftLength/2+1, featureSize]

    public init(
        featureSize: Int = 128,
        sampleRate: Int = 16_000,
        frameLengthMs: Double = 20.0,
        hopLengthMs: Double = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        melFloor: Float = 1e-3,
        maxSamples: Int = 480_000,
        padToMultipleOf: Int = 128
    ) {
        self.featureSize = featureSize
        self.sampleRate = sampleRate
        self.frameLength = Int((Double(sampleRate) * frameLengthMs / 1000.0).rounded())
        self.hopLength = Int((Double(sampleRate) * hopLengthMs / 1000.0).rounded())
        self.fftLength = 1 << Int(ceil(log2(Double(self.frameLength))))
        self.melFloor = melFloor
        self.maxSamples = maxSamples
        self.padToMultipleOf = padToMultipleOf

        // Periodic Hann window: w[n] = 0.5 - 0.5·cos(2π n / frameLength).
        let n = MLXArray(Array(0 ..< self.frameLength)).asType(.float32)
        self.window = 0.5 - 0.5 * cos(2.0 * Float.pi * n / Float(self.frameLength))

        let numFreqBins = self.fftLength / 2 + 1
        let filters = Self.melFilterBank(
            numFreqBins: numFreqBins, numMel: featureSize,
            minFrequency: minFrequency, maxFrequency: maxFrequency, sampleRate: sampleRate)
        self.melFilters = MLXArray(filters, [numFreqBins, featureSize])
    }

    /// - Parameter waveform: mono samples `[N]` at `sampleRate` (16 kHz).
    /// - Returns: log-mel features `[1, T, 128]` and a validity mask `[1, T]`
    ///   (`true == valid frame`, i.e. the HF `input_features_mask` convention).
    public func callAsFunction(_ waveform: MLXArray) -> (features: MLXArray, mask: MLXArray) {
        var samples = waveform.asType(.float32)
        if samples.ndim > 1 { samples = samples.reshaped(-1) }
        let n0 = min(samples.dim(0), maxSamples)
        samples = samples[..<n0]

        // Pad the real audio up to a multiple of `padToMultipleOf` samples; the
        // frame mask marks the real prefix. Then semicausal left-pad by
        // `frameLength/2` so the first frame is centred at t=0.
        var target = n0
        if padToMultipleOf > 0 && target % padToMultipleOf != 0 {
            target = (target / padToMultipleOf + 1) * padToMultipleOf
        }
        let padRight = target - n0
        let padLeft = frameLength / 2

        var w = samples
        if padRight > 0 { w = padded(w, widths: [IntOrPair((0, padRight))]) }
        w = padded(w, widths: [IntOrPair((padLeft, 0))])
        let paddedLen = padLeft + target

        // Reference unfolds with size `frameLength + 1` (a preemphasis affordance),
        // then, with preemphasis disabled, keeps the first `frameLength` samples.
        let frameSizeForUnfold = frameLength + 1
        let numFrames = (paddedLen - frameSizeForUnfold) / hopLength + 1
        guard numFrames > 0 else {
            return (
                MLXArray.zeros([1, 0, featureSize]),
                MLXArray.zeros([1, 0]).asType(.bool)
            )
        }

        // Frame f uses samples [f·hop, f·hop + frameLength).
        var idx = [Int32]()
        idx.reserveCapacity(numFrames * frameLength)
        for f in 0 ..< numFrames {
            let start = f * hopLength
            for j in 0 ..< frameLength { idx.append(Int32(start + j)) }
        }
        var frames = take(w, MLXArray(idx), axis: 0).reshaped(numFrames, frameLength)
        frames = frames * window

        // rfft -> magnitude (not power) -> mel -> log.
        let stft = rfft(frames, n: fftLength, axis: -1)  // [numFrames, fftLength/2+1] complex
        let re = stft.realPart()
        let im = stft.imaginaryPart()
        let magnitude = sqrt(re * re + im * im)
        let mel = matmul(magnitude, melFilters)  // [numFrames, featureSize]
        var logMel = log(mel + MLXArray(melFloor))

        // Per-frame validity from the padded sample mask: a frame is valid when
        // its end sample falls inside the real-audio region [padLeft, padLeft+n0).
        var frameValid = [Int32]()
        frameValid.reserveCapacity(numFrames)
        let validLo = padLeft
        let validHi = padLeft + n0
        for f in 0 ..< numFrames {
            let end = f * hopLength + frameSizeForUnfold - 1
            frameValid.append((end >= validLo && end < validHi) ? 1 : 0)
        }
        let maskArr = MLXArray(frameValid)
        logMel = logMel * expandedDimensions(maskArr.asType(logMel.dtype), axis: -1)

        return (
            expandedDimensions(logMel, axis: 0),  // [1, T, 128]
            expandedDimensions(maskArr.asType(.bool), axis: 0)  // [1, T]
        )
    }

    /// Number of audio soft tokens the encoder yields for `validFrames` valid mel
    /// frames: the SSCP's two stride-2 stages applied to a valid prefix collapse to
    /// `ceil(ceil(V/2)/2)`. Expanding the audio placeholder to exactly this many
    /// tokens keeps the prompt's audio-token count equal to the scattered encoder
    /// frames (the tower zeroes/removes padding frames).
    public static func audioTokenCount(validFrames: Int) -> Int {
        let step1 = (validFrames + 1) / 2
        return (step1 + 1) / 2
    }

    /// HTK mel filterbank, `norm=None`, row-major `[numFreqBins, numMel]`.
    static func melFilterBank(
        numFreqBins: Int, numMel: Int, minFrequency: Float, maxFrequency: Float, sampleRate: Int
    ) -> [Float] {
        func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

        let melMin = hzToMel(Double(minFrequency))
        let melMax = hzToMel(Double(maxFrequency))
        var freqPoints = [Double](repeating: 0, count: numMel + 2)
        for i in 0 ..< numMel + 2 {
            let mel = melMin + (melMax - melMin) * Double(i) / Double(numMel + 1)
            freqPoints[i] = melToHz(mel)
        }

        // Frequency of each rfft bin: k · sampleRate / (2·(numFreqBins − 1)).
        let binWidth = Double(sampleRate) / (2.0 * Double(numFreqBins - 1))

        var bank = [Float](repeating: 0, count: numFreqBins * numMel)
        for i in 0 ..< numMel {
            let lower = freqPoints[i]
            let center = freqPoints[i + 1]
            let upper = freqPoints[i + 2]
            let ldenom = max(center - lower, 1e-10)
            let rdenom = max(upper - center, 1e-10)
            for k in 0 ..< numFreqBins {
                let freq = Double(k) * binWidth
                let rising = (freq - lower) / ldenom
                let falling = (upper - freq) / rdenom
                bank[k * numMel + i] = Float(max(0.0, min(rising, falling)))
            }
        }
        return bank
    }
}
