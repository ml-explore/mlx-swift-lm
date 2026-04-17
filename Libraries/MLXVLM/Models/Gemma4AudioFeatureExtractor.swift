//
//  Gemma4AudioFeatureExtractor.swift
//  MLXVLM
//
//  Audio feature extractor for Gemma 4 — extracts log-mel spectrograms
//  from raw audio waveforms using USM preprocessing pipeline.
//
//  Ported from: mlx_vlm/models/gemma4/audio_feature_extractor.py
//

import Accelerate
import Foundation
import MLX

// MARK: - Mel Filter Bank

/// Create a mel filter bank matrix [numFrequencyBins, numMelFilters] using HTK scale.
public func gemma4MelFilterBank(
    numFrequencyBins: Int,
    numMelFilters: Int,
    minFrequency: Float,
    maxFrequency: Float,
    samplingRate: Int
) -> MLXArray {
    func hzToMel(_ freq: Float) -> Float {
        2595.0 * log10(1.0 + freq / 700.0)
    }
    func melToHz(_ mel: Float) -> Float {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let melMin = hzToMel(minFrequency)
    let melMax = hzToMel(maxFrequency)

    // Linearly spaced mel points
    var melPoints = [Float](repeating: 0, count: numMelFilters + 2)
    for i in 0 ..< (numMelFilters + 2) {
        melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(numMelFilters + 1)
    }
    let freqPoints = melPoints.map { melToHz($0) }

    // All frequency bins
    var allFreqs = [Float](repeating: 0, count: numFrequencyBins)
    let freqStep = Float(samplingRate) / Float(2 * (numFrequencyBins - 1))
    for i in 0 ..< numFrequencyBins {
        allFreqs[i] = Float(i) * freqStep
    }

    // Build triangular filter bank
    var filterBank = [Float](repeating: 0, count: numFrequencyBins * numMelFilters)
    for i in 0 ..< numMelFilters {
        let lower = freqPoints[i]
        let center = freqPoints[i + 1]
        let upper = freqPoints[i + 2]

        for j in 0 ..< numFrequencyBins {
            let rising = (allFreqs[j] - lower) / max(center - lower, 1e-10)
            let falling = (upper - allFreqs[j]) / max(upper - center, 1e-10)
            filterBank[j * numMelFilters + i] = max(0, min(rising, falling))
        }
    }

    return MLXArray(filterBank, [numFrequencyBins, numMelFilters])
}

// MARK: - Feature Extractor

/// Gemma4 audio feature extractor — converts raw waveform to log-mel spectrogram.
public struct Gemma4AudioFeatureExtractor {
    public let featureSize: Int
    public let samplingRate: Int
    public let frameLength: Int
    public let hopLength: Int
    public let fftLength: Int
    public let melFloor: Float
    public let preemphasis: Float
    public let preemphasisHTKFlavor: Bool
    public let inputScaleFactor: Float

    /// Hanning window [frameLength]
    private let window: [Float]
    /// Mel filter bank [fftLength/2+1, featureSize]
    private let melFilters: MLXArray

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        frameLengthMs: Float = 20.0,
        hopLengthMs: Float = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        preemphasis: Float = 0.0,
        preemphasisHTKFlavor: Bool = true,
        fftOverdrive: Bool = true,
        inputScaleFactor: Float = 1.0,
        melFloor: Float = 1e-3
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.preemphasis = preemphasis
        self.preemphasisHTKFlavor = preemphasisHTKFlavor
        self.inputScaleFactor = inputScaleFactor
        self.melFloor = melFloor

        self.frameLength = Int(round(Float(samplingRate) * frameLengthMs / 1000.0))
        self.hopLength = Int(round(Float(samplingRate) * hopLengthMs / 1000.0))

        var fftLen = 1
        while fftLen < frameLength { fftLen *= 2 }
        if fftOverdrive { fftLen *= 2 }
        self.fftLength = fftLen

        // Hanning window (non-zero at endpoints, matching Python)
        let arg = Float.pi * 2.0 / Float(frameLength)
        var win = [Float](repeating: 0, count: frameLength)
        for i in 0 ..< frameLength {
            win[i] = 0.5 - 0.5 * cos(arg * (Float(i) + 0.5))
        }
        self.window = win

        self.melFilters = gemma4MelFilterBank(
            numFrequencyBins: fftLen / 2 + 1,
            numMelFilters: featureSize,
            minFrequency: minFrequency,
            maxFrequency: maxFrequency,
            samplingRate: samplingRate
        )
    }

    /// Extract log-mel spectrogram from raw audio samples.
    /// - Parameters:
    ///   - audio: Raw waveform [numSamples] as Float array
    ///   - maxLength: Maximum number of samples (default 480000 = 30s at 16kHz)
    /// - Returns: (melSpectrogram: [frames, featureSize], mask: [frames])
    public func extract(audio: [Float], maxLength: Int = 480_000) -> (MLXArray, MLXArray) {
        var waveform = audio
        if waveform.count > maxLength {
            waveform = Array(waveform.prefix(maxLength))
        }

        // Pad to multiple of 128
        let padTarget = ((waveform.count + 127) / 128) * 128
        var mask = [Float](repeating: 1.0, count: padTarget)
        if waveform.count < padTarget {
            mask.replaceSubrange(
                waveform.count ..< padTarget,
                with: repeatElement(0.0, count: padTarget - waveform.count))
            waveform.append(contentsOf: repeatElement(0.0, count: padTarget - waveform.count))
        }

        // Scale
        if inputScaleFactor != 1.0 {
            for i in 0 ..< waveform.count {
                waveform[i] *= inputScaleFactor
            }
        }

        // Frame extraction (unfold)
        let frameSizeForUnfold = frameLength + 1
        let numFrames = (waveform.count - frameSizeForUnfold) / hopLength + 1
        guard numFrames > 0 else {
            return (MLXArray.zeros([0, featureSize]), MLXArray.zeros([0]))
        }

        // Extract frames with preemphasis
        var frames = [Float](repeating: 0, count: numFrames * frameLength)
        for f in 0 ..< numFrames {
            let start = f * hopLength
            if preemphasis > 0 && preemphasisHTKFlavor {
                frames[f * frameLength] = waveform[start] * (1.0 - preemphasis)
                for j in 1 ..< frameLength {
                    frames[f * frameLength + j] =
                        waveform[start + j] - preemphasis * waveform[start + j - 1]
                }
            } else {
                for j in 0 ..< frameLength {
                    frames[f * frameLength + j] = waveform[start + j]
                }
            }
        }

        // Apply window
        for f in 0 ..< numFrames {
            for j in 0 ..< frameLength {
                frames[f * frameLength + j] *= window[j]
            }
        }

        // RFFT using Accelerate
        let melSpec = computeMelSpectrogram(frames: frames, numFrames: numFrames)

        // Build frame-level mask
        var frameMask = [Float](repeating: 0, count: numFrames)
        for f in 0 ..< numFrames {
            let idx = f * hopLength
            if idx < mask.count {
                frameMask[f] = mask[idx]
            }
        }

        return (melSpec, MLXArray(frameMask))
    }

    /// Compute mel spectrogram from windowed frames using Accelerate FFT.
    private func computeMelSpectrogram(frames: [Float], numFrames: Int) -> MLXArray {
        let halfFFT = fftLength / 2

        // Use vDSP for FFT
        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            // Fallback: zero output
            return MLXArray.zeros([numFrames, featureSize])
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var allMagnitudes = [Float](repeating: 0, count: numFrames * (halfFFT + 1))

        for f in 0 ..< numFrames {
            // Zero-pad frame to fftLength
            var paddedFrame = [Float](repeating: 0, count: fftLength)
            for j in 0 ..< frameLength {
                paddedFrame[j] = frames[f * frameLength + j]
            }

            // Split complex
            var realPart = [Float](repeating: 0, count: halfFFT)
            var imagPart = [Float](repeating: 0, count: halfFFT)

            // Pack into split complex (even/odd interleave)
            for i in 0 ..< halfFFT {
                realPart[i] = paddedFrame[2 * i]
                imagPart[i] = paddedFrame[2 * i + 1]
            }

            var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

            // Extract magnitudes
            // vDSP_fft_zrip output is scaled by 2x compared to standard DFT.
            // Divide by 2 to match numpy.fft.rfft normalization.
            let scale: Float = 0.5
            // DC component
            allMagnitudes[f * (halfFFT + 1)] = abs(splitComplex.realp[0]) * scale
            // Nyquist
            allMagnitudes[f * (halfFFT + 1) + halfFFT] = abs(splitComplex.imagp[0]) * scale
            // Other bins
            for i in 1 ..< halfFFT {
                let re = splitComplex.realp[i]
                let im = splitComplex.imagp[i]
                allMagnitudes[f * (halfFFT + 1) + i] = sqrt(re * re + im * im) * scale
            }
        }

        // Apply mel filter bank: [numFrames, halfFFT+1] @ [halfFFT+1, featureSize]
        let magnitudeArray = MLXArray(allMagnitudes, [numFrames, halfFFT + 1])
        let melSpec = matmul(magnitudeArray, melFilters)

        // Log mel
        let logMelSpec = log(maximum(melSpec, MLXArray(melFloor)))

        return logMelSpec.asType(.float32)
    }
}
