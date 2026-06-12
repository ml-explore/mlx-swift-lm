// Copyright © 2026 Apple Inc.

// Audio feature extractor for Gemma 4 — extracts log-mel spectrograms from raw
// audio waveforms using the USM preprocessing pipeline.
// Ported from: mlx_vlm/models/gemma4/audio_feature_extractor.py

import Accelerate
import Foundation
import MLX

// MARK: - Mel Filter Bank

/// Create a mel filter bank matrix [numFrequencyBins, numMelFilters] using HTK scale.
func gemma4MelFilterBank(
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
    // Map mel points to rounded FFT bin indices and build the triangles in
    // bin-index space (matching the reference Gemma 4 extractor). Building them
    // in continuous-Hz space yields different triangle widths/areas on the
    // nonlinear mel scale, scaling the mel energies (~1.6x) so the audio tower
    // can't interpret them.
    let fftLen = (numFrequencyBins - 1) * 2
    let binPoints = melPoints.map { mel -> Int in
        let hz = melToHz(mel)
        return Int((hz * Float(fftLen) / Float(samplingRate)).rounded())
    }

    var filterBank = [Float](repeating: 0, count: numFrequencyBins * numMelFilters)
    for i in 0 ..< numMelFilters {
        let left = binPoints[i]
        let center = binPoints[i + 1]
        let right = binPoints[i + 2]

        if center > left {
            for k in left ..< min(center, numFrequencyBins) where k >= 0 {
                filterBank[k * numMelFilters + i] = Float(k - left) / Float(center - left)
            }
        }
        if right > center {
            for k in center ..< min(right, numFrequencyBins) where k >= 0 {
                filterBank[k * numMelFilters + i] = Float(right - k) / Float(right - center)
            }
        }
    }

    return MLXArray(filterBank, [numFrequencyBins, numMelFilters])
}

// MARK: - Feature Extractor

/// Gemma4 audio feature extractor — converts raw waveform to log-mel spectrogram.
struct Gemma4AudioFeatureExtractor {
    let featureSize: Int
    let samplingRate: Int
    let frameLength: Int
    let hopLength: Int
    let fftLength: Int
    let melFloor: Float
    let preemphasis: Float
    let preemphasisHTKFlavor: Bool
    let inputScaleFactor: Float

    /// Hanning window [frameLength]
    private let window: [Float]
    /// Mel filter bank [fftLength/2+1, featureSize]
    private let melFilters: MLXArray

    init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        frameLengthMs: Float = 20.0,
        hopLengthMs: Float = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        preemphasis: Float = 0.0,
        preemphasisHTKFlavor: Bool = true,
        // The reference extractor does not double the FFT length ("overdrive");
        // doubling to 1024 yields the wrong spectrum/mel and the audio tower
        // can't interpret it.
        fftOverdrive: Bool = false,
        inputScaleFactor: Float = 1.0,
        melFloor: Float = 1e-3,
        // Direct sample-count overrides, used when the checkpoint's
        // processor_config.json specifies hop_length / fft_length explicitly.
        hopLength: Int? = nil,
        fftLength: Int? = nil
    ) {
        // Only the HTK pre-emphasis flavor is implemented; the defaults disable
        // pre-emphasis entirely (preemphasis = 0), matching the reference
        // Gemma 4 extractor.
        precondition(
            preemphasis == 0 || preemphasisHTKFlavor,
            "non-HTK pre-emphasis is not implemented")
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.preemphasis = preemphasis
        self.preemphasisHTKFlavor = preemphasisHTKFlavor
        self.inputScaleFactor = inputScaleFactor
        self.melFloor = melFloor

        self.frameLength = Int(round(Float(samplingRate) * frameLengthMs / 1000.0))
        self.hopLength = hopLength ?? Int(round(Float(samplingRate) * hopLengthMs / 1000.0))

        if let fftLength {
            self.fftLength = fftLength
        } else {
            var fftLen = 1
            while fftLen < self.frameLength { fftLen *= 2 }
            if fftOverdrive { fftLen *= 2 }
            self.fftLength = fftLen
        }
        let fftLen = self.fftLength
        // The radix-2 vDSP FFT below requires a power-of-two length.
        precondition(
            fftLen > 0 && fftLen & (fftLen - 1) == 0,
            "fftLength must be a power of two")

        // Periodic Hann window (torch.hann_window(periodic=true) → divisor N, not
        // N-1; zero at i=0), matching the reference extractor.
        var win = [Float](repeating: 0, count: frameLength)
        for i in 0 ..< frameLength {
            win[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(frameLength))
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
    func extract(audio: [Float], maxLength: Int = 480_000) -> (MLXArray, MLXArray) {
        var waveform = audio
        if waveform.count > maxLength {
            waveform = Array(waveform.prefix(maxLength))
        }

        // Semicausal padding: prepend frameLength/2 zeros before framing (ref
        // Google feature_extraction_gemma4). Without it the frame alignment is
        // shifted relative to what the audio tower was trained on.
        waveform = [Float](repeating: 0, count: frameLength / 2) + waveform

        // No multiple-of-128 padding: the reference extractor frames the
        // semicausal-padded waveform directly (all frames valid). The extra
        // 128-pad added trailing zero frames that shifted the token count.
        // For this single-utterance path the mask is therefore always all-ones;
        // it exists so batch padding can be represented without API changes.
        let mask = [Float](repeating: 1.0, count: waveform.count)

        // Scale
        if inputScaleFactor != 1.0 {
            for i in 0 ..< waveform.count {
                waveform[i] *= inputScaleFactor
            }
        }

        // Frame extraction (unfold) — frame size = frameLength (ref uses N, not N+1)
        let frameSizeForUnfold = frameLength
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

            realPart.withUnsafeMutableBufferPointer { realBuffer in
                imagPart.withUnsafeMutableBufferPointer { imagBuffer in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuffer.baseAddress!,
                        imagp: imagBuffer.baseAddress!
                    )
                    vDSP_fft_zrip(
                        fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // Extract magnitudes. vDSP_fft_zrip output is scaled by 2x
                    // compared to standard DFT, so divide by 2 to match
                    // numpy.fft.rfft normalization.
                    let scale: Float = 0.5
                    allMagnitudes[f * (halfFFT + 1)] = abs(splitComplex.realp[0]) * scale
                    allMagnitudes[f * (halfFFT + 1) + halfFFT] =
                        abs(splitComplex.imagp[0]) * scale
                    for i in 1 ..< halfFFT {
                        let re = splitComplex.realp[i]
                        let im = splitComplex.imagp[i]
                        allMagnitudes[f * (halfFFT + 1) + i] = sqrt(re * re + im * im) * scale
                    }
                }
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
