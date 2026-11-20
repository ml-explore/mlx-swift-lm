import Foundation
import MLX

// Reference: LiquidAI/LFM3-Omni-2B, revision 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67.

/// Converts mono audio to float32 log-mel features shaped [128, frames].
struct LFM3OmniAudioFeatures {
    private let fftSize = 400
    private let hopSize = 160
    private let melBins = 128

    func callAsFunction(_ samples: MLXArray, sampleRate: Int = 16_000) throws -> MLXArray {
        guard sampleRate == 16_000 else {
            throw VLMError.processing("LFM3-Omni requires audio resampled to 16 kHz")
        }
        guard samples.ndim == 1, samples.size > 0,
            samples.dtype.isFloatingPoint, all(isFinite(samples)).item(Bool.self)
        else {
            throw VLMError.processing("LFM3-Omni requires finite, nonempty mono audio at 16 kHz")
        }
        let frameCount = 1 + max(0, samples.size - fftSize + hopSize - 1) / hopSize
        let paddedSize = (frameCount - 1) * hopSize + fftSize
        let padded = concatenated([
            samples.asType(.float32), MLXArray.zeros([paddedSize - samples.size]),
        ])
        let frames = asStrided(padded, [frameCount, fftSize], strides: [hopSize, 1])
        let window = MLXArray(
            (0 ..< fftSize).map {
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize)))
            })
        let spectrum = abs(rfft(frames * window))
        let power = spectrum * spectrum
        let mel = matmul(power, filters())
        let features = ((maximum(log10(mel), -12) + 4) / 4).transposed()
        guard all(isFinite(features)).item(Bool.self) else {
            throw VLMError.processing("LFM3-Omni audio overflows the log-mel frontend")
        }
        return features
    }

    func stream(
        _ chunk: MLXArray, state: LFM3OmniFeatureStreamingState,
        finalize: Bool = false, sampleRate: Int = 16_000
    ) throws -> MLXArray {
        guard sampleRate == 16000, !state.finished, chunk.ndim == 1,
            chunk.dtype.isFloatingPoint, all(isFinite(chunk)).item(Bool.self)
        else { throw VLMError.processing("Invalid LFM3-Omni audio stream or sample chunk") }
        let samples =
            state.samples.map { concatenated([$0, chunk.asType(.float32)]) }
            ?? chunk.asType(.float32)
        let total = state.sampleCount + chunk.size
        let count: Int
        if finalize {
            let frames = total == 0 ? 0 : 1 + max(0, total - fftSize + hopSize - 1) / hopSize
            count = frames - state.frameCount
        } else {
            count = samples.size < fftSize ? 0 : 1 + (samples.size - fftSize) / hopSize
        }
        let features: MLXArray
        if count > 0 {
            let size = (count - 1) * hopSize + fftSize
            features = try self(samples[..<min(size, samples.size)])
        } else {
            features = MLXArray.zeros([melBins, 0])
        }
        state.samples = finalize ? nil : contiguous(samples[(count * hopSize)...])
        state.sampleCount = total
        state.frameCount += count
        state.finished = finalize
        return features
    }

    private func filters() -> MLXArray {
        let frequencyStep = 200.0 / 3.0
        let logStep = log(6.4) / 27.0
        let maxMel = 15 + log(8000.0 / 1000) / logStep
        let edges = (0 ..< melBins + 2).map { index -> Double in
            let mel = maxMel * Double(index) / Double(melBins + 1)
            return mel < 15 ? mel * frequencyStep : 1000 * exp((mel - 15) * logStep)
        }
        var weights = [Float]()
        weights.reserveCapacity((fftSize / 2 + 1) * melBins)
        for bin in 0 ... fftSize / 2 {
            let frequency = Double(bin) * 16000 / Double(fftSize)
            for band in 0 ..< melBins {
                let rising = (frequency - edges[band]) / (edges[band + 1] - edges[band])
                let falling = (edges[band + 2] - frequency) / (edges[band + 2] - edges[band + 1])
                let normalization = 2 / (edges[band + 2] - edges[band])
                weights.append(Float(max(0, min(rising, falling)) * normalization))
            }
        }
        return MLXArray(weights).reshaped(fftSize / 2 + 1, melBins)
    }
}

final class LFM3OmniFeatureStreamingState {
    fileprivate var samples: MLXArray?
    fileprivate var sampleCount = 0
    fileprivate var frameCount = 0
    fileprivate var finished = false
}
