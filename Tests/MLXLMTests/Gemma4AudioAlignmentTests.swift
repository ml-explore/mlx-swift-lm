// Copyright © 2026 Apple Inc.

// Verify the Swift Gemma 4 mel spectrogram output matches Python reference data
// (fixture generated from the reference extractor; see Fixtures/gemma4_mel_alignment.json).

import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Gemma4 Audio Alignment")
struct Gemma4AudioAlignmentTests {

    private func loadFixture(_ name: String) throws -> [String: Any] {
        let fixtureURL = try #require(
            Bundle.module.url(forResource: name, withExtension: "json"),
            "Fixture \(name).json missing from test bundle")
        let data = try Data(contentsOf: fixtureURL)
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "Fixture \(name).json is not a JSON object")
    }

    /// 0.5 s, 440 Hz sine at 16 kHz — the input the Python reference used.
    private var sineAudio: [Float] {
        var audio = [Float](repeating: 0, count: 8000)
        for i in 0 ..< 8000 {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0)
        }
        return audio
    }

    @Test
    func melSpectrogramShapeAndStats() throws {
        let ref = try loadFixture("gemma4_mel_alignment")
        let refShape = try #require(ref["mel_shape"] as? [Int])  // [49, 128]
        let refStats = try #require(ref["mel_stats"] as? [String: Double])

        // The fixture's shape/stats reference was generated with the overdriven
        // (1024-point) FFT, so mirror that configuration here; the per-frame
        // value test below covers the production default (512-point) FFT.
        let extractor = Gemma4AudioFeatureExtractor(
            featureSize: 128,
            samplingRate: 16000,
            frameLengthMs: 20.0,
            hopLengthMs: 10.0,
            minFrequency: 0.0,
            maxFrequency: 8000.0,
            preemphasis: 0.0,
            preemphasisHTKFlavor: true,
            fftOverdrive: true,
            inputScaleFactor: 1.0,
            melFloor: 1e-3
        )

        let (mel, mask) = extractor.extract(audio: sineAudio)
        eval(mel, mask)

        let shape = mel.shape
        #expect(shape[0] == refShape[0], "Frame count mismatch: \(shape[0]) vs \(refShape[0])")
        #expect(shape[1] == refShape[1], "Mel bins mismatch: \(shape[1]) vs \(refShape[1])")

        let swiftMean = Double(mel.mean().item(Float.self))
        let swiftStd = Double(MLX.sqrt(mel.variance()).item(Float.self))
        let refMean = try #require(refStats["mean"])
        let refStd = try #require(refStats["std"])

        #expect(abs(swiftMean - refMean) < 0.5, "Mean too far: \(swiftMean) vs \(refMean)")
        #expect(abs(swiftStd - refStd) < 0.5, "Std too far: \(swiftStd) vs \(refStd)")
    }

    @Test
    func melSpectrogramFrameValues() throws {
        let ref = try loadFixture("gemma4_mel_alignment")
        // 10 frames × 128 bins of reference values.
        let refFrames = try #require(ref["mel_frames_0_to_9"] as? [[Double]])

        let extractor = Gemma4AudioFeatureExtractor()
        let (mel, _) = extractor.extract(audio: sineAudio)
        eval(mel)

        // Compare every reference value, not just a corner of the matrix, so a
        // regression in windowing, filter-bank edges, or frame alignment past
        // frame 0 cannot slip through. The 1.0 log-mel tolerance absorbs the
        // FFT-implementation difference between vDSP and the Python reference.
        var maxDiff: Float = 0
        var maxFrame = 0
        var maxBin = 0
        for f in 0 ..< min(refFrames.count, mel.dim(0)) {
            let swiftFrame = mel[f].asArray(Float.self)
            for b in 0 ..< min(refFrames[f].count, swiftFrame.count) {
                let diff = abs(swiftFrame[b] - Float(refFrames[f][b]))
                if diff > maxDiff {
                    maxDiff = diff
                    maxFrame = f
                    maxBin = b
                }
            }
        }

        #expect(
            maxDiff < 1.0,
            "Mel values too far from Python: maxDiff=\(maxDiff) at frame \(maxFrame) bin \(maxBin)")
    }

    @Test
    func melFilterBankShape() {
        // 512 FFT → 257 frequency bins, 128 mel filters
        let bank = gemma4MelFilterBank(
            numFrequencyBins: 257,
            numMelFilters: 128,
            minFrequency: 0,
            maxFrequency: 8000,
            samplingRate: 16000
        )
        eval(bank)

        #expect(bank.shape == [257, 128])

        // Filter bank should be non-negative
        let minVal = bank.min().item(Float.self)
        #expect(minVal >= 0, "Filter bank has negative values: \(minVal)")

        // Filter coverage: at this configuration (128 mel filters over 0–8000 Hz
        // with a 512-point FFT → 31.25 Hz bin spacing), the lowest triangular
        // filter has upper edge ≈27.9 Hz < bin spacing, so no FFT bin lands
        // inside it and that filter column is legitimately all-zero. This
        // matches HTK's unnormalized mel filter bank definition (and what
        // librosa produces with `htk=True, norm=None`). Assert that at most
        // one filter is empty and that all filters from index 1 onward carry
        // non-zero coefficients.
        let colSums = bank.sum(axis: 0)
        eval(colSums)
        let colSumsArray = colSums.asArray(Float.self)
        let zeroCount = colSumsArray.filter { $0 == 0 }.count
        #expect(zeroCount <= 1, "Too many all-zero mel filters: \(zeroCount)")
        for i in 1 ..< colSumsArray.count {
            #expect(colSumsArray[i] > 0, "Filter \(i) is unexpectedly all-zero")
        }
    }
}
