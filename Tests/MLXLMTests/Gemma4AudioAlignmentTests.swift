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

    // Fixture provenance: gemma4_mel_alignment.json was generated from the
    // Python reference extractor BEFORE the Swift extractor adopted semicausal
    // padding (frameLength/2 leading zeros). The padding prepends exactly one
    // hop of zeros, so Swift frame f+1 covers the same samples as reference
    // frame f, and the padded run yields one extra leading frame
    // (50 = 49 + 1). The fixture was generated with the overdriven
    // (1024-point) FFT configuration. Both tests below account for this.

    @Test
    func melSpectrogramShapeAndStats() throws {
        let ref = try loadFixture("gemma4_mel_alignment")
        let refShape = try #require(ref["mel_shape"] as? [Int])  // [49, 128]
        let refStats = try #require(ref["mel_stats"] as? [String: Double])

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
        // Semicausal padding adds exactly one frame over the reference run.
        #expect(
            shape[0] == refShape[0] + 1,
            "Frame count mismatch: \(shape[0]) vs ref \(refShape[0]) + 1 padding frame")
        #expect(shape[1] == refShape[1], "Mel bins mismatch: \(shape[1]) vs \(refShape[1])")

        // Global stats of a steady sine are insensitive to the one-frame shift,
        // so they compare directly (the extra frame is onset/zeros, which the
        // tolerance absorbs).
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

        // Match the fixture's generation configuration (overdriven FFT).
        let extractor = Gemma4AudioFeatureExtractor(fftOverdrive: true)
        let (mel, _) = extractor.extract(audio: sineAudio)
        eval(mel)
        #expect(mel.dim(0) > refFrames.count, "Not enough frames to compare against the fixture")

        // Compare every reference value (not just a corner of the matrix) so a
        // regression in windowing, FFT scaling, or frame alignment cannot slip
        // through. Swift frame f+1 corresponds to reference frame f (see
        // provenance note above).
        //
        // Bins below 16 are excluded: the rounded-bin filter bank collapses
        // sub-bin-width low-frequency filters to zero (= mel floor), where the
        // fixture's generation-era bank kept small leakage energies — a known,
        // deliberate divergence (see gemma4MelFilterBank), confined to
        // near-floor bins. The signal-carrying bins must agree within 1.0
        // log-mel, which absorbs the vDSP-vs-numpy FFT difference.
        let lowBinCutoff = 16
        var maxDiff: Float = 0
        var maxFrame = 0
        var maxBin = 0
        for f in 0 ..< min(refFrames.count, mel.dim(0) - 1) {
            let swiftFrame = mel[f + 1].asArray(Float.self)
            for b in lowBinCutoff ..< min(refFrames[f].count, swiftFrame.count) {
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
            "Mel values too far from Python: maxDiff=\(maxDiff) at ref frame \(maxFrame) bin \(maxBin)")
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

        // Filter coverage: the reference extractor builds triangles in
        // *rounded FFT-bin space* (see gemma4MelFilterBank). At this
        // configuration (128 mel filters over 0–8000 Hz, 512-point FFT →
        // 31.25 Hz bin spacing) low-frequency mel filters are narrower than
        // one FFT bin, so their rounded edges intermittently collapse and
        // those columns are legitimately all-zero — a property of the
        // reference filter bank, not a porting bug. The collapsed set is
        // fully determined by the configuration; pin it exactly so any change
        // to the bank construction is caught.
        let colSums = bank.sum(axis: 0)
        eval(colSums)
        let colSumsArray = colSums.asArray(Float.self)
        let zeroColumns = colSumsArray.enumerated().filter { $0.element == 0 }.map { $0.offset }
        #expect(
            zeroColumns == [1, 3, 5, 7, 9, 11, 14, 16, 19, 22, 26, 30, 39],
            "Collapsed mel filter set changed: \(zeroColumns)")
    }
}
