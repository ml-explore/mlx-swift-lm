//
//  Gemma4AudioAlignmentTest.swift
//  MLXVLMTests
//
//  Verify Swift mel spectrogram output matches Python reference data.
//

import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Gemma4 Audio Alignment")
struct Gemma4AudioAlignmentTest {

    /// Load JSON fixture from Tests/MLXVLMTests/Fixtures/
    private func loadFixture(_ name: String) throws -> [String: Any] {
        // #filePath points to source file location
        let sourceDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let fixtureURL = sourceDir.appendingPathComponent("Fixtures/\(name)")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Fixture not found: \(fixtureURL.path)"])
        }
        let data = try Data(contentsOf: fixtureURL)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    @Test
    func melSpectrogramShapeAndStats() throws {
        // Load Python reference: 0.5s 440Hz sine wave
        let ref = try loadFixture("gemma4_mel_alignment.json")
        let refShape = ref["mel_shape"] as! [Int]  // [49, 128]
        let refStats = ref["mel_stats"] as! [String: Double]

        // Generate same input in Swift
        var audio = [Float](repeating: 0, count: 8000)
        for i in 0 ..< 8000 {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0)
        }

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

        let (mel, mask) = extractor.extract(audio: audio)
        eval(mel, mask)

        let shape = mel.shape
        print("Swift mel shape: \(shape), Python ref: \(refShape)")

        // Shape should match
        #expect(shape[0] == refShape[0], "Frame count mismatch: \(shape[0]) vs \(refShape[0])")
        #expect(shape[1] == refShape[1], "Mel bins mismatch: \(shape[1]) vs \(refShape[1])")

        // Stats should be close
        let swiftMean = Double(mel.mean().item(Float.self))
        let swiftStd = Double(MLX.sqrt(mel.variance()).item(Float.self))
        let refMean = refStats["mean"]!
        let refStd = refStats["std"]!

        print("Swift mean=\(swiftMean), Python mean=\(refMean)")
        print("Swift std=\(swiftStd), Python std=\(refStd)")

        #expect(abs(swiftMean - refMean) < 0.5, "Mean too far: \(swiftMean) vs \(refMean)")
        #expect(abs(swiftStd - refStd) < 0.5, "Std too far: \(swiftStd) vs \(refStd)")
    }

    @Test
    func melSpectrogramFrameValues() throws {
        // Compare first few frames against Python values
        let ref = try loadFixture("gemma4_mel_alignment.json")
        let refFrames = ref["mel_frames_0_to_9"] as! [[Double]]  // 10 frames × 128 bins

        var audio = [Float](repeating: 0, count: 8000)
        for i in 0 ..< 8000 {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0)
        }

        let extractor = Gemma4AudioFeatureExtractor()
        let (mel, _) = extractor.extract(audio: audio)
        eval(mel)

        // Compare first frame
        let firstFrame = mel[0]
        eval(firstFrame)

        var maxDiff: Float = 0
        for i in 0 ..< min(10, refFrames[0].count) {
            let swiftVal = firstFrame[i].item(Float.self)
            let pyVal = Float(refFrames[0][i])
            let diff = abs(swiftVal - pyVal)
            if diff > maxDiff { maxDiff = diff }
        }

        print("Max diff in first frame (first 10 bins): \(maxDiff)")
        // Allow some tolerance due to FFT implementation differences
        #expect(maxDiff < 1.0, "First frame values too far from Python: maxDiff=\(maxDiff)")
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
