import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import XCTest

final class AudioInputTests: XCTestCase {
    func testBigEndianAudioLoadsAsNativeFloats() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString + ".aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let rate = 16_000.0
        let expected = (0 ..< 1600).map { Float(0.2 * sin(2 * .pi * 440 * Double($0) / rate)) }
        do {
            let file = try AVAudioFile(
                forWriting: url,
                settings: [
                    AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate,
                    AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                    AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: true,
                ])
            let buffer = try XCTUnwrap(
                AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600))
            buffer.frameLength = 1600
            let channel = try XCTUnwrap(buffer.floatChannelData)[0]
            for (index, sample) in expected.enumerated() { channel[index] = sample }
            try file.write(from: buffer)
        }
        var processing = UserInput.AudioProcessing()
        processing.sampleRate = rate
        let loaded = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        XCTAssertEqual(loaded.shape, [expected.count])
        XCTAssertTrue(all(isFinite(loaded)).item(Bool.self))
        XCTAssertLessThan(abs(loaded - MLXArray(expected)).max().item(Float.self), 1.0 / 32768)
    }
}
