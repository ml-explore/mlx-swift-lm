import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

#if canImport(AVFoundation)
final class UserInputAudioTests: XCTestCase {
    func testAudioURLMaterializesFloatSamples() async throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "audio_only", withExtension: "mov"))

        let samples = try await UserInput.Audio.url(url).asMLXArray()
        eval(samples)

        XCTAssertEqual(samples.ndim, 1)
        XCTAssertGreaterThan(samples.size, 0)
        XCTAssertEqual(samples.dtype, .float32)
    }

    func testArrayAudioPreservesIdentityShape() async throws {
        let input = MLXArray([Float](repeating: 0.25, count: 64))

        let output = try await UserInput.Audio.array(input).asMLXArray()

        XCTAssertEqual(output.shape, [64])
        XCTAssertEqual(output.asArray(Float.self), input.asArray(Float.self))
    }
}
#endif
