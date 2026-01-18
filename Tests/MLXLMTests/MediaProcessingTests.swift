// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

public class MediaProcesingTests: XCTestCase {

    func testResize() {
        // resampleBicubic should produce an image with the desired dimensions
        let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
        inputFilter.setValue(CIColor.red, forKey: "inputColor")
        let input = inputFilter.outputImage!.cropped(
            to: CGRect(x: 0, y: 0, width: 1536, height: 1106))

        let target = CGSize(width: 1540, height: 1120)
        let output = MediaProcessing.resampleBicubic(input, to: target)

        XCTAssertEqual(output.extent.size, target)
    }

    func testVideoFileAsSimpleProcessedSequence() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        // We know video is exactly 5 seconds long, expect 5 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)

        XCTAssert(frames.frames.count == 5)
    }

    func testVideoFileValidationThisShouldFail() async throws {
        guard let fileURL = Bundle.module.url(forResource: "audio_only", withExtension: "mov")
        else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        do {
            let _ = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)
        } catch {
            XCTAssertEqual(error as? VLMError, VLMError.noVideoTrackFound)
        }
    }

    func testVideoFileAsProcessedSequence() async throws {
        // Bogus preprocessing values
        func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
            image
                .toSRGB()
                .resampled(to: resizedSize, method: .bicubic)
                .normalized(mean: (0.1, 0.2, 0.3), std: (0.4, 0.5, 0.6))
        }

        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        // We know video is exactly 5 seconds long, expect 10 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
            frame in
            let image = preprocess(image: frame.frame, resizedSize: .init(width: 224, height: 224))

            return VideoFrame.init(frame: image, timeStamp: frame.timeStamp)
        }

        XCTAssert(frames.frames.count == 10)
        XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
    }

    func testVideoFramesAsProcessedSequence() async throws {
        // a function to make a set of frames from images
        func imageWithColor(_ color: CIColor) -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(color, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(
                to: CGRect(x: 0, y: 0, width: 1920, height: 1080))
        }

        let colors: [CIColor] = [
            .red, .green, .blue, .cyan, .magenta, .yellow, .white, .black, .gray, .clear,
        ]

        let seconds = 5
        let framerate = 30
        var rawFrames: [VideoFrame] = []

        for i in 0 ..< (seconds * framerate) {
            let image = imageWithColor(colors.randomElement()!)
            let timeStamp: CMTime = .init(value: Int64(i), timescale: Int32(framerate))
            rawFrames.append(VideoFrame(frame: image, timeStamp: timeStamp))
        }

        // Bogus preprocessing values
        func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
            image
                .toSRGB()
                .resampled(to: resizedSize, method: .bicubic)
                .normalized(mean: (0.1, 0.2, 0.3), std: (0.4, 0.5, 0.6))
        }

        let video = UserInput.Video.frames(rawFrames)

        // We know video is exactly 5 seconds long, expect 10 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
            frame in
            let image = preprocess(image: frame.frame, resizedSize: .init(width: 224, height: 224))

            return VideoFrame.init(frame: image, timeStamp: frame.timeStamp)
        }

        XCTAssert(frames.frames.count == 10)
        XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
    }
}
