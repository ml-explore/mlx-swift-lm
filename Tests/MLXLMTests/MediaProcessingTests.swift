// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

public class MediaProcesingTests: XCTestCase {

    func testResize() {
        Device.withDefaultDevice(.cpu) {
            // resampleBicubic should produce an image with the desired dimensions
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(CIColor.red, forKey: "inputColor")
            let input = inputFilter.outputImage!.cropped(
                to: CGRect(x: 0, y: 0, width: 1536, height: 1106))

            let target = CGSize(width: 1540, height: 1120)
            let output = MediaProcessing.resampleBicubic(input, to: target)

            XCTAssertEqual(output.extent.size, target)
        }
    }

    func testVideoFileAsSimpleProcessedSequence() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        try await Device.withDefaultDevice(.cpu) {
            // We know video is exactly 5 seconds long, expect 5 samples
            let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)
            XCTAssert(frames.frames.count == 5)
        }
    }

    func testVideoFileValidationThisShouldFail() async throws {
        guard let fileURL = Bundle.module.url(forResource: "audio_only", withExtension: "mov")
        else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        await Device.withDefaultDevice(.cpu) {
            do {
                let _ = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)
            } catch {
                XCTAssertEqual(error as? VLMError, VLMError.noVideoTrackFound)
            }
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

        try await Device.withDefaultDevice(.cpu) {
            // We know video is exactly 5 seconds long, expect 10 samples
            let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
                frame in
                let image = preprocess(
                    image: try frame.image.asCIImage(), resizedSize: .init(width: 224, height: 224))

                return VideoFrame.init(image: .ciImage(image), timeStamp: frame.timeStamp)
            }

            XCTAssert(frames.frames.count == 10)
            XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
        }
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
            rawFrames.append(VideoFrame(image: .ciImage(image), timeStamp: timeStamp))
        }

        // Bogus preprocessing values
        func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
            image
                .toSRGB()
                .resampled(to: resizedSize, method: .bicubic)
                .normalized(mean: (0.1, 0.2, 0.3), std: (0.4, 0.5, 0.6))
        }

        let video = UserInput.Video.frames(rawFrames)

        try await Device.withDefaultDevice(.cpu) {
            // We know video is exactly 5 seconds long, expect 10 samples
            let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
                frame in
                let image = preprocess(
                    image: try frame.image.asCIImage(), resizedSize: .init(width: 224, height: 224))

                return VideoFrame.init(image: .ciImage(image), timeStamp: frame.timeStamp)
            }

            XCTAssert(frames.frames.count == 10)
            XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
        }
    }

    // MARK: - VideoProcessing Option Tests

    func testVideoFileAsProcessedSequenceWithTargetFrames() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)
        let videoProcessing = UserInput.VideoProcessing(targetFrames: 3)

        try await Device.withDefaultDevice(.cpu) {
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                processing: videoProcessing
            )

            XCTAssertEqual(frames.frames.count, 3)
            XCTAssertEqual(frames.timestamps.count, 3)
        }
    }

    func testVideoFileAsProcessedSequenceWithTargetFPS() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)
        let videoProcessing = UserInput.VideoProcessing(targetFramesPerSecond: 3.0)

        try await Device.withDefaultDevice(.cpu) {
            // 5 seconds @ 3 fps = 15 frames
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                processing: videoProcessing
            )

            XCTAssertEqual(frames.frames.count, 15)
            XCTAssertEqual(frames.timestamps.count, 15)
        }
    }

    func testVideoFramesAsProcessedSequenceWithTargetFrames() async throws {
        func testImage() -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(CIColor.blue, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        }

        let framerate = 30
        var rawFrames: [VideoFrame] = []
        for i in 0 ..< 150 {
            let timeStamp = CMTime(value: Int64(i), timescale: Int32(framerate))
            rawFrames.append(VideoFrame(image: .ciImage(testImage()), timeStamp: timeStamp))
        }

        let video = UserInput.Video.frames(rawFrames)
        try await Device.withDefaultDevice(.cpu) {
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                processing: .init(targetFrames: 6)
            )

            XCTAssertEqual(frames.frames.count, 6)
            XCTAssertEqual(frames.timestamps.count, 6)
        }
    }

    func testVideoFramesAsProcessedSequenceWithTargetFramesPerSecond() async throws {
        func testImage() -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(CIColor.green, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        }

        let framerate = 30
        var rawFrames: [VideoFrame] = []
        for i in 0 ..< 150 {
            let timeStamp = CMTime(value: Int64(i), timescale: Int32(framerate))
            rawFrames.append(VideoFrame(image: .ciImage(testImage()), timeStamp: timeStamp))
        }

        let video = UserInput.Video.frames(rawFrames)
        try await Device.withDefaultDevice(.cpu) {
            // 5s duration * 4.0 fps = 20 frames
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                processing: .init(targetFramesPerSecond: 4.0)
            )

            XCTAssertEqual(frames.frames.count, 20)
            XCTAssertEqual(frames.timestamps.count, 20)
        }
    }

    func testVideoProcessingWithUserInputProcessingWrapper() async throws {
        func testImage() -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(CIColor.red, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        }

        let rawFrames = (0 ..< 60).map {
            VideoFrame(
                image: .ciImage(testImage()), timeStamp: CMTime(value: Int64($0), timescale: 30))
        }
        let video = UserInput.Video.frames(rawFrames)

        let userInputProcessing = UserInput.Processing(video: .init(targetFrames: 4))
        try await Device.withDefaultDevice(.cpu) {
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                userProcessing: userInputProcessing
            )

            XCTAssertEqual(frames.frames.count, 4)
        }
    }

    func testVideoProcessingAmbiguousInitInference() async throws {
        func testImage() -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(CIColor.red, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        }

        let rawFrames = (0 ..< 60).map {
            VideoFrame(
                image: .ciImage(testImage()), timeStamp: CMTime(value: Int64($0), timescale: 30))
        }
        let video = UserInput.Video.frames(rawFrames)

        try await Device.withDefaultDevice(.cpu) {
            // Calling processing: .init() should cleanly and unambiguously resolve to VideoProcessing
            let frames = try await MediaProcessing.asProcessedSequence(
                video,
                processing: .init()
            )

            XCTAssertGreaterThan(frames.frames.count, 0)
        }
    }

    func testSampleTimesHelper() {
        let timescale: Int32 = 600
        let duration5s = CMTime(seconds: 5.0, preferredTimescale: timescale)

        // 1. targetFrames spans across endpoints: [0, 2.5s, 5.0s]
        let times1 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(targetFrames: 3),
            targetFPS: nil,
            maxFrames: 50
        )
        XCTAssertEqual(times1.map { $0.seconds }, [0.0, 2.5, 5.0])

        // 2. targetFramesPerSecond derives the count and spans the full duration
        // 5s @ 2 FPS -> 10 uniformly distributed frames including both endpoints
        let times2 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(targetFramesPerSecond: 2.0),
            targetFPS: nil,
            maxFrames: 50
        )
        XCTAssertEqual(times2.count, 10)
        XCTAssertEqual(
            times2.map(\.value), [0, 333, 667, 1000, 1333, 1667, 2000, 2333, 2667, 3000])

        // 3. Fallback targetFPS closure preserves the previous full-duration sampling behavior
        let times3 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(),
            targetFPS: { _ in 1.0 },
            maxFrames: 50
        )
        XCTAssertEqual(times3.map { $0.seconds }, [0.0, 1.25, 2.5, 3.75, 5.0])

        // 4. Clamping with maxFrames still covers the entire video
        let times4 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(targetFramesPerSecond: 2.0),
            targetFPS: nil,
            maxFrames: 4
        )
        XCTAssertEqual(times4.map(\.value), [0, 1000, 2000, 3000])

        // 5. Clamping with totalAvailableFrames
        let times5 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(targetFrames: 10),
            targetFPS: nil,
            maxFrames: 50,
            totalAvailableFrames: 2
        )
        XCTAssertEqual(times5.count, 2)
        XCTAssertEqual(times5.map { $0.seconds }, [0.0, 5.0])

        // 6. Clamping FPS sampling to the available frames still covers the entire video
        let times6 = MediaProcessing.sampleTimes(
            duration: duration5s,
            processing: .init(targetFramesPerSecond: 4.0),
            targetFPS: nil,
            maxFrames: 50,
            totalAvailableFrames: 2
        )
        XCTAssertEqual(times6.map { $0.seconds }, [0.0, 5.0])

        // 7. Minimum 1 frame guarantee for 0-duration or empty requests
        let duration0s = CMTime(seconds: 0.0, preferredTimescale: timescale)
        let times7 = MediaProcessing.sampleTimes(
            duration: duration0s,
            processing: .init(targetFrames: 0),
            targetFPS: nil,
            maxFrames: 10
        )
        XCTAssertEqual(times7.count, 1)
        XCTAssertEqual(times7[0].value, 0)
    }
}
