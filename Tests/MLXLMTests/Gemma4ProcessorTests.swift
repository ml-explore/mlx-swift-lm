// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

public class Gemma4ProcessorTests: XCTestCase {

    /// Create a solid-color CIImage for testing.
    private func createTestImage(width: Int, height: Int) -> CIImage {
        let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
        inputFilter.setValue(CIColor.red, forKey: "inputColor")
        return inputFilter.outputImage!.cropped(
            to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    /// Decode a Gemma4ProcessorConfiguration from minimal JSON with defaults.
    private func createConfig() throws -> Gemma4ProcessorConfiguration {
        let json = """
            {
                "processor_class": "Gemma4Processor",
                "patch_size": 16,
                "max_soft_tokens": 280,
                "pooling_kernel_size": 3,
                "do_normalize": false,
                "do_rescale": true,
                "do_resize": true
            }
            """
        return try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: json.data(using: .utf8)!)
    }

    func testConfigSideMult() throws {
        let config = try createConfig()
        // sideMult = poolingKernelSize (3) * patchSize (16) = 48
        XCTAssertEqual(config.sideMult, 48, "sideMult should be 48")
    }

    func testConfigMaxPatches() throws {
        let config = try createConfig()
        // maxPatches = maxSoftTokens (280) * poolingKernelSize^2 (9) = 2520
        XCTAssertEqual(config.maxPatches, 2520, "maxPatches should be 2520")
    }

    func testPreprocessOutputDimensions() throws {
        let config = try createConfig()
        let tokenizer = TestTokenizer()
        let processor = Gemma4Processor(config, tokenizer: tokenizer)

        // Create a 100x100 test image
        let image = createTestImage(width: 100, height: 100)

        // Preprocess
        let (array, thw) = try processor.preprocess(images: [image], processing: nil)

        // Output dimensions should be divisible by 48 (sideMult)
        let h = thw.h
        let w = thw.w
        XCTAssertEqual(h % 48, 0, "Height \(h) should be divisible by 48")
        XCTAssertEqual(w % 48, 0, "Width \(w) should be divisible by 48")

        // Array shape should be [1, 3, H, W]
        XCTAssertEqual(array.dim(0), 1, "Batch dim should be 1")
        XCTAssertEqual(array.dim(1), 3, "Channel dim should be 3")
        XCTAssertEqual(array.dim(2), h, "Height should match THW")
        XCTAssertEqual(array.dim(3), w, "Width should match THW")
    }

    func testPreprocessLargeImage() throws {
        let config = try createConfig()
        let tokenizer = TestTokenizer()
        let processor = Gemma4Processor(config, tokenizer: tokenizer)

        // Create a large test image (1920x1080)
        let image = createTestImage(width: 1920, height: 1080)

        // Preprocess
        let (array, thw) = try processor.preprocess(images: [image], processing: nil)

        // Output dimensions should be divisible by 48
        let h = thw.h
        let w = thw.w
        XCTAssertEqual(h % 48, 0, "Height \(h) should be divisible by 48")
        XCTAssertEqual(w % 48, 0, "Width \(w) should be divisible by 48")

        // Array shape check
        XCTAssertEqual(array.dim(0), 1)
        XCTAssertEqual(array.dim(1), 3)
        XCTAssertEqual(array.dim(2), h)
        XCTAssertEqual(array.dim(3), w)
    }

    func testPreprocessSmallImage() throws {
        let config = try createConfig()
        let tokenizer = TestTokenizer()
        let processor = Gemma4Processor(config, tokenizer: tokenizer)

        // Create a very small test image (30x30) - smaller than sideMult
        let image = createTestImage(width: 30, height: 30)

        // Preprocess
        let (array, thw) = try processor.preprocess(images: [image], processing: nil)

        // Even small images should produce dimensions divisible by 48
        let h = thw.h
        let w = thw.w
        XCTAssertEqual(h % 48, 0, "Height \(h) should be divisible by 48")
        XCTAssertEqual(w % 48, 0, "Width \(w) should be divisible by 48")

        // Minimum dimension should be at least sideMult (48)
        XCTAssertGreaterThanOrEqual(h, 48, "Height should be at least 48")
        XCTAssertGreaterThanOrEqual(w, 48, "Width should be at least 48")
    }

    func testPreprocessAspectRatioPreservation() throws {
        let config = try createConfig()
        let tokenizer = TestTokenizer()
        let processor = Gemma4Processor(config, tokenizer: tokenizer)

        // Create a wide image
        let wideImage = createTestImage(width: 800, height: 200)
        let (_, wideThw) = try processor.preprocess(images: [wideImage], processing: nil)

        // Create a tall image
        let tallImage = createTestImage(width: 200, height: 800)
        let (_, tallThw) = try processor.preprocess(images: [tallImage], processing: nil)

        // Wide image should have width >= height
        XCTAssertGreaterThanOrEqual(
            wideThw.w, wideThw.h,
            "Wide image should preserve width >= height")

        // Tall image should have height >= width
        XCTAssertGreaterThanOrEqual(
            tallThw.h, tallThw.w,
            "Tall image should preserve height >= width")
    }
}
