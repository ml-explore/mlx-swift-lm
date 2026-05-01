// Copyright © 2026 Apple Inc.
//
// Verifies the #208 fix: small images must not be upscaled to
// `maxProcessingImageSize` before tiling. Previously a 512×384 input was
// upscaled to 2048×1536 and chopped into a 3×4 grid (12 tiles + 1 global =
// 13 patches, ~1140 prompt tokens, ~9× slower). After the fix the same
// input flows through as a single 512×384 frame.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SmolVLM2TilingTests: XCTestCase {

    private func makeProcessor() throws -> SmolVLMProcessor {
        let json = """
            {
                "processor_class": "SmolVLMProcessor",
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "image_seq_len": 64,
                "size": { "longest_edge": 2048 },
                "max_image_size": { "longest_edge": 512 },
                "video_sampling": { "fps": 1, "max_frames": 12, "video_size": { "longest_edge": 512 } }
            }
            """
        let config = try JSONDecoder().decode(
            SmolVLMProcessorConfiguration.self,
            from: Data(json.utf8))
        return SmolVLMProcessor(config, tokenizer: TestTokenizer())
    }

    private func makeImage(width: CGFloat, height: CGFloat) -> CIImage {
        let filter = CIFilter(name: "CIConstantColorGenerator")!
        filter.setValue(CIColor(red: 0.5, green: 0.5, blue: 0.5), forKey: "inputColor")
        return filter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    /// 512×384 must produce exactly one tile. Pre-fix, this image was
    /// upscaled to 2048×1536 and produced a 3×4 grid (12 tiles).
    func testSmallImageNotUpscaled() throws {
        let processor = try makeProcessor()
        let image = makeImage(width: 512, height: 384)
        let result = processor.tiles(from: image)
        XCTAssertEqual(
            result.tiles.count, 1,
            "512x384 input should not be upscaled into a tile grid; got \(result.tiles.count) tiles"
        )
        XCTAssertEqual(result.rows, 1)
        XCTAssertEqual(result.cols, 1)
    }

    /// 1024×768 — under maxProcessingImageSize (2048). Tiles based on actual
    /// size (no upscale): longest edge 1024 → 2 tiles wide × ceil(768/512) =
    /// 2 tiles tall = 4 tiles.
    func testMediumImageUsesActualSize() throws {
        let processor = try makeProcessor()
        let image = makeImage(width: 1024, height: 768)
        let result = processor.tiles(from: image)
        XCTAssertEqual(
            result.tiles.count, 4,
            "1024x768 should tile based on actual size, got \(result.tiles.count) tiles")
        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 2)
    }

    /// 4096×3072 — larger than maxProcessingImageSize (2048). The downscaling
    /// branch is unaffected by the fix; the original 3×4 grid still applies.
    func testLargeImageStillDownscales() throws {
        let processor = try makeProcessor()
        let image = makeImage(width: 4096, height: 3072)
        let result = processor.tiles(from: image)
        XCTAssertEqual(
            result.tiles.count, 12,
            "4096x3072 should still downscale + tile, got \(result.tiles.count) tiles")
        XCTAssertEqual(result.rows, 3)
        XCTAssertEqual(result.cols, 4)
    }
}
