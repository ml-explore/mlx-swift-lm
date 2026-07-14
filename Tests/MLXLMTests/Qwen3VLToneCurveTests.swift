// Copyright © 2026 Apple Inc.

import CoreGraphics
import CoreImage
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// The Qwen3VL image path must hand the model gamma-encoded sRGB values —
/// the byte values the HF reference preprocesses — not CoreImage's
/// linear-light working-space values. Without the tone-curve step, near-black
/// content (e.g. dark-theme text at luminance 23/255 on a 6/255 background)
/// reaches the ViT with ~12x less contrast than the reference and becomes
/// unreadable, while bright content is barely affected.
final class Qwen3VLToneCurveTests: XCTestCase {

    private func makeProcessor() throws -> Qwen3VLProcessor {
        let json = """
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 16,
              "temporal_patch_size": 2,
              "image_processor_type": "Qwen2VLImageProcessor"
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: Data(json.utf8))
        return Qwen3VLProcessor(config, tokenizer: TestTokenizer())
    }

    /// A 64x64 sRGB image: background gray level 6 with a centered 32x32
    /// block at gray level 23 — the luminances of a dark-theme screenshot's
    /// background and a faint headline.
    private func faintBlockImage() throws -> CIImage {
        let side = 64
        let block = 16 ..< 48
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< side {
            for x in 0 ..< side {
                let level: UInt8 = block.contains(x) && block.contains(y) ? 23 : 6
                let offset = (y * side + x) * 4
                pixels[offset] = level
                pixels[offset + 1] = level
                pixels[offset + 2] = level
                pixels[offset + 3] = 255
            }
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        let cgImage = try XCTUnwrap(
            CGImage(
                width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: side * 4,
                space: try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB)),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent))
        return CIImage(cgImage: cgImage)
    }

    func testDarkContentKeepsGammaEncodedContrast() throws {
        let processor = try makeProcessor()
        let (pixels, _) = try processor.preprocess(
            images: [faintBlockImage()], processing: nil)

        // Patchify rearranges layout but preserves values, so global extrema
        // are layout-independent. Reference (HF byte-value preprocessing with
        // mean/std 0.5): block = (23/255 - 0.5)/0.5 ≈ -0.820, background =
        // (6/255 - 0.5)/0.5 ≈ -0.953. The linear-space failure mode produced
        // ≈ -0.984 / -0.996 — far outside these tolerances.
        let maxValue = pixels.max().item(Float.self)
        let minValue = pixels.min().item(Float.self)
        XCTAssertEqual(maxValue, -0.820, accuracy: 0.05)
        XCTAssertEqual(minValue, -0.953, accuracy: 0.05)
    }
}
