// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Guards the multi-image pixel packing used by the Gemma4 (non-unified) vision
/// path. Each image is aspect-ratio-preserved to its own `(height, width)`, so
/// the images cannot be stacked into one dense batch: `Gemma4Processor.prepare`
/// flattens every image to 1D and concatenates them, and `Gemma4.getInputEmbeddings`
/// slices that buffer back apart using each image's `(height, width)` from
/// `frames`. This verifies the flatten → concat → slice → reshape round-trip
/// recovers every image exactly for a mix of different sizes and aspect ratios.
struct Gemma4MultiImagePackingTests {
    private static let channels = 3

    // (height, width) — deliberately different sizes and aspect ratios, each a
    // multiple of `patch_size * pooling_kernel_size` (48) like real targets.
    private static let sizes: [(h: Int, w: Int)] = [
        (48, 96),  // wide
        (96, 48),  // tall
        (144, 144),  // square, larger
    ]

    @Test("Flatten/concat then slice/reshape recovers each image exactly")
    func packingRoundTrips() {
        // Pure data movement (flatten/concat/slice/reshape); scope to the CPU
        // backend so it runs under `swift test` without the Metal library.
        Device.withDefaultDevice(.cpu) {
            // One distinct dense image per size, mirroring the (1, C, H, W)
            // layout that MediaProcessing.asMLXArray produces. Globally unique
            // values catch any ordering or off-by-one error in the pack/unpack.
            var originals: [MLXArray] = []
            var base = 0
            for size in Self.sizes {
                let count = Self.channels * size.h * size.w
                let image = MLXArray(base ..< (base + count))
                    .reshaped(1, Self.channels, size.h, size.w)
                originals.append(image)
                base += count
            }

            // Processor side (Gemma4Processor.prepare): flatten each to 1D, concat.
            let packed = concatenated(originals.map { $0.flattened() })

            let expectedLength = Self.sizes.reduce(0) { $0 + Self.channels * $1.h * $1.w }
            #expect(packed.ndim == 1)
            #expect(packed.dim(0) == expectedLength)

            // Model side (Gemma4.getInputEmbeddings): slice by count, reshape back.
            var offset = 0
            for (index, size) in Self.sizes.enumerated() {
                let count = Self.channels * size.h * size.w
                let recovered = packed[offset ..< (offset + count)]
                    .reshaped(1, Self.channels, size.h, size.w)
                offset += count

                #expect(recovered.shape == [1, Self.channels, size.h, size.w])
                #expect((recovered .== originals[index]).all().item(Bool.self))
            }

            // The model consumes exactly the whole buffer, in order.
            #expect(offset == expectedLength)
        }
    }
}
