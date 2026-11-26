// Copyright © 2026 Apple Inc.
//
// Pillow's bicubic `Image.resize`, which d1-3B's `cap_pixels` runs before the LFM2-VL processor.
// Ports `precompute_coeffs`, `normalize_coeffs_8bpc` and `bicubic_filter` from
// `src/libImaging/Resample.c`, and the pass order of `Image.resize`, from Pillow
// (https://github.com/python-pillow/Pillow). Copyright © 1997-2011 by Secret Labs AB,
// © 1995-2011 by Fredrik Lundh and contributors, © 2010 by Jeffrey 'Alex' Clark and
// contributors. MIT-CMU; see NOTICE_MLXDecisions.md.

import Foundation
import MLXVLM

extension ResampleTaps {
    /// Pillow `precompute_coeffs` and `normalize_coeffs_8bpc` with the bicubic filter: weights
    /// in double, then int32 with 22 fractional bits.
    init(pillowBicubicInSize inSize: Int, outSize: Int) {
        precondition(inSize >= 1 && outSize >= 1, "resize from \(inSize) to \(outSize)")
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        let kernelSize = Int(support.rounded(.up)) * 2 + 1
        let inverseScale = 1 / filterScale
        let precision = 22
        let one = Double(1 << precision)

        var start = [Int](repeating: 0, count: outSize)
        var count = [Int](repeating: 0, count: outSize)
        var weights = [Int32](repeating: 0, count: outSize * kernelSize)
        var k = [Double](repeating: 0, count: kernelSize)
        for i in 0 ..< outSize {
            let center = (Double(i) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let n = min(Int(center + support + 0.5), inSize) - xmin
            var total = 0.0
            for x in 0 ..< n {
                k[x] = pillowBicubic((Double(x + xmin) - center + 0.5) * inverseScale)
                total += k[x]
            }
            for x in 0 ..< n {
                let w = total != 0 ? k[x] / total : k[x]
                weights[i * kernelSize + x] = Int32(w < 0 ? -0.5 + w * one : 0.5 + w * one)
            }
            start[i] = xmin
            count[i] = n
        }
        self.init(
            kernelSize: kernelSize, shift: precision, start: start, count: count, weights: weights)
    }
}

/// Pillow `bicubic_filter`: the Keys cubic with a = -0.5, in Pillow's arithmetic.
private func pillowBicubic(_ x: Double) -> Double {
    let a = -0.5
    let x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0
}

extension RGB8Image {
    /// PIL `Image.resize((width, height), Image.Resampling.BICUBIC)` on the RGB image
    /// `oriented(orientation)`, without `box` or `reducing_gap`. Large images run on the GPU
    /// with the same bytes, as `resized` does.
    func resizedPILBicubic(
        width newWidth: Int, height newHeight: Int, orientation: Int = 1
    ) -> RGB8Image {
        precondition(newWidth >= 1 && newHeight >= 1, "height and width must be > 0")
        return resampled(
            pillowBicubicPasses(width: newWidth, height: newHeight, orientation: orientation),
            orientation: orientation)
    }

    /// The passes of `resizedPILBicubic` from the upright size: width, then height, each
    /// skipped when its size already matches. Pillow sums in a C `int`, as the GPU kernel does.
    func pillowBicubicPasses(
        width newWidth: Int, height newHeight: Int, orientation: Int = 1
    ) -> [ResamplePass] {
        let transposed = (5 ... 8).contains(orientation)
        let (width, height) = transposed ? (self.height, self.width) : (self.width, self.height)
        var passes = [ResamplePass]()
        if newWidth != width {
            passes.append(
                ResamplePass(
                    horizontal: true,
                    taps: ResampleTaps(pillowBicubicInSize: width, outSize: newWidth)))
        }
        if newHeight != height {
            passes.append(
                ResamplePass(
                    horizontal: false,
                    taps: ResampleTaps(pillowBicubicInSize: height, outSize: newHeight)))
        }
        // `Image.resize` shrinks very tall images in two calls, height first.
        if height > width * 100 && newHeight < height {
            passes.reverse()
        }
        return passes
    }
}
