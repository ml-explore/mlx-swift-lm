// Copyright © 2026 Apple Inc.
//
// torchvision's antialiased uint8 resize, as transformers' `TorchvisionBackend.resize` runs it,
// and the float32 weights of `F.interpolate(mode="bilinear", antialias=True)`. The LFM2-VL image
// processor and the SigLIP2 position embeddings use them.
//
// Core Image and `bicubicInterpolate` resize in float and differ from torchvision by up to 33
// levels when shrinking. This runs on the CPU, or for large images in a Metal kernel with the
// same integer arithmetic. `resampled(_:orientation:)` also runs passes with other taps.
//
// Both port ATen's `aten/src/ATen/native/cpu/UpSampleKernel.cpp`. PyTorch distributes it under
// this licence, from PyTorch's LICENSE file (the Caffe2 part does not apply):
//
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import Foundation
import MLX

/// The `PIL.Image.Resampling` ids that torchvision's antialiased resize supports, as image
/// processor configs store them in `resample`.
enum ResampleFilter: Int, Sendable, CaseIterable {
    case bilinear = 2
    case bicubic = 3

    /// Half of ATen's `interp_size`.
    fileprivate var support: Double {
        switch self {
        case .bilinear: 1
        case .bicubic: 2
        }
    }

    /// ATen `aa_filter`: the triangle, or the Keys cubic with a = -0.5.
    fileprivate func weight(_ x: Double) -> Double {
        let x = abs(x)
        switch self {
        case .bilinear:
            return x < 1 ? 1 - x : 0
        case .bicubic:
            let a = -0.5
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
            return 0
        }
    }
}

/// One axis of a separable uint8 resize in fixed point: output sample `i` is
/// `(2^(shift - 1) + Σ input[start[i] + j] * weights[i * kernelSize + j]) >> shift` over
/// `j < count[i]`, clamped to 0...255.
struct ResampleTaps: Equatable, Sendable {
    var kernelSize: Int
    var shift: Int
    var start: [Int]
    var count: [Int]
    var weights: [Int32]

    /// ATen `_compute_index_ranges_int16_weights` with antialiasing: weights in double, then
    /// int16 at the largest precision below 22 bits that keeps the largest weight in range.
    init(inSize: Int, outSize: Int, filter: ResampleFilter) {
        precondition(inSize >= 1 && outSize >= 1, "resize from \(inSize) to \(outSize)")
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1 ? filter.support * scale : filter.support
        let kernelSize = Int(support.rounded(.up)) * 2 + 1
        let inverseScale = scale >= 1 ? 1 / scale : 1

        var start = [Int](repeating: 0, count: outSize)
        var count = [Int](repeating: 0, count: outSize)
        var normalized = [Double](repeating: 0, count: outSize * kernelSize)
        var maxWeight = 0.0
        for i in 0 ..< outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            let n = min(max(min(Int(center + support + 0.5), inSize) - xmin, 0), kernelSize)
            var total = 0.0
            for x in 0 ..< n {
                let w = filter.weight((Double(x + xmin) - center + 0.5) * inverseScale)
                normalized[i * kernelSize + x] = w
                total += w
            }
            if total != 0 {
                for x in 0 ..< n {
                    normalized[i * kernelSize + x] /= total
                    maxWeight = max(maxWeight, normalized[i * kernelSize + x])
                }
            }
            start[i] = xmin
            count[i] = n
        }
        var precision = 0
        while precision < 22, Int(0.5 + maxWeight * Double(1 << (precision + 1))) < 1 << 15 {
            precision += 1
        }
        let one = Double(1 << precision)
        self.kernelSize = kernelSize
        self.shift = precision
        self.start = start
        self.count = count
        self.weights = normalized.map {
            let v = $0 * one
            return Int32(Int16(truncatingIfNeeded: Int(v < 0 ? v - 0.5 : v + 0.5)))
        }
    }
}

/// One pass of a separable uint8 resize: the axis it resamples and its taps.
struct ResamplePass: Equatable, Sendable {
    var horizontal: Bool
    var taps: ResampleTaps
}

extension RGB8Image {
    /// torchvision `resize(uint8 image, antialias=True)` of `oriented(orientation)`: the width
    /// pass, then the height pass, each rounded to uint8. Large images run on the GPU, same bytes.
    func resized(
        width newWidth: Int, height newHeight: Int, filter: ResampleFilter, orientation: Int = 1
    ) -> RGB8Image {
        precondition(newWidth >= 1 && newHeight >= 1, "height and width must be > 0")
        return resampled(
            resizePasses(
                width: newWidth, height: newHeight, filter: filter, orientation: orientation),
            orientation: orientation)
    }

    /// The passes of `resized` from the upright size: width, then height, each skipped when its
    /// size already matches.
    func resizePasses(
        width newWidth: Int, height newHeight: Int, filter: ResampleFilter, orientation: Int = 1
    ) -> [ResamplePass] {
        let transposed = (5 ... 8).contains(orientation)
        let (uprightWidth, uprightHeight) = transposed ? (height, width) : (width, height)
        var passes = [ResamplePass]()
        if newWidth != uprightWidth {
            passes.append(
                ResamplePass(
                    horizontal: true,
                    taps: ResampleTaps(inSize: uprightWidth, outSize: newWidth, filter: filter)))
        }
        if newHeight != uprightHeight {
            passes.append(
                ResamplePass(
                    horizontal: false,
                    taps: ResampleTaps(inSize: uprightHeight, outSize: newHeight, filter: filter)))
        }
        return passes
    }

    /// `oriented(orientation)` after `passes` in order, each rounded to uint8. It runs on the
    /// GPU or the CPU as `resized` does, with the same bytes.
    func resampled(_ passes: [ResamplePass], orientation: Int = 1) -> RGB8Image {
        #if canImport(Metal)
        if prefersGPU(passes, orientation: orientation) {
            return gpuResampled(passes, orientation: orientation)
        }
        #endif
        return cpuResampled(passes, orientation: orientation)
    }

    /// `resampled` on the CPU. This is the reference for `gpuResampled`.
    func cpuResampled(_ passes: [ResamplePass], orientation: Int = 1) -> RGB8Image {
        var image = oriented(orientation)
        for pass in passes {
            image =
                pass.horizontal ? image.horizontalPass(pass.taps) : image.verticalPass(pass.taps)
        }
        return image
    }

    /// Resamples every row of the image to `taps.start.count` columns.
    private func horizontalPass(_ taps: ResampleTaps) -> RGB8Image {
        let outWidth = taps.start.count
        let rowLength = width * 3
        var out = [UInt8](repeating: 0, count: outWidth * height * 3)
        bytes.withUnsafeBufferPointer { source in
            out.withUnsafeMutableBufferPointer { out in
                taps.withPointers { start, count, weights in
                    let (src, dst) = (source.baseAddress!, out.baseAddress!)
                    let (kernelSize, shift) = (taps.kernelSize, taps.shift)
                    let half = 1 << (shift - 1)
                    forEachChunk(height, work: outWidth * height * kernelSize) { range in
                        for y in range {
                            let row = src + y * rowLength
                            var o = y * outWidth * 3
                            var x = 0
                            while x < outWidth {
                                var p = start[x] * 3
                                var t = x * kernelSize
                                let end = t + count[x]
                                var (s0, s1, s2) = (half, half, half)
                                while t < end {
                                    let w = Int(weights[t])
                                    s0 += Int(row[p]) * w
                                    s1 += Int(row[p + 1]) * w
                                    s2 += Int(row[p + 2]) * w
                                    p += 3
                                    t += 1
                                }
                                dst[o] = clampedByte(s0 >> shift)
                                dst[o + 1] = clampedByte(s1 >> shift)
                                dst[o + 2] = clampedByte(s2 >> shift)
                                o += 3
                                x += 1
                            }
                        }
                    }
                }
            }
        }
        return RGB8Image(width: outWidth, height: height, bytes: out)
    }

    /// Resamples the columns of the image to `taps.start.count` rows. Each output row sums
    /// whole input rows, so the reads are sequential.
    private func verticalPass(_ taps: ResampleTaps) -> RGB8Image {
        let outHeight = taps.start.count
        let rowLength = width * 3
        var out = [UInt8](repeating: 0, count: rowLength * outHeight)
        bytes.withUnsafeBufferPointer { source in
            out.withUnsafeMutableBufferPointer { out in
                taps.withPointers { start, count, weights in
                    let (src, dst) = (source.baseAddress!, out.baseAddress!)
                    let (kernelSize, shift) = (taps.kernelSize, taps.shift)
                    let half = 1 << (shift - 1)
                    let work = rowLength * outHeight * kernelSize / 3
                    forEachChunk(outHeight, work: work) { range in
                        let sums = UnsafeMutablePointer<Int>.allocate(capacity: rowLength)
                        defer { sums.deallocate() }
                        for y in range {
                            var e = 0
                            while e < rowLength {
                                sums[e] = half
                                e += 1
                            }
                            for j in 0 ..< count[y] {
                                let row = src + (start[y] + j) * rowLength
                                let w = Int(weights[y * kernelSize + j])
                                e = 0
                                while e < rowLength {
                                    sums[e] += Int(row[e]) * w
                                    e += 1
                                }
                            }
                            let o = dst + y * rowLength
                            e = 0
                            while e < rowLength {
                                o[e] = clampedByte(sums[e] >> shift)
                                e += 1
                            }
                        }
                    }
                }
            }
        }
        return RGB8Image(width: width, height: outHeight, bytes: out)
    }
}

// MARK: - GPU

#if canImport(Metal)
extension RGB8Image {
    /// Whether `resampled` runs on the GPU: the default device is the GPU, every image the
    /// passes read or write has fewer than `Int32.max` bytes, and the CPU passes would do at
    /// least `gpuMinimumWork` multiply-adds per channel.
    func prefersGPU(_ passes: [ResamplePass], orientation: Int) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu else { return false }
        let transposed = (5 ... 8).contains(orientation)
        var (width, height) = transposed ? (self.height, self.width) : (self.width, self.height)
        var largest = bytes.count
        var work = 0
        for pass in passes {
            let taps = pass.taps.count.reduce(0, +)
            if pass.horizontal {
                work += height * taps
                width = pass.taps.start.count
            } else {
                work += width * taps
                height = pass.taps.start.count
            }
            largest = max(largest, width * height * 3)
        }
        // The kernel indexes bytes with int32.
        return largest < Int(Int32.max) && work >= gpuMinimumWork
    }

    /// `cpuResampled` in one Metal dispatch per pass. The first pass reads the stored pixels
    /// through `orientation`, so the image is never turned at full size. The kernel sums in
    /// int32, which the taps must not overflow.
    func gpuResampled(_ passes: [ResamplePass], orientation: Int = 1) -> RGB8Image {
        let orientation = (1 ... 8).contains(orientation) ? orientation : 1
        guard !passes.isEmpty else { return oriented(orientation) }
        var (width, height) =
            orientation >= 5 ? (self.height, self.width) : (self.width, self.height)
        var pixels = mappedArray(bytes)
        var steps = orientationSteps(orientation)
        for pass in passes {
            if pass.horizontal {
                width = pass.taps.start.count
            } else {
                height = pass.taps.start.count
            }
            pixels = resamplePass(
                pixels, steps: steps, horizontal: pass.horizontal, width: width, height: height,
                taps: pass.taps)
            steps = (0, 1, width)
        }
        return RGB8Image(width: width, height: height, bytes: pixels.asArray(UInt8.self))
    }
}

/// `bytes` copied to fresh pages that the GPU reads in place. The pages are unmapped when MLX
/// frees the array, so a full-size image does not stay in MLX's buffer cache. `onFree` runs then.
func mappedArray(_ bytes: [UInt8], onFree: @escaping () -> Void = {}) -> MLXArray {
    let page = Int(getpagesize())
    let length = (bytes.count + page - 1) / page * page
    guard length > 0,
        let memory = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0),
        memory != UnsafeMutableRawPointer(bitPattern: -1)
    else {
        onFree()
        return MLXArray(bytes)
    }
    bytes.withUnsafeBytes { memory.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    // If Metal cannot wrap the pages, MLX copies them and calls this at once.
    return MLXArray(rawPointer: memory, [length], dtype: .uint8) {
        munmap(memory, length)
        onFree()
    }
}

/// Multiply-adds per channel from which `resampled` runs on the GPU. There, on an M3 Ultra,
/// the CPU passes take about 0.8 ms and the GPU about 0.25 ms. The first GPU call in a
/// process also compiles the kernel.
private let gpuMinimumWork = 400_000

/// One pass of `resampleKernel` into a `width` x `height` image. `steps` are the input pixel
/// at output (0, 0) and the input steps per output column and row.
private func resamplePass(
    _ input: MLXArray, steps: (start: Int, columnStep: Int, rowStep: Int), horizontal: Bool,
    width: Int, height: Int, taps: ResampleTaps
) -> MLXArray {
    // Neighbouring threads take neighbouring output rows when that reads adjacent pixels.
    let transposed = abs(steps.columnStep) != 1
    var packed = (taps.start + taps.count).map { Int32($0) } + taps.weights
    // An input under 8 elements goes to constant memory, which compiles another kernel.
    packed += [Int32](repeating: 0, count: max(0, 8 - packed.count))
    let params = [
        steps.start, steps.columnStep, steps.rowStep, width, height, taps.kernelSize,
        taps.shift, (horizontal ? 1 : 0) | (transposed ? 2 : 0),
    ].map { Int32($0) }
    return resampleKernel(
        [input, MLXArray(packed), MLXArray(params)],
        grid: transposed ? (height, width, 1) : (width, height, 1),
        threadGroup: (32, 8, 1), outputShapes: [[width * height * 3]], outputDTypes: [.uint8],
        stream: .gpu)[0]
}

/// One output pixel per thread, in int32 as the CPU passes: start at half of `1 << shift`,
/// add `src * weight` over the taps, shift right, clamp to 0...255. `taps` holds the start
/// and count of every output, then the weights.
private let resampleKernel = MLXFast.metalKernel(
    name: "antialiased_resample_uint8",
    inputNames: ["src", "taps", "params"],
    outputNames: ["dst"],
    source: """
        const int start = params[0];
        const int column_step = params[1];
        const int row_step = params[2];
        const int width = params[3];
        const int height = params[4];
        const int kernel_size = params[5];
        const int shift = params[6];
        const bool horizontal = (params[7] & 1) != 0;
        const bool transposed = (params[7] & 2) != 0;
        const int x = int(transposed ? thread_position_in_grid.y : thread_position_in_grid.x);
        const int y = int(transposed ? thread_position_in_grid.x : thread_position_in_grid.y);
        if (x >= width || y >= height) {
            return;
        }
        const int n = horizontal ? width : height;
        const int i = horizontal ? x : y;
        const int first = taps[i];
        const int count = taps[n + i];
        const int weights = 2 * n + i * kernel_size;
        const int tap_step = horizontal ? column_step : row_step;
        int p = start + (horizontal ? first : x) * column_step
            + (horizontal ? y : first) * row_step;
        const int bias = 1 << (shift - 1);
        int r = bias;
        int g = bias;
        int b = bias;
        for (int j = 0; j < count; ++j) {
            const int w = taps[weights + j];
            r += int(src[3 * p]) * w;
            g += int(src[3 * p + 1]) * w;
            b += int(src[3 * p + 2]) * w;
            p += tap_step;
        }
        // A negative sum clamps to 0; this avoids shifting a negative number.
        const int o = 3 * (y * width + x);
        dst[o] = uint8_t(r < 0 ? 0 : min(r >> shift, 255));
        dst[o + 1] = uint8_t(g < 0 ? 0 : min(g >> shift, 255));
        dst[o + 2] = uint8_t(b < 0 ? 0 : min(b >> shift, 255));
        """)
#endif

/// Dense `[outputSize, inputSize]` row-major weights of `F.interpolate(mode="bilinear",
/// align_corners=False, antialias=True)` on a float32 CPU tensor: ATen's
/// `_compute_indices_min_size_weights_aa` with its float32 and double arithmetic.
func antialiasedBilinearWeights(inputSize: Int, outputSize: Int) -> [Float] {
    var weights = [Float](repeating: 0, count: outputSize * inputSize)
    let scale = Float(inputSize) / Float(outputSize)
    let support: Float = scale >= 1 ? scale : 1
    let inverseScale: Float = scale >= 1 ? Float(1 / Double(scale)) : 1
    let maxTaps = Int(support.rounded(.up)) * 2 + 1
    var taps = [Float](repeating: 0, count: maxTaps)
    for i in 0 ..< outputSize {
        let center = Float(Double(scale) * (Double(i) + 0.5))
        let xmin = max(Int(Double(center - support) + 0.5), 0)
        let count = min(
            max(min(Int(Double(center + support) + 0.5), inputSize) - xmin, 0), maxTaps)
        var total: Float = 0
        for j in 0 ..< count {
            let x = abs(Float((Double(Float(j + xmin) - center) + 0.5) * Double(inverseScale)))
            taps[j] = x < 1 ? 1 - x : 0
            total += taps[j]
        }
        for j in 0 ..< count {
            weights[i * inputSize + xmin + j] = total != 0 ? taps[j] / total : taps[j]
        }
    }
    return weights
}

extension ResampleTaps {
    fileprivate func withPointers<R>(
        _ body: (UnsafePointer<Int>, UnsafePointer<Int>, UnsafePointer<Int32>) -> R
    ) -> R {
        start.withUnsafeBufferPointer { start in
            count.withUnsafeBufferPointer { count in
                weights.withUnsafeBufferPointer { weights in
                    body(start.baseAddress!, count.baseAddress!, weights.baseAddress!)
                }
            }
        }
    }
}

/// `value` clamped to 0...255. `UInt8(clamping:)` is generic, which is slow in Debug builds.
private func clampedByte(_ value: Int) -> UInt8 {
    value < 0 ? 0 : value > 255 ? 255 : UInt8(truncatingIfNeeded: value)
}

/// Runs `body` over contiguous chunks of `0..<count`, in parallel when `work` is large.
private func forEachChunk(_ count: Int, work: Int, _ body: (Range<Int>) -> Void) {
    let chunks = min(count, max(1, work / 4_000_000), 64)
    if chunks <= 1 {
        body(0 ..< count)
        return
    }
    // Chunks write disjoint rows, and `concurrentPerform` returns only when all are done.
    withoutActuallyEscaping(body) { body in
        let body = UncheckedSendable(body)
        DispatchQueue.concurrentPerform(iterations: chunks) { i in
            body.value(count * i / chunks ..< count * (i + 1) / chunks)
        }
    }
}

private struct UncheckedSendable<Value>: @unchecked Sendable {
    let value: Value
    init(_ value: Value) { self.value = value }
}
