// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// Tests of RGB8Image decoding and resizing. Expected values come from
// tools/lfm2_vl_test_vectors.py (torchvision, and Pillow, which `load_image` uses to decode).

private func decode(base64: String) throws -> RGB8Image {
    try RGB8Image(encoded: Data(base64Encoded: base64)!)
}

/// The largest difference between the bytes of two images of one size.
private func maxDifference(_ a: RGB8Image, _ b: RGB8Image) -> Int {
    zip(a.bytes, b.bytes).reduce(0) { (worst: Int, pair: (UInt8, UInt8)) -> Int in
        max(worst, abs(Int(pair.0) - Int(pair.1)))
    }
}

/// `oriented` as it was before it worked in tiles: one pass over the upright rows.
private func rowLoopOriented(_ image: RGB8Image, _ exifOrientation: Int) -> RGB8Image {
    guard (2 ... 8).contains(exifOrientation) else { return image }
    let (outWidth, outHeight) =
        exifOrientation >= 5 ? (image.height, image.width) : (image.width, image.height)
    let (start, columnStep, rowStep) = image.orientationSteps(exifOrientation)
    var out = [UInt8](repeating: 0, count: image.bytes.count)
    var d = 0
    var rowStart = start
    for _ in 0 ..< outHeight {
        var s = rowStart * 3
        for _ in 0 ..< outWidth {
            out[d] = image.bytes[s]
            out[d + 1] = image.bytes[s + 1]
            out[d + 2] = image.bytes[s + 2]
            d += 3
            s += columnStep * 3
        }
        rowStart += rowStep
    }
    return RGB8Image(width: outWidth, height: outHeight, bytes: out)
}

/// Saturated colour blocks. Their sharp chroma edges are where JPEG decoders differ most.
private func colourBlocks(width: Int, height: Int) -> RGB8Image {
    let colours: [[UInt8]] = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    ]
    var bytes: [UInt8] = []
    for y in 0 ..< height {
        for x in 0 ..< width {
            bytes += colours[(x / 5 + 2 * (y / 3)) % 6]
        }
    }
    return RGB8Image(width: width, height: height, bytes: bytes)
}

private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
private let displayP3 = CGColorSpace(name: CGColorSpace.displayP3)!

/// `image` as an opaque `CGImage` in `space`.
private func cgImage(_ image: RGB8Image, in space: CGColorSpace = sRGB) throws -> CGImage {
    let provider = try #require(CGDataProvider(data: Data(image.bytes) as CFData))
    return try #require(
        CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: image.width * 3, space: space, bitmapInfo: [], provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
}

/// The samples that store `image`, read in `source`, in `space`, as CoreGraphics converts them.
private func samples(
    _ image: RGB8Image, from source: CGColorSpace = sRGB, in space: CGColorSpace
) throws -> RGB8Image {
    let (width, height) = (image.width, image.height)
    var rgbx = [UInt8](repeating: 0, count: width * height * 4)
    try rgbx.withUnsafeMutableBytes { buffer in
        let context = try #require(
            CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.draw(
            try cgImage(image, in: source), in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    let bytes = (0 ..< width * height).flatMap { rgbx[$0 * 4 ..< $0 * 4 + 3] }
    return RGB8Image(width: width, height: height, bytes: bytes)
}

/// `image`, stored in `space`, encoded by ImageIO as `type`. ImageIO embeds the profile of any
/// space but sRGB.
private func encoded(_ image: RGB8Image, in space: CGColorSpace, type: String) throws -> Data {
    let data = NSMutableData()
    let destination = try #require(
        CGImageDestinationCreateWithData(data, type as CFString, 1, nil))
    let properties = [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary
    CGImageDestinationAddImage(destination, try cgImage(image, in: space), properties)
    try #require(CGImageDestinationFinalize(destination))
    return data as Data
}

/// A PNG without its colour chunks and EXIF, whose colour space tag also marks sRGB.
private func untagged(png: Data) -> Data {
    let bytes = [UInt8](png)
    var file = Data(bytes[..<8])
    var at = 8
    while at + 12 <= bytes.count {
        let length = bytes[at ..< at + 4].reduce(0) { $0 << 8 | Int($1) }
        let type = String(decoding: bytes[at + 4 ..< at + 8], as: UTF8.self)
        if !["iCCP", "sRGB", "gAMA", "cHRM", "cICP", "eXIf"].contains(type) {
            file.append(contentsOf: bytes[at ..< at + 12 + length])
        }
        at += 12 + length
    }
    return file
}

/// The sRGB profile under another maker's name, as cameras embed their own copy of it.
private func vendorSRGB() throws -> CGColorSpace {
    var icc = [UInt8](try #require(sRGB.copyICCData()) as Data)
    icc.replaceSubrange(48 ..< 56, with: Array("TESTCOPY".utf8))
    // A zero profile ID means it is not computed.
    icc.replaceSubrange(84 ..< 100, with: repeatElement(0, count: 16))
    let space = try #require(CGColorSpace(iccData: Data(icc) as CFData))
    try #require(!CFEqual(space, sRGB))
    return space
}

/// sRGB primaries with a gamma 2.2 curve, like Apple's older "sRGB Profile".
private let gamma22SRGB: CGColorSpace = {
    let white: [CGFloat] = [0.9505, 1, 1.089]
    let black: [CGFloat] = [0, 0, 0]
    let gamma: [CGFloat] = [2.2, 2.2, 2.2]
    let matrix: [CGFloat] = [
        0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505,
    ]
    return CGColorSpace(
        calibratedRGBWhitePoint: white, blackPoint: black, gamma: gamma, matrix: matrix)!
}()

/// `image` as a JPEG written by ImageIO, which subsamples chroma 4:2:0 at this quality.
private func jpegData(_ image: RGB8Image, orientation: Int = 1) throws -> Data {
    let data = NSMutableData()
    let destination = try #require(
        CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil))
    let properties: [CFString: Any] = [
        kCGImageDestinationLossyCompressionQuality: 0.75,
        kCGImagePropertyOrientation: orientation,
    ]
    CGImageDestinationAddImage(destination, try cgImage(image), properties as CFDictionary)
    try #require(CGImageDestinationFinalize(destination))
    return data as Data
}

/// A two-image MPO as Pillow writes it: `primary` with `segments` and an MP index (APP2 "MPF")
/// after its APP0 and APP1 segments, then `second`.
private func multiPictureJPEG(primary: Data, second: Data, segments: [UInt8] = []) -> Data {
    let bytes = [UInt8](primary)
    var at = 2
    while bytes[at] == 0xFF, bytes[at + 1] == 0xE0 || bytes[at + 1] == 0xE1 {
        let length = Int(bytes[at + 2]) * 256 + Int(bytes[at + 3])
        at += 2 + length
    }
    // APP2 of 88 bytes: "MPF\0", then a little-endian TIFF header.
    var index: [UInt8] = [0xFF, 0xE2, 0, 88, 0x4D, 0x50, 0x46, 0, 0x49, 0x49, 0x2A, 0]
    func put(_ value: Int, _ count: Int) {
        for byte in 0 ..< count {
            index.append(UInt8(value >> (8 * byte) & 0xFF))
        }
    }
    put(8, 4)
    // Three tags: version "0100", 2 images, and their 16-byte entries at offset 50.
    let tags = [(0xB000, 7, 4, 0x3030_3130), (0xB001, 4, 1, 2), (0xB002, 7, 32, 50)]
    put(tags.count, 2)
    for (tag, type, count, value) in tags {
        put(tag, 2)
        put(type, 2)
        put(count, 4)
        put(value, 4)
    }
    put(0, 4)
    // Entries: attributes, size, offset from the TIFF header (8 bytes into the segment).
    let firstSize = bytes.count + segments.count + 90
    let secondOffset = firstSize - (at + segments.count + 8)
    let entries = [(0x2003_0000, firstSize, 0), (0, second.count, secondOffset)]
    for (attributes, size, offset) in entries {
        put(attributes, 4)
        put(size, 4)
        put(offset, 4)
        put(0, 4)
    }
    var file = Data(bytes[..<at])
    file.append(contentsOf: segments)
    file.append(contentsOf: index)
    file.append(contentsOf: bytes[at...])
    file.append(second)
    return file
}

/// An APP1 segment with an XMP packet around `description`.
private func xmpSegment(_ description: String) -> [UInt8] {
    let parts: [String] = [
        "http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">",
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">",
        description, "</rdf:RDF></x:xmpmeta>",
    ]
    let packet = Array(parts.joined().utf8)
    let length = packet.count + 2
    var segment: [UInt8] = [0xFF, 0xE1, UInt8(length >> 8), UInt8(length & 0xFF)]
    segment += packet
    return segment
}

/// An Ultra HDR JPEG: `primary` with a gain-map directory and an MP index, then `gainMap` with
/// its metadata.
private func ultraHDRJPEG(primary: Data, gainMap: Data) -> Data {
    let version = "xmlns:hdrgm=\"http://ns.adobe.com/hdr-gain-map/1.0/\" hdrgm:Version=\"1.0\""
    var second = Data(gainMap.prefix(2))
    second.append(
        contentsOf: xmpSegment(
            "<rdf:Description \(version) hdrgm:GainMapMax=\"2\" hdrgm:HDRCapacityMax=\"2\"/>"))
    second.append(gainMap.dropFirst(2))
    let item = "<rdf:li rdf:parseType=\"Resource\"><Container:Item Item:Mime=\"image/jpeg\""
    let directory: [String] = [
        "<rdf:Description xmlns:Container=\"http://ns.google.com/photos/1.0/container/\"",
        " xmlns:Item=\"http://ns.google.com/photos/1.0/container/item/\" \(version)>",
        "<Container:Directory><rdf:Seq>\(item) Item:Semantic=\"Primary\"/></rdf:li>",
        "\(item) Item:Semantic=\"GainMap\" Item:Length=\"\(second.count)\"/></rdf:li>",
        "</rdf:Seq></Container:Directory></rdf:Description>",
    ]
    return multiPictureJPEG(
        primary: primary, second: second, segments: xmpSegment(directory.joined()))
}

@Suite struct ImagePixelsResizeTests {
    // (inW, inH, outW, outH, filter, byte sum, first 6 bytes, last 3 bytes) from torchvision
    // `resize(uint8, antialias=True)`.
    static let torchCases: [(Int, Int, Int, Int, ResampleFilter, Int, [UInt8], [UInt8])] = [
        (37, 53, 96, 64, .bicubic, 2_362_452, [0, 55, 114, 0, 59, 118], [0, 66, 151]),
        (37, 53, 15, 20, .bicubic, 115_394, [59, 122, 164, 130, 162, 133], [181, 123, 67]),
        (64, 48, 23, 17, .bicubic, 150_259, [72, 132, 159, 128, 156, 124], [203, 106, 76]),
        (50, 40, 50, 13, .bicubic, 250_263, [19, 78, 137, 80, 145, 169], [115, 174, 233]),
        (50, 40, 7, 40, .bicubic, 107_729, [106, 136, 141, 137, 129, 123], [124, 132, 142]),
        (3, 500, 2, 37, .bicubic, 28_318, [113, 135, 136, 129, 137, 120], [118, 128, 136]),
        (3, 500, 3, 41, .bicubic, 47_055, [108, 135, 147, 119, 141, 117], [110, 131, 140]),
        (300, 2, 31, 3, .bicubic, 35_542, [116, 134, 136, 131, 123, 119], [125, 130, 123]),
        (1, 1, 5, 3, .bicubic, 2_655, [0, 59, 118, 0, 59, 118], [0, 59, 118]),
        (37, 53, 96, 64, .bilinear, 2_361_175, [0, 59, 118, 2, 61, 120], [21, 80, 139]),
        (37, 53, 15, 20, .bilinear, 115_390, [71, 125, 158, 128, 153, 130], [175, 128, 75]),
        (3, 500, 2, 37, .bilinear, 28_324, [116, 135, 136, 129, 135, 122], [119, 128, 136]),
        (1024, 768, 1536, 1024, .bicubic, 601_904_847, [0, 56, 115, 12, 71, 130], [37, 96, 172]),
        (
            768, 1024, 416, 576, .bilinear, 91_661_761, [36, 95, 153, 101, 157, 171],
            [162, 149, 96]
        ),
    ]

    /// Checks `resize` against `torchCases`.
    static func expectTorchvisionValues(
        _ resize: (RGB8Image, Int, Int, ResampleFilter) -> RGB8Image
    ) {
        for (w, h, ow, oh, filter, sum, first, last) in torchCases {
            let out = resize(RGB8Image.testPattern(width: w, height: h), ow, oh, filter)
            let label = "\(w)x\(h) -> \(ow)x\(oh) \(filter)"
            #expect(out.width == ow && out.height == oh, "\(label)")
            #expect(out.byteSum == sum, "\(label)")
            #expect(Array(out.bytes.prefix(6)) == first, "\(label)")
            #expect(Array(out.bytes.suffix(3)) == last, "\(label)")
        }
    }

    @Test func resizeMatchesTorchvision() {
        Device.withDefaultDevice(.cpu) {
            Self.expectTorchvisionValues {
                $0.resized(width: $1, height: $2, filter: $3)
            }
            let image = RGB8Image.testPattern(width: 37, height: 53)
            #expect(image.resized(width: 37, height: 53, filter: .bicubic) == image)
        }
    }

    @Test func orientationMatchesExifTranspose() {
        let image = RGB8Image(width: 3, height: 2, bytes: (0 ..< 18).map { UInt8($0) })
        // Orientation -> (width, height, bytes) from `load_image` (`ImageOps.exif_transpose`).
        let expected: [Int: (Int, Int, [UInt8])] = [
            1: (3, 2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            2: (3, 2, [6, 7, 8, 3, 4, 5, 0, 1, 2, 15, 16, 17, 12, 13, 14, 9, 10, 11]),
            3: (3, 2, [15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]),
            4: (3, 2, [9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
            5: (2, 3, [0, 1, 2, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17]),
            6: (2, 3, [9, 10, 11, 0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17, 6, 7, 8]),
            7: (2, 3, [15, 16, 17, 6, 7, 8, 12, 13, 14, 3, 4, 5, 9, 10, 11, 0, 1, 2]),
            8: (2, 3, [6, 7, 8, 15, 16, 17, 3, 4, 5, 12, 13, 14, 0, 1, 2, 9, 10, 11]),
        ]
        for (orientation, (width, height, bytes)) in expected {
            #expect(
                image.oriented(orientation) == RGB8Image(width: width, height: height, bytes: bytes)
            )
        }
        #expect(image.oriented(0) == image && image.oriented(9) == image)
    }

    @Test func tiledOrientationMatchesTheRowLoop() {
        // Smaller than, equal to and larger than the 32x32 tiles, and not multiples of them.
        let sizes = [(1, 1), (1, 70), (70, 1), (31, 33), (32, 32), (64, 96), (100, 37), (129, 67)]
        for (width, height) in sizes {
            let image = RGB8Image.testPattern(width: width, height: height)
            for orientation in 0 ... 9 {
                let same = image.oriented(orientation) == rowLoopOriented(image, orientation)
                #expect(same, "\(width)x\(height) orientation \(orientation)")
            }
        }
        let empty = RGB8Image(width: 0, height: 5, bytes: [])
        #expect(empty.oriented(6) == RGB8Image(width: 5, height: 0, bytes: []))
    }

    @Test func cropFillsOutsideWithBlack() {
        let image = RGB8Image(width: 2, height: 2, bytes: (1 ... 12).map { UInt8($0) })
        #expect(image.cropped(x: 1, y: 0, width: 1, height: 2).bytes == [4, 5, 6, 10, 11, 12])
        #expect(
            image.cropped(x: -1, y: 1, width: 2, height: 2).bytes == [
                0, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ])
    }
}

#if canImport(Metal)
@Suite(.enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
struct ImagePixelsGPUResizeTests {
    /// Random bytes, a fifth of them 0 or 255, so bicubic overshoots clamp at both ends.
    private func randomImage(
        width: Int, height: Int, using rng: inout some RandomNumberGenerator
    ) -> RGB8Image {
        let bytes = (0 ..< width * height * 3).map { _ -> UInt8 in
            switch Int.random(in: 0 ..< 10, using: &rng) {
            case 0: 0
            case 1: 255
            default: UInt8.random(in: 0 ... 255, using: &rng)
            }
        }
        return RGB8Image(width: width, height: height, bytes: bytes)
    }

    /// Compares the GPU and CPU resize of `image` for every orientation and filter. Each
    /// target is `size(uprightWidth, uprightHeight)`.
    private func expectSameBytes(
        _ image: RGB8Image, _ size: (Int, Int) -> (Int, Int),
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        for orientation in 1 ... 8 {
            let upright =
                orientation >= 5 ? (image.height, image.width) : (image.width, image.height)
            let (width, height) = size(upright.0, upright.1)
            for filter in ResampleFilter.allCases {
                let passes = image.resizePasses(
                    width: width, height: height, filter: filter, orientation: orientation)
                let gpu = image.gpuResampled(passes, orientation: orientation)
                let cpu = image.cpuResampled(passes, orientation: orientation)
                let same = gpu == cpu
                #expect(
                    same,
                    "\(image.width)x\(image.height) orientation \(orientation) -> \(width)x\(height) \(filter)",
                    sourceLocation: sourceLocation)
            }
        }
    }

    @Test func gpuMatchesTorchvision() {
        ImagePixelsResizeTests.expectTorchvisionValues {
            $0.gpuResampled($0.resizePasses(width: $1, height: $2, filter: $3))
        }
    }

    @Test func gpuMatchesCPUOnRandomSizes() {
        var rng = SplitMix64(seed: 1)
        for _ in 0 ..< 30 {
            let image = randomImage(
                width: Int.random(in: 1 ... 257, using: &rng),
                height: Int.random(in: 1 ... 257, using: &rng), using: &rng)
            let size = (
                Int.random(in: 1 ... 257, using: &rng), Int.random(in: 1 ... 257, using: &rng)
            )
            expectSameBytes(image) { _, _ in size }
        }
    }

    @Test func gpuMatchesCPUOnExtremeRatios() {
        var rng = SplitMix64(seed: 2)
        let cases: [(Int, Int, Int, Int)] = [
            (1, 1, 300, 7), (2, 2, 1, 1), (2000, 3, 9, 1), (3, 2000, 1, 11), (1000, 1, 1, 1000),
            // The weights of a large shrink use the full 22-bit precision.
            (900, 700, 23, 31), (13, 17, 600, 450),
        ]
        for (width, height, newWidth, newHeight) in cases {
            let image = randomImage(width: width, height: height, using: &rng)
            expectSameBytes(image) { _, _ in (newWidth, newHeight) }
        }
    }

    @Test func gpuSkipsPassesLikeCPU() {
        var rng = SplitMix64(seed: 3)
        let image = randomImage(width: 61, height: 37, using: &rng)
        expectSameBytes(image) { width, _ in (width, 90) }
        expectSameBytes(image) { _, height in (20, height) }
        expectSameBytes(image) { width, height in (width, height) }
    }

    @Test("The GPU reads the stored pixels and frees them after")
    func storedPixelsAreFreed() async throws {
        let bytes = RGB8Image.testPattern(width: 640, height: 480).bytes
        let freed = FreedFlag()
        var pixels: MLXArray? = mappedArray(bytes) { freed.set() }
        let sum = try #require(pixels)[0 ..< bytes.count].asType(.int32).sum().item(Int32.self)
        #expect(
            Int(sum) == bytes.reduce(0) { (total: Int, byte: UInt8) -> Int in total + Int(byte) })
        pixels = nil
        // MLX can free the buffer on its completion thread.
        for _ in 0 ..< 500 where !freed.isSet {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(freed.isSet)
    }
}

/// Set once, from any thread.
private final class FreedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool { lock.withLock { value } }

    func set() { lock.withLock { value = true } }
}

/// SplitMix64, a small seeded generator.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}
#endif

@Suite struct ImagePixelsDecodingTests {
    @Test func pngDecodesLikeLoadImage() throws {
        // 3x2 RGB with an eXIf orientation of 6: `load_image` turns it upright.
        let turned = try decode(
            base64:
                "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAGmVYSWZNTQAqAAAACAABARIAAwAAAAEABgAAAAAAANZnS2kAAAAUSURBVHicY2RgZGIGAxZOTk4ICwACjgBIUiuFMgAAAABJRU5ErkJggg=="
        )
        #expect(
            turned
                == RGB8Image(
                    width: 2, height: 3,
                    bytes: [9, 10, 11, 0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17, 6, 7, 8]))

        let gray = try decode(
            base64:
                "iVBORw0KGgoAAAANSUhEUgAAAAMAAAABCAAAAAA+i0toAAAADElEQVR4nGNg8P0PAAGdAU2K0fFtAAAAAElFTkSuQmCC"
        )
        #expect(gray.bytes == [0, 0, 0, 77, 77, 77, 255, 255, 255])

        // 16-bit gray with alpha keeps the high byte.
        let grayAlpha16 = try decode(
            base64:
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABEAQAAAAOu2tCAAAAEUlEQVR4nGNgfsHAcOL///8AD5MEsdbArBoAAAAASUVORK5CYII="
        )
        #expect(grayAlpha16.bytes == [3, 3, 3, 200, 200, 200])

        // Alpha is dropped without compositing, even where it is 0.
        let rgba = try decode(
            base64:
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGPgEpFjOJFi1AAABskCGzDccEoAAAAASUVORK5CYII="
        )
        #expect(rgba.bytes == [10, 20, 30, 200, 100, 50])

        // A 2-bit palette with tRNS reads the palette colours.
        let palette = try decode(
            base64:
                "iVBORw0KGgoAAAANSUhEUgAAAAQAAAABAgMAAACEUudeAAAADFBMVEX/AAAA/wAAAP8JCAei0mLAAAAABHRSTlMA/4AANojKPAAAAApJREFUeJxjkAYAAB0AHI709SEAAAAASUVORK5CYII="
        )
        #expect(palette.bytes == [255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 8, 7])
    }

    @Test func urlsDecodeFilesAndDataURLs() throws {
        let base64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGPgEpFjOJFi1AAABskCGzDccEoAAAAASUVORK5CYII="
        let dataURL = URL(string: "data:image/png;base64,\(base64)")!
        #expect(try RGB8Image(.url(dataURL)).bytes == [10, 20, 30, 200, 100, 50])

        let file = FileManager.default.temporaryDirectory.appending(
            component: "\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data(base64Encoded: base64)!.write(to: file)
        #expect(try RGB8Image(.url(file)).bytes == [10, 20, 30, 200, 100, 50])

        #expect(throws: VLMError.self) {
            try RGB8Image(.url(file.appending(component: "missing.png")))
        }
        #expect(throws: VLMError.self) {
            try RGB8Image(encoded: Data("not an image".utf8))
        }
    }

    @Test func jpegDecodesUprightAndClose() throws {
        // 12x8 4:2:0 JPEG of colour blocks with EXIF orientation 6. Core Image and libjpeg-turbo
        // differ by a few levels, and the exact amount depends on the OS.
        let jpeg = try decode(
            base64:
                "/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAIAAwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD0/Rv2Xf8AhTnnf8VN/a/9o4/5cPI8vy8/9NWznzPbGPevg3/hnX/qYP8AyS/+2UUV9/8ARor1KPCf9swdsRiv4sv5vZ1a0IafDHljp7qV93d6n2XhzlWD494/4txXEcPbzh9Rs7uFr0Jp6U3FbRjuunmz/9k="
        )
        // `load_image` output: upright 8x12.
        let reference = [UInt8](
            Data(
                base64Encoded:
                    "/f8B+P4A/f8L/+Uv1xsA+wAA/QYB9wMA+f8A/P8C9/4D/+Uq1RkA/gIB+AAA+QIA+v0A//8C9/0A/+cp1xkA/wQE+gAA/wQDzv9Rxv9JzP9Q6ftbjT8AqCsAsTEAqSkAMey1Ove/Lu61QvSWQdQuVdcSTcwLWNUXBP//APv/A///B//KAPo1AP8AAf8CAPsAAv//Af//A///Cv/KAPs9AP8GAP8HAv8HUc//WdX/U8v/UNztHdOGHORoGNhfIt5lsyvlsyvntSrpkzrqAhGeACKgAB+gAB6i/wD1/wD5/wD+1hH/LwDnAgD5BAD8BAD8+QT1+wP6/AD/0RL/LADnAAH5AAH8AAL+9Qb19QX69wL/zRT/KwDnAAL3AAP8AAT+"
            )!)
        #expect(jpeg.width == 8 && jpeg.height == 12)
        #expect(jpeg.bytes.count == reference.count)
        let total = zip(jpeg.bytes, reference).reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }
        #expect(Double(total) / Double(reference.count) < 4)
    }

    @Test func mpoAndUltraHDRDecodeLikePlainJPEG() throws {
        // ImageIO's own decode of these 4:2:0 blocks is about 50 levels from Core Image's.
        let plain = try jpegData(colourBlocks(width: 48, height: 32), orientation: 6)
        let reference = try RGB8Image(encoded: plain)
        #expect(reference.width == 32 && reference.height == 48)

        let second = try jpegData(RGB8Image.testPattern(width: 24, height: 16))
        let files = [
            "MPO": multiPictureJPEG(primary: plain, second: second),
            "Ultra HDR": ultraHDRJPEG(primary: plain, gainMap: second),
        ]
        for (name, file) in files {
            let decoded = try RGB8Image(encoded: file)
            #expect(decoded.width == 32 && decoded.height == 48, "\(name)")
            let worst = maxDifference(decoded, reference)
            #expect(worst <= 4, "\(name): \(worst) levels from the plain JPEG")
        }
    }

    @Test func stripRendersGiveTheSameBytes() throws {
        // An opaque image renders its stored bytes. The crop moves the extent off the origin.
        let pattern = RGB8Image.testPattern(width: 101, height: 75)
        let cropped = CIImage(cgImage: try cgImage(pattern)).cropped(
            to: CGRect(x: 3, y: 5, width: 90, height: 61))
        let expected = pattern.cropped(x: 3, y: 75 - 5 - 61, width: 90, height: 61)
        // One strip, 2 strips, 3 strips and a strip per row.
        for stripPixels in [Int.max, 90 * 31, 90 * 25, 50] {
            let same = try RGB8Image(rendering: cropped, stripPixels: stripPixels) == expected
            #expect(same, "\(stripPixels) pixels per strip")
        }

        // A 4:2:0 JPEG of odd size, so strips cut through chroma rows.
        let jpeg = try jpegData(pattern)
        let image = try #require(CIImage(data: jpeg, options: [.colorSpace: NSNull()]))
        let whole = try RGB8Image(rendering: image, stripPixels: .max)
        #expect(whole.width == 101 && whole.height == 75)
        for stripPixels in [101 * 38, 101 * 30, 50] {
            let strips = try RGB8Image(rendering: image, stripPixels: stripPixels)
            #expect(strips.width == 101 && strips.height == 75)
            #expect(strips == whole, "\(stripPixels) pixels per strip")
        }
    }

    @Test func arraysAreUsedAsGiven() throws {
        try Device.withDefaultDevice(.cpu) { () throws in
            let values = (0 ..< 30).map { UInt8($0 * 8) }
            let hwc = MLXArray(values, [2, 5, 3])
            #expect(try RGB8Image(array: hwc) == RGB8Image(width: 5, height: 2, bytes: values))

            let chw = hwc.transposed(2, 0, 1)
            #expect(try RGB8Image(array: chw) == RGB8Image(array: hwc))

            let rgba = MLXArray([10, 20, 30, 0, 40, 50, 60, 255] as [UInt8], [2, 1, 4])
            #expect(try RGB8Image(array: rgba).bytes == [10, 20, 30, 40, 50, 60])

            let unit = MLXArray([0, 0.5, 1, 0.25] as [Float], [2, 2])
            #expect(
                try RGB8Image(array: unit).bytes == [
                    0, 0, 0, 128, 128, 128, 255, 255, 255, 64, 64, 64,
                ])

            let wide = MLXArray([-5, 300, 17, 0, 0, 0] as [Float], [2, 1, 3])
            #expect(try RGB8Image(array: wide).bytes == [0, 255, 17, 0, 0, 0])

            let extreme = MLXArray(
                [0, 1e20, -.infinity, .infinity, 7, -1e30] as [Float], [2, 1, 3])
            #expect(try RGB8Image(array: extreme).bytes == [0, 255, 0, 255, 7, 0])

            #expect(throws: VLMError.self) {
                try RGB8Image(array: MLXArray([0.2, .nan, 0.7] as [Float], [1, 1, 3]))
            }
            #expect(throws: VLMError.self) {
                try RGB8Image(array: MLXArray.zeros([5, 6, 7]))
            }
        }
    }

    @Test func ambiguousArraysReadLikeTransformers() throws {
        // `infer_channel_dimension_format`: a first dimension of 1 or 3 is the channel axis.
        try Device.withDefaultDevice(.cpu) { () throws in
            let values = (0 ..< 36).map { UInt8($0 * 7) }
            let square = MLXArray(values, [3, 4, 3])
            let image = try RGB8Image(array: square)
            #expect(image.width == 3 && image.height == 4)
            #expect(try image == RGB8Image(array: square.transposed(1, 2, 0)))

            // `UserInput.Image.asCIImage()` reads this as a 4x3 image too.
            let wide = try RGB8Image(array: MLXArray(values, [3, 3, 4]))
            #expect(wide.width == 4 && wide.height == 3)

            let gray = try RGB8Image(array: MLXArray(Array(values[0 ..< 15]), [1, 5, 3]))
            #expect(gray.width == 3 && gray.height == 5)
            #expect(gray.bytes.prefix(6) == [0, 0, 0, 7, 7, 7])

            let tall = try RGB8Image(array: MLXArray(Array(values[0 ..< 24]), [4, 2, 3]))
            #expect(tall.width == 2 && tall.height == 4)
        }
    }

    @Test func sRGBCIImagesAreUsedAsGiven() throws {
        let pixels = RGB8Image(width: 2, height: 1, bytes: [1, 2, 3, 250, 251, 252])
        // Wrapped sRGB CGImages keep their stored samples.
        #expect(try RGB8Image(.ciImage(CIImage(cgImage: cgImage(pixels)))) == pixels)
    }

    @Test func derivedSRGBCIImagesRenderTheirStoredValues() throws {
        // 4x3 sRGB RGBA with straight alpha; one pixel is half transparent.
        let (width, height) = (4, 3)
        var rgba: [UInt8] = (0 ..< width * height).flatMap { (i: Int) -> [UInt8] in
            let red = UInt8(10 + i * 20)
            let green = UInt8(250 - i * 15)
            let blue = UInt8((i * 67) % 256)
            return [red, green, blue, 255]
        }
        rgba[(1 * width + 2) * 4 + 3] = 128
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let cgImage = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: sRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let source = try RGB8Image(cgImage: cgImage)

        // Core Image puts the origin at the bottom left: this is columns 1...3 of rows 1...2.
        // Its rendering can move the half-transparent pixel by a level or two.
        let derived = CIImage(cgImage: cgImage).cropped(
            to: CGRect(x: 1, y: 0, width: 3, height: 2))
        #expect(derived.cgImage == nil)
        let rendered = try RGB8Image(.ciImage(derived))
        let expected = source.cropped(x: 1, y: 1, width: 3, height: 2)
        #expect(rendered.width == expected.width && rendered.height == expected.height)
        #expect(zip(rendered.bytes, expected.bytes).allSatisfy { abs(Int($0) - Int($1)) <= 2 })
    }

    @Test func extendedRangeFloatImagesDrawInSRGB() throws {
        // 2x1 float16 RGBA in extended linear sRGB, as ImageIO decodes OpenEXR: (0.5, 0.2, 0.1)
        // and (1, 2, 0). CoreGraphics has no 8-bit context in that space.
        let halves: [UInt16] = [0x3800, 0x3266, 0x2E66, 0x3C00, 0x3C00, 0x4000, 0x0000, 0x3C00]
        let bytes = halves.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
        let cgImage = CGImage(
            width: 2, height: 1, bitsPerComponent: 16, bitsPerPixel: 64, bytesPerRow: 16,
            space: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            bitmapInfo: [
                .floatComponents,
                CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Little.rawValue),
            ],
            provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)!
        let pixels = try RGB8Image(cgImage: cgImage)
        // About the sRGB encoding of the linear values, clipped to 255; CoreGraphics decides.
        let expected = [188, 124, 89, 255, 255, 0]
        #expect(
            zip(pixels.bytes, expected).allSatisfy { abs(Int($0) - $1) <= 8 }, "\(pixels.bytes)")
    }
}

@Suite struct ImagePixelsColourTests {
    @Test func filesWithAnotherProfileAreConvertedToSRGB() throws {
        let source = RGB8Image.testPattern(width: 24, height: 16)
        let spaces = [
            ("Display P3", displayP3),
            ("Adobe RGB", CGColorSpace(name: CGColorSpace.adobeRGB1998)!),
        ]
        for (name, space) in spaces {
            let stored = try samples(source, in: space)
            #expect(maxDifference(stored, source) > 20, "\(name): the file must hold other values")
            let png = try RGB8Image(encoded: encoded(stored, in: space, type: "public.png"))
            #expect(maxDifference(png, source) <= 3, "\(name) PNG: \(maxDifference(png, source))")

            // A JPEG's own samples, converted as CoreGraphics converts them.
            let jpeg = try encoded(stored, in: space, type: "public.jpeg")
            let unmanaged = try #require(CIImage(data: jpeg, options: [.colorSpace: NSNull()]))
            let jpegSamples = try RGB8Image(rendering: unmanaged)
            let expected = try samples(jpegSamples, from: space, in: sRGB)
            #expect(maxDifference(jpegSamples, expected) > 20, "\(name) JPEG")
            let decoded = try RGB8Image(encoded: jpeg)
            #expect(
                maxDifference(decoded, expected) <= 2,
                "\(name) JPEG: \(maxDifference(decoded, expected))")
        }
    }

    @Test func sRGBAndUntaggedFilesKeepTheirStoredSamples() throws {
        let source = RGB8Image.testPattern(width: 24, height: 16)
        let png = try encoded(source, in: sRGB, type: "public.png")
        let untaggedPNG = untagged(png: png)
        // ImageIO reads an untagged file as sRGB, but names no profile for it.
        let reader = try #require(CGImageSourceCreateWithData(untaggedPNG as CFData, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(reader, 0, nil) as? [CFString: Any]
        #expect(properties?[kCGImagePropertyProfileName] == nil)
        let files = [
            ("sRGB", png), ("untagged", untaggedPNG),
            ("gamma 2.2 sRGB", try encoded(source, in: gamma22SRGB, type: "public.png")),
        ]
        for (name, file) in files {
            #expect(try RGB8Image(encoded: file) == source, "\(name)")
        }

        // The same JPEG data, untagged and with a camera maker's copy of the sRGB profile.
        let vendorJPEG = try encoded(source, in: vendorSRGB(), type: "public.jpeg")
        let vendorReader = try #require(CGImageSourceCreateWithData(vendorJPEG as CFData, nil))
        let space = CGImageSourceCreateImageAtIndex(vendorReader, 0, nil)?.colorSpace
        #expect(space.map { !CFEqual($0, sRGB) } == true)
        let jpeg = try RGB8Image(encoded: encoded(source, in: sRGB, type: "public.jpeg"))
        #expect(try RGB8Image(encoded: vendorJPEG) == jpeg)
    }

    /// A `CGImage` of `bytes` laid out as given.
    private func bitmap(
        _ bytes: [UInt8], width: Int, bitsPerComponent: Int = 8, bitsPerPixel: Int,
        space: CGColorSpace, bitmapInfo: CGBitmapInfo = []
    ) throws -> CGImage {
        let rowBytes = width * bitsPerPixel / 8
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        return try #require(
            CGImage(
                width: width, height: bytes.count / rowBytes, bitsPerComponent: bitsPerComponent,
                bitsPerPixel: bitsPerPixel, bytesPerRow: rowBytes, space: space,
                bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent))
    }

    @Test func cgImagesInAnotherSpaceAreConverted() throws {
        let source = RGB8Image.testPattern(width: 16, height: 16)
        let stored = try samples(source, in: displayP3)
        let rgb = try RGB8Image(cgImage: cgImage(stored, in: displayP3))
        #expect(maxDifference(rgb, source) <= 3, "RGB: \(maxDifference(rgb, source))")

        // Alpha is dropped before the conversion, so colour stays straight.
        var rgba: [UInt8] = []
        for pixel in 0 ..< 256 { rgba += stored.bytes[pixel * 3 ..< pixel * 3 + 3] + [128] }
        let last = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        let alpha = try RGB8Image(
            cgImage: bitmap(rgba, width: 16, bitsPerPixel: 32, space: displayP3, bitmapInfo: last))
        #expect(maxDifference(alpha, source) <= 3, "RGBA: \(maxDifference(alpha, source))")

        // 16-bit samples keep their high byte, then convert.
        let wide = try bitmap(
            stored.bytes.flatMap { [$0, $0] }, width: 16, bitsPerComponent: 16, bitsPerPixel: 48,
            space: displayP3)
        let sixteen = try RGB8Image(cgImage: wide)
        #expect(maxDifference(sixteen, source) <= 3, "16-bit: \(maxDifference(sixteen, source))")

        // A palette converts from its base space; each pixel has its own entry.
        let palette = try #require(
            CGColorSpace(indexedBaseSpace: displayP3, last: 255, colorTable: stored.bytes))
        let indices = (0 ... 255).map(UInt8.init)
        let indexed = try RGB8Image(
            cgImage: bitmap(indices, width: 16, bitsPerPixel: 8, space: palette))
        #expect(maxDifference(indexed, source) <= 3, "palette: \(maxDifference(indexed, source))")
    }

    @Test func ciImagesAreReadInTheirColourSpace() throws {
        let source = RGB8Image.testPattern(width: 24, height: 16)
        let stored = try samples(source, in: displayP3)
        let p3Image = try cgImage(stored, in: displayP3)
        let png = try encoded(stored, in: displayP3, type: "public.png")
        let file = FileManager.default.temporaryDirectory.appending(
            component: "\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: file) }
        try png.write(to: file)

        // Core Image's origin is at the bottom left: this is columns 2...21 of rows 3...12.
        let crop = CGRect(x: 2, y: 3, width: 20, height: 10)
        let cropped = CIImage(cgImage: p3Image).cropped(to: crop)
        // This filter leaves no colour space, so Core Image converts the source.
        let inverted = CIImage(cgImage: p3Image).applyingFilter("CIColorInvert")
        let inputs: [(String, CIImage, RGB8Image)] = [
            ("wrapped CGImage", CIImage(cgImage: p3Image), source),
            ("data", try #require(CIImage(data: png)), source),
            ("file", try #require(CIImage(contentsOf: file)), source),
            ("cropped", cropped, source.cropped(x: 2, y: 3, width: 20, height: 10)),
            ("filtered", inverted.applyingFilter("CIColorInvert"), source),
        ]
        for (name, image, expected) in inputs {
            let decoded = try RGB8Image(.ciImage(image))
            #expect(decoded.width == expected.width && decoded.height == expected.height, "\(name)")
            #expect(
                maxDifference(decoded, expected) <= 3,
                "\(name): \(maxDifference(decoded, expected))")
        }

        // With colour management off, the stored samples.
        let unmanaged = try #require(CIImage(data: png, options: [.colorSpace: NSNull()]))
        #expect(try RGB8Image(.ciImage(unmanaged)) == stored)
    }

    @Test func profilesNearSRGBReadAsSRGB() throws {
        #expect(SRGBConversion(from: nil) == nil)
        #expect(SRGBConversion(from: sRGB) == nil)
        #expect(SRGBConversion(from: try vendorSRGB()) == nil)
        #expect(SRGBConversion(from: gamma22SRGB) == nil)
        #expect(SRGBConversion(from: CGColorSpace(name: CGColorSpace.extendedSRGB)) == nil)
        #expect(SRGBConversion(from: CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)) == nil)
        #expect(SRGBConversion(from: CGColorSpace(name: CGColorSpace.genericCMYK)) == nil)

        #expect(SRGBConversion(from: displayP3) != nil)
        #expect(SRGBConversion(from: CGColorSpace(name: CGColorSpace.adobeRGB1998)) != nil)
        #expect(SRGBConversion(from: CGColorSpace(name: CGColorSpace.linearSRGB)) != nil)

        // vImage cannot read HLG; CoreGraphics draws it.
        let hlg = try #require(SRGBConversion(from: CGColorSpace(name: CGColorSpace.displayP3_HLG)))
        var gray = RGB8Image(width: 1, height: 1, bytes: [128, 128, 128])
        #expect(hlg.apply(to: &gray))
        #expect(gray.bytes != [128, 128, 128])
    }
}
