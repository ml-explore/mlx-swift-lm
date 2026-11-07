// Copyright © 2026 Apple Inc.
// 8-bit RGB decoding as transformers' `load_image` does: the stored samples and the EXIF
// transpose of files. RGB in a colour space other than sRGB is converted to sRGB first. The
// LFM2-VL processor uses it.

import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXLMCommon

// MARK: - RGB8Image

/// 8-bit RGB pixels, row-major, interleaved, no alpha.
///
/// Decoding gives the stored samples, as transformers' `load_image` does, except that RGB in a
/// colour space other than sRGB is converted to sRGB (see `SRGBConversion`). Per-pixel loops use
/// raw pointers and `while`, which are faster in Debug builds.
struct RGB8Image: Equatable, Sendable {
    var width: Int
    var height: Int
    /// `width * height * 3` bytes.
    var bytes: [UInt8]

    init(width: Int, height: Int, bytes: [UInt8]) {
        precondition(width >= 0 && height >= 0, "negative image size \(width)x\(height)")
        precondition(
            bytes.count == width * height * 3,
            "\(bytes.count) bytes for a \(width)x\(height) RGB image")
        self.width = width
        self.height = height
        self.bytes = bytes
    }
}

// MARK: Decoding

extension RGB8Image {
    /// Decode as transformers' `load_image` does: `.url` reads the file and applies its EXIF
    /// orientation. The processor's own path loading does not rotate. `.ciImage` is read in its
    /// colour space, and `.array` is used as given.
    init(_ image: UserInput.Image) throws {
        self = try OrientedImage(image).upright
    }

    /// Decodes the first frame of encoded image bytes and turns it upright; see
    /// `OrientedImage.init(encoded:)`.
    init(encoded data: Data) throws {
        self = try OrientedImage(encoded: data).upright
    }

    /// The stored pixels of a `CGImage`, like `convert("RGB")`: alpha is dropped, gray is
    /// replicated, a palette is looked up. RGB is then converted to sRGB as `SRGBConversion`
    /// decides.
    init(cgImage image: CGImage) throws {
        try self.init(cgImage: image, colorSpace: image.colorSpace)
    }

    /// Reads the samples of `image` as being in `colorSpace`, which Core Image can override.
    private init(cgImage image: CGImage, colorSpace: CGColorSpace?) throws {
        let conversion = SRGBConversion(from: colorSpace)
        if var pixels = Self.storedPixels(image) {
            conversion?.apply(to: &pixels)
            self = pixels
        } else {
            self = try Self.drawnPixels(image, inOwnSpace: conversion == nil)
        }
    }

    /// A `CIImage` in its colour space: its `CGImage` when it wraps one, else a render,
    /// unpremultiplied, so fully transparent pixels come out black. Some filters leave an image
    /// with no colour space; Core Image then converts each of its sources to sRGB.
    init(ciImage image: CIImage) throws {
        if let cgImage = image.cgImage {
            self = try RGB8Image(cgImage: cgImage, colorSpace: image.colorSpace)
        } else if let space = image.colorSpace {
            self = try RGB8Image(rendering: image)
            SRGBConversion(from: space)?.apply(to: &self)
        } else {
            self = try RGB8Image(rendering: image, inSRGB: true)
        }
    }

    /// Most pixels in one render strip: 64 MB of RGBA. Each extra strip costs 1 ms or more.
    static let stripPixels = 1 << 24

    /// Renders with colour management off, or with `inSRGB` in sRGB. A render above
    /// `stripPixels` goes in equal row strips, which lowers peak memory and gives the same bytes.
    init(
        rendering image: CIImage, stripPixels: Int = RGB8Image.stripPixels, inSRGB: Bool = false
    ) throws {
        precondition(stripPixels >= 1, "\(stripPixels) pixels per strip")
        let extent = image.extent
        guard !extent.isInfinite, !extent.isNull, extent.width >= 1, extent.height >= 1 else {
            throw VLMError.imageProcessingFailure("CIImage has no finite extent")
        }
        let bounds = extent.integral
        let width = Int(bounds.width)
        let height = Int(bounds.height)
        let pixels = width * height
        let strips = (pixels - 1) / stripPixels + 1
        let stripRows = (height + strips - 1) / strips
        let straight = image.unpremultiplyingAlpha()
        var rgba = [UInt8](repeating: 0, count: width * stripRows * 4)
        let bytes = try [UInt8](unsafeUninitializedCapacity: pixels * 3) { out, count in
            try rgba.withUnsafeMutableBytes { strip in
                var y = 0
                while y < height {
                    let rows = min(stripRows, height - y)
                    // Core Image's y axis points up; the bitmap's rows run top down.
                    let rect = CGRect(
                        x: bounds.minX, y: bounds.maxY - CGFloat(y + rows), width: bounds.width,
                        height: CGFloat(rows))
                    (inSRGB ? sRGBRenderContext : renderContext).render(
                        straight, toBitmap: strip.baseAddress!, rowBytes: width * 4, bounds: rect,
                        format: .RGBA8, colorSpace: inSRGB ? sRGB : nil)
                    var source = vImage_Buffer(
                        data: strip.baseAddress, height: vImagePixelCount(rows),
                        width: vImagePixelCount(width), rowBytes: width * 4)
                    var destination = vImage_Buffer(
                        data: out.baseAddress! + y * width * 3, height: vImagePixelCount(rows),
                        width: vImagePixelCount(width), rowBytes: width * 3)
                    let flags = vImage_Flags(kvImageNoFlags)
                    guard
                        vImageConvert_RGBA8888toRGB888(&source, &destination, flags)
                            == kvImageNoError
                    else {
                        throw VLMError.imageProcessingFailure("unsupported pixel format")
                    }
                    y += rows
                }
            }
            count = pixels * 3
        }
        self.init(width: width, height: height, bytes: bytes)
    }

    /// An `HW`, `HWC` or `CHW` array with 1 (gray), 2 (gray, alpha), 3 (RGB) or 4 (RGBA) channels.
    /// The channel axis is picked as transformers' `infer_channel_dimension_format` picks it:
    /// `CHW` when the first dimension is 1 or 3, else `HWC` when the last is 1 to 4, else `CHW`
    /// for 2 or 4. A first dimension of 3 reads as `CHW` in `UserInput.Image.asCIImage()` too.
    /// Floats with a maximum of at most 1 are scaled by 255; values are rounded and clipped to
    /// 0...255. NaN throws.
    init(array: MLXArray) throws {
        var shape = array.shape
        if shape.count == 2 { shape.insert(1, at: 0) }
        guard shape.count == 3 else {
            throw VLMError.imageProcessingFailure(
                "an image array needs 2 or 3 dimensions: \(array.shape)")
        }
        let channelsFirst: Bool
        if [1, 3].contains(shape[0]) {
            channelsFirst = true
        } else if (1 ... 4).contains(shape[2]) {
            channelsFirst = false
        } else if [2, 4].contains(shape[0]) {
            channelsFirst = true
        } else {
            throw VLMError.imageProcessingFailure(
                "an image array needs 1 to 4 channels: \(array.shape)")
        }
        let (height, width, channels) =
            channelsFirst ? (shape[1], shape[2], shape[0]) : (shape[0], shape[1], shape[2])
        guard height >= 1, width >= 1 else {
            throw VLMError.imageProcessingFailure("empty image array: \(array.shape)")
        }

        // Conversions run on the CPU, which also handles float64.
        let values: [UInt8] = try Device.withDefaultDevice(.cpu) {
            let bytes: MLXArray
            switch array.dtype {
            case .uint8:
                bytes = array
            case .bool:
                bytes = array.asType(.uint8) * 255
            case let dtype where dtype.isFloatingPoint:
                let floats = array.asType(.float32)
                guard !any(isNaN(floats)).item(Bool.self) else {
                    throw VLMError.imageProcessingFailure("the image array contains NaN")
                }
                let scale: Float = floats.max().item(Float.self) <= 1 ? 255 : 1
                // `round` is half to even. Clip before the conversion, which is undefined for
                // infinity and huge values.
                bytes = clip(round(floats * scale), min: 0, max: 255).asType(.uint8)
            default:
                bytes = clip(array, min: 0, max: 255).asType(.uint8)
            }
            // `asArray` copies a strided array element by element.
            return contiguous(bytes).asArray(UInt8.self)
        }

        let plane = width * height
        // Steps between pixels and between channels; gray repeats channel 0.
        let pixelStep = channelsFirst ? 1 : channels
        let channelStep = channels < 3 ? 0 : channelsFirst ? plane : 1
        var bytes = [UInt8](repeating: 0, count: plane * 3)
        values.withUnsafeBufferPointer { values in
            bytes.withUnsafeMutableBufferPointer { bytes in
                let (source, rgb) = (values.baseAddress!, bytes.baseAddress!)
                var pixel = 0
                while pixel < plane {
                    let s = pixel * pixelStep
                    rgb[pixel * 3] = source[s]
                    rgb[pixel * 3 + 1] = source[s + channelStep]
                    rgb[pixel * 3 + 2] = source[s + 2 * channelStep]
                    pixel += 1
                }
            }
        }
        self.init(width: width, height: height, bytes: bytes)
    }

    /// Unpacks 8- and 16-bit RGB, gray and palette bitmaps. 16-bit samples keep their high byte
    /// (PIL clips 16-bit gray to 255 instead). Returns nil for other layouts.
    private static func storedPixels(_ image: CGImage) -> RGB8Image? {
        let bitsPerComponent = image.bitsPerComponent
        guard bitsPerComponent == 8 || bitsPerComponent == 16,
            !image.bitmapInfo.contains(.floatComponents),
            let space = image.colorSpace,
            let data = image.dataProvider?.data,
            let base = CFDataGetBytePtr(data)
        else { return nil }

        let width = image.width
        let height = image.height
        let rowBytes = image.bytesPerRow
        let bytesPerComponent = bitsPerComponent / 8
        let bytesPerPixel = image.bitsPerPixel / 8
        let components = bytesPerPixel / bytesPerComponent
        let colorComponents: Int
        switch space.model {
        case .rgb: colorComponents = 3
        case .monochrome, .indexed: colorComponents = 1
        default: return nil
        }
        let alpha = image.alphaInfo
        let alphaFirst = [.premultipliedFirst, .first, .noneSkipFirst].contains(alpha)
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let hasAlphaSlot = alpha != .none && alpha != .alphaOnly
        guard alpha != .alphaOnly, components == colorComponents + (hasAlphaSlot ? 1 : 0),
            image.bitsPerPixel % 8 == 0, width >= 1, height >= 1,
            CFDataGetLength(data) >= (height - 1) * rowBytes + width * bytesPerPixel
        else { return nil }

        // Byte offset of each logical component; little-endian words reverse the order.
        let order = image.bitmapInfo.rawValue & CGBitmapInfo.byteOrderMask.rawValue
        let little32 = order == CGImageByteOrderInfo.order32Little.rawValue
        let little16 = order == CGImageByteOrderInfo.order16Little.rawValue
        var offsets = (0 ..< components).map { $0 * bytesPerComponent }
        var littleComponents = false
        if bytesPerComponent == 1 && (little32 || little16) {
            guard bytesPerPixel == (little32 ? 4 : 2) else { return nil }
            offsets.reverse()
        } else if bytesPerComponent == 2 {
            guard !little32 else { return nil }
            littleComponents = little16
        }
        let first = alphaFirst ? 1 : 0
        let alphaIndex = alphaFirst ? 0 : components - 1

        if let decode = image.decode {
            let ranges = (0 ..< colorComponents).map { (decode[2 * $0], decode[2 * $0 + 1]) }
            guard ranges.allSatisfy({ $0 == (0, 1) }) else { return nil }
        }

        let model = space.model
        if bytesPerComponent == 1 && !premultiplied && (model == .rgb || model == .monochrome) {
            let rgb = (0 ..< 3).map { offsets[model == .rgb ? first + $0 : first] }
            let pixels = withExtendedLifetime(data) {
                unpacked(
                    base, width: width, height: height, rowBytes: rowBytes,
                    pixelBytes: bytesPerPixel, rgb: rgb)
            }
            if let pixels { return pixels }
        }

        var table: [UInt8] = []
        var tableComponents = 3
        if model == .indexed {
            guard bytesPerComponent == 1, let base = space.baseColorSpace,
                base.model == .rgb || base.model == .monochrome
            else { return nil }
            tableComponents = base.model == .rgb ? 3 : 1
            table = space.colorTable ?? []
        }
        let tableCount = table.count / tableComponents
        let tableStep = tableComponents == 3 ? 1 : 0

        // Offsets of the high (or only) byte of the colour components, the last repeated for
        // gray and palette, and of alpha.
        let high = offsets.map { littleComponents ? $0 + 1 : $0 }
        let c = (0 ..< 3).map { high[min(first + $0, components - 1)] }
        let (c0, c1, c2, a0) = (c[0], c[1], c[2], high[alphaIndex])

        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        withExtendedLifetime(data) {
            table.withUnsafeBufferPointer { table in
                bytes.withUnsafeMutableBufferPointer { bytes in
                    let out = bytes.baseAddress!
                    var o = 0
                    var y = 0
                    while y < height {
                        var p = y * rowBytes
                        var x = 0
                        while x < width {
                            var (r, g, b): (Int, Int, Int)
                            switch model {
                            case .rgb:
                                r = Int(base[p + c0])
                                g = Int(base[p + c1])
                                b = Int(base[p + c2])
                            case .indexed:
                                let i = Int(base[p + c0])
                                if i < tableCount {
                                    let e = i * tableComponents
                                    r = Int(table[e])
                                    g = Int(table[e + tableStep])
                                    b = Int(table[e + 2 * tableStep])
                                } else {
                                    (r, g, b) = (0, 0, 0)
                                }
                            default:
                                r = Int(base[p + c0])
                                (g, b) = (r, r)
                            }
                            if premultiplied {
                                let a = Int(base[p + a0])
                                (r, g, b) = (
                                    unpremultiplied(r, a), unpremultiplied(g, a),
                                    unpremultiplied(b, a)
                                )
                            }
                            out[o] = UInt8(r)
                            out[o + 1] = UInt8(g)
                            out[o + 2] = UInt8(b)
                            o += 3
                            p += bytesPerPixel
                            x += 1
                        }
                        y += 1
                    }
                }
            }
        }
        return RGB8Image(width: width, height: height, bytes: bytes)
    }

    /// 8-bit pixels unpacked by vImage. `rgb` holds the byte offsets of the red, green and blue
    /// samples in a pixel (gray repeats one offset). Nil for layouts vImage cannot unpack.
    private static func unpacked(
        _ base: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int, pixelBytes: Int,
        rgb: [Int]
    ) -> RGB8Image? {
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        let error = bytes.withUnsafeMutableBytes { out -> vImage_Error? in
            var source = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: base), height: vImagePixelCount(height),
                width: vImagePixelCount(width), rowBytes: rowBytes)
            var destination = vImage_Buffer(
                data: out.baseAddress, height: vImagePixelCount(height),
                width: vImagePixelCount(width), rowBytes: width * 3)
            let flags = vImage_Flags(kvImageNoFlags)
            switch (pixelBytes, rgb) {
            case (1, [0, 0, 0]):
                return withUnsafePointer(to: source) {
                    vImageConvert_Planar8toRGB888($0, $0, $0, &destination, flags)
                }
            case (3, [0, 1, 2]): return vImageCopyBuffer(&source, &destination, 3, flags)
            case (4, [0, 1, 2]): return vImageConvert_RGBA8888toRGB888(&source, &destination, flags)
            case (4, [1, 2, 3]): return vImageConvert_ARGB8888toRGB888(&source, &destination, flags)
            case (4, [2, 1, 0]): return vImageConvert_BGRA8888toRGB888(&source, &destination, flags)
            default: return nil
            }
        }
        guard error == kvImageNoError else { return nil }
        return RGB8Image(width: width, height: height, bytes: bytes)
    }

    /// Tightly packed RGBA or RGBX rows without the fourth byte.
    private static func droppingAlpha(_ rgba: [UInt8], width: Int, height: Int) throws
        -> RGB8Image
    {
        let pixels = rgba.withUnsafeBufferPointer {
            unpacked(
                $0.baseAddress!, width: width, height: height, rowBytes: width * 4,
                pixelBytes: 4, rgb: [0, 1, 2])
        }
        guard let pixels else {
            throw VLMError.imageProcessingFailure("unsupported pixel format")
        }
        return pixels
    }

    /// Fallback for layouts `storedPixels` cannot read (CMYK, float, Lab, unusual packing): draws
    /// into an 8-bit RGB context, in the image's own colour space when it is RGB and `inOwnSpace`
    /// is set, else in sRGB. Not exact.
    fileprivate static func drawnPixels(_ image: CGImage, inOwnSpace: Bool) throws -> RGB8Image {
        let width = image.width
        let height = image.height
        guard width >= 1, height >= 1 else {
            throw VLMError.imageProcessingFailure("empty image")
        }
        let own = inOwnSpace ? image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } : nil
        let space = own ?? sRGB
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            func context(_ space: CGColorSpace) -> CGContext? {
                CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            }
            // Extended-range spaces (EXR, float TIFF) have no 8-bit context.
            guard let context = context(space) ?? context(sRGB) else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw VLMError.imageProcessingFailure("unsupported pixel format")
        }
        return try droppingAlpha(rgba, width: width, height: height)
    }
}

// MARK: - OrientedImage

/// Decoded pixels as stored, and the EXIF orientation (1...8) that turns them upright as
/// `ImageOps.exif_transpose` does. `resized` reads the stored pixels through the orientation,
/// so a large photo is not rotated at full size first.
struct OrientedImage: Sendable {
    var stored: RGB8Image
    var orientation: Int

    /// Orientations outside 1...8 read as 1, as `exif_transpose` ignores them.
    init(_ stored: RGB8Image, orientation: Int = 1) {
        self.stored = stored
        self.orientation = (1 ... 8).contains(orientation) ? orientation : 1
    }

    /// The upright size.
    var width: Int { orientation >= 5 ? stored.height : stored.width }
    var height: Int { orientation >= 5 ? stored.width : stored.height }

    var upright: RGB8Image { stored.oriented(orientation) }

    /// The upright image resized as `RGB8Image.resized(width:height:filter:)` does.
    func resized(width: Int, height: Int, filter: ResampleFilter) -> RGB8Image {
        stored.resized(width: width, height: height, filter: filter, orientation: orientation)
    }

    /// `.url` reads the file and keeps its EXIF orientation, as `load_image` applies it.
    /// `.ciImage` is read in its colour space, and `.array` is used as given.
    init(_ image: UserInput.Image) throws {
        switch image.source {
        case .url(let url):
            self = try OrientedImage(contentsOf: url)
        case .ciImage(let image):
            self.init(try RGB8Image(ciImage: image))
        case .array(let array):
            self.init(try RGB8Image(array: array))
        }
    }

    /// Reads a file, http(s) or `data:` URL and decodes it with `init(encoded:)`.
    init(contentsOf url: URL) throws {
        let data: Data
        if url.scheme?.lowercased() == "data" {
            // `load_image` decodes everything after the first comma as base64.
            let text = url.absoluteString
            guard let comma = text.firstIndex(of: ","),
                let decoded = Data(
                    base64Encoded: String(text[text.index(after: comma)...]),
                    options: .ignoreUnknownCharacters)
            else {
                throw VLMError.imageProcessingFailure("invalid data URL")
            }
            data = decoded
        } else {
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw VLMError.imageProcessingFailure(
                    "\(url.absoluteString): \(error.localizedDescription)")
            }
        }
        self = try OrientedImage(encoded: data)
    }

    /// Decodes the first frame of encoded image bytes and reads its EXIF orientation. RGB JPEG
    /// renders through Core Image, because ImageIO's raw 4:2:0 samples differ from libjpeg-turbo
    /// by up to about 50 levels; other formats, HEIC included, take the CGImage path.
    init(encoded data: Data) throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            CGImageSourceGetCount(source) > 0
        else {
            throw VLMError.imageProcessingFailure("unsupported or damaged image data")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        // By signature: ImageIO can report MPO and Ultra HDR files as another type.
        let isJPEG = data.starts(with: [0xFF, 0xD8, 0xFF])
        let model = properties?[kCGImagePropertyColorModel] as? String
        if isJPEG && model == kCGImagePropertyColorModelRGB as String,
            let image = CIImage(data: data, options: [.colorSpace: NSNull()])
        {
            var pixels = try RGB8Image(rendering: image)
            // A CGImage made without caching gives the file's colour space without decoding.
            let space = CGImageSourceCreateImageAtIndex(source, 0, nil)?.colorSpace
            SRGBConversion(from: space)?.apply(to: &pixels)
            self.init(pixels, orientation: orientation)
            return
        }
        let options = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, options) else {
            throw VLMError.imageProcessingFailure("unsupported or damaged image data")
        }
        self.init(try RGB8Image(cgImage: image), orientation: orientation)
    }
}

/// Straight colour from premultiplied: `round(c * 255 / a)`, capped at 255; 0 where alpha is 0.
private func unpremultiplied(_ c: Int, _ a: Int) -> Int {
    a == 0 ? 0 : min(255, (c * 255 + a / 2) / a)
}

private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

/// Renders `CIImage`s with colour management off, or in sRGB. CIContext is thread-safe and
/// costly to make.
private func makeRenderContext(_ space: CGColorSpace?) -> CIContext {
    CIContext(options: [
        .workingColorSpace: space ?? NSNull(), .outputColorSpace: space ?? NSNull(),
        .workingFormat: NSNumber(value: CIFormat.RGBAf.rawValue),
        .useSoftwareRenderer: true, .cacheIntermediates: false,
    ])
}

#if compiler(>=6.2)  // proxy check for macOS 26 SDK, where CIContext is Sendable
private let renderContext = makeRenderContext(nil)
private let sRGBRenderContext = makeRenderContext(sRGB)
#else
nonisolated(unsafe) private let renderContext = makeRenderContext(nil)
nonisolated(unsafe) private let sRGBRenderContext = makeRenderContext(sRGB)
#endif

// MARK: - SRGBConversion

/// Converts 8-bit RGB to sRGB from another RGB colour space, in place with vImage. CoreGraphics
/// draws what vImage cannot convert in place, such as HLG and PQ.
struct SRGBConversion {
    /// RGB spaces that move no colour by more than this many levels read as sRGB. Some sRGB
    /// profiles use gamma 2.2, which is up to 9 levels from the sRGB curve.
    static let tolerance = 10

    private let space: CGColorSpace
    private let converter: vImageConverter?

    /// Nil when samples in `space` are kept as stored: no space, gray, CMYK, or an RGB space
    /// within `tolerance` levels of sRGB. A palette converts from its base space.
    init?(from space: CGColorSpace?) {
        guard var space else { return nil }
        if space.model == .indexed, let base = space.baseColorSpace { space = base }
        guard space.model == .rgb, !CFEqual(space, sRGB) else { return nil }
        self.space = space
        if let source = Self.format(space), let destination = Self.format(sRGB) {
            converter = try? vImageConverter.make(
                sourceFormat: source, destinationFormat: destination)
        } else {
            converter = nil
        }
        let probe = Self.probe
        var converted = probe
        guard apply(to: &converted),
            zip(converted.bytes, probe.bytes).contains(where: {
                abs(Int($0) - Int($1)) > Self.tolerance
            })
        else { return nil }
    }

    /// 16 levels per channel, then a gray ramp.
    private static let probe: RGB8Image = {
        let levels = (0 ..< 16).map { UInt8($0 * 17) }
        var bytes: [UInt8] = []
        for r in levels {
            for g in levels {
                for b in levels { bytes += [r, g, b] }
            }
        }
        for value in 0 ... 255 { bytes += [UInt8(value), UInt8(value), UInt8(value)] }
        return RGB8Image(width: bytes.count / 3, height: 1, bytes: bytes)
    }()

    private static func format(_ space: CGColorSpace) -> vImage_CGImageFormat? {
        vImage_CGImageFormat(
            bitsPerComponent: 8, bitsPerPixel: 24, colorSpace: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue))
    }

    /// Converts `image`. False, and `image` unchanged, when neither vImage nor CoreGraphics can.
    @discardableResult
    func apply(to image: inout RGB8Image) -> Bool {
        let (width, height) = (image.width, image.height)
        guard width >= 1, height >= 1 else { return true }
        if let converter {
            let converted = image.bytes.withUnsafeMutableBytes { bytes -> Bool in
                var pixels = vImage_Buffer(
                    data: bytes.baseAddress, height: vImagePixelCount(height),
                    width: vImagePixelCount(width), rowBytes: width * 3)
                let source = pixels
                guard
                    (try? converter.mustOperateOutOfPlace(source: source, destination: pixels))
                        == false
                else { return false }
                return (try? converter.convert(source: source, destination: &pixels)) != nil
            }
            if converted { return true }
        }
        let provider = CGDataProvider(data: Data(image.bytes) as CFData)
        guard let provider,
            let cgImage = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: width * 3, space: space, bitmapInfo: [], provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let drawn = try? RGB8Image.drawnPixels(cgImage, inOwnSpace: false)
        else { return false }
        image = drawn
        return true
    }
}

// MARK: Geometry

extension RGB8Image {
    /// The upright image for an EXIF orientation (1...8), as `ImageOps.exif_transpose` makes it.
    func oriented(_ exifOrientation: Int) -> RGB8Image {
        guard (2 ... 8).contains(exifOrientation) else { return self }
        let (outWidth, outHeight) = exifOrientation >= 5 ? (height, width) : (width, height)
        let (start, columnStep, rowStep) = orientationSteps(exifOrientation)
        // Orientations 5...8 read one pixel per stored row. Tiles keep those reads in the cache.
        let (tileWidth, tileHeight) = exifOrientation >= 5 ? (32, 32) : (outWidth, outHeight)
        var out = [UInt8](repeating: 0, count: bytes.count)
        bytes.withUnsafeBufferPointer { source in
            out.withUnsafeMutableBufferPointer { out in
                guard let src = source.baseAddress, let dst = out.baseAddress else { return }
                var tileY = 0
                while tileY < outHeight {
                    let yEnd = min(tileY + tileHeight, outHeight)
                    var tileX = 0
                    while tileX < outWidth {
                        let xEnd = min(tileX + tileWidth, outWidth)
                        var yo = tileY
                        while yo < yEnd {
                            var s = (start + yo * rowStep + tileX * columnStep) * 3
                            var d = (yo * outWidth + tileX) * 3
                            var xo = tileX
                            while xo < xEnd {
                                dst[d] = src[s]
                                dst[d + 1] = src[s + 1]
                                dst[d + 2] = src[s + 2]
                                d += 3
                                s += columnStep * 3
                                xo += 1
                            }
                            yo += 1
                        }
                        tileX = xEnd
                    }
                    tileY = yEnd
                }
            }
        }
        return RGB8Image(width: outWidth, height: outHeight, bytes: out)
    }

    /// For an EXIF orientation (1...8): the stored pixel at upright (0, 0), and the stored
    /// pixel steps per upright column and row.
    func orientationSteps(_ exifOrientation: Int) -> (start: Int, columnStep: Int, rowStep: Int) {
        let (w, h) = (width, height)
        switch exifOrientation {
        case 2: return (w - 1, -1, w)
        case 3: return (h * w - 1, -1, -w)
        case 4: return ((h - 1) * w, 1, -w)
        case 5: return (0, w, 1)
        case 6: return ((h - 1) * w, -w, 1)
        case 7: return (h * w - 1, -w, -1)
        case 8: return (w - 1, w, -1)
        default: return (0, 1, w)
        }
    }

    /// The `width` x `height` region at (`x`, `y`). Pixels outside the image are black.
    func cropped(x: Int, y: Int, width cropWidth: Int, height cropHeight: Int) -> RGB8Image {
        precondition(cropWidth >= 0 && cropHeight >= 0, "negative crop size")
        var out = [UInt8](repeating: 0, count: cropWidth * cropHeight * 3)
        let x0 = max(x, 0)
        let x1 = min(x + cropWidth, width)
        let y0 = max(y, 0)
        let y1 = min(y + cropHeight, height)
        guard x0 < x1, y0 < y1 else {
            return RGB8Image(width: cropWidth, height: cropHeight, bytes: out)
        }
        bytes.withUnsafeBytes { source in
            out.withUnsafeMutableBytes { out in
                for row in y0 ..< y1 {
                    (out.baseAddress! + ((row - y) * cropWidth + x0 - x) * 3).copyMemory(
                        from: source.baseAddress! + (row * width + x0) * 3,
                        byteCount: (x1 - x0) * 3)
                }
            }
        }
        return RGB8Image(width: cropWidth, height: cropHeight, bytes: out)
    }
}
