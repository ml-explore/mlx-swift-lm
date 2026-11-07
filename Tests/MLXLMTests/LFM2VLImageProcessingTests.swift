// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXScriptedLM
import Testing

@testable import MLXVLM

// Expected values come from tools/lfm2_vl_test_vectors.py: transformers' `Lfm2VlImageProcessor`
// and `Lfm2VlProcessor` token expansion.

/// The LFM2-VL chat layout for scripted tests: `<|startoftext|>`, the tools as JSON in a system
/// turn, ChatML turns, and `<image>` for each image part. The tokens in `omitting` are left out
/// of the vocabulary.
struct LFM2VLTestTemplate: ScriptedChatTemplate {
    var omitting: Set<String> = []

    var specialTokens: [String] {
        var tokens = [
            "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<image>", "<|image_start|>",
            "<|image_end|>", "<|img_thumbnail|>",
        ]
        // The grids of up to 10 tiles, the ones the processor looks up.
        for row in 1 ... 10 {
            for column in 1 ... 10 / row {
                tokens.append("<|img_row_\(row)_col_\(column)|>")
            }
        }
        return tokens.filter { !omitting.contains($0) }
    }
    var corpus: [String] { ["user\n", "assistant\n", "\n"] }
    var endOfTurnToken: String? { "<|im_end|>" }

    func render(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String] {
        var segments = ["<|startoftext|>"]
        if let tools {
            segments += [
                "<|im_start|>", "system\n", try ScriptedTemplateMessage.canonicalJSON(tools),
                "<|im_end|>", "\n",
            ]
        }
        for message in messages {
            segments += ["<|im_start|>", "\(message["role"] as? String ?? "")\n"]
            if let parts = message["content"] as? [[String: String]] {
                segments += parts.map { $0["type"] == "image" ? "<image>" : $0["text"] ?? "" }
            } else {
                segments.append(ScriptedTemplateMessage.content(of: message))
            }
            if let calls = message["tool_calls"] {
                segments.append(try ScriptedTemplateMessage.canonicalJSON(calls))
            }
            segments += ["<|im_end|>", "\n"]
        }
        return segments + ["<|im_start|>", "assistant\n"]
    }
}

/// A scripted tokenizer with the LFM2-VL markers, and the ids of those markers.
enum LFM2VLTestTokens {
    static let tokenizer = make()

    static func make(omitting: Set<String> = []) -> ScriptedTokenizer {
        ScriptedTokenizer(
            corpus: ["Hi", "A", "B", "C"], template: LFM2VLTestTemplate(omitting: omitting))
    }

    static let image = id("<image>")
    static let thumbnail = id("<|img_thumbnail|>")
    static let imageStart = id("<|image_start|>")
    static let imageEnd = id("<|image_end|>")

    static func tile(row: Int, column: Int) -> Int {
        id("<|img_row_\(row)_col_\(column)|>")
    }

    static func id(_ token: String) -> Int {
        tokenizer.convertTokenToId(token)!
    }

    /// The token run transformers writes for a single view of `tokens` image tokens.
    static func singleViewRun(_ tokens: Int) -> [Int] {
        [imageStart] + Array(repeating: image, count: tokens) + [imageEnd]
    }

    /// The token run transformers writes for a tiled image with a thumbnail.
    static func tiledRun(columns: Int, rows: Int, thumbnailTokens: Int) -> [Int] {
        var run = [imageStart]
        for row in 1 ... rows {
            for column in 1 ... columns {
                run.append(tile(row: row, column: column))
                run += Array(repeating: image, count: 256)
            }
        }
        run.append(thumbnail)
        run += Array(repeating: image, count: thumbnailTokens)
        return run + [imageEnd]
    }
}

@Suite struct LFM2VLImageLayoutTests {
    // ((width, height), tiled, columns, rows, thumbnailWidth, thumbnailHeight) from
    // `Lfm2VlImageProcessor.smart_resize`, `_is_image_too_large` and `_get_grid_layout`,
    // including 400x600 (half-to-even rounding).
    static let defaultCases: [((Int, Int), Bool, Int, Int, Int, Int)] = [
        ((640, 480), false, 1, 1, 576, 416),
        ((64, 48), false, 1, 1, 320, 224),
        ((300, 1400), false, 1, 1, 224, 1088),
        ((1, 1), false, 1, 1, 256, 256),
        ((2000, 1200), true, 3, 2, 640, 384),
        ((3000, 260), true, 10, 1, 1728, 128),
        ((1000, 1000), true, 2, 2, 512, 512),
        ((1100, 1100), true, 3, 3, 512, 512),
        ((48, 1040), false, 1, 1, 64, 1024),
        ((1250, 1000), true, 3, 3, 544, 448),
        ((2000, 1000), true, 4, 2, 704, 352),
        ((1920, 1080), true, 4, 2, 672, 384),
        ((8000, 20), false, 1, 1, 8000, 32),
        ((48, 80), false, 1, 1, 224, 352),
        ((1536, 1024), true, 3, 2, 608, 416),
        ((400, 600), false, 1, 1, 384, 608),
        ((1024, 768), true, 3, 2, 576, 416),
        ((1182, 886), true, 3, 2, 576, 416),
        ((886, 1182), true, 2, 3, 416, 576),
        ((686, 1526), true, 2, 5, 320, 736),
        ((2560, 1440), true, 4, 2, 672, 384),
        ((960, 540), false, 1, 1, 672, 384),
    ]

    @Test func layoutMatchesTransformers() throws {
        let settings = LFM2VLImageSettings()
        for ((width, height), tiled, columns, rows, thumbnailWidth, thumbnailHeight) in Self
            .defaultCases
        {
            let layout = try LFM2VLImageLayout.plan(
                width: width, height: height, settings: settings)
            #expect(
                layout
                    == LFM2VLImageLayout(
                        tiled: tiled, gridColumns: columns, gridRows: rows,
                        thumbnailWidth: thumbnailWidth, thumbnailHeight: thumbnailHeight,
                        includesThumbnail: true),
                "\(width)x\(height)")
        }
    }

    @Test func layoutFollowsOtherSettings() throws {
        var settings = LFM2VLImageSettings()
        settings.maxImageTokens = 128
        settings.minImageTokens = 32
        settings.maxPixelsTolerance = 1.5
        let cases: [((Int, Int), Int, Int, Int, Int)] = [
            ((640, 480), 3, 2, 416, 288),
            ((1000, 1000), 2, 2, 352, 352),
            ((700, 300), 5, 2, 544, 224),
            ((300, 700), 2, 5, 224, 544),
            ((3000, 260), 10, 1, 1216, 96),
            ((512, 512), 2, 2, 352, 352),
            ((1100, 1100), 3, 3, 352, 352),
        ]
        for ((width, height), columns, rows, thumbnailWidth, thumbnailHeight) in cases {
            #expect(
                try LFM2VLImageLayout.plan(width: width, height: height, settings: settings)
                    == LFM2VLImageLayout(
                        tiled: true, gridColumns: columns, gridRows: rows,
                        thumbnailWidth: thumbnailWidth, thumbnailHeight: thumbnailHeight,
                        includesThumbnail: true),
                "\(width)x\(height)")
        }
    }

    @Test func layoutRejectsViewsThatDoNotFit() throws {
        #expect(throws: VLMError.self) {
            try LFM2VLImageLayout.plan(width: 0, height: 10, settings: LFM2VLImageSettings())
        }
        // A 32x8864 thumbnail needs 1108 patches, more than the 1024 rows of a view.
        #expect(throws: VLMError.self) {
            try LFM2VLImageLayout.plan(width: 9000, height: 30, settings: LFM2VLImageSettings())
        }
    }

    @Test func targetRatiosSortByTileCountThenColumns() {
        // transformers' `_target_ratios(min, max)` grids as "columns x rows".
        let cases: [((Int, Int), String)] = [
            (
                (2, 10),
                "1x2 2x1 1x3 3x1 1x4 2x2 4x1 1x5 5x1 1x6 2x3 3x2 6x1 1x7 7x1 1x8 2x4 4x2 8x1 1x9 "
                    + "3x3 9x1 1x10 2x5 5x2 10x1"
            ),
            ((1, 4), "1x1 1x2 2x1 1x3 3x1 1x4 2x2 4x1"),
            ((6, 6), "1x6 2x3 3x2 6x1"),
            ((1, 1), "1x1"),
        ]
        for ((minTiles, maxTiles), expected) in cases {
            let ratios = LFM2VLImageSettings.targetRatios(minTiles: minTiles, maxTiles: maxTiles)
            #expect(
                ratios.map { "\($0.0)x\($0.1)" }.joined(separator: " ") == expected,
                "\(minTiles)...\(maxTiles)")
        }
        #expect(LFM2VLImageSettings.targetRatios(minTiles: 3, maxTiles: 2).isEmpty)
    }

    @Test func exactRatioTiesPickTheLaterGrid() throws {
        // Halfway between two grids, a large image takes the later one. In these narrow ranges
        // transformers 5.19 picks 2x2 for 5000x2000 and 1x5 for 6500x2500, from Python's set order.
        let cases: [((Int, Int), (Int, Int), (Int, Int))] = [
            ((4, 4), (5000, 2000), (4, 1)),
            ((4, 4), (2000, 5000), (1, 4)),
            ((5, 5), (6500, 2500), (5, 1)),
        ]
        for ((minTiles, maxTiles), (width, height), (columns, rows)) in cases {
            var settings = LFM2VLImageSettings()
            settings.minTiles = minTiles
            settings.maxTiles = maxTiles
            let layout = try LFM2VLImageLayout.plan(
                width: width, height: height, settings: settings)
            #expect(layout.tiled, "\(width)x\(height)")
            let grid = [layout.gridColumns, layout.gridRows]
            #expect(grid == [columns, rows], "\(width)x\(height)")
        }
    }

    @Test func cropsAreTilesRowMajorThenThumbnail() {
        let layout = LFM2VLImageLayout(
            tiled: true, gridColumns: 3, gridRows: 2, thumbnailWidth: 5, thumbnailHeight: 4,
            includesThumbnail: true)
        var requested: [[Int]] = []
        let crops = layout.crops(of: RGB8Image.testPattern(width: 7, height: 9), tileSize: 8) {
            _, width, height in
            requested.append([width, height])
            return RGB8Image.testPattern(width: width, height: height)
        }
        #expect(requested == [[24, 16], [5, 4]])
        #expect(crops.count == 7)
        let big = RGB8Image.testPattern(width: 24, height: 16)
        #expect(crops[1] == big.cropped(x: 8, y: 0, width: 8, height: 8))
        #expect(crops[3] == big.cropped(x: 0, y: 8, width: 8, height: 8))
        #expect(crops[6] == RGB8Image.testPattern(width: 5, height: 4))

        var single = layout
        single.tiled = false
        #expect(
            single.crops(of: RGB8Image.testPattern(width: 7, height: 9), tileSize: 8) {
                $0.resized(width: $1, height: $2, filter: .bicubic)
            }.count == 1)
    }
}

@Suite struct LFM2VLImageSettingsTests {
    /// Bytes and their float32 bit patterns after transformers' fused `rescale_and_normalize`,
    /// `(x - 127.5) / 127.5`, with mean and std 0.5.
    static let normalizedBytes: [UInt8] = [0, 1, 77, 127, 128, 200, 254, 255]
    static let halfBits: [UInt32] = [
        0xbf80_0000, 0xbf7d_fdfe, 0xbeca_cacb, 0xbb80_8081, 0x3b80_8081, 0x3f11_9192,
        0x3f7d_fdfe, 0x3f80_0000,
    ]

    @Test func decodesNestedAndFlatConfigs() throws {
        // LFM2.5-VL: everything under `image_processor`.
        let nested = """
            {"image_processor": {"do_image_splitting": false, "min_tiles": 4, "max_tiles": 6,
             "use_thumbnail": false, "max_num_patches": 256, "resample": 2, "tile_size": 384,
             "image_mean": [0.25, 0.5, 0.75], "image_std": 0.5, "max_pixels_tolerance": 1.5},
             "processor_class": "Lfm2VlProcessor"}
            """
        let settings = try JSONDecoder().decode(LFM2VLImageSettings.self, from: Data(nested.utf8))
        var expected = LFM2VLImageSettings()
        expected.resample = .bilinear
        expected.tileSize = 384
        expected.imageMean = [0.25, 0.5, 0.75]
        expected.imageStd = [0.5]
        expected.maxPixelsTolerance = 1.5
        expected.doImageSplitting = false
        expected.minTiles = 4
        expected.maxTiles = 6
        expected.useThumbnail = false
        #expect(settings == expected)
        #expect(settings.maxNumPatches == 1024)
        #expect(settings.tileRange.min == 1 && settings.tileRange.max == 1)

        // Without splitting, a big tile does not raise `max_num_patches` (1024 from the
        // generator), as in `_preprocess`.
        var bigTiles = LFM2VLImageSettings()
        bigTiles.tileSize = 1024
        #expect(bigTiles.maxNumPatches == 4096)
        bigTiles.doImageSplitting = false
        #expect(bigTiles.maxNumPatches == 1024)
    }

    @Test func resampleDefaultsToBilinear() throws {
        func resample(_ json: String) throws -> ResampleFilter {
            try JSONDecoder().decode(LFM2VLImageSettings.self, from: Data(json.utf8)).resample
        }
        // `Lfm2VlImageProcessor().resample` is 2.
        #expect(try resample("{}") == .bilinear)
        #expect(try resample(#"{"resample": null}"#) == .bilinear)
        #expect(try resample(#"{"resample": 3}"#) == .bicubic)
        #expect(try resample(#"{"resample": 2.0}"#) == .bilinear)
        for unsupported in ["0", "1", "4", "9", #""2""#, #""nonsense""#] {
            #expect(throws: DecodingError.self, "\(unsupported)") {
                try resample(#"{"resample": \#(unsupported)}"#)
            }
        }
    }

    @Test func normalizationTableMatchesTransformers() {
        let table = LFM2VLImageSettings().normalizationTable()
        for channel in 0 ..< 3 {
            #expect(
                Self.normalizedBytes.map { table[channel * 256 + Int($0)].bitPattern }
                    == Self.halfBits)
        }
    }
}

@Suite struct LFM2VLImageProcessingTests {

    /// Test sizes, including common photo sizes, with transformers' spatial shapes and token
    /// counts: ((width, height), (columns, rows), thumbnail shape, image tokens, run length).
    static let assetCases: [((Int, Int), (Int, Int), [Int], Int, Int)] = [
        ((640, 480), (1, 1), [26, 36], 234, 236),
        ((1024, 768), (3, 2), [26, 36], 1770, 1779),
        ((1182, 886), (3, 2), [26, 36], 1770, 1779),
        ((886, 1182), (2, 3), [36, 26], 1770, 1779),
        ((400, 600), (1, 1), [38, 24], 228, 230),
        ((686, 1526), (2, 5), [46, 20], 2790, 2803),
        ((1600, 1200), (3, 2), [26, 36], 1770, 1779),
        ((720, 1600), (2, 4), [46, 20], 2278, 2289),
        ((3024, 4032), (2, 3), [36, 26], 1770, 1779),
        ((1000, 1000), (2, 2), [32, 32], 1280, 1287),
        ((64, 48), (1, 1), [14, 20], 70, 72),
        ((1, 1), (1, 1), [16, 16], 64, 66),
        ((3000, 260), (10, 1), [8, 108], 2776, 2789),
    ]

    @Test func assetLayoutsAndTokenRuns() throws {
        let processing = try LFM2VLImageProcessing(
            settings: LFM2VLImageSettings(), tokenizer: LFM2VLTestTokens.tokenizer)
        for ((width, height), (columns, rows), thumbnail, tokens, length) in Self.assetCases {
            let layout = try LFM2VLImageLayout.plan(
                width: width, height: height, settings: processing.settings)
            #expect(layout.gridColumns == columns && layout.gridRows == rows, "\(width)x\(height)")
            #expect(
                [layout.thumbnailHeight / 16, layout.thumbnailWidth / 16] == thumbnail,
                "\(width)x\(height)")

            let run = try processing.tokenRun(layout)
            #expect(run.count == length, "\(width)x\(height)")
            #expect(run.filter { $0 == LFM2VLTestTokens.image }.count == tokens)
            let thumbnailTokens = ((thumbnail[0] + 1) / 2) * ((thumbnail[1] + 1) / 2)
            if columns * rows > 1 {
                #expect(
                    run
                        == LFM2VLTestTokens.tiledRun(
                            columns: columns, rows: rows, thumbnailTokens: thumbnailTokens),
                    "\(width)x\(height)")
            } else {
                #expect(
                    run == LFM2VLTestTokens.singleViewRun(thumbnailTokens), "\(width)x\(height)")
            }
        }
    }

    @Test func viewsMatchTheLayout() throws {
        let processing = try LFM2VLImageProcessing(
            settings: LFM2VLImageSettings(), tokenizer: LFM2VLTestTokens.tokenizer)
        try Device.withDefaultDevice(.cpu) { () throws in
            for ((width, height), (columns, rows), thumbnail, _, _) in Self.assetCases
            where width * height <= 1024 * 768 {
                let output = try processing.process([
                    RGB8Image.testPattern(width: width, height: height)
                ]).patches
                let tiles =
                    columns * rows > 1 ? Array(repeating: [32, 32], count: columns * rows) : []
                #expect(output.spatialShapes.map { [$0.0, $0.1] } == tiles + [thumbnail])
                #expect(output.pixels.shape == [tiles.count + 1, 1024, 768])
                #expect(
                    output.mask.sum(axis: 1).asArray(Int32.self)
                        == (tiles + [thumbnail]).map { Int32($0[0] * $0[1]) })
            }
        }
    }

    @Test func imagesAreReleasedBeforeTheNextIsDecoded() throws {
        let processing = try LFM2VLImageProcessing(
            settings: LFM2VLImageSettings(), tokenizer: LFM2VLTestTokens.tokenizer)
        try Device.withDefaultDevice(.cpu) { () throws in
            // `last` shares the bytes of the image decoded last. Writing to it copies them
            // only while something else still holds that image.
            var last: [UInt8] = []
            var held: [Bool] = []
            let decode = { (side: Int) -> OrientedImage in
                if !last.isEmpty {
                    let before = last.withUnsafeBufferPointer { $0.baseAddress }
                    last.withUnsafeMutableBufferPointer { _ in }
                    held.append(last.withUnsafeBufferPointer { $0.baseAddress } != before)
                }
                let image = RGB8Image.testPattern(width: side, height: side)
                last = image.bytes
                return OrientedImage(image)
            }
            _ = try processing.process([64, 80, 96], decode: decode)
            #expect(held == [false, false])

            // Decoding every image first keeps them all.
            (last, held) = ([], [])
            _ = try processing.process([64, 80, 96].map(decode)) { (image: OrientedImage) in image }
            #expect(held == [true, true])
        }
    }

    /// sha256 of transformers' float32 `pixel_values` (`Lfm2VlImageProcessor`) for the test
    /// pattern, for bicubic (LFM2-VL-450M) and bilinear (LFM2.5-VL-450M).
    static let pixelHashes: [(ResampleFilter, (Int, Int), String)] = [
        (
            .bicubic, (64, 48),
            "e07eb7d39b5027309f81ae1b714344713fabd721109c51402d98d9fe2df366d0"
        ),
        (
            .bicubic, (400, 600),
            "2901853fbedb2300ff6164665f73508290642a759dde67b8c8174346c743a88b"
        ),
        (
            .bicubic, (1024, 768),
            "39e4942bb04d4373a735c3c80a554c7ba040d7c08f8ca5f9706528789c3403ca"
        ),
        (
            .bilinear, (64, 48),
            "657bae734362becde1a9e808bc7bcdd6e419e2cc33b455edcaf912e159a77235"
        ),
        (
            .bilinear, (400, 600),
            "1db801803f628c81b2872344ee64e3558e4b90b25e47847a3083ce9f36d86614"
        ),
        (
            .bilinear, (1024, 768),
            "89cd469414abb4b21b994679ede2071c72876ac210c76e97c0a889c12ea5ab6f"
        ),
    ]

    @Test func pixelsMatchTransformersBitForBit() throws {
        try Device.withDefaultDevice(.cpu) { () throws in
            for (resample, (width, height), expected) in Self.pixelHashes {
                var settings = LFM2VLImageSettings()
                settings.resample = resample
                let processing = try LFM2VLImageProcessing(
                    settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
                let output = try processing.process([
                    RGB8Image.testPattern(width: width, height: height)
                ]).patches
                #expect(sha256Hex(output.pixels) == expected, "\(resample) \(width)x\(height)")
            }
        }
    }

    #if canImport(Metal)
    @Test(.enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
    func pixelsMatchTransformersWithTheGPUResize() throws {
        for (resample, (width, height), expected) in Self.pixelHashes where width >= 400 {
            var settings = LFM2VLImageSettings()
            settings.resample = resample
            let processing = try LFM2VLImageProcessing(
                settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
            let image = RGB8Image.testPattern(width: width, height: height)
            let layout = try LFM2VLImageLayout.plan(
                width: width, height: height, settings: settings)
            let passes = image.resizePasses(
                width: layout.thumbnailWidth, height: layout.thumbnailHeight, filter: resample)
            #expect(image.prefersGPU(passes, orientation: 1))
            let output = try processing.process([image]).patches
            #expect(sha256Hex(output.pixels) == expected, "\(resample) \(width)x\(height)")
        }
    }
    #endif

    /// Processes a landscape image with each EXIF orientation and checks it against the same
    /// image turned upright first. Orientations 5 to 8 make it portrait, with another grid.
    private func expectOrientationsMatchUprightImages(onGPU: Bool = false) throws {
        let settings = LFM2VLImageSettings()
        let processing = try LFM2VLImageProcessing(
            settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
        let stored = RGB8Image.testPattern(width: 1024, height: 768)
        for orientation in 1 ... 8 {
            let image = OrientedImage(stored, orientation: orientation)
            let upright = stored.oriented(orientation)
            let label = "orientation \(orientation)"
            #expect(image.width == upright.width && image.height == upright.height, "\(label)")
            #expect((image.width < image.height) == (orientation >= 5), "\(label)")
            #if canImport(Metal)
            if onGPU {
                let layout = try LFM2VLImageLayout.plan(
                    width: image.width, height: image.height, settings: settings)
                let sizes = [
                    (layout.gridColumns * settings.tileSize, layout.gridRows * settings.tileSize),
                    (layout.thumbnailWidth, layout.thumbnailHeight),
                ]
                for (width, height) in sizes {
                    let passes = stored.resizePasses(
                        width: width, height: height, filter: settings.resample,
                        orientation: orientation)
                    #expect(stored.prefersGPU(passes, orientation: orientation), "\(label)")
                }
            }
            #endif
            let output = try processing.process([image]) { (image: OrientedImage) in image }
            let expected = try processing.process([upright])
            #expect(output.tokenRuns == expected.tokenRuns, "\(label)")
            #expect(
                output.patches.spatialShapes.map { [$0.0, $0.1] }
                    == expected.patches.spatialShapes.map { [$0.0, $0.1] }, "\(label)")
            #expect(
                arrayEqual(output.patches.pixels, expected.patches.pixels).item(Bool.self),
                "\(label)")
        }
    }

    @Test func orientationsMatchUprightImages() throws {
        try Device.withDefaultDevice(.cpu) { () throws in
            try expectOrientationsMatchUprightImages()
        }
    }

    #if canImport(Metal)
    @Test(.enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
    func orientationsMatchUprightImagesWithTheGPUResize() throws {
        try expectOrientationsMatchUprightImages(onGPU: true)
    }
    #endif

    @Test func normalizationMatchesTransformers() throws {
        // transformers on uniform 16x16 images, as float32 bit patterns per channel.
        let bytes = LFM2VLImageSettingsTests.normalizedBytes
        let half = LFM2VLImageSettingsTests.halfBits
        let clip: [[UInt32]] = [
            [
                0xbfe5_68dc, 0xbfe3_8a80, 0xbf2b_0e17, 0x3d7c_e08c, 0x3d9c_560e, 0x3f90_4f63,
                0x3ff5_36e6, 0x3ff7_1542,
            ],
            [
                0xbfe0_44b7, 0xbfde_58f1, 0xbf18_b428, 0x3e1d_952e, 0x3e2c_f360, 0x3f9f_ee30,
                0x4003_d501, 0x4004_cae4,
            ],
            [
                0xbfbd_77d8, 0xbfbb_a5e2, 0xbec5_42ba, 0x3ea6_c5e7, 0x3eae_0dc1, 0x3fae_90c9,
                0x4008_6d66, 0x4009_5661,
            ],
        ]
        var clipSettings = LFM2VLImageSettings()
        clipSettings.imageMean = [0.48145466, 0.4578275, 0.40821073]
        clipSettings.imageStd = [0.26862954, 0.26130258, 0.27577711]

        try Device.withDefaultDevice(.cpu) { () throws in
            for (settings, expected) in [
                (LFM2VLImageSettings(), [half, half, half]), (clipSettings, clip),
            ] {
                let processing = try LFM2VLImageProcessing(
                    settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
                // Uniform crops stay uniform through the resize, so patch 0 holds the value.
                let images = bytes.map {
                    RGB8Image(width: 16, height: 16, bytes: Array(repeating: $0, count: 768))
                }
                let pixels = try processing.process(images).patches.pixels
                let first = pixels[0..., 0, 0 ..< 3].asArray(Float.self).map(\.bitPattern)
                for channel in 0 ..< 3 {
                    #expect(
                        (0 ..< bytes.count).map { first[$0 * 3 + channel] } == expected[channel])
                }
            }
        }
    }

    @Test func paddingAndMask() throws {
        try Device.withDefaultDevice(.cpu) { () throws in
            let images = [
                RGB8Image.testPattern(width: 64, height: 48),
                RGB8Image.testPattern(width: 1, height: 1),
            ]
            let padded = try LFM2VLImageProcessing(
                settings: LFM2VLImageSettings(), tokenizer: LFM2VLTestTokens.tokenizer
            ).process(images).patches
            #expect(padded.pixels.shape == [2, 1024, 768])
            #expect(padded.spatialShapes.map { [$0.0, $0.1] } == [[14, 20], [16, 16]])
            func row(ones: Int) -> [Int32] {
                [Int32](repeating: 1, count: ones) + [Int32](repeating: 0, count: 1024 - ones)
            }
            #expect(padded.mask.asArray(Int32.self) == row(ones: 280) + row(ones: 256))
            #expect(abs(padded.pixels[0, 280...]).sum().item(Float.self) == 0)
            #expect(abs(padded.pixels[1, 256...]).sum().item(Float.self) == 0)

            // Without `do_pad`, views pad to the longest one.
            var settings = LFM2VLImageSettings()
            settings.doPad = false
            let unpadded = try LFM2VLImageProcessing(
                settings: settings, tokenizer: LFM2VLTestTokens.tokenizer
            ).process(images).patches
            #expect(unpadded.pixels.shape == [2, 280, 768])
            #expect(unpadded.mask.sum(axis: 1).asArray(Int32.self) == [280, 256])
            #expect(
                abs(unpadded.pixels - padded.pixels[0..., 0 ..< 280]).max().item(Float.self) == 0)
        }
    }

    @Test func expandReplacesEachPlaceholderInOrder() throws {
        let processing = try LFM2VLImageProcessing(
            settings: LFM2VLImageSettings(), tokenizer: LFM2VLTestTokens.tokenizer)
        let image = LFM2VLTestTokens.image
        let expanded = try processing.expand(
            [1, image, image, 5, image], tokenRuns: [[10], [11, 12], [13]])
        #expect(expanded == [1, 10, 11, 12, 5, 13])
        #expect(throws: VLMError.self) {
            try processing.expand([1, image, image], tokenRuns: [[10]])
        }
        #expect(throws: VLMError.self) {
            try processing.expand([1, image], tokenRuns: [[10], [11]])
        }
    }

    @Test func tilingKeysWorkAsInTransformers() throws {
        // (processor config, (width, height), spatial shapes, image tokens, run length).
        let cases: [(String, (Int, Int), [[Int]], Int, Int)] = [
            (#"{"do_image_splitting": false}"#, (1024, 768), [[26, 36]], 234, 236),
            (#"{"do_image_splitting": false}"#, (3000, 260), [[8, 108]], 216, 218),
            (#"{"do_image_splitting": false}"#, (2000, 2000), [[32, 32]], 256, 258),
            (#"{"use_thumbnail": false}"#, (1024, 768), Self.tiles(6), 1536, 1544),
            (#"{"use_thumbnail": false}"#, (3000, 260), Self.tiles(10), 2560, 2572),
            (#"{"use_thumbnail": false}"#, (724, 724), Self.tiles(4), 1024, 1030),
            (#"{"use_thumbnail": false}"#, (64, 48), [[14, 20]], 70, 72),
            (#"{"max_tiles": 4}"#, (1024, 768), Self.tiles(4) + [[26, 36]], 1258, 1265),
            (#"{"max_tiles": 4}"#, (3000, 260), Self.tiles(4) + [[8, 108]], 1240, 1247),
            (#"{"max_tiles": 4}"#, (724, 724), Self.tiles(5), 1280, 1287),
            (#"{"min_tiles": 1, "max_tiles": 4}"#, (724, 724), [[32, 32]], 256, 258),
            (#"{"min_tiles": 1, "max_tiles": 4}"#, (2000, 2000), Self.tiles(5), 1280, 1287),
            (
                #"{"min_tiles": 6, "max_tiles": 6}"#, (1024, 768), Self.tiles(6) + [[26, 36]], 1770,
                1779
            ),
            (#"{"min_tiles": 6, "max_tiles": 6}"#, (2000, 2000), Self.tiles(7), 1792, 1801),
        ]
        for (json, (width, height), shapes, tokens, length) in cases {
            let settings = try JSONDecoder().decode(LFM2VLImageSettings.self, from: Data(json.utf8))
            let processing = try LFM2VLImageProcessing(
                settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
            let layout = try LFM2VLImageLayout.plan(
                width: width, height: height, settings: settings)
            let tile = settings.tileSize / settings.encoderPatchSize
            var views = Array(
                repeating: [tile, tile],
                count: layout.tiled ? layout.gridColumns * layout.gridRows : 0)
            if layout.includesThumbnail {
                views.append([layout.thumbnailHeight / 16, layout.thumbnailWidth / 16])
            }
            let label = "\(json) \(width)x\(height)"
            #expect(views == shapes, "\(label)")
            let run = try processing.tokenRun(layout)
            #expect(run.filter { $0 == LFM2VLTestTokens.image }.count == tokens, "\(label)")
            #expect(run.count == length, "\(label)")
        }

        // sha256 of transformers' `pixel_values` for two of the settings.
        let pixelCases: [(String, (Int, Int), String)] = [
            (
                #"{"use_thumbnail": false}"#, (1000, 1000),
                "2f49bcd243ae65bf8e3b6c52edcefd2db5eda41e11b16e636591a6aedcd340b5"
            ),
            (
                #"{"do_image_splitting": false}"#, (1024, 768),
                "bd21060b9fc8cd4f645153a5ebe8abf9b5939e4b5eb4463c46d6ae3e78bea437"
            ),
        ]
        try Device.withDefaultDevice(.cpu) { () throws in
            for (json, (width, height), expected) in pixelCases {
                let settings = try JSONDecoder().decode(
                    LFM2VLImageSettings.self, from: Data(json.utf8))
                let output = try LFM2VLImageProcessing(
                    settings: settings, tokenizer: LFM2VLTestTokens.tokenizer
                ).process([RGB8Image.testPattern(width: width, height: height)]).patches
                #expect(sha256Hex(output.pixels) == expected, "\(json)")
            }
        }
    }

    private static func tiles(_ count: Int) -> [[Int]] {
        Array(repeating: [32, 32], count: count)
    }

    @Test func badTilingSettingsThrow() throws {
        var settings = LFM2VLImageSettings()
        settings.minTiles = 5
        settings.maxTiles = 4
        #expect(throws: VLMError.self) {
            try LFM2VLImageProcessing(settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
        }
        // `_preprocess` only checks the range when splitting is on.
        settings.doImageSplitting = false
        #expect(throws: Never.self) {
            try LFM2VLImageProcessing(settings: settings, tokenizer: LFM2VLTestTokens.tokenizer)
        }
    }

    @Test func imageMeanAndStdNeedOneOrThreeValues() throws {
        let image = RGB8Image.testPattern(width: 64, height: 48)
        for (mean, std) in [([0.5, 0.5], [0.5]), ([0.5], []), ([0.5], [0.5, 0.5, 0.5, 0.5])] {
            var settings = LFM2VLImageSettings()
            settings.imageMean = mean
            settings.imageStd = std
            #expect(throws: VLMError.self, "\(mean) \(std)") {
                try LFM2VLImageProcessing(
                    settings: settings, tokenizer: LFM2VLTestTokens.tokenizer
                ).process([image])
            }
        }
        var settings = LFM2VLImageSettings()
        settings.imageMean = [0.5]
        settings.imageStd = [0.25, 0.5, 0.75]
        #expect(throws: Never.self) {
            try LFM2VLImageProcessing(
                settings: settings, tokenizer: LFM2VLTestTokens.tokenizer
            ).process([image])
        }
    }

    @Test func missingMarkerTokensThrow() throws {
        for token in ["<image>", "<|image_start|>", "<|img_thumbnail|>"] {
            let tokenizer = LFM2VLTestTokens.make(omitting: [token])
            #expect(throws: VLMError.self, "\(token)") {
                try LFM2VLImageProcessing(settings: LFM2VLImageSettings(), tokenizer: tokenizer)
            }
        }
        // A tile marker is needed only by the grids that use it.
        let tokenizer = LFM2VLTestTokens.make(omitting: ["<|img_row_2_col_2|>"])
        let processing = try LFM2VLImageProcessing(
            settings: LFM2VLImageSettings(), tokenizer: tokenizer)
        let wide = try LFM2VLImageLayout.plan(
            width: 3000, height: 260, settings: processing.settings)
        #expect(throws: Never.self) { try processing.tokenRun(wide) }
        let square = try LFM2VLImageLayout.plan(
            width: 1000, height: 1000, settings: processing.settings)
        #expect(throws: VLMError.self) { try processing.tokenRun(square) }

        // The thumbnail marker is needed only by settings that can tile with a thumbnail.
        let noThumbnail = LFM2VLTestTokens.make(omitting: ["<|img_thumbnail|>"])
        for json in [
            #"{"do_image_splitting": false}"#, #"{"use_thumbnail": false}"#,
            #"{"min_tiles": 1, "max_tiles": 1}"#,
        ] {
            let settings = try JSONDecoder().decode(LFM2VLImageSettings.self, from: Data(json.utf8))
            let processing = try LFM2VLImageProcessing(settings: settings, tokenizer: noThumbnail)
            let layout = try LFM2VLImageLayout.plan(width: 1000, height: 1000, settings: settings)
            #expect(throws: Never.self, "\(json)") { try processing.tokenRun(layout) }
        }
    }
}
