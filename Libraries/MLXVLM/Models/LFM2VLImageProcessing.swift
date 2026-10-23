// Copyright © 2026 Apple Inc.
// Port of transformers' `Lfm2VlImageProcessor` (models/lfm2_vl/image_processing_lfm2_vl.py)
// and the image-token expansion of `Lfm2VlProcessor` (processing_lfm2_vl.py).

import Foundation
import MLX
import MLXLMCommon

// MARK: - Settings

/// LFM2-VL image processor settings, with the defaults of transformers' `Lfm2VlImageProcessor`.
struct LFM2VLImageSettings: Equatable, Sendable {
    var tileSize = 512
    var encoderPatchSize = 16
    var downsampleFactor = 2
    var minImageTokens = 64
    var maxImageTokens = 256
    var maxPixelsTolerance = 2.0
    var doImageSplitting = true
    var minTiles = 2
    var maxTiles = 10
    var useThumbnail = true
    var resample = ResampleFilter.bilinear
    var imageMean: [Double] = [0.5, 0.5, 0.5]
    var imageStd: [Double] = [0.5, 0.5, 0.5]
    var rescaleFactor = 1.0 / 255.0
    var doRescale = true
    var doNormalize = true
    var doPad = true

    init() {}

    /// The tile range `_preprocess` uses: 1 to 1 when splitting is off.
    var tileRange: (min: Int, max: Int) { doImageSplitting ? (minTiles, maxTiles) : (1, 1) }

    /// Rows per vision-tower view. transformers derives it and ignores a configured
    /// `max_num_patches`.
    var maxNumPatches: Int {
        let side = tileSize / encoderPatchSize
        let tilePatches = doImageSplitting ? side * side : 0
        return max(maxImageTokens * downsampleFactor * downsampleFactor, tilePatches)
    }

    /// Normalised value of every byte per channel in float32, as transformers'
    /// `rescale_and_normalize` computes it: `table[channel * 256 + byte]`. Not
    /// `MediaProcessing.normalize`, a Core Image filter that is not bit-exact.
    func normalizationTable() -> [Float] {
        var table = [Float](repeating: 0, count: 3 * 256)
        for c in 0 ..< 3 {
            var mean = Float(imageMean.count == 1 ? imageMean[0] : imageMean[c])
            var std = Float(imageStd.count == 1 ? imageStd[0] : imageStd[c])
            let rescale = Float(rescaleFactor)
            if doRescale && doNormalize {
                // `_fuse_mean_std_and_rescale_factor` folds the rescale into mean and std.
                mean *= Float(1 / rescaleFactor)
                std *= Float(1 / rescaleFactor)
            }
            for v in 0 ..< 256 {
                var x = Float(v)
                if doNormalize {
                    x = (x - mean) / std
                } else if doRescale {
                    x *= rescale
                }
                table[c * 256 + v] = x
            }
        }
        return table
    }

    /// `smart_resize`: the single view or thumbnail size, multiples of patch * downsample.
    /// Not `QwenVL.targetSize`: Python rounds half to even in float64, and LFM2 clamps the floor
    /// branch.
    func smartResize(width: Int, height: Int) -> (width: Int, height: Int) {
        let factor = encoderPatchSize * downsampleFactor
        let minPixels =
            minImageTokens * encoderPatchSize * encoderPatchSize * downsampleFactor
            * downsampleFactor
        let maxPixels =
            maxImageTokens * encoderPatchSize * encoderPatchSize * downsampleFactor
            * downsampleFactor
        let (w, h, f) = (Double(width), Double(height), Double(factor))
        var hBar = max(factor, pythonRound(h / f) * factor)
        var wBar = max(factor, pythonRound(w / f) * factor)
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((h / beta / f).rounded(.down)) * factor)
            wBar = max(factor, Int((w / beta / f).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = Int((h * beta / f).rounded(.up)) * factor
            wBar = Int((w * beta / f).rounded(.up)) * factor
        }
        return (wBar, hBar)
    }

    /// `_is_image_too_large`: whether the image is big enough to split into tiles.
    func isImageTooLarge(width: Int, height: Int) -> Bool {
        let factor = Double(encoderPatchSize * downsampleFactor)
        let hBar = max(encoderPatchSize, pythonRound(Double(height) / factor) * Int(factor))
        let wBar = max(encoderPatchSize, pythonRound(Double(width) / factor) * Int(factor))
        let limit =
            Double(
                maxImageTokens * encoderPatchSize * encoderPatchSize * downsampleFactor
                    * downsampleFactor)
            * maxPixelsTolerance
        return Double(hBar * wBar) > limit
    }

    /// `_get_grid_layout`: the (columns, rows) grid closest to the image's aspect ratio.
    func gridLayout(width: Int, height: Int) -> (columns: Int, rows: Int) {
        let aspectRatio = Double(width) / Double(height)
        let area = Double(width * height)
        var best = (1, 1)
        var bestDiff = Double.infinity
        let tiles = tileRange
        for (x, y) in Self.targetRatios(minTiles: tiles.min, maxTiles: tiles.max) {
            let diff = abs(aspectRatio - Double(x) / Double(y))
            if diff < bestDiff {
                bestDiff = diff
                best = (x, y)
            } else if diff == bestDiff && area > 0.5 * Double(tileSize * tileSize * x * y) {
                best = (x, y)
            }
        }
        return best
    }

    /// `_target_ratios`: every (columns, rows) grid with `minTiles...maxTiles` tiles, sorted by
    /// tile count, then columns. Python keeps set order for equal counts. That changes exact
    /// aspect-ratio ties only in narrow ranges, such as 4 to 4 tiles.
    static func targetRatios(minTiles: Int, maxTiles: Int) -> [(Int, Int)] {
        guard minTiles <= maxTiles, maxTiles >= 1 else { return [] }
        var ratios: [(Int, Int)] = []
        for w in 1 ... maxTiles {
            for h in 1 ... maxTiles / w where minTiles <= w * h {
                ratios.append((w, h))
            }
        }
        return ratios.sorted { ($0.0 * $0.1, $0.0) < ($1.0 * $1.1, $1.0) }
    }
}

/// Python 3 `round` on a float: half to even.
private func pythonRound(_ x: Double) -> Int {
    Int(x.rounded(.toNearestOrEven))
}

extension LFM2VLImageSettings: Codable {
    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case rescaleFactor = "rescale_factor"
        case doRescale = "do_rescale"
        case doNormalize = "do_normalize"
        case doPad = "do_pad"
        case resample
        case downsampleFactor = "downsample_factor"
        case encoderPatchSize = "encoder_patch_size"
        case patchSize = "patch_size"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case tileSize = "tile_size"
        case maxPixelsTolerance = "max_pixels_tolerance"
        case doImageSplitting = "do_image_splitting"
        case minTiles = "min_tiles"
        case maxTiles = "max_tiles"
        case useThumbnail = "use_thumbnail"
    }

    /// Decodes the image processor settings, flat or nested under `image_processor`. Malformed
    /// values keep their defaults, except `resample`, which must be bilinear (2) or bicubic (3).
    init(from decoder: any Decoder) throws {
        var container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.imageProcessor) {
            container = try container.nestedContainer(
                keyedBy: CodingKeys.self, forKey: .imageProcessor)
        }
        self.init()
        func value<T: Decodable>(_ key: CodingKeys, _ type: T.Type = T.self) -> T? {
            try? container.decodeIfPresent(T.self, forKey: key)
        }
        func channels(_ key: CodingKeys) -> [Double]? {
            value(key, [Double].self) ?? value(key, Double.self).map { [$0] }
        }
        imageMean = channels(.imageMean) ?? imageMean
        imageStd = channels(.imageStd) ?? imageStd
        rescaleFactor = value(.rescaleFactor) ?? rescaleFactor
        doRescale = value(.doRescale) ?? doRescale
        doNormalize = value(.doNormalize) ?? doNormalize
        doPad = value(.doPad) ?? doPad
        if container.contains(.resample), try !container.decodeNil(forKey: .resample) {
            let id =
                value(.resample, Int.self)
                ?? value(.resample, Double.self).flatMap(Int.init(exactly:))
            guard let filter = id.flatMap(ResampleFilter.init(rawValue:)) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .resample, in: container,
                    debugDescription:
                        "LFM2-VL resizes images with resample 2 (bilinear) or 3 (bicubic) only")
            }
            resample = filter
        }
        downsampleFactor = value(.downsampleFactor) ?? downsampleFactor
        encoderPatchSize = value(.encoderPatchSize) ?? value(.patchSize) ?? encoderPatchSize
        minImageTokens = value(.minImageTokens) ?? minImageTokens
        maxImageTokens = value(.maxImageTokens) ?? maxImageTokens
        tileSize = value(.tileSize) ?? tileSize
        maxPixelsTolerance = value(.maxPixelsTolerance) ?? maxPixelsTolerance
        doImageSplitting = value(.doImageSplitting) ?? doImageSplitting
        minTiles = value(.minTiles) ?? minTiles
        maxTiles = value(.maxTiles) ?? maxTiles
        useThumbnail = value(.useThumbnail) ?? useThumbnail
    }

    /// Writes the settings flat, in the form `init(from:)` reads back.
    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(imageMean, forKey: .imageMean)
        try container.encode(imageStd, forKey: .imageStd)
        try container.encode(rescaleFactor, forKey: .rescaleFactor)
        try container.encode(doRescale, forKey: .doRescale)
        try container.encode(doNormalize, forKey: .doNormalize)
        try container.encode(doPad, forKey: .doPad)
        try container.encode(resample.rawValue, forKey: .resample)
        try container.encode(downsampleFactor, forKey: .downsampleFactor)
        try container.encode(encoderPatchSize, forKey: .encoderPatchSize)
        try container.encode(minImageTokens, forKey: .minImageTokens)
        try container.encode(maxImageTokens, forKey: .maxImageTokens)
        try container.encode(tileSize, forKey: .tileSize)
        try container.encode(maxPixelsTolerance, forKey: .maxPixelsTolerance)
        try container.encode(doImageSplitting, forKey: .doImageSplitting)
        try container.encode(minTiles, forKey: .minTiles)
        try container.encode(maxTiles, forKey: .maxTiles)
        try container.encode(useThumbnail, forKey: .useThumbnail)
    }

    /// The image processor settings of a model directory, nested under `image_processor`, read
    /// as mlx-vlm reads them. `processor_config.json` overrides `preprocessor_config.json`, and
    /// a missing `resample` is bicubic. The tiling settings keep their defaults, because
    /// LiquidAI's MLX repos turn tiling off.
    static func resolvedConfiguration(directory: URL) throws -> Data? {
        func read(_ name: String) throws -> [String: Any]? {
            let url = directory.appending(component: name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            do {
                return try JSONSerialization.jsonObject(
                    with: Data(contentsOf: url), options: .json5Allowed) as? [String: Any] ?? [:]
            } catch {
                throw ProcessorConfigError(filename: name, underlying: error)
            }
        }
        let files = try ["preprocessor_config.json", "processor_config.json"].compactMap(read)
        guard !files.isEmpty else { return nil }
        var settings: [String: Any] = [:]
        for file in files {
            settings.merge(file["image_processor"] as? [String: Any] ?? file) { _, new in new }
        }
        if settings["resample"] == nil || settings["resample"] is NSNull {
            settings["resample"] = ResampleFilter.bicubic.rawValue
        }
        let defaults = LFM2VLImageSettings()
        settings["do_image_splitting"] = defaults.doImageSplitting
        settings["min_tiles"] = defaults.minTiles
        settings["max_tiles"] = defaults.maxTiles
        settings["use_thumbnail"] = defaults.useThumbnail
        return try JSONSerialization.data(withJSONObject: ["image_processor": settings])
    }
}

// MARK: - Layout

/// LFM2-VL smart resize, tile grid and thumbnail, as `resize_and_split` plans them.
struct LFM2VLImageLayout: Equatable, Sendable {
    var tiled: Bool
    /// (1, 1) when not tiled.
    var gridColumns: Int
    var gridRows: Int
    /// The single view when not tiled.
    var thumbnailWidth: Int
    var thumbnailHeight: Int
    /// Whether the views end with the thumbnail: always when not tiled, else with
    /// `use_thumbnail` and more than one tile.
    var includesThumbnail: Bool

    static func plan(width: Int, height: Int, settings: LFM2VLImageSettings) throws -> Self {
        guard width >= 1, height >= 1 else {
            throw VLMError.imageProcessingFailure("empty image (\(width)x\(height))")
        }
        let thumbnail = settings.smartResize(width: width, height: height)
        let tiles = settings.tileRange
        let tiled =
            settings.isImageTooLarge(width: width, height: height)
            && !(tiles.min == 1 && tiles.max == 1)
        let grid: (columns: Int, rows: Int) =
            tiled ? settings.gridLayout(width: width, height: height) : (1, 1)
        let layout = LFM2VLImageLayout(
            tiled: tiled, gridColumns: grid.columns, gridRows: grid.rows,
            thumbnailWidth: thumbnail.width, thumbnailHeight: thumbnail.height,
            includesThumbnail: !tiled || (settings.useThumbnail && grid.columns * grid.rows != 1))

        let patch = settings.encoderPatchSize
        let thumbnailPatches = (thumbnail.width / patch) * (thumbnail.height / patch)
        if layout.includesThumbnail && thumbnailPatches > settings.maxNumPatches {
            throw VLMError.imageProcessingFailure(
                "a \(width)x\(height) image needs \(thumbnailPatches) patches in one view, "
                    + "more than \(settings.maxNumPatches)")
        }
        return layout
    }

    /// The vision-tower views in transformers' order: tiles row-major, then the thumbnail.
    func crops<Image>(
        of image: Image, tileSize: Int, resize: (Image, _ width: Int, _ height: Int) -> RGB8Image
    ) -> [RGB8Image] {
        var crops: [RGB8Image] = []
        if tiled {
            let big = resize(image, gridColumns * tileSize, gridRows * tileSize)
            for row in 0 ..< gridRows {
                for column in 0 ..< gridColumns {
                    crops.append(
                        big.cropped(
                            x: column * tileSize, y: row * tileSize, width: tileSize,
                            height: tileSize))
                }
            }
        }
        if includesThumbnail {
            crops.append(resize(image, thumbnailWidth, thumbnailHeight))
        }
        return crops
    }
}

// MARK: - Processing

/// transformers' `Lfm2VlImageProcessor` and the image-token expansion of `Lfm2VlProcessor`, for
/// already-decoded images: per image, the crops in order (tiles row-major, then thumbnail),
/// their patches, and the token run that replaces one `<image>`.
struct LFM2VLImageProcessing: Sendable {
    let settings: LFM2VLImageSettings
    let imageTokenId: Int
    private let imageStartId: Int
    private let imageEndId: Int
    /// nil when the tokenizer lacks it. Only settings that tile with a thumbnail need it.
    private let thumbnailId: Int?
    /// `<|img_row_R_col_C|>` at `[R - 1][C - 1]`, for every `R * C <= maxTiles`; nil when the
    /// tokenizer lacks it.
    private let tileMarkerIds: [[Int?]]

    /// Resolves `<image>` and the image markers by name.
    init(settings: LFM2VLImageSettings, tokenizer: any Tokenizer) throws {
        guard settings.encoderPatchSize > 0, settings.downsampleFactor > 0,
            settings.tileSize >= settings.encoderPatchSize
        else {
            throw VLMError.processing(
                "Invalid LFM2-VL image settings: patch \(settings.encoderPatchSize), "
                    + "downsample \(settings.downsampleFactor), tile \(settings.tileSize)")
        }
        let tiles = settings.tileRange
        guard tiles.min <= tiles.max else {
            throw VLMError.processing("min_tiles must be less than or equal to max_tiles")
        }
        guard [1, 3].contains(settings.imageMean.count), [1, 3].contains(settings.imageStd.count)
        else {
            throw VLMError.processing(
                "image_mean and image_std need 1 or 3 values, got \(settings.imageMean.count) "
                    + "and \(settings.imageStd.count)")
        }
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw VLMError.processing("The tokenizer has no \(token) token")
            }
            return id
        }
        self.settings = settings
        imageTokenId = try id("<image>")
        imageStartId = try id("<|image_start|>")
        imageEndId = try id("<|image_end|>")
        thumbnailId =
            settings.useThumbnail && tiles.max > 1
            ? try id("<|img_thumbnail|>") : tokenizer.convertTokenToId("<|img_thumbnail|>")
        tileMarkerIds = (0 ..< max(tiles.max, 0)).map { r in
            (0 ..< tiles.max / (r + 1)).map { c in
                tokenizer.convertTokenToId("<|img_row_\(r + 1)_col_\(c + 1)|>")
            }
        }
    }

    /// The vision-tower input for the upright images that `decode` reads, and one token run per
    /// image. Images are decoded one at a time, so only one full-size image is alive at once.
    func process<Source>(
        _ sources: [Source], decode: (Source) throws -> OrientedImage
    ) throws -> (patches: Siglip2Patches, tokenRuns: [[Int]]) {
        var crops: [RGB8Image] = []
        var tokenRuns: [[Int]] = []
        for source in sources {
            let image = try decode(source)
            let layout = try LFM2VLImageLayout.plan(
                width: image.width, height: image.height, settings: settings)
            crops += layout.crops(of: image, tileSize: settings.tileSize) {
                $0.resized(width: $1, height: $2, filter: settings.resample)
            }
            tokenRuns.append(try tokenRun(layout))
        }
        let patch = settings.encoderPatchSize
        let length =
            settings.doPad
            ? settings.maxNumPatches
            : crops.map { ($0.width / patch) * ($0.height / patch) }.max() ?? 0
        let patches = siglip2Patches(
            crops, patchSize: patch, maxPatches: length, table: settings.normalizationTable())
        return (patches, tokenRuns)
    }

    /// `process` for images that are already upright.
    func process(_ images: [RGB8Image]) throws -> (patches: Siglip2Patches, tokenRuns: [[Int]]) {
        try process(images) { OrientedImage($0) }
    }

    /// Replace each `<image>` id in `ids`, in order, with its token run.
    func expand(_ ids: [Int], tokenRuns: [[Int]]) throws -> [Int] {
        let count = ids.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
        guard count == tokenRuns.count else {
            throw VLMError.processing(
                "The prompt has \(count) image placeholders for \(tokenRuns.count) images")
        }
        var expanded: [Int] = []
        expanded.reserveCapacity(ids.count + tokenRuns.reduce(0) { $0 + $1.count })
        var next = 0
        for id in ids {
            if id == imageTokenId {
                expanded += tokenRuns[next]
                next += 1
            } else {
                expanded.append(id)
            }
        }
        return expanded
    }

    /// `_build_image_tokens` for one image. Tile and thumbnail counts round odd patch grids up.
    func tokenRun(_ layout: LFM2VLImageLayout) throws -> [Int] {
        let patch = settings.encoderPatchSize
        let factor = settings.downsampleFactor
        func tokens(rows: Int, columns: Int) -> Int {
            ((rows + factor - 1) / factor) * ((columns + factor - 1) / factor)
        }
        let tilePatches = settings.tileSize / patch
        let perTile = tokens(rows: tilePatches, columns: tilePatches)
        let perImage = tokens(
            rows: layout.thumbnailHeight / patch, columns: layout.thumbnailWidth / patch)

        var run = [imageStartId]
        if layout.gridRows > 1 || layout.gridColumns > 1 {
            for row in 0 ..< layout.gridRows {
                for column in 0 ..< layout.gridColumns {
                    guard let marker = tileMarkerIds[row][column] else {
                        throw VLMError.processing(
                            "The tokenizer has no <|img_row_\(row + 1)_col_\(column + 1)|> token")
                    }
                    run.append(marker)
                    run += repeatElement(imageTokenId, count: perTile)
                }
            }
            if layout.includesThumbnail {
                guard let thumbnailId else {
                    throw VLMError.processing("The tokenizer has no <|img_thumbnail|> token")
                }
                run.append(thumbnailId)
                run += repeatElement(imageTokenId, count: perImage)
            }
        } else {
            // A tiled 1x1 grid (min_tiles 1) also lands here and miscounts its tile, as in
            // transformers. `inputEmbeddings` then throws.
            run += repeatElement(imageTokenId, count: perImage)
        }
        run.append(imageEndId)
        return run
    }
}
