// Copyright © 2026 Apple Inc.
// Port of mlx-vlm's d1-omni model (Blaizzy/mlx-vlm#2467), which follows LiquidAI's reference code.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

extension LFM2VLImageSettings {
    /// The settings `vision.py` hardcodes, whatever the checkpoint stores: the LFM2-VL
    /// defaults, with `(x - 127.5) / 127.5`.
    static let d1Omni: LFM2VLImageSettings = {
        var settings = LFM2VLImageSettings()
        settings.doRescale = false
        settings.imageMean = [127.5, 127.5, 127.5]
        settings.imageStd = [127.5, 127.5, 127.5]
        return settings
    }()
}

/// Python `preprocess` for one image: LFM2-VL tiling, torchvision's uint8 antialiased bilinear
/// resize and `(x - 127.5) / 127.5`. The tiles come row-major, then the thumbnail, each zero
/// padded to 1024 patches. The resize reads the stored pixels through the EXIF orientation.
func d1OmniPatches(_ image: OrientedImage) throws -> Siglip2Patches {
    let settings = LFM2VLImageSettings.d1Omni
    let image = d1OmniResizeInput(image)
    let layout = try LFM2VLImageLayout.plan(
        width: image.width, height: image.height, settings: settings)
    let crops = layout.crops(of: image, tileSize: settings.tileSize) {
        $0.resized(width: $1, height: $2, filter: settings.resample)
    }
    return siglip2Patches(
        crops, patchSize: settings.encoderPatchSize, maxPatches: settings.maxNumPatches,
        table: settings.normalizationTable())
}

/// The image the resizes of `d1OmniPatches` read. Each CPU resize turns the stored pixels at
/// full size, so without the GPU a turned image is made upright once here.
func d1OmniResizeInput(_ image: OrientedImage) -> OrientedImage {
    guard image.orientation != 1, Device.defaultDevice().deviceType != .gpu else { return image }
    return OrientedImage(image.upright)
}

/// 2x2 pixel unshuffle, then a GELU MLP into the trunk width. One row per 2x2 patches.
final class D1OmniProjector: Module, UnaryLayer {
    @ModuleInfo(key: "pixel_unshuffle") var pixelUnshuffle: PixelUnshuffleBlock
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
        _pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: 2)
        _linear1.wrappedValue = Linear(visionDimensions * 2 * 2, hiddenDimensions)
        _linear2.wrappedValue = Linear(hiddenDimensions, outputDimensions)
    }

    /// `[1, rows, columns, visionDimensions]` -> `[ceil(rows / 2) * ceil(columns / 2), D]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = linear2(gelu(linear1(pixelUnshuffle(x))))
        return y.reshaped(-1, y.dim(-1))
    }
}

/// Images -> prefix embeddings: the SigLIP2 NaFlex tower and the projector.
final class D1OmniVision: Module {
    @ModuleInfo var tower: Siglip2NaFlex.VisionModel
    @ModuleInfo var projector: D1OmniProjector

    init(
        _ config: LFM2VLConfiguration.VisionConfiguration, projectorHiddenSize: Int,
        outputDimensions: Int
    ) {
        _tower.wrappedValue = Siglip2NaFlex.VisionModel(
            Siglip2NaFlex.Configuration(config, visionFeatureLayer: -1))
        _projector.wrappedValue = D1OmniProjector(
            visionDimensions: config.hiddenSize, hiddenDimensions: projectorHiddenSize,
            outputDimensions: outputDimensions)
    }

    /// Prefix rows `[P, D]` for one image: the tower over its crops cut to their patches, then
    /// the projector per crop.
    func callAsFunction(_ patches: Siglip2Patches) -> MLXArray {
        let states = tower.viewStates(patches.pixels, spatialShapes: patches.spatialShapes)
        let crops = zip(states, patches.spatialShapes).map { state, shape in
            projector(state.reshaped(1, shape.0, shape.1, -1))
        }
        return concatenated(crops)
    }

    /// Prefix rows `[P, D]` for the images that `decode` reads, in order. Each image is decoded,
    /// cropped and released before the next, so only one full-size image is alive at a time.
    func callAsFunction<Source>(
        _ sources: [Source], decode: (Source) throws -> OrientedImage
    ) throws -> MLXArray {
        concatenated(try sources.map { self(try d1OmniPatches(try decode($0))) })
    }
}
