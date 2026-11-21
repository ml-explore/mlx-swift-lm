// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Each padded patch-merge block becomes one image token.

/// gelu in float32, as the reference.
private func preciseGelu(_ x: MLXArray) -> MLXArray {
    gelu(x.asType(.float32)).asType(x.dtype)
}

/// The row and column of each patch, with the patches of each merge block consecutive and the
/// blocks in raster order.
private func gridPositions(gridH: Int, gridW: Int, merge: Int) -> (row: MLXArray, col: MLXArray) {
    let i = MLXArray.arange(gridH * gridW, dtype: .int32)
    let blocksW = gridW / merge
    let row = i / Int32(merge * merge * blocksW) * Int32(merge) + i / Int32(merge) % Int32(merge)
    let col = i / Int32(merge * merge) % Int32(blocksW) * Int32(merge) + i % Int32(merge)
    return (row, col)
}

/// The two bilinear taps of each index and their weights, on a grid of `side` positions.
private func bilinearTaps(_ index: MLXArray, size: Int, side: Int) -> (MLXArray, MLXArray) {
    let source = (index.asType(.float32) + 0.5) * Float(side) / Float(size) - 0.5
    let taps = floor(source)[0..., .newAxis] + MLXArray([Float(0), 1])
    let weights = maximum(1 - abs(source[0..., .newAxis] - taps), 0)
    return (clip(taps, min: 0, max: side - 1).asType(.int32), weights)
}

private final class Inkling2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var classPositionEmbedding: MLXArray

    let mergeSize: Int
    let gridSize: Int

    init(_ config: Inkling2VisionConfiguration) {
        mergeSize = config.mergeSize
        gridSize = config.positionGridSize
        _patchEmbedding.wrappedValue = Linear(
            3 * config.patchSize * config.patchSize, config.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: gridSize * gridSize, dimensions: config.hiddenSize)
        _classEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _classPositionEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        super.init()
    }

    func callAsFunction(_ patches: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        var embeddings = patchEmbedding(patches)

        let side = gridSize
        let (row, col) = gridPositions(gridH: gridH, gridW: gridW, merge: mergeSize)
        let (rowTaps, rowWeights) = bilinearTaps(row, size: gridH, side: side)
        let (colTaps, colWeights) = bilinearTaps(col, size: gridW, side: side)
        let indices = rowTaps[0..., 0..., .newAxis] * Int32(side) + colTaps[0..., .newAxis, 0...]
        let weights = rowWeights[0..., 0..., .newAxis] * colWeights[0..., .newAxis, 0...]
        var positions = positionEmbedding(indices.reshaped(-1, 4)).asType(.float32)
        positions = (positions * weights.reshaped(-1, 4, 1)).sum(axis: 1)
        embeddings = embeddings + positions.asType(embeddings.dtype)

        let classToken = (classEmbedding + classPositionEmbedding).asType(embeddings.dtype)
        return concatenated([classToken.expandedDimensions(axis: 0), embeddings])
    }
}

/// Rotates the interleaved pairs of each head vector, in float32.
private func applyVisionRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let states = x.asType(.float32)
    let pairs = states.reshaped(Array(states.shape.dropLast()) + [-1, 2])
    let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
        .reshaped(states.shape)
    return (states * cos + rotated * sin).asType(x.dtype)
}

private final class Inkling2VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outputProj: Linear

    let heads: Int
    let scale: Float

    init(_ config: Inkling2VisionConfiguration) {
        let dims = config.hiddenSize
        heads = config.attentionHeads
        scale = 1 / Float(config.headDim).squareRoot()
        _queryProj.wrappedValue = Linear(dims, dims)
        _keyProj.wrappedValue = Linear(dims, dims)
        _valueProj.wrappedValue = Linear(dims, dims)
        _outputProj.wrappedValue = Linear(dims, dims)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let cos = cos[0..., .newAxis, 0...]
        let sin = sin[0..., .newAxis, 0...]
        let q = applyVisionRotary(queryProj(x).reshaped(n, heads, -1), cos: cos, sin: sin)
        let k = applyVisionRotary(keyProj(x).reshaped(n, heads, -1), cos: cos, sin: sin)
        let v = valueProj(x).reshaped(n, heads, -1)
        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2)[.newAxis], keys: k.transposed(1, 0, 2)[.newAxis],
            values: v.transposed(1, 0, 2)[.newAxis], scale: scale, mask: .none)
        return outputProj(output[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

private final class Inkling2VisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: Inkling2VisionConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(preciseGelu(fc1(x)))
    }
}

private final class Inkling2VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "layernorm_before") var normBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var normAfter: LayerNorm
    @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
    @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

    init(_ config: Inkling2VisionConfiguration) {
        let dims = config.hiddenSize
        _attention.wrappedValue = Inkling2VisionAttention(config)
        _mlp.wrappedValue = Inkling2VisionMLP(config)
        _normBefore.wrappedValue = LayerNorm(dimensions: dims, eps: config.layerNormEps)
        _normAfter.wrappedValue = LayerNorm(dimensions: dims, eps: config.layerNormEps)
        _lambda1.wrappedValue = MLXArray.ones([dims])
        _lambda2.wrappedValue = MLXArray.ones([dims])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + lambda1 * attention(normBefore(x), cos: cos, sin: sin)
        return h + lambda2 * mlp(normAfter(h))
    }
}

private final class Inkling2VisionMerger: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: Inkling2VisionConfiguration, textHiddenSize: Int) {
        let dims = config.mergeSize * config.mergeSize * config.hiddenSize
        _linear1.wrappedValue = Linear(
            dims, config.projectionHiddenSize, bias: config.projectorBias)
        _linear2.wrappedValue = Linear(
            config.projectionHiddenSize, textHiddenSize, bias: config.projectorBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(preciseGelu(linear1(x)))
    }
}

final class Inkling2VisionModel: Module {
    @ModuleInfo fileprivate var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "pre_layernorm") var preNorm: LayerNorm
    @ModuleInfo fileprivate var layers: [Inkling2VisionEncoderLayer]
    @ModuleInfo fileprivate var merger: Inkling2VisionMerger

    let config: Inkling2VisionConfiguration
    private let inverseFrequencies: [Float]

    init(_ config: Inkling2VisionConfiguration, textHiddenSize: Int) {
        self.config = config
        let dims = config.headDim / 2
        inverseFrequencies = stride(from: 0, to: dims, by: 2).map {
            1 / pow(config.ropeTheta, Float($0) / Float(dims))
        }
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(config)
        _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Inkling2VisionEncoderLayer(config)
        }
        _merger.wrappedValue = Inkling2VisionMerger(config, textHiddenSize: textHiddenSize)
        super.init()
    }

    private func rotary(row: MLXArray, col: MLXArray) -> (MLXArray, MLXArray) {
        let inverse = MLXArray(inverseFrequencies)
        let rows = row[0..., .newAxis].asType(.float32) * inverse
        let cols = col[0..., .newAxis].asType(.float32) * inverse
        let frequencies = repeated(concatenated([cols, rows], axis: -1), count: 2, axis: -1)
        return (cos(frequencies), sin(frequencies))
    }

    /// The image tokens of one image's patches, `[rows * cols, textHidden]`.
    private func encodeImage(_ patches: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let merge = config.mergeSize
        let dtype = embeddings.patchEmbedding.weight.dtype
        var hidden = embeddings(patches.asType(dtype), gridH: gridH, gridW: gridW)
        hidden = preNorm(hidden)

        // The class token is at the origin; the patches start at 1.
        let (row, col) = gridPositions(gridH: gridH, gridW: gridW, merge: merge)
        let zero = MLXArray.zeros([1], dtype: .int32)
        let (cos, sin) = rotary(
            row: concatenated([zero, row + 1]), col: concatenated([zero, col + 1]))
        for layer in layers {
            hidden = layer(hidden, cos: cos, sin: sin)
        }
        let tokens = gridH * gridW / (merge * merge)
        return merger(hidden[1...].reshaped(tokens, -1))
    }

    /// The image tokens of all images, in order.
    ///
    /// - Parameters:
    ///   - pixels: `[patches, patch * patch * 3]`, every image's patches in the order
    ///     ``Inkling2Processor`` makes them.
    ///   - grids: the patch grid of each image.
    func encode(pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        var start = 0
        var features = [MLXArray]()
        for grid in grids {
            let count = grid.h * grid.w
            guard start + count <= pixels.dim(0) else {
                throw VLMError.processing("The image grids need more patches than were given")
            }
            features.append(
                encodeImage(pixels[start ..< (start + count)], gridH: grid.h, gridW: grid.w))
            start += count
        }
        return concatenated(features)
    }
}

// MARK: - Processor

/// The image settings of the processor, read from `config.json`.
public struct Inkling2ProcessorConfiguration: Codable, Sendable {
    public let patchSize: Int
    public let mergeSize: Int
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
        case imageTokenId = "image_token_id"
    }

    private struct Vision: Codable {
        let patchSize: Int?
        let mergeSize: Int?

        enum CodingKeys: String, CodingKey {
            case patchSize = "native_patch_size"
            case mergeSize = "adapter_merge_size"
        }
    }

    public init(patchSize: Int = 20, mergeSize: Int = 2, imageTokenId: Int = 200054) {
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.imageTokenId = imageTokenId
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let vision = try c.decodeIfPresent(Vision.self, forKey: .visionConfiguration)
        patchSize = vision?.patchSize ?? 20
        mergeSize = vision?.mergeSize ?? 2
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200054
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(
            Vision(patchSize: patchSize, mergeSize: mergeSize), forKey: .visionConfiguration)
        try c.encode(imageTokenId, forKey: .imageTokenId)
    }
}

/// Builds the processor from `config.json`, as Inkling 2 checkpoints ship no processor
/// configuration.
public struct Inkling2ProcessorLoadingResolver: VLMProcessorLoadingResolver {
    public static let modelTypes: Set<String> = ["inkling2", "inkling2_loop"]

    public init() {}

    public func fallbackProcessorConfiguration(for context: VLMProcessorLoadingContext) throws
        -> VLMProcessorConfiguration?
    {
        guard Self.modelTypes.contains(context.modelType) else { return nil }
        return VLMProcessorConfiguration(
            data: context.configurationData, processorType: "Inkling2Processor")
    }

    public func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? {
        Self.modelTypes.contains(context.modelType) ? "Inkling2Processor" : nil
    }
}

public struct Inkling2Processor: UserInputProcessor {
    private let config: Inkling2ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Inkling2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// The normalized patches of one image and its patch grid.
    ///
    /// Transparent pixels go on a gray checkerboard and the padding is -1 on the 0-255 scale,
    /// as in the reference.
    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        var image = MediaProcessing.apply(image, processing: processing)
        if let maxPixels = processing?.maxPixels {
            let area = image.extent.width * image.extent.height
            if area > CGFloat(maxPixels) {
                let scale = (CGFloat(maxPixels) / area).squareRoot()
                let size = CGSize(
                    width: max(1, (image.extent.width * scale).rounded(.down)),
                    height: max(1, (image.extent.height * scale).rounded(.down)))
                image = MediaProcessing.resampleBicubic(image, to: size)
            }
        }
        image = MediaProcessing.inSRGBToneCurveSpace(image)
        let rgba = renderRGBA(image)
        let (height, width) = (rgba.dim(0), rgba.dim(1))
        guard height > 0, width > 0 else {
            throw VLMError.imageProcessingFailure("The image is empty")
        }

        // Premultiplied colors over the checkerboard.
        let y = MLXArray.arange(height, dtype: .int32)[0..., .newAxis]
        let x = MLXArray.arange(width, dtype: .int32)[.newAxis, 0...]
        let light = ((x / 8 + y / 8) % 2) .== 0
        let background = which(light, MLXArray(Float(204) / 255), MLXArray(Float(153) / 255))
        let alpha = rgba[0..., 0..., 3...]
        let rgb = rgba[0..., 0..., ..<3] + background[0..., 0..., .newAxis] * (1 - alpha)

        let block = config.patchSize * config.mergeSize
        let rows = (height + block - 1) / block
        let cols = (width + block - 1) / block
        let normalized = (rgb - 0.5) / 0.5
        let paddingValue = (Float(-1) / 255 - 0.5) / 0.5
        let paddedImage = padded(
            normalized,
            widths: [[0, rows * block - height], [0, cols * block - width], [0, 0]],
            value: MLXArray(paddingValue))

        // The merge blocks in raster order, then the patches of each block.
        let (p, m) = (config.patchSize, config.mergeSize)
        let patches = paddedImage.reshaped(rows, m, p, cols, m, p, 3)
            .transposed(0, 3, 1, 4, 2, 5, 6)
            .reshaped(-1, p * p * 3)
        return (patches, THW(1, rows * m, cols * m))
    }

    /// The image as float32 `[H, W, 4]` premultiplied RGBA, top row first.
    private func renderRGBA(_ image: CIImage) -> MLXArray {
        let extent = image.extent
        let width = Int(extent.width.rounded())
        let height = Int(extent.height.rounded())
        let bytesPerRow = width * 4 * MemoryLayout<Float32>.size
        var data = Data(count: height * bytesPerRow)
        data.withUnsafeMutableBytes { pointer in
            Inkling2Processor.context.render(
                image, toBitmap: pointer.baseAddress!, rowBytes: bytesPerRow, bounds: extent,
                format: .RGBAf, colorSpace: nil)
        }
        return clip(MLXArray(data, [height, width, 4], type: Float32.self), min: 0, max: 1)
    }

    private static let context = CIContext()

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else { throw VLMError.singleMediaTypeAllowed }

        let messages = Qwen2VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        guard !input.images.isEmpty else {
            return LMInput(text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)))
        }

        let processed = try input.images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let merge = config.mergeSize * config.mergeSize
        var expanded = [Int]()
        var imageIndex = 0
        for token in tokens {
            guard token == config.imageTokenId else {
                expanded.append(token)
                continue
            }
            guard imageIndex < processed.count else {
                throw VLMError.processing("The prompt has more image tokens than images")
            }
            let count = processed[imageIndex].1.product / merge
            expanded += Array(repeating: config.imageTokenId, count: count)
            imageIndex += 1
        }
        guard imageIndex == processed.count else {
            throw VLMError.processing(
                "The prompt has \(imageIndex) image tokens for \(processed.count) images")
        }
        tokens = expanded

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: .init(
                pixels: concatenated(processed.map(\.0)), frames: processed.map(\.1)))
    }
}
