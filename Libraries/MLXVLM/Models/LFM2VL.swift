// Port of transformers' LFM2-VL (models/lfm2_vl, with the LFM2 text model from models/lfm2),
// based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm2_vl

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Language Model Components (LFM2)

private enum Language {

    fileprivate class LFM2Attention: Module {
        let scale: Float
        let headDim: Int
        let heads: Int
        let kvHeads: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        let rope: RoPE

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class LFM2ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.lCache = config.convLCache
            self.hiddenSize = config.hiddenSize
            let bias = config.convBias

            self._conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize,
                outputChannels: config.hiddenSize,
                kernelSize: lCache,
                groups: config.hiddenSize,
                bias: bias
            )

            self._inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: bias)
            self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
            let BCx = inProj(x)
            let BCxSplit = BCx.split(parts: 3, axis: -1)
            let B = BCxSplit[0]
            let C = BCxSplit[1]
            let xPart = BCxSplit[2]
            var Bx = B * xPart

            var state: MLXArray? = nil
            if let cache {
                state = cache[0]
            }
            if state == nil {
                state = MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            }

            Bx = concatenated([state!, Bx], axis: -2)
            if let cache {
                cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
                cache.advance(x.dim(1))
            }

            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    fileprivate class LFM2MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            let adjustedFFDim = config.feedForwardDimensions
            self._w1.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, config.blockDim, bias: false)
            self._w3.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    fileprivate class LFM2DecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: LFM2Attention?
        @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM2MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = config.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._attention.wrappedValue = LFM2Attention(config)
            } else {
                self._conv.wrappedValue = LFM2ShortConv(config, layerIdx: layerIdx)
            }

            self._feedForward.wrappedValue = LFM2MLP(config)
            self._operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = attention!(operatorNorm(x), mask: mask, cache: cache)
            } else {
                r = conv!(operatorNorm(x), cache: cache as? MambaCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    fileprivate class LFM2ModelInner: Module {
        let config: LFM2VLConfiguration.TextConfiguration
        let vocabularySize: Int
        let numHiddenLayers: Int

        let layers: [LFM2DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.vocabularySize = config.vocabularySize
            self.numHiddenLayers = config.hiddenLayers

            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: config.hiddenSize)

            self.layers = (0 ..< numHiddenLayers).map { i in
                LFM2DecoderLayer(config, layerIdx: i)
            }

            self._embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens(inputs)

            let mask =
                mask
                ?? {
                    let firstAttnIdx = config.fullAttnIdxs.first ?? 0
                    let c =
                        cache != nil && firstAttnIdx < cache!.count ? cache![firstAttnIdx] : nil
                    return createAttentionMask(h: h, cache: c)
                }()

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return embeddingNorm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: LFM2VLConfiguration.TextConfiguration
        let modelType: String
        let model: LFM2ModelInner

        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        init(_ config: LFM2VLConfiguration.TextConfiguration, tieWordEmbeddings: Bool) {
            self.config = config
            self.modelType = config.modelType

            self.model = LFM2ModelInner(config)
            if !tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< config.hiddenLayers).map { layerIdx in
                config.fullAttnIdxs.contains(layerIdx) ? config.kvHeads : 0
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            mask: MLXArray? = nil,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> LMOutput {
            let out = model(
                inputs ?? MLXArray([0]), cache: cache, inputEmbeddings: inputsEmbeds)
            return LMOutput(logits: vocabularyLogits(out))
        }

        func vocabularyLogits(_ hidden: MLXArray) -> MLXArray {
            if let lmHead {
                return lmHead(hidden)
            }
            return model.embedTokens.asLinear(hidden)
        }
    }
}

// MARK: - Multi-modal Projector

private class Lfm2VlMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(config: LFM2VLConfiguration) {
        let inChannels =
            config.visionConfiguration.hiddenSize
            * (config.downsampleFactor * config.downsampleFactor)

        if config.projectorUseLayernorm {
            self._layerNorm.wrappedValue = LayerNorm(dimensions: inChannels)
        }

        self._linear1.wrappedValue = Linear(
            inChannels, config.projectorHiddenSize, bias: config.projectorBias)
        self._linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let layerNorm {
            x = layerNorm(x)
        }
        x = linear1(x)
        x = gelu(x.asType(.float32)).asType(x.dtype)
        x = linear2(x)
        return x
    }
}

/// Space-to-depth over a `[N, W, H, C]` feature map, as `Lfm2VlMultiModalProjector.pixel_unshuffle`
/// does. Odd sizes are zero padded, which transformers does not do; the processor never makes
/// them.
final class PixelUnshuffleBlock: Module, UnaryLayer {
    let factor: Int

    init(factor: Int) {
        self.factor = factor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        if w % factor != 0 {
            let padding = MLXArray.zeros([n, factor - (w % factor), h, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 1)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        if h % factor != 0 {
            let padding = MLXArray.zeros([n, w, factor - (h % factor), c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 2)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        x = x.reshaped(n, w, h / factor, c * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(n, h / factor, w / factor, c * factor * factor)
        return x.transposed(0, 2, 1, 3)
    }
}

// MARK: - Processor

/// LFM2 VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``LFM2VL`` and is typically created by ``VLMModelFactory``.
/// Images follow transformers' `Lfm2VlProcessor`: smart resize, or tiles plus a thumbnail,
/// torchvision's antialiased resize on the decoded 8-bit pixels, and the image start, tile and
/// end marker tokens.
///
/// `UserInput.Processing.resize` first fits each image in that size, keeping its aspect ratio, as
/// `MediaProcessing.apply` does for other models. `ChatSession` passes 512 × 512 by default. Pass
/// `processing: .init()` to read images at full size.
///
/// File images get their EXIF orientation. sRGB and untagged PNG, GIF, WebP, BMP and TIFF files
/// decode to the same pixels as transformers' `load_image`, except 16-bit gray PNGs and
/// transparent GIFs. RGB images with another colour profile are converted to sRGB, and a
/// `CIImage` is read in its colour space. JPEG decodes through Core Image, within 4 levels of
/// `load_image`, and HEIC through ImageIO. CMYK can be off by more than 100 levels.
///
/// An image that cannot be decoded or does not fit the vision tower throws
/// `VLMError.imageProcessingFailure`. A count of `<image>` that does not match the images throws
/// `VLMError.processing`, except that extra `<image>` outside user messages are read as text. A
/// video also throws.
public struct LFM2VLProcessor: UserInputProcessor {
    private let config: LFM2VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: LFM2VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // transformers' `Lfm2VlProcessor` has no video input. Without this check the video would
        // be dropped, and the LFM2.5-VL template would print its part as JSON.
        guard input.videos.isEmpty else {
            throw VLMError.processing("LFM2-VL does not accept video")
        }

        let messages = Qwen2VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let promptTokens = try promptTokens(messages: messages, input: input)

        if input.images.isEmpty {
            let imageTokenId = tokenizer.convertTokenToId("<image>")
            let count = promptTokens.filter { $0 == imageTokenId }.count
            guard count == 0 else {
                throw VLMError.processing(
                    "The prompt has \(count) image placeholders but no images")
            }
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Settings are checked per call, as transformers checks them in `_preprocess`.
        let settings = config.settings
        let processing = try LFM2VLImageProcessing(settings: settings, tokenizer: tokenizer)
        let resize = input.processing.resize
        let (patches, tokenRuns) = try processing.process(input.images) {
            let image = try OrientedImage($0)
            guard let resize else { return image }
            return try Self.fitted(image, in: resize, filter: settings.resample)
        }
        let ids = try processing.expand(promptTokens, tokenRuns: tokenRuns)

        let promptArray = MLXArray(ids).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: LMInput.ProcessedImage(
                pixels: patches.pixels,
                frames: patches.spatialShapes.map { THW(1, $0.0, $0.1) })
        )
    }

    /// `image` scaled to fit in `size`, as `MediaProcessing.apply` scales it.
    private static func fitted(
        _ image: OrientedImage, in size: CGSize, filter: ResampleFilter
    ) throws -> OrientedImage {
        let upright = CGSize(width: image.width, height: image.height)
        let scale = MediaProcessing.bestFitScale(upright, in: size)
        guard scale.isFinite, scale > 0 else {
            throw VLMError.processing("resize must be a positive size, not \(size)")
        }
        let fit = MediaProcessing.bestFit(upright, in: size)
        let (width, height) = (max(1, Int(fit.width)), max(1, Int(fit.height)))
        guard width != image.width || height != image.height else { return image }
        return OrientedImage(image.resized(width: width, height: height, filter: filter))
    }

    /// The chat template's token ids. When the prompt has more `<image>` than images, those
    /// typed in system, assistant and tool messages or in tool specs are written as text
    /// tokens, so a reply that holds `<image>` does not break the turns after it.
    private func promptTokens(messages: [MLXLMCommon.Message], input: UserInput) throws -> [Int] {
        func render(_ messages: [MLXLMCommon.Message], tools: [ToolSpec]?) throws -> [Int] {
            try tokenizer.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: input.additionalContext)
        }
        let tokens = try render(messages, tools: input.tools)
        let marker = "<|image_start|>"
        guard let imageId = tokenizer.convertTokenToId("<image>"),
            tokens.count(where: { $0 == imageId }) > input.images.count,
            let markerId = tokenizer.convertTokenToId(marker)
        else { return tokens }

        // Render again with those `<image>` swapped for another single token. The ids that
        // change are the typed ones. Match code units, as the tokenizer does.
        var typed = false
        func swappedText(_ text: String) -> String {
            guard text.range(of: "<image>", options: .literal) != nil else { return text }
            typed = true
            return text.replacingOccurrences(of: "<image>", with: marker, options: .literal)
        }
        func swappedValue(_ value: any Sendable) -> any Sendable {
            switch value {
            case let text as String: return swappedText(text)
            case let parts as [[String: String]]: return parts.map { $0.mapValues(swappedText) }
            case let items as [any Sendable]: return items.map(swappedValue)
            case let fields as [String: any Sendable]: return fields.mapValues(swappedValue)
            default: return value
            }
        }
        let swapped = messages.map { (message: MLXLMCommon.Message) -> MLXLMCommon.Message in
            message["role"] as? String == "user" ? message : message.mapValues(swappedValue)
        }
        let swappedTools = input.tools?.map { (tool: ToolSpec) -> ToolSpec in
            tool.mapValues(swappedValue)
        }
        // Left as placeholders when the second render fails, or it or the text tokens do not
        // line up.
        guard typed, let swappedTokens = try? render(swapped, tools: swappedTools) else {
            return tokens
        }
        let text =
            tokenizer.encode(text: "<", addSpecialTokens: false)
            + tokenizer.encode(text: "image>", addSpecialTokens: false)
        guard swappedTokens.count == tokens.count, !text.contains(imageId),
            tokenizer.decode(tokenIds: text, skipSpecialTokens: false) == "<image>"
        else { return tokens }
        var ids: [Int] = []
        ids.reserveCapacity(tokens.count)
        for (id, swappedId) in zip(tokens, swappedTokens) {
            if id == swappedId {
                ids.append(id)
            } else if id == imageId && swappedId == markerId {
                ids += text
            } else {
                return tokens
            }
        }
        return ids
    }
}

// MARK: - Model

/// LFM2 VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class LFM2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Siglip2NaFlex.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Lfm2VlMultiModalProjector
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "pixel_unshuffle") private var pixelUnshuffle: PixelUnshuffleBlock?

    public let config: LFM2VLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public init(_ config: LFM2VLConfiguration) {
        self.config = config

        self._visionModel.wrappedValue = Siglip2NaFlex.VisionModel(
            Siglip2NaFlex.Configuration(
                config.visionConfiguration, visionFeatureLayer: config.visionFeatureLayer))

        if config.downsampleFactor > 1 {
            self._pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: config.downsampleFactor)
        }

        self._multiModalProjector.wrappedValue = Lfm2VlMultiModalProjector(config: config)
        self._languageModel.wrappedValue = Language.LanguageModel(
            config.textConfiguration, tieWordEmbeddings: config.tieWordEmbeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        var features: MLXArray? = nil
        if let pixels = input.image?.pixels {
            let spatialShapes: [(Int, Int)]
            if let frames = input.image?.frames, !frames.isEmpty {
                spatialShapes = frames.map { ($0.h, $0.w) }
            } else {
                // No frames: assume one square view per row.
                let side = Int(Double(pixels.dim(1)).squareRoot())
                spatialShapes = Array(repeating: (side, side), count: pixels.dim(0))
            }
            features = imageFeatures(pixels: pixels, spatialShapes: spatialShapes)
        }

        let embeddings = try inputEmbeddings(
            tokens: input.text.tokens, imageFeatures: features)

        let result = try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let totalPositions = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: totalPositions) { range in
                _ = languageModel(
                    nil, cache: cache, inputsEmbeds: embeddings[0..., range, 0...])
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }

            let result = languageModel(
                nil, cache: cache, inputsEmbeds: embeddings[0..., processed..., 0...])
            prefill.progress?(totalPositions, totalPositions)
            return result
        }

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transformKey(_ key: String) -> String {
            var key = key

            if key.contains("vision_tower") {
                key =
                    key
                    .replacingOccurrences(of: "model.", with: "")
                    .replacingOccurrences(of: "vision_encoder", with: "encoder")
                    .replacingOccurrences(of: "vision_embeddings", with: "embeddings")
                    .replacingOccurrences(of: "vision_post_layernorm", with: "post_layernorm")
            }

            if key.contains("language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            if key.contains("multi_modal_projector") {
                key = key.replacingOccurrences(
                    of: "model.multi_modal_projector", with: "multi_modal_projector")
            }

            // HF checkpoints keep an untied `lm_head` at the top level.
            if key.hasPrefix("model.") || key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            return key
        }

        let weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.tieWordEmbeddings)
        var sanitizedWeights = [String: MLXArray]()
        for (k, v) in weights {
            let newKey = transformKey(k)
            if newKey.contains("position_ids")
                || (!config.projectorUseLayernorm
                    && newKey.hasPrefix("multi_modal_projector.layer_norm."))
            {
                continue
            }

            // Handle conv weight transposition
            var value = v
            if newKey.contains("conv.weight") {
                if v.shape[v.shape.count - 1] > v.dim(1) {
                    value = v.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[newKey] = value
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let textConfig = config.textConfiguration
        return try (0 ..< textConfig.hiddenLayers).map { layerIdx in
            if textConfig.fullAttnIdxs.contains(layerIdx) {
                try makeAttentionKVCache(parameters: parameters)
            } else {
                MambaCache()
            }
        }
    }
}

// MARK: - Image features

extension LFM2VL {
    /// Vision features per view, already projected: `[Nimage, hidden]`. Port of `encode_images`.
    /// `pixels` may be padded past each view's `h * w` patches; the padding is not read.
    func imageFeatures(pixels: MLXArray, spatialShapes: [(Int, Int)]) -> MLXArray {
        let states = visionModel.viewStates(pixels, spatialShapes: spatialShapes)
        let features = zip(states, spatialShapes).map { (state, shape) -> MLXArray in
            let (h, w) = shape
            var feature = state.reshaped(1, h, w, -1)
            if let pixelUnshuffle {
                feature = pixelUnshuffle(feature)
            }
            let projected = multiModalProjector(feature)
            return projected.reshaped(-1, projected.dim(-1))
        }
        return concatenated(features, axis: 0)
    }

    /// Token embeddings with image features scattered at `image_token_id`: `[B, L, hidden]`.
    /// Features fill the image tokens in row-major order, as `masked_scatter` does.
    /// Throws when the number of image tokens and feature rows differ.
    func inputEmbeddings(tokens: MLXArray, imageFeatures: MLXArray?) throws -> MLXArray {
        let ids = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        let embeddings = languageModel.model.embedTokens(ids)
        guard let imageFeatures else { return embeddings }

        let flat = ids.flattened().asArray(Int.self)
        let positions = flat.indices.filter { flat[$0] == config.imageTokenIndex }
        guard positions.count == imageFeatures.dim(0) else {
            throw VLMError.processing(
                "Image features and image tokens do not match: tokens: \(positions.count), features \(imageFeatures.dim(0))"
            )
        }
        guard !positions.isEmpty else { return embeddings }

        let rows = embeddings.reshaped(-1, embeddings.dim(-1))
        rows[MLXArray(positions.map { Int32($0) })] = imageFeatures.asType(rows.dtype)
        return rows.reshaped(embeddings.shape)
    }
}

extension Siglip2NaFlex.Configuration {
    /// The LFM2-VL tower, truncated after `visionFeatureLayer` unless it is -1.
    init(_ vision: LFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int) {
        var layers = vision.numHiddenLayers
        if visionFeatureLayer != -1 {
            let index =
                visionFeatureLayer < 0
                ? vision.numHiddenLayers + visionFeatureLayer
                : visionFeatureLayer
            if index >= 0 && index < vision.numHiddenLayers {
                layers = index + 1
            }
        }
        self.init(
            hiddenSize: vision.hiddenSize, intermediateSize: vision.intermediateSize,
            numHiddenLayers: layers, numAttentionHeads: vision.numAttentionHeads,
            numChannels: vision.numChannels, numPatches: vision.numPatches,
            patchSize: vision.patchSize, layerNormEps: vision.layerNormEps)
    }
}

// MARK: - Configuration

/// Configuration for ``LFM2VL``
public struct LFM2VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _normEps: Float?
        public var normEps: Float { _normEps ?? 1e-5 }
        private let _convBias: Bool?
        public var convBias: Bool { _convBias ?? false }
        private let _convLCache: Int?
        public var convLCache: Int { _convLCache ?? 3 }
        private let _blockDim: Int?
        public var blockDim: Int { _blockDim ?? hiddenSize }
        private let _blockFFDim: Int?
        private let _intermediateSize: Int?
        /// MLP width before auto-adjustment: `block_ff_dim`, else `intermediate_size`.
        public var blockFFDim: Int { _blockFFDim ?? _intermediateSize ?? hiddenSize }
        private let _blockMultipleOf: Int?
        public var blockMultipleOf: Int { _blockMultipleOf ?? 256 }
        private let _blockFFNDimMultiplier: Double?
        public var blockFFNDimMultiplier: Float { Float(_blockFFNDimMultiplier ?? 1.0) }
        private let _blockAutoAdjustFFDim: Bool?
        public var blockAutoAdjustFFDim: Bool { _blockAutoAdjustFFDim ?? true }
        private let _tieWordEmbeddings: Bool?
        private let _tieEmbedding: Bool?
        /// `text_config`'s `tie_embedding`, else `tie_word_embeddings`, else true. ``LFM2VL``
        /// uses `LFM2VLConfiguration.tieWordEmbeddings`, which reads the top level first.
        var tieWordEmbeddings: Bool { _tieEmbedding ?? _tieWordEmbeddings ?? true }

        /// MLP width after transformers' `block_auto_adjust_ff_dim` rule. transformers skips the
        /// rounding when `block_ffn_dim_multiplier` is an explicit null; this rounds anyway.
        var feedForwardDimensions: Int {
            var width = blockFFDim
            if blockAutoAdjustFFDim {
                width = Int(Double(2 * width) / 3)
                if let multiplier = _blockFFNDimMultiplier {
                    width = Int(multiplier * Double(width))
                }
                width = blockMultipleOf * ((width + blockMultipleOf - 1) / blockMultipleOf)
            }
            return width
        }
        private let _fullAttnIdxs: [Int]?
        private let layerTypes: [String]?
        public var fullAttnIdxs: [Int] {
            if let fullAttnIdxs = _fullAttnIdxs {
                return fullAttnIdxs
            }

            if let layerTypes {
                return layerTypes.enumerated().compactMap { index, layerType in
                    layerType == "full_attention" ? index : nil
                }
            }

            return Array(0 ..< hiddenLayers)
        }
        private struct RopeParameters: Codable, Sendable {
            let ropeTheta: Float?
            enum CodingKeys: String, CodingKey {
                case ropeTheta = "rope_theta"
            }
        }
        private let _ropeTheta: Float?
        private let _ropeParameters: RopeParameters?
        /// `rope_parameters.rope_theta`, else `rope_theta`, else 1e6.
        public var ropeTheta: Float { _ropeParameters?.ropeTheta ?? _ropeTheta ?? 1000000.0 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _normEps = "norm_eps"
            case _convBias = "conv_bias"
            case _convLCache = "conv_L_cache"
            case _blockDim = "block_dim"
            case _blockFFDim = "block_ff_dim"
            case _intermediateSize = "intermediate_size"
            case _blockMultipleOf = "block_multiple_of"
            case _blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case _blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case _fullAttnIdxs = "full_attn_idxs"
            case layerTypes = "layer_types"
            case _ropeTheta = "rope_theta"
            case _ropeParameters = "rope_parameters"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _tieEmbedding = "tie_embedding"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _imageSize: Int?
        /// `image_size` from config.json. The tower does not use it.
        public var imageSize: Int { _imageSize ?? 224 }
        private let _patchSize: Int?
        public var patchSize: Int { _patchSize ?? 16 }
        private let _numPatches: Int?
        public var numPatches: Int { _numPatches ?? 256 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numChannels = "num_channels"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _numPatches = "num_patches"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _downsampleFactor: Int?
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }
    private let _imageTokenId: Int?
    public var imageTokenIndex: Int { _imageTokenId ?? 396 }
    private let _projectorBias: Bool?
    public var projectorBias: Bool { _projectorBias ?? true }
    private let _projectorHiddenSize: Int?
    public var projectorHiddenSize: Int { _projectorHiddenSize ?? 2560 }
    private let _projectorUseLayernorm: Bool?
    public var projectorUseLayernorm: Bool { _projectorUseLayernorm ?? true }
    private let _visionFeatureLayer: Int?
    /// Which vision encoder layer to use for features. -1 means use all layers (default).
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _doImageSplitting: Bool?
    /// `do_image_splitting` from config.json; the processor does not read it.
    public var doImageSplitting: Bool { _doImageSplitting ?? true }
    private let _maxImageTokens: Int?
    /// `max_image_tokens` from config.json; the processor does not read it.
    public var maxImageTokens: Int { _maxImageTokens ?? 256 }
    private let _maxNumPatches: Int?
    /// `max_num_patches` from config.json; the processor does not read it.
    public var maxNumPatches: Int { _maxNumPatches ?? 1024 }
    private let _minImageTokens: Int?
    /// `min_image_tokens` from config.json; the processor does not read it.
    public var minImageTokens: Int { _minImageTokens ?? 64 }
    private let _minTiles: Int?
    /// `min_tiles` from config.json; the processor does not read it.
    public var minTiles: Int { _minTiles ?? 2 }
    private let _useThumbnail: Bool?
    /// `use_thumbnail` from config.json; the processor does not read it.
    public var useThumbnail: Bool { _useThumbnail ?? false }
    private let _tieWordEmbeddings: Bool?
    private let _tieEmbedding: Bool?
    /// Whether the output projection reuses `embed_tokens`: the top-level `tie_embedding`, then
    /// `tie_word_embeddings`, then the `text_config` flags, where mlx-vlm conversions keep them.
    var tieWordEmbeddings: Bool {
        _tieEmbedding ?? _tieWordEmbeddings ?? textConfiguration.tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _downsampleFactor = "downsample_factor"
        case _imageTokenId = "image_token_id"
        case _projectorBias = "projector_bias"
        case _projectorHiddenSize = "projector_hidden_size"
        case _projectorUseLayernorm = "projector_use_layernorm"
        case _visionFeatureLayer = "vision_feature_layer"
        case _doImageSplitting = "do_image_splitting"
        case _maxImageTokens = "max_image_tokens"
        case _maxNumPatches = "max_num_patches"
        case _minImageTokens = "min_image_tokens"
        case _minTiles = "min_tiles"
        case _useThumbnail = "use_thumbnail"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _tieEmbedding = "tie_embedding"
    }
}

/// Configuration for ``LFM2VLProcessor``.
///
/// Decodes the image processor settings, flat or nested under `image_processor`.
/// ``VLMModelFactory`` reads them as mlx-vlm does: `processor_config.json` overrides
/// `preprocessor_config.json`, a missing `resample` is bicubic, and the tiling settings
/// (`do_image_splitting`, `min_tiles`, `max_tiles`, `use_thumbnail`) keep their defaults.
public struct LFM2VLProcessorConfiguration: Codable, Sendable {
    let settings: LFM2VLImageSettings

    init(settings: LFM2VLImageSettings) {
        self.settings = settings
    }

    /// `image_mean`, one value per channel. Kept for compatibility.
    public var imageMean: [CGFloat] { Self.channels(settings.imageMean) }
    /// `image_std`, one value per channel. Kept for compatibility.
    public var imageStd: [CGFloat] { Self.channels(settings.imageStd) }
    /// `tile_size`. Kept for compatibility.
    public var tileSize: Int { settings.tileSize }
    /// `encoder_patch_size`. Kept for compatibility.
    public var encoderPatchSize: Int { settings.encoderPatchSize }
    /// `max_tiles`. Kept for compatibility.
    public var maxTiles: Int { settings.maxTiles }
    /// `downsample_factor`. Kept for compatibility.
    public var downsampleFactor: Int { settings.downsampleFactor }

    /// ``imageMean`` as a tuple. Kept for compatibility.
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    /// ``imageStd`` as a tuple. Kept for compatibility.
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    /// One value per RGB channel; a single value applies to all three.
    private static func channels(_ values: [Double]) -> [CGFloat] {
        let values = values.map { CGFloat($0) }
        return values.count == 1 ? Array(repeating: values[0], count: 3) : values
    }

    public init(from decoder: any Decoder) throws {
        settings = try LFM2VLImageSettings(from: decoder)
    }

    /// Writes the settings flat, in the form ``init(from:)`` reads back.
    public func encode(to encoder: any Encoder) throws {
        try settings.encode(to: encoder)
    }
}

// MARK: - Chat conventions

extension LFM2VL {
    public var toolCallFormat: ToolCallFormat? { .lfm2 }
}
