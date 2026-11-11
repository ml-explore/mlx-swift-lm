// LFM3-VL architecture from LiquidAI/LFM3-VL-2B.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LFM3VLConfiguration: Decodable, Sendable {
    let text: LFM3Configuration
    let vision: LFM3VisionConfiguration
    let imageToken: Int
    let videoToken: Int
    let downsample: Int
    let projectorSize: Int
    let projectorBias: Bool
    let projectorEps: Float

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case imageToken = "image_token_id"
        case videoToken = "video_token_id"
        case downsample = "downsample_factor"
        case projectorSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorEps = "projector_layer_norm_eps"
        case activation = "projector_hidden_act"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(LFM3Configuration.self, forKey: .text)
        vision = try c.decode(LFM3VisionConfiguration.self, forKey: .vision)
        imageToken = try c.decode(Int.self, forKey: .imageToken)
        videoToken = try c.decode(Int.self, forKey: .videoToken)
        downsample = try c.decode(Int.self, forKey: .downsample)
        projectorSize = try c.decode(Int.self, forKey: .projectorSize)
        projectorBias = try c.decode(Bool.self, forKey: .projectorBias)
        projectorEps = try c.decode(Float.self, forKey: .projectorEps)
        let activation = try c.decode(String.self, forKey: .activation)
        guard activation == "gelu", downsample == vision.mergeSize,
            downsample <= Int.max / downsample / vision.hiddenSize,
            projectorSize > 0, projectorEps.isFinite, projectorEps > 0,
            imageToken >= 0, imageToken < text.vocabularySize,
            videoToken >= 0, videoToken < text.vocabularySize, imageToken != videoToken
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid LFM3-VL projector or media token configuration"))
        }
    }
}

final class LFM3VisionProjector: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear
    let inputSize: Int

    init(_ c: LFM3VLConfiguration) {
        inputSize = c.vision.hiddenSize * c.downsample * c.downsample
        _norm.wrappedValue = LayerNorm(dimensions: c.vision.hiddenSize, eps: c.projectorEps)
        _first.wrappedValue = Linear(inputSize, c.projectorSize, bias: c.projectorBias)
        _second.wrappedValue = Linear(c.projectorSize, c.text.hiddenSize, bias: c.projectorBias)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        second(gelu(first(norm(features).reshaped(-1, inputSize))))
    }
}

final class LFM3VLLanguageModel: Module {
    let model: LFM3ModelInner
    @ModuleInfo(key: "lm_head") var head: Linear?

    init(_ c: LFM3Configuration) {
        model = LFM3ModelInner(c)
        if !c.tieWordEmbeddings {
            _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        model.projectLogits(hidden, lmHead: head)
    }
}

final class LFM3VLVision: Module {
    @ModuleInfo(key: "vision_tower") var tower: LFM3VisionModel
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM3VisionProjector

    init(_ config: LFM3VLConfiguration) {
        _tower.wrappedValue = LFM3VisionModel(config.vision)
        _projector.wrappedValue = LFM3VisionProjector(config)
    }
}

final class LFM3VL: Module, LanguageModel, KVCacheDimensionProvider, LoRAModel {
    let config: LFM3VLConfiguration
    @ModuleInfo(key: "language_model") var languageModel: LFM3VLLanguageModel
    @ModuleInfo(key: "vision_tower") private var vision: LFM3VisionModel?
    @ModuleInfo(key: "multi_modal_projector") private var projector: LFM3VisionProjector?
    private let onDemandVision: OnDemandComponent<LFM3VLVision>

    var vocabularySize: Int { config.text.vocabularySize }
    var kvHeads: [Int] { config.text.layerTypes.map { $0 == .conv ? 0 : config.text.kvHeads } }
    var loraLayers: [Module] { languageModel.model.layers }
    var toolCallFormat: ToolCallFormat? { .xmlFunction }
    var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }

    init(_ c: LFM3VLConfiguration) {
        config = c
        _languageModel.wrappedValue = LFM3VLLanguageModel(c.text)
        let stack = LFM3VLVision(c)
        _vision.wrappedValue = stack.tower
        _projector.wrappedValue = stack.projector
        onDemandVision = OnDemandComponent(.vision) { LFM3VLVision(c) }
    }

    func validate(tokenizer: any Tokenizer) throws {
        try LFM3ModelInner.validate(tokenizer: tokenizer)
        for (token, id) in [("<|image|>", config.imageToken), ("<|video|>", config.videoToken)] {
            guard tokenizer.convertTokenToId(token) == id,
                tokenizer.encode(text: token, addSpecialTokens: false) == [id]
            else {
                throw VLMError.processing(
                    "LFM3-VL tokenizer does not match the configured media tokens")
            }
        }
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.newCache(parameters: parameters)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.logits(languageModel.model(inputs, cache: cache))
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.audio == nil else {
            throw VLMError.processing("LFM3-VL does not support audio input")
        }
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        guard tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0,
            cache.count == config.text.hiddenLayers
        else {
            throw VLMError.processing("LFM3-VL expects one nonempty prompt and one cache per layer")
        }
        if let mask = input.text.mask {
            guard mask.shape == tokens.shape, all(mask .!= 0).item(Bool.self) else {
                throw VLMError.processing("LFM3-VL requires an unpadded prompt")
            }
        }
        let ids = tokens.asArray(Int.self)
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw VLMError.processing("LFM3-VL token ID is outside the vocabulary")
        }
        var embeddings = languageModel.model.embed(tokens)
        func insert(pixels: MLXArray?, frames: [THW]?, token: Int) throws {
            let indices = ids.indices.filter { ids[$0] == token }
            guard let pixels else {
                guard indices.isEmpty else {
                    throw VLMError.processing("Missing LFM3-VL media for placeholder tokens")
                }
                return
            }
            guard let frames else { throw VLMError.processing("Missing LFM3-VL media grid") }
            let stack: (tower: LFM3VisionModel, projector: LFM3VisionProjector)
            if let vision, let projector {
                stack = (vision, projector)
            } else {
                let loaded = try onDemandVision.load()
                stack = (loaded.tower, loaded.projector)
            }
            let features = try stack.projector(stack.tower(pixels, frames: frames))
            guard indices.count == features.dim(0) else {
                throw VLMError.processing(
                    "LFM3-VL media token count does not match projected features")
            }
            embeddings[0, MLXArray(indices), 0...] = features.asType(embeddings.dtype)
        }
        try insert(
            pixels: input.image?.pixels, frames: input.image?.frames, token: config.imageToken)
        try insert(
            pixels: input.video?.pixels, frames: input.video?.frames, token: config.videoToken)
        let total = tokens.dim(1)
        let consumed = try prefill.forEachChunk(total: total, reserving: 1) { range in
            let hidden = languageModel.model(
                embeddings: embeddings[0..., range, 0...], cache: cache)
            eval(hidden, cache)
        }
        let hidden = languageModel.model(
            embeddings: embeddings[0..., consumed..., 0...], cache: cache)
        let projection: Module = languageModel.head ?? languageModel.model.embedTokens
        let logits = languageModel.logits(
            quantizedVocabularyProjectionInput(hidden, projection: projection))[0..., (-1)..., 0...]
        prefill.progress?(total, total)
        return .logits(.init(logits: logits))
    }

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames { name in
            var name = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
            if name.hasPrefix("language_model."), name != "language_model.model",
                name != "language_model.lm_head", !name.hasPrefix("language_model.model."),
                !name.hasPrefix("language_model.lm_head.")
            {
                name = "language_model.model." + name.dropFirst("language_model.".count)
            } else if name == "lm_head" || name.hasPrefix("lm_head.") {
                name = "language_model." + name
            }
            return name
        }
        checkpoint.weights = sanitize(weights: checkpoint.weights)
        return checkpoint
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        if config.text.tieWordEmbeddings {
            for key in result.keys where key.hasPrefix("language_model.lm_head.") {
                result.removeValue(forKey: key)
            }
        }
        for (key, value) in result {
            if key.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                result[key] = value.transposed(0, 2, 1)
            } else if key.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1 {
                result[key] = value.transposed(0, 2, 3, 1)
            }
        }
        return result
    }
}

extension LFM3VL: ModelComponentsProviding {
    var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [.vision: .modules("vision_tower", "multi_modal_projector")]
    }

    var onDemandComponents: [AnyOnDemandComponent] { [onDemandVision] }
}
