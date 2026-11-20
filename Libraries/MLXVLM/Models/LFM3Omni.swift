// LFM3-Omni architecture from LiquidAI/LFM3-Omni-2B.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private final class LFM3OmniAudio: Module {
    @ModuleInfo(key: "audio_encoder") var encoder: LFM3OmniAudioEncoder
    @ModuleInfo(key: "audio_projector") var projector: LFM3OmniAudioProjector

    init(_ config: LFM3OmniConfiguration) {
        _encoder.wrappedValue = LFM3OmniAudioEncoder(config.audioEncoder)
        _projector.wrappedValue = LFM3OmniAudioProjector(
            inputSize: config.audioEncoder.hiddenSize,
            outputSize: config.visionLanguage.text.hiddenSize)
    }
}

final class LFM3OmniSpeech: Module {
    @ModuleInfo(key: "depth_decoder") var decoder: LFM3OmniDepthDecoder
    @ModuleInfo(key: "codebook_embedding") var embedding: Embedding
    @ModuleInfo(key: "audio_tokenizer") var codec: LiquidCodec

    init(_ config: LFM3OmniConfiguration, codecConfiguration: LiquidCodecConfiguration) throws {
        _decoder.wrappedValue = LFM3OmniDepthDecoder(config)
        _embedding.wrappedValue = Embedding(
            embeddingCount: config.codebooks * config.codebookSize,
            dimensions: config.visionLanguage.text.hiddenSize)
        _codec.wrappedValue = try LiquidCodec(codecConfiguration)
    }
}

final class LFM3Omni: Module, LanguageModel, KVCacheDimensionProvider, LoRAModel {
    let config: LFM3OmniConfiguration
    @ModuleInfo(key: "language_model") var languageModel: LFM3VLLanguageModel
    @ModuleInfo(key: "vision_tower") private var vision: LFM3VisionModel?
    @ModuleInfo(key: "multi_modal_projector") private var projector: LFM3VisionProjector?
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: LFM3OmniAudioEncoder?
    @ModuleInfo(key: "audio_projector") private var audioProjector: LFM3OmniAudioProjector?
    @ModuleInfo(key: "depth_decoder") private var depthDecoder: LFM3OmniDepthDecoder?
    @ModuleInfo(key: "codebook_embedding") private var codebookEmbedding: Embedding?

    @ModuleInfo(key: "audio_tokenizer") private var codec: LiquidCodec?
    private let onDemandVision: OnDemandComponent<LFM3VLVision>
    private let onDemandAudio: OnDemandComponent<LFM3OmniAudio>
    private let onDemandSpeech: OnDemandComponent<LFM3OmniSpeech>

    var vocabularySize: Int { config.visionLanguage.text.vocabularySize }
    var kvHeads: [Int] {
        config.visionLanguage.text.layerTypes.map {
            $0 == .conv ? 0 : config.visionLanguage.text.kvHeads
        }
    }
    var loraLayers: [Module] { languageModel.model.layers }
    var toolCallFormat: ToolCallFormat? { .xmlFunction }

    init(
        _ config: LFM3OmniConfiguration,
        codecConfiguration: LiquidCodecConfiguration = .lfm3Omni
    ) throws {
        try codecConfiguration.validate()
        guard codecConfiguration.codebooks == config.codebooks,
            codecConfiguration.levels.reduce(1, *) == config.audioEndCode,
            config.codebookSize == config.audioEndCode + 1
        else { throw VLMError.processing("LiquidCodec codebooks do not match the Omni model") }
        self.config = config
        _languageModel.wrappedValue = LFM3VLLanguageModel(config.visionLanguage.text)
        let vision = LFM3VLVision(config.visionLanguage)
        _vision.wrappedValue = vision.tower
        _projector.wrappedValue = vision.projector
        let audio = LFM3OmniAudio(config)
        _audioEncoder.wrappedValue = audio.encoder
        _audioProjector.wrappedValue = audio.projector
        let speech = try LFM3OmniSpeech(config, codecConfiguration: codecConfiguration)
        _depthDecoder.wrappedValue = speech.decoder
        _codebookEmbedding.wrappedValue = speech.embedding
        _codec.wrappedValue = speech.codec
        onDemandVision = OnDemandComponent(.vision) { LFM3VLVision(config.visionLanguage) }
        onDemandAudio = OnDemandComponent(.audio) { LFM3OmniAudio(config) }
        onDemandSpeech = OnDemandComponent(.speech) {
            try LFM3OmniSpeech(config, codecConfiguration: codecConfiguration)
        }
    }

    func speech() throws -> (
        decoder: LFM3OmniDepthDecoder, embedding: Embedding, codec: LiquidCodec
    ) {
        if let depthDecoder, let codebookEmbedding, let codec {
            return (depthDecoder, codebookEmbedding, codec)
        }
        let stack = try onDemandSpeech.load()
        return (stack.decoder, stack.embedding, stack.codec)
    }

    func validate(tokenizer: any Tokenizer) throws {
        try LFM3ModelInner.validate(tokenizer: tokenizer)
        for (token, id) in [
            ("<|image|>", config.visionLanguage.imageToken),
            ("<|video|>", config.visionLanguage.videoToken),
            ("<|audio|>", config.audioToken), ("<|audio_out|>", config.audioOutputToken),
            ("<|audio_start|>", config.audioStartToken),
            ("<|mixed_start|>", config.mixedStartToken),
            ("<|text_end|>", config.textEndToken),
        ] {
            guard tokenizer.convertTokenToId(token) == id,
                tokenizer.encode(text: token, addSpecialTokens: false) == [id]
            else {
                throw VLMError.processing("LFM3-Omni tokenizer does not match its media tokens")
            }
        }
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.newCache(parameters: parameters)
    }

    func codebookEmbeddings(_ codes: MLXArray) throws -> MLXArray {
        guard codes.ndim == 2, codes.dim(1) == config.codebooks,
            codes.dtype == .int32 || codes.dtype == .int64,
            all((codes .>= 0) .&& (codes .< config.codebookSize)).item(Bool.self)
        else { throw VLMError.processing("Invalid LFM3-Omni speech codebook indices") }
        let offsets = MLXArray(0 ..< config.codebooks) * config.codebookSize
        return try speech().embedding(codes + offsets).sum(axis: 1)
    }

    func embeddings(_ input: LMInput, audioOutputCodes: MLXArray? = nil) throws -> MLXArray {
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        guard tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0,
            tokens.dtype == .int32 || tokens.dtype == .int64
        else { throw VLMError.processing("LFM3-Omni expects one nonempty tokenized prompt") }
        if let mask = input.text.mask {
            guard mask.shape == tokens.shape, all(mask .!= 0).item(Bool.self) else {
                throw VLMError.processing("LFM3-Omni requires an unpadded prompt")
            }
        }
        let ids = tokens.asArray(Int.self)
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw VLMError.processing("LFM3-Omni token ID is outside the vocabulary")
        }
        let embeddings = languageModel.model.embed(tokens)
        func insert(token: Int, features: MLXArray?) throws {
            let positions = ids.indices.filter { ids[$0] == token }
            guard positions.count == (features?.dim(0) ?? 0) else {
                throw VLMError.processing(
                    "LFM3-Omni media placeholders do not match the supplied features")
            }
            if let features, !positions.isEmpty {
                embeddings[0, MLXArray(positions), 0...] = features.asType(embeddings.dtype)
            }
        }
        func visualFeatures(pixels: MLXArray?, grid: [THW]?) throws -> MLXArray? {
            guard let pixels else { return nil }
            guard let grid else { throw VLMError.processing("Missing LFM3-Omni media grid") }
            if let vision, let projector {
                return try projector(vision(pixels, frames: grid))
            }
            let stack = try onDemandVision.load()
            return try stack.projector(stack.tower(pixels, frames: grid))
        }
        try insert(
            token: config.visionLanguage.imageToken,
            features: visualFeatures(pixels: input.image?.pixels, grid: input.image?.frames))
        try insert(
            token: config.visionLanguage.videoToken,
            features: visualFeatures(pixels: input.video?.pixels, grid: input.video?.frames))
        var audio: MLXArray?
        if let prepared = input.audio {
            guard let mask = prepared.mask, prepared.features.ndim == 3,
                mask.shape == [prepared.features.dim(0), prepared.features.dim(2)],
                all((mask .== 0) .|| (mask .== 1)).item(Bool.self)
            else { throw VLMError.processing("LFM3-Omni audio requires a binary frame mask") }
            let lengths = mask.asType(.int32).sum(axis: -1)
            let expected = MLXArray(0 ..< mask.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]
            guard all(mask.asType(.bool) .== expected).item(Bool.self) else {
                throw VLMError.processing("LFM3-Omni audio masks must describe right padding")
            }
            if let audioEncoder, let audioProjector {
                audio = try audioProjector(audioEncoder(prepared.features, lengths: lengths))
            } else {
                let stack = try onDemandAudio.load()
                audio = try stack.projector(stack.encoder(prepared.features, lengths: lengths))
            }
        }
        try insert(token: config.audioToken, features: audio)
        try insert(
            token: config.audioOutputToken,
            features: audioOutputCodes.map { try codebookEmbeddings($0) })
        return embeddings
    }

    func prefillHidden(_ embeddings: MLXArray, cache: [KVCache], prefill: PrefillParameters) throws
        -> MLXArray
    {
        guard cache.count == config.visionLanguage.text.hiddenLayers else {
            throw VLMError.processing("LFM3-Omni requires one cache per language layer")
        }
        let total = embeddings.dim(1)
        let consumed = try prefill.forEachChunk(total: total, reserving: 1) { range in
            eval(
                languageModel.model(embeddings: embeddings[0..., range, 0...], cache: cache), cache)
        }
        let hidden = languageModel.model(
            embeddings: embeddings[0..., consumed..., 0...], cache: cache)
        prefill.progress?(total, total)
        return hidden
    }

    private func textLogits(_ hidden: MLXArray) -> MLXArray {
        let projection: Module = languageModel.head ?? languageModel.model.embedTokens
        let logits = languageModel.logits(
            quantizedVocabularyProjectionInput(hidden, projection: projection))
        // Ordinary text generation cannot consume speech codebooks.
        let suppressed = MLXArray([
            config.audioToken, config.audioOutputToken, config.audioStartToken,
            config.mixedStartToken, config.visionLanguage.imageToken,
            config.visionLanguage.videoToken,
        ])
        logits[0..., 0..., suppressed] = MLXArray(-Float.infinity)
        return logits
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        textLogits(languageModel.model(inputs, cache: cache))
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let hidden = try prefillHidden(embeddings(input), cache: cache, prefill: prefill)
        return .logits(.init(logits: textLogits(hidden)[0..., (-1)..., 0...]))
    }

    private static func checkpointName(_ original: String) -> String {
        var key = original.hasPrefix("model.") ? String(original.dropFirst(6)) : original
        if key.hasPrefix("language_model."), key != "language_model.model",
            key != "language_model.lm_head", !key.hasPrefix("language_model.model."),
            !key.hasPrefix("language_model.lm_head.")
        {
            key = "language_model.model." + key.dropFirst("language_model.".count)
        } else if key == "lm_head" || key.hasPrefix("lm_head.") {
            key = "language_model." + key
        } else if LiquidCodec.checkpointNamespaces.contains(where: {
            key == $0 || key.hasPrefix($0 + ".")
        }) {
            key = "audio_tokenizer." + key
        }
        return key
    }

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let rawCodec = checkpoint.weights.keys.contains(where: Self.isRawCodecWeight)
        var checkpoint = try checkpoint.mapNames(Self.checkpointName)
        checkpoint.weights = try sanitize(weights: checkpoint.weights, rawCodec: rawCodec)
        return checkpoint
    }

    private static func isRawCodecWeight(_ name: String) -> Bool {
        let name = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
        return LiquidCodec.checkpointNamespaces.contains { name.hasPrefix($0 + ".") }
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try sanitize(
            weights: weights, rawCodec: weights.keys.contains(where: Self.isRawCodecWeight))
    }

    private func sanitize(weights: [String: MLXArray], rawCodec: Bool) throws -> [String: MLXArray]
    {
        var result = [String: MLXArray]()
        var audioWeights = [String: MLXArray]()
        var codecWeights = [String: MLXArray]()
        for (original, value) in weights {
            let key = Self.checkpointName(original)
            guard result[key] == nil else {
                throw VLMError.processing("Duplicate normalized LFM3-Omni weight: \(key)")
            }
            result[key] = value
            if key.hasPrefix("audio_encoder.") {
                audioWeights[String(key.dropFirst("audio_encoder.".count))] = value
            } else if key.hasPrefix("audio_tokenizer.") {
                codecWeights[String(key.dropFirst("audio_tokenizer.".count))] =
                    value.dtype.isFloatingPoint ? value.asType(.float32) : value
            } else if key.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                result[key] = value.transposed(0, 2, 1)
            } else if key.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1 {
                result[key] = value.transposed(0, 2, 3, 1)
            }
        }
        if !audioWeights.isEmpty {
            let encoder = audioEncoder ?? LFM3OmniAudioEncoder(config.audioEncoder)
            for (key, value) in encoder.sanitize(weights: audioWeights) {
                result["audio_encoder." + key] = value
            }
        }
        let preparedCodec = rawCodec ? LiquidCodec.sanitizeWeights(codecWeights) : codecWeights
        for (key, value) in preparedCodec {
            result["audio_tokenizer." + key] = value
        }
        return filterLMHeadWeights(
            from: result, tiedWordEmbeddings: config.visionLanguage.text.tieWordEmbeddings)
    }

}

extension LFM3Omni: ModelComponentsProviding, AdditionalWeightFilesProviding {
    var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [
            .vision: .modules("vision_tower", "multi_modal_projector"),
            .audio: .modules("audio_encoder", "audio_projector"),
            .speech: .modules("depth_decoder", "codebook_embedding") + [
                CheckpointComponent(
                    name: "audio_tokenizer",
                    namespaces: ["audio_tokenizer", "model.audio_tokenizer"]
                        + LiquidCodec.checkpointNamespaces
                        + LiquidCodec.checkpointNamespaces.map { "model." + $0 },
                    destination: "audio_tokenizer")
            ],
        ]
    }

    var onDemandComponents: [AnyOnDemandComponent] {
        [onDemandVision, onDemandAudio, onDemandSpeech]
    }

    var additionalWeightFiles: [String] { ["audio_tokenizer/model.safetensors"] }
}
