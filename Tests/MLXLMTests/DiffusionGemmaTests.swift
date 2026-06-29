import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

struct DiffusionGemmaTests {
    @Test("DiffusionGemma decodes released config shape")
    func decodesReleasedConfigShape() throws {
        let config = try Self.configuration()
        let model = DiffusionGemmaLanguageCore(config)

        #expect(model.vocabularySize == 32)
        #expect(model.diffusionCanvasLength == 4)
        #expect(model.diffusionMaxDenoisingSteps == 3)
        #expect(model.diffusionEntropyBound == 0.2)
        #expect(model.diffusionTemperatureMin == 0.3)
        #expect(model.diffusionTemperatureMax == 0.7)
        #expect(model.diffusionStabilityThreshold == 2)
        #expect(model.diffusionConfidenceThreshold == 0.01)
        #expect(model.diffusionDefaultMaxTokens == 5)
        #expect(model.capabilities.contains(.blockDiffusion))
        #expect(model.newCache(parameters: nil).count == 2)
    }

    @Test("DiffusionGemma produces canvas logits from encoder cache")
    func producesCanvasLogits() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        eval(model)

        let cache = model.newCache(parameters: nil)
        let prompt = LMInput(tokens: MLXArray([2, 7, 11]).reshaped([1, 3]))
        try model.prepareDiffusion(prompt, cache: cache, windowSize: nil)
        #expect(cache.allSatisfy { $0.offset == 3 })

        let canvas = MLXArray([4, 5, 6, 7]).reshaped([1, 4])
        let logits = model.diffusionLogits(
            canvasTokens: canvas,
            cache: cache,
            selfConditioningLogits: nil)
        eval(logits)

        #expect(logits.shape == [1, 4, 32])

        model.acceptDiffusionTokens(canvas, cache: cache, windowSize: 2)
        #expect(cache.allSatisfy { $0.offset == 7 })
    }

    @Test("DiffusionGemma rejects autoregressive prepare path")
    func rejectsAutoregressivePreparePath() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        let cache = model.newCache(parameters: nil)
        let prompt = LMInput(tokens: MLXArray([2, 7, 11]).reshaped([1, 3]))

        do {
            _ = try model.prepare(prompt, cache: cache, windowSize: nil)
            #expect(Bool(false))
        } catch GenerateError.unsupportedAutoregressiveGeneration(let modelName) {
            #expect(modelName.contains("DiffusionGemmaLanguageCore"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("DiffusionGemma rejects speculative decoding")
    func rejectsSpeculativeDecoding() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        let prompt = LMInput(tokens: MLXArray([2, 7, 11]).reshaped([1, 3]))

        do {
            _ = try SpeculativeTokenIterator(
                input: prompt,
                mainModel: model,
                draftModel: model,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                numDraftTokens: 2)
            #expect(Bool(false))
        } catch GenerateError.unsupportedSpeculativeDecoding(let modelName) {
            #expect(modelName.contains("DiffusionGemmaLanguageCore"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("DiffusionGemma language core rejects direct multimodal input")
    func languageCoreRejectsDirectMultimodalInput() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        let cache = model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(tokens: MLXArray([2, 7, 11]).reshaped([1, 3])),
            image: .init(pixels: MLXArray.zeros([1, 3, 16, 16])))

        do {
            try model.prepareDiffusion(input, cache: cache, windowSize: nil)
            #expect(Bool(false))
        } catch GenerateError.unsupportedMultimodalGeneration(let modelName) {
            #expect(modelName.contains("DiffusionGemmaLanguageCore"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("DiffusionGemma iterator batch-flushes emitted tokens into cache")
    func iteratorBatchFlushesEmittedTokens() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        eval(model)

        let cache = model.newCache(parameters: nil)
        let prompt = LMInput(tokens: MLXArray([2, 7, 11]).reshaped([1, 3]))
        var iterator = try BlockDiffusionTokenIterator(
            input: prompt,
            model: model,
            cache: cache,
            parameters: GenerateParameters(maxTokens: 4, temperature: 0, prefillStepSize: 2))

        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens.count == 4)
        #expect(cache.allSatisfy { $0.offset == 7 })
    }

    @Test("DiffusionGemma keeps official shared text checkpoint layout")
    func keepsOfficialSharedTextCheckpointLayout() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        let tensor = MLXArray.ones([1])
        let weights = [
            "model.decoder.embed_tokens.weight": tensor,
            "model.decoder.embed_tokens.scales": tensor,
            "model.decoder.norm.weight": tensor,
            "model.decoder.layers.0.self_attn.q_proj.weight": tensor,
            "model.encoder.language_model.layers.0.layer_scalar": tensor,
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight": tensor,
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight": tensor,
            "lm_head.weight": tensor,
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.decoder.embed_tokens.weight"] != nil)
        #expect(sanitized["model.decoder.embed_tokens.scales"] != nil)
        #expect(sanitized["model.decoder.norm.weight"] != nil)
        #expect(sanitized["model.decoder.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["model.encoder.language_model.layers.0.layer_scalar"] != nil)
        #expect(sanitized["model.encoder.language_model.layers.0.self_attn.q_proj.weight"] == nil)
        #expect(
            sanitized["model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
    }

    @Test("DiffusionGemma strict-loads official tied text checkpoint layout")
    func strictLoadsOfficialTiedTextCheckpointLayout() throws {
        let model = DiffusionGemmaLanguageCore(try Self.configuration())
        var officialStyleWeights = [String: MLXArray]()

        for (key, value) in model.parameters().flattened() {
            if key.hasPrefix("model.decoder.") {
                officialStyleWeights[key] = value
            } else if key.hasPrefix("model.encoder.language_model.layers.")
                && key.hasSuffix(".layer_scalar")
            {
                officialStyleWeights[key] = value
            }
        }

        let sanitized = model.sanitize(weights: officialStyleWeights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    @Test("DiffusionGemma VLM decodes released processor config")
    func vlmDecodesReleasedProcessorConfig() throws {
        let json = """
            {
              "processor_class": "Gemma4Processor",
              "image_processor": {
                "do_normalize": false,
                "image_mean": [0.0, 0.0, 0.0],
                "image_std": [1.0, 1.0, 1.0],
                "image_seq_length": 280,
                "max_soft_tokens": 280
              },
              "video_processor": {
                "do_normalize": true,
                "max_soft_tokens": 70,
                "num_frames": 32
              }
            }
            """
        let config = try JSONDecoder().decode(
            DiffusionGemma4ProcessorConfiguration.self, from: Data(json.utf8))

        #expect(config.processorClass == "Gemma4Processor")
        #expect(config.imageSeqLength == 280)
        #expect(config.videoSeqLength == 70)
        #expect(config.videoFrameLimit == 32)
        #expect(config.imageTokenId == 258_880)
    }

    @Test("DiffusionGemma VLM processor expands image tokens and token types")
    func vlmProcessorExpandsImageTokensAndTokenTypes() async throws {
        let config = try JSONDecoder().decode(
            DiffusionGemma4ProcessorConfiguration.self,
            from: Data(
                #"{"processor_class":"DiffusionGemma4Processor","image_seq_length":4}"#.utf8))
        let processor = DiffusionGemma4Processor(
            config,
            tokenizer: DiffusionGemmaPromptTokenizer(tokens: [7, 258_880, 9]))
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        #expect(
            input.text.tokens.asArray(Int.self) == [
                7, 255_999, 258_880, 258_880, 258_880, 258_880, 258_882, 9,
            ])
        #expect(input.multimodalTokenTypes?.asArray(Int32.self) == [0, 0, 1, 1, 1, 1, 0, 0])
        #expect(input.image?.pixels.shape == [1, 3, 224, 224])
    }

    @Test("DiffusionGemma VLM processor expands video tokens and token types")
    func vlmProcessorExpandsVideoTokensAndTokenTypes() async throws {
        let config = try JSONDecoder().decode(
            DiffusionGemma4ProcessorConfiguration.self,
            from: Data(
                #"{"processor_class":"DiffusionGemma4Processor","image_seq_length":4,"video_processor":{"max_soft_tokens":3,"num_frames":1}}"#
                    .utf8))
        let videoTokenId = 262_143
        let processor = DiffusionGemma4Processor(
            config,
            tokenizer: DiffusionGemmaPromptTokenizer(
                tokens: [7, videoTokenId, 9], videoTokenId: videoTokenId))
        let frame = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let video = UserInput.Video.frames([.init(frame: frame, timeStamp: .zero)])

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", videos: [video]))

        #expect(
            input.text.tokens.asArray(Int.self) == [
                7, 255_999, videoTokenId, videoTokenId, videoTokenId, 258_882, 9,
            ])
        #expect(input.multimodalTokenTypes?.asArray(Int32.self) == [0, 0, 2, 2, 2, 0, 0])
        #expect(input.video?.pixels.shape == [1, 3, 224, 224])
    }

    @Test("DiffusionGemma VLM processor preserves mixed visual prompt order")
    func vlmProcessorPreservesMixedVisualPromptOrder() async throws {
        let config = try JSONDecoder().decode(
            DiffusionGemma4ProcessorConfiguration.self,
            from: Data(
                #"{"processor_class":"DiffusionGemma4Processor","image_seq_length":2,"video_processor":{"max_soft_tokens":3,"num_frames":1}}"#
                    .utf8))
        let videoTokenId = 262_143
        let processor = DiffusionGemma4Processor(
            config,
            tokenizer: DiffusionGemmaPromptTokenizer(
                tokens: [7, videoTokenId, 8, 258_880, 9], videoTokenId: videoTokenId))
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let frame = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let video = UserInput.Video.frames([.init(frame: frame, timeStamp: .zero)])

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)], videos: [video]))

        #expect(
            input.text.tokens.asArray(Int.self) == [
                7, 255_999, videoTokenId, videoTokenId, videoTokenId, 258_882, 8, 255_999, 258_880,
                258_880, 258_882, 9,
            ])
        #expect(
            input.multimodalTokenTypes?.asArray(Int32.self) == [0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0])
    }

    @Test("DiffusionGemma VLM processor preserves separate video placeholders")
    func vlmProcessorPreservesSeparateVideoPlaceholders() async throws {
        let config = try JSONDecoder().decode(
            DiffusionGemma4ProcessorConfiguration.self,
            from: Data(
                #"{"processor_class":"DiffusionGemma4Processor","video_processor":{"max_soft_tokens":2,"num_frames":2}}"#
                    .utf8))
        let videoTokenId = 262_143
        let processor = DiffusionGemma4Processor(
            config,
            tokenizer: DiffusionGemmaPromptTokenizer(
                tokens: [7, videoTokenId, 8, videoTokenId, 9], videoTokenId: videoTokenId))
        let blueFrame = CIImage(color: .blue).cropped(
            to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let greenFrame = CIImage(color: .green).cropped(
            to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let firstVideo = UserInput.Video.frames([.init(frame: blueFrame, timeStamp: .zero)])
        let secondVideo = UserInput.Video.frames([.init(frame: greenFrame, timeStamp: .zero)])

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", videos: [firstVideo, secondVideo]))

        #expect(
            input.text.tokens.asArray(Int.self) == [
                7, 255_999, videoTokenId, videoTokenId, 258_882, 8, 255_999, videoTokenId,
                videoTokenId, 258_882, 9,
            ])
        #expect(
            input.multimodalTokenTypes?.asArray(Int32.self) == [
                0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0,
            ])
        #expect(input.video?.pixels.shape == [2, 3, 224, 224])
    }

    @Test("TokenIterator preserves state returned from logits prefill")
    func tokenIteratorPreservesLogitsPrefillState() throws {
        let model = LogitsPrefillStateModel()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([0]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0))

        _ = iterator.next()

        #expect(model.sawPrefillState)
    }

    @Test("DiffusionGemma VLM registry creates diffusion model")
    func vlmRegistryCreatesDiffusionModel() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(Self.vlmConfigurationJSON.utf8),
            modelType: "diffusion_gemma")

        #expect(model is DiffusionGemma)
    }

    @Test("DiffusionGemma VLM sanitizer keeps text and vision checkpoint layout")
    func vlmSanitizerKeepsTextAndVisionCheckpointLayout() throws {
        let config = try JSONDecoder().decode(
            DiffusionGemmaVLMConfiguration.self, from: Data(Self.vlmConfigurationJSON.utf8))
        let model = DiffusionGemma(config)
        let tensor = MLXArray.ones([1])
        let weights = [
            "model.decoder.embed_tokens.weight": tensor,
            "model.encoder.language_model.layers.0.layer_scalar": tensor,
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight": tensor,
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight": tensor,
            "model.encoder.embed_vision.embedding_projection.weight": tensor,
            "model.encoder.audio_tower.layers.0.weight": tensor,
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["diffusion_core.model.decoder.embed_tokens.weight"] != nil)
        #expect(
            sanitized["diffusion_core.model.encoder.language_model.layers.0.layer_scalar"] != nil)
        #expect(
            sanitized[
                "diffusion_core.model.encoder.language_model.layers.0.self_attn.q_proj.weight"]
                == nil)
        #expect(sanitized["vision_tower.encoder.layers.0.input_layernorm.weight"] != nil)
        #expect(sanitized["embed_vision.embedding_projection.weight"] != nil)
        #expect(sanitized["model.encoder.audio_tower.layers.0.weight"] == nil)
    }

    @Test("Block diffusion iterator emits stable argmax canvas")
    func blockDiffusionIteratorEmitsStableArgmaxCanvas() throws {
        let model = StableCanvasDiffusionModel()
        var iterator = try BlockDiffusionTokenIterator(
            input: LMInput(tokens: MLXArray([9]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(maxTokens: 3, temperature: 0))

        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == [1, 2, 3])
        #expect(model.decoderCalls == 2)
    }

    @Test("Block diffusion iterator uses configured default max tokens")
    func blockDiffusionIteratorUsesConfiguredDefaultMaxTokens() throws {
        let model = StableCanvasDiffusionModel(defaultMaxTokens: 2)
        var iterator = try BlockDiffusionTokenIterator(
            input: LMInput(tokens: MLXArray([9]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(temperature: 0))

        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == [1, 2])
    }

    @Test("Block diffusion iterator uses mlx-vlm default minimum canvas")
    func blockDiffusionIteratorUsesDefaultMinimumCanvas() throws {
        let model = StableCanvasDiffusionModel(canvasLength: 5)
        var iterator = try BlockDiffusionTokenIterator(
            input: LMInput(tokens: MLXArray([9]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0))

        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == [1, 2])
        #expect(model.requestedCanvasLengths.first == 5)
    }

    @Test("Block diffusion iterator supports confidence-threshold sampler")
    func blockDiffusionIteratorSupportsConfidenceThresholdSampler() throws {
        let model = StableCanvasDiffusionModel()
        var iterator = try BlockDiffusionTokenIterator(
            input: LMInput(tokens: MLXArray([9]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(
                maxTokens: 3,
                temperature: 0,
                diffusionSampler: .confidenceThreshold))

        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == [1, 2, 3])
    }

    @Test("Block diffusion iterator does not apply autoregressive logit processors")
    func blockDiffusionIteratorDoesNotApplyAutoregressiveLogitProcessors() throws {
        let model = PenaltySensitiveDiffusionModel()
        var iterator = try BlockDiffusionTokenIterator(
            input: LMInput(tokens: MLXArray([1]).reshaped([1, 1])),
            model: model,
            parameters: GenerateParameters(
                maxTokens: 1,
                temperature: 0,
                repetitionPenalty: 10,
                diffusionMinCanvasLength: 1,
                diffusionMaxCanvasLength: 1))

        #expect(iterator.next() == 1)
        #expect(iterator.next() == nil)
    }

    private static func configuration() throws -> DiffusionGemmaConfiguration {
        let json = """
            {
              "model_type": "diffusion_gemma",
              "canvas_length": 4,
              "generation_config": {
                "confidence_threshold": 0.01,
                "max_denoising_steps": 3,
                "max_new_tokens": 5,
                "sampler_config": {
                  "_cls_name": "EntropyBoundSamplerConfig",
                  "entropy_bound": 0.2
                },
                "stability_threshold": 2,
                "t_max": 0.7,
                "t_min": 0.3
              },
              "text_config": {
                "model_type": "diffusion_gemma_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "moe_intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "vocab_size": 32,
                "sliding_window": 8,
                "rms_norm_eps": 0.000001,
                "final_logit_softcapping": 30.0,
                "num_experts": 4,
                "top_k_experts": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                  "full_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                  },
                  "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                  }
                }
              }
            }
            """
        return try JSONDecoder().decode(DiffusionGemmaConfiguration.self, from: Data(json.utf8))
    }

    private static let vlmConfigurationJSON = """
        {
          "model_type": "diffusion_gemma",
          "canvas_length": 4,
          "text_config": {
            "model_type": "diffusion_gemma_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "sliding_window": 8,
            "rms_norm_eps": 0.000001,
            "num_experts": 4,
            "top_k_experts": 2,
            "layer_types": ["sliding_attention", "full_attention"]
          },
          "vision_config": {
            "model_type": "gemma4_vision",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "patch_size": 16,
            "default_output_length": 4,
            "position_embedding_size": 16,
            "pooling_kernel_size": 1,
            "rms_norm_eps": 0.000001
          },
          "vision_soft_tokens_per_image": 4
        }
        """
}

private struct DiffusionGemmaPromptTokenizer: Tokenizer {
    let tokens: [Int]
    var videoTokenId: Int? = nil
    var vocabularySize: Int { 262_144 }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { 1 }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { tokens }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? {
        token == "<|video|>" ? videoTokenId : nil
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        tokens
    }
}

private final class PenaltySensitiveDiffusionModel: Module, BlockDiffusionLanguageModel {
    let diffusionCanvasLength = 1
    let diffusionMinimumCanvasLength = 1
    let diffusionMaxDenoisingSteps = 1
    let diffusionEntropyBound: Float = 0.1
    let diffusionVocabularySize = 3

    func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {}

    func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {}

    func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        MLXArray([Float(1), Float(2), Float(0)]).reshaped([1, 1, diffusionVocabularySize])
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, diffusionVocabularySize])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

private final class StableCanvasDiffusionModel: Module, BlockDiffusionLanguageModel {
    let diffusionCanvasLength: Int
    let diffusionMinimumCanvasLength: Int
    let diffusionMaxDenoisingSteps = 4
    let diffusionEntropyBound: Float = 0.1
    let diffusionVocabularySize = 5
    let diffusionDefaultMaxTokens: Int?
    var decoderCalls = 0
    var requestedCanvasLengths = [Int]()

    init(
        canvasLength: Int = 3,
        minimumCanvasLength: Int = 64,
        defaultMaxTokens: Int? = nil
    ) {
        self.diffusionCanvasLength = canvasLength
        self.diffusionMinimumCanvasLength = minimumCanvasLength
        self.diffusionDefaultMaxTokens = defaultMaxTokens
        super.init()
    }

    func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {}

    func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {}

    func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        decoderCalls += 1
        let length = canvasTokens.dim(1)
        requestedCanvasLengths.append(length)
        var values = Array(repeating: Float(-20), count: length * diffusionVocabularySize)
        for (position, token) in [1, 2, 3].prefix(length).enumerated() {
            values[position * diffusionVocabularySize + token] = 20
        }
        return MLXArray(values).reshaped([1, length, diffusionVocabularySize])
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, diffusionVocabularySize])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

private final class LogitsPrefillStateModel: Module, LanguageModel {
    static let key = LMOutput.Key<Int>("logits-prefill-state")

    var sawPrefillState = false

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        var state = LMOutput.State()
        state[Self.key] = 42
        return .logits(LMOutput(logits: Self.logits(for: 1), state: state))
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        sawPrefillState = state?[Self.key] == 42
        return LMOutput(logits: Self.logits(for: 2), state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private static func logits(for token: Int) -> MLXArray {
        var values = Array(repeating: Float(-20), count: 4)
        values[token] = 20
        return MLXArray(values).reshaped([1, 1, 4])
    }
}
