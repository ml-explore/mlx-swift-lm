import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

struct DiffusionGemmaTests {
    @Test("DiffusionGemma decodes released config shape")
    func decodesReleasedConfigShape() throws {
        let config = try Self.configuration()
        let model = DiffusionGemmaModel(config)

        #expect(model.vocabularySize == 32)
        #expect(model.diffusionCanvasLength == 4)
        #expect(model.newCache(parameters: nil).count == 2)
    }

    @Test("DiffusionGemma produces canvas logits from encoder cache")
    func producesCanvasLogits() throws {
        let model = DiffusionGemmaModel(try Self.configuration())
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
        let model = DiffusionGemmaModel(try Self.configuration())
        let cache = model.newCache(parameters: nil)
        let prompt = LMInput(tokens: MLXArray([2, 7, 11]).reshaped([1, 3]))

        do {
            _ = try model.prepare(prompt, cache: cache, windowSize: nil)
            #expect(Bool(false))
        } catch GenerateError.unsupportedAutoregressiveGeneration(let modelName) {
            #expect(modelName.contains("DiffusionGemmaModel"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("DiffusionGemma rejects speculative decoding")
    func rejectsSpeculativeDecoding() throws {
        let model = DiffusionGemmaModel(try Self.configuration())
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
            #expect(modelName.contains("DiffusionGemmaModel"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("DiffusionGemma iterator batch-flushes emitted tokens into cache")
    func iteratorBatchFlushesEmittedTokens() throws {
        let model = DiffusionGemmaModel(try Self.configuration())
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

    @Test("DiffusionGemma mirrors official tied checkpoint weights")
    func mirrorsOfficialTiedCheckpointWeights() throws {
        let model = DiffusionGemmaModel(try Self.configuration())
        let tensor = MLXArray.ones([1])
        let weights = [
            "model.decoder.embed_tokens.weight": tensor,
            "model.decoder.norm.weight": tensor,
            "model.decoder.layers.0.self_attn.q_proj.weight": tensor,
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight": tensor,
            "lm_head.weight": tensor,
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.encoder.language_model.embed_tokens.weight"] != nil)
        #expect(sanitized["model.encoder.language_model.norm.weight"] != nil)
        #expect(sanitized["model.encoder.language_model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
    }

    @Test("DiffusionGemma strict-loads official tied text checkpoint layout")
    func strictLoadsOfficialTiedTextCheckpointLayout() throws {
        let model = DiffusionGemmaModel(try Self.configuration())
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

    private static func configuration() throws -> DiffusionGemmaConfiguration {
        let json = """
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
}

private final class StableCanvasDiffusionModel: Module, BlockDiffusionLanguageModel {
    let diffusionCanvasLength = 3
    let diffusionMaxDenoisingSteps = 4
    let diffusionEntropyBound: Float = 0.1
    let diffusionVocabularySize = 5
    var decoderCalls = 0

    func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {}

    func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {}

    func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        decoderCalls += 1
        var values = Array(repeating: Float(-20), count: diffusionCanvasLength * diffusionVocabularySize)
        for (position, token) in [1, 2, 3].enumerated() {
            values[position * diffusionVocabularySize + token] = 20
        }
        return MLXArray(values).reshaped([1, diffusionCanvasLength, diffusionVocabularySize])
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
