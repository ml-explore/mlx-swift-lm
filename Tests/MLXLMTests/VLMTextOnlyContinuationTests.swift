// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

private struct ContinuationTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let value = Int(scalar.value)
            return (256 ..< 356).contains(value) ? value - 256 : value % 100
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.map { UnicodeScalar($0 + 256)! }))
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            tokens.append(message["role"] as? String == "assistant" ? 2 : 1)
            let content =
                (message["content"] as? String)
                ?? (message["content"] as? [[String: String]])?
                .compactMap { $0["text"] }.joined() ?? ""
            tokens += encode(text: content, addSpecialTokens: false)
            tokens.append(3)
        }
        return tokens + [2]
    }
}

@Suite("VLM text-only continuation", .serialized)
struct VLMTextOnlyContinuationTests {
    static let modelTypes = [
        "fastvlm", "llava_qwen2", "mistral3", "ministral3", "idefics3", "smolvlm",
    ]

    private func model(_ modelType: String) async throws -> any LanguageModel {
        var text: [String: Any] = [
            "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "vocab_size": 100, "rms_norm_eps": 1e-5,
            "rope_theta": 10000, "max_position_embeddings": 512,
        ]
        var configuration: [String: Any] = [
            "model_type": modelType, "image_token_index": 10, "image_token_id": 10,
            "vocab_size": 100,
        ]
        switch modelType {
        case "fastvlm", "llava_qwen2":
            text["model_type"] = "llava_qwen2"
            configuration.merge(text) { _, new in new }
            configuration.merge([
                "eos_token_id": 99, "mm_projector_type": "mlp2x_gelu",
                "mm_hidden_size": 16, "tokenizer_model_max_length": 512,
                "tokenizer_padding_side": "right",
                "vision_config": [
                    "cls_ratio": 1, "down_patch_size": 3, "down_stride": 2,
                    "downsamples": [false], "embed_dims": [16], "hidden_size": 16,
                    "image_size": 16, "intermediate_size": 32, "layers": [1],
                    "layer_scale_init_value": 1e-5, "mlp_ratios": [2], "num_classes": 0,
                    "patch_size": 4, "pos_embs_shapes": [NSNull()], "projection_dim": 16,
                    "repmixer_kernel_size": 3, "token_mixers": ["repmixer"],
                ],
            ]) { _, new in new }
        case "mistral3", "ministral3":
            text["model_type"] = modelType == "mistral3" ? "mistral" : "ministral3"
            text["rope_parameters"] = ["rope_theta": 10000]
            if modelType == "ministral3" {
                text["layer_types"] = ["sliding_attention", "full_attention"]
                text["sliding_window"] = 4
            }
            configuration["text_config"] = text
            configuration["spatial_merge_size"] = 1
            configuration["vision_feature_layer"] = -1
            configuration["vision_config"] = [
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 64, "patch_size": 8,
                "image_size": 16,
            ]
        default:
            configuration["text_config"] = text
            configuration["scale_factor"] = 2
            configuration["vision_config"] = [
                "model_type": "siglip_vision_model", "hidden_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
                "patch_size": 4, "image_size": 16,
            ]
        }
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: configuration),
            modelType: modelType == "ministral3" ? "mistral3" : modelType)
        model.train(false)
        eval(model)
        return model
    }

    private func prepare(
        _ model: any LanguageModel, tokens: MLXArray, cache: [KVCache],
        prefill: PrefillParameters, image: LMInput.ProcessedImage? = nil, mask: MLXArray? = nil
    ) throws -> MLXArray {
        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens, mask: mask), image: image),
            cache: cache, state: nil, prefill: prefill)
        guard case .logits(let output) = result else {
            Issue.record("Expected the VLM to return logits")
            throw CancellationError()
        }
        eval(output.logits, cache)
        return output.logits
    }

    @Test("Flat prompts and cached suffixes match batched inputs", arguments: modelTypes)
    func flatAndBatchedInputsMatch(modelType: String) async throws {
        let model = try await model(modelType)
        let prefix = MLXArray([Int32(1), 2, 3, 4]).expandedDimensions(axis: 0)
        for warm in [false, true] {
            let baseCache = try model.newCache(parameters: nil)
            if warm {
                _ = try prepare(model, tokens: prefix, cache: baseCache, prefill: .init())
            }
            for count in [1, 2, 3, 4, 5, 9] {
                for chunking: PrefillParameters.Chunking in [.balanced, .remainder, .unchunked] {
                    let flatCache = baseCache.map { $0.copy() }
                    let batchedCache = baseCache.map { $0.copy() }
                    let flat = MLXArray((5 ..< 5 + count).map(Int32.init))
                    let prefill = PrefillParameters(stepSize: 3, chunking: chunking)
                    let actual = try prepare(
                        model, tokens: flat, cache: flatCache, prefill: prefill)
                    let expected = try prepare(
                        model, tokens: flat.expandedDimensions(axis: 0), cache: batchedCache,
                        prefill: prefill)
                    #expect(actual.shape == expected.shape)
                    #expect(actual.dim(0) == 1)
                    #expect(actual.dim(-1) == 100)
                    #expect(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
                    for (flat, batched) in zip(flatCache, batchedCache) {
                        #expect(flat.offset == (warm ? 4 : 0) + count)
                        #expect(flat.offset == batched.offset)
                        #expect(flat.state.count == batched.state.count)
                        for (a, b) in zip(flat.state, batched.state) {
                            #expect(a.shape == b.shape)
                            #expect(allClose(a, b, rtol: 0, atol: 0).item(Bool.self))
                        }
                    }
                }
            }
        }
    }

    @Test("Cached suffix and decode match a full prompt", arguments: modelTypes)
    func continuationMatchesFullPrompt(modelType: String) async throws {
        let model = try await model(modelType)
        let cache = try model.newCache(parameters: nil)
        let referenceCache = try model.newCache(parameters: nil)
        _ = try prepare(
            model, tokens: MLXArray([Int32(1), 2, 3, 4]).expandedDimensions(axis: 0),
            cache: cache, prefill: .init(stepSize: 2))
        let actual = try prepare(
            model, tokens: MLXArray([Int32(5), 6, 7, 8, 9]), cache: cache,
            prefill: .init(stepSize: 2))
        let expected = try prepare(
            model, tokens: MLXArray((1 ... 9).map(Int32.init)).expandedDimensions(axis: 0),
            cache: referenceCache, prefill: .init(chunking: .unchunked))
        expectClose(
            actual[0..., -1, 0...], expected[0..., -1, 0...], rtol: 1e-4, atol: 1e-4,
            reducedAtol: 2e-2)
        let token = MLXArray([Int32(11)]).expandedDimensions(axis: 0)
        let decoded = model(token, cache: cache)
        let reference = model(token, cache: referenceCache)
        eval(decoded, reference, cache, referenceCache)
        expectClose(decoded, reference, rtol: 1e-4, atol: 1e-4, reducedAtol: 2e-2)
        #expect(cache.allSatisfy { $0.offset == 10 })
    }

    @Test("An existing batch dimension is preserved", arguments: modelTypes)
    func multipleRowsKeepTheirBatchDimension(modelType: String) async throws {
        let model = try await model(modelType)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(2, 3)
        let output = try prepare(
            model, tokens: tokens, cache: model.newCache(parameters: nil),
            prefill: .init(chunking: .unchunked))
        #expect(output.shape == [2, 3, 100])
        for row in 0 ..< 2 {
            let expected = try prepare(
                model, tokens: tokens[row], cache: model.newCache(parameters: nil),
                prefill: .init(chunking: .unchunked))
            #expect(
                allClose(output[row], expected[0], rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test("A media prefix can be followed by a flat text suffix", arguments: modelTypes)
    func mediaPrefixContinuation(modelType: String) async throws {
        let model = try await model(modelType)
        let isFastVLM = modelType == "fastvlm" || modelType == "llava_qwen2"
        let isIdefics3 = modelType == "idefics3" || modelType == "smolvlm"
        let pixels = MLXArray.ones(isIdefics3 ? [1, 16, 16, 3] : [1, 3, 16, 16])
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [.init(1, 16, 16)])
        let ids: [Int32] = [1] + Array(repeating: 10, count: isFastVLM ? 1 : 4) + [2]
        let tokens = MLXArray(ids)
        let mask = MLXArray.ones([1, ids.count], type: Int32.self)
        let cache = try model.newCache(parameters: nil)
        let referenceCache = try model.newCache(parameters: nil)
        let actual = try prepare(
            model, tokens: tokens, cache: cache, prefill: .init(stepSize: 3), image: image,
            mask: mask)
        let expected = try prepare(
            model, tokens: tokens.expandedDimensions(axis: 0), cache: referenceCache,
            prefill: .init(stepSize: 3), image: image, mask: mask)
        #expect(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
        let suffix = MLXArray([Int32(3), 4, 5])
        let continuation = try prepare(
            model, tokens: suffix, cache: cache, prefill: .init(stepSize: 2))
        let reference = try prepare(
            model, tokens: suffix.expandedDimensions(axis: 0), cache: referenceCache,
            prefill: .init(stepSize: 2))
        #expect(allClose(continuation, reference, rtol: 0, atol: 0).item(Bool.self))
        #expect(cache.map(\.offset) == referenceCache.map(\.offset))
    }

    @Test("ChatSession reuses text-only VLM prompts across turns", arguments: modelTypes)
    func sessionReusesCache(modelType: String) async throws {
        let model = try await model(modelType)
        let processorType: String
        switch modelType {
        case "fastvlm", "llava_qwen2": processorType = "FastVLMProcessor"
        case "mistral3", "ministral3": processorType = "Mistral3Processor"
        default: processorType = "SmolVLMProcessor"
        }
        let configuration = try #require(
            VLMTextOnlyPrepareTests.processors.first { $0.type == processorType })
        let tokenizer = ContinuationTokenizer()
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(configuration.configuration.utf8),
            processorType: processorType, tokenizer: tokenizer)
        let session = ChatSession(
            ModelContext(
                configuration: .init(id: "test/\(modelType)"), model: model,
                processor: processor, tokenizer: tokenizer),
            generateParameters: .init(maxTokens: 2, temperature: 0, prefill: .init(stepSize: 3)))
        for turn in 0 ..< 3 {
            var cached: Int?
            for try await event in session.streamDetails(to: "Turn \(turn)") {
                if case .info(let info) = event { cached = info.cachedPromptTokenCount }
            }
            let cachedCount = try #require(cached)
            #expect(turn == 0 ? cachedCount == 0 : cachedCount > 0)
        }
    }
}
