// Copyright © 2026 Apple Inc.
//
// Six VLM processors used to pair every prompt with an all-ones attention mask
// their models never read. `ChatSession` refuses to splice a cache under a mask,
// so those models prefilled the whole conversation on every turn. The processors
// now attach none, and these tests pin both halves of why that is safe: a
// text-only prompt carries no mask, and each model prefilled on top of a warm
// cache lands where one cold prefill of the whole prompt does.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("VLM prompt cache reuse")
struct VLMPromptCacheReuseTests {

    // MARK: - Processors

    @Test("text-only prompts carry no attention mask")
    func textOnlyPromptsCarryNoMask() async throws {
        let input = UserInput(chat: [.user("hello")])
        let tokenizer = PromptTokenizer()
        let processors: [(String, any UserInputProcessor)] = [
            (
                "Gemma4",
                Gemma4Processor(
                    try Gemma4VideoInputTests.makeProcessorConfig(), tokenizer: tokenizer)
            ),
            (
                "Gemma4Unified",
                Gemma4UnifiedProcessor(
                    try decode(
                        """
                        { "processor_class": "Gemma4UnifiedProcessor", "image_token_id": 31,
                          "boi_token_id": 28, "eoi_token_id": 29 }
                        """), tokenizer: tokenizer)
            ),
            (
                "Mistral3",
                Mistral3VLMProcessor(
                    try decode(Self.mistralProcessorJSON), tokenizer: tokenizer)
            ),
            (
                "Pixtral",
                PixtralProcessor(
                    try decode(Self.mistralProcessorJSON), tokenizer: tokenizer)
            ),
            (
                "Idefics3",
                Idefics3Processor(
                    try decode(
                        """
                        { "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                          "size": { "longest_edge": 384 } }
                        """), tokenizer: tokenizer)
            ),
            (
                "SmolVLM",
                SmolVLMProcessor(
                    try decode(
                        """
                        { "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                          "image_seq_len": 64, "size": { "longest_edge": 2048 },
                          "max_image_size": { "longest_edge": 512 },
                          "video_sampling": { "fps": 1, "max_frames": 12 } }
                        """), tokenizer: tokenizer)
            ),
            (
                "MuseGlimmer",
                MuseGlimmerProcessor(
                    MuseGlimmerProcessorConfiguration(imageProcessor: .init(maxImageTokens: 4096)),
                    tokenizer: tokenizer)
            ),
        ]

        for (name, processor) in processors {
            let prepared = try await processor.prepare(input: input)
            #expect(prepared.text.mask == nil, "\(name) attached a mask to a text-only prompt")
        }
    }

    // MARK: - Warm prefill

    @Test("Gemma4 prefilled on a warm cache matches a cold prefill")
    func gemma4() throws {
        try expectWarmPrefillMatchesColdPrefill(
            Gemma4ChunkedPrefillTests.makeTinyModel(), tokens: 3 ..< 199)
    }

    @Test("Gemma4 Unified prefilled on a warm cache matches a cold prefill")
    func gemma4Unified() throws {
        let config: Gemma4UnifiedConfiguration = try decode(
            """
            {
              "model_type": "gemma4_unified",
              "vocab_size": 64, "image_token_id": 63, "audio_token_id": 62,
              "video_token_id": 61,
              "text_config": {
                "model_type": "gemma4_unified_text", "hidden_size": 64,
                "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 2,
                "num_key_value_heads": 1, "num_global_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 16, "vocab_size": 64, "vocab_size_per_layer_input": 64,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
                "sliding_window": 8, "sliding_window_pattern": 2, "attention_k_eq_v": true,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "full_attention"],
                "tie_word_embeddings": true
              },
              "vision_config": null,
              "audio_config": null
            }
            """)
        try expectWarmPrefillMatchesColdPrefill(
            withRandomState(MLXRandom.RandomState(seed: 42)) { Gemma4Unified(config) },
            tokens: 2 ..< 61)
    }

    @Test("Mistral3 prefilled on a warm cache matches a cold prefill")
    func mistral3() throws {
        try expectWarmPrefillMatchesColdPrefill(
            VLMImageTokenMismatchTests.makeTinyMistral3(), tokens: 11 ..< 100)
    }

    @Test("Pixtral prefilled on a warm cache matches a cold prefill")
    func pixtral() throws {
        let config: PixtralConfiguration = try decode(
            """
            {
              "model_type": "pixtral",
              "image_token_index": 10,
              "text_config": {
                "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 16, "rms_norm_eps": 1e-5, "vocab_size": 100,
                "rope_theta": 10000.0
              },
              "vision_config": {
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 64, "patch_size": 8,
                "image_size": 32
              }
            }
            """)
        try expectWarmPrefillMatchesColdPrefill(
            withRandomState(MLXRandom.RandomState(seed: 42)) { PixtralVLM(config) },
            tokens: 11 ..< 100)
    }

    @Test("Idefics3 and SmolVLM2 prefilled on a warm cache match a cold prefill")
    func idefics3() throws {
        let config: Idefics3Configuration = try decode(
            """
            {
              "model_type": "idefics3",
              "vocab_size": 100,
              "image_token_id": 10,
              "scale_factor": 2,
              "text_config": {
                "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
                "rms_norm_eps": 1e-5, "vocab_size": 100, "rope_theta": 10000.0
              },
              "vision_config": {
                "model_type": "idefics3", "hidden_size": 16, "num_hidden_layers": 1,
                "intermediate_size": 32, "num_attention_heads": 2, "patch_size": 8,
                "image_size": 32
              }
            }
            """)
        try expectWarmPrefillMatchesColdPrefill(
            withRandomState(MLXRandom.RandomState(seed: 42)) { Idefics3(config) },
            tokens: 11 ..< 100)
    }

    @Test("MuseGlimmer prefilled on a warm cache matches a cold prefill")
    func museGlimmer() throws {
        try expectWarmPrefillMatchesColdPrefill(
            withRandomState(MLXRandom.RandomState(seed: 42)) {
                try MuseGlimmerForwardTests.model()
            },
            tokens: 9 ..< 64)
    }

    // MARK: - Fixtures

    /// The processor configuration Mistral3 and Pixtral share.
    private static let mistralProcessorJSON = """
        {
          "image_processor": {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "size": { "longest_edge": 32 }, "patch_size": 8
          },
          "image_token": "[IMG]",
          "patch_size": 8
        }
        """

    private func decode<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    /// Prefills a prefix, then a suffix on top of it, and compares the suffix's
    /// last logits with one cold prefill of the whole prompt.
    ///
    /// The bound is the decode path's own drift from the cold prefill: decoding the
    /// suffix token by token is the session's offset-correct control, so the warm
    /// prefill may differ from the cold one by no more than splitting the forward
    /// already does. The suffix prefilled with no prefix must land outside it, or the
    /// bound could not tell a warm prefill from one that ignored the cache.
    private func expectWarmPrefillMatchesColdPrefill(
        _ model: some LanguageModel, tokens range: Range<Int>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let ids = (0 ..< 48).map { range.lowerBound + ($0 * 13 + 7) % range.count }
        let prefix = Array(ids[..<40])
        let suffix = Array(ids[40...])

        let cold = try lastLogits(model, ids, cache: model.newCache(parameters: nil))

        let decodeCache = try model.newCache(parameters: nil)
        var decoded = try lastLogits(model, prefix, cache: decodeCache)
        for token in suffix {
            decoded =
                model(
                    LMInput.Text(tokens: MLXArray([Int32(token)])[.newAxis]), cache: decodeCache,
                    state: nil
                ).logits[0..., -1, 0...]
        }
        let noiseFloor = maxAbsDiff(decoded, cold)

        let warmCache = try model.newCache(parameters: nil)
        _ = try lastLogits(model, prefix, cache: warmCache)
        let warm = try lastLogits(model, suffix, cache: warmCache)

        let bound = max(noiseFloor * 10, MatmulPrecision.splitTolerance)
        let unanchored = try lastLogits(model, suffix, cache: model.newCache(parameters: nil))

        #expect(
            maxAbsDiff(warm, cold) <= bound,
            "warm prefill diverged from the cold prefill (noise floor \(noiseFloor))",
            sourceLocation: sourceLocation)
        #expect(
            maxAbsDiff(unanchored, cold) > bound,
            "the bound also accepts a prefill without the prefix",
            sourceLocation: sourceLocation)
    }

    private func lastLogits(
        _ model: some LanguageModel, _ tokens: [Int], cache: [any KVCache]
    ) throws -> MLXArray {
        let input = LMInput(tokens: MLXArray(tokens.map(Int32.init))[.newAxis])
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: PrefillParameters())
        else {
            throw PrepareResultError.expectedLogits
        }
        return output.logits[0..., -1, 0...]
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private enum PrepareResultError: Error {
        case expectedLogits
    }
}

/// Renders every prompt as the same three tokens, outside every fixture's special ids.
private struct PromptTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [40, 41, 42] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [40, 41, 42]
    }
}
