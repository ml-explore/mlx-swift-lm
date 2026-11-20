// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Regression: KV-shared layers must not declare local K/V projections
//
// Gemma 4's tail layers (`layer_idx >= num_hidden_layers - num_kv_shared_layers`)
// reuse an earlier same-type layer's K/V and own no `k_proj` / `v_proj` / `k_norm`.
// `Gemma4TextLanguageModel.sanitize` drops those tensors from the checkpoint, and
// the runtime forward feeds the shared layers a `sharedKV` instead of computing
// their own. If the module tree still *constructs* those projections (the default
// `kvSharedOnly: false`), loading any real Gemma 4 checkpoint fails under the
// strict loader verify with e.g.
//
//   keyNotFound(path: ["language_model","model","layers","24","self_attn",
//                      "v_proj","weight"], modules: [... "Gemma4TextAttention","Linear"])
//
// (`mlx-community/gemma-4-e4b-it-4bit`: num_hidden_layers=42, num_kv_shared_layers=18
//  → the first shared layer is 24.) These tests pin the construction side to the
// sanitize side so the two can't drift apart again.

struct Gemma4KVSharedLoadTests {

    @Test(
        "Gemma4 VLM text rows preserve rotary positions through KV-owning and shared layers",
        arguments: [0, 2], [2, 3])
    func raggedTextPositions(numKVSharedLayers: Int, batchSize: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma4VLMTextForBatchPositions(numKVSharedLayers: numKVSharedLayers)
            let prompts = Array(
                [[1, 2, 3], Array(4 ..< 11), Array(11 ..< 20)].prefix(batchSize))
            var singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                let output = model(MLXArray(prompt.map { Int32($0) })[.newAxis], cache: cache)
                eval(output.logits)
                eval(cache)
                #expect(output.state == nil)
                return cache
            }
            #expect(singles[0].count == 4 - numKVSharedLayers)
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer] })
            }

            var generated = 0
            for length in [1, 2, 1] {
                let tokens = MLXArray(
                    singles.indices.flatMap { row in
                        (0 ..< length).map { Int32(21 + row * 7 + generated + $0) }
                    }
                ).reshaped([singles.count, length])
                try Self.expectTextChunk(model, tokens: tokens, batch: batch, singles: singles)
                generated += length
            }

            let keep = batchSize == 2 ? [1] : [2, 0]
            let indices = MLXArray(keep.map { Int32($0) })
            for layer in batch { layer.filter(batchIndices: indices) }
            singles = keep.map { singles[$0] }
            let reorderedTokens = MLXArray(
                singles.indices.flatMap { row in [Int32(41 + row), Int32(44 + row)] }
            ).reshaped([singles.count, 2])
            try Self.expectTextChunk(
                model, tokens: reorderedTokens, batch: batch, singles: singles)

            if singles.count > 1 {
                for layer in batch { layer.filter(batchIndices: MLXArray([Int32(1)])) }
                singles = [singles[1]]
            }
            let single = try #require(batch.first as? BatchKVCache)
            guard case .scalar(let position) = single.ropeOffset else {
                Issue.record("one remaining text row must keep the scalar rotary path")
                return
            }
            #expect(position == singles[0][0].offset)
            try Self.expectTextChunk(
                model, tokens: MLXArray([Int32(51), 52])[.newAxis], batch: batch,
                singles: singles)
        }
    }

    /// The module tree of a model with KV sharing must not contain
    /// `k_proj` / `v_proj` / `k_norm` for the shared tail layers, while the
    /// non-shared head layers must still own them.
    @Test("KV-shared layers own no local k_proj/v_proj/k_norm")
    func kvSharedLayersOwnNoLocalKVProjections() {
        // 4 layers, 2 KV-shared → layers 0,1 own K/V; layers 2,3 share it.
        let model = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 2))
        let keys = Set(model.parameters().flattened().map(\.0))

        // Head (KV-owning) layers keep their projections.
        for owning in [0, 1] {
            #expect(keys.contains { $0.contains("layers.\(owning).self_attn.k_proj") })
            #expect(keys.contains { $0.contains("layers.\(owning).self_attn.v_proj") })
        }
        // Tail (KV-shared) layers must own none of them.
        for shared in [2, 3] {
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.k_proj") })
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.v_proj") })
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.k_norm") })
        }
    }

    /// End-to-end reproduction of the production load failure: build a
    /// checkpoint that carries K/V projections on every layer (a dense export),
    /// sanitize it the way real Gemma 4 checkpoints are sanitized (drop the
    /// KV-shared tail's `k_proj`/`v_proj`/`k_norm`), then load it into the
    /// KV-sharing model through the exact call the production loader uses.
    /// Before the fix this threw `keyNotFound(… self_attn.v_proj.weight …)`.
    @Test("Sanitized Gemma 4 checkpoint loads into a KV-sharing model")
    func sanitizedCheckpointLoadsWithoutKeyNotFound() throws {
        // A "dense" sibling (no KV sharing) supplies weights for every layer,
        // independent of how the KV-sharing model constructs its own tree.
        let dense = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 0))
        eval(dense)

        let firstKVSharedLayer = 4 - 2
        var checkpoint = [String: MLXArray]()
        for (key, value) in dense.parameters().flattened() {
            if let layerIdx = Self.layerIndex(in: key), layerIdx >= firstKVSharedLayer,
                key.contains(".self_attn.k_proj")
                    || key.contains(".self_attn.v_proj")
                    || key.contains(".self_attn.k_norm")
            {
                continue  // dropped by sanitize for KV-shared layers
            }
            checkpoint[key] = value
        }

        let model = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 2))
        // `[.all]` is what `MLXLMCommon.loadWeights` applies — it throws on any
        // module parameter that has no matching weight (the original bug).
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
    }

    // MARK: - Helpers

    private static func layerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        return Int(key[range.upperBound...].prefix { $0.isNumber })
    }

    private static func expectTextChunk(
        _ model: Gemma4TextLanguageModel, tokens: MLXArray, batch: [KVCache],
        singles: [[KVCache]], sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let output = model(tokens, cache: batch)
        eval(output.logits)
        eval(batch)
        #expect(output.state == nil, sourceLocation: sourceLocation)
        try #require(
            output.logits.shape == [singles.count, tokens.dim(1), 64],
            sourceLocation: sourceLocation)
        for row in singles.indices {
            let expected = model(tokens[row ..< row + 1], cache: singles[row])
            eval(expected.logits)
            eval(singles[row])
            #expect(
                maxAbsDifference(output.logits[row ..< row + 1], expected.logits) < 1e-3,
                "row \(row) must retain its owner's rotary positions through the KV-shared tail",
                sourceLocation: sourceLocation)
        }
    }

    /// Tiny all-`full_attention` text config. `use_double_wide_mlp` is off so the
    /// dense and KV-sharing models differ *only* in the shared layers' K/V tensors,
    /// keeping every other weight shape-compatible across the two.
    static func textConfig(
        numKVSharedLayers: Int, headDim: Int = 4, vocabularySize: Int = 12
    ) -> Gemma4TextConfiguration {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": \(headDim * 2),
              "num_hidden_layers": 4,
              "intermediate_size": \(headDim * 4),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": \(headDim),
              "global_head_dim": \(headDim),
              "vocab_size": \(vocabularySize),
              "vocab_size_per_layer_input": \(vocabularySize),
              "num_kv_shared_layers": \(numKVSharedLayers),
              "hidden_size_per_layer_input": 0,
              "sliding_window": 8,
              "sliding_window_pattern": 1,
              "max_position_embeddings": 32,
              "rms_norm_eps": 1e-6,
              "rope_traditional": false,
              "use_double_wide_mlp": false,
              "enable_moe_block": false,
              "attention_k_eq_v": false,
              "layer_types": [
                "full_attention", "full_attention", "full_attention", "full_attention"
              ],
              "rope_parameters": {},
              "tie_word_embeddings": true
            }
            """
        return try! JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }
}

/// Text-only Gemma4 VLM backbone with 64-wide heads for float and packed-cache parity tests.
func tinyGemma4VLMTextForBatchPositions(numKVSharedLayers: Int) -> Gemma4TextLanguageModel {
    withRandomState(MLXRandom.RandomState(seed: UInt64(923 + numKVSharedLayers))) {
        let model = Gemma4TextLanguageModel(
            Gemma4KVSharedLoadTests.textConfig(
                numKVSharedLayers: numKVSharedLayers, headDim: 64, vocabularySize: 64))
        model.train(false)
        eval(model)
        return model
    }
}
