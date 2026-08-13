// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
@_spi(GemmaEncoder) import MLXLMCommon
import MLXNN
import Testing

/// Verifies that the `@_spi(GemmaEncoder)` Gemma 4 surface
/// (`Gemma4TextModel.model`, `Gemma4TextModelInner.embedTokens` / `.layers` / `.norm` /
/// `.embedScale`, `Gemma4DecoderLayer.callAsFunction`, and the config's
/// `hiddenSize`/`numHiddenLayers`) is sufficient for a CLIENT module to implement an
/// encoder-style all-hidden-states tap — e.g. using Gemma 4 as a frozen text encoder
/// whose per-layer states condition a downstream model.
///
/// Mirrors `Gemma3EncoderAccessTests` (#387). Deliberately imports `MLXLLM` /
/// `MLXLMCommon` without `@testable`, opting in via `@_spi(GemmaEncoder)` exactly as a
/// client module would: everything below must compile against declared API only, with no
/// internal access. Runs on a tiny randomly-initialized model; no weights are downloaded.
struct Gemma4EncoderAccessTests {

    /// A small text-only config. PLE, MoE and KV sharing are OFF — the configuration an
    /// encoder tap actually sees, and the one that makes every layer self-contained.
    private static let configJSON = """
        {
          "model_type": "gemma4_text",
          "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
          "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 32,
          "num_key_value_heads": 2, "num_global_key_value_heads": 1,
          "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
          "enable_moe_block": false, "use_double_wide_mlp": false,
          "vocab_size": 128, "rms_norm_eps": 1e-6,
          "sliding_window": 32, "sliding_window_pattern": 2,
          "rope_theta": 10000, "rope_local_base_freq": 10000,
          "attention_k_eq_v": true, "max_position_embeddings": 512
        }
        """

    private static func makeModel() throws -> Gemma4TextModel {
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(configJSON.utf8))
        let model = Gemma4TextModel(config)
        eval(model)
        return model
    }

    /// The client-side tap this access surface exists for: embedding output plus one state
    /// per decoder layer, with the final state normed — using ONLY the exposed surface.
    private static func allHiddenStates(
        _ model: Gemma4TextModel, tokens: MLXArray
    ) -> [MLXArray] {
        let inner = model.model
        var h = inner.embedTokens(tokens) * inner.embedScale
        var states = [h]
        for layer in inner.layers {
            // Not using per-layer inputs, KV sharing, or a position offset: pass nil and
            // ignore the corresponding returned values.
            let (out, _, _) = layer(h, mask: nil, cache: nil,
                                    perLayerInput: nil, sharedKV: nil, positionOffset: nil)
            h = out
            states.append(h)
        }
        // Gemma 4's final norm is a plain RMSNorm — no `1 + weight` shift, unlike Gemma 3.
        states[states.count - 1] = inner.norm(states[states.count - 1])
        eval(states)
        return states
    }

    @Test func exposedSurfaceSupportsAllHiddenStatesTap() throws {
        let model = try Self.makeModel()
        let config = model.model.config
        let (B, T) = (1, 6)
        let tokens = MLXArray((0 ..< B * T).map { Int32($0 % 128) }).reshaped(B, T)

        let states = Self.allHiddenStates(model, tokens: tokens)

        // One state per layer, plus the embedding state.
        #expect(states.count == config.numHiddenLayers + 1)
        for state in states {
            #expect(state.shape == [B, T, config.hiddenSize])
            #expect(state.dtype == states[0].dtype)
        }
    }

    @Test func firstStateIsScaledEmbedding() throws {
        let model = try Self.makeModel()
        let inner = model.model
        let tokens = MLXArray((0 ..< 6).map { Int32($0) }).reshaped(1, 6)

        let states = Self.allHiddenStates(model, tokens: tokens)
        let expected = inner.embedTokens(tokens) * inner.embedScale
        eval(expected)

        #expect(allClose(states[0], expected, atol: 0).all().item(Bool.self))
    }

    @Test func tapIsDeterministic() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 6).map { Int32($0) }).reshaped(1, 6)

        let a = Self.allHiddenStates(model, tokens: tokens)
        let b = Self.allHiddenStates(model, tokens: tokens)

        #expect(a.count == b.count)
        for (x, y) in zip(a, b) {
            #expect(allClose(x, y, atol: 0).all().item(Bool.self))
        }
    }

    /// The layer stack must actually transform: if states were all equal, the checks above
    /// would pass on a tap that never ran a layer.
    @Test func layersChangeTheState() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 6).map { Int32($0) }).reshaped(1, 6)

        let states = Self.allHiddenStates(model, tokens: tokens)
        let delta = (states[states.count - 1] - states[0]).abs().max()
        eval(delta)

        #expect(delta.item(Float.self) > 0)
    }
}
