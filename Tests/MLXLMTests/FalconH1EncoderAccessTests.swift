// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(FalconH1Encoder) import MLXLLM
import MLXLMCommon
import XCTest

/// Proves the `@_spi(FalconH1Encoder)` surface is sufficient for a client that builds its
/// own slow-stack input — the DualAR-TTS shape, where the embedding entering the stack is a
/// text embedding summed with several codebook embeddings rather than one token lookup.
///
/// Deliberately imports MLXLLM WITHOUT `@testable`, so the suite fails to compile if any
/// piece of the needed surface stops being exposed. Runs on a tiny randomly-initialized
/// model; no weights are downloaded.
final class FalconH1EncoderAccessTests: XCTestCase {

    private func tinyConfiguration(embeddingMultiplier: Float = 0.125) throws
        -> FalconH1Configuration
    {
        let json = """
            {
                "model_type": "falcon_h1",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 100,
                "mamba_d_ssm": 16,
                "mamba_d_state": 8,
                "mamba_d_head": 8,
                "mamba_n_heads": 2,
                "mamba_n_groups": 1,
                "mamba_d_conv": 4,
                "mamba_chunk_size": 64,
                "embedding_multiplier": \(embeddingMultiplier)
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(FalconH1Configuration.self, from: json)
    }

    // MARK: - The seam

    /// A composite embedding runs through the stack and produces a well-formed hidden state.
    func testInputsEmbedsEntryPointRunsTheStack() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)
        let inner = model.model

        let (batch, length) = (1, 5)
        let embeds = MLXRandom.normal([batch, length, config.hiddenSize])
        let out = inner(inputsEmbeds: embeds)
        eval(out)

        XCTAssertEqual(out.shape, [batch, length, config.hiddenSize])
        // The stack must actually transform the input — otherwise this test would pass on a
        // seam that returned its argument (or ran zero layers).
        XCTAssertGreaterThan(
            (out - embeds).abs().max().item(Float.self), 1e-3,
            "the layer stack did not transform the injected embedding")
    }

    /// Deterministic for identical input.
    func testInputsEmbedsIsDeterministic() throws {
        let config = try tinyConfiguration()
        let inner = FalconH1Model(config).model

        let embeds = MLXRandom.normal([1, 4, config.hiddenSize])
        let a = inner(inputsEmbeds: embeds)
        let b = inner(inputsEmbeds: embeds)
        eval(a, b)
        XCTAssertEqual((a - b).abs().max().item(Float.self), 0.0)
    }

    /// The token path and the seam agree when the seam is fed the same embedding the token
    /// path would have built. This is what makes the seam a refactor rather than a fork.
    func testInputsEmbedsMatchesTokenPath() throws {
        let config = try tinyConfiguration()
        let inner = FalconH1Model(config).model

        let ids = MLXArray([3, 17, 42, 8]).reshaped([1, 4])
        let viaTokens = inner(ids)
        let viaEmbeds = inner(inputsEmbeds: inner.embedTokens(ids))
        eval(viaTokens, viaEmbeds)

        XCTAssertLessThan((viaTokens - viaEmbeds).abs().max().item(Float.self), 1e-5)
    }

    // MARK: - The multiplier trap

    /// `(raw + other) * m == folded + other * m`.
    ///
    /// `sanitize()` folds `embedding_multiplier` into `embed_tokens.weight`, so a client
    /// summing extra contributions onto the folded lookup must scale those contributions
    /// too. This test pins that equivalence, and — by construction — fails if someone
    /// "simplifies" a composite-embedding client to add unscaled contributions: that
    /// variant is computed here and asserted to be DIFFERENT, so the trap cannot quietly
    /// become the expected behaviour.
    func testCompositeEmbeddingMultiplierEquivalence() throws {
        let multiplier: Float = 0.125
        let config = try tinyConfiguration(embeddingMultiplier: multiplier)
        let inner = FalconH1Model(config).model

        let ids = MLXArray([5, 11, 2]).reshaped([1, 3])
        let raw = inner.embedTokens(ids)
        let other = MLXRandom.normal([1, 3, config.hiddenSize])

        let folded = raw * multiplier  // what sanitize() would bake into the table
        let correctA = (raw + other) * multiplier  // unfolded table, scale the sum
        let correctB = folded + other * multiplier  // folded table, scale the remainder
        let wrong = folded + other  // folded table, remainder left unscaled
        eval(correctA, correctB, wrong)

        XCTAssertLessThan(
            (correctA - correctB).abs().max().item(Float.self), 1e-6,
            "the two correct orderings must agree")
        XCTAssertGreaterThan(
            (correctA - wrong).abs().max().item(Float.self), 1e-3,
            "leaving the non-lookup contribution unscaled must be observably different")

        // And the difference survives the stack, so it is a real output error rather than
        // something the first layer norm washes out.
        let outCorrect = inner(inputsEmbeds: correctA)
        let outWrong = inner(inputsEmbeds: wrong)
        eval(outCorrect, outWrong)
        XCTAssertGreaterThan(
            (outCorrect - outWrong).abs().max().item(Float.self), 1e-3,
            "the unscaled-remainder error must be visible in the stack output")
    }

    // MARK: - Layer-level access

    /// A client can also drive the layers itself, e.g. to tap every hidden state.
    func testLayerStackIsDrivableDirectly() throws {
        let config = try tinyConfiguration()
        let inner = FalconH1Model(config).model

        var h = MLXRandom.normal([1, 4, config.hiddenSize])
        var states: [MLXArray] = [h]
        let caches: [CacheList?] = Array(repeating: nil, count: inner.layers.count)
        let mambaMask = createSSMMask(h: h, cache: nil)
        let attnMask = createAttentionMask(h: h, cache: nil)
        for (layer, cache) in zip(inner.layers, caches) {
            h = layer(h, cache: cache, attnMask: attnMask, mambaMask: mambaMask)
            states.append(h)
        }
        let out = inner.finalLayerNorm(h)
        eval(out, states.last!)

        XCTAssertEqual(states.count, config.numHiddenLayers + 1)
        XCTAssertEqual(out.shape, [1, 4, config.hiddenSize])
        // Same result as the seam, reached the long way round.
        let viaSeam = inner(inputsEmbeds: states[0])
        eval(viaSeam)
        XCTAssertLessThan((out - viaSeam).abs().max().item(Float.self), 1e-5)
    }
}
