// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXNN
import Testing

/// Verifies that the `@_spi(GemmaEncoder)` Gemma 3 surface
/// (`Gemma3Model.embedTokens`, `Gemma3Model.layers`,
/// `Gemma3TransformerBlock.callAsFunction`, and the config's
/// `hiddenSize`/`hiddenLayers`) is sufficient for a CLIENT module to implement
/// an encoder-style all-hidden-states tap — e.g. using Gemma 3 as a frozen text
/// encoder whose per-layer states condition a downstream model.
///
/// Deliberately imports `MLXLLM` without `@testable`, opting in via
/// `@_spi(GemmaEncoder)` exactly as a client module would: everything below
/// must compile against declared API only, with no internal access. The SPI
/// opt-in keeps this surface off the advertised public API of MLXLLM while
/// still making it usable outside the module. Runs on a tiny
/// randomly-initialized model; no weights are downloaded.
struct Gemma3EncoderAccessTests {

    private static func makeModel() -> Gemma3TextModel {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        eval(model)
        return model
    }

    /// The client-side tap this access surface exists for: embedding output
    /// plus each transformer layer's output — `numHiddenLayers + 1` states,
    /// each shaped `(B, T, hiddenSize)` — with a single caller-supplied mask
    /// applied uniformly to every layer.
    private static func allHiddenStates(
        of model: Gemma3TextModel,
        inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> [MLXArray] {
        let inner = model.model
        var h = inner.embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(inner.config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        var states: [MLXArray] = [h]
        for layer in inner.layers {
            h = layer(h, mask: mask, cache: nil)
            eval(h)
            states.append(h)
        }
        return states
    }

    @Test("Tap returns numHiddenLayers + 1 states with (B, T, hiddenSize) shapes")
    func testStateCountAndShapes() {
        let model = Self.makeModel()
        let tokens = MLXArray([3, 14, 15, 92, 65, 35], [1, 6])
        let states = Self.allHiddenStates(of: model, inputs: tokens, mask: .causal)

        #expect(states.count == model.config.hiddenLayers + 1)
        for state in states {
            #expect(state.shape == [1, 6, model.config.hiddenSize])
        }
    }

    @Test("First state is the scaled embedding; layers change the state")
    func testFirstStateIsScaledEmbeddingAndLayersTransform() {
        let model = Self.makeModel()
        let tokens = MLXArray([3, 14, 15, 92], [1, 4])
        let states = Self.allHiddenStates(of: model, inputs: tokens, mask: .causal)

        let embedded = model.model.embedTokens(tokens)
        let scale = MLXArray(sqrt(Float(model.config.hiddenSize)), dtype: .bfloat16)
        let expected = embedded * scale.asType(embedded.dtype)
        #expect(allClose(states[0], expected).item(Bool.self))

        // A randomly-initialized layer stack must actually transform the input.
        let firstVsLast = allClose(states[0], states[states.count - 1], atol: 1e-3)
            .item(Bool.self)
        #expect(!firstVsLast)
    }

    @Test("Deterministic: repeated calls produce identical states")
    func testDeterminism() {
        let model = Self.makeModel()
        let tokens = MLXArray([7, 42, 9, 88, 21], [1, 5])
        let a = Self.allHiddenStates(of: model, inputs: tokens, mask: .causal)
        let b = Self.allHiddenStates(of: model, inputs: tokens, mask: .causal)

        #expect(a.count == b.count)
        for (x, y) in zip(a, b) {
            #expect(allClose(x, y).item(Bool.self))
        }
    }
}
