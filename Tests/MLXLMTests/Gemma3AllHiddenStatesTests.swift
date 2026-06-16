// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Unit tests for `Gemma3TextModel.allHiddenStates` — the encoder-style
/// multi-layer hidden-state extraction (frozen-text-encoder use). Runs on a
/// tiny randomly-initialized model; no weights are downloaded.
struct Gemma3AllHiddenStatesTests {

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

    @Test("Returns numHiddenLayers + 1 states with (B, T, hiddenSize) shapes")
    func testStateCountAndShapes() {
        let model = Self.makeModel()
        let tokens = MLXArray([3, 14, 15, 92, 65, 35], [1, 6])
        let states = model.allHiddenStates(tokens, mask: .causal)

        #expect(states.count == 8 + 1)
        for state in states {
            #expect(state.shape == [1, 6, 64])
        }
    }

    @Test("First state is the scaled embedding; layers change the state")
    func testFirstStateIsScaledEmbeddingAndLayersTransform() {
        let model = Self.makeModel()
        let tokens = MLXArray([3, 14, 15, 92], [1, 4])
        let states = model.allHiddenStates(tokens, mask: .causal)

        let embedded = model.model.embedTokens(tokens)
        let scale = MLXArray(sqrt(Float(64)), dtype: .bfloat16)
        let expected = embedded * scale.asType(embedded.dtype)
        #expect(allClose(states[0], expected).item(Bool.self))

        // A randomly-initialized layer stack must actually transform the input.
        let firstVsLast = allClose(states[0], states[8], atol: 1e-3).item(Bool.self)
        #expect(!firstVsLast)
    }

    @Test("Deterministic: repeated calls produce identical states")
    func testDeterminism() {
        let model = Self.makeModel()
        let tokens = MLXArray([7, 42, 9, 88, 21], [1, 5])
        let a = model.allHiddenStates(tokens, mask: .causal)
        let b = model.allHiddenStates(tokens, mask: .causal)

        #expect(a.count == b.count)
        for (x, y) in zip(a, b) {
            #expect(allClose(x, y).item(Bool.self))
        }
    }
}
