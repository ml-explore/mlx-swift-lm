// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct Gemma3nMaskTests {

    @Test func slidingDecoderLayerAcceptsBooleanMask() {
        withRandomState(MLXRandom.RandomState(seed: 42)) {
            let config = Gemma3nTextConfiguration()
            let layer = Gemma3nDecoderLayer(config, layerIdx: 0)
            let sequenceLength = 4
            let hiddenStates = MLXRandom.normal([
                config.altupNumInputs, 1, sequenceLength, config.hiddenSize,
            ])
            let perLayerInput = MLXRandom.normal([
                1, sequenceLength, config.hiddenSizePerLayerInput,
            ])
            let mask = createCausalMask(n: sequenceLength, offset: 0, windowSize: 2)

            let output = layer(
                hiddenStates,
                mask: .array(mask),
                perLayerInput: perLayerInput
            )
            eval(output)

            #expect(output.shape == hiddenStates.shape)

            // The Boolean mask must block the same positions as the equivalent
            // additive float mask, rather than acting as an additive 1/0 mask.
            let additiveMask = MLX.where(
                mask, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude))
            let reference = layer(
                hiddenStates,
                mask: .array(additiveMask),
                perLayerInput: perLayerInput
            )
            eval(reference)

            #expect(allClose(output, reference, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        }
    }

    @Test func booleanAttentionMaskKeepsItsDType() {
        let mask = createCausalMask(n: 4, offset: 0, windowSize: 2)
        let unchangedMask = gemma3nAdjustedAttentionMask(
            .array(mask),
            keySequenceLength: 4
        )
        let adjustedMask = gemma3nAdjustedAttentionMask(
            .array(mask),
            keySequenceLength: 3
        )

        guard case .array(let unchangedMaskArray) = unchangedMask else {
            Issue.record("Expected an unchanged array attention mask")
            return
        }
        guard case .array(let maskArray) = adjustedMask else {
            Issue.record("Expected a sliced array attention mask")
            return
        }

        #expect(unchangedMaskArray.dtype == .bool)
        #expect(unchangedMaskArray.shape == [4, 4])
        #expect(maskArray.dtype == .bool)
        #expect(maskArray.shape == [4, 3])
        #expect(maskArray[0, 0].item(Bool.self))
        #expect(!maskArray[0, 1].item(Bool.self))
        // Blocked by the sliding window but allowed under a plain causal mask.
        #expect(!maskArray[2, 0].item(Bool.self))
        #expect(maskArray[2, 1].item(Bool.self))
    }
}
