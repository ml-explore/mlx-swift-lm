// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("Parameter count")
struct NumParametersTests {

    /// The modules MLX quantizes, under a module that holds a parameter of its own.
    private final class Layers: Module {
        @ParameterInfo var gain = MLXArray.ones([8])
        @ModuleInfo var embedding = Embedding(embeddingCount: 32, dimensions: 64)
        @ModuleInfo var projection = Linear(64, 32, bias: true)
        @ModuleInfo var experts = SwitchLinear(inputDims: 64, outputDims: 32, numExperts: 4)
    }

    @Test(
        "Quantization keeps the parameter count",
        arguments: [
            (QuantizationMode.affine, 4, 64), (.affine, 8, 32), (.mxfp4, 4, 32), (.nvfp4, 4, 16),
        ])
    func quantizedCount(mode: QuantizationMode, bits: Int, groupSize: Int) {
        let layers = Layers()
        let dense = layers.numParameters()

        quantize(model: layers, groupSize: groupSize, bits: bits, mode: mode)

        #expect(layers.experts is QuantizedSwitchLinear)
        #expect(dense == 8 + 32 * 64 + (64 * 32 + 32) + (4 * 64 * 32 + 4 * 32))
        #expect(layers.numParameters() == dense)
    }
}
