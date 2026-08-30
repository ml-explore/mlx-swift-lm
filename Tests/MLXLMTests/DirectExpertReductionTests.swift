import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class DirectExpertReductionTests: XCTestCase {
    func testWeightedExpertUnsortMatchesLegacyReductionExactly() {
        MLXRandom.seed(17)

        for (tokens, hidden) in [(8, 64), (17, 128), (257, 256)] {
            let assignmentCount = tokens * 8
            let expertIndexValues: [UInt32] = (0 ..< assignmentCount).map { index in
                UInt32((index * 13 + index / 7) % 29)
            }
            let expertIndices = MLXArray(expertIndexValues)
            let order = argSort(expertIndices)
            let inverseOrder = argSort(order)

            let outputs = MLXRandom.normal([tokens, 8, hidden]).asType(.bfloat16)
            let weights = softmax(MLXRandom.normal([tokens, 8]), axis: -1).asType(.bfloat16)
            let sortedOutputs = outputs.reshaped(assignmentCount, hidden)[order]

            let expected = weightedExpertSum(outputs, weights)
            let actual = weightedExpertUnsort(
                sortedOutputs: sortedOutputs,
                inverseOrder: inverseOrder,
                weights: weights)
            eval(expected, actual)

            XCTAssertTrue(
                allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self),
                "tokens=\(tokens), hidden=\(hidden)")
        }
    }

    func testQuantizedSwitchGLUDirectReductionMatchesLegacyExactly() {
        MLXRandom.seed(23)

        let layer = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 8)
        quantize(model: layer, groupSize: 32, bits: 4)

        let tokens = 19
        let input = MLXRandom.normal([tokens, 64]).asType(.bfloat16)
        let indexValues: [UInt32] = (0 ..< tokens * 8).map { index in
            UInt32((index * 5 + index / 8) % 8)
        }
        let indices = MLXArray(indexValues).reshaped(tokens, 8)
        let weights = softmax(MLXRandom.normal([tokens, 8]), axis: -1).asType(.bfloat16)

        XCTAssertTrue(layer.supportsDirectWeightedReduction(input, indices, weights: weights))

        let expected = layer.callAndWeightedReduce(
            input, indices, weights: weights, fuseSortedReduction: false)
        let actual = layer.callAndWeightedReduce(
            input, indices, weights: weights, fuseSortedReduction: true)
        eval(expected, actual)

        XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    func testDirectReductionEligibilityRetainsFallbacks() {
        MLXRandom.seed(29)

        let unquantized = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 8)
        let input = MLXRandom.normal([8, 64]).asType(.bfloat16)
        let indices = MLXArray((0 ..< 64).map { UInt32($0 % 8) }).reshaped(8, 8)
        let weights = MLXArray.ones([8, 8]).asType(.bfloat16) / 8
        XCTAssertFalse(
            unquantized.supportsDirectWeightedReduction(input, indices, weights: weights))

        let quantized = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 8)
        quantize(model: quantized, groupSize: 32, bits: 4)

        XCTAssertFalse(
            quantized.supportsDirectWeightedReduction(
                input[0 ..< 1], indices[0 ..< 1], weights: weights[0 ..< 1]),
            "decode-sized calls must stay on the established path")
        XCTAssertFalse(
            quantized.supportsDirectWeightedReduction(
                input.asType(.float32), indices, weights: weights),
            "non-bfloat16 calls must stay on the established path")
    }
}
