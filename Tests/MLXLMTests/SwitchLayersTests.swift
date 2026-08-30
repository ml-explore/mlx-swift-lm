import MLX
import MLXLMCommon
import XCTest

final class SwitchLayersTests: XCTestCase {
    private func assertBitIdentical(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape", file: file, line: line)
        let a = actual.asType(.float32).asArray(Float.self)
        let b = expected.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    private func referenceRouterTopK(
        selection: MLXArray, values: MLXArray, k: Int,
        normalize: Bool, order: FusedRouterTopKOrder
    ) -> (MLXArray, MLXArray) {
        let indices: MLXArray
        switch order {
        case .ascending:
            let kth = selection.dim(-1) - k
            indices = MLX.argPartition(selection, kth: kth, axis: -1)[.ellipsis, kth...]
        case .descending:
            indices = MLX.argPartition(-selection, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        }
        var scores = MLX.takeAlong(values, indices, axis: -1)
        if normalize {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (indices, scores)
    }

    func testWeightedExpertSumMatchesGenericExpression() {
        let outputs = MLXArray(0 ..< 24).asType(.float32).reshaped(2, 3, 4)
        let weights = MLXArray([Float](arrayLiteral: 0.25, 0.75, 1.0, 0.0, 0.5, 0.5))
            .reshaped(2, 3)

        let expected = (outputs * weights[.ellipsis, .newAxis]).sum(axis: -2)
        let actual = weightedExpertSum(outputs, weights)

        eval(expected, actual)
        XCTAssertTrue(allClose(actual, expected).item(Bool.self))
    }

    func testFusedRouterTopKPreservesOrderAndSeparateScoreValues() {
        let rows = 32
        let e = 128
        let k = 8

        for dtype in [DType.float16, DType.bfloat16, DType.float32] {
            for order in [FusedRouterTopKOrder.ascending, .descending] {
                for normalize in [false, true] {
                    MLXRandom.seed(UInt64(1000 + (normalize ? 1 : 0)))
                    // Rounded selection values force stable-tie behavior; the
                    // positive score tensor exercises selection and weighting
                    // from different inputs as used by Qwen 3 MoE.
                    let selection = (MLX.round(MLXRandom.normal([rows, e]) * 2) / 2)
                        .asType(dtype)
                    let values = MLX.softmax(
                        MLXRandom.normal([rows, e]), axis: -1, precise: true
                    ).asType(dtype)

                    let reference = referenceRouterTopK(
                        selection: selection, values: values, k: k,
                        normalize: normalize, order: order)
                    let fused = fusedRouterTopK(
                        selection: selection, values: values, k: k,
                        normalize: normalize, order: order)
                    eval(reference.0, reference.1, fused.0, fused.1)

                    XCTAssertEqual(
                        fused.0.reshaped(-1).asArray(UInt32.self),
                        reference.0.reshaped(-1).asArray(UInt32.self),
                        "\(dtype) \(order) norm=\(normalize): expert order")
                    assertBitIdentical(
                        fused.1, reference.1,
                        "\(dtype) \(order) norm=\(normalize): selected scores")
                }
            }
        }
    }
}
