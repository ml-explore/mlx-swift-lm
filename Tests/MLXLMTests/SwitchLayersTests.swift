import MLX
import MLXLMCommon
import XCTest

final class SwitchLayersTests: XCTestCase {
    func testWeightedExpertSumMatchesGenericExpression() {
        let outputs = MLXArray(0 ..< 24).asType(.float32).reshaped(2, 3, 4)
        let weights = MLXArray([Float](arrayLiteral: 0.25, 0.75, 1.0, 0.0, 0.5, 0.5))
            .reshaped(2, 3)

        let expected = (outputs * weights[.ellipsis, .newAxis]).sum(axis: -2)
        let actual = weightedExpertSum(outputs, weights)

        eval(expected, actual)
        XCTAssertTrue(allClose(actual, expected).item(Bool.self))
    }
}
