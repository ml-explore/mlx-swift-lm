import MLX
import XCTest

@testable import MLXVLM

final class Qwen35MRoPETests: XCTestCase {
    func testDefaultSectionsMatchSliceReference() {
        assertMatchesReference(dim: 64, sections: [11, 11, 10])
    }

    func testTruncatedSectionsMatchSliceReference() {
        assertMatchesReference(dim: 16, sections: [2, 1, 1])
    }

    func testMissingSectionsUseDefaults() {
        assertMatchesReference(dim: 64, sections: [])
    }

    private func assertMatchesReference(dim: Int, sections: [Int]) {
        let embedding = Qwen35Language.RotaryEmbedding(
            dim: dim, base: 100_000, mropeSection: sections)
        let positionIds = MLXArray((0 ..< 24).map(Int32.init)).reshaped(3, 2, 4)
        let x = MLXArray.zeros([1], dtype: .float32)

        let actual = embedding(x: x, positionIds: positionIds)
        let expected = reference(
            dim: dim, base: 100_000, sections: sections, positionIds: positionIds)

        eval(actual.0, actual.1, expected.0, expected.1)
        XCTAssertTrue(allClose(actual.0, expected.0, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(actual.1, expected.1, atol: 1e-6).item(Bool.self))
    }

    private func reference(
        dim: Int, base: Float, sections: [Int], positionIds: MLXArray
    ) -> (MLXArray, MLXArray) {
        let sections = sections.count >= 3 ? sections : [11, 11, 10]
        let safeDim = max(1, dim)
        var frequency = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
        frequency = frequency / Float(safeDim)
        var inverseFrequency = 1.0 / pow(MLXArray(base), frequency)
        inverseFrequency = inverseFrequency[.newAxis, .newAxis, .newAxis, 0...]

        let positions = positionIds.asType(.float32)
        let frequencies = positions[0..., 0..., 0..., .newAxis] * inverseFrequency
        let temporal = frequencies[0, 0..., 0..., 0...]
        var slices: [MLXArray] = []
        slices.reserveCapacity(temporal.dim(-1))

        for index in 0 ..< temporal.dim(-1) {
            var slice = temporal[0..., 0..., index]
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, temporal.dim(-1))
                if index >= offset && index < end && (index - offset) % 3 == 0 {
                    slice = frequencies[dimension, 0..., 0..., index]
                    break
                }
            }
            slices.append(slice)
        }

        let interleaved = stacked(slices, axis: -1)
        let embedding = concatenated([interleaved, interleaved], axis: -1)
        return (cos(embedding), sin(embedding))
    }
}
