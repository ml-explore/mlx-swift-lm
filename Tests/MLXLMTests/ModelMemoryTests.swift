// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class SharedTensorModel: Module, BaseLanguageModel {
    let weight: MLXArray
    let view: MLXArray

    init(weight: MLXArray, view: MLXArray) {
        self.weight = weight
        self.view = view
    }
}

private struct FixedMemoryReporter: ModelMemoryReporting {
    func memory() async throws -> ModelMemory { ModelMemory(model: 1024) }
}

final class ModelMemoryTests: XCTestCase {
    func testSharedBuffersAreCountedOnce() throws {
        let weight = MLXArray(Array(repeating: Float(1), count: 128))
        let view = weight[0 ..< 16]
        eval(weight, view)
        let model = SharedTensorModel(weight: weight, view: view)

        XCTAssertEqual(try model.memory().total, Memory.bufferSize([weight]))
    }

    func testUnevaluatedTensorsReturnAnError() {
        let weight = MLXArray(Array(repeating: Float(1), count: 128))
        eval(weight)
        let model = SharedTensorModel(weight: weight, view: weight + 1)

        XCTAssertThrowsError(try model.memory())
    }

    func testRerankerReportsOptionalFootprint() async throws {
        let measured = RerankerContainer(
            modelID: "test", scoreKind: .cosineSimilarity, memoryReporter: FixedMemoryReporter()
        ) { _, _, _, _ in [] }
        let custom = RerankerContainer(modelID: "test", scoreKind: .cosineSimilarity) {
            _, _, _, _ in []
        }

        let measuredFootprint = try await measured.memory()
        let customFootprint = try await custom.memory()
        XCTAssertEqual(measuredFootprint?.total, 1024)
        XCTAssertNil(customFootprint)
    }
}
