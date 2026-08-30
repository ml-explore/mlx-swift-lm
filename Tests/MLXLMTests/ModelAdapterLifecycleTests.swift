// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class AdapterLifecycleModel: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "projection") var projection: Linear

    let kvHeads: [Int] = []
    private(set) var preparationCount = 0

    override init() {
        _projection.wrappedValue = Linear(weight: MLXArray.eye(2))
        super.init()
    }

    func prepare() throws {
        preparationCount += 1
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projection(inputs)
    }
}

private struct ReplacingModelAdapter: ModelAdapter {
    enum ExpectedFailure: Error { case load }

    let failWhileLoading: Bool

    init(failWhileLoading: Bool = false) {
        self.failWhileLoading = failWhileLoading
    }

    func load(into model: LanguageModel) throws {
        guard let model = model as? AdapterLifecycleModel else {
            throw ModelAdapterError.incompatibleModelType
        }
        try replaceProjection(in: model)
        if failWhileLoading {
            throw ExpectedFailure.load
        }
    }

    func fuse(with model: LanguageModel) throws {
        try load(into: model)
    }

    func unload(from model: LanguageModel) {
        guard let model = model as? AdapterLifecycleModel else { return }
        do {
            try replaceProjection(in: model)
        } catch {
            XCTFail("adapter test fixture could not restore its projection: \(error)")
        }
    }

    private func replaceProjection(in model: AdapterLifecycleModel) throws {
        try model.update(
            modules: ModuleChildren(values: [
                "projection": .value(Linear(weight: MLXArray.eye(2)))
            ]), verify: [])
    }
}

final class ModelAdapterLifecycleTests: XCTestCase {
    func testConvenienceOperationsPrepareExactlyOncePerTopologyBoundary() throws {
        let model = AdapterLifecycleModel()
        let adapter = ReplacingModelAdapter()

        try model.load(adapter: adapter)
        XCTAssertEqual(model.preparationCount, 1)

        try model.fuse(with: adapter)
        XCTAssertEqual(model.preparationCount, 2)

        model.unload(adapter: adapter)
        XCTAssertEqual(model.preparationCount, 3)

        try model.perform(with: adapter) {
            XCTAssertEqual(model.preparationCount, 4)
        }
        XCTAssertEqual(model.preparationCount, 5)
    }

    func testThrowingAdapterStillPreparesPartiallyMutatedModel() {
        let model = AdapterLifecycleModel()
        let adapter = ReplacingModelAdapter(failWhileLoading: true)

        XCTAssertThrowsError(try model.load(adapter: adapter))
        XCTAssertEqual(model.preparationCount, 1)
    }
}
