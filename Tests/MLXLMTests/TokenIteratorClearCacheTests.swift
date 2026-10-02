// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class TokenIteratorClearCacheTests: XCTestCase {

    private func makeTinyModel() -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        return model
    }

    // Short generations never reach token 256, so the first token must clear the cache too,
    // as mlx-lm does. Otherwise each request leaves its buffers in the cache.
    func testFirstTokenClearsBufferCache() throws {
        let model = makeTinyModel()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3, 4, 5])), model: model,
            parameters: GenerateParameters(maxTokens: 8))

        let seeded = 256 * 1024 * 1024
        do {
            let buffer = MLXArray.zeros([seeded], dtype: .uint8) + 1
            eval(buffer)
        }
        // A freed buffer reaches the cache when the GPU has finished with it, which happens
        // on another thread. Checking straight away races that, and loses when earlier tests
        // have left the device busier.
        let deadline = Date().addingTimeInterval(1)
        while Memory.cacheMemory < seeded, Date() < deadline {
            usleep(1000)
        }
        XCTAssertGreaterThanOrEqual(Memory.cacheMemory, seeded)

        _ = iterator.next()

        XCTAssertLessThan(Memory.cacheMemory, seeded / 4)
    }
}
