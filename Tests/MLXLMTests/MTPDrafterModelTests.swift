// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Minimal `MTPDrafterModel` conformance for shape-testing.
///
/// Returns deterministic dummy tokens of the requested shape so the protocol
/// contract can be exercised end-to-end without bringing in a real drafter.
private final class MockMTPDrafter: Module, MTPDrafterModel {
    private(set) var bindCallCount = 0
    private(set) var draftCallCount = 0

    func bind(target: any LanguageModel) {
        bindCallCount += 1
    }

    func draftBlock(
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionIds: MLXArray,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftCallCount += 1
        let batch = lastToken.dim(0)
        // Return [B, blockSize - 1] zeros — the contract is shape, not value.
        return MLXArray.zeros([batch, blockSize - 1], dtype: .int32)
    }
}

@Test
func testMTPDrafterModelProtocolShape() {
    let drafter = MockMTPDrafter()

    // Mock target (not actually used; we just exercise the call).
    let target: any LanguageModel = DummyLanguageModel()

    drafter.bind(target: target)
    #expect(drafter.bindCallCount == 1)

    let result = drafter.draftBlock(
        lastToken: MLXArray([Int32(7)]),
        lastHidden: MLXArray.zeros([1, 1, 4]),
        sharedKV: [
            "full_attention": (MLXArray.zeros([1, 1, 8, 4]), MLXArray.zeros([1, 1, 8, 4])),
            "sliding_attention": (MLXArray.zeros([1, 1, 8, 4]), MLXArray.zeros([1, 1, 8, 4])),
        ],
        positionIds: MLXArray(Int32(0)).reshaped([1, 1]),
        blockSize: 4,
        sampler: ArgMaxSampler()
    )
    #expect(drafter.draftCallCount == 1)
    #expect(result.shape == [1, 3])
}

@Test
func testMTPDrafterContextRoundtrip() {
    let drafter = MockMTPDrafter()
    let config = ModelConfiguration(id: "test/mock-drafter", defaultPrompt: "")
    let ctx = MTPDrafterContext(configuration: config, model: drafter)
    #expect(ctx.configuration.name == "test/mock-drafter")
    #expect(ctx.model is MockMTPDrafter)
}

@Test
func testMTPDrafterContainerPerform() async {
    let drafter = MockMTPDrafter()
    let config = ModelConfiguration(id: "test/mock-drafter", defaultPrompt: "")
    let container = MTPDrafterContainer(
        context: MTPDrafterContext(configuration: config, model: drafter))

    let name = await container.configuration.name
    #expect(name == "test/mock-drafter")

    let modelIsMock = await container.perform { ctx in
        ctx.model is MockMTPDrafter
    }
    #expect(modelIsMock)
}

/// Minimal LanguageModel implementation for test plumbing only.
private final class DummyLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}
