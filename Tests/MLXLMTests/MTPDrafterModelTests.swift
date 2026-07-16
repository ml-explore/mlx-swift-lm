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
    private(set) var draftCallCount = 0

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        queryOffset: Int,
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

    // Mock target (not actually used by MockMTPDrafter; the protocol shape
    // requires it as a parameter).
    let target: any LanguageModel = DummyLanguageModel()

    let result = drafter.draftBlock(
        target: target,
        lastToken: MLXArray([Int32(7)]),
        lastHidden: MLXArray.zeros([1, 1, 4]),
        sharedKV: [
            "full_attention": (MLXArray.zeros([1, 1, 8, 4]), MLXArray.zeros([1, 1, 8, 4])),
            "sliding_attention": (MLXArray.zeros([1, 1, 8, 4]), MLXArray.zeros([1, 1, 8, 4])),
        ],
        queryOffset: 0,
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

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?)
        throws -> PrepareResult
    {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

// MARK: - Cross-model state key round-trips

@Test
func testMTPLastHiddenStatesKeyRoundtrip() {
    var state = LMOutput.State()
    let hidden = MLXArray(
        (0 ..< 8).map { Float($0) }, [1, 1, 8])

    state[mtpLastHiddenStatesKey] = hidden

    let recovered = state[mtpLastHiddenStatesKey]
    #expect(recovered != nil)
    #expect(recovered!.shape == [1, 1, 8])
    // Sample-element identity at the boundaries.
    #expect(recovered![0, 0, 0].item(Float.self) == Float(0))
    #expect(recovered![0, 0, 7].item(Float.self) == Float(7))
}

@Test
func testMTPSharedKVStatesKeyRoundtrip() {
    var state = LMOutput.State()
    let kFull = MLXArray.ones([1, 2, 4, 8])
    let vFull = MLXArray.ones([1, 2, 4, 8]) * MLXArray(Float(2))
    let kSlide = MLXArray.zeros([1, 2, 4, 8])
    let vSlide = MLXArray.ones([1, 2, 4, 8]) * MLXArray(Float(3))

    state[mtpSharedKVStatesKey] = [
        "full_attention": (kFull, vFull),
        "sliding_attention": (kSlide, vSlide),
    ]

    let recovered = state[mtpSharedKVStatesKey]
    #expect(recovered != nil)
    #expect(Set(recovered!.keys) == ["full_attention", "sliding_attention"])

    let full = recovered!["full_attention"]!
    #expect(full.0.shape == [1, 2, 4, 8])
    #expect(full.1.shape == [1, 2, 4, 8])
    #expect(full.0[0, 0, 0, 0].item(Float.self) == Float(1))
    #expect(full.1[0, 0, 0, 0].item(Float.self) == Float(2))

    let slide = recovered!["sliding_attention"]!
    #expect(slide.0[0, 0, 0, 0].item(Float.self) == Float(0))
    #expect(slide.1[0, 0, 0, 0].item(Float.self) == Float(3))
}

@Test
func testMTPEmitFlagKeyDefaultsToFalse() {
    let state = LMOutput.State()
    // Absent key reads as nil; iterator-side code treats nil as false.
    #expect(state[mtpEmitFlagKey] == nil)

    var withTrue = LMOutput.State()
    withTrue[mtpEmitFlagKey] = true
    #expect(withTrue[mtpEmitFlagKey] == true)

    var withFalse = LMOutput.State()
    withFalse[mtpEmitFlagKey] = false
    #expect(withFalse[mtpEmitFlagKey] == false)
}

@Test
func testMTPStateKeysAreDistinct() {
    var state = LMOutput.State()
    let hidden = MLXArray.zeros([1, 1, 4])
    let kv: [String: (MLXArray, MLXArray)] = [
        "full_attention": (MLXArray.zeros([1, 1, 2, 4]), MLXArray.zeros([1, 1, 2, 4]))
    ]

    state[mtpLastHiddenStatesKey] = hidden
    state[mtpSharedKVStatesKey] = kv
    state[mtpEmitFlagKey] = true

    #expect(state[mtpLastHiddenStatesKey]?.shape == [1, 1, 4])
    #expect(state[mtpSharedKVStatesKey]?.count == 1)
    #expect(state[mtpEmitFlagKey] == true)
}
