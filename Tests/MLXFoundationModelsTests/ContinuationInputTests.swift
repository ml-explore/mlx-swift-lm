// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Unit tests for `MLXLanguageModel.Executor.continuationInput(from:appending:)`,
/// the Phase-2 (think-then-call) input builder.
///
/// The prompt's token rank must be preserved: LLM processors produce 1-D
/// `[N]` token arrays (the default `LLMModel.prepare` adds the batch axis
/// itself), while VLM processors produce 2-D `[1, N]` (VLM `prepare`
/// implementations index `dim(1)` and fatally abort on 1-D input — see
/// ml-explore/mlx-swift-lm#433). Media must survive so a VLM's Phase-2
/// prefill still sees its pixels.
@Suite
struct ContinuationInputTests {

    @Test func appendsToRank1PromptKeepingRank() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let prompt = LMInput(tokens: MLXArray([1, 2, 3].map { Int32($0) }))
        let result = MLXLanguageModel.Executor.continuationInput(
            from: prompt, appending: [7, 8])

        #expect(result.text.tokens.ndim == 1)
        #expect(result.text.tokens.asArray(Int.self) == [1, 2, 3, 7, 8])
    }

    @Test func appendsToRank2PromptKeepingRank() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tokens = MLXArray([1, 2, 3].map { Int32($0) }).expandedDimensions(axis: 0)
        let prompt = LMInput(tokens: tokens)
        let result = MLXLanguageModel.Executor.continuationInput(
            from: prompt, appending: [7, 8])

        #expect(result.text.tokens.shape == [1, 5])
        #expect(result.text.tokens.asArray(Int.self) == [1, 2, 3, 7, 8])
    }

    @Test func preservesPromptTokenDType() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let prompt = LMInput(tokens: MLXArray([1, 2, 3]))  // int64
        let result = MLXLanguageModel.Executor.continuationInput(
            from: prompt, appending: [7])

        #expect(result.text.tokens.dtype == prompt.text.tokens.dtype)
    }

    @Test func carriesProcessedMediaThrough() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tokens = MLXArray([1, 2, 3].map { Int32($0) }).expandedDimensions(axis: 0)
        let image = LMInput.ProcessedImage(
            pixels: MLXArray.zeros([4, 4]), frames: [THW(1, 2, 2)])
        let prompt = LMInput(text: .init(tokens: tokens), image: image)

        let result = MLXLanguageModel.Executor.continuationInput(
            from: prompt, appending: [7])

        #expect(result.image != nil)
        let frame = result.image?.frames?.first
        #expect(frame?.t == 1)
        #expect(frame?.h == 2)
        #expect(frame?.w == 2)
        #expect(result.video == nil)
    }
}

#endif
