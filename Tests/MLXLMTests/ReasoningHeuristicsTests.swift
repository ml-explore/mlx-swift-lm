// Copyright © 2025 Apple Inc.

import Testing

@testable import MLXLMCommon

@Suite
struct ReasoningHeuristicsTests {

    @Test func headlineReasoningIdsAreDetected() {
        #expect(ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Qwen3-4B-4bit"))
        #expect(ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Qwen3-0.6B-4bit"))
        #expect(
            ReasoningHeuristics.isLikelyReasoningModel(
                "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"))
        #expect(ReasoningHeuristics.isLikelyReasoningModel("mlx-community/DeepSeek-R1-4bit"))
    }

    /// QwQ is a `<think>`-delimiter reasoning model, but until its mechanism is
    /// verified and something actually resolves a config for it, declaring the
    /// capability here would advertise reasoning that doesn't route (nothing
    /// resolves → leak). So it is deliberately NOT detected in v1.
    @Test func qwqNotDeclaredUntilSomethingResolvesIt() {
        #expect(!ReasoningHeuristics.isLikelyReasoningModel("mlx-community/QwQ-32B-4bit"))
        #expect(
            ChatConventionsRegistry.shared.reasoningConfig(
                modelId: "mlx-community/QwQ-32B-4bit", modelType: "qwen2") == nil)
    }

    @Test func nonReasoningIdsAreNotDetected() {
        #expect(
            !ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Qwen2.5-3B-Instruct-4bit"))
        #expect(!ReasoningHeuristics.isLikelyReasoningModel("mlx-community/gemma-3-270m-it-4bit"))
        #expect(
            !ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Llama-3.2-3B-Instruct-4bit"))
    }

    /// Consistency check (not a proof) for the id-keyed families: where
    /// ``ChatConventionsRegistry`` resolves a config from the repo id, the
    /// heuristic must also fire, keeping the two hand-maintained id lists aligned.
    ///
    /// Type-keyed families (Qwen3, DeepSeek-V3) resolve through the model's own
    /// ``ChatConventionsProviding`` declaration instead, which needs a model
    /// instance; that side is covered in `ChatConventionsTests`. The heuristic
    /// must still fire for them, which is asserted separately below.
    @Test func heuristicCoversRegistryResolvedIds() {
        let idKeyed = [
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            "mlx-community/DeepSeek-R1-4bit",
        ]
        for id in idKeyed {
            #expect(
                ChatConventionsRegistry.shared.reasoningConfig(modelId: id, modelType: "qwen2")
                    != nil, "registry should resolve \(id)")
            #expect(ReasoningHeuristics.isLikelyReasoningModel(id), "heuristic missed \(id)")
        }
    }

    /// Type-keyed reasoning families still have to be detectable from the id alone,
    /// since that is all this heuristic ever sees.
    @Test func heuristicCoversTypeKeyedFamilies() {
        #expect(ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Qwen3-4B-4bit"))
    }

    @Test func heuristicAndRegistryAgreeOnNonReasoning() {
        let nonReasoningPairs: [(type: String, id: String)] = [
            ("qwen2", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            ("gemma3", "mlx-community/gemma-3-270m-it-4bit"),
            ("llama", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        ]
        for pair in nonReasoningPairs {
            #expect(
                ChatConventionsRegistry.shared.reasoningConfig(
                    modelId: pair.id, modelType: pair.type) == nil)
            #expect(!ReasoningHeuristics.isLikelyReasoningModel(pair.id))
        }
    }
}
