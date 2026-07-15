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

    /// QwQ is a `<think>`-delimiter reasoning model, but until its mechanism is verified
    /// mechanism and it is added to `infer`, declaring the capability here would
    /// advertise reasoning that doesn't route (infer returns nil → leak). So it
    /// is deliberately NOT detected in v1.
    @Test func qwqNotDeclaredUntilInferSupportsIt() {
        #expect(!ReasoningHeuristics.isLikelyReasoningModel("mlx-community/QwQ-32B-4bit"))
        #expect(ReasoningConfig.infer(from: "qwen2", modelId: "mlx-community/QwQ-32B-4bit") == nil)
    }

    @Test func nonReasoningIdsAreNotDetected() {
        #expect(
            !ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Qwen2.5-3B-Instruct-4bit"))
        #expect(!ReasoningHeuristics.isLikelyReasoningModel("mlx-community/gemma-3-270m-it-4bit"))
        #expect(
            !ReasoningHeuristics.isLikelyReasoningModel("mlx-community/Llama-3.2-3B-Instruct-4bit"))
    }

    /// Consistency check (not a proof): for curated `(modelType,
    /// modelId)` pairs where `infer` resolves a config, the heuristic must also
    /// fire — keeping the two hand-maintained lists aligned for known families.
    /// It does not (and cannot) cover arbitrary re-uploads; that's what the
    /// runtime emit-only-when-declared gate + drift log handle.
    @Test func heuristicCoversInferForKnownFamilies() {
        let reasoningPairs: [(type: String, id: String)] = [
            ("qwen3", "mlx-community/Qwen3-4B-4bit"),
            ("qwen2", "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"),
            ("deepseek_v3", "mlx-community/DeepSeek-R1-4bit"),
        ]
        for pair in reasoningPairs {
            #expect(
                ReasoningConfig.infer(from: pair.type, modelId: pair.id) != nil,
                "infer should resolve \(pair.id)")
            #expect(
                ReasoningHeuristics.isLikelyReasoningModel(pair.id),
                "heuristic missed \(pair.id)")
        }
    }

    @Test func heuristicAndInferAgreeOnNonReasoning() {
        let nonReasoningPairs: [(type: String, id: String)] = [
            ("qwen2", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            ("gemma3", "mlx-community/gemma-3-270m-it-4bit"),
            ("llama", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        ]
        for pair in nonReasoningPairs {
            #expect(ReasoningConfig.infer(from: pair.type, modelId: pair.id) == nil)
            #expect(!ReasoningHeuristics.isLikelyReasoningModel(pair.id))
        }
    }
}
