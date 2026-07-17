// Copyright © 2025 Apple Inc.

/// Pre-load heuristics for deciding whether a model identifier looks like a
/// reasoning-capable family.
///
/// This is a standalone, opt-in helper — nothing in `MLXLMCommon` calls it.
/// It exists for callers that need to guess reasoning capability from a repo
/// id alone (e.g. before any model files are downloaded, when no other signal
/// is available). Callers that have a stronger signal, or that simply declare
/// their capabilities explicitly, should not use it.
///
/// It is intentionally NOT a provable superset of
/// ``ReasoningConfig/infer(from:modelId:configData:)``: `infer` also keys on
/// `model_type`, which this heuristic never sees. A community re-upload with a
/// non-matching repo name but a reasoning `model_type` resolves a
/// `ReasoningConfig` yet may not match here. Callers who need a stricter
/// guarantee should declare `.reasoning` themselves.
public enum ReasoningHeuristics {

    /// Lowercased substrings that mark a likely reasoning-capable model id.
    private static let reasoningModelMarkers = [
        "qwen3",  // Qwen3 family
        "deepseek-r1",  // DeepSeek-R1 and R1-Distill
        "r1-distill",  // R1-Distill re-uploads not prefixed "deepseek-"
    ]

    /// Whether the model identifier looks like a reasoning-capable model.
    public static func isLikelyReasoningModel(_ modelID: String) -> Bool {
        let lower = modelID.lowercased()
        return reasoningModelMarkers.contains { lower.contains($0) }
    }
}
