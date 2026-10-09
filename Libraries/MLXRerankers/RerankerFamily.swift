// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// The prompt and scoring protocol a reranker checkpoint was trained with.
///
/// ``RerankerModelFactory`` identifies published checkpoints from their name and
/// configuration. Pass a family when loading a renamed or fine-tuned checkpoint that keeps
/// its base model's protocol. Encoder rerankers declare their protocol in their
/// classification head and have no family.
public enum RerankerFamily: Sendable, Hashable, CaseIterable {
    /// `Qwen/Qwen3-Reranker-*`: yes/no relevance normalized to `0...1`.
    case qwen3Reranker

    /// `zeroentropy/zerank-2`: raw logit of `Yes`.
    case zerank2

    /// `ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b`: raw logit of token 0.
    case contextual

    /// `jinaai/jina-reranker-v3`: listwise cosine similarity over at most 64 documents.
    case jinaV3

    /// `jinaai/jina-reranker-v3.5`: listwise cosine similarity with fused query blocks.
    case jinaV35

    /// Identifies a family from the last path component of a checkpoint identifier, so
    /// conversions such as `mlx-community/zerank-2-4bit` resolve to their source protocol.
    init?(checkpointName: String) {
        let name = checkpointName.split(separator: "/").last.map { $0.lowercased() } ?? ""
        guard
            let family = Self.allCases.first(where: {
                name == $0.checkpointStem || name.hasPrefix($0.checkpointStem + "-")
            })
        else { return nil }
        self = family
    }

    private var checkpointStem: String {
        switch self {
        case .qwen3Reranker: "qwen3-reranker"
        case .zerank2: "zerank-2"
        case .contextual: "ctxl-rerank-v2-instruct-multilingual-1b"
        case .jinaV3: "jina-reranker-v3"
        case .jinaV35: "jina-reranker-v3.5"
        }
    }

    var architecture: RerankerArchitecture {
        switch self {
        case .qwen3Reranker: .causal(.qwen3)
        case .zerank2: .causal(.zerank2)
        case .contextual: .causal(.contextual)
        case .jinaV3: .jina(.v3)
        case .jinaV35: .jina(.v35)
        }
    }
}
