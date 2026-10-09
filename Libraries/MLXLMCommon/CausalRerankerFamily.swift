// Copyright © 2026 Apple Inc.

import MLX

/// The prompt, scoring rule, and context limit a causal reranker was trained with.
///
/// A family is independent of the language model backbone: Qwen3-Reranker, Zerank-2, and
/// ContextualAI share the Qwen3 architecture but not their protocol.
package struct CausalRerankerFamily: Sendable {
    package let name: String
    package let scoring: CausalRerankerScoring
    package let maxInputTokens: Int
    private let makeInputProcessor: @Sendable (_ instruction: String?) -> any RerankerInputProcessor

    package var scoreKind: RerankScoreKind {
        switch scoring {
        case .margin: .normalizedRelevance
        case .logit: .logit
        }
    }

    package func inputProcessor(instruction: String?) -> any RerankerInputProcessor {
        makeInputProcessor(instruction)
    }

    /// Resolves the classifier tokens from checkpoint metadata, or else from the tokenizer.
    package func scorePolicy(
        metadata: CausalRerankerMetadata?, tokenizer: any Tokenizer, vocabularySize: Int?
    ) throws -> CausalRerankerScorePolicy {
        let policy =
            try declaredScorePolicy(metadata)
            ?? scoring.defaultPolicy(tokenizer: tokenizer)
        try policy.validate(vocabularySize: vocabularySize)
        return policy
    }

    /// The policy declared by `1_LogitScore/config.json`, checked against the family's score shape.
    package func declaredScorePolicy(
        _ metadata: CausalRerankerMetadata?
    ) throws -> CausalRerankerScorePolicy? {
        guard let metadata else { return nil }
        if let moduleInputName = metadata.moduleInputName, moduleInputName != "causal_logits" {
            throw RerankerError.unsupportedModel(
                "Unsupported reranker module_input_name '\(moduleInputName)'.")
        }
        switch (scoring, metadata.falseTokenID) {
        case (.margin, let negative?):
            return .binaryMargin(positive: metadata.trueTokenID, negative: negative)
        case (.margin, nil):
            throw RerankerError.unsupportedModel(
                "\(name) requires both true_token_id and false_token_id.")
        case (.logit(_, let roundsToBFloat16), nil):
            return .logit(tokenID: metadata.trueTokenID, roundsToBFloat16: roundsToBFloat16)
        case (.logit, _?):
            throw RerankerError.unsupportedModel("\(name) requires a single raw-logit score.")
        }
    }
}

extension CausalRerankerFamily: Equatable {
    package static func == (lhs: Self, rhs: Self) -> Bool { lhs.name == rhs.name }
}

extension CausalRerankerFamily {
    /// `Qwen/Qwen3-Reranker-*`: yes/no margin under an instruction prompt.
    package static let qwen3 = Self(
        name: "Qwen3-Reranker", scoring: .margin(positive: "yes", negative: "no"),
        maxInputTokens: 8_192
    ) { Qwen3RerankerInputProcessor(instruction: $0) }

    /// `zeroentropy/zerank-2`: raw `Yes` logit; the query is the system message.
    package static let zerank2 = Self(
        name: "Zerank-2", scoring: .logit(.text("Yes"), roundsToBFloat16: false),
        maxInputTokens: 32_768
    ) { _ in
        RenderedRerankerInputProcessor { query, document in
            "<|im_start|>system\n\(query)<|im_end|>\n<|im_start|>user\n\(document)<|im_end|>\n<|im_start|>assistant\n"
        }
    }

    /// `ContextualAI/ctxl-rerank-v2-*`: the reference reads the bfloat16 logit of token 0.
    package static let contextual = Self(
        name: "ContextualAI Reranker v2", scoring: .logit(.id(0), roundsToBFloat16: true),
        maxInputTokens: 32_768
    ) { instruction in
        let instruction = instruction.flatMap { $0.isEmpty ? nil : " " + $0 } ?? ""
        return RenderedRerankerInputProcessor { query, document in
            "Check whether a given document contains information helpful to answer the query.\n<Document> \(document)\n<Query> \(query)\(instruction) ??"
        }
    }
}

/// How a causal reranker turns next-token logits into a score.
package enum CausalRerankerScoring: Sendable, Equatable {
    package enum Token: Sendable, Equatable {
        case text(String)
        case id(Int)
    }

    /// `sigmoid(logit[positive] - logit[negative])`, in `0...1`.
    case margin(positive: String, negative: String)

    /// The raw logit of one token.
    case logit(Token, roundsToBFloat16: Bool)

    func defaultPolicy(tokenizer: any Tokenizer) throws -> CausalRerankerScorePolicy {
        switch self {
        case .margin(let positive, let negative):
            .binaryMargin(
                positive: try resolveClassifierToken(positive, tokenizer: tokenizer),
                negative: try resolveClassifierToken(negative, tokenizer: tokenizer))
        case .logit(.id(let id), let roundsToBFloat16):
            .logit(tokenID: id, roundsToBFloat16: roundsToBFloat16)
        case .logit(.text(let token), let roundsToBFloat16):
            .logit(
                tokenID: try resolveClassifierToken(token, tokenizer: tokenizer),
                roundsToBFloat16: roundsToBFloat16)
        }
    }
}

package struct CausalRerankerMetadata: Decodable, Sendable {
    package let trueTokenID: Int
    package let falseTokenID: Int?
    package let moduleInputName: String?

    private enum CodingKeys: String, CodingKey {
        case trueTokenID = "true_token_id"
        case falseTokenID = "false_token_id"
        case moduleInputName = "module_input_name"
    }
}

/// Classifier token IDs resolved against a concrete tokenizer and vocabulary.
package enum CausalRerankerScorePolicy: Sendable, Equatable {
    case binaryMargin(positive: Int, negative: Int)
    case logit(tokenID: Int, roundsToBFloat16: Bool)

    package func validate(vocabularySize: Int?) throws {
        let ids: [Int]
        switch self {
        case .binaryMargin(let positive, let negative):
            guard positive != negative else {
                throw RerankerError.unsupportedModel(
                    "Reranker positive and negative token IDs must differ.")
            }
            ids = [positive, negative]
        case .logit(let tokenID, _):
            ids = [tokenID]
        }
        guard ids.allSatisfy({ $0 >= 0 && $0 < (vocabularySize ?? .max) }) else {
            throw RerankerError.unsupportedModel(
                "Reranker classifier token IDs are outside the model vocabulary.")
        }
    }

    /// Scores each row of `[batch, vocabulary]` logits with one device read.
    package func callAsFunction(_ logits: MLXArray) -> [Double] {
        let logits = logits.reshaped(-1, logits.dim(-1))
        switch self {
        case .binaryMargin(let positive, let negative):
            let pairs = logits.take(MLXArray([Int32(positive), Int32(negative)]), axis: 1)
                .asType(.float32).asArray(Float.self)
            return stride(from: 0, to: pairs.count, by: 2).map {
                RerankerScoreTransform.sigmoid(Double(pairs[$0]) - Double(pairs[$0 + 1]))
            }
        case .logit(let tokenID, let roundsToBFloat16):
            let column = logits[0..., tokenID]
            return (roundsToBFloat16 ? column.asType(.bfloat16) : column)
                .asType(.float32).asArray(Float.self).map(Double.init)
        }
    }
}

private func resolveClassifierToken(_ token: String, tokenizer: any Tokenizer) throws -> Int {
    if let id = tokenizer.convertTokenToId(token) { return id }
    let ids = tokenizer.encode(text: token, addSpecialTokens: false)
    guard let id = ids.first else { throw RerankerError.missingClassifierToken(token) }
    guard ids.count == 1 else { throw RerankerError.classifierTokenIsNotSingleToken(token, ids) }
    return id
}

/// Retokenize complete prompts after truncation, preserving their structural delimiters.
private struct RenderedRerankerInputProcessor: RerankerInputProcessor {
    let render: @Sendable (String, String) -> String

    func encode(
        query: String, document: String, tokenizer: any Tokenizer,
        maxInputTokens: Int?, truncation: RerankTruncationPolicy
    ) throws -> RerankerInput {
        func encode(_ query: String, _ document: String) -> [Int] {
            tokenizer.encode(
                text: render(query, document), addSpecialTokens: false)
        }
        let tokens = encode(query, document)
        guard let limit = maxInputTokens, tokens.count > limit else {
            return RerankerInput(tokenIds: tokens)
        }
        guard truncation == .truncate else {
            throw RerankerError.inputTooLong(actual: tokens.count, maximum: limit)
        }
        var best = encode("", "")
        guard best.count <= limit else {
            throw RerankerError.tokenLimitTooSmall(
                maxInputTokens: limit, requiredTemplateTokens: best.count)
        }
        let queryTokens = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentTokens = tokenizer.encode(text: document, addSpecialTokens: false)
        var lower = 1
        var upper = queryTokens.count + documentTokens.count
        while lower <= upper {
            let budget = lower + (upper - lower) / 2
            let pair = try preparePair(
                first: queryTokens, second: documentTokens,
                maxInputTokens: budget, specialTokenCount: 0, truncation: .truncate)
            let candidate = encode(
                tokenizer.decode(tokenIds: pair.first), tokenizer.decode(tokenIds: pair.second))
            if candidate.count <= limit {
                best = candidate
                lower = budget + 1
            } else {
                upper = budget - 1
            }
        }
        return RerankerInput(tokenIds: best)
    }
}
