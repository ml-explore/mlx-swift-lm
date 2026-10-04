// Copyright © 2026 Apple Inc.

import MLX

/// A prompt protocol is independent of the language model's backbone.
package enum CausalRerankerProtocol: Sendable, Equatable {
    case qwen3
    case zerank2
    case contextual

    package var scoreKind: RerankScoreKind {
        self == .qwen3 ? .normalizedRelevance : .logit
    }

    package var maxInputTokens: Int { self == .qwen3 ? 8_192 : 32_768 }

    package func configuration(
        tokenizer: any Tokenizer, vocabularySize: Int,
        metadata: CausalRerankerMetadata?
    ) throws -> CausalRerankerConfiguration {
        let policy: CausalRerankerScorePolicy =
            if let metadata {
                try metadata.scorePolicy(for: self)
            } else {
                switch self {
                case .qwen3:
                    .binaryMargin(
                        positive: try resolveToken("yes", tokenizer: tokenizer),
                        negative: try resolveToken("no", tokenizer: tokenizer))
                case .zerank2:
                    .logit(
                        tokenID: try resolveToken("Yes", tokenizer: tokenizer),
                        roundToBFloat16: false)
                case .contextual:
                    .logit(tokenID: 0, roundToBFloat16: true)
                }
            }
        try policy.validate(vocabularySize: vocabularySize)
        return CausalRerankerConfiguration(family: self, scorePolicy: policy)
    }

    private func resolveToken(_ token: String, tokenizer: any Tokenizer) throws -> Int {
        if let id = tokenizer.convertTokenToId(token) { return id }
        let ids = tokenizer.encode(text: token, addSpecialTokens: false)
        guard ids.count == 1, let id = ids.first else {
            throw RerankerError.classifierTokenIsNotSingleToken(token, ids)
        }
        return id
    }
}

package struct CausalRerankerMetadata: Decodable, Sendable {
    package let trueTokenID: Int
    package let falseTokenID: Int?
    package let moduleInputName: String?

    package func scorePolicy(for family: CausalRerankerProtocol) throws -> CausalRerankerScorePolicy
    {
        if let moduleInputName, moduleInputName != "causal_logits" {
            throw RerankerError.unsupportedModel(
                "Unsupported reranker module_input_name '\(moduleInputName)'.")
        }
        switch family {
        case .qwen3:
            guard let falseTokenID else {
                throw RerankerError.unsupportedModel(
                    "Qwen3-Reranker requires both true_token_id and false_token_id.")
            }
            return .binaryMargin(positive: trueTokenID, negative: falseTokenID)
        case .zerank2, .contextual:
            guard falseTokenID == nil else {
                throw RerankerError.unsupportedModel(
                    "This reranker requires a single raw-logit score.")
            }
            return .logit(tokenID: trueTokenID, roundToBFloat16: family == .contextual)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case trueTokenID = "true_token_id"
        case falseTokenID = "false_token_id"
        case moduleInputName = "module_input_name"
    }
}

package enum CausalRerankerScorePolicy: Sendable, Equatable {
    case binaryMargin(positive: Int, negative: Int)
    case logit(tokenID: Int, roundToBFloat16: Bool)

    package func validate(vocabularySize: Int) throws {
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
        guard ids.allSatisfy({ (0 ..< vocabularySize).contains($0) }) else {
            throw RerankerError.unsupportedModel(
                "Reranker classifier token IDs are outside the model vocabulary.")
        }
    }

    package func callAsFunction(_ logits: MLXArray) -> Double {
        let logits = logits.reshaped(-1)
        switch self {
        case .binaryMargin(let positive, let negative):
            let margin =
                Double(logits[positive].item(Float.self))
                - Double(logits[negative].item(Float.self))
            return RerankerScoreTransform.sigmoid(margin)
        case .logit(let tokenID, let roundToBFloat16):
            let logit = logits[tokenID]
            return Double((roundToBFloat16 ? logit.asType(.bfloat16) : logit).item(Float.self))
        }
    }
}

package struct CausalRerankerConfiguration: Sendable {
    package let family: CausalRerankerProtocol
    package let scorePolicy: CausalRerankerScorePolicy

    package func inputProcessor(instruction: String?) -> any RerankerInputProcessor {
        switch family {
        case .qwen3:
            return Qwen3RerankerInputProcessor(instruction: instruction)
        case .zerank2:
            return RenderedRerankerInputProcessor { query, document in
                "<|im_start|>system\n\(query)<|im_end|>\n<|im_start|>user\n\(document)<|im_end|>\n<|im_start|>assistant\n"
            }
        case .contextual:
            let instruction = instruction.flatMap { $0.isEmpty ? nil : " " + $0 } ?? ""
            return RenderedRerankerInputProcessor { query, document in
                "Check whether a given document contains information helpful to answer the query.\n<Document> \(document)\n<Query> \(query)\(instruction) ??"
            }
        }
    }
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
