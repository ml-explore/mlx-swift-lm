// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// A test tokenizer -- this can be used in place of a real tokenizer for unit/integration tests.
struct TestTokenizer: MLXLMCommon.Tokenizer {

    let length = 8
    let maxLength = 50

    /// a token outside the range that the model will generate, see vocabularySize
    let _eosTokenId = 101
    let _unknownTokenId = 102

    let vocabularySize: Int
    let vocabulary: [Int: String]

    init(vocabularySize: Int = 100) {
        let letters = "abcdefghijklmnopqrstuvwxyz"
        self.vocabularySize = vocabularySize
        self.vocabulary = Dictionary(
            uniqueKeysWithValues: (0 ..< vocabularySize)
                .map { t in
                    (
                        t,
                        String(
                            (0 ..< ((3 ..< 8).randomElement() ?? 3)).compactMap { _ in
                                letters.randomElement()
                            })
                    )
                }
        )
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (0 ..< length).map { _ in
            Int.random(in: 1 ..< vocabularySize)
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var tokenIds = tokenIds
        if tokenIds.count > maxLength {
            tokenIds.append(_eosTokenId)
        }
        return tokenIds.map { convertIdToToken($0) ?? "" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int.random(in: 1 ..< vocabularySize)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == eosTokenId {
            return "EOS"
        }
        return vocabulary[id]
    }

    var bosToken: String? = nil
    var eosToken: String? = nil
    var eosTokenId: Int? { _eosTokenId }

    var unknownToken: String? = nil

    var unknownTokenId: Int? { _unknownTokenId }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(text: "")
    }

}

struct TestInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
    }

    internal init() {
        self.configuration = ModelConfiguration(id: "test")
        self.tokenizer = TestTokenizer()
        self.messageGenerator = DefaultMessageGenerator()
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        return LMInput(tokens: MLXArray(promptTokens))
    }
}
