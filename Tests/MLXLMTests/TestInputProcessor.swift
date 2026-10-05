// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXScriptedLM

/// Prepares chat input for tests: renders messages with the tokenizer's chat template.
///
/// The default uses ``PseudoWordTokenizer``, whose ids fit the tiny models' `vocabularySize: 100`.
struct TestInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator
    /// When set, every prompt id must be below this, so it fits the model's embedding.
    let vocabularySize: Int?

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator, vocabularySize: Int? = nil
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
        self.vocabularySize = vocabularySize
    }

    internal init() {
        let tokenizer = PseudoWordTokenizer()
        self.configuration = ModelConfiguration(id: "test")
        self.tokenizer = tokenizer
        self.messageGenerator = DefaultMessageGenerator()
        self.vocabularySize = tokenizer.vocabularySize
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if let vocabularySize {
            precondition(
                promptTokens.allSatisfy { (0 ..< vocabularySize).contains($0) },
                "prompt id out of range for vocabularySize \(vocabularySize): \(promptTokens)")
        }

        return LMInput(tokens: MLXArray(promptTokens))
    }
}
