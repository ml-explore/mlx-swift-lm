// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Renders chat input with a tokenizer's chat template.
public struct ScriptedInputProcessor: UserInputProcessor {
    public let tokenizer: any Tokenizer
    public let messageGenerator: any MessageGenerator

    public init(
        tokenizer: any Tokenizer, messageGenerator: any MessageGenerator = DefaultMessageGenerator()
    ) {
        self.tokenizer = tokenizer
        self.messageGenerator = messageGenerator
    }

    public func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        return LMInput(tokens: MLXArray(tokens))
    }
}

/// A transcript, a tokenizer built from it, and a model that follows it.
///
/// ```swift
/// // @_spi(Testing) import MLXLLM
/// let scenario = try ScriptedScenario(
///     transcript: Transcript([.user("hello"), .assistant("hi there")])
/// ) { vocabularySize in
///     Gemma3TextModel(.smallConfiguration(vocabularySize: vocabularySize))
/// }
/// let session = ChatSession(scenario.context)
/// let reply = try await session.respond(to: "hello")  // "hi there"
/// ```
public struct ScriptedScenario {
    public let transcript: Transcript
    public let tokenizer: ScriptedTokenizer
    public let model: ScriptedLogitsModel
    public let context: ModelContext

    /// - Parameters:
    ///   - transcript: the conversation to script.
    ///   - template: the chat template; the generation prompt marks each assistant turn.
    ///   - margin: how strongly the model favors the scripted token.
    ///   - configuration: the context's model configuration. Its `eosTokenIds` and
    ///     `extraEOSTokens` are blocked after a script, like the tokenizer's EOS.
    ///   - innerModel: makes the wrapped model, given the vocabulary size it must support.
    ///     Usually a small model with random weights.
    public init(
        transcript: Transcript,
        template: any ScriptedChatTemplate = MinimalChatTemplate(),
        margin: ScriptedLogitsModel.Margin = .dominant,
        configuration: ModelConfiguration = ModelConfiguration(id: "scripted"),
        innerModel: (_ vocabularySize: Int) throws -> any LanguageModel
    ) throws {
        let tokenizer = ScriptedTokenizer(transcript: transcript, template: template)
        let model = ScriptedLogitsModel(
            wrapping: try innerModel(tokenizer.vocabulary.size),
            scripts: try TurnScripts(
                transcript: transcript, tokenizer: tokenizer,
                stopTokens: configuration.eosTokenIds.union(
                    configuration.extraEOSTokens.compactMap(tokenizer.convertTokenToId))),
            margin: margin)

        self.transcript = transcript
        self.tokenizer = tokenizer
        self.model = model
        self.context = ModelContext(
            configuration: configuration, model: model,
            processor: ScriptedInputProcessor(tokenizer: tokenizer), tokenizer: tokenizer)
    }
}
