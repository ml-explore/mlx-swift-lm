// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

/// Exercises Qwen's real templates with scripted tokens and no model weights.
public enum QwenReasoningTemplateTests {
    public enum Continuation: Sendable, CaseIterable {
        case canonical, reorderedParameters, noncanonicalWhitespace, implicitEnd, incomplete
    }

    private static let reasoning = "Read the requested line."
    private static let result = "Qwen structured continuation fixture."
    private static let context: [String: any Sendable] = [
        "enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "medium",
    ]
    private static let call = ToolCall(
        function: .init(
            name: "read_file", arguments: ["path": "README.md", "offset": 1, "limit": 1]),
        id: "read_fixture")
    private static let tools: [ToolSpec] = [
        [
            "type": "function",
            "function": [
                "name": "read_file",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"], "offset": ["type": "integer"],
                        "limit": ["type": "integer"],
                    ],
                    "required": ["path", "offset", "limit"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    private static func wireCall(reordered: Bool = false) -> String {
        let arguments =
            reordered
            ? [("path", "README.md"), ("offset", "1"), ("limit", "1")]
            : [("limit", "1"), ("offset", "1"), ("path", "README.md")]
        return "<tool_call>\n<function=read_file>\n"
            + arguments.map { "<parameter=\($0.0)>\n\($0.1)\n</parameter>\n" }.joined()
            + "</function>\n</tool_call>"
    }

    private static func check(_ condition: Bool, _ message: String) throws {
        guard condition else { throw IntegrationTestFailure(message) }
    }

    public static func importedHistory(tokenizer: any Tokenizer, supportsInlineReasoning: Bool)
        throws
    {
        let generator = Qwen3VLMessageGenerator()
        let assistant = Chat.Message.assistant(
            "<think>\n\(reasoning)\n</think>\n\n", toolCalls: [call])
        let history: [Chat.Message] = [
            .user("Read README.md."), assistant, .tool(result, id: "read_fixture"),
        ]
        let normalized = generator.generate(messages: history)
        let tokens = try tokenizer.applyChatTemplate(
            messages: normalized, tools: tools, additionalContext: context)
        let text = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        try check(
            text.contains("<think>\n\(reasoning)\n</think>\n\n" + wireCall()),
            "The real Qwen template did not reconstruct completed reasoning and all arguments")

        let inlineTokens = try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: history),
            tools: tools, additionalContext: context)
        if supportsInlineReasoning {
            try check(
                tokens == inlineTokens, "Normalization changed Qwen 3.6's rendered token sequence")
        } else {
            try check(
                tokens != inlineTokens,
                "The Qwen 3.8 fixture no longer reproduces inline reasoning loss")
        }

        let userOnly = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: [.user("Read README.md.")]),
            tools: tools, additionalContext: context)
        let generated = tokenizer.encode(
            text: reasoning + "\n</think>\n\n" + wireCall() + "<|im_end|>",
            addSpecialTokens: false)
        try check(
            tokens.starts(with: userOnly + generated),
            "The canonical generated prefix, including the complete tool call, changed")
        if !supportsInlineReasoning {
            try check(
                !inlineTokens.starts(with: userOnly + generated),
                "The inline Qwen 3.8 assistant mapping unexpectedly preserved the cached prefix")
        }

        var primed = Chat.Message.assistant(
            reasoning + "\n</think>\n\n", toolCalls: [call])
        primed.prefilledReasoningStartDelimiter = "<think>"
        let primedHistory: [Chat.Message] = [
            .user("Read README.md."), primed, .tool(result, id: "read_fixture"),
        ]
        let primedTokens = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: primedHistory), tools: tools,
            additionalContext: context)
        let inlinePrimedTokens = try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: primedHistory), tools: tools,
            additionalContext: context)
        try check(
            primedTokens == tokens, "Prompt-primed and explicit reasoning rendered differently")
        try check(
            (inlinePrimedTokens == primedTokens) == supportsInlineReasoning,
            "Prompt-primed reasoning did not preserve Qwen 3.6 or reproduce Qwen 3.8's mismatch")

        let unproven = Chat.Message.assistant(reasoning + "\n</think>\n\nAnswer")
        try check(
            generator.generate(message: unproven)["reasoning_content"] == nil,
            "Imported text without an opening tag acquired invented reasoning provenance")

        let empty = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: [
                .user("First"), .assistant("<think>\n\n</think>\n\nAnswer"), .user("Second"),
            ]), tools: nil, additionalContext: context)
        try check(
            tokenizer.decode(tokenIds: empty, skipSpecialTokens: false)
                .contains("<think>\n\n</think>\n\nAnswer<|im_end|>"),
            "An empty completed reasoning block changed the template's history")

        let completed: [Chat.Message] = [
            .user("First"), .assistant("<think>\nPrivate thought.\n</think>\n\nAnswer"),
            .user("Second"),
        ]
        var dropContext = context
        dropContext["preserve_thinking"] = false
        let dropped = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: completed), tools: nil,
            additionalContext: dropContext)
        try check(
            !tokenizer.decode(tokenIds: dropped, skipSpecialTokens: false).contains(
                "Private thought."),
            "preserve_thinking=false no longer drops earlier reasoning")

        var disabledContext = context
        disabledContext["enable_thinking"] = false
        let disabled = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: [.user("First")]), tools: nil,
            additionalContext: disabledContext)
        try check(
            tokenizer.decode(tokenIds: disabled, skipSpecialTokens: false)
                .hasSuffix("<think>\n\n</think>\n\n"),
            "enable_thinking=false changed the model's generation prompt")
    }

    public static func liveContinuation(
        tokenizer: any Tokenizer, variant: Continuation
    ) async throws {
        guard let eos = tokenizer.convertTokenToId("<|im_end|>") else {
            throw IntegrationTestFailure("Missing Qwen end-of-message token")
        }
        let firstOutput: String
        switch variant {
        case .canonical, .reorderedParameters:
            firstOutput =
                reasoning + "\n</think>\n\n"
                + wireCall(reordered: variant == .reorderedParameters)
        case .noncanonicalWhitespace:
            firstOutput = reasoning + "  \n</think>\n\n" + wireCall()
        case .implicitEnd:
            firstOutput = reasoning + "\n" + wireCall()
        case .incomplete:
            firstOutput = "Unfinished reasoning"
        }
        let scriptedText = [
            firstOutput, firstOutput, "Done.\n</think>\n\n" + result,
            "Done.\n</think>\n\nSecond answer.",
        ]
        let scripts = scriptedText.map {
            tokenizer.encode(text: $0, addSpecialTokens: false) + [eos]
        }
        let vocabularySize = (scripts.flatMap { $0 }.max() ?? eos) + 1
        let model = ScriptedModel(scripts: scripts, vocabularySize: vocabularySize, eos: eos)
        let processor = RecordingProcessor(tokenizer: tokenizer)
        let configuration = ModelConfiguration(
            id: "qwen-reasoning-template-test", eosTokenIds: [eos], toolCallFormat: .qwen35,
            reasoningConfig: QwenReasoningProtocol.tagged)
        let session = ChatSession(
            ModelContext(
                configuration: configuration, model: model, processor: processor,
                tokenizer: tokenizer),
            generateParameters: .init(maxTokens: 256, temperature: 0),
            additionalContext: context, tools: tools)

        let first = try await collect(session.streamDetails(to: "Read README.md."))
        try check(first.info?.cachedPromptTokenCount == 0, "The first prompt was unexpectedly warm")
        if variant != .incomplete {
            try check(first.calls.count == 1, "The scripted read_file call was not parsed once")
            try check(
                first.calls.first?.function.arguments == call.function.arguments,
                "The real tokenizer/parser changed read_file arguments")
        }
        let firstCachedCount = model.lastOffset
        let firstPrompt = processor.prompts[0]
        let appended: [Chat.Message] =
            variant == .incomplete
            ? [.user("Continue.")] : [.tool(result, id: first.calls.first?.id)]
        let second = try await collect(session.streamDetails(to: appended))
        let secondPrompt = processor.prompts[1]
        try check(firstCachedCount >= firstPrompt.count, "The first prompt was not fully processed")
        let representedTokens =
            firstPrompt + Array(scripts[0].prefix(firstCachedCount - firstPrompt.count))
        try check(
            secondPrompt.starts(with: representedTokens) == (variant == .canonical),
            "The actual rendered continuation has an unexpected token-prefix relationship")
        if variant == .canonical {
            try check(
                second.info?.cachedPromptTokenCount == firstCachedCount,
                "The live session did not reuse the complete canonical cached prefix")
            try check(
                model.prefills[1] == Array(secondPrompt.dropFirst(firstCachedCount)),
                "The model received more than the actual continuation suffix")
            let secondCachedCount = model.lastOffset
            let third = try await collect(
                session.streamDetails(to: [.tool(result, id: second.calls.first?.id)]))
            try check(
                third.info?.cachedPromptTokenCount == secondCachedCount,
                "A second tool continuation lost exact cache reuse")
            let thirdCachedCount = model.lastOffset
            let fourth = try await collect(session.streamDetails(to: "Another question."))
            try check(
                fourth.info?.cachedPromptTokenCount == thirdCachedCount,
                "A second user turn lost preserved-reasoning cache reuse")
        } else {
            try check(
                second.info?.cachedPromptTokenCount == 0 && model.prefills[1] == secondPrompt,
                "A mismatching hybrid prompt was reused instead of rebuilt")
        }
    }

    private struct Collected {
        var calls: [ToolCall] = []
        var info: GenerateCompletionInfo?
    }

    private static func collect(_ stream: AsyncThrowingStream<Generation, Error>) async throws
        -> Collected
    {
        var collected = Collected()
        for try await event in stream {
            if let call = event.toolCall { collected.calls.append(call) }
            if let info = event.info { collected.info = info }
        }
        return collected
    }

    private final class RecordingProcessor: UserInputProcessor, @unchecked Sendable {
        let tokenizer: any Tokenizer
        private(set) var prompts: [[Int]] = []

        init(tokenizer: any Tokenizer) { self.tokenizer = tokenizer }

        func prepare(input: UserInput) throws -> LMInput {
            let tokens = try tokenizer.applyChatTemplate(
                messages: Qwen3VLMessageGenerator().generate(from: input),
                tools: input.tools, additionalContext: input.additionalContext)
            prompts.append(tokens)
            return LMInput(tokens: MLXArray(tokens))
        }
    }

    private final class ScriptedModel: Module, LanguageModel, @unchecked Sendable {
        let vocabularySize: Int
        private let scripts: [[Int]]
        private let eos: Int
        private var pass = -1
        private var cursor = 0
        private(set) var prefills: [[Int]] = []
        private(set) var lastOffset = 0

        init(scripts: [[Int]], vocabularySize: Int, eos: Int) {
            self.scripts = scripts
            self.vocabularySize = vocabularySize
            self.eos = eos
            super.init()
        }

        func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
            [KVCacheSimple(), MambaCache()]
        }

        func prepare(
            _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
            prefill: PrefillParameters
        ) throws -> PrepareResult {
            pass += 1
            cursor = 0
            prefills.append(input.text.tokens.asArray(Int.self))
            return .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
            let count = inputs.dim(-1)
            let entry = MLXArray.zeros([1, 1, count, 1])
            _ = cache?.first?.update(keys: entry, values: entry)
            if let recurrent = cache?.last as? MambaCache { recurrent.offset += count }
            lastOffset = cache?.first?.offset ?? 0
            let script = scripts[min(pass, scripts.count - 1)]
            let token = cursor < script.count ? script[cursor] : eos
            cursor += 1
            var logits = [Float](repeating: -30, count: vocabularySize)
            logits[token] = 30
            return MLXArray(logits, [1, 1, vocabularySize])
        }
    }
}
