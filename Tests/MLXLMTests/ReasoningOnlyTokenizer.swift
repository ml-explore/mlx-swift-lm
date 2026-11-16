// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM

/// Decodes everything as an unterminated `<think>` block, so with a `ReasoningConfig` in
/// play the whole turn routes to `.reasoning` and `.chunk` never fires.
///
/// The same `decode` is used on the prompt tail. Text follows its `<think>`, so
/// `promptEndsInsideReasoning` does not treat the prompt as primed: the emitter opens on
/// the `<think>` that starts the generated text instead, and nothing ever closes it.
///
/// The decoded text grows with the token count so the streaming detokenizer sees a
/// non-empty delta per step.
struct ReasoningOnlyTokenizer: MLXLMCommon.Tokenizer {

    private let base = PseudoWordTokenizer()

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        "<think>" + String(repeating: "thinking ", count: tokenIds.count)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var eosTokenId: Int? { base.eosTokenId }
    var unknownToken: String? { base.unknownToken }
    var unknownTokenId: Int? { base.unknownTokenId }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
