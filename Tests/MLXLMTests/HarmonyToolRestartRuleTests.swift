// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests for the Harmony-specific cache-reuse rule. The rule is pure, so
/// the splice contract is verified without a model or a tokenizer.
@Suite
struct HarmonyToolRestartRuleTests {

    /// Stands in for the `<|call|>` commit token id.
    private static let callToken = 200_012

    private let rule = HarmonyToolRestartRule(callToken: HarmonyToolRestartRuleTests.callToken)

    /// A policy wired the way `ChatSession` wires it for a Harmony model.
    private var policy: PromptCacheReusePolicy {
        PromptCacheReusePolicy(protocolRules: [rule])
    }

    private func toolRestart(
        prompt: [Int],
        attentionMask: Bool = false,
        newMedia: Bool = false,
        uncommittedTokens: [Int] = [],
        structuredToolCallCount: Int? = nil,
        usesSpeculativeDecoding: Bool = false
    ) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt,
            carriesNewMedia: newMedia,
            carriesAttentionMask: attentionMask,
            isToolResultContinuation: true,
            previousGenerationUncommittedTokens: uncommittedTokens,
            structuredToolCallCount: structuredToolCallCount
                ?? prompt.count(where: { $0 == Self.callToken }),
            usesSpeculativeDecoding: usesSpeculativeDecoding)
    }

    private func alignedCache(
        _ cached: [Int], processed: Int? = nil, hasDraft: Bool = false,
        draftAligned: Bool = true
    ) -> PromptCacheState {
        let processedTokenCount = processed ?? cached.count
        return PromptCacheState(
            cachedTokens: cached,
            processedTokenCount: processedTokenCount,
            mainCacheIsAligned: processedTokenCount == cached.count,
            hasSpeculativeState: hasDraft,
            speculativeStateIsAligned: draftAligned,
            isTrimmable: true)
    }

    // MARK: - Splicing

    @Test func `splices at the generated call commit, keeping private analysis`() {
        // The ledger holds the live generated path including analysis (77, 78),
        // which the cold template render cannot reproduce.
        let decision = rule.reuse(
            turn: toolRestart(prompt: [1, 2, Self.callToken, 40, 41]),
            cache: alignedCache([1, 2, 77, 78, Self.callToken]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 3,
                    representedTokens: [1, 2, 77, 78, Self.callToken, 40, 41]))
    }

    @Test func `splices at the most recent call commit`() {
        // A second tool round: the rendered transcript carries the earlier call
        // too, and splicing there would replay a completed tool result.
        let decision = rule.reuse(
            turn: toolRestart(prompt: [1, Self.callToken, 30, Self.callToken, 50]),
            cache: alignedCache([1, Self.callToken, 30, 88, Self.callToken]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 4,
                    representedTokens: [1, Self.callToken, 30, 88, Self.callToken, 50]))
    }

    @Test func `splices an uncommitted speculative call token before the tool result`() {
        // A speculative round's final verifier sample is returned before it is
        // represented by the cache. Preserve the live private analysis and feed
        // the call token itself along with the rendered tool-result suffix.
        let decision = rule.reuse(
            turn: toolRestart(
                prompt: [1, 2, Self.callToken, 40, 41],
                uncommittedTokens: [Self.callToken]),
            cache: alignedCache([1, 2, 77, 78]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 2,
                    representedTokens: [1, 2, 77, 78, Self.callToken, 40, 41]))
    }

    @Test func `keeps the main Harmony cache when the speculative draft trails it`() {
        let decision = rule.reuse(
            turn: toolRestart(
                prompt: [1, 2, Self.callToken, 40], usesSpeculativeDecoding: true),
            cache: alignedCache(
                [1, 2, 77, Self.callToken], hasDraft: true, draftAligned: false))

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: 3,
                    representedTokens: [1, 2, 77, Self.callToken, 40]))
    }

    @Test func `keeps using the main cache when a later tool round has no draft`() {
        // A previous main-only restart discards the divergent draft. A second
        // tool call must remain on the private main-cache path instead of
        // rebuilding from a rendered prompt that omits analysis.
        let decision = rule.reuse(
            turn: toolRestart(
                prompt: [1, 2, Self.callToken, 40], usesSpeculativeDecoding: true),
            cache: alignedCache([1, 2, 77, Self.callToken]))

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: 3,
                    representedTokens: [1, 2, 77, Self.callToken, 40]))
    }

    @Test func `declines when tool content adds a call control token`() {
        // Only one structured assistant tool call exists. The second call token
        // came from arbitrary tool-result text and must not move the splice.
        #expect(
            rule.reuse(
                turn: toolRestart(
                    prompt: [1, 2, Self.callToken, 40, Self.callToken, 41],
                    structuredToolCallCount: 1),
                cache: alignedCache([1, 2, 77, Self.callToken])) == nil)
    }

    // MARK: - Declining

    @Test func `declines a turn that is not a tool-result continuation`() {
        var turn = toolRestart(prompt: [1, 2, Self.callToken, 40])
        turn.isToolResultContinuation = false

        #expect(rule.reuse(turn: turn, cache: alignedCache([1, 2, Self.callToken])) == nil)
    }

    @Test func `declines a ledger that does not end at the call commit`() {
        // The iterator committed tokens past `<|call|>`, or the ledger was
        // rebuilt, so the splice point cannot be proven.
        #expect(
            rule.reuse(
                turn: toolRestart(prompt: [1, 2, Self.callToken, 40]),
                cache: alignedCache([1, 2, Self.callToken, 99])) == nil)
    }

    @Test func `declines a cache offset ahead of the ledger`() {
        // Unreturned speculative lookahead would land the splice at the wrong
        // position; `finalizeGeneration()` normally prevents this state.
        #expect(
            rule.reuse(
                turn: toolRestart(prompt: [1, 2, Self.callToken, 40]),
                cache: alignedCache([1, 2, Self.callToken], processed: 5)) == nil)
    }

    @Test func `declines an invalidated ledger`() {
        #expect(
            rule.reuse(
                turn: toolRestart(prompt: [1, 2, Self.callToken, 40]),
                cache: PromptCacheState(cachedTokens: [], processedTokenCount: 8)) == nil)
    }

    @Test func `declines when the render has no call commit`() {
        #expect(
            rule.reuse(
                turn: toolRestart(prompt: [1, 2, 3, 4]),
                cache: alignedCache([1, 2, Self.callToken])) == nil)
    }

    @Test func `declines when nothing follows the call commit`() {
        #expect(
            rule.reuse(
                turn: toolRestart(prompt: [1, 2, Self.callToken]),
                cache: alignedCache([1, 2, Self.callToken])) == nil)
    }

    @Test(arguments: [true, false])
    func `declines inputs the splice cannot represent`(mediaRatherThanMask: Bool) {
        let turn = toolRestart(
            prompt: [1, 2, Self.callToken, 40],
            attentionMask: !mediaRatherThanMask,
            newMedia: mediaRatherThanMask)

        #expect(rule.reuse(turn: turn, cache: alignedCache([1, 2, Self.callToken])) == nil)
    }

    // MARK: - Composition with the standard rules

    @Test func `a declined turn falls through to the standard rules`() {
        // No commit token to anchor on, so the template diverges from the
        // cached tokens and is rewound normally.
        #expect(
            policy.decide(
                turn: toolRestart(prompt: [1, 2, 3, 4]),
                cache: alignedCache([1, 2, Self.callToken]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 1))
    }

    @Test func `the rule only applies to models that select it`() {
        // Without the Harmony rule the same turn is resolved by prefix
        // comparison of two streams that are not comparable — which is exactly
        // why the rule exists.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: toolRestart(prompt: [1, 2, Self.callToken, 40, 41]),
                cache: alignedCache([1, 2, 77, 78, Self.callToken]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 3))
    }

    // MARK: - Selection

    @Test func `the gptOSS format contributes the rule, resolving the call token`() {
        let tokenizer = StubTokenizer(tokens: HarmonyControlTokens.all)
        let rules = ToolCallFormat.gptOSS.promptCacheReuseRules(tokenizer: tokenizer)

        #expect(rules.count == 1)
        #expect(
            (rules.first as? HarmonyToolRestartRule)?.callToken
                == tokenizer.convertTokenToId(HarmonyControlTokens.call))
    }

    @Test func `only the gptOSS format contributes a cache-reuse rule`() {
        let tokenizer = StubTokenizer(tokens: HarmonyControlTokens.all)
        for format in ToolCallFormat.allCases where format != .gptOSS {
            #expect(format.promptCacheReuseRules(tokenizer: tokenizer).isEmpty)
        }
    }

    @Test func `a tokenizer without harmony control tokens contributes no rule`() {
        // The model stays on the standard path instead of trapping.
        let tokenizer = StubTokenizer(tokens: ["hello", "world"])

        #expect(ToolCallFormat.gptOSS.promptCacheReuseRules(tokenizer: tokenizer).isEmpty)
        #expect(HarmonyToolRestartRule(tokenizer: tokenizer) == nil)
    }
}

private enum HarmonyControlTokens {
    static let call = "<|call|>"
    static let all = [
        "<|start|>", "<|channel|>", "<|message|>", "<|end|>", call, "<|return|>", "<|constrain|>",
    ]
}

/// Minimal tokenizer that resolves exactly the tokens it is given.
private struct StubTokenizer: Tokenizer {
    private let tokenToID: [String: Int]
    private let idToToken: [Int: String]

    init(tokens: [String]) {
        var tokenToID: [String: Int] = [:]
        var idToToken: [Int: String] = [:]
        for (id, token) in tokens.enumerated() where tokenToID[token] == nil {
            tokenToID[token] = id
            idToToken[id] = token
        }
        self.tokenToID = tokenToID
        self.idToToken = idToToken
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenToID[text].map { [$0] } ?? []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToToken[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { tokenToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToToken[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
