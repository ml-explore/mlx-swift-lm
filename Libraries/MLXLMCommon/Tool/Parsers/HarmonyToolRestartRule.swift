// Copyright © 2026 Apple Inc.

/// Cache-reuse rule for the Harmony protocol's tool restart.
///
/// Harmony analysis is private: the live KV cache holds the true generated
/// path — analysis frames included — while a cold chat-template render omits
/// them. The two token streams therefore stop being prefix-equivalent at the
/// first analysis frame, and the generic prefix rules would compare streams
/// that are not comparable. This rule claims such a turn and splices the tool
/// result onto the cache at the generated `<|call|>` commit instead.
///
/// It is the only place in the cache pipeline that knows about Harmony.
struct HarmonyToolRestartRule: PromptCacheReuseRule {

    /// The commit token separating an assistant function call from its tool
    /// result in a rendered Harmony conversation.
    let callToken: Int

    init(callToken: Int) {
        self.callToken = callToken
    }

    /// Fails when the tokenizer has no Harmony control tokens, in which case
    /// the model is not running the Harmony protocol and the standard rules
    /// apply unchanged.
    init?(tokenizer: any Tokenizer) {
        guard let callToken = HarmonyFrameParser.callTokenID(tokenizer: tokenizer) else {
            return nil
        }
        self.callToken = callToken
    }

    /// Splices only against a cache whose ledger ends at the generated call or
    /// whose sole uncommitted token is that call.
    ///
    /// ``TokenIterator/next()`` commits the token it returns (`step` feeds it
    /// before sampling the next one), so a normal Harmony tool turn leaves the
    /// cache ending at `<|call|>`. A speculative iterator can instead return
    /// `<|call|>` as its final verifier sample before committing it; in that
    /// case the call token is included in the suffix. Any other state — a fresh
    /// or restored cache, an invalidated or rewound ledger — is declined so the
    /// standard rules decide, rather than risking a fragment prefill with no
    /// transcript behind it.
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard turn.isToolResultContinuation,
            !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            let structuredToolCallCount = turn.structuredToolCallCount,
            structuredToolCallCount > 0,
            turn.promptTokens.count(where: { $0 == callToken }) == structuredToolCallCount,
            let callIndex = turn.promptTokens.lastIndex(of: callToken)
        else {
            return nil
        }

        let suffixStart: Int
        if cache.cachedTokens.last == callToken,
            turn.previousGenerationUncommittedTokens.isEmpty
        {
            suffixStart = turn.promptTokens.index(after: callIndex)
        } else if turn.previousGenerationUncommittedTokens == [callToken] {
            // The final speculative sample was visible to the parser but has
            // not been evaluated into the cache yet.
            suffixStart = callIndex
        } else {
            return nil
        }

        guard suffixStart < turn.promptTokens.endIndex else { return nil }

        let representedTokens = cache.cachedTokens + turn.promptTokens[suffixStart...]
        let canContinueWithDraft =
            !turn.usesSpeculativeDecoding
            || (cache.hasDraftCache && cache.draftCacheIsAligned)
        if canContinueWithDraft {
            return .appendSuffix(
                suffixStart: suffixStart,
                representedTokens: representedTokens)
        } else {
            return .appendSuffixToMain(
                suffixStart: suffixStart,
                representedTokens: representedTokens)
        }
    }
}
