// Copyright © 2026 Apple Inc.

/// Cache-reuse rule for an Onyx tool-result continuation.
///
/// Muse Glimmer's live cache contains private reasoning frames which the cold
/// chat-template render intentionally omits. This rule identifies the last
/// structured assistant→tool frame in the render and appends only the tokens
/// following its `<|eot|>` commit to the live trajectory.
struct OnyxToolRestartRule: PromptCacheReuseRule {
    let startToken: Int
    let messageToken: Int
    let eomToken: Int
    let eotToken: Int
    let tokenizer: any Tokenizer

    init?(tokenizer: any Tokenizer) {
        guard let startToken = tokenizer.convertTokenToId("<|start|>"),
            let messageToken = tokenizer.convertTokenToId("<|message|>"),
            let eomToken = tokenizer.convertTokenToId("<|eom|>"),
            let eotToken = tokenizer.convertTokenToId("<|eot|>"),
            Set([startToken, messageToken, eomToken, eotToken]).count == 4
        else { return nil }
        self.startToken = startToken
        self.messageToken = messageToken
        self.eomToken = eomToken
        self.eotToken = eotToken
        self.tokenizer = tokenizer
    }

    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard turn.isToolResultContinuation,
            !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            let expectedCalls = turn.structuredToolCallCount,
            expectedCalls > 0
        else { return nil }

        let commits = assistantToolCommitIndices(in: turn.promptTokens)
        guard commits.count == expectedCalls, let commit = commits.last else { return nil }

        let suffixStart: Int
        if cache.cachedTokens.last == eotToken,
            turn.previousGenerationUncommittedTokens.isEmpty
        {
            suffixStart = turn.promptTokens.index(after: commit)
        } else if turn.previousGenerationUncommittedTokens == [eotToken] {
            suffixStart = commit
        } else {
            return nil
        }
        guard suffixStart < turn.promptTokens.endIndex else { return nil }

        let represented = cache.cachedTokens + turn.promptTokens[suffixStart...]
        let draftCanContinue =
            !turn.usesSpeculativeDecoding
            || (cache.hasDraftCache && cache.draftCacheIsAligned)
        return draftCanContinue
            ? .appendSuffix(suffixStart: suffixStart, representedTokens: represented)
            : .appendSuffixToMain(suffixStart: suffixStart, representedTokens: represented)
    }

    private func assistantToolCommitIndices(in tokens: [Int]) -> [Int] {
        var commits: [Int] = []
        var index = tokens.startIndex
        while index < tokens.endIndex {
            guard tokens[index] == startToken else {
                index += 1
                continue
            }
            let headerStart = index + 1
            guard let message = tokens[headerStart...].firstIndex(of: messageToken) else { break }
            let header = tokenizer.decode(
                tokenIds: Array(tokens[headerStart ..< message]), skipSpecialTokens: false)
            guard
                let terminator = tokens[(message + 1)...].firstIndex(where: {
                    $0 == eomToken || $0 == eotToken
                })
            else { break }

            let fields = header.split(whereSeparator: \.isWhitespace)
            if fields.first == "assistant",
                let to = fields.dropFirst().first(where: { $0.hasPrefix("to=") })
            {
                let recipient = to.dropFirst(3)
                if recipient != "self", recipient != "user", !recipient.isEmpty {
                    commits.append(terminator)
                }
            }
            index = terminator + 1
        }
        return commits
    }
}
