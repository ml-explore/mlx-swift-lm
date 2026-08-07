// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests for the protocol-agnostic cache-reuse rules that `ChatSession`
/// applies at the start of every turn. The policy is pure, so the whole
/// decision table is exercised without a model, a tokenizer, or a KV cache.
@Suite
struct PromptCacheReusePolicyTests {

    // MARK: - Fixtures

    private func turn(
        prompt: [Int],
        newMedia: Bool = false,
        preparedMedia: Bool = false,
        attentionMask: Bool = false,
        modelState: Bool = false,
        toolResultContinuation: Bool = false
    ) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt,
            carriesNewMedia: newMedia,
            carriesPreparedMedia: preparedMedia,
            carriesAttentionMask: attentionMask,
            carriesModelState: modelState,
            isToolResultContinuation: toolResultContinuation)
    }

    /// A cache whose model-wide timeline agrees with `cached`, unless
    /// `processed` overrides it.
    private func alignedCache(
        _ cached: [Int],
        processed: Int? = nil,
        draftAligned: Bool = true,
        trimmable: Bool = true
    ) -> PromptCacheState {
        let processedTokenCount = processed ?? cached.count
        return PromptCacheState(
            cachedTokens: cached,
            processedTokenCount: processedTokenCount,
            mainCacheIsAligned: processedTokenCount == cached.count,
            draftCacheIsAligned: draftAligned,
            isTrimmable: trimmable)
    }

    // MARK: - Fresh and empty caches

    @Test func `an empty cache prefills the whole prompt`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3]), cache: alignedCache([]))

        #expect(decision == .prefillAll)
        #expect(decision.reusesCachedPrefix == false)
    }

    @Test func `a populated cache with an invalidated ledger rebuilds`() {
        // A cancelled generation clears the ledger but leaves tokens in the
        // cache: there is nothing to splice onto and the cache is stale.
        let cache = PromptCacheState(
            cachedTokens: [],
            processedTokenCount: 12,
            mainCacheIsAligned: false,
            isTrimmable: true)

        #expect(
            PromptCacheReusePolicy().decide(turn: turn(prompt: [1, 2, 3]), cache: cache) == .rebuild
        )
    }

    // MARK: - Prefix extension

    @Test func `a prompt extending the cache appends only its suffix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4, 5]), cache: alignedCache([1, 2, 3]))

        #expect(decision == .appendSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4, 5]))
        #expect(decision.reusesCachedPrefix)
    }

    @Test func `an identical prompt has no suffix to append and rebuilds`() {
        // No new tokens: the common prefix spans the whole prompt, so there is
        // nothing to prefill and nothing safe to rewind to.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3]), cache: alignedCache([1, 2, 3])) == .rebuild)
    }

    @Test func `a misaligned draft cache blocks suffix reuse`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4]),
            cache: alignedCache([1, 2, 3], draftAligned: false))

        #expect(decision == .rebuild)
    }

    @Test func `new media invalidates the cached text prefix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], newMedia: true), cache: alignedCache([1, 2, 3]))

        #expect(decision == .rebuild)
    }

    @Test func `an explicit attention mask blocks suffix reuse`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], attentionMask: true), cache: alignedCache([1, 2, 3]))

        #expect(decision == .rebuild)
    }

    // MARK: - Rewind to the common prefix

    @Test func `a divergent prompt rewinds to the common prefix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 9, 9]), cache: alignedCache([1, 2, 3, 4, 5]))

        #expect(decision == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 3))
        #expect(decision.reusesCachedPrefix)
    }

    @Test func `a fully divergent prompt rebuilds rather than rewinding to nothing`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [9, 9, 9]), cache: alignedCache([1, 2, 3])) == .rebuild)
    }

    @Test(arguments: [
        "prepared media", "attention mask", "model state", "not trimmable",
    ])
    func `a rewind is refused when it cannot be performed safely`(blocker: String) {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(
                prompt: [1, 2, 9, 9],
                preparedMedia: blocker == "prepared media",
                attentionMask: blocker == "attention mask",
                modelState: blocker == "model state"),
            cache: alignedCache([1, 2, 3, 4, 5], trimmable: blocker != "not trimmable"))

        #expect(decision == .rebuild)
    }

    // MARK: - Rule composition

    @Test func `a protocol rule is consulted before the standard rules`() {
        // A response protocol whose token stream is not reproducible by the
        // chat template must be able to claim a turn that the standard rules
        // would otherwise resolve by prefix comparison.
        struct ClaimEverythingRule: PromptCacheReuseRule {
            func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
            {
                .appendSuffix(suffixStart: 0, representedTokens: [42])
            }
        }

        let policy = PromptCacheReusePolicy(protocolRules: [ClaimEverythingRule()])

        #expect(
            policy.decide(turn: turn(prompt: [1, 2, 3, 4]), cache: alignedCache([1, 2, 3]))
                == .appendSuffix(suffixStart: 0, representedTokens: [42]))
    }

    @Test func `a declining protocol rule leaves the standard behavior intact`() {
        struct DecliningRule: PromptCacheReuseRule {
            func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
            {
                nil
            }
        }

        let withRule = PromptCacheReusePolicy(protocolRules: [DecliningRule()])
        let standard = PromptCacheReusePolicy()
        let turn = turn(prompt: [1, 2, 3, 4])
        let cache = alignedCache([1, 2, 3])

        #expect(
            withRule.decide(turn: turn, cache: cache) == standard.decide(turn: turn, cache: cache))
    }
}
