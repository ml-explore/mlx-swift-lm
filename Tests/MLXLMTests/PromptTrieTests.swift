// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("PromptTrie")
struct PromptTrieTests {

    @Test func addGetReturnsStoredValue() {
        let trie = PromptTrie<String, Int>()
        #expect(trie.add(model: "m", tokens: [1, 2, 3], value: 42) == nil)
        #expect(trie.get(model: "m", tokens: [1, 2, 3]) == 42)
    }

    @Test func addReturnsDisplacedValue() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2], value: 1)
        #expect(trie.add(model: "m", tokens: [1, 2], value: 2) == 1)
        #expect(trie.get(model: "m", tokens: [1, 2]) == 2)
    }

    @Test func exactMatch() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2, 3], value: 1)
        let r = trie.search(model: "m", tokens: [1, 2, 3])
        #expect(r.exact == [1, 2, 3])
        #expect(r.shorter == nil)
        #expect(r.longer == nil)
    }

    @Test func longerMatchIsShortestExtension() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2, 3, 4], value: 1)
        _ = trie.add(model: "m", tokens: [1, 2, 3, 4, 5, 6], value: 2)
        // Query [1,2] is a strict prefix of both stored sequences; the shortest
        // extension is [1,2,3,4].
        let r = trie.search(model: "m", tokens: [1, 2])
        #expect(r.exact == nil)
        #expect(r.longer == [1, 2, 3, 4])
        #expect(r.commonPrefix == 2)
    }

    @Test func shorterMatchIsLongestStoredPrefix() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2], value: 1)
        // Query [1,2,9] has stored prefix [1,2] and then diverges.
        let r = trie.search(model: "m", tokens: [1, 2, 9])
        #expect(r.shorter == [1, 2])
        #expect(r.exact == nil)
        #expect(r.commonPrefix == 2)
        // Parity quirk with Python `mlx_lm`: `search` also reports the prefix
        // itself as `longer` here, but `fetchNearestCache` ignores it because
        // `commonPrefix (2) > shortLength (2)` is false, so the shorter hit wins.
        #expect(r.longer == [1, 2])
    }

    @Test func missingModelReturnsEmptyResult() {
        let trie = PromptTrie<String, Int>()
        let r = trie.search(model: "absent", tokens: [1, 2])
        #expect(r.exact == nil)
        #expect(r.shorter == nil)
        #expect(r.longer == nil)
        #expect(r.commonPrefix == 0)
    }

    @Test func popPrunesEmptyNodes() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2, 3], value: 1)
        #expect(trie.pop(model: "m", tokens: [1, 2, 3]) == 1)
        // Fully pruned: the sequence no longer matches anything.
        let r = trie.search(model: "m", tokens: [1, 2, 3])
        #expect(r.exact == nil)
        #expect(r.shorter == nil)
        #expect(r.longer == nil)
    }

    @Test func popKeepsSiblingBranches() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1, 2], value: 1)
        _ = trie.add(model: "m", tokens: [1, 3], value: 2)
        _ = trie.pop(model: "m", tokens: [1, 2])
        #expect(trie.get(model: "m", tokens: [1, 3]) == 2)
    }

    @Test func popPrefixesRemovesOnlyProperPrefixes() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [1], value: 1)
        _ = trie.add(model: "m", tokens: [1, 2], value: 2)
        _ = trie.add(model: "m", tokens: [1, 2, 3], value: 3)
        let popped = trie.popPrefixes(model: "m", tokens: [1, 2, 3])
        // Proper prefixes [1] (len 1) and [1,2] (len 2) are popped; the full
        // path [1,2,3] is not.
        #expect(popped.map { $0.0 } == [1, 2])
        #expect(popped.map { $0.1 } == [1, 2])
        #expect(trie.get(model: "m", tokens: [1, 2, 3]) == 3)
    }

    // MARK: - Regression: upstream mlx-lm #1495

    /// A value stored at a single-token prefix must be returned as `shorter`.
    /// Buggy upstream code guards with `last_index > 0`, which drops the
    /// `last_index == 0` case; the fix is `>= 0`. Reverting the guard in
    /// `PromptTrie.search` makes this fail.
    @Test func regressionSingleTokenPrefixIsFound_1495() {
        let trie = PromptTrie<String, Int>()
        _ = trie.add(model: "m", tokens: [5], value: 1)
        let r = trie.search(model: "m", tokens: [5, 9])
        #expect(r.shorter == [5])
        #expect(r.exact == nil)
    }
}
