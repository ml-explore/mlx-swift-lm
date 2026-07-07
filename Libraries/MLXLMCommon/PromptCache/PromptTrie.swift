// Copyright © 2025 Apple Inc.

import Foundation

/// Result of a ``PromptTrie/search(model:tokens:)``.
///
/// Ported from the Python `mlx_lm` `PromptTrieResult` dataclass
/// (`mlx_lm/models/cache.py`). Exactly one of `exact`/`shorter`/`longer` is the
/// useful hit for a given query; `commonPrefix` is the length of the longest
/// path shared with any stored sequence.
public struct PromptTrieResult<Model: Hashable> {
    public let model: Model
    /// The queried tokens, when an exact stored sequence matches them.
    public let exact: [Int]?
    /// The longest stored *prefix* of the queried tokens that carries a value.
    public let shorter: [Int]?
    /// The shortest stored sequence that *extends beyond* the queried tokens.
    public let longer: [Int]?
    /// Length of the common prefix walked into the trie.
    public let commonPrefix: Int
}

/// A prefix trie keyed by integer tokens, storing an arbitrary `Value` at any
/// node reached by a token sequence.
///
/// Direct port of Python `mlx_lm`'s `PromptTrie` (`mlx_lm/models/cache.py`),
/// generalised over the value type. Used by ``LRUPromptCache`` to find the
/// nearest reusable KV-cache for an incoming prompt.
///
/// Not thread-safe (mirrors the Python single-server-thread assumption); wrap in
/// a `SerialAccessContainer` for concurrent use.
public final class PromptTrie<Model: Hashable, Value> {

    private final class Node {
        var children: [Int: Node] = [:]
        var value: Value?

        /// A node is prunable once it holds neither a value nor any children —
        /// the Swift analogue of Python's `len(node) > 0` dict check.
        var isEmpty: Bool { value == nil && children.isEmpty }
    }

    private var roots: [Model: Node] = [:]

    public init() {}

    /// Insert `value` at `tokens` for `model`, returning the previous value (if any).
    @discardableResult
    public func add(model: Model, tokens: [Int], value: Value) -> Value? {
        let root: Node
        if let existing = roots[model] {
            root = existing
        } else {
            root = Node()
            roots[model] = root
        }
        var current = root
        for tok in tokens {
            if let next = current.children[tok] {
                current = next
            } else {
                let next = Node()
                current.children[tok] = next
                current = next
            }
        }
        let prev = current.value
        current.value = value
        return prev
    }

    /// Fetch the value stored at exactly `tokens`. The caller must know the path
    /// exists (mirrors Python's `KeyError`-on-miss contract) — it is only called
    /// with a sequence a prior ``search(model:tokens:)`` returned.
    public func get(model: Model, tokens: [Int]) -> Value {
        var current = roots[model]!
        for tok in tokens {
            current = current.children[tok]!
        }
        return current.value!
    }

    /// Remove and return the value at `tokens`, pruning any nodes left empty.
    @discardableResult
    public func pop(model: Model, tokens: [Int]) -> Value {
        let root = roots[model]!
        var path: [Node] = [root]
        for tok in tokens {
            path.append(path[path.count - 1].children[tok]!)
        }
        let value = path[path.count - 1].value!
        path[path.count - 1].value = nil
        var i = tokens.count
        while i > 0 {
            let node = path[i]
            let parent = path[i - 1]
            let tok = tokens[i - 1]
            if !node.isEmpty { break }
            parent.children[tok] = nil
            i -= 1
        }
        return value
    }

    /// Remove and return every value stored at a *proper* prefix of `tokens`,
    /// as `(prefixLength, value)` pairs. Used when inserting a trimmable cache:
    /// its prefixes are redundant and just take space.
    @discardableResult
    public func popPrefixes(model: Model, tokens: [Int]) -> [(Int, Value)] {
        var values: [(Int, Value)] = []
        guard var current = roots[model] else { return values }
        for (i, tok) in tokens.enumerated() {
            if let v = current.value {
                values.append((i, v))
                current.value = nil
            }
            guard let next = current.children[tok] else { break }
            current = next
        }
        return values
    }

    /// Find the nearest stored sequence to `tokens`.
    ///
    /// Baked-in upstream fix (mlx-lm #1495): the shorter-prefix guard is
    /// `lastIndex >= 0` (Python shipped `last_index > 0`), so a value stored at a
    /// single-token prefix — reached at `lastIndex == 0` — is actually returned.
    public func search(model: Model, tokens: [Int]) -> PromptTrieResult<Model> {
        guard let root = roots[model] else {
            return PromptTrieResult(
                model: model, exact: nil, shorter: nil, longer: nil, commonPrefix: 0)
        }

        var current = root
        if tokens.isEmpty && current.value != nil {
            return PromptTrieResult(
                model: model, exact: [], shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Walk the tokens as far as the trie allows.
        var lastIndex = -1
        var index = 0
        while index < tokens.count, let next = current.children[tokens[index]] {
            current = next
            if current.value != nil { lastIndex = index }
            index += 1
        }

        // Exact match.
        if lastIndex == tokens.count - 1 && lastIndex >= 0 {
            return PromptTrieResult(
                model: model, exact: tokens, shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Longest stored prefix carrying a value. (mlx-lm#1495 fix: `>= 0`, not `> 0`.)
        var shorter: [Int]? = nil
        if lastIndex >= 0 {
            shorter = Array(tokens[0 ..< (lastIndex + 1)])
        }

        // Shortest stored sequence extending beyond the queried tokens.
        var longer: [Int]? = nil
        let commonPrefix = index
        if index > 0 {
            var best: [Int]? = nil
            var stack: [(Node, [Int])] = [(current, [])]
            while let (node, extra) = stack.popLast() {
                if node.value != nil {
                    if best == nil || extra.count < best!.count { best = extra }
                } else if best == nil || extra.count < best!.count {
                    for (tok, child) in node.children {
                        stack.append((child, extra + [tok]))
                    }
                }
            }
            if let best {
                longer = Array(tokens[0 ..< index]) + best
            }
        }

        return PromptTrieResult(
            model: model, exact: nil, shorter: shorter, longer: longer, commonPrefix: commonPrefix)
    }
}
