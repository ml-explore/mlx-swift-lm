# Design: Cache tokenizer-derived logit biases per model

**Date:** 2026-06-25
**Context:** PR #334 (`ml-explore/mlx-swift-lm`, branch `mlx-foundationmodels`), David Koski review comment #7.
**Status:** Approved (brainstorm), pre-implementation.

> **Artifact note:** This file is a local working artifact. `docs/` is untracked on
> this branch by convention (the branch feeds public PR #334), so this spec is **not**
> committed. It lives on disk for reference only.

## 1. Problem

Two logit-bias arrays are recomputed on **every** guided-generation and tool-calling
response, in two near-identical blocks:

- Tool-calling path: `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1097-1112`
- Guided-JSON path: `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1247-1272`

The recomputed values:

| Computation | Inputs | Cost | Varies per |
|---|---|---|---|
| `ClosingTokenBias.compute(tokenizer:, eosTokenId:)` | tokenizer + eosTokenId | full vocab scan → `[Float]` + `MLXArray[vocabSize]` | model only |
| `WhitespaceTokenBias.compute(tokenizer:)` | tokenizer | full vocab scan **+ per-token byte-decode (BPE inverse)** → `MLXArray` + `Set<Int>` | model only |

Both are pure functions of the tokenizer, so they are identical for a model's entire
lifetime, yet they are rebuilt per request. `WhitespaceTokenBias` is the costly one: on a
~150k-token vocab it does ~150k `convertIdToToken` calls plus a byte-decode per token,
every response.

This is David's comment #7 ("`WhitespaceTokenBias` & friends computed per-response → hold
in `ModelCache`"). The comment is a syllogism: *computed per-response → therefore maybe
cache it*, with the hidden premise *...and it doesn't need to be*. The "friends" are
exactly the values incidentally coupled to the response loop with no per-request reason to
be there.

## 2. Scope

**In scope — cache these (model-invariant):**
- `ClosingTokenBias` result
- `WhitespaceTokenBias` result (bias array + whitespace token-ID set)

**Out of scope — stays inline (per-request):**
- `CompletionReserve.estimate(schemaJSON:, tokenizer:)` — depends on the request's
  `schemaJSON` (the `@Generable` schema for guided gen; the tool envelope JSON for tool
  calling). The schema can change response-to-response, so a `modelID`-keyed cache would
  hand request N the reserve computed for request N-1's schema — a correctness bug, not a
  missed optimization. It is genuinely coupled to the response loop, so it is not a member
  of the "friend group." (Internal due-diligence rationale only; not raised to David, who
  never named it.)
- Derived scalars (`completionReserve`, `hardReserve`) — cheap arithmetic over the
  schema-dependent `structuralReserve`; per-request by transitivity.

## 3. Components & data flow

New reference type bundling the three cached values (they share a key, a lifecycle, and a
call site — Option A of the brainstorm). It is a `final class` marked `@unchecked Sendable`
because `MLXArray` is a non-`Sendable` `final class`, and the bundle is stored in the
`ModelCache` actor and returned across its `async` boundary. This mirrors the existing
precedent in `Libraries/MLXGuidedGeneration/XGrammarBridge.swift:63-66, 180-184`, where
`GrammarTokenizer` and `GrammarConstraint` are `final class … @unchecked Sendable` for the
same reason (cached on the model cache, shared across actors). Safety holds because every
field is `let` and read-only after construction — the biases are only *added* to logits in
`GuidedGenerationLoop`, never mutated.

```swift
/// Tokenizer-derived logit biases, cached per model. `@unchecked Sendable`:
/// the arrays are immutable after construction and only read (added to logits),
/// matching the GrammarTokenizer/GrammarConstraint cache pattern.
final class TokenizerBias: @unchecked Sendable {
    let closing: MLXArray
    let whitespace: MLXArray
    let whitespaceTokenIDs: Set<Int>

    init(closing: MLXArray, whitespace: MLXArray, whitespaceTokenIDs: Set<Int>) {
        self.closing = closing
        self.whitespace = whitespace
        self.whitespaceTokenIDs = whitespaceTokenIDs
    }
}
```

`ModelCache` (nested `private actor` in `MLXLanguageModel`) gains:

- one slice: `private var tokenizerBiases: [String: TokenizerBias] = [:]`
- one accessor `makeTokenizerBias(modelID:tokenizer:)` mirroring `makeXGTokenizer`
  (`:141-156`): return the cached entry if present, else call the two enums, store the
  bundle, return it.

`MLXLanguageModel` gains a `static func makeTokenizerBias(modelID:tokenizer:)` wrapper that
delegates into the actor, matching the existing static wrappers (`:383-407`).

Both response blocks replace their inline `ClosingTokenBias.compute(...)` +
`WhitespaceTokenBias.compute(...)` with a single
`await MLXLanguageModel.makeTokenizerBias(modelID:tokenizer:)` fetch, then read `.closing`
/ `.whitespace` / `.whitespaceTokenIDs` off the result. `CompletionReserve.estimate(...)`
and the derived scalars stay exactly where they are.

## 4. Eviction integration

The slice joins the per-model teardown established by the #8 eviction work, in both places:

- `evictAll()` (`:207-214`): add `tokenizerBiases.removeAll()`
- `remove(modelID:)` (`:223-234`): add `tokenizerBiases.removeValue(forKey: modelID)`

This preserves the invariant that eviction clears all per-model state.

## 5. The two enums are untouched

`ClosingTokenBias` and `WhitespaceTokenBias` remain independent pure functions in
`Libraries/MLXGuidedGeneration/`. We cache their *results*; we do **not** fuse their two
vocab scans into one pass. Once cached, the scan is once-per-model — negligible — so fusing
would only muddy two cleanly-tested public utilities for no real-world gain.

## 6. Testing (TDD)

- **Array correctness** is already covered by `Tests/MLXGuidedGenerationTests/ClosingTokenBiasTests.swift`
  and `WhitespaceTokenBiasTests.swift`. Not duplicated.
- **Cache hit (new):** a tokenizer stub that counts `convertIdToToken` calls. Call
  `makeTokenizerBias` twice for one `modelID`; assert the vocab is scanned **once**. Assert
  a different `modelID` triggers a fresh scan. (Counting the scan is implementation-agnostic
  and avoids relying on `MLXArray` identity.)
- **Eviction (extend `Tests/MLXFoundationModelsTests/ModelCacheEvictionTests.swift`):**
  after `evictAll()` / `remove(modelID:)`, the next `makeTokenizerBias` recomputes (scan
  count increments).
- All host-green (macOS 27), following the serialized-suite isolation the eviction tests
  already use (the cache is process-global `static`).

## 7. Sendability (resolved)

`MLXArray` is a `public final class` with no `Sendable` conformance
(`.build/checkouts/mlx-swift/Source/MLX/MLXArray.swift:7`; no conforming extension exists),
so storing it in the actor and returning it across the `async` boundary requires the bundle
to be `@unchecked Sendable` (see §3). This is the same resolution the codebase already uses
for its other actor-cached, non-`Sendable` types — `GrammarTokenizer` and `GrammarConstraint`
in `XGrammarBridge.swift`. Safety rests on immutability: all fields are `let` and the arrays
are only read (added to logits), never mutated, so concurrent reads across requests are safe.
This is distinct from the `@unchecked Sendable` removed from `CompositeLogitProcessor` in
review comment #10 — that one was gratuitous; this one is load-bearing and follows the
established cache precedent.

## 8. Reviewer-facing outcome

Reply to David #7 addresses only the actual friend group:

> Good catch — the closing and whitespace biases were being recomputed per response with no
> reason to be, so they're now cached per-model.

No mention of `CompletionReserve` (never named by David; not a member of the group).
