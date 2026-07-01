# Tokenizer-Bias Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the two model-invariant logit-bias arrays (closing + whitespace) per model in `ModelCache` instead of recomputing them on every guided-generation and tool-calling response.

**Architecture:** Add a `final class TokenizerBias: @unchecked Sendable` bundling the closing bias, whitespace bias, and whitespace token-ID set. Store it in a new `[String: TokenizerBias]` slice in the `ModelCache` actor, fetched via a `makeTokenizerBias(modelID:tokenizer:)` accessor that mirrors the existing `makeXGTokenizer`. Wire the slice into both eviction paths. Replace the two inline `ClosingTokenBias.compute` + `WhitespaceTokenBias.compute` blocks with a single cached fetch. `CompletionReserve` stays inline (it is per-request / schema-dependent).

**Tech Stack:** Swift 6, Swift Concurrency (actors), Swift Testing, MLX, xcodebuild.

## Global Constraints

- All adapter code lives inside `#if FoundationModelsIntegration` + `#if canImport(FoundationModels, _version: 2)`. Test files mirror this gate exactly.
- `MLXArray` is a non-`Sendable` `public final class` — any type holding it that crosses an actor boundary must be `@unchecked Sendable`, justified by `let`-only fields + read-only use (matches `GrammarTokenizer`/`GrammarConstraint` in `XGrammarBridge.swift`).
- Scope discipline: touch only files this change owns (`MLXLanguageModel.swift` + a new test file). Do not move/rename pre-existing files. `CompletionReserve` and the two bias enums are NOT modified.
- Cache-touching tests run under the serialized `FoundationModelsCacheTests` parent suite (the process-global `static let cache` + `evictAll()` make unsynchronized suites race).
- Host-green target: macOS 27 (`-destination 'platform=macOS'`). Confirm the active Xcode points at a 27 SDK before building (`xcode-select -p`); values drift.
- `docs/` is untracked on this branch (feeds public PR #334) — do not commit plan/spec docs.

---

### Task 1: `TokenizerBias` type, cache slice, accessor, static wrapper

**Files:**
- Modify: `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (add type after `ConstraintKind` ~:29; add slice ~:54; add actor method after `makeXGTokenizer` ~:156; add static wrapper after `makeXGTokenizer` static wrapper ~:388)
- Test: `Tests/MLXFoundationModelsTests/TokenizerBiasCacheTests.swift` (create)

**Interfaces:**
- Produces:
  - `final class TokenizerBias: @unchecked Sendable` with `let closing: MLXArray`, `let whitespace: MLXArray`, `let whitespaceTokenIDs: Set<Int>`, and a memberwise `init`.
  - actor method `func makeTokenizerBias(modelID: String, tokenizer: any Tokenizer) -> TokenizerBias`
  - static wrapper `static func makeTokenizerBias(modelID: String, tokenizer: any Tokenizer) async -> TokenizerBias`
- Consumes: `ClosingTokenBias.compute(tokenizer:eosTokenId:)`, `WhitespaceTokenBias.compute(tokenizer:)` (existing, in `MLXGuidedGeneration`).

- [ ] **Step 1: Write the failing tests**

Create `Tests/MLXFoundationModelsTests/TokenizerBiasCacheTests.swift`:

```swift
// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    // Extends the serialized parent declared in ModelCacheEvictionTests.swift so these
    // cache-touching tests never run concurrently with the other cache suites (the
    // process-global `static let cache` + key-agnostic `evictAll()` would otherwise race).
    extension FoundationModelsCacheTests {

        @Suite("MLXLanguageModel tokenizer-bias cache")
        struct TokenizerBiasCaching {

            @Test("makeTokenizerBias scans the vocab once, then serves from cache")
            func cachesPerModel() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let id = "org/bias-\(UUID().uuidString)"

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                let afterFirst = tok.idLookupCount
                #expect(afterFirst > 0, "first call must scan the vocab")

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                #expect(
                    tok.idLookupCount == afterFirst,
                    "second call for the same model must hit the cache, not rescan")
            }

            @Test("a different modelID computes a fresh bias")
            func isPerModel() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let idA = "org/bias-a-\(UUID().uuidString)"
                let idB = "org/bias-b-\(UUID().uuidString)"

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: idA, tokenizer: tok)
                let afterA = tok.idLookupCount
                _ = await MLXLanguageModel.makeTokenizerBias(modelID: idB, tokenizer: tok)
                #expect(
                    tok.idLookupCount > afterA,
                    "a new modelID must trigger a fresh vocab scan")
            }
        }
    }

    // MARK: - Fixtures

    /// Tokenizer with a fixed vocab that counts `convertIdToToken` calls, so a test can
    /// assert whether a bias computation re-scanned the vocab (cache miss) or not (hit).
    /// `@unchecked Sendable`: the counter is mutated only from serialized test calls.
    private final class CountingTokenizer: Tokenizer, @unchecked Sendable {
        let tokens: [String]
        private(set) var idLookupCount = 0

        init(tokens: [String]) { self.tokens = tokens }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { tokens.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            idLookupCount += 1
            guard id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (confirm `xcode-select -p` points at a 27-SDK Xcode first):
```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXFoundationModelsTests/FoundationModelsCacheTests \
  2>&1 | tee /tmp/build-biascache-t1-fail.log | tail -5
```
Expected: COMPILE FAILURE — `type 'MLXLanguageModel' has no member 'makeTokenizerBias'`. (Confirm via `grep -E "error:|makeTokenizerBias" /tmp/build-biascache-t1-fail.log`.)

- [ ] **Step 3: Add the `TokenizerBias` type**

In `MLXLanguageModel.swift`, immediately after the `ConstraintKind` enum (after line 29), add:

```swift
        // MARK: - Tokenizer Bias Cache Entry

        /// Tokenizer-derived logit biases, cached per model. Both arrays are pure
        /// functions of the tokenizer, so they are identical for a model's lifetime.
        /// `@unchecked Sendable`: every field is `let` and read-only after construction
        /// (the arrays are only *added* to logits in `GuidedGenerationLoop`, never
        /// mutated), and the entry is shared across actors via `ModelCache` — the same
        /// pattern as `GrammarTokenizer`/`GrammarConstraint` in `XGrammarBridge.swift`.
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

- [ ] **Step 4: Add the cache slice**

In `ModelCache`, after the `constraintTemplates` declaration (line 54), add:

```swift
            /// Cached per-model logit biases (closing + whitespace). Pure functions of
            /// the tokenizer, so computed once per model and reused across requests.
            private var tokenizerBiases: [String: TokenizerBias] = [:]
```

- [ ] **Step 5: Add the actor accessor**

In `ModelCache`, after `makeXGTokenizer(modelID:tokenizer:)` (after line 156), add:

```swift
            /// Gets or creates the cached tokenizer-derived logit biases for a model.
            func makeTokenizerBias(
                modelID: String,
                tokenizer: any Tokenizer
            ) -> TokenizerBias {
                if let cached = tokenizerBiases[modelID] {
                    return cached
                }
                let closing = ClosingTokenBias.compute(
                    tokenizer: tokenizer,
                    eosTokenId: tokenizer.eosTokenId
                )
                let (whitespace, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                    tokenizer: tokenizer
                )
                let bias = TokenizerBias(
                    closing: closing,
                    whitespace: whitespace,
                    whitespaceTokenIDs: whitespaceTokenIDs
                )
                tokenizerBiases[modelID] = bias
                return bias
            }
```

- [ ] **Step 6: Add the static wrapper**

In `MLXLanguageModel`, after the `makeXGTokenizer` static wrapper (after line 388), add:

```swift
            /// Gets the cached per-model tokenizer-derived logit biases (closing +
            /// whitespace), computing them on first use.
            static func makeTokenizerBias(
                modelID: String,
                tokenizer: any Tokenizer
            ) async -> TokenizerBias {
                await cache.makeTokenizerBias(modelID: modelID, tokenizer: tokenizer)
            }
```

- [ ] **Step 7: Run the tests to verify they pass**

Run:
```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXFoundationModelsTests/FoundationModelsCacheTests \
  2>&1 | tee /tmp/build-biascache-t1-pass.log | tail -5
```
Expected: `** TEST SUCCEEDED **`. Verify count, not just the banner: `grep -E "Test Suite 'TokenizerBiasCaching'|passed|failed|error:" /tmp/build-biascache-t1-pass.log` — expect 2 tests passed in `TokenizerBiasCaching`, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add Libraries/MLXFoundationModels/MLXLanguageModel.swift Tests/MLXFoundationModelsTests/TokenizerBiasCacheTests.swift
git commit -m "feat(mlx-fm): cache per-model tokenizer logit biases in ModelCache"
```

---

### Task 2: Wire the slice into eviction

**Files:**
- Modify: `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (`evictAll()` ~:207-214; `remove(modelID:)` ~:223-234)
- Test: `Tests/MLXFoundationModelsTests/TokenizerBiasCacheTests.swift` (add two tests to the existing `TokenizerBiasCaching` suite)

**Interfaces:**
- Consumes: `MLXLanguageModel.makeTokenizerBias(modelID:tokenizer:)`, `MLXLanguageModel.evictAll()`, `MLXLanguageModel.evict()` (Task 1 + existing #8 work).

- [ ] **Step 1: Write the failing tests**

In `TokenizerBiasCacheTests.swift`, add inside the `TokenizerBiasCaching` struct (after `isPerModel`):

```swift
            @Test("evictAll() forces a recompute on the next call")
            func evictAllClearsBias() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let id = "org/bias-evictall-\(UUID().uuidString)"

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                let afterFirst = tok.idLookupCount

                await MLXLanguageModel.evictAll()

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                #expect(
                    tok.idLookupCount > afterFirst,
                    "evictAll() must drop the cached bias so the next call rescans")
            }

            @Test("evict() drops only this model's cached bias")
            func evictIsPerModel() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let id = "org/bias-evict-\(UUID().uuidString)"
                let model = MLXLanguageModel(
                    modelID: id,
                    capabilities: LanguageModelCapabilities(capabilities: []),
                    from: EvictBiasStubDownloader(),
                    using: EvictBiasStubTokenizerLoader(),
                    locatedBy: { _ in URL(fileURLWithPath: "/no/such/path") }
                )

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                let afterFirst = tok.idLookupCount

                await model.evict()

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                #expect(
                    tok.idLookupCount > afterFirst,
                    "evict() must drop this model's cached bias so the next call rescans")
            }
```

And add the two minimal stubs the `evict()` test's model needs (file-private, after `CountingTokenizer`):

```swift
    /// Minimal no-op transport stubs so an `MLXLanguageModel` can be constructed purely to
    /// exercise the instance `evict()` path. They are never driven to a real load here.
    private final class EvictBiasStubDownloader: Downloader, @unchecked Sendable {
        func download(
            id: String,
            revision: String?,
            matching patterns: [String],
            useLatest: Bool,
            progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> URL { URL(fileURLWithPath: "/no/such/path") }
    }

    private final class EvictBiasStubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
        func load(from directory: URL) async throws -> any Tokenizer {
            CountingTokenizer(tokens: [])
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXFoundationModelsTests/FoundationModelsCacheTests \
  2>&1 | tee /tmp/build-biascache-t2-fail.log | tail -10
```
Expected: `evictAllClearsBias` and `evictIsPerModel` FAIL — the second `makeTokenizerBias` still serves the cached value (count unchanged), because eviction does not yet clear `tokenizerBiases`. Confirm: `grep -E "evictAllClearsBias|evictIsPerModel|failed" /tmp/build-biascache-t2-fail.log`.

- [ ] **Step 3: Clear the slice in `evictAll()`**

In `ModelCache.evictAll()` (line 207-214), add the `tokenizerBiases` line alongside the others:

```swift
            func evictAll() {
                containers.removeAll()
                loadingTasks.removeAll()
                suppressedLoadIDs.removeAll()
                xgTokenizers.removeAll()
                constraintTemplates.removeAll()
                tokenizerBiases.removeAll()
                lastErrors.removeAll()
            }
```

- [ ] **Step 4: Clear the slice in `remove(modelID:)`**

In `ModelCache.remove(modelID:)` (line 223-234), add the per-model removal alongside the others (e.g. after the `constraintTemplates` filter):

```swift
                constraintTemplates = constraintTemplates.filter {
                    !$0.key.hasPrefix("\(modelID):")
                }
                tokenizerBiases.removeValue(forKey: modelID)
                lastErrors.removeValue(forKey: modelID)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXFoundationModelsTests/FoundationModelsCacheTests \
  2>&1 | tee /tmp/build-biascache-t2-pass.log | tail -5
```
Expected: `** TEST SUCCEEDED **`. Verify: `grep -E "Test Suite 'TokenizerBiasCaching'|passed|failed" /tmp/build-biascache-t2-pass.log` — 4 tests in `TokenizerBiasCaching`, 0 failures. Also confirm the sibling `CacheEviction` suite still passes (no regression from the teardown edits).

- [ ] **Step 6: Commit**

```bash
git add Libraries/MLXFoundationModels/MLXLanguageModel.swift Tests/MLXFoundationModelsTests/TokenizerBiasCacheTests.swift
git commit -m "feat(mlx-fm): evict cached tokenizer biases in evictAll()/evict()"
```

---

### Task 3: Replace inline bias computation at both call sites with the cached fetch

**Files:**
- Modify: `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (tool-calling block ~:1096-1112; guided-JSON block ~:1246-1272)

**Interfaces:**
- Consumes: `MLXLanguageModel.makeTokenizerBias(modelID:tokenizer:)` (Task 1). `modelID` and `context.tokenizer` are already in scope at both sites (used for `makeXGTokenizer` at :1077 / :1231).
- The local names `closingBias`, `whitespaceBias`, `whitespaceTokenIDs` are preserved so the downstream `GuidedGenerationLoop.run(...)` calls (:1177-1179, :1301-1303) are untouched. `CompletionReserve.estimate(...)` and the derived `completionReserve`/`hardReserve` scalars stay exactly as-is.

- [ ] **Step 1: Edit the tool-calling block**

Replace the inline computation at lines 1097-1112. Old:

```swift
                                let closingBias = ClosingTokenBias.compute(
                                    tokenizer: context.tokenizer,
                                    eosTokenId: context.tokenizer.eosTokenId
                                )
                                let structuralReserve = CompletionReserve.estimate(
                                    schemaJSON: toolCallingEnvelopeJSON,
                                    tokenizer: context.tokenizer
                                )
                                let completionReserve = Swift.max(
                                    structuralReserve * 3, maxTokens / 4)
                                let hardReserve = structuralReserve * 8

                                let (whitespaceBias, whitespaceTokenIDs) =
                                    WhitespaceTokenBias.compute(
                                        tokenizer: context.tokenizer
                                    )
```

New:

```swift
                                let bias = await MLXLanguageModel.makeTokenizerBias(
                                    modelID: modelID,
                                    tokenizer: context.tokenizer
                                )
                                let closingBias = bias.closing
                                let structuralReserve = CompletionReserve.estimate(
                                    schemaJSON: toolCallingEnvelopeJSON,
                                    tokenizer: context.tokenizer
                                )
                                let completionReserve = Swift.max(
                                    structuralReserve * 3, maxTokens / 4)
                                let hardReserve = structuralReserve * 8

                                let whitespaceBias = bias.whitespace
                                let whitespaceTokenIDs = bias.whitespaceTokenIDs
```

- [ ] **Step 2: Edit the guided-JSON block**

Replace the inline computation at lines 1247-1272. Old:

```swift
                                let closingBias = ClosingTokenBias.compute(
                                    tokenizer: context.tokenizer,
                                    eosTokenId: context.tokenizer.eosTokenId
                                )
                                let structuralReserve = CompletionReserve.estimate(
                                    schemaJSON: schemaJSON,
                                    tokenizer: context.tokenizer
                                )
                                // The structural reserve is the bare minimum tokens for
                                // JSON skeleton (empty strings). Use the larger of 3x
                                // structural minimum or 25% of maxTokens, so closing
                                // bias activates early enough for the model to generate
                                // actual content in closing fields.
                                let completionReserve = Swift.max(
                                    structuralReserve * 3, maxTokens / 4)
                                // Hard reserve: the point at which we force structural
                                // completion by penalizing non-closing tokens. Must be
                                // larger than the raw estimate because grammar-forced
                                // key names (FF tokens) and model-inserted whitespace
                                // cost more tokens than the compact minimal JSON string.
                                let hardReserve = structuralReserve * 8

                                let (whitespaceBias, whitespaceTokenIDs) =
                                    WhitespaceTokenBias.compute(
                                        tokenizer: context.tokenizer
                                    )
```

New (preserve the explanatory comments on the reserve scalars):

```swift
                                let bias = await MLXLanguageModel.makeTokenizerBias(
                                    modelID: modelID,
                                    tokenizer: context.tokenizer
                                )
                                let closingBias = bias.closing
                                let structuralReserve = CompletionReserve.estimate(
                                    schemaJSON: schemaJSON,
                                    tokenizer: context.tokenizer
                                )
                                // The structural reserve is the bare minimum tokens for
                                // JSON skeleton (empty strings). Use the larger of 3x
                                // structural minimum or 25% of maxTokens, so closing
                                // bias activates early enough for the model to generate
                                // actual content in closing fields.
                                let completionReserve = Swift.max(
                                    structuralReserve * 3, maxTokens / 4)
                                // Hard reserve: the point at which we force structural
                                // completion by penalizing non-closing tokens. Must be
                                // larger than the raw estimate because grammar-forced
                                // key names (FF tokens) and model-inserted whitespace
                                // cost more tokens than the compact minimal JSON string.
                                let hardReserve = structuralReserve * 8

                                let whitespaceBias = bias.whitespace
                                let whitespaceTokenIDs = bias.whitespaceTokenIDs
```

- [ ] **Step 3: Confirm the bias enums are no longer referenced inline in the adapter**

Run:
```bash
grep -n "ClosingTokenBias.compute\|WhitespaceTokenBias.compute" Libraries/MLXFoundationModels/MLXLanguageModel.swift
```
Expected: only the two references inside the `ModelCache.makeTokenizerBias` accessor (Task 1) remain; zero references in the `respond()` call sites.

- [ ] **Step 4: Build and run the host-green FM + guided-gen suites**

Run:
```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXFoundationModelsTests \
  -only-testing:MLXGuidedGenerationTests \
  2>&1 | tee /tmp/build-biascache-t3.log | tail -8
```
Expected: `** TEST SUCCEEDED **`. Verify no masked failures: `grep -E "error:|failed|Recorded an issue" /tmp/build-biascache-t3.log` returns nothing; `grep -E "Executed .* tests" /tmp/build-biascache-t3.log` shows a non-zero count for both targets. This is a behavior-preserving refactor — the cached arrays are the same values the enums produced (already proven by `ClosingTokenBiasTests`/`WhitespaceTokenBiasTests` + the Task 1/2 cache tests), so the existing suites passing is the regression gate.

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXFoundationModels/MLXLanguageModel.swift
git commit -m "refactor(mlx-fm): fetch cached tokenizer biases instead of recomputing per response"
```

- [ ] **Step 6: Run the real-model integration suites on host**

The guided-generation and tool-calling **integration** tests in `IntegrationTesting/` load real models and run MLX/Metal inference — they are the end-to-end proof that streaming output is byte-identical after the swap. They run on **this Mac** (Apple Silicon + Metal, macOS 27), same `platform=macOS` destination — no device. First inference is slow (~120s: HF download + Metal shader JIT), so allow a generous timeout and `caffeinate` to avoid screen-sleep throttling.

```bash
caffeinate -dimsu xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/MLXFoundationModelsIntegration \
  2>&1 | tee /tmp/build-biascache-t3-integration.log | tail -10
```
Expected: `** TEST SUCCEEDED **`; `grep -E "error:|failed|Recorded an issue" /tmp/build-biascache-t3-integration.log` returns nothing; `grep -E "Executed .* tests" /tmp/build-biascache-t3-integration.log` shows a non-zero count. (Confirm the exact `-only-testing` suite identifier against `xcodebuild -showTestPlans` / the IntegrationTesting layout if it differs.) Treat a guided-gen/tool-calling regression here as a real failure to investigate, not a flake — but rule out the screen-sleep throttling confound first.

---

## Self-Review

**1. Spec coverage:**
- §1/§2 problem & scope → Task 3 swaps both sites; `CompletionReserve` left inline (Task 3 explicitly preserves it). ✓
- §3 `TokenizerBias` + slice + accessor + static wrapper → Task 1. ✓
- §4 eviction integration (both paths) → Task 2. ✓
- §5 enums untouched / no fusion → no task modifies them; Task 3 Step 3 asserts they're only referenced in the accessor. ✓
- §6 testing (correctness already covered; cache-hit; eviction) → Task 1 (cache-hit, per-model) + Task 2 (evictAll, evict). Correctness not duplicated. ✓
- §7 Sendability resolved → `@unchecked Sendable` class in Task 1 Step 3, justified by precedent. ✓
- §8 reviewer outcome → not a code task (David reply drafted at hand-off).

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; every command has expected output. The only "verify in environment" item is Step 6 (device run), which is a genuine host limitation, explicitly flagged, not a placeholder.

**3. Type consistency:** `makeTokenizerBias(modelID:tokenizer:)` signature identical across actor method (Task 1.5), static wrapper (Task 1.6), and all call sites (Tasks 1/2 tests, Task 3). `TokenizerBias` fields `closing`/`whitespace`/`whitespaceTokenIDs` consistent between the type (1.3), the accessor (1.5), and the call-site reads (3.1/3.2). `CountingTokenizer.idLookupCount` consistent across all tests. ✓
