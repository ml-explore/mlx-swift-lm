// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon

// MARK: - Constraint Cache Kind

/// Selects which xgrammar constructor a cached template was compiled
/// with. Used by the constraint cache so a JSON-schema source and a
/// structural-tag source can never alias even if their text collides.
enum ConstraintKind {
    case json
    case structuralTag
}

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

// MARK: - Model Cache Actor

/// Thread-safe model cache using Swift actor isolation.
/// Prevents race conditions when multiple concurrent requests try to load the model.
/// Supports caching multiple models by their identifiers.
actor ModelCache {
    /// Class wrapper around `Task` so actor-reentrancy supersession guards can
    /// use `===` identity comparison. `Task` is a value type; a wrapper lets us
    /// detect whether `evictAll()` replaced a loading entry mid-flight.
    private final class LoadTask {
        let task: Task<ModelContainer, Error>
        init(_ task: Task<ModelContainer, Error>) { self.task = task }
    }

    private var containers: [String: ModelContainer] = [:]
    private var loadingTasks: [String: LoadTask] = [:]
    /// In-flight loads tagged as a warmup of an already-present model, which
    /// must NOT surface as `.downloading` (there is no user-facing download).
    /// A subset of `loadingTasks`' keys. See `load` and `isDownloading`.
    private var suppressedLoadIDs: Set<String> = []
    private var xgTokenizers: [String: GrammarTokenizer] = [:]
    /// Cached compiled constraint templates keyed by (modelID, schemaJSON).
    /// Clone from template instead of recompiling the grammar each request.
    private var constraintTemplates: [String: GrammarConstraint] = [:]
    /// Cached per-model logit biases (closing + whitespace). Pure functions of
    /// the tokenizer, so computed once per model and reused across requests.
    private var tokenizerBiases: [String: TokenizerBias] = [:]
    /// Most recent load error per model. Cleared on a subsequent successful
    /// load. Surfaced through `MLXLanguageModel.availability` so callers can
    /// distinguish "never tried" from "tried and failed".
    private var lastErrors: [String: any Error] = [:]

    /// Gets the cached model container for the given model ID, loading it if necessary.
    /// Concurrent callers for the same model will share the same loading task, preventing duplicate loads.
    ///
    /// The `loader` closure carries the transport types (downloader, tokenizer
    /// loader). Keeping them out of the cache means the cache itself stays
    /// agnostic of how a container is acquired -- first caller wins; later
    /// callers reuse the cached container regardless of which loader they
    /// brought along.
    func load(
        modelID: String,
        suppressDownloadingState: Bool = false,
        loader: @Sendable @escaping () async throws -> ModelContainer
    ) async throws -> ModelContainer {
        if let cached = containers[modelID] {
            return cached
        }

        if let existingLoadTask = loadingTasks[modelID] {
            // Coalesced onto an in-flight load: the first caller's
            // classification (downloading vs. suppressed) stands — we do not
            // re-tag. This collision is benign because the suppress decision is
            // conditioned on disk-presence: a warmup and a genuine download for
            // a not-yet-present model both classify as downloading, so they
            // agree; when the model IS present, `availability` resolves to
            // `.available` regardless of the in-flight load.
            return try await existingLoadTask.task.value
        }

        let loadTask = LoadTask(
            Task<ModelContainer, Error> {
                try await loader()
            })
        loadingTasks[modelID] = loadTask
        // Tag a warmup-of-an-already-present model out of the `.downloading`
        // signal (computed by the caller as warmup AND modelExistsOnDisk()).
        if suppressDownloadingState {
            suppressedLoadIDs.insert(modelID)
        }

        do {
            let loaded = try await loadTask.task.value
            // Supersession guard: `evict()`/`evictAll()` may have removed this
            // load while it was suspended (actor reentrancy). If we are no longer
            // the registered task, hand the awaiter its container but do NOT
            // re-populate the cache — ARC frees the weights when the awaiter
            // releases it.
            guard loadingTasks[modelID] === loadTask else { return loaded }
            containers[modelID] = loaded
            loadingTasks[modelID] = nil
            suppressedLoadIDs.remove(modelID)
            lastErrors[modelID] = nil
            return loaded
        } catch {
            // Same guard on the failure path: a superseded load must not re-add a
            // stale lastErrors entry for a model nobody holds.
            if loadingTasks[modelID] === loadTask {
                loadingTasks[modelID] = nil
                suppressedLoadIDs.remove(modelID)
                lastErrors[modelID] = error
            }
            throw error
        }
    }

    /// Whether a *genuine download* is in flight for the given model: a load
    /// task is running and it was not tagged as a warmup of an already-present
    /// model. Drives `availability`'s `.downloading` state, so a background
    /// warmup of an already-downloaded model does not spuriously report
    /// `.downloading`. (A warmup that triggers a real fetch is not tagged and
    /// does report here.)
    func isDownloading(modelID: String) -> Bool {
        loadingTasks[modelID] != nil && !suppressedLoadIDs.contains(modelID)
    }

    /// The most recent load error for the given model, if a previous attempt
    /// failed and no successful load has happened since.
    func lastError(modelID: String) -> (any Error)? {
        lastErrors[modelID]
    }

    /// Gets or creates a cached GrammarTokenizer for the given model.
    func makeXGTokenizer(
        modelID: String,
        tokenizer: any Tokenizer
    ) throws -> GrammarTokenizer {
        if let cached = xgTokenizers[modelID] {
            return cached
        }
        let vocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
        let xgTok = try GrammarTokenizer(
            vocab: vocab.vocab,
            vocabType: vocab.vocabType,
            eosTokenId: Int32(tokenizer.eosTokenId ?? 0)
        )
        xgTokenizers[modelID] = xgTok
        return xgTok
    }

    /// Whether an `GrammarTokenizer` is already cached for the given model.
    /// Used by `MLXLanguageModel.hasCachedXGTokenizer` so tests can assert
    /// that `warmUp()` pre-created it (a genuine cache hit) rather than only
    /// that a later guided respond happens to succeed.
    func hasCachedXGTokenizer(modelID: String) -> Bool {
        xgTokenizers[modelID] != nil
    }

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

    /// Gets a fresh constraint by cloning a cached template, or compiles and caches one first.
    ///
    /// Grammar compilation is expensive (~5-20ms). By caching the compiled template
    /// and cloning it (~0.1ms), repeated requests with the same schema skip recompilation.
    /// When Fork() is unavailable (xgrammar < v0.1.34), the clone attempt fails gracefully
    /// and each request compiles a fresh constraint instead. Any other clone failure
    /// reaches the caller.
    func makeConstraint(
        modelID: String,
        kind: ConstraintKind,
        source: String,
        tokenizer: GrammarTokenizer,
        hostTokenizer: any Tokenizer,
        fastForward: Bool
    ) throws -> GrammarConstraint {
        let cacheKey = "\(modelID):\(kind):\(source)"
        if let template = constraintTemplates[cacheKey] {
            do {
                return try template.clone()
            } catch GrammarError.forkFailed {
                constraintTemplates.removeValue(forKey: cacheKey)
            }
        }
        let constraint: GrammarConstraint
        switch kind {
        case .json:
            constraint = try GrammarConstraint(
                tokenizer: tokenizer,
                jsonSchema: source,
                fastForward: fastForward,
                hostTokenizer: hostTokenizer
            )
        case .structuralTag:
            constraint = try GrammarConstraint(
                tokenizer: tokenizer,
                structuralTag: source,
                fastForward: fastForward,
                hostTokenizer: hostTokenizer
            )
        }
        do {
            let clone = try constraint.clone()
            constraintTemplates[cacheKey] = constraint
            return clone
        } catch GrammarError.forkFailed {
            return constraint
        }
    }

    /// Evicts all cached state: model containers, tokenizers, constraint
    /// templates, and per-model tokenizer biases. No GPU-stream synchronization
    /// is required — in-flight callers retain their own `ModelContainer` and
    /// free it via ARC on completion.
    func evictAll() {
        containers.removeAll()
        loadingTasks.removeAll()
        suppressedLoadIDs.removeAll()
        xgTokenizers.removeAll()
        constraintTemplates.removeAll()
        tokenizerBiases.removeAll()
        lastErrors.removeAll()
    }

    /// Evicts a single model's state across every per-model cache: its container,
    /// xgrammar tokenizer, all compiled constraint templates, tokenizer bias,
    /// last load error, the suppressed-download tag, and any in-flight load
    /// registration.
    /// Best-effort cancels an in-flight load (the load path is not
    /// cancellation-aware today, so this is a no-op safety net); the
    /// load-completion guard in `load()` is what prevents a superseded load
    /// from re-populating after removal.
    func remove(modelID: String) {
        // `loadingTasks` holds a `LoadTask` box; cancel the wrapped `Task`.
        loadingTasks[modelID]?.task.cancel()
        loadingTasks.removeValue(forKey: modelID)
        suppressedLoadIDs.remove(modelID)
        containers.removeValue(forKey: modelID)
        xgTokenizers.removeValue(forKey: modelID)
        constraintTemplates = constraintTemplates.filter {
            !$0.key.hasPrefix("\(modelID):")
        }
        tokenizerBiases.removeValue(forKey: modelID)
        lastErrors.removeValue(forKey: modelID)
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
