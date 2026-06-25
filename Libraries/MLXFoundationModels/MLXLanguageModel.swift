// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    // `_version: 2` gates on the FoundationModels *framework* major version, which
    // is 1.4.x on the macOS/iOS 26 SDK and 2.0.x on 27. The third-party-model
    // surface this adapter uses (`LanguageModel`, `LanguageModelCapabilities`, the
    // generic `LanguageModelSession(model:)` init) only exists on the 27 SDK, so
    // this excludes the whole adapter from older SDKs where those symbols are
    // absent. A plain `canImport(FoundationModels)` is insufficient — the module
    // also ships in 26 — and `@available` cannot help, since it gates runtime
    // availability, not the compile-time presence of a symbol in the SDK.
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import FoundationModels
        import MLXLMCommon
        import MLX
        import os.log
        import MLXGuidedGeneration

        // MARK: - Constraint Cache Kind

        /// Selects which xgrammar constructor a cached template was compiled
        /// with. Used by the constraint cache so a JSON-schema source and a
        /// structural-tag source can never alias even if their text collides.
        enum ConstraintKind {
            case json
            case structuralTag
        }

        // MARK: - Model Cache Actor

        /// Thread-safe model cache using Swift actor isolation.
        /// Prevents race conditions when multiple concurrent requests try to load the model.
        /// Supports caching multiple models by their identifiers.
        private actor ModelCache {
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

            /// Gets a fresh constraint by cloning a cached template, or compiles and caches one first.
            ///
            /// Grammar compilation is expensive (~5-20ms). By caching the compiled template
            /// and cloning it (~0.1ms), repeated requests with the same schema skip recompilation.
            /// When Fork() is unavailable (xgrammar < v0.1.34), the clone attempt fails gracefully
            /// and each request compiles a fresh constraint instead.
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
                if let cloned = try? constraint.clone() {
                    constraintTemplates[cacheKey] = constraint
                    return cloned
                }
                return constraint
            }

            /// Evicts all cached state: model containers, tokenizers, and constraint
            /// templates. No GPU-stream synchronization is required — in-flight callers
            /// retain their own `ModelContainer` and free it via ARC on completion.
            func evictAll() {
                containers.removeAll()
                loadingTasks.removeAll()
                suppressedLoadIDs.removeAll()
                xgTokenizers.removeAll()
                constraintTemplates.removeAll()
                lastErrors.removeAll()
            }

            /// Evicts a single model's state across every per-model cache: its container,
            /// xgrammar tokenizer, all compiled constraint templates, last load error,
            /// the suppressed-download tag, and any in-flight load registration.
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
                lastErrors.removeValue(forKey: modelID)
            }
        }

        // MARK: - MLXLanguageModel

        /// A language model implementation that uses MLX for local inference.
        ///
        /// Conforms to the FoundationModels `LanguageModel` protocol, allowing MLX models
        /// to be used with `LanguageModelSession`.
        ///
        /// Example usage:
        /// ```swift
        /// import MLXFoundationModels
        /// import MLXLMHFAPI      // HubClient (Downloader)
        /// import MLXLMTokenizers // TokenizersLoader
        ///
        /// let cache = HubCache.default
        /// let repoID = Repo.ID(rawValue: "mlx-community/Qwen2.5-3B-Instruct-4bit")!
        /// let model = MLXLanguageModel(
        ///     modelID: repoID.rawValue,
        ///     capabilities: LanguageModelCapabilities(
        ///         capabilities: [.guidedGeneration, .toolCalling]),
        ///     from: HubClient.default,
        ///     using: TokenizersLoader(),
        ///     locatedBy: { id in
        ///         guard let r = Repo.ID(rawValue: id) else { return URL(fileURLWithPath: "/") }
        ///         return cache.snapshotPath(repo: r, kind: .model, revision: "main")
        ///             ?? cache.repoDirectory(repo: r, kind: .model)
        ///     }
        /// )
        /// let session = LanguageModelSession(model: model, tools: [], instructions: nil)
        /// let response = try await session.respond(to: "Hello!")
        /// print(response.content)
        /// ```
        ///
        /// **Factory registration**: this target deliberately does not depend on
        /// `MLXLLM`. Consumers who want LLM inference must import `MLXLLM` (or another
        /// factory provider) in their own target so that
        /// `MLXLLM.TrampolineModelFactory` is linked into the binary; otherwise
        /// `loadModelContainer` fails with `noModelFactoryAvailable`.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        public struct MLXLanguageModel: FoundationModels.LanguageModel, Sendable {

            // MARK: - Model Caching (CRITICAL for performance)

            /// Shared model cache - thread-safe via actor isolation.
            /// Without caching, model loading takes 2-30 seconds per request.
            private static let cache = ModelCache()

            /// The model identifier to load.
            public let modelID: String

            /// Downloader used to fetch model snapshots when the cache misses.
            public let downloader: any Downloader

            /// Tokenizer loader used by `loadModelContainer` to materialize the tokenizer.
            public let tokenizerLoader: any TokenizerLoader

            /// Resolves a model identifier to the on-disk weights URL. Currently
            /// stored on the struct for future use by load paths that bypass the
            /// downloader; the standard cache miss path uses
            /// `loadModelContainer(from:using:configuration:)` which discovers
            /// weights itself via the model factory.
            public let weightsLocation: @Sendable (String) -> URL

            /// Loads the model container for the specified model, returning a cached
            /// instance when one exists.
            ///
            /// The first call downloads the model and loads it into memory; subsequent
            /// calls return the cached container immediately. Concurrent callers for
            /// the same model share a single loading task, so a model is never loaded
            /// twice.
            ///
            /// Returns the lower-level `ModelContainer`, which is not part of the
            /// adapter's public surface. Consumers load a model ahead of use through
            /// the public ``preload()`` or `session.prewarm()`.
            ///
            /// - Parameter suppressDownloadingState: When `true`, an in-flight load of
            ///   a model already present on disk is kept out of the `.downloading`
            ///   availability signal. This is an availability-state-machine detail: a
            ///   load that triggers a genuine fetch for a model not yet on disk still
            ///   reports `.downloading` regardless of this flag. Defaults to `false`,
            ///   the behavior for a normal consumer-driven load.
            static func loadContainer(
                modelID: String,
                from downloader: any Downloader,
                using tokenizerLoader: any TokenizerLoader,
                suppressDownloadingState: Bool = false
            ) async throws -> ModelContainer {
                try await cache.load(
                    modelID: modelID,
                    suppressDownloadingState: suppressDownloadingState,
                    loader: makeContainerLoader(
                        modelID: modelID, from: downloader, using: tokenizerLoader)
                )
            }

            /// Builds the `@Sendable` loader closure that
            /// ``loadContainer(modelID:from:using:suppressDownloadingState:)`` hands to
            /// the cache: sets the MLX buffer-reuse pool limit, loads the container via
            /// the model factory, and reports download progress.
            private static func makeContainerLoader(
                modelID: String,
                from downloader: any Downloader,
                using tokenizerLoader: any TokenizerLoader
            ) -> @Sendable () async throws -> ModelContainer {
                {
                    // MLX buffer-reuse pool. Higher = less allocator thrash (fewer
                    // Metal malloc/free round-trips through IOGPU) at the cost of
                    // slightly higher resident GPU memory. 256MB comfortably holds
                    // activations and KV cache for a 3B-parameter model without
                    // forcing pool evictions mid-forward-pass. Well under iOS's
                    // per-app jetsam ceiling on current-generation devices.
                    //
                    // NOTE: this is a process-global setting called on every model
                    // load. Should move to once-per-process init and/or a
                    // configurable surface so consumers can tune for their own footprint.
                    MLX.Memory.cacheLimit = 256 * 1024 * 1024
                    let container = try await loadModelContainer(
                        from: downloader,
                        using: tokenizerLoader,
                        configuration: .init(id: modelID)
                    ) { progress in
                        MLXDownloadProgress.report(progress: progress, modelID: modelID)
                    }
                    MLXDownloadProgress.reportCompleted()
                    return container
                }
            }

            /// Gets or creates a cached GrammarTokenizer for the given model.
            static func makeXGTokenizer(
                modelID: String,
                tokenizer: any Tokenizer
            ) async throws -> GrammarTokenizer {
                try await cache.makeXGTokenizer(modelID: modelID, tokenizer: tokenizer)
            }

            /// Gets a constraint by cloning a cached compiled template (or compiling one first).
            static func makeConstraint(
                modelID: String,
                kind: ConstraintKind,
                source: String,
                tokenizer: GrammarTokenizer,
                hostTokenizer: any Tokenizer,
                fastForward: Bool
            ) async throws -> GrammarConstraint {
                try await cache.makeConstraint(
                    modelID: modelID,
                    kind: kind,
                    source: source,
                    tokenizer: tokenizer,
                    hostTokenizer: hostTokenizer,
                    fastForward: fastForward
                )
            }

            /// Evicts every cached model, tokenizer, and constraint template, freeing
            /// the GPU memory held by model weights. Subsequent requests reload from the
            /// on-disk cache.
            ///
            /// Safe to call during in-flight `respond()`/`warmUp()` work: each holds its
            /// own strong reference to the `ModelContainer` and synchronizes the GPU on
            /// exit, so dropping the cache's reference cannot free weights out from under
            /// a live kernel — the weights free via ARC once that work returns.
            public static func evictAll() async {
                await cache.evictAll()
            }

            /// Drops this model from the shared cache, freeing the GPU memory held by its
            /// weights. A subsequent `respond()`/`preload()` triggers a fresh load
            /// (reusing the on-disk snapshot if the model was previously downloaded).
            ///
            /// Safe to call during an in-flight `respond()`: that call retains its own
            /// `ModelContainer` and finishes normally; the weights free via ARC once it
            /// returns. Evicting a model whose load is still in flight removes it cleanly
            /// — the in-flight load completes but does not re-populate the cache.
            public func evict() async {
                await Self.cache.remove(modelID: modelID)
            }

            /// Whether the shared cache has a *genuine download* in flight for the
            /// given model — excludes a warmup of an already-present model. Used by
            /// ``availability`` to surface a `.downloading` state.
            static func isDownloadingInCache(modelID: String) async -> Bool {
                await cache.isDownloading(modelID: modelID)
            }

            /// The most recent load error for the given model, if any. Cleared on a
            /// subsequent successful load. Used by ``availability`` to surface a
            /// `.downloadFailed` state after a failed ``preload()``.
            static func lastLoadErrorInCache(modelID: String) async -> (any Error)? {
                await cache.lastError(modelID: modelID)
            }

            // MARK: - LanguageModel Conformance

            /// MLX supports guided generation via xgrammar grammar-constrained
            /// decoding (provided by the MLXGuidedGeneration library), tool
            /// calling via the synthetic-final-answer envelope, and reasoning
            /// (chain-of-thought) routing on the unconstrained generation path.
            ///
            /// Capabilities are declared explicitly by the caller at ``init(modelID:capabilities:configurationResolver:from:using:locatedBy:)``
            /// and stored verbatim. The caller includes
            /// `.guidedGeneration`/`.toolCalling`/`.reasoning` as appropriate; the
            /// adapter does not consult ``ReasoningHeuristics`` (which remains a
            /// standalone helper a caller may use to compute their own capability set).
            ///
            /// Declaring `.reasoning` matters for request routing: the framework only
            /// forwards a `reasoningLevel` to executors that declare `.reasoning`, and
            /// auto-rejects one otherwise (on the developer's behalf) before `respond`
            /// runs. The executor in turn emits `.reasoning` events only when this
            /// capability was declared.
            public let capabilities: LanguageModelCapabilities

            /// The configuration resolver that patches a per-call ``ModelConfiguration``
            /// for this instance. Defaults to ``DefaultConfigurationResolver`` via the
            /// convenience init.
            public let configurationResolver: any ModelConfigurationResolver

            /// Configuration the framework uses to create and cache executors.
            public var executorConfiguration: Executor.Configuration {
                Executor.Configuration(modelID: modelID)
            }

            // MARK: - Initialization

            /// Creates an MLXLanguageModel instance with explicitly declared
            /// capabilities and a configuration resolver.
            ///
            /// Model loading is deferred until first inference or preload.
            ///
            /// - Parameters:
            ///   - modelID: The model identifier (e.g., "mlx-community/Qwen3-4B-4bit").
            ///   - capabilities: The capabilities this model supports
            ///     (`.guidedGeneration`, `.toolCalling`, `.reasoning`). Declared
            ///     verbatim; the adapter does not infer or expand the set.
            ///   - configurationResolver: The ``ModelConfigurationResolver`` that
            ///     patches a per-call ``ModelConfiguration`` (reasoning config, extra
            ///     stop tokens) for this instance.
            ///   - downloader: The ``Downloader`` used to fetch model snapshots.
            ///   - tokenizerLoader: The ``TokenizerLoader`` used to materialize the tokenizer.
            ///   - weightsLocation: Resolves a model identifier to the on-disk weights URL.
            public init(
                modelID: String,
                capabilities: LanguageModelCapabilities,
                configurationResolver: any ModelConfigurationResolver,
                from downloader: any Downloader,
                using tokenizerLoader: any TokenizerLoader,
                locatedBy weightsLocation: @Sendable @escaping (String) -> URL
            ) {
                self.modelID = modelID
                self.capabilities = capabilities
                self.configurationResolver = configurationResolver
                self.downloader = downloader
                self.tokenizerLoader = tokenizerLoader
                self.weightsLocation = weightsLocation
            }

            /// Convenience init that defaults the configuration resolver to
            /// ``DefaultConfigurationResolver`` — the zero-config path where the
            /// factory-inferred ``ModelConfiguration`` drives all per-model behavior.
            public init(
                modelID: String,
                capabilities: LanguageModelCapabilities,
                from downloader: any Downloader,
                using tokenizerLoader: any TokenizerLoader,
                locatedBy weightsLocation: @Sendable @escaping (String) -> URL
            ) {
                self.init(
                    modelID: modelID,
                    capabilities: capabilities,
                    configurationResolver: DefaultConfigurationResolver(),
                    from: downloader,
                    using: tokenizerLoader,
                    locatedBy: weightsLocation)
            }

            /// Downloads the model and loads its weights into memory.
            ///
            /// This is a weights-only load: it runs no forward pass, compiles no Metal
            /// shaders, and performs no GPU work, so the first generation request after
            /// `preload()` still pays the one-time Metal shader JIT cost. The call is
            /// awaitable and fully caller-owned — you decide when it runs and handle
            /// any error it throws.
            ///
            /// Call it early, for example when a view appears, to move the
            /// download-and-load portion of cold-start latency off the first
            /// generation request.
            ///
            /// Safe to call multiple times; once the model is loaded, subsequent calls
            /// return immediately from cache.
            public func preload() async throws {
                _ = try await Self.loadContainer(
                    modelID: modelID,
                    from: downloader,
                    using: tokenizerLoader
                )
            }

            /// Loads the model weights and compiles Metal shaders, so the first
            /// `respond()` afterward pays no (or materially reduced) cold-start
            /// shader-JIT cost.
            ///
            /// Metal kernels JIT-compile lazily on the first *synchronous* readback
            /// (`.item()` inside the generate loop) — scheduling work with `asyncEval`
            /// alone does not compile them — so this runs a minimal throwaway forward
            /// pass to force compilation ahead of a real request.
            ///
            /// The forward pass and its single `Stream.gpu.synchronize()` run inside
            /// `container.perform { }`, the same `SerialAccessContainer` lock the
            /// `respond` path holds for its entire generation, so a warmup cannot race
            /// a concurrent `respond` on the process-global `Stream.gpu`. The 1-token
            /// generate ends naturally and is consumed to completion — never cancelled
            /// mid-flight — honoring the Metal teardown invariant (`docs/solutions/002`,
            /// `004`).
            ///
            /// Internal by design: it touches process-global Metal and is driven
            /// fire-and-forget by ``Executor/prewarm(model:transcript:)``, reached
            /// publicly through `session.prewarm()`. Safe to call multiple times and
            /// concurrently; subsequent calls reuse the cached container.
            func warmUp() async throws {
                // Distinguish a warmup of an already-present model (suppress the
                // spurious `.available → .downloading → .available` flip) from a
                // genuine first fetch (which still reports `.downloading`). Conditioning
                // on disk-presence — not "is a warmup" alone — is what makes the
                // loadingTasks-dedup collision benign (see `ModelCache.load`) and keeps
                // the partial-download guard intact: we suppress the in-flight
                // `.downloading` signal rather than reorder the availability checks
                // (reordering would let a partial download with only `config.json`
                // present falsely report `.available`).
                let alreadyOnDisk = modelExistsOnDisk()
                let container = try await Self.loadContainer(
                    modelID: modelID,
                    from: downloader,
                    using: tokenizerLoader,
                    suppressDownloadingState: alreadyOnDisk
                )

                // Pre-create the model-keyed GrammarTokenizer so a guided / tool-calling
                // consumer skips the expensive vocab-extraction step on first
                // respond(). It's keyed on modelID alone — the same cache entry
                // respond()'s guided path reads — so this is a genuine cache hit.
                //
                // CPU-only (xgrammar is C++; no Stream.gpu, no Metal), so it adds no
                // GPU-teardown-race exposure: the safe half of warmup. It runs *after*
                // loadContainer because it needs the live Tokenizer from the container,
                // and *before* the forward pass below so the GPU-touching work stays a
                // single contiguous, serialized block.
                //
                // We deliberately do NOT pre-build a constraint template here:
                // makeConstraint is keyed on modelID:kind:source, where `source` is the
                // per-request schema/tool grammar that prewarm doesn't possess — a
                // pre-built constraint would land under a key no real respond() reads.
                let tokenizer = await container.tokenizer
                _ = try await Self.makeXGTokenizer(
                    modelID: modelID, tokenizer: tokenizer)

                // Force Metal shader JIT with a minimal 1-token generate, run inside
                // `perform` so the forward pass + synchronize serialize against any
                // concurrent `respond`. `maxTokens: 1` makes the stream end on
                // its own; we consume it fully (no early break) so generation runs to
                // completion and leaves no dangling GPU work to race the teardown sync.
                try await container.perform { context in
                    // Exactly one synchronize on every exit path (success or throw),
                    // per the Metal teardown invariant. `prepare` is CPU-only, so on a
                    // pre-forward-pass throw this just synchronizes an idle stream.
                    defer { Stream.gpu.synchronize() }
                    let input = try await context.processor.prepare(
                        input: UserInput(chat: [.user("warmup")]))
                    let params = GenerateParameters(maxTokens: 1)
                    for await _ in try MLXLMCommon.generate(
                        input: input, parameters: params, context: context
                    ) {
                        // Drain to completion.
                    }
                }
            }

            // MARK: - Executor

            /// Executes inference requests for the model.
            public struct Executor: LanguageModelExecutor, Sendable {

                /// Default `maxTokens` when the caller doesn't set
                /// `GenerationOptions.maximumResponseTokens`. Applied uniformly
                /// across guided-JSON, tool-calling, and unconstrained generation
                /// paths so all three share a single definition.
                ///
                /// The guided paths *require* a budget to activate the zone-based
                /// closing bias in `GuidedGenerationLoop` -- without it, open-source
                /// models tend to wander in JSON whitespace before reaching
                /// structural close. 4096 is generous for typical tool calls and
                /// structured outputs. Consumers can override via
                /// `GenerationOptions(maximumResponseTokens:)`.
                private static let defaultMaxTokens = 4096

                /// Map FoundationModels' optional `Double` `GenerationOptions.temperature`
                /// to MLXLMCommon's `Float` `GenerateParameters.temperature`, clamping
                /// negatives to 0.
                ///
                /// - Returns: `nil` when the caller did not request a specific
                ///   temperature, leaving `GenerateParameters`' built-in default in
                ///   place. Otherwise the clamped `Float`.
                ///
                /// Negative sampling temperatures land in `CategoricalSampler` and
                /// produce inverted distributions; we clamp at 0 so the worst the
                /// caller can get is greedy. `0` itself is honored unchanged because
                /// MLXLMCommon's `GenerateParameters.sampler()` routes
                /// `temperature == 0` to `ArgMaxSampler` (greedy) -- no division-by-
                /// zero hazard.
                static func clampedTemperature(_ value: Double?) -> Float? {
                    guard let value else { return nil }
                    return Float(max(0, value))
                }

                /// Translate FoundationModels' `GenerationOptions.SamplingMode` into the
                /// backend-local `MLXSamplingMode`, dropping the best-effort `seed`
                /// (MLX's samplers expose no seed-injection hook). No mode set (`nil`)
                /// and any future/unknown `Kind` both map to `nil` -- "use the provider
                /// default" -- so an unrecognized case never traps and never reaches the
                /// resolver. All value policy lives in `resolveSamplingParameters`; this
                /// shim is a pure 1:1 case translation.
                static func samplingMode(
                    from samplingMode: GenerationOptions.SamplingMode?
                ) -> MLXSamplingMode? {
                    guard let kind = samplingMode?.kind else { return nil }
                    switch kind {
                    case .greedy:
                        return .greedy
                    case .top(let k, _):
                        return .topK(k)
                    case .nucleus(let threshold, _):
                        return .nucleus(threshold)
                    @unknown default:
                        return nil
                    }
                }

                /// Build the `GenerateParameters` for a generation pass, threading the
                /// caller's temperature and sampling mode through the shared resolver so
                /// every real-sampler path (unconstrained, reasoning, tool-call
                /// reasoning) honors `samplingMode` identically. `maxTokens` is the
                /// already-resolved budget -- callers keep their own default/budget
                /// arithmetic, so this helper owns only temperature + sampling resolution.
                static func makeParameters(
                    maxTokens: Int,
                    requestedTemperature: Double?,
                    samplingMode: MLXSamplingMode?
                ) -> GenerateParameters {
                    var params = GenerateParameters(maxTokens: maxTokens)
                    resolveSamplingParameters(
                        mode: samplingMode,
                        clampedTemperature: clampedTemperature(requestedTemperature)
                    ).apply(to: &params)
                    return params
                }

                /// Map xgrammar errors to typed `LanguageModelError` cases where the
                /// cause is provably the user's input; pass everything else through
                /// unchanged.
                ///
                /// Only `GrammarError.invalidJSONSchema` is mapped: that case fires when
                /// xgrammar's JSON-Schema validator outright rejects the schema text
                /// we synthesized from `GenerationSchema`, which is a problem the
                /// developer can fix (simplify the schema, drop an unsupported
                /// construct). `LanguageModelError.unsupportedGenerationGuide` is the
                /// framework's idiomatic surface for that.
                ///
                /// `constraintCompilationFailed` is deliberately NOT mapped to
                /// `unsupportedGenerationGuide`: its origin is ambiguous (could be
                /// schema-level, could be an internal shim failure), and claiming
                /// user-fault when the cause is actually our infrastructure
                /// misleads developers who pattern-match on typed errors.
                ///
                /// `tokenizerCreationFailed` and `bitmaskRetrievalFailed` are
                /// internal shim failures with no recovery path on the developer's
                /// side -- surfacing them untyped is honest.
                static func mapGrammarError(_ grammarError: GrammarError) -> Error {
                    switch grammarError {
                    case .invalidJSONSchema(let message):
                        return LanguageModelError.unsupportedGenerationGuide(
                            .init(schemaName: nil, debugDescription: message)
                        )
                    default:
                        return grammarError
                    }
                }

                /// Configuration for creating and caching executors.
                public struct Configuration: Hashable, Sendable {
                    /// The model identifier this executor uses for loading and metadata.
                    public let modelID: String
                }

                /// The model identifier this executor uses for loading and metadata.
                let modelID: String

                /// Creates an executor from a configuration.
                public init(configuration: Configuration) throws {
                    self.modelID = configuration.modelID
                }

                /// Logs warmup failures from the fire-and-forget `prewarm` path. A
                /// failed warmup is otherwise invisible (no throw reaches the caller),
                /// so this is the only diagnostic surface for a persistently-failing
                /// prewarm (bad id, network gone, OOM). Note it cannot intercept a
                /// Metal command-buffer assertion abort — that is a process crash, not
                /// a catchable Swift error.
                private static let logger = Logger(
                    subsystem: "com.apple.FoundationModels-MLX", category: "Prewarm")

                /// Prewarms the model: loads weights and pre-compiles Metal shaders so
                /// the first `respond()` pays no cold-start shader-JIT cost.
                ///
                /// This is the protocol witness for `LanguageModelExecutor`'s
                /// `prewarm(model:transcript:)`. The signature must match the
                /// requirement *exactly* — concrete `Transcript`, not a generic
                /// `some Collection<Transcript.Entry>` — otherwise it fails to bind as
                /// the witness and the framework's no-op default silently wins instead.
                /// The session hands us the live model instance, so we route through
                /// its downloader/loader pair.
                ///
                /// Fire-and-forget, mirroring Apple's SLM/PCCLM executors and the
                /// framework's own `session.prewarm()`: the method is synchronous and
                /// non-throwing, so it spawns a detached warmup `Task` and returns
                /// immediately. The `Task` is best-effort — a failure is logged, never
                /// surfaced to or crashed on the caller.
                ///
                /// - Parameters:
                ///   - model: The live model instance to warm.
                ///   - transcript: Accepted per protocol; the shader warmup uses a
                ///     fixed dummy prompt and does not depend on it.
                public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
                    Task {
                        do {
                            try await model.warmUp()
                        } catch {
                            Self.logger.error(
                                "MLX prewarm failed for \(model.modelID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                            )
                        }
                    }
                }

                /// Generates a response for the given request, streaming events into the channel.
                ///
                /// - Parameters:
                ///   - request: The generation request containing transcript, tools, and options
                ///   - model: The model instance for this request
                ///   - channel: The channel to send response events into
                public func respond(
                    to request: LanguageModelExecutorGenerationRequest,
                    model: MLXLanguageModel,
                    streamingInto channel: LanguageModelExecutorGenerationChannel
                ) async throws {
                    var collected = TranscriptConverter.mlxMessages(for: request.transcript)
                    // MLX tokenizer crashes on empty chat input; provide a fallback.
                    if collected.isEmpty {
                        collected = [Chat.Message.user("")]
                    }
                    let messages = collected
                    let container = try await MLXLanguageModel.loadContainer(
                        modelID: model.modelID,
                        from: model.downloader,
                        using: model.tokenizerLoader
                    )

                    // Encode schema to JSON if present
                    let schemaJSON: String?
                    if let schema = request.schema {
                        schemaJSON = try SchemaConverter.encodeToJSON(schema)
                    } else {
                        schemaJSON = nil
                    }

                    let modelID = self.modelID
                    let requestedMaxTokens = request.generationOptions.maximumResponseTokens
                    // Translate the SDK sampling mode once, here where generationOptions
                    // is in scope; thread the bridge-local value down to every
                    // real-sampler path so they honor it identically.
                    let requestedSamplingMode = Self.samplingMode(
                        from: request.generationOptions.samplingMode)
                    // Per SKILL.md: response and tool-calls entries each need a fresh
                    // UUID — they live in separate transcript entries. We preserve the
                    // framework-supplied `request.id` for tracing by stamping it into
                    // the response metadata below, rather than reusing it as an entry id.
                    let entryID = UUID().uuidString
                    let toolCallsEntryID = UUID().uuidString
                    let reasoningEntryID = UUID().uuidString
                    // Captured before the actor hop so the perform closure doesn't
                    // capture `model`. Reasoning is gated strictly on the declared
                    // capability; the resolver-patched configuration supplies the
                    // reasoning config we route on.
                    let declaresReasoning = model.capabilities.contains(.reasoning)
                    let configurationResolver = model.configurationResolver

                    do {
                        // Send metadata first
                        await channel.send(
                            .response(
                                entryID: entryID,
                                action: .updateMetadata([
                                    "modelID": modelID,
                                    "requestID": request.id.uuidString,
                                ])))

                        // Generate tokens inside actor isolation. `messages` carries
                        // non-Sendable `Chat.Message` instances (UserInput.Image and
                        // .Video are not Sendable), so route the array through
                        // perform(nonSendable:_:) which boxes it across the actor hop.
                        try await container.perform(nonSendable: messages) { context, messages in
                            // Render the prompt through the model's UserInputProcessor.
                            let userInput = UserInput(chat: messages)
                            let input = try await context.processor.prepare(input: userInput)

                            // Single-turn tool-calling cap: if the transcript already
                            // contains prior tool-call or tool-output entries, this
                            // is a continuation round from `LanguageModelSession`'s
                            // auto-loop (it executed the tool and re-invoked us with
                            // the result appended). Our `TranscriptConverter` drops
                            // those entries, so re-entering the tool-calling branch
                            // would just make the model emit the same tool call
                            // again -- an infinite loop. Fall through to text
                            // generation so the session terminates cleanly after
                            // one round.
                            //
                            // Multi-turn tool calling -- where the model sees tool
                            // outputs in the transcript and continues with a
                            // data-aware response -- is not supported.
                            let isContinuationAfterToolCall = request.transcript.contains { entry in
                                switch entry {
                                case .instructions, .prompt, .response: return false
                                case .reasoning: return false
                                case .toolCalls, .toolOutput: return true
                                @unknown default: return true
                                }
                            }

                            // Resolve the per-instance configuration. Held strictly as
                            // a local; it never lands in context.configuration or
                            // Executor.Configuration, so two instances with the same id
                            // but different resolvers don't cross-contaminate through
                            // the shared caches. Identity is read from
                            // context.configuration (above, at load time) and never
                            // from `resolved`.
                            let configData = try? Data(
                                contentsOf:
                                    context.configuration.modelDirectory
                                    .appendingPathComponent("config.json"))
                            let modelType =
                                configData.flatMap {
                                    try? JSONDecoder.json5().decode(
                                        BaseConfiguration.self, from: $0
                                    ).modelType
                                } ?? ""
                            let descriptor = ModelDescriptor(
                                modelType: modelType,
                                modelId: modelID,
                                configData: configData,
                                tokenizer: context.tokenizer)
                            let resolved = configurationResolver.resolve(
                                context.configuration, for: descriptor)

                            // Capability gate. When the caller omits `.reasoning`
                            // but the resolved configuration carries a reasoning
                            // config, the model must not be allowed to think:
                            //
                            // - Toggleable strategies (`.templateFlag`) re-render the
                            //   prompt with thinking off (handled below per path).
                            // - Non-suppressible strategies (`.alwaysOn`) raise
                            //   `unsupportedCapability` BEFORE generation, regardless
                            //   of which path (tools / schema / unconstrained) the
                            //   request would otherwise take. The throw is
                            //   path-independent so a tool-calling or schema-guided
                            //   request on a model that always reasons surfaces the
                            //   same typed error the unconstrained path does, never a
                            //   silent leak through the grammar's malformed-output
                            //   fallback.
                            if !declaresReasoning, let suppressionConfig = resolved.reasoningConfig
                            {
                                do {
                                    _ = try suppressionConfig.promptStrategy
                                        .additionalContext(forThinkingEnabled: false)
                                } catch ReasoningError.cannotDisableReasoning {
                                    throw LanguageModelError.unsupportedCapability(
                                        LanguageModelError.UnsupportedCapability(
                                            capability: .reasoning,
                                            debugDescription:
                                                "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."
                                        ))
                                }
                            }

                            // Reasoning is only consumed by the unconstrained path
                            // (no tools, no schema). On the guided/tool paths the
                            // grammar already constrains output, so suppression-prep
                            // would be wasted work.
                            let mayRunReasoningPath =
                                (request.enabledToolDefinitions.isEmpty
                                    || isContinuationAfterToolCall)
                                && request.schema == nil

                            // When .reasoning is OMITTED on the unconstrained path,
                            // re-render the prompt with thinking off so the model
                            // doesn't emit `<think>`. Toggleable-only;
                            // .alwaysOn was already rejected above.
                            let suppressedInput: LMInput?
                            if mayRunReasoningPath, !declaresReasoning,
                                let suppressionConfig = resolved.reasoningConfig
                            {
                                suppressedInput = try await Self.preparedInput(
                                    messages: messages, config: suppressionConfig,
                                    thinkingEnabled: false, processor: context.processor,
                                    cannotDisableMessage:
                                        "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."
                                )
                            } else {
                                suppressedInput = nil
                            }

                            let reasoningSetup:
                                (input: LMInput, config: ReasoningConfig, primedInside: Bool)?
                            if mayRunReasoningPath, declaresReasoning,
                                let reasoningConfig = resolved.reasoningConfig
                            {
                                let thinkingEnabled = Self.thinkingEnabled(
                                    for: request.contextOptions.reasoningLevel)
                                let reasoningInput = try await Self.preparedInput(
                                    messages: messages, config: reasoningConfig,
                                    thinkingEnabled: thinkingEnabled, processor: context.processor,
                                    cannotDisableMessage:
                                        "This model always reasons; reasoning cannot be disabled via reasoningLevel."
                                )
                                reasoningSetup = (
                                    reasoningInput, reasoningConfig,
                                    Self.reasoningPrimedInside(
                                        input: reasoningInput, config: reasoningConfig,
                                        tokenizer: context.tokenizer)
                                )
                            } else {
                                reasoningSetup = nil
                            }

                            // The prompt actually fed into generation: the suppressed
                            // prompt when we're forcing thinking off, otherwise the
                            // baseline `input` rendered above.
                            let effectiveInput = suppressedInput ?? input

                            if !request.enabledToolDefinitions.isEmpty
                                && !isContinuationAfterToolCall
                            {
                                // Tool-calling path. Force the model to emit a JSON
                                // object matching one of the declared tools --
                                // including a synthetic "final answer" tool whose
                                // arguments carry the free-text response. After
                                // generation, parse the output to route to either a
                                // toolCallDelta (real tool) or textDelta (final
                                // answer) event.
                                //
                                // Buffers the full output before emitting; streaming
                                // within the final-answer path (reparse-each-delta) is
                                // not yet implemented.
                                let finalAnswerDef = FinalAnswerTool.makeToolDefinition(
                                    responseSchema: request.schema
                                )
                                let allTools =
                                    Array(request.enabledToolDefinitions) + [finalAnswerDef]

                                // Re-tokenize using the model's native tool-aware chat
                                // template (Qwen/Llama/Phi/Gemma all ship one in their
                                // tokenizer_config.json). This is what teaches the model
                                // *what* tools exist and how to decide between them; the
                                // grammar constraint below only enforces the *shape* of
                                // whatever tool call it emits.
                                let toolSpecs = try ToolCallingConversions.makeToolSpecs(
                                    from: allTools)
                                let tokenizerMessages = DefaultMessageGenerator().generate(
                                    messages: messages)

                                // Think-then-call is gated to the enable_thinking
                                // family (Qwen3/QwQ): their template both renders the tool
                                // block AND honors `enable_thinking`. R1-style `.alwaysOn`
                                // models are tool-blind (template ignores `tools:`), so
                                // they fall through to the single-phase path unchanged;
                                // thinking-disabled requests stay single-phase too.
                                let thinkThenCallConfig: ReasoningConfig? = {
                                    guard declaresReasoning,
                                        let cfg = resolved.reasoningConfig,
                                        case .templateFlag = cfg.promptStrategy,
                                        Self.thinkingEnabled(
                                            for: request.contextOptions.reasoningLevel) != false
                                    else { return nil }
                                    return cfg
                                }()
                                // Thread `enable_thinking` through the tool-aware template
                                // (3-arg form) so the prompt is both tool-aware and
                                // thinking-primed; nil on the single-phase path.
                                let reasoningContext = try thinkThenCallConfig.flatMap {
                                    try $0.promptStrategy.additionalContext(
                                        forThinkingEnabled: Self.thinkingEnabled(
                                            for: request.contextOptions.reasoningLevel))
                                }
                                let toolAwareTokens = try context.tokenizer.applyChatTemplate(
                                    messages: tokenizerMessages,
                                    tools: toolSpecs,
                                    additionalContext: reasoningContext
                                )
                                let toolAwareInput = LMInput(tokens: MLXArray(toolAwareTokens))

                                let toolCallingGrammar =
                                    try SchemaConverter.encodeToolCallingGrammar(
                                        tools: allTools
                                    )
                                // The inner JSON envelope is still needed separately to
                                // seed `CompletionReserve` -- the wrapper tokens
                                // (`<tool_call>`, two `\n`s, `</tool_call>`) are small
                                // and fixed, so padding the reserve with their
                                // tokenized size adds noise rather than accuracy.
                                let toolCallingEnvelopeJSON =
                                    try SchemaConverter.encodeToolCallingEnvelopeJSON(
                                        tools: allTools
                                    )

                                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                                    modelID: modelID,
                                    tokenizer: context.tokenizer
                                )
                                let constraint = try await MLXLanguageModel.makeConstraint(
                                    modelID: modelID,
                                    kind: .structuralTag,
                                    source: toolCallingGrammar,
                                    tokenizer: xgTokenizer,
                                    hostTokenizer: context.tokenizer,
                                    fastForward: true
                                )

                                // Always partition into zones -- the grammar has
                                // wiggle room (JSON whitespace before the outer
                                // `}`, whitespace before `\n</tool_call>`) that
                                // open-source models tend to exploit into infinite
                                // loops when not pushed toward structural close.
                                // Use the caller's budget when set, otherwise the
                                // Executor's default.
                                let maxTokens = requestedMaxTokens ?? Self.defaultMaxTokens
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

                                // PHASE 1 (think-then-call): reason unconstrained until
                                // `</think>`, retaining the token IDs to prefill into the
                                // constrained phase below. Empty on the single-phase path.
                                var reasoningTokenIDs: [Int] = []
                                if let cfg = thinkThenCallConfig {
                                    let primedInside = Self.reasoningPrimedInside(
                                        input: toolAwareInput, config: cfg,
                                        tokenizer: context.tokenizer)
                                    let phase1 = try await runToolCallReasoningPhase(
                                        input: toolAwareInput, config: cfg,
                                        primedInside: primedInside, maxTokens: maxTokens,
                                        requestedTemperature: request.generationOptions
                                            .temperature,
                                        samplingMode: requestedSamplingMode,
                                        reasoningEntryID: reasoningEntryID,
                                        responseEntryID: entryID,
                                        context: context, channel: channel)
                                    reasoningTokenIDs = phase1.tokenIDs
                                    if !phase1.closed {
                                        // Cut off mid-thought (budget exhausted before
                                        // `</think>`). Don't prefill a truncated thought
                                        // into the grammar — signal and finish. Phase 1
                                        // already synchronized the GPU on its way out.
                                        await channel.send(
                                            .response(
                                                entryID: entryID,
                                                action: .updateMetadata([
                                                    "incompleteOutput": true
                                                ])))
                                        return
                                    }
                                }

                                // Phase 2 continues from the model's completed reasoning;
                                // carry the raw IDs (no decode/re-encode) so the grammar
                                // starts from the exact post-`</think>` state.
                                let phase2Input =
                                    reasoningTokenIDs.isEmpty
                                    ? toolAwareInput
                                    : LMInput(
                                        tokens: MLXArray(toolAwareTokens + reasoningTokenIDs))
                                // Shared budget (match the unconstrained path): the
                                // envelope continues under the remaining budget, floored
                                // at the completion reserve so it always has room to close
                                // the tool call.
                                let phase2MaxTokens =
                                    reasoningTokenIDs.isEmpty
                                    ? maxTokens
                                    : Swift.max(
                                        maxTokens - reasoningTokenIDs.count, completionReserve)

                                var outputBuffer = ""
                                var incomplete = false
                                var generatedTokenCount: Int?
                                do {
                                    generatedTokenCount = try GuidedGenerationLoop.run(
                                        input: phase2Input,
                                        context: context,
                                        constraint: constraint,
                                        maxTokens: phase2MaxTokens,
                                        vocabSize: Int(xgTokenizer.vocabSize),
                                        completionReserve: completionReserve,
                                        hardReserve: hardReserve,
                                        closingBias: closingBias,
                                        whitespaceBias: whitespaceBias,
                                        whitespaceTokenIDs: whitespaceTokenIDs,
                                        additionalStopTokens: resolved.extraEOSTokens
                                    ) { text in
                                        outputBuffer += text
                                        return !Task.isCancelled
                                    }
                                } catch GuidedGenerationError.incompleteOutput {
                                    incomplete = true
                                }

                                try await emitToolCallingEvent(
                                    outputBuffer: outputBuffer,
                                    userResponseSchema: request.schema,
                                    entryID: entryID,
                                    toolCallsEntryID: toolCallsEntryID,
                                    channel: channel
                                )

                                if let generatedTokenCount {
                                    // Output total spans both phases (reasoning + envelope);
                                    // the reasoning subset is the Phase-1 token count,
                                    // clamped ≤ total.
                                    let reasoningCount = reasoningTokenIDs.count
                                    let totalOutput = generatedTokenCount + reasoningCount
                                    await channel.send(
                                        .response(
                                            entryID: entryID,
                                            action: .updateUsage(
                                                input: .init(
                                                    totalTokenCount: toolAwareInput.text.tokens
                                                        .size,
                                                    cachedTokenCount: 0
                                                ),
                                                output: .init(
                                                    totalTokenCount: totalOutput,
                                                    reasoningTokenCount: Swift.min(
                                                        reasoningCount, totalOutput)
                                                )
                                            )
                                        ))
                                }

                                if incomplete {
                                    await channel.send(
                                        .response(
                                            entryID: entryID,
                                            action: .updateMetadata(["incompleteOutput": true]))
                                    )
                                }
                            } else if let schemaJSON {
                                // Guided generation: stream text deltas as they arrive.
                                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                                    modelID: modelID,
                                    tokenizer: context.tokenizer
                                )

                                let constraint = try await MLXLanguageModel.makeConstraint(
                                    modelID: modelID,
                                    kind: .json,
                                    source: schemaJSON,
                                    tokenizer: xgTokenizer,
                                    hostTokenizer: context.tokenizer,
                                    fastForward: true
                                )
                                // Bias and reserve computation: only when a token
                                // budget is set. Without a budget, the grammar mask
                                // and model's natural EOS tendency control termination.
                                let maxTokens = requestedMaxTokens ?? Self.defaultMaxTokens
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

                                // GuidedGenerationLoop.run's emit closure is synchronous (for
                                // performance -- it runs inside the tight MLX generation loop).
                                // channel.send is async. Bridge via an AsyncStream + concurrent
                                // forwarder so text deltas stream to the channel in order.
                                let (textStream, textContinuation) = AsyncStream<String>
                                    .makeStream()
                                async let forwarder: Void = {
                                    for await text in textStream {
                                        await channel.send(
                                            .response(
                                                entryID: entryID,
                                                action: .appendText(text, tokenCount: 1)
                                            ))
                                    }
                                }()

                                var incomplete = false
                                var generatedTokenCount: Int?
                                do {
                                    generatedTokenCount = try GuidedGenerationLoop.run(
                                        input: input,
                                        context: context,
                                        constraint: constraint,
                                        maxTokens: maxTokens,
                                        vocabSize: Int(xgTokenizer.vocabSize),
                                        completionReserve: completionReserve,
                                        hardReserve: hardReserve,
                                        closingBias: closingBias,
                                        whitespaceBias: whitespaceBias,
                                        whitespaceTokenIDs: whitespaceTokenIDs,
                                        additionalStopTokens: resolved.extraEOSTokens
                                    ) { text in
                                        textContinuation.yield(text)
                                        return !Task.isCancelled
                                    }
                                } catch GuidedGenerationError.incompleteOutput {
                                    // Grammar exhausted maxTokens before reaching a stop state.
                                    // Text deltas already emitted are best-effort output.
                                    incomplete = true
                                }
                                textContinuation.finish()
                                await forwarder

                                if let generatedTokenCount {
                                    await channel.send(
                                        .response(
                                            entryID: entryID,
                                            action: .updateUsage(
                                                input: .init(
                                                    totalTokenCount: input.text.tokens.size,
                                                    cachedTokenCount: 0
                                                ),
                                                output: .init(
                                                    totalTokenCount: generatedTokenCount,
                                                    reasoningTokenCount: 0
                                                )
                                            )
                                        ))
                                }

                                if incomplete {
                                    await channel.send(
                                        .response(
                                            entryID: entryID,
                                            action: .updateMetadata(["incompleteOutput": true]))
                                    )
                                }
                            } else {
                                try await runTextGeneration(
                                    reasoningSetup: reasoningSetup,
                                    fallbackInput: effectiveInput,
                                    requestedMaxTokens: requestedMaxTokens,
                                    requestedTemperature: request.generationOptions.temperature,
                                    samplingMode: requestedSamplingMode,
                                    additionalStopTokens: resolved.extraEOSTokens,
                                    responseEntryID: entryID,
                                    reasoningEntryID: reasoningEntryID,
                                    context: context,
                                    channel: channel
                                )
                            }

                            Stream.gpu.synchronize()
                        }
                    } catch is CancellationError {
                        // Synchronize GPU before rethrowing to ensure in-flight operations complete.
                        // Without this, process teardown can crash with Metal assertions.
                        Stream.gpu.synchronize()
                        throw CancellationError()
                    } catch {
                        // Synchronize GPU before rethrowing to ensure in-flight operations complete
                        Stream.gpu.synchronize()
                        // Re-map xgrammar errors to typed `LanguageModelError` cases
                        // where the cause is provably user input (see `mapGrammarError`).
                        // Internal-shim failures pass through unchanged.
                        if let grammarError = error as? GrammarError {
                            throw Self.mapGrammarError(grammarError)
                        }
                        throw error
                    }
                }

                /// Unconstrained text generation. Used on the no-tools/no-schema
                /// path when the model has no reasoning config to route through.
                private func runUnconstrained(
                    input: LMInput,
                    requestedMaxTokens: Int?,
                    requestedTemperature: Double?,
                    samplingMode: MLXSamplingMode?,
                    additionalStopTokens: Set<String>,
                    entryID: String,
                    context: ModelContext,
                    channel: LanguageModelExecutorGenerationChannel
                ) async throws {
                    // Use a finite default when the framework doesn't specify a
                    // token limit; there's no grammar to stop the model naturally.
                    let params = Self.makeParameters(
                        maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
                        requestedTemperature: requestedTemperature,
                        samplingMode: samplingMode
                    )

                    for await generation in try generate(
                        input: input,
                        parameters: params,
                        context: context,
                        additionalStopTokens: additionalStopTokens
                    ) {
                        try Task.checkCancellation()
                        switch generation {
                        case .chunk(let text):
                            await channel.send(
                                .response(
                                    entryID: entryID,
                                    action: .appendText(text, tokenCount: 1)
                                ))
                        case .info(let info):
                            // MLX-LM emits one .info event at end-of-generation with
                            // authoritative scalar token counts (`promptTokenCount`
                            // is the prompt; `generationTokenCount` is the
                            // model-generated completion -- see Evaluate.swift's
                            // `GenerateCompletionInfo` definition).
                            await channel.send(
                                .response(
                                    entryID: entryID,
                                    action: .updateUsage(
                                        input: .init(
                                            totalTokenCount: info.promptTokenCount,
                                            cachedTokenCount: 0
                                        ),
                                        output: .init(
                                            totalTokenCount: info.generationTokenCount,
                                            reasoningTokenCount: 0
                                        )
                                    )
                                ))
                        case .toolCall(_):
                            break
                        }
                    }
                }

                /// Dispatches the no-tools/no-schema path: reasoning routing when a
                /// config resolved, otherwise plain unconstrained text.
                private func runTextGeneration(
                    reasoningSetup: (input: LMInput, config: ReasoningConfig, primedInside: Bool)?,
                    fallbackInput: LMInput,
                    requestedMaxTokens: Int?,
                    requestedTemperature: Double?,
                    samplingMode: MLXSamplingMode?,
                    additionalStopTokens: Set<String>,
                    responseEntryID: String,
                    reasoningEntryID: String,
                    context: ModelContext,
                    channel: LanguageModelExecutorGenerationChannel
                ) async throws {
                    if let reasoning = reasoningSetup {
                        try await runReasoning(
                            input: reasoning.input,
                            reasoningConfig: reasoning.config,
                            primedInside: reasoning.primedInside,
                            requestedMaxTokens: requestedMaxTokens,
                            requestedTemperature: requestedTemperature,
                            samplingMode: samplingMode,
                            additionalStopTokens: additionalStopTokens,
                            responseEntryID: responseEntryID,
                            reasoningEntryID: reasoningEntryID,
                            context: context,
                            channel: channel)
                    } else {
                        try await runUnconstrained(
                            input: fallbackInput,
                            requestedMaxTokens: requestedMaxTokens,
                            requestedTemperature: requestedTemperature,
                            samplingMode: samplingMode,
                            additionalStopTokens: additionalStopTokens,
                            entryID: responseEntryID,
                            context: context,
                            channel: channel)
                    }
                }

                /// Reasoning-aware unconstrained generation.
                ///
                /// Routes thinking delimited by the model's reasoning markers to
                /// `.reasoning` events and the rest to `.response`, using a raw
                /// `generateTokens` stream + a self-owned `NaiveStreamingDetokenizer`
                /// (bypassing `ToolCallProcessor`) so the scanner sees clean detokenized
                /// text — no second fragmentation source — and the loop sees real token
                /// IDs for an accurate reasoning token count.
                private func runReasoning(
                    input: LMInput,
                    reasoningConfig: ReasoningConfig,
                    primedInside: Bool,
                    requestedMaxTokens: Int?,
                    requestedTemperature: Double?,
                    samplingMode: MLXSamplingMode?,
                    additionalStopTokens: Set<String>,
                    responseEntryID: String,
                    reasoningEntryID: String,
                    context: ModelContext,
                    channel: LanguageModelExecutorGenerationChannel
                ) async throws {
                    let params = Self.makeParameters(
                        maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
                        requestedTemperature: requestedTemperature,
                        samplingMode: samplingMode
                    )

                    var emitter = ReasoningEventEmitter(
                        config: reasoningConfig, primedInside: primedInside)
                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
                    var reasoningTokenCount = 0
                    var completionInfo: GenerateCompletionInfo?

                    for await generation in try generateTokens(
                        input: input, parameters: params, context: context,
                        additionalStopTokens: additionalStopTokens
                    ) {
                        try Task.checkCancellation()
                        switch generation {
                        case .token(let token):
                            // One `.token` == one real token, so this is a true token
                            // count (not a chunk count). Attribute it to reasoning while
                            // the scanner is inside a thinking span. This generously
                            // counts the closing-delimiter tokens as reasoning (the
                            // emitter only flips state once `process` consumes the full
                            // `</think>`); it remains a true token count and the clamp
                            // below keeps it ≤ total.
                            if emitter.isInsideReasoning {
                                reasoningTokenCount += 1
                            }
                            detokenizer.append(token: token)
                            if let chunk = detokenizer.next() {
                                for segment in emitter.process(chunk) {
                                    await Self.send(
                                        segment, responseEntryID: responseEntryID,
                                        reasoningEntryID: reasoningEntryID, channel: channel)
                                }
                            }
                        case .info(let info):
                            completionInfo = info
                        }
                    }

                    for segment in emitter.finalize() {
                        await Self.send(
                            segment, responseEntryID: responseEntryID,
                            reasoningEntryID: reasoningEntryID, channel: channel)
                    }

                    // If generation ended while still inside a thinking block, the model
                    // was cut off mid-thought (e.g. it exhausted the token budget before
                    // emitting `</think>`). Signal it so a consumer doesn't mistake an
                    // empty or partial answer for the model's chosen response — mirrors
                    // the guided path's `incompleteOutput` convention.
                    if emitter.isInsideReasoning {
                        await channel.send(
                            .response(
                                entryID: responseEntryID,
                                action: .updateMetadata(["incompleteOutput": true])))
                    }

                    if let info = completionInfo {
                        // Single source of truth for usage: one authoritative
                        // `.updateUsage` (the framework's aggregator replaces wholesale,
                        // so we must not also rely on per-delta auto-summing). The
                        // reasoning count is clamped to never exceed the total.
                        await channel.send(
                            .response(
                                entryID: responseEntryID,
                                action: .updateUsage(
                                    input: .init(
                                        totalTokenCount: info.promptTokenCount,
                                        cachedTokenCount: 0
                                    ),
                                    output: .init(
                                        totalTokenCount: info.generationTokenCount,
                                        reasoningTokenCount: min(
                                            reasoningTokenCount, info.generationTokenCount)
                                    )
                                )
                            ))
                    }
                }

                /// Routes one scanned segment to the appropriate channel entry.
                private static func send(
                    _ segment: ReasoningEventEmitter.Segment,
                    responseEntryID: String,
                    reasoningEntryID: String,
                    channel: LanguageModelExecutorGenerationChannel
                ) async {
                    switch segment {
                    case .reasoning(let text):
                        await channel.send(
                            .reasoning(
                                entryID: reasoningEntryID,
                                action: .appendText(text, tokenCount: 1)))
                    case .response(let text):
                        await channel.send(
                            .response(
                                entryID: responseEntryID,
                                action: .appendText(text, tokenCount: 1)))
                    }
                }

                /// Prepares an `LMInput` for the unconstrained reasoning path with
                /// thinking explicitly on, off, or unspecified. Maps the package-
                /// internal `cannotDisableReasoning` to the framework's
                /// `unsupportedCapability` so always-on models surface a typed error
                /// before generation rather than leaking `<think>` into `.response`.
                private static func preparedInput(
                    messages: [Chat.Message],
                    config: ReasoningConfig,
                    thinkingEnabled: Bool?,
                    processor: any UserInputProcessor,
                    cannotDisableMessage: String
                ) async throws -> LMInput {
                    let additionalContext: [String: any Sendable]?
                    do {
                        additionalContext = try config.promptStrategy
                            .additionalContext(forThinkingEnabled: thinkingEnabled)
                    } catch ReasoningError.cannotDisableReasoning {
                        throw LanguageModelError.unsupportedCapability(
                            LanguageModelError.UnsupportedCapability(
                                capability: .reasoning,
                                debugDescription: cannotDisableMessage))
                    }
                    return try await processor.prepare(
                        input: UserInput(chat: messages, additionalContext: additionalContext))
                }

                /// Maps a requested reasoning level to a thinking on/off/unspecified
                /// flag. `nil` (no opinion) defers to the strategy's default; any
                /// concrete level means "think" (v1 does not modulate depth); only the
                /// package convention `.custom("no_think")` means "off".
                static func thinkingEnabled(for level: ContextOptions.ReasoningLevel?) -> Bool? {
                    guard let level else { return nil }
                    switch level {
                    case .light, .moderate, .deep:
                        return true
                    case .custom(let value):
                        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
                            .lowercased()
                        return normalized == "no_think" ? false : true
                    @unknown default:
                        // A future level we don't recognize → default to thinking on.
                        return true
                    }
                }

                /// Decodes the rendered prompt's tail and asks whether it ends inside an
                /// open reasoning block (some model families prefill the opening
                /// delimiter).
                private static func reasoningPrimedInside(
                    input: LMInput, config: ReasoningConfig, tokenizer: any Tokenizer
                ) -> Bool {
                    let tokens = input.text.tokens.asArray(Int.self)
                    let renderedTail = tokenizer.decode(tokenIds: Array(tokens.suffix(64)))
                    return ReasoningEventEmitter.promptEndsInsideReasoning(
                        renderedPromptTail: renderedTail, config: config)
                }

                /// Think-then-call Phase 1: generate reasoning unconstrained until
                /// the model closes its thinking block, routing reasoning text to
                /// `.reasoning` events and retaining the raw token IDs to prefill into the
                /// constrained Phase 2.
                ///
                /// Uses the `Task`-returning `generateTokensTask` so the GPU loop is
                /// cancelled and drained at the phase boundary — without that, Phase 2's
                /// prefill could overlap Phase 1's in-flight forward pass on the shared
                /// `Stream` and trip a Metal command-buffer assertion.
                ///
                /// Returns the accumulated token IDs and whether `</think>` actually
                /// closed. If it did not (budget exhausted mid-thought), the caller must
                /// skip Phase 2 rather than prefill a truncated thought into the grammar.
                private func runToolCallReasoningPhase(
                    input: LMInput,
                    config: ReasoningConfig,
                    primedInside: Bool,
                    maxTokens: Int,
                    requestedTemperature: Double?,
                    samplingMode: MLXSamplingMode?,
                    reasoningEntryID: String,
                    responseEntryID: String,
                    context: ModelContext,
                    channel: LanguageModelExecutorGenerationChannel
                ) async throws -> (tokenIDs: [Int], closed: Bool) {
                    let params = Self.makeParameters(
                        maxTokens: maxTokens,
                        requestedTemperature: requestedTemperature,
                        samplingMode: samplingMode
                    )
                    var collector = ReasoningTokenCollector(
                        config: config, primedInside: primedInside, tokenizer: context.tokenizer
                    )

                    let (stream, task) = try generateTokensTask(
                        input: input, parameters: params, context: context)
                    var closed = false
                    do {
                        for await generation in stream {
                            try Task.checkCancellation()
                            guard case .token(let token) = generation else { continue }
                            for segment in collector.ingest(token) {
                                await Self.send(
                                    segment, responseEntryID: responseEntryID,
                                    reasoningEntryID: reasoningEntryID, channel: channel)
                            }
                            if collector.shouldStopAfterReasoning {
                                closed = true
                                break
                            }
                        }
                    } catch {
                        // Drain the generation task before propagating, but do NOT sync
                        // here: respond's outer `catch` is the single GPU-sync point for
                        // this exit path. Keep one clean GPU sync per exit path —
                        // cascading syncs across nested catches can race the Metal
                        // command-buffer state during teardown.
                        task.cancel()
                        _ = await task.value
                        throw error
                    }
                    // Drain the generation task before Phase 2 reuses the Stream.
                    task.cancel()
                    _ = await task.value
                    Stream.gpu.synchronize()

                    for segment in collector.finalize() {
                        await Self.send(
                            segment, responseEntryID: responseEntryID,
                            reasoningEntryID: reasoningEntryID, channel: channel)
                    }
                    return (collector.reasoningTokenIDs, closed)
                }

                /// Parses a tool-calling envelope JSON object and emits the
                /// appropriate channel event.
                ///
                /// The output buffer is expected to be a JSON object matching the
                /// shape `{"name": <tool-name>, "arguments": <args>}`. Grammars from
                /// `SchemaConverter.encodeToolCallingGrammar` guarantee either that
                /// shape directly (bare JSON) or that shape wrapped in Qwen's
                /// `<tool_call>\n...\n</tool_call>` special-token delimiters --
                /// `unwrapToolCallMarkers` below strips the wrapper if present. The
                /// best-effort fallback only exists so that unexpected upstream
                /// changes don't silently swallow output.
                ///
                /// - If `name` is the synthetic final-answer tool:
                ///   - With no developer response schema: unwrap `arguments.response`
                ///     into a `.textDelta` event.
                ///   - With a developer response schema: re-serialize `arguments`
                ///     back to JSON text and emit as a single `.textDelta`. The
                ///     session's normal response-parsing path will decode the JSON
                ///     through the developer's `GenerationSchema`.
                /// - If `name` is any real tool: emit a single `.toolCallDelta`
                ///   with the arguments JSON and a freshly minted toolCallID.
                ///
                /// `entryID` and `toolCallsEntryID` must be distinct: SKILL.md requires
                /// `.response` and `.toolCalls` to live in separate transcript entries.
                private func emitToolCallingEvent(
                    outputBuffer: String,
                    userResponseSchema: GenerationSchema?,
                    entryID: String,
                    toolCallsEntryID: String,
                    channel: LanguageModelExecutorGenerationChannel
                ) async throws {
                    let unwrapped = Self.unwrapToolCallMarkers(outputBuffer)
                    let data = Data(unwrapped.utf8)
                    guard
                        let obj = try? JSONSerialization.jsonObject(with: data)
                            as? [String: Any],
                        let name = obj["name"] as? String
                    else {
                        // Malformed output. The grammar should have prevented this;
                        // emit the raw buffer as text so failures surface loudly.
                        await channel.send(
                            .response(
                                entryID: entryID,
                                action: .appendText(outputBuffer, tokenCount: 1)
                            ))
                        return
                    }

                    if name == FinalAnswerTool.toolName {
                        let text: String
                        if userResponseSchema == nil {
                            let args = obj["arguments"] as? [String: Any]
                            text = (args?["response"] as? String) ?? ""
                        } else if let args = obj["arguments"],
                            let argsData = try? JSONSerialization.data(withJSONObject: args),
                            let argsStr = String(data: argsData, encoding: .utf8)
                        {
                            text = argsStr
                        } else {
                            text = ""
                        }
                        await channel.send(
                            .response(
                                entryID: entryID,
                                action: .appendText(text, tokenCount: 1)
                            ))
                    } else {
                        guard
                            let args = obj["arguments"],
                            let argsData = try? JSONSerialization.data(withJSONObject: args),
                            let argsStr = String(data: argsData, encoding: .utf8)
                        else {
                            return
                        }
                        await channel.send(
                            .toolCalls(
                                entryID: toolCallsEntryID,
                                action: .toolCall(
                                    id: UUID().uuidString,
                                    name: name,
                                    action: .appendArguments(argsStr, tokenCount: 1)
                                )
                            ))
                    }
                }

                /// Strips Qwen-style `<tool_call>\n...\n</tool_call>` wrapper markers
                /// if present, returning the inner JSON text. Untouched if the buffer
                /// doesn't start with a wrapper -- the `bare_call` grammar alternative
                /// is valid output and parses directly.
                ///
                /// The inner newlines around the JSON come from the Qwen training
                /// format; we're tolerant of whitespace on either side of the markers
                /// so that tokenizer decoding quirks (extra spaces, missing newlines)
                /// don't cause the JSON parse to fail.
                private static func unwrapToolCallMarkers(_ buffer: String) -> String {
                    let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                    let openMarker = "<tool_call>"
                    let closeMarker = "</tool_call>"
                    guard trimmed.hasPrefix(openMarker) else { return buffer }
                    let afterOpen = trimmed.dropFirst(openMarker.count)
                    let inner: Substring
                    if let closeRange = afterOpen.range(of: closeMarker, options: .backwards) {
                        inner = afterOpen[afterOpen.startIndex ..< closeRange.lowerBound]
                    } else {
                        inner = afterOpen
                    }
                    return inner.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
