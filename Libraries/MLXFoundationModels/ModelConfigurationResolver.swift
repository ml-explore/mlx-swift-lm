// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// The configuration seam for ``MLXLanguageModel``: adjust the loaded,
        /// already-inferred ``ModelConfiguration`` for a model on a per-call basis.
        ///
        /// `LLMModelFactory._load` fully infers reasoning, tool-call format, and eos
        /// tokens (passing the load-bearing `modelId`) before the adapter runs, so the
        /// configuration handed to ``resolve(_:for:)`` is already complete. A resolver
        /// patches a per-call copy; it does not perform inference.
        ///
        /// - Important: The returned value is consumed as a per-call local. It is
        ///   **never** written back to `context.configuration`, `Executor.Configuration`,
        ///   or any cache, so two instances with the same id but different resolvers
        ///   cannot cross-contaminate through the shared container. The adapter consumes
        ///   only `reasoningConfig` from the resolved value. Stop tokens come from the
        ///   loaded `ModelConfiguration` (`extraEOSTokens` / `eosTokenIds`), and
        ///   tool-call format and identity (`id` / `tokenizerSource` / `modelDirectory`)
        ///   are likewise taken from `context.configuration` (read at load time before
        ///   resolution) — so patching `extraEOSTokens`, `eosTokenIds`, `toolCallFormat`,
        ///   or any identity field in a resolver is inert. Carry extra stop tokens in the
        ///   model configuration itself, not via a resolver.
        ///
        /// Composition follows the same `any Protocol`-injected-at-init convention as
        /// ``Downloader`` / ``TokenizerLoader`` in `MLXLMCommon`, with a trivial default
        /// (``DefaultConfigurationResolver``) wired up by ``MLXLanguageModel``'s
        /// convenience init so the common case stays zero-config. The ``default`` static
        /// sugar is provided here for ergonomic call sites.
        public protocol ModelConfigurationResolver: Sendable {
            /// Adjust the loaded, already-inferred configuration for a model.
            ///
            /// Called per ``MLXLanguageModel/Executor/respond(to:model:streamingInto:)``
            /// call, after the weights container is loaded. The returned value is
            /// consumed as a per-call local and never written back to caches.
            func resolve(
                _ configuration: ModelConfiguration,
                for descriptor: ModelDescriptor
            ) -> ModelConfiguration
        }

        /// The zero-config default: returns the already-inferred configuration
        /// unchanged. Wired in by ``MLXLanguageModel``'s convenience init so the
        /// common case (let the factory infer everything) stays zero-config.
        public struct DefaultConfigurationResolver: ModelConfigurationResolver {
            public init() {}

            public func resolve(
                _ configuration: ModelConfiguration,
                for descriptor: ModelDescriptor
            ) -> ModelConfiguration {
                configuration
            }
        }

        extension ModelConfigurationResolver where Self == DefaultConfigurationResolver {
            /// The zero-config default: returns the configuration unchanged.
            public static var `default`: Self { DefaultConfigurationResolver() }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
