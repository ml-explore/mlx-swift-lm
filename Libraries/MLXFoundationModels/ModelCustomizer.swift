// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation

        /// The customization seam for ``MLXLanguageModel``: vend a ``ModelProfile``
        /// for a loaded-model context.
        ///
        /// Composition follows the same convention as ``Downloader`` / ``TokenizerLoader``
        /// in `MLXLMCommon`: behavior is injected as `any Protocol` at init, with a
        /// trivial default conformer (``InferringCustomizer``) wired up by a
        /// convenience init so the common case stays zero-config.
        ///
        /// A custom conformer typically starts from the inferred baseline and patches
        /// individual fields:
        ///
        /// ```swift
        /// struct MyQwen3Customizer: ModelCustomizer {
        ///     func profile(for context: LoadedModelContext) -> ModelProfile {
        ///         var profile = context.inferred
        ///         profile.reasoningConfig?.startDelimiter = "<reason>"
        ///         return profile
        ///     }
        /// }
        /// ```
        public protocol ModelCustomizer: Sendable {
            /// Resolve the model profile to use for the given loaded-model context.
            ///
            /// Called per ``MLXLanguageModel/Executor/respond(to:model:streamingInto:)``
            /// call, after the weights container is loaded; the returned profile is
            /// consumed as a per-call local and never written back to caches.
            func profile(for context: LoadedModelContext) -> ModelProfile
        }

        extension ModelCustomizer where Self == InferringCustomizer {
            /// The zero-config default: return ``ModelProfile/inferred(for:)``
            /// unchanged.
            public static var inferring: Self { InferringCustomizer() }
        }

        /// The default ``ModelCustomizer``: returns ``ModelProfile/inferred(for:)``
        /// unchanged. Wired in by ``MLXLanguageModel``'s convenience init so the
        /// common case (let the framework infer everything) stays zero-config.
        public struct InferringCustomizer: ModelCustomizer {
            public init() {}

            public func profile(for context: LoadedModelContext) -> ModelProfile {
                .inferred(for: context)
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
