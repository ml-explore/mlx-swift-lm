// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// A focused, externally-constructable bundle of per-model behavioral quirks
        /// for the FoundationModels-backed MLX adapter.
        ///
        /// `ModelProfile` is the data half of the customization seam:
        /// per-call resolution lives on ``ModelCustomizer/profile(for:)``, but the
        /// values it returns are this plain value type. A `ModelProfile` carries
        /// reasoning, tool-call format, and extra stop tokens — none of which are
        /// always meaningful on every code path:
        ///
        /// - `reasoningConfig` drives the unconstrained-generation reasoning gate.
        /// - `toolCallFormat` is carried for data-layer parity with the direct
        ///   `MLXLLM` path. It is inert on the FoundationModels adapter today, which
        ///   uses xgrammar grammar-constrained decoding for tool calls rather than the
        ///   `ToolCallFormat` parser; carry-only here.
        /// - `extraEOSTokens` is unioned into the stop-token set per call without
        ///   mutating the cached configuration.
        ///
        /// Inference lives on ``ModelProfile/inferred(for:)`` — the single source of
        /// inference and the baseline a customizer patches from
        /// (`var p = context.inferred; p.reasoningConfig = ...`).
        public struct ModelProfile: Sendable, Equatable {

            /// Reasoning configuration (delimiters + prompt strategy), or `nil` for a
            /// non-reasoning model.
            public var reasoningConfig: ReasoningConfig?

            /// Tool-call format for parser selection on the direct `MLXLLM` path.
            /// Carried for parity; inert on the FoundationModels adapter.
            public var toolCallFormat: ToolCallFormat?

            /// Extra stop tokens to union into the per-call stop-token set. Inferred
            /// profiles return an empty set; customizers supply additions per-model.
            public var extraEOSTokens: Set<String>

            public init(
                reasoningConfig: ReasoningConfig? = nil,
                toolCallFormat: ToolCallFormat? = nil,
                extraEOSTokens: Set<String> = []
            ) {
                self.reasoningConfig = reasoningConfig
                self.toolCallFormat = toolCallFormat
                self.extraEOSTokens = extraEOSTokens
            }

            /// Derive a profile for the given loaded-model context from MLXLMCommon's
            /// shared inference functions. This is the single source of inference and
            /// the baseline a custom ``ModelCustomizer`` starts from.
            ///
            /// `extraEOSTokens` is always empty; the framework does not maintain a
            /// per-family stop-token table. Models that need extra stop tokens supply
            /// them through their own customizer.
            public static func inferred(for context: LoadedModelContext) -> ModelProfile {
                ModelProfile(
                    reasoningConfig: ReasoningConfig.infer(
                        from: context.modelType,
                        modelId: context.modelId,
                        configData: context.configData),
                    toolCallFormat: ToolCallFormat.infer(
                        from: context.modelType, configData: context.configData),
                    extraEOSTokens: [])
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
