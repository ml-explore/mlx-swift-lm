// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// The loaded-model handle that a ``ModelCustomizer`` sees: model identity,
        /// the raw `config.json` data, and the tokenizer.
        ///
        /// The shape is wide because ``ModelProfile/inferred(for:)`` needs `configData`
        /// (Llama 3 tool-call detection inspects `vocab_size`/`rope_scaling`) and
        /// custom customizers may need the tokenizer to translate stop-token strings
        /// to ids or inspect chat-template internals. These fields are inputs to a
        /// public protocol method, so narrowing them later would be a breaking change.
        public struct LoadedModelContext: Sendable {

            /// The `model_type` value read from `config.json`.
            public let modelType: String

            /// The Hugging Face repo id (e.g. `mlx-community/Qwen3-4B-4bit`).
            public let modelId: String

            /// The raw `config.json` contents, or `nil` when unavailable. Inference and
            /// customizers can inspect secondary signals (e.g. `vocab_size`) from this.
            public let configData: Data?

            /// The loaded tokenizer for the model.
            public let tokenizer: any Tokenizer

            public init(
                modelType: String,
                modelId: String,
                configData: Data?,
                tokenizer: any Tokenizer
            ) {
                self.modelType = modelType
                self.modelId = modelId
                self.configData = configData
                self.tokenizer = tokenizer
            }

            /// The inferred baseline profile for this context — the value
            /// ``InferringCustomizer`` returns unchanged, and the value a custom
            /// customizer typically starts from before patching individual fields.
            ///
            /// Implemented as a direct shortcut to ``ModelProfile/inferred(for:)``;
            /// never routes through a ``ModelCustomizer`` (no recursion).
            public var inferred: ModelProfile {
                .inferred(for: self)
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
