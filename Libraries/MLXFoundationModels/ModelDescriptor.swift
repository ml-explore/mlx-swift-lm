// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// Inspection-only inputs a ``ModelConfigurationResolver`` may branch on:
        /// model identity, the raw `config.json` data, and the tokenizer.
        ///
        /// Unlike the configuration it accompanies, a `ModelDescriptor` performs no
        /// inference — the configuration handed to the resolver is already fully
        /// inferred by `LLMModelFactory._load`. The descriptor exists so a resolver
        /// can key per-model adjustments on `model_type`/`modelId` or inspect
        /// secondary `config.json` signals.
        public struct ModelDescriptor: Sendable {

            /// The `model_type` value read from `config.json`.
            public let modelType: String

            /// The Hugging Face repo id (e.g. `mlx-community/Qwen3-4B-4bit`).
            public let modelId: String

            /// The raw `config.json` contents, or `nil` when unavailable.
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
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
