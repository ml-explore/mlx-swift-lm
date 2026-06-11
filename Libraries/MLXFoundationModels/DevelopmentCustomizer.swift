// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// Internal customizer carrying the known per-model stop-token additions used
        /// by the package's examples and tests.
        ///
        /// This deliberately does not maintain a public family→token table:
        /// EOS is not family-predictable (gemma-2 has none, gemma-3 ships
        /// `<end_of_turn>`, gemma-4 ships `<turn|>`), and most coverage already comes
        /// from `eos_token_id`. This customizer demonstrates the supply path without
        /// committing the framework to a maintenance burden.
        ///
        /// Internal-only by design — `MLXFoundationModels` test and sample code can
        /// wire it in via the customizer parameter at `MLXLanguageModel.init`. App
        /// developers building their own models should write their own customizer.
        struct DevelopmentCustomizer: ModelCustomizer {

            init() {}

            func profile(for context: LoadedModelContext) -> ModelProfile {
                var profile = context.inferred
                profile.extraEOSTokens.formUnion(
                    Self.knownStopTokens(forModelType: context.modelType))
                return profile
            }

            /// Known package-test stop tokens by model_type. Adds, does not replace.
            private static func knownStopTokens(forModelType modelType: String) -> Set<String> {
                let type = modelType.lowercased()
                if type.hasPrefix("gemma3") {
                    return ["<end_of_turn>"]
                }
                if type.hasPrefix("phi3") {
                    return ["<|end|>"]
                }
                return []
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
