// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon

        /// Internal resolver carrying the known per-model stop-token additions used by
        /// the package's examples and tests.
        ///
        /// This deliberately does not maintain a public family→token table: EOS is not
        /// family-predictable (gemma-2 has none, gemma-3 ships `<end_of_turn>`), and
        /// most coverage already comes from `eos_token_id`. This resolver demonstrates
        /// the supply path without committing the framework to a maintenance burden.
        ///
        /// Internal-only by design — `MLXFoundationModels` test and sample code can
        /// wire it in via the resolver parameter at `MLXLanguageModel.init`. App
        /// developers building their own models should write their own resolver.
        struct DevelopmentConfigurationResolver: ModelConfigurationResolver {

            init() {}

            func resolve(
                _ configuration: ModelConfiguration,
                for descriptor: ModelDescriptor
            ) -> ModelConfiguration {
                var c = configuration
                c.extraEOSTokens.formUnion(
                    Self.knownStopTokens(forModelType: descriptor.modelType))
                return c
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
