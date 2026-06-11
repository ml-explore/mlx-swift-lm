// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    import Foundation
    import Testing
    import FoundationModels

    @testable import MLXFoundationModels
    import MLXLMCommon

    /// Verifies the authoritative-capabilities contract: the adapter stores what
    /// the caller passes, never inferring from the model id. The convenience init
    /// wires in `InferringCustomizer`.
    @Suite("MLXLanguageModel capabilities")
    struct MLXLanguageModelCapabilitiesTests {

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func model(
            id: String,
            capabilities: [LanguageModelCapabilities.Capability],
            customizer: (any ModelCustomizer)? = nil
        ) -> MLXLanguageModel {
            let caps = LanguageModelCapabilities(capabilities: capabilities)
            if let customizer {
                return MLXLanguageModel(
                    modelIdentifier: id,
                    capabilities: caps,
                    customizer: customizer,
                    from: CapabilitiesStubDownloader(),
                    using: CapabilitiesStubTokenizerLoader(),
                    locatedBy: { _ in URL(fileURLWithPath: "/tmp") })
            }
            return MLXLanguageModel(
                modelIdentifier: id,
                capabilities: caps,
                from: CapabilitiesStubDownloader(),
                using: CapabilitiesStubTokenizerLoader(),
                locatedBy: { _ in URL(fileURLWithPath: "/tmp") })
        }

        @Test("Declaring [.reasoning, .toolCalling] reports exactly those, regardless of repo id")
        func declaredCapabilitiesAreVerbatim() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let m = model(
                id: "non-reasoning-looking-id",
                capabilities: [.reasoning, .toolCalling])
            #expect(m.capabilities.contains(.reasoning))
            #expect(m.capabilities.contains(.toolCalling))
            #expect(!m.capabilities.contains(.guidedGeneration))
        }

        @Test("Declaring [] reports no .reasoning even for a Qwen3 id (heuristics not consulted)")
        func emptyCapabilitiesIgnoreQwen3Heuristic() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let m = model(id: "mlx-community/Qwen3-4B-4bit", capabilities: [])
            #expect(!m.capabilities.contains(.reasoning))
            #expect(!m.capabilities.contains(.guidedGeneration))
            #expect(!m.capabilities.contains(.toolCalling))
        }

        @Test("Convenience init (no customizer) stores InferringCustomizer")
        func convenienceInitDefaultsCustomizer() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let m = model(id: "any", capabilities: [])
            #expect(m.customizer is InferringCustomizer)
        }

        @Test("Designated init stores the supplied customizer")
        func designatedInitHoldsExplicitCustomizer() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            struct CustomCustomizer: ModelCustomizer {
                func profile(for context: LoadedModelContext) -> ModelProfile {
                    ModelProfile(extraEOSTokens: ["<|done|>"])
                }
            }
            let m = model(id: "any", capabilities: [], customizer: CustomCustomizer())
            #expect(m.customizer is CustomCustomizer)
        }
    }

    // MARK: - Stubs (no download/load occurs in these tests; we only check stored state)

    private final class CapabilitiesStubDownloader: Downloader, @unchecked Sendable {
        func download(
            id: String,
            revision: String?,
            matching patterns: [String],
            useLatest: Bool,
            progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> URL {
            URL(fileURLWithPath: "/tmp/\(id)")
        }
    }

    private final class CapabilitiesStubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            struct EmptyTokenizer: MLXLMCommon.Tokenizer {
                func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
                func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
                func convertTokenToId(_ token: String) -> Int? { nil }
                func convertIdToToken(_ id: Int) -> String? { nil }
                var bosToken: String? { nil }
                var eosToken: String? { nil }
                var unknownToken: String? { nil }
                func applyChatTemplate(
                    messages: [[String: any Sendable]],
                    tools: [[String: any Sendable]]?,
                    additionalContext: [String: any Sendable]?
                ) throws -> [Int] { [] }
            }
            return EmptyTokenizer()
        }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
