// Copyright © 2026 Apple Inc.
//
// TraitMatrixTests: symbol-surface + behavioral checks across the orthogonal
// `FoundationModelsIntegration` × `GuidedGenerationSupport` traits.
//
// Each `#if` block below is active for exactly one of the four trait
// combinations. Successfully compiling this file under a given trait set is
// the primary structural assertion — the test bodies reference the symbols
// that must be present in that set.
//
// The two `FoundationModelsIntegration`-on arms additionally require
// `canImport(FoundationModels, _version: 2)`: the adapter surface
// (`MLXLanguageModel` et al.) only exists on the 27 SDK, so on the 26 SDK
// those arms compile to nothing even when the trait is on. The guided-
// generation primitives (`GuidedGenerationLoop`, `XGConstraint`) are gated on
// `GuidedGenerationSupport` alone and are SDK-independent.
//
// A few behavioral tests are gated on the FM-on combinations because they
// rely on `MLXLanguageModel.Executor`. The MLXFoundationModelsTests target
// compiles with the package defaults (both traits on), so only the
// "both on" block runs in the normal test pass.

import Testing

#if GuidedGenerationSupport
    import CXGrammar
    import MLXGuidedGeneration
#endif

#if FoundationModelsIntegration
    @testable import MLXFoundationModels
    import FoundationModels
#else
    @testable import MLXFoundationModels
#endif

@Suite("Trait matrix: FoundationModelsIntegration × GuidedGenerationSupport")
struct TraitMatrixTests {

    // MARK: - Both traits on (default)

    #if FoundationModelsIntegration && canImport(FoundationModels, _version: 2) && GuidedGenerationSupport
        @Test("Both traits on: MLXLanguageModel + guided-generation primitives compile")
        func bothTraitsOnSurface() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            _ = MLXLanguageModel.self
            _ = MLXLanguageModel.Executor.self
            _ = GuidedGenerationLoop.self
            _ = XGConstraint.self
            _ = MLXDownloadProgress.self
        }

        @Test("Both traits on: capabilities stored verbatim from init")
        func capabilitiesStoredVerbatim() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Capabilities are authoritative: the adapter stores what the caller
            // passes, never inferring from the model id.
            let reasoning = makeStubModel(
                "mlx-community/Qwen3-4B-4bit",
                capabilities: LanguageModelCapabilities(capabilities: [
                    .reasoning, .guidedGeneration, .toolCalling,
                ])
            ).capabilities
            #expect(reasoning.contains(.reasoning))
            #expect(reasoning.contains(.guidedGeneration))
            #expect(reasoning.contains(.toolCalling))

            let nonReasoning = makeStubModel(
                TestFixtures.gemmaModelID,
                capabilities: LanguageModelCapabilities(capabilities: [
                    .guidedGeneration, .toolCalling,
                ])
            ).capabilities
            #expect(!nonReasoning.contains(.reasoning))
            #expect(nonReasoning.contains(.guidedGeneration))
        }
    #endif

    // MARK: - FoundationModels on, guided generation off
    //
    // These "throws guidedGenerationDisabled" tests don't require actual model
    // inference — `respond(to:)` checks `request.schema` and
    // `request.enabledTools` before loading weights, so the error surfaces
    // early and stub-backed model construction suffices. The real-inference
    // chat-fallthrough variant lives in the IntegrationTesting xcodeproj
    // (`PlainChatGenerationTests`), since it loads a model.

    #if FoundationModelsIntegration && canImport(FoundationModels, _version: 2) && !GuidedGenerationSupport
        @Test("FM on, GG off: MLXLanguageModel compiles; guidedGenerationDisabled defined")
        func fmOnlySurface() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            _ = MLXLanguageModel.self
            _ = MLXLanguageModelError.guidedGenerationDisabled
            _ = MLXDownloadProgress.self
        }

        @Test("FM on, GG off: caller can declare .reasoning without GG")
        func reasoningCapabilityWithoutGG() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Capabilities are independent of the trait: the unconstrained
            // reasoning path exists under !GuidedGenerationSupport, so a caller
            // may declare `.reasoning` even when GG isn't compiled in. The
            // adapter does not police the set against the trait.
            let reasoning = makeStubModel(
                "mlx-community/Qwen3-4B-4bit",
                capabilities: LanguageModelCapabilities(capabilities: [.reasoning])
            ).capabilities
            #expect(reasoning.contains(.reasoning))
            #expect(!reasoning.contains(.guidedGeneration))
            #expect(!reasoning.contains(.toolCalling))

            let nonReasoning = makeStubModel(
                TestFixtures.gemmaModelID,
                capabilities: LanguageModelCapabilities(capabilities: [])
            ).capabilities
            #expect(!nonReasoning.contains(.reasoning))
            #expect(!nonReasoning.contains(.guidedGeneration))
        }

        @Test("FM on, GG off: respond(to:) with schema throws guidedGenerationDisabled")
        func schemaRequestThrowsWithoutGG() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeStubModel(TestFixtures.gemmaModelID)
            let executor = try makeMLXExecutor(for: model)
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Pick a number."))
                        ], responseFormat: nil))
            ])
            let request = makeExecutorRequest(
                transcript: transcript,
                schema: Int.generationSchema
            )
            let channel = LanguageModelExecutorGenerationChannel()

            await #expect(throws: MLXLanguageModelError.guidedGenerationDisabled) {
                try await executor.respond(to: request, model: model, streamingInto: channel)
            }
        }

        @Test("FM on, GG off: respond(to:) with enabled tools throws guidedGenerationDisabled")
        func toolsRequestThrowsWithoutGG() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeStubModel(TestFixtures.gemmaModelID)
            let executor = try makeMLXExecutor(for: model)
            let tool = Transcript.ToolDefinition(
                name: "noop",
                description: "Does nothing; only needs a schema to exist.",
                parameters: Int.generationSchema
            )
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Call a tool."))
                        ], responseFormat: nil))
            ])
            let request = makeExecutorRequest(
                transcript: transcript,
                enabledTools: [tool]
            )
            let channel = LanguageModelExecutorGenerationChannel()

            await #expect(throws: MLXLanguageModelError.guidedGenerationDisabled) {
                try await executor.respond(to: request, model: model, streamingInto: channel)
            }
        }
    #endif

    // MARK: - FoundationModels off, guided generation on

    #if !FoundationModelsIntegration && GuidedGenerationSupport
        @Test("FM off, GG on: guided-generation primitives compile; MLXLanguageModel absent")
        func ggOnlySurface() {
            _ = GuidedGenerationLoop.self
            _ = XGConstraint.self
            _ = MLXDownloadProgress.self
            // MLXLanguageModel is not a type in this configuration; the fact that
            // this file compiles without referencing it is the assertion.
        }

        @Test("FM off, GG on: XGConstraint compiles a simple JSON schema")
        func xgConstraintUsableWithoutFM() throws {
            let vocabSize = 256
            let vocab: [String] = (0 ..< vocabSize).map { byte in
                String(format: "<0x%02X>", byte)
            }
            let tokenizer = try XGTokenizer(
                vocab: vocab,
                vocabType: XG_VOCAB_TYPE_BYTE_FALLBACK,
                eosTokenId: Int32(vocabSize - 1)
            )
            let schema = #"{ "type": "integer" }"#
            _ = try XGConstraint(tokenizer: tokenizer, jsonSchema: schema)
        }
    #endif

    // MARK: - Neither trait

    #if !FoundationModelsIntegration && !GuidedGenerationSupport
        @Test("Neither trait: MLXFoundationModels exposes only MLXDownloadProgress")
        func neitherTrait() {
            _ = MLXDownloadProgress.self
            // No MLXLanguageModel, no GuidedGenerationLoop, no XGConstraint.
        }
    #endif
}
