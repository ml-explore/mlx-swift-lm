// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    @Suite("MLXLanguageModel initialization")
    struct MLXLanguageModelInitTests {

        @Test("stores modelIdentifier on construction")
        func identifier() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let model = MLXLanguageModel(
                modelIdentifier: "mlx-community/Qwen3-4B-4bit",
                capabilities: LanguageModelCapabilities(capabilities: [.reasoning]),
                from: StubDownloader(),
                using: StubTokenizerLoader(),
                locatedBy: { _ in URL(fileURLWithPath: "/tmp") }
            )
            #expect(model.modelIdentifier == "mlx-community/Qwen3-4B-4bit")
        }
    }

    // MARK: - Test Stubs

    /// Minimal `Downloader` conformance. The tests in this suite only verify
    /// MLXLanguageModel's construction surface; no download is actually invoked.
    private final class StubDownloader: Downloader, @unchecked Sendable {
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

    /// Minimal `TokenizerLoader` conformance. As above, never invoked here.
    private final class StubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
        func load(from directory: URL) async throws -> any Tokenizer {
            StubTokenizer()
        }
    }

    /// Empty `Tokenizer` conformance returned by `StubTokenizerLoader.load`.
    /// All operations no-op or return empty results -- this exists only so the
    /// loader has something to hand back.
    private struct StubTokenizer: Tokenizer {
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
        ) throws -> [Int] {
            []
        }
    }

    // MARK: - Temperature plumbing

    /// Pure-function tests for the `Double?` (FoundationModels) →
    /// `Float?` (MLXLMCommon `GenerateParameters.temperature`) translation
    /// done by the unconstrained-generation path. Verifies the clamp
    /// semantics that prevent negative sampling temperatures from landing
    /// in `CategoricalSampler` and producing inverted distributions.
    @Suite("Temperature plumbing")
    struct TemperaturePlumbingTests {

        @Test("nil temperature returns nil so the sampler default is used")
        func nilPassesThrough() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.clampedTemperature(nil) == nil)
        }

        @Test("zero passes through unchanged — greedy via ArgMaxSampler")
        func zeroPassesThrough() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.clampedTemperature(0) == 0)
        }

        @Test("positive value passes through unchanged")
        func positivePassesThrough() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.clampedTemperature(0.7) == Float(0.7))
        }

        @Test("negative value clamps to zero")
        func negativeClampsToZero() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.clampedTemperature(-0.5) == 0)
        }

        @Test("Double precision narrows to Float without surprise")
        func doubleNarrowsToFloat() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Sanity check: 0.1 in Double rounds slightly differently than 0.1 in
            // Float. The helper's contract is `Float(max(0, value))`, so we assert
            // exactly that, not arbitrary equality.
            #expect(MLXLanguageModel.Executor.clampedTemperature(0.1) == Float(0.1))
        }
    }

    // MARK: - Typed error mapping

    /// Pure-function tests for the `XGError → Error` translation in
    /// `Executor.mapXGError(_:)`. Verifies that the one xgrammar case where
    /// user-fault is provable (`invalidJSONSchema`) maps to the typed
    /// `LanguageModelError.unsupportedGenerationGuide`, and everything else
    /// passes through untyped so internal-shim failures don't masquerade as
    /// developer mistakes.
    #if GuidedGenerationSupport
        @Suite("XGError typed mapping")
        struct XGErrorMappingTests {

            @Test("invalidJSONSchema maps to LanguageModelError.unsupportedGenerationGuide")
            func invalidJSONSchemaMapsToTypedError() throws {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
                let mapped = MLXLanguageModel.Executor.mapXGError(
                    .invalidJSONSchema(
                        "xgrammar rejected the schema: top-level type must be a string")
                )

                guard case LanguageModelError.unsupportedGenerationGuide(let payload) = mapped
                else {
                    Issue.record(
                        "Expected LanguageModelError.unsupportedGenerationGuide, got \(type(of: mapped)): \(mapped)"
                    )
                    return
                }
                #expect(
                    payload.schemaName == nil,
                    "We can't recover the schema name from the xgrammar error path")
                #expect(
                    payload.debugDescription
                        == "xgrammar rejected the schema: top-level type must be a string",
                    "Provider's raw error message should pass through verbatim into debugDescription"
                )
            }

            @Test("constraintCompilationFailed passes through unchanged (origin is ambiguous)")
            func constraintCompilationFailedPassesThrough() throws {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
                let original = XGError.constraintCompilationFailed("matcher init failed")
                let mapped = MLXLanguageModel.Executor.mapXGError(original)

                guard case XGError.constraintCompilationFailed(let msg) = mapped else {
                    Issue.record(
                        "Expected XGError.constraintCompilationFailed unchanged, got \(type(of: mapped)): \(mapped)"
                    )
                    return
                }
                #expect(msg == "matcher init failed")
            }

            @Test("tokenizerCreationFailed passes through unchanged (internal shim failure)")
            func tokenizerCreationFailedPassesThrough() throws {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
                let original = XGError.tokenizerCreationFailed("vocab extraction failed")
                let mapped = MLXLanguageModel.Executor.mapXGError(original)

                guard case XGError.tokenizerCreationFailed(let msg) = mapped else {
                    Issue.record(
                        "Expected XGError.tokenizerCreationFailed unchanged, got \(type(of: mapped)): \(mapped)"
                    )
                    return
                }
                #expect(msg == "vocab extraction failed")
            }
        }
    #endif  // GuidedGenerationSupport

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
