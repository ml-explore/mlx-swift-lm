// Copyright © 2025 Apple Inc.
//
// Model-free test helpers for the in-package `MLXFoundationModelsTests` target.
//
// Model-DOWNLOADING infrastructure (the swift-transformers-backed
// `TestHubDownloader` / `TestHuggingFaceTokenizerLoader`, `loadTestModelContainer`,
// `makeTestModel`, `TestResponseStream`, etc.) lives in
// `IntegrationTesting/IntegrationTestingTests/FMTestHelpers.swift` — the
// IntegrationTesting xcodeproj is the only place that carries the
// `swift-transformers` dependency. This file keeps only what the model-free
// in-package tests need: fake tokenizers, a stub-backed model constructor for
// construction / capability / gate-rejection tests, and the download-free
// executor machinery.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    // MARK: - Stub Downloader / TokenizerLoader
    //
    // For tests that construct an `MLXLanguageModel` but never actually load one:
    // capability assertions and construction paths. No network, no weights.

    private struct StubDownloader: MLXLMCommon.Downloader, @unchecked Sendable {
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

    private struct StubTokenizerLoader: MLXLMCommon.TokenizerLoader, @unchecked Sendable {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { StubTokenizer() }
    }

    private struct StubTokenizer: MLXLMCommon.Tokenizer {
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

    // MARK: - Model Construction (no download)

    /// Constructs an `MLXLanguageModel` wired to stub download / tokenizer infra,
    /// for tests that exercise construction, stored capabilities, or gate-rejection
    /// paths WITHOUT loading a real model. Tests that need a real model live in the
    /// IntegrationTesting xcodeproj (`FMTestHelpers.makeTestModel`).
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeStubModel(
        _ id: String,
        capabilities: LanguageModelCapabilities? = nil
    ) -> MLXLanguageModel {
        let resolved =
            capabilities
            ?? {
                var set: [LanguageModelCapabilities.Capability] = []
                set += [.guidedGeneration, .toolCalling]
                return LanguageModelCapabilities(capabilities: set)
            }()
        return MLXLanguageModel(
            modelIdentifier: id,
            capabilities: resolved,
            from: StubDownloader(),
            using: StubTokenizerLoader(),
            locatedBy: { _ in URL(fileURLWithPath: "/tmp") }
        )
    }

    // MARK: - Executor Helpers (download-free machinery)

    /// Creates an MLX executor for the given model. Construction only — no download.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeMLXExecutor(for model: MLXLanguageModel) throws -> MLXLanguageModel.Executor {
        try MLXLanguageModel.Executor(
            configuration: MLXLanguageModel.Executor.Configuration(
                modelIdentifier: model.modelIdentifier)
        )
    }

    /// Creates a `LanguageModelExecutorGenerationRequest` with sensible defaults.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeExecutorRequest(
        id: UUID = UUID(),
        transcript: Transcript,
        enabledTools: [Transcript.ToolDefinition] = [],
        schema: GenerationSchema? = nil,
        generationOptions: GenerationOptions = GenerationOptions(),
        contextOptions: ContextOptions = ContextOptions(),
        metadata: [String: any Sendable & Codable & Equatable] = [:]
    ) -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: id,
            transcript: transcript,
            enabledTools: enabledTools,
            schema: schema,
            generationOptions: generationOptions,
            contextOptions: contextOptions,
            metadata: metadata
        )
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)

// MARK: - Shared Test Fixtures (model-free)

enum TestFixtures {

    /// The exact JSON schema emitted by `@Generable Itinerary` in the TripPlanner sample app.
    static let itinerarySchemaProduction = """
        {"properties":{"rationale":{"type":"string","description":"An explanation of how the itinerary meets the person's special requests."},"days":{"type":"array","items":{"$ref":"#/$defs/DayPlan"},"maxItems":3,"description":"A list of day-by-day plans.","minItems":3},"title":{"type":"string","description":"An exciting name for the trip."},"destinationName":{"type":"string","enum":["Sahara Desert","Serengeti","Deadvlei","Grand Canyon","Niagara Falls","Joshua Tree","Rocky Mountains","Monument Valley","Muir Woods","Amazon Rainforest","Lençóis Maranhenses","Uyuni Salt Flat","White Cliffs of Dover","Alps","Mount Fuji","Wulingyuan","Mount Everest","Great Barrier Reef","South Shetland Islands"]},"description":{"type":"string"}},"type":"object","required":["title","destinationName","description","rationale","days"],"x-order":["title","destinationName","description","rationale","days"],"title":"Itinerary","$defs":{"Activity":{"additionalProperties":false,"title":"Activity","type":"object","properties":{"type":{"type":"string","enum":["sightseeing","foodAndDining","shopping","hotelAndLodging"]},"title":{"type":"string"},"description":{"type":"string"}},"x-order":["type","title","description"],"required":["type","title","description"]},"DayPlan":{"properties":{"activities":{"type":"array","minItems":3,"items":{"$ref":"#/$defs/Activity"},"maxItems":3},"subtitle":{"type":"string"},"destination":{"type":"string"},"title":{"description":"A unique and exciting title for this day plan.","type":"string"}},"required":["title","subtitle","destination","activities"],"additionalProperties":false,"x-order":["title","subtitle","destination","activities"],"type":"object","title":"DayPlan"}},"additionalProperties":false}
        """

    /// Variant with maxLength constraints on all string fields, suitable for generation tests
    /// where bounded output keeps test time reasonable.
    static let itinerarySchemaConstrained = """
        {
            "type": "object",
            "properties": {
                "title": { "type": "string", "maxLength": 100 },
                "destinationName": {
                    "type": "string",
                    "enum": ["Sahara Desert", "Serengeti", "Deadvlei", "Grand Canyon", "Niagara Falls", "Joshua Tree", "Rocky Mountains", "Monument Valley", "Muir Woods", "Amazon Rainforest", "White Cliffs of Dover", "Alps", "Mount Fuji", "Wulingyuan", "Mount Everest", "Great Barrier Reef", "South Shetland Islands"]
                },
                "description": { "type": "string", "maxLength": 100 },
                "rationale": { "type": "string", "maxLength": 100 },
                "days": {
                    "type": "array",
                    "items": { "$ref": "#/$defs/DayPlan" },
                    "minItems": 3,
                    "maxItems": 3
                }
            },
            "required": ["title", "destinationName", "description", "rationale", "days"],
            "additionalProperties": false,
            "$defs": {
                "Activity": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                        },
                        "title": { "type": "string", "maxLength": 40 },
                        "description": { "type": "string", "maxLength": 40 }
                    },
                    "required": ["type", "title", "description"],
                    "additionalProperties": false,
                    "x-order": ["type", "title", "description"]
                },
                "DayPlan": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "maxLength": 60 },
                        "subtitle": { "type": "string", "maxLength": 60 },
                        "destination": { "type": "string", "maxLength": 60 },
                        "activities": {
                            "type": "array",
                            "items": { "$ref": "#/$defs/Activity" },
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["title", "subtitle", "destination", "activities"],
                    "additionalProperties": false,
                    "x-order": ["title", "subtitle", "destination", "activities"]
                }
            },
            "x-order": ["title", "destinationName", "description", "rationale", "days"]
        }
        """

    static let itineraryPrompt =
        "Generate a 3-day travel itinerary to Mount Fuji with 3 activities per day. Respond as JSON."

    static let gemmaModelID = "mlx-community/gemma-3-270m-it-4bit"

    /// Default model ID for tests that don't care which specific MLX model runs,
    /// but do need a model known to exercise the full guided-generation and
    /// tool-calling paths.
    static let defaultModelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
}

// MARK: - Test Tokenizers (model-free)

/// Minimal 256 single-byte tokenizer for tests.
/// Each byte is its own token ID, enabling exact character-to-ID mapping.
struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0 && id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { String(UnicodeScalar(UInt8(255))) }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
