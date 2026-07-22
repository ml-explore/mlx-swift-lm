// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

// MARK: - Resource Bundle
//
// `Bundle.module` is synthesized only for SwiftPM resource-bearing targets; the
// hand-authored IntegrationTesting xcodeproj test target has no such accessor.
// The golden-fixture tests resolve their resources through this instead.
private final class FixturesBundleToken {}

/// The test bundle carrying the golden `Fixtures/` resources. The hand-authored
/// xcodeproj test target has no synthesized `Bundle.module`, so resources resolve
/// through this bundle token instead.
var fixturesBundle: Bundle { Bundle(for: FixturesBundleToken.self) }

// MARK: - Model Construction
//
// Model download + tokenizer wiring uses the SAME production path the
// `#huggingFaceLanguageModel` macro synthesizes: `#hubDownloader()` (a
// `HuggingFace.HubClient` bridge) paired with a `HubCache.default`-backed
// `weightsLocation:` closure (see `testLoad()` / `testWeightsLocation(modelID:)`).
// Both resolve against the same HubClient cache, so a downloaded model is seen
// by `MLXLanguageModel.modelExistsOnDisk()` — the shipping behavior. The
// `IntegrationTestingTests` target links swift-huggingface (`HuggingFace` +
// `MLXHuggingFace`), so the macros are available here.
//
// The rest of this file is gated on FoundationModelsIntegration. Consumers
// building the test target with `--disable-default-traits` (or the FM-trait
// explicitly turned off) can still use TestFixtures, ByteTokenizer, and
// SmallTokenizer — all of which live outside the gate — for tests that
// exercise xgrammar / MLXLMCommon directly.

#if FoundationModelsIntegration

/// Constructs an `MLXLanguageModel` using the production test downloader /
/// tokenizer loader and a `HubCache.default`-backed `weightsLocation:` closure.
///
/// Capabilities default to `[.guidedGeneration, .toolCalling]` (the common
/// case for tests that do not exercise reasoning). Pass an explicit set for
/// reasoning models or any other shape: capabilities are authoritative.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func makeTestModel(
    _ id: String,
    capabilities: [LanguageModelCapabilities.Capability]? = nil,
    resolver: (any ModelConfigurationResolver)? = nil
) -> MLXLanguageModel {
    let resolved = capabilities ?? defaultTestCapabilities()
    if let resolver {
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: id),
            capabilities: resolved,
            configurationResolver: resolver,
            weightsLocation: testWeightsLocation(modelID:),
            load: testLoad())
    }
    return MLXLanguageModel(
        configuration: ModelConfiguration(id: id),
        capabilities: resolved,
        weightsLocation: testWeightsLocation(modelID:),
        load: testLoad())
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func defaultTestCapabilities() -> [LanguageModelCapabilities.Capability] {
    [.guidedGeneration, .toolCalling]
}

/// Constructs an `MLXLanguageModel` for a reasoning-capable model id, declaring
/// `.reasoning` on top of the default capability set. Use for Qwen3 / R1-Distill
/// tests where `.reasoning` is load-bearing.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func makeReasoningTestModel(
    _ id: String,
    resolver: (any ModelConfigurationResolver)? = nil
) -> MLXLanguageModel {
    makeTestModel(
        id,
        capabilities: [.reasoning, .guidedGeneration, .toolCalling],
        resolver: resolver)
}

/// A `ContainerLoader` backed by the production Hugging Face wiring: the same
/// `#hubDownloader()` (a `HuggingFace.HubClient` bridge) and
/// `#huggingFaceTokenizerLoader()` the `#huggingFaceLanguageModel` macro
/// synthesizes. Paired with `testWeightsLocation(modelID:)`, so downloading
/// and on-disk availability resolve against the same HubClient cache — exactly
/// as they do in shipping code.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func testLoad() -> MLXLanguageModel.ContainerLoader {
    { configuration, progress in
        try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration, progressHandler: progress)
    }
}

/// Loads a `ModelContainer` for the given model identifier using the production
/// downloader/tokenizer pair (`#hubDownloader()` / `#huggingFaceTokenizerLoader()`).
///
/// On device (iOS 27), this MUST be invoked from a single xctest worker
/// process. xcodebuild's default `-parallel-testing-enabled YES` splits test
/// methods of one test target across N concurrent xctest processes. Each
/// worker has its own `MLXLanguageModel.cache` (`ModelCache` actor) singleton,
/// so cross-process dedup of the HubClient snapshot download does not exist.
/// Workers then race on the shared on-disk HubClient cache for the same repo,
/// and the losers can fail while moving a partially downloaded file into place.
///
/// `ModelCache.load` is correct, so the race is purely cross-process. Run the
/// model-dependent tests with parallel testing disabled
/// (`-parallel-testing-enabled NO`, a single worker) to avoid it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func loadTestModelContainer(id: String) async throws -> ModelContainer {
    try await makeTestModel(id).loadContainer()
}

// MARK: - Weights Location

/// Resolves the on-disk weights directory for a HuggingFace repo through the
/// same `HubCache.default` lookup the `#huggingFaceLanguageModel` macro
/// synthesizes, so it matches the cache `#hubDownloader()`'s `HubClient` writes
/// into — the two must agree so `MLXLanguageModel.modelExistsOnDisk()` can
/// probe for `config.json`.
func testWeightsLocation(modelID: String) -> URL {
    let cache = HuggingFace.HubCache.default
    guard let repo = HuggingFace.Repo.ID(rawValue: modelID) else {
        return cache.cacheDirectory
    }
    if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
        let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit)
    {
        return snapshot
    }
    return cache.repoDirectory(repo: repo, kind: .model)
}

// MARK: - Executor Helpers

/// Creates an MLX executor for the given model.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func makeMLXExecutor(for model: MLXLanguageModel) throws -> MLXLanguageModel.Executor {
    try MLXLanguageModel.Executor(
        configuration: MLXLanguageModel.Executor.Configuration(
            modelID: model.modelID)
    )
}

/// Creates a LanguageModelExecutorGenerationRequest with sensible defaults.
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

/// Bundles the framework channel + respond task into a single AsyncSequence.
///
/// Termination strategy: `LanguageModelExecutorGenerationChannel` has no
/// public `finish()`. In production the framework closes the channel after
/// respond returns; tests bypass the framework, so iterating the channel
/// directly hangs forever. We relay events into an `AsyncThrowingStream`
/// that we own. A producer task attaches an observer that yields readable
/// `GenerationEvent`s into our stream, then runs `respond()` and cancels a
/// collector task that drains and discards the channel's opaque events (just
/// enough to keep `respond()`'s sends from stalling). Our stream's
/// continuation is finished once both tasks settle, so `for try await`
/// terminates naturally. Early break from iteration cancels both tasks via
/// `deinit`, so tests that stop reading mid-generation don't waste GPU
/// compute on tokens nobody wants.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class TestResponseStream: AsyncSequence, @unchecked Sendable {
    typealias Element = MLXLanguageModel.Executor.GenerationEvent
    typealias AsyncIterator = AsyncThrowingStream<Element, Error>.AsyncIterator

    private let stream: AsyncThrowingStream<Element, Error>
    private let producerTask: Task<Void, Never>
    private let collectorTask: Task<Void, Never>

    init(
        executor: MLXLanguageModel.Executor,
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) {
        let channel = LanguageModelExecutorGenerationChannel()
        let (stream, continuation) = AsyncThrowingStream<Element, Error>.makeStream()
        self.stream = stream

        // The SDK's channel events are opaque, so we read generated content via
        // the executor's observation hook instead. We still must drain the
        // channel: respond() sends into it, and without a consumer those sends
        // would stall. The drained events are discarded; content comes from the
        // observer below.
        let collector = Task<Void, Never> {
            do {
                for try await _ in channel {}
            } catch {
                // Including CancellationError; we don't depend on cancellation here.
            }
        }
        self.collectorTask = collector

        // Producer: attach the observer (task-local, so it also reaches the
        // guided-path forwarder child task), run respond(), then finish our
        // stream so the test's iteration terminates.
        self.producerTask = Task<Void, Never> {
            defer { collector.cancel() }
            await MLXLanguageModel.Executor.$generationObserver.withValue({ event in
                continuation.yield(event)
            }) {
                do {
                    try await executor.respond(
                        to: request, model: model, streamingInto: channel)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    deinit {
        producerTask.cancel()
        collectorTask.cancel()
    }

    func makeAsyncIterator() -> AsyncIterator {
        stream.makeAsyncIterator()
    }
}

/// Starts executor.respond(...) on a background task and returns a wrapper that
/// iterates the generation channel. Errors from respond() surface when iteration ends.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func executeResponse(
    _ executor: MLXLanguageModel.Executor,
    request: LanguageModelExecutorGenerationRequest,
    model: MLXLanguageModel
) async throws -> TestResponseStream {
    TestResponseStream(executor: executor, request: request, model: model)
}

// MARK: - GPU Memory Management

/// Releases all GPU memory: synchronizes pending GPU work, evicts cached models,
/// then clears the Metal buffer pool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func releaseAllGPUMemory() async {
    Stream.gpu.synchronize()
    await MLXLanguageModel.evictAll()
    Stream.gpu.synchronize()
    GPU.clearCache()
}

#endif  // FoundationModelsIntegration

// MARK: - Shared Test Fixtures

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

// MARK: - Test Tokenizers

/// Minimal 256 single-byte tokenizer for tests.
/// Each byte is its own token ID, enabling exact character-to-ID mapping.
///
/// Conforms to `MLXLMCommon.Tokenizer` because every consumer (`GrammarTokenizer`
/// initialiser, `ClosingTokenBias.compute`, `WhitespaceTokenBias.compute`)
/// expects that protocol.
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

/// Configurable tokenizer with an arbitrary token list.
/// Token at index i has ID i. No EOS token.
struct SmallTokenizer: MLXLMCommon.Tokenizer {
    let tokens: [String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        self.tokens.firstIndex(of: token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < self.tokens.count else { return nil }
        return self.tokens[id]
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
