// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class SeedRecorder: @unchecked Sendable, Hashable {
    private let lock = NSLock()
    private var storage: UInt64?

    static func == (lhs: SeedRecorder, rhs: SeedRecorder) -> Bool {
        lhs === rhs
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }

    func record(_ seed: UInt64?) {
        lock.lock()
        storage = seed
        lock.unlock()
    }

    var seed: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RecordingLanguageModel: LanguageModel {
    typealias Executor = RecordingLanguageModelExecutor

    let recorder: SeedRecorder

    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([])
    }

    var executorConfiguration: SeedRecorder {
        recorder
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RecordingLanguageModelExecutor: LanguageModelExecutor {
    typealias Configuration = SeedRecorder
    typealias Model = RecordingLanguageModel

    let recorder: SeedRecorder

    init(configuration: SeedRecorder) throws {
        recorder = configuration
    }

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: RecordingLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let seed: UInt64?
        switch request.generationOptions.samplingMode?.kind {
        case .some(.randomTopK(_, let value)):
            seed = value
        default:
            seed = nil
        }
        recorder.record(seed)
        await channel.send(
            .response(action: .appendText("ok", tokenCount: 1)))
    }
}

@Suite("GenerationOptions forwarding")
struct GenerationOptionsForwardingTests {
    @Test("LanguageModelSession forwards an exact UInt64 seed to the executor")
    func seedReachesExecutorWithoutNarrowing() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let recorder = SeedRecorder()
        let session = LanguageModelSession(
            model: RecordingLanguageModel(recorder: recorder))

        _ = try await session.respond(
            to: "Record the generation options.",
            options: GenerationOptions(
                samplingMode: .random(top: 40, seed: UInt64.max),
                maximumResponseTokens: 1))

        #expect(recorder.seed == UInt64.max)
    }
}

#endif
