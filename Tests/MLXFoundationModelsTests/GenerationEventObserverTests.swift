// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Synchronization
import Testing

@testable import MLXFoundationModels

/// Proves the behavior-neutral observation hook: when an observer is attached
/// via the task-local, each emit helper both sends to the channel (drained and
/// discarded here) and hands the observer a readable GenerationEvent mirror.
@Suite("GenerationEvent observer")
struct GenerationEventObserverTests {

    /// Drains `channel` until it closes or the caller cancels this task.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func drain(
        _ channel: LanguageModelExecutorGenerationChannel
    ) -> Task<Void, any Error> {
        Task {
            for try await _ in channel {}
        }
    }

    /// Awaits a cancelled drain task, recording any error other than the
    /// expected `CancellationError`.
    private func assertDrainCancelled(_ task: Task<Void, any Error>) async {
        do {
            try await task.value
        } catch is CancellationError {
            // expected: the drain loop was still running when we cancelled it.
        } catch {
            Issue.record("drain task failed with unexpected error: \(error)")
        }
    }

    /// Runs `body` with an attached observer and a drained channel, returning
    /// every mirrored event the observer received.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func capture(
        _ body: (LanguageModelExecutorGenerationChannel) async -> Void
    ) async -> [MLXLanguageModel.Executor.GenerationEvent] {
        let channel = LanguageModelExecutorGenerationChannel()
        let drainTask = drain(channel)
        let box = EventBox()
        await MLXLanguageModel.Executor.$generationObserver.withValue(
            { box.append($0) },
            operation: { await body(channel) }
        )
        drainTask.cancel()
        await assertDrainCancelled(drainTask)
        return box.events
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class EventBox: Sendable {
        private let storage = Mutex<[MLXLanguageModel.Executor.GenerationEvent]>([])

        func append(_ event: MLXLanguageModel.Executor.GenerationEvent) {
            storage.withLock { $0.append(event) }
        }

        var events: [MLXLanguageModel.Executor.GenerationEvent] {
            storage.withLock { $0 }
        }
    }

    @Test("appendText is mirrored with destination and entryID")
    func mirrorsText() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let events = await capture { channel in
            await MLXLanguageModel.Executor.emit(
                text: "hi", entryID: "e1", destination: .response, into: channel)
        }
        #expect(events.count == 1)
        guard case .appendText(let text, "e1", .response) = events.first else {
            Issue.record("expected .appendText mirror, got \(String(describing: events.first))")
            return
        }
        #expect(text == "hi")
    }

    @Test("toolCall is mirrored with name and arguments")
    func mirrorsToolCall() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let events = await capture { channel in
            await MLXLanguageModel.Executor.emitToolCall(
                id: "id1", name: "get_weather", arguments: "{\"a\":1}",
                entryID: "tc1", into: channel)
        }
        guard case .toolCall(_, let name, let arguments) = events.first else {
            Issue.record("expected .toolCall mirror, got \(String(describing: events.first))")
            return
        }
        #expect(name == "get_weather")
        #expect(arguments == "{\"a\":1}")
    }

    @Test("no observer attached means no crash and events are simply sent")
    func noObserverIsSafe() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Not inside withValue: generationObserver is nil (shipping behavior).
        let channel = LanguageModelExecutorGenerationChannel()
        let drainTask = drain(channel)
        await MLXLanguageModel.Executor.emit(
            text: "x", entryID: nil, destination: .reasoning, into: channel)
        drainTask.cancel()
        // Reaching here without trapping is the assertion.
        await assertDrainCancelled(drainTask)
    }
}

#endif
