// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class SteeringTests: XCTestCase {
    private struct FinalizingIterator: GenerationFinalizingTokenIterator {
        let tokens: [Int]
        var onNext: @Sendable (Int) -> Void = { _ in }
        var tokenCount = 0
        var maxTokens: Int? { tokens.count }
        var promptPrefillTime: TimeInterval { 0 }
        var state: LMOutput.State? = LMOutput.State()

        mutating func next() -> Int? {
            guard tokenCount < tokens.count else { return nil }
            onNext(tokenCount)
            defer { tokenCount += 1 }
            return tokens[tokenCount]
        }

        mutating func finalizeGeneration() {
            state?[LMOutput.Key<Int>("finalizedAt")] = tokenCount
        }
    }

    private struct BoundaryTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { "<eos>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func convertTokenToId(_ token: String) -> Int? { token == "<eos>" ? 99 : nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let pieces = [
                1: "hello ", 2: "<en", 3: "ough", 4: " more",
                5: "{\"name\":\"weather\",\"arguments\":", 6: "{}}",
            ]
            if tokenIds == [10] { return "\u{fffd}" }
            if tokenIds == [10, 11] { return "é" }
            return tokenIds.compactMap { pieces[$0] }.joined()
        }
    }

    func testDecoderWaitsForCompleteUnicodeAndStopPrefix() {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: BoundaryTokenizer(), format: .json, tools: nil, stopStrings: ["<end>"])
        _ = decoder.push(10) { _ in true }
        XCTAssertFalse(decoder.canEndForSteering)
        _ = decoder.push(11) { _ in true }
        XCTAssertTrue(decoder.canEndForSteering)

        decoder = StandardTokenStreamDecoder(
            tokenizer: BoundaryTokenizer(), format: .json, tools: nil, stopStrings: ["<end>"])
        _ = decoder.push(1) { _ in true }
        XCTAssertTrue(decoder.canEndForSteering)
        _ = decoder.push(2) { _ in true }
        XCTAssertFalse(decoder.canEndForSteering)
        _ = decoder.push(3) { _ in true }
        XCTAssertTrue(decoder.canEndForSteering)
    }

    func testSteeredStepReturnsFinalizedStateAndExactTokens() async throws {
        let control = SteeringControl()
        _ = try control.enqueue("now", policy: .nextSafeBoundary)
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 2, modelConfiguration: ModelConfiguration(id: "test"),
            tokenizer: BoundaryTokenizer(), iterator: FinalizingIterator(tokens: [1, 4, 4]),
            steering: control)
        var info: GenerateCompletionInfo?
        for await event in stream { info = event.info ?? info }
        let result = await task.value
        XCTAssertEqual(result.tokens, [1])
        XCTAssertEqual(result.state.consume()?[LMOutput.Key<Int>("finalizedAt")], 1)
        XCTAssertEqual(info?.stopReason, .steered)
        // Ending a step does not consume or close its turn's mailbox.
        XCTAssertTrue(control.requestsEarlyBoundary)
    }

    func testReasoningModelUsesNaturalStepBoundary() async throws {
        let control = SteeringControl()
        _ = try control.enqueue("now", policy: .nextSafeBoundary)
        let configuration = ModelConfiguration(
            id: "test", reasoningConfig: QwenReasoningProtocol.qwen3)
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 2, modelConfiguration: configuration,
            tokenizer: BoundaryTokenizer(), iterator: FinalizingIterator(tokens: [1, 4, 4]),
            steering: control)
        var info: GenerateCompletionInfo?
        for await event in stream { info = event.info ?? info }
        let result = await task.value
        XCTAssertEqual(result.tokens, [1, 4, 4])
        XCTAssertEqual(info?.stopReason, .length)
    }

    func testInstructionDuringToolPayloadDoesNotTruncateCall() async throws {
        let control = SteeringControl()
        let iterator = FinalizingIterator(
            tokens: [1, 5, 6, 99],
            onNext: { index in
                if index == 1 { _ = try? control.enqueue("now", policy: .nextSafeBoundary) }
            })
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 2, modelConfiguration: ModelConfiguration(id: "test"),
            tokenizer: BoundaryTokenizer(), iterator: iterator, steering: control)
        var info: GenerateCompletionInfo?
        var calls = 0
        for await event in stream {
            info = event.info ?? info
            if event.toolCall != nil { calls += 1 }
        }
        let result = await task.value
        XCTAssertEqual(result.tokens, [1, 5, 6, 99])
        XCTAssertEqual(info?.stopReason, .stop)
        XCTAssertEqual(calls, 1)
        XCTAssertTrue(control.requestsEarlyBoundary)
    }

    func testInstructionArrivingDuringDecodingStopsAtNextBoundary() async throws {
        let control = SteeringControl()
        let iterator = FinalizingIterator(
            tokens: [1, 4, 4, 4, 4],
            onNext: { index in
                if index == 2 { _ = try? control.enqueue("now", policy: .nextSafeBoundary) }
            })
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 2, modelConfiguration: ModelConfiguration(id: "test"),
            tokenizer: BoundaryTokenizer(), iterator: iterator, steering: control)
        for await _ in stream {}
        let result = await task.value
        XCTAssertEqual(result.tokens, [1, 4, 4])
    }

    func testMailboxBoundsAndPromotion() throws {
        let control = SteeringControl()
        var ids: [UUID] = []
        for _ in 0 ..< SteeringLimits.maxPendingInstructions - 1 {
            ids.append(try control.enqueue("queued", policy: .nextStepBoundary))
        }
        XCTAssertFalse(control.requestsEarlyBoundary)
        ids.append(try control.enqueue("now", policy: .nextSafeBoundary))
        XCTAssertTrue(control.requestsEarlyBoundary)
        XCTAssertThrowsError(try control.enqueue("full", policy: .nextStepBoundary))
        XCTAssertEqual(try control.take(closeIfEmpty: true).map(\.id), ids)
        XCTAssertFalse(control.requestsEarlyBoundary)
        _ = try control.enqueue(
            String(repeating: "a", count: SteeringLimits.maxPendingBytes),
            policy: .nextStepBoundary)
        XCTAssertThrowsError(try control.enqueue("é", policy: .nextStepBoundary))
        XCTAssertEqual(try control.take(closeIfEmpty: false).count, 1)
        XCTAssertTrue(try control.take(closeIfEmpty: true).isEmpty)
        XCTAssertThrowsError(try control.enqueue("closed", policy: .nextSafeBoundary))
    }

    func testAcceptanceRacingCompletionIsNeverLost() async {
        for _ in 0 ..< 100 {
            let control = SteeringControl()
            async let accepted: UUID? = try? control.enqueue("race", policy: .nextSafeBoundary)
            async let taken = control.take(closeIfEmpty: true)
            let id = await accepted
            let batch = try? await taken
            if let id {
                XCTAssertEqual(batch?.map(\.id), [id])
            } else {
                XCTAssertEqual(batch?.count, 0)
            }
            control.finish()
        }
    }

    func testFinalDrainClosesAcceptanceWithPendingInstructions() throws {
        let control = SteeringControl()
        let id = try control.enqueue("late", policy: .nextSafeBoundary)
        XCTAssertEqual(try control.closeAndTake().map(\.id), [id])
        XCTAssertFalse(control.isOpen)
        XCTAssertFalse(control.requestsEarlyBoundary)
        XCTAssertThrowsError(try control.enqueue("too late", policy: .nextStepBoundary)) {
            XCTAssertEqual($0 as? SteeringError, .responseEnded)
        }
    }

    func testAcceptanceRacingFinalDrainIsNeverLost() async throws {
        for _ in 0 ..< 100 {
            let control = SteeringControl()
            let pending = try control.enqueue("pending", policy: .nextStepBoundary)
            async let accepted: UUID? = try? control.enqueue("race", policy: .nextSafeBoundary)
            async let drained = control.closeAndTake()
            let id = await accepted
            let batch = try await drained
            XCTAssertEqual(batch.map(\.id), [pending] + (id.map { [$0] } ?? []))
            XCTAssertFalse(control.isOpen)
        }
    }

    func testCancelBeforeWorkerInstallation() async throws {
        let control = SteeringControl()
        _ = try control.enqueue("pending", policy: .nextSafeBoundary)
        control.cancel()
        let cancelled = expectation(description: "worker cancelled on attachment")
        let task = Task {
            do {
                try await Task.sleep(for: .seconds(10))
                XCTFail("Expected cancellation")
            } catch {
                XCTAssertTrue(error is CancellationError)
                cancelled.fulfill()
            }
        }
        control.setTask(task)
        await fulfillment(of: [cancelled], timeout: 1)
        await control.synchronize()
        XCTAssertFalse(control.requestsEarlyBoundary)
        XCTAssertThrowsError(try control.take(closeIfEmpty: true)) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertThrowsError(try control.enqueue("late", policy: .nextStepBoundary))
    }

    func testSessionRoutesToActiveRequestBeforeQueuedRequests() throws {
        let session = SessionSteering()
        XCTAssertThrowsError(try session.steer("idle", policy: .nextSafeBoundary, response: nil)) {
            XCTAssertEqual($0 as? SteeringError, .noActiveResponse)
        }
        let first = SteeringControl()
        let second = SteeringControl()
        session.register(first)
        let beforeStart = try session.steer(
            "before start", policy: .nextStepBoundary, response: nil)
        session.start(first)
        session.register(second)
        let active = try session.steer("active", policy: .nextSafeBoundary, response: nil)
        XCTAssertEqual(try first.take(closeIfEmpty: false).map(\.id), [beforeStart, active])
        XCTAssertTrue(try second.take(closeIfEmpty: false).isEmpty)
        XCTAssertTrue(try first.take(closeIfEmpty: true).isEmpty)
        session.remove(first)
        let queued = try session.steer("queued", policy: .nextStepBoundary, response: nil)
        XCTAssertEqual(try second.take(closeIfEmpty: false).map(\.id), [queued])
    }

    func testRoutingSkipsAResponseThatHasNotBeenRemovedYet() throws {
        // A cancelled runner closes its mailbox long before its `defer` removes it.
        // Input must reach the response the session can still deliver to.
        let session = SessionSteering()
        let cancelled = SteeringControl()
        let live = SteeringControl()
        session.register(cancelled)
        session.register(live)
        session.start(cancelled)
        cancelled.cancel()
        XCTAssertFalse(session.canSteer(cancelled.responseID))
        XCTAssertTrue(session.canSteer(nil))

        let id = try session.steer("after cancel", policy: .nextSafeBoundary, response: nil)
        XCTAssertEqual(try live.take(closeIfEmpty: false).map(\.id), [id])

        // Naming the cancelled response reports it instead of retargeting.
        XCTAssertThrowsError(
            try session.steer("named", policy: .nextSafeBoundary, response: cancelled.responseID)
        ) {
            XCTAssertEqual($0 as? SteeringError, .responseEnded)
        }
        XCTAssertTrue(try live.take(closeIfEmpty: false).isEmpty)
    }

    func testNamedResponseIsNeverRetargeted() throws {
        let session = SessionSteering()
        let first = SteeringControl()
        let second = SteeringControl()
        session.register(first)
        session.register(second)
        session.start(first)
        XCTAssertEqual(session.latestResponse, second.responseID)

        let id = try session.steer(
            "for second", policy: .nextStepBoundary, response: second.responseID)
        XCTAssertEqual(try second.take(closeIfEmpty: false).map(\.id), [id])
        XCTAssertTrue(try first.take(closeIfEmpty: false).isEmpty)

        session.remove(second)
        XCTAssertFalse(session.canSteer(second.responseID))
        XCTAssertThrowsError(
            try session.steer("gone", policy: .nextStepBoundary, response: second.responseID)
        ) {
            XCTAssertEqual($0 as? SteeringError, .noActiveResponse)
        }
    }

    func testWhitespaceInstructionIsRejectedWhateverTheSessionIsDoing() throws {
        let session = SessionSteering()
        XCTAssertThrowsError(try session.steer(" \n", policy: .nextSafeBoundary, response: nil)) {
            XCTAssertEqual($0 as? SteeringError, .emptyInstruction)
        }
        let control = SteeringControl()
        session.register(control)
        session.start(control)
        XCTAssertThrowsError(try session.steer(" \n", policy: .nextSafeBoundary, response: nil)) {
            XCTAssertEqual($0 as? SteeringError, .emptyInstruction)
        }
    }

    /// `canSteer` and `steer` must select the same response, including when an
    /// unsteerable response is followed by a steerable one.
    func testCanSteerAgreesWithSteerSelection() throws {
        let session = SessionSteering()
        let unsteerable = SteeringControl()
        let queued = SteeringControl()
        session.register(unsteerable)
        session.register(queued)
        session.start(unsteerable)
        _ = unsteerable.setSteerable(false)

        XCTAssertFalse(session.canSteer(nil))
        XCTAssertThrowsError(try session.steer("no", policy: .nextSafeBoundary, response: nil)) {
            XCTAssertEqual($0 as? SteeringError, .notSteerable)
        }
        XCTAssertTrue(try queued.take(closeIfEmpty: false).isEmpty)

        session.remove(unsteerable)
        XCTAssertTrue(session.canSteer(nil))
        let id = try session.steer("yes", policy: .nextSafeBoundary, response: nil)
        XCTAssertEqual(try queued.take(closeIfEmpty: false).map(\.id), [id])
    }

    func testUnsteerableResponseRejectsInputAndReleasesWhatItAccepted() throws {
        let session = SessionSteering()
        let control = SteeringControl()
        session.register(control)
        session.start(control)
        let early = try session.steer(
            "before the cache was known", policy: .nextSafeBoundary, response: nil)

        // The runner discovers it restored a raw cache with no transcript.
        XCTAssertEqual(control.setSteerable(false).map(\.id), [early])
        XCTAssertFalse(control.requestsEarlyBoundary)
        XCTAssertFalse(session.canSteer(nil))
        XCTAssertThrowsError(try session.steer("late", policy: .nextSafeBoundary, response: nil)) {
            XCTAssertEqual($0 as? SteeringError, .notSteerable)
        }
    }

    func testActualCacheOwnerWinsOverRegistrationOrder() throws {
        let session = SessionSteering()
        let first = SteeringControl()
        let second = SteeringControl()
        session.register(first)
        session.register(second)
        session.start(second)
        let id = try session.steer("active", policy: .nextSafeBoundary, response: nil)
        XCTAssertEqual(try second.take(closeIfEmpty: false).map(\.id), [id])
        XCTAssertTrue(try first.take(closeIfEmpty: false).isEmpty)
        session.start(first)
        session.remove(second)
        let next = try session.steer("next", policy: .nextStepBoundary, response: nil)
        XCTAssertEqual(try first.take(closeIfEmpty: false).map(\.id), [next])
    }

    func testPartialToolPayloadCannotEndForSteering() {
        let processor = ToolCallProcessor(format: .json)
        _ = processor.processChunkOutputs("Some text ")
        XCTAssertTrue(processor.canEndForSteering)
        _ = processor.processChunkOutputs("{\"name\":\"weather\",\"arguments\":")
        XCTAssertFalse(processor.canEndForSteering)
        let output = processor.processChunkOutputs("{\"city\":\"Paris\"}}")
        XCTAssertTrue(output.contains { if case .toolCall = $0 { true } else { false } })
        XCTAssertTrue(processor.canEndForSteering)
    }
}
