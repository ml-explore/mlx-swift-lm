// Copyright © 2026 Apple Inc.

import Dispatch
import MLX
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

final class GenerationExecutionTests: XCTestCase {
    private enum Context {
        @TaskLocal static var value = 0
    }

    private struct CancellingPolicy: WiredMemoryPolicy {
        let identifier = UUID()
        var id: AnyHashable { identifier }

        func limit(baseline: Int, activeSizes: [Int]) -> Int { baseline }

        func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
            withUnsafeCurrentTask { $0?.cancel() }
            return true
        }
    }

    private struct Iterator: GenerationFinalizingTokenIterator {
        var maxTokens: Int? = 1
        var tokenCount = 0
        var promptPrefillTime: TimeInterval { 0 }
        var evictedTokenCount = 0
        var reasoningTokenCount: Int?
        var onNext: @Sendable () -> Void = {}
        var onFinalize: @Sendable () -> Void = {}

        mutating func next() -> Int? {
            guard tokenCount < (maxTokens ?? Int.max) else { return nil }
            onNext()
            tokenCount += 1
            return 42
        }

        mutating func finalizeGeneration() {
            onFinalize()
        }
    }

    // Run with LIBDISPATCH_COOPERATIVE_POOL_STRICT=1 to expose pool starvation
    // deterministically. The timeout also lets a regressed implementation finish.
    func testBlockingIteratorLeavesCooperativeWorkerAvailable() async {
        for useTicket in [false, true] {
            let iterator = Iterator(
                onNext: Self.assertCooperativeProgress,
                onFinalize: Self.assertCooperativeProgress)
            let ticket = useTicket ? MLX.WiredSumPolicy().ticket(size: 0) : nil
            let (stream, task) = generateTask(
                promptTokenCount: 1, modelConfiguration: .init(id: "test"),
                tokenizer: PseudoWordTokenizer(), iterator: iterator, wiredMemoryTicket: ticket)
            for await _ in stream {}
            await task.value
        }
    }

    private static func assertCooperativeProgress() {
        let resumed = DispatchSemaphore(value: 0)
        Task { resumed.signal() }
        XCTAssertEqual(resumed.wait(timeout: .now() + 5), .success)
    }

    func testGenerationPreservesDeviceOverrideAndCooperativeProgress() async {
        await Device.withDefaultDevice(.cpu) {
            let (stream, task) = generateTaskRecordingTokens(
                promptTokenCount: 1, modelConfiguration: .init(id: "test"),
                tokenizer: PseudoWordTokenizer(),
                iterator: Iterator(
                    maxTokens: 3,
                    onNext: {
                        XCTAssertEqual(Device.defaultDevice().deviceType, .cpu)
                        Self.assertCooperativeProgress()
                    },
                    onFinalize: {
                        XCTAssertEqual(Device.defaultDevice().deviceType, .cpu)
                        Self.assertCooperativeProgress()
                    }))
            var completion: GenerateCompletionInfo?
            for await event in stream {
                if case .info(let info) = event { completion = info }
            }
            let tokens = await task.value
            XCTAssertEqual(tokens, [42, 42, 42])
            XCTAssertEqual(completion?.generationTokenCount, 3)
            XCTAssertEqual(completion?.stopReason, .length)
        }
    }

    func testCancellingOneGenerationDoesNotCancelAnother() async {
        let firstEntered = expectation(description: "first generation entered next")
        let secondEntered = expectation(description: "second generation entered next")
        let resumeFirst = DispatchSemaphore(value: 0)
        let resumeSecond = DispatchSemaphore(value: 0)
        let (firstStream, first) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                maxTokens: 10,
                onNext: {
                    firstEntered.fulfill()
                    XCTAssertEqual(resumeFirst.wait(timeout: .now() + 5), .success)
                    XCTAssertTrue(Task.isCancelled)
                },
                onFinalize: { XCTAssertTrue(Task.isCancelled) }))
        let (secondStream, second) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                onNext: {
                    secondEntered.fulfill()
                    XCTAssertEqual(resumeSecond.wait(timeout: .now() + 5), .success)
                    XCTAssertFalse(Task.isCancelled)
                },
                onFinalize: { XCTAssertFalse(Task.isCancelled) }))

        await fulfillment(of: [firstEntered, secondEntered], timeout: 5)
        first.cancel()
        resumeFirst.signal()
        let firstTokens = await first.value
        XCTAssertEqual(firstTokens, [42])
        XCTAssertFalse(second.isCancelled)
        resumeSecond.signal()
        let secondTokens = await second.value
        XCTAssertEqual(secondTokens, [42])

        var firstCompletion: GenerateCompletionInfo?
        for await event in firstStream {
            if case .info(let info) = event { firstCompletion = info }
        }
        var secondCompletion: GenerateCompletionInfo?
        for await event in secondStream {
            if case .info(let info) = event { secondCompletion = info }
        }
        XCTAssertEqual(firstCompletion?.stopReason, .cancelled)
        XCTAssertEqual(secondCompletion?.stopReason, .length)
    }

    func testIndependentGenerationsCanMakeProgressConcurrently() async {
        let firstStarted = DispatchSemaphore(value: 0)
        let secondStarted = DispatchSemaphore(value: 0)
        let (firstStream, first) = generateTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(onNext: {
                firstStarted.signal()
                XCTAssertEqual(secondStarted.wait(timeout: .now() + 5), .success)
            }))
        let (secondStream, second) = generateTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(onNext: {
                XCTAssertEqual(firstStarted.wait(timeout: .now() + 5), .success)
                secondStarted.signal()
            }))
        await first.value
        await second.value
        withExtendedLifetime((firstStream, secondStream)) {}
    }

    func testCancellationDuringNextStillFinalizesBeforeTaskCompletes() async {
        let entered = expectation(description: "entered next")
        let resume = DispatchSemaphore(value: 0)
        let finalized = expectation(description: "finalized after cancellation")
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                maxTokens: 10,
                onNext: {
                    entered.fulfill()
                    XCTAssertEqual(resume.wait(timeout: .now() + 5), .success)
                    XCTAssertTrue(Task.isCancelled)
                },
                onFinalize: {
                    XCTAssertTrue(Task.isCancelled)
                    finalized.fulfill()
                }))
        await fulfillment(of: [entered], timeout: 5)
        task.cancel()
        resume.signal()
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let info) = event { completion = info }
        }
        let tokens = await task.value
        XCTAssertEqual(tokens, [42])
        XCTAssertEqual(completion?.stopReason, .cancelled)
        XCTAssertEqual(completion?.generationTokenCount, 1)
        await fulfillment(of: [finalized], timeout: 0)
    }

    func testCancellationDuringTicketAdmissionSkipsNextAndFinalizes() async throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "Admission requires a GPU backend")
        let finalized = expectation(description: "cancelled iterator finalized")
        let ticket = CancellingPolicy().ticket(size: 0)
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                onNext: { XCTFail("Cancelled generation must not call next") },
                onFinalize: {
                    XCTAssertTrue(Task.isCancelled)
                    finalized.fulfill()
                }),
            wiredMemoryTicket: ticket)
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let info) = event { completion = info }
        }
        let tokens = await task.value
        XCTAssertEqual(tokens, [])
        XCTAssertEqual(completion?.stopReason, .cancelled)
        await fulfillment(of: [finalized], timeout: 0)
    }

    func testConsumerCancellationReachesBlockedGeneration() async {
        let entered = expectation(description: "entered next")
        let resume = DispatchSemaphore(value: 0)
        let finalized = expectation(description: "consumer cancellation finalized")
        let (stream, task) = generateTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                onNext: {
                    entered.fulfill()
                    XCTAssertEqual(resume.wait(timeout: .now() + 5), .success)
                    XCTAssertTrue(Task.isCancelled)
                },
                onFinalize: { finalized.fulfill() }))
        let consumer = Task {
            for await _ in stream {}
        }
        await fulfillment(of: [entered], timeout: 5)
        consumer.cancel()
        await consumer.value
        resume.signal()
        await task.value
        XCTAssertTrue(task.isCancelled)
        await fulfillment(of: [finalized], timeout: 0)
    }

    @MainActor
    func testMainActorCallerDoesNotRunIteratorOnMainThread() async {
        let (stream, task) = generateTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(onNext: {
                XCTAssertFalse(Thread.isMainThread)
                Self.assertCooperativeProgress()
            }))
        for await _ in stream {}
        await task.value
    }

    func testEmptyGenerationStillFinalizes() async {
        let finalized = expectation(description: "empty iterator finalized")
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                maxTokens: 0, onNext: { XCTFail("Empty iterator must not produce a token") },
                onFinalize: { finalized.fulfill() }))
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let info) = event { completion = info }
        }
        let tokens = await task.value
        XCTAssertEqual(tokens, [])
        XCTAssertEqual(completion?.stopReason, .length)
        XCTAssertEqual(completion?.generationTokenCount, 0)
        await fulfillment(of: [finalized], timeout: 0)
    }

    func testGenerationReportsCompletionObservability() async {
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 3,
            modelConfiguration: .init(id: "test"),
            tokenizer: PseudoWordTokenizer(),
            iterator: Iterator(
                maxTokens: 2,
                evictedTokenCount: 4,
                reasoningTokenCount: 1))

        var completion: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .info(let info): completion = info
            case .chunk, .toolCall, .rejectedToolCall: break
            }
        }

        _ = await task.value
        XCTAssertEqual(completion?.evictedTokenCount, 4)
        XCTAssertEqual(completion?.reasoningTokenCount, 1)
        XCTAssertEqual(completion?.answerTokenCount, 1)
        XCTAssertTrue(completion?.summary().contains("4 context tokens dropped") == true)
    }

    func testTaskContextAndRecordedTokensSurviveExecutionHop() async {
        let finalized = expectation(description: "iterator finalized")
        let (stream, task) = Context.$value.withValue(123) {
            generateTaskRecordingTokens(
                promptTokenCount: 1, modelConfiguration: .init(id: "test"),
                tokenizer: PseudoWordTokenizer(),
                iterator: Iterator(
                    onNext: {
                        XCTAssertEqual(Context.value, 123)
                        XCTAssertTrue(withUnsafeCurrentTask { $0 != nil })
                    },
                    onFinalize: {
                        XCTAssertEqual(Context.value, 123)
                        finalized.fulfill()
                    }))
        }
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let info) = event { completion = info }
        }
        let tokens = await task.value
        XCTAssertEqual(tokens, [42])
        XCTAssertEqual(completion?.stopReason, .length)
        await fulfillment(of: [finalized], timeout: 5)
    }
}
