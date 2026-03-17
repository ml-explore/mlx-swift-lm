// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

private final class WiredMemorySchedulerMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int
    let numLayers: Int
    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }

    init(vocabSize: Int = 64, numLayers: Int = 1) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        let batch = tokens.dim(0)
        let steps = tokens.dim(1)

        var logitsFlat = [Float]()
        logitsFlat.reserveCapacity(batch * steps * vocabSize)

        for b in 0 ..< batch {
            for s in 0 ..< steps {
                let lastToken = Int(tokens[b, s].item(Int32.self))
                let predictedToken = ((lastToken + 3) % (vocabSize - 1)) + 1

                var row = [Float](repeating: -100, count: vocabSize)
                row[predictedToken] = 0
                logitsFlat.append(contentsOf: row)
            }
        }

        return LMOutput(logits: MLXArray(logitsFlat, [batch, steps, vocabSize]))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private struct WiredMemoryMockInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    var messageGenerator: MessageGenerator { DefaultMessageGenerator() }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(promptTokens))
    }
}

private actor WiredMemoryEventRecorder {
    private var events = [WiredMemoryEvent]()

    func append(_ event: WiredMemoryEvent) {
        events.append(event)
    }

    func snapshot() -> [WiredMemoryEvent] {
        events
    }
}

private actor AsyncFlag {
    private var value = false

    func set() {
        value = true
    }

    func get() -> Bool {
        value
    }
}

final class SchedulerWiredMemoryIntegrationTests: XCTestCase {
    private func makeSchedulerParts() -> (
        scheduler: InferenceScheduler,
        model: WiredMemorySchedulerMockModel,
        tokenizer: TestTokenizer,
        configuration: ModelConfiguration
    ) {
        (
            scheduler: InferenceScheduler(),
            model: WiredMemorySchedulerMockModel(),
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "wired-memory-test-model")
        )
    }

    private func makeModelContainer(scheduler: InferenceScheduler? = nil) -> ModelContainer {
        let model = WiredMemorySchedulerMockModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "wired-memory-test-model")
        let processor = WiredMemoryMockInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration
        )

        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )

        let container = ModelContainer(context: context)
        container.scheduler = scheduler
        return container
    }

    private func makeTestManager(baseline: Int = 100) -> WiredMemoryManager {
        WiredMemoryManager.makeForTesting(
            configuration: .init(
                policyOnlyWhenUnsupported: true,
                baselineOverride: baseline,
                useRecommendedWorkingSetWhenUnsupported: false
            )
        )
    }

    private func startRecording(
        manager: WiredMemoryManager
    ) -> (WiredMemoryEventRecorder, Task<Void, Never>) {
        let recorder = WiredMemoryEventRecorder()
        let task = Task {
            for await event in await manager.events() {
                await recorder.append(event)
            }
        }
        return (recorder, task)
    }

    private func ticketEvents(
        _ events: [WiredMemoryEvent],
        ticket: WiredMemoryTicket,
        kind: WiredMemoryEvent.Kind? = nil
    ) -> [WiredMemoryEvent] {
        events.filter { event in
            event.ticketID == ticket.id && (kind == nil || event.kind == kind)
        }
    }

    private func settleEvents() async {
        try? await Task.sleep(nanoseconds: 20_000_000)
    }

    func testSchedulerSinglePathStartsAndEndsWiredMemoryTicket() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager()
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 200)
        let ticket = policy.ticket(size: 40, manager: manager)
        let parts = makeSchedulerParts()

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 4, temperature: 0)

        let stream = try await parts.scheduler.submit(
            input: input,
            parameters: params,
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket
        )

        for await _ in stream {}
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketEnded).count, 1)
    }

    func testIncompatibleSingleFallbackStartsAndEndsWiredMemoryTicket() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager()
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 200)
        let ticket = policy.ticket(size: 36, manager: manager)
        let parts = makeSchedulerParts()

        let stream = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(2), Int32(3), Int32(4)])),
            parameters: GenerateParameters(maxTokens: 4, kvBits: 4, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket
        )

        let schedulerState = await parts.scheduler.currentState
        XCTAssertEqual(schedulerState, "idle")

        for await _ in stream {}
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketEnded).count, 1)
    }

    func testModelContainerSchedulerForwardsWiredMemoryTicket() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager()
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 220)
        let ticket = policy.ticket(size: 48, manager: manager)
        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let input = LMInput(tokens: MLXArray([Int32(4), Int32(5), Int32(6)]))
        let params = GenerateParameters(maxTokens: 4, temperature: 0)

        let stream = try await container.generate(
            input: input,
            parameters: params,
            wiredMemoryTicket: ticket
        )

        for await _ in stream {}
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket, kind: .ticketEnded).count, 1)
    }

    func testUpgradeEndsEachRequestTicketOnItsOwnCompletion() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager(baseline: 120)
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 260)
        let ticket1 = policy.ticket(size: 40, manager: manager)
        let ticket2 = policy.ticket(size: 30, manager: manager)
        let parts = makeSchedulerParts()

        let stream1 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2)])),
            parameters: GenerateParameters(maxTokens: 3, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket1
        )

        let stream2 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(9), Int32(10)])),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket2
        )

        async let consume1: Void = { for await _ in stream1 {} }()
        async let consume2: Void = { for await _ in stream2 {} }()
        _ = await (consume1, consume2)
        await settleEvents()

        let events = await recorder.snapshot()
        let firstEnd = try XCTUnwrap(
            ticketEvents(events, ticket: ticket1, kind: .ticketEnded).first)
        let secondEnd = try XCTUnwrap(
            ticketEvents(events, ticket: ticket2, kind: .ticketEnded).first)

        XCTAssertEqual(ticketEvents(events, ticket: ticket1, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket2, kind: .ticketStarted).count, 1)
        XCTAssertLessThan(firstEnd.sequence, secondEnd.sequence)
    }

    func testWaitingSecondTicketDoesNotInterruptFirstRequest() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager(baseline: 100)
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 140)
        let blockerTicket = policy.ticket(size: 30, manager: manager)
        let firstTicket = policy.ticket(size: 10, manager: manager)
        let secondTicket = policy.ticket(size: 20, manager: manager)
        let parts = makeSchedulerParts()
        var blockerReleased = false
        _ = await blockerTicket.start()
        defer {
            if !blockerReleased {
                Task { _ = await blockerTicket.end() }
            }
        }

        let stream1 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 20, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: firstTicket
        )

        let secondReturned = AsyncFlag()
        let secondTask = Task<Void, Error> {
            let stream2 = try await parts.scheduler.submit(
                input: LMInput(tokens: MLXArray([Int32(11), Int32(12)])),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                model: parts.model,
                cache: nil,
                tokenizer: parts.tokenizer,
                configuration: parts.configuration,
                wiredMemoryTicket: secondTicket
            )
            await secondReturned.set()
            for await _ in stream2 {}
        }
        defer { secondTask.cancel() }

        try? await Task.sleep(nanoseconds: 50_000_000)

        let didSecondReturn = await secondReturned.get()
        XCTAssertFalse(didSecondReturn)

        let firstChunkSeen = AsyncFlag()
        let firstConsumer = Task {
            for await generation in stream1 {
                if case .chunk = generation {
                    await firstChunkSeen.set()
                }
            }
        }
        defer { firstConsumer.cancel() }

        var sawChunk = false
        for _ in 0 ..< 50 {
            if await firstChunkSeen.get() {
                sawChunk = true
                break
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(sawChunk)

        _ = await firstConsumer.value
        _ = await blockerTicket.end()
        blockerReleased = true
        _ = try await secondTask.value
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertFalse(ticketEvents(events, ticket: secondTicket, kind: .admissionWait).isEmpty)

        let firstEnd = try XCTUnwrap(
            ticketEvents(events, ticket: firstTicket, kind: .ticketEnded).first)
        let secondStart = try XCTUnwrap(
            ticketEvents(events, ticket: secondTicket, kind: .ticketStarted).first)
        XCTAssertLessThan(firstEnd.sequence, secondStart.sequence)
    }

    func testJoinedBatchRequestEndsItsOwnTicketOnCancellation() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager(baseline: 120)
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 320)
        let ticket1 = policy.ticket(size: 30, manager: manager)
        let ticket2 = policy.ticket(size: 30, manager: manager)
        let ticket3 = policy.ticket(size: 30, manager: manager)
        let parts = makeSchedulerParts()

        let stream1 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2)])),
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket1
        )

        let stream2 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(8), Int32(9)])),
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket2
        )

        let stream3 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(20), Int32(21)])),
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket3
        )

        async let stopReason1: GenerateStopReason? = {
            var stopReason: GenerateStopReason?
            for await generation in stream1 {
                if case .info(let info) = generation {
                    stopReason = info.stopReason
                }
            }
            return stopReason
        }()
        async let stopReason2: GenerateStopReason? = {
            var stopReason: GenerateStopReason?
            for await generation in stream2 {
                if case .info(let info) = generation {
                    stopReason = info.stopReason
                }
            }
            return stopReason
        }()
        async let consume3: Void = {
            var chunkCount = 0
            for await generation in stream3 {
                if case .chunk = generation {
                    chunkCount += 1
                    if chunkCount >= 2 {
                        break
                    }
                }
            }
        }()

        let (reason1, reason2, _) = await (stopReason1, stopReason2, consume3)
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertNotEqual(reason1, .cancelled)
        XCTAssertNotEqual(reason2, .cancelled)
        XCTAssertEqual(ticketEvents(events, ticket: ticket3, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket3, kind: .ticketEnded).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket1, kind: .ticketEnded).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket2, kind: .ticketEnded).count, 1)
    }

    func testDelayedJoinedBatchTicketFallsBackToSingleAfterBatchDrains() async throws {
        try skipIfMetalUnavailable()

        let manager = makeTestManager(baseline: 120)
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 160)
        let blockerTicket = policy.ticket(size: 20, manager: manager)
        let ticket1 = policy.ticket(size: 10, manager: manager)
        let ticket2 = policy.ticket(size: 10, manager: manager)
        let ticket3 = policy.ticket(size: 30, manager: manager)
        let parts = makeSchedulerParts()
        var blockerReleased = false
        _ = await blockerTicket.start()
        defer {
            if !blockerReleased {
                Task { _ = await blockerTicket.end() }
            }
        }

        let stream1 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2)])),
            parameters: GenerateParameters(maxTokens: 10, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket1
        )

        let stream2 = try await parts.scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(6), Int32(7)])),
            parameters: GenerateParameters(maxTokens: 10, temperature: 0),
            model: parts.model,
            cache: nil,
            tokenizer: parts.tokenizer,
            configuration: parts.configuration,
            wiredMemoryTicket: ticket2
        )

        let thirdReturned = AsyncFlag()
        let thirdTask = Task<Void, Error> {
            let stream3 = try await parts.scheduler.submit(
                input: LMInput(tokens: MLXArray([Int32(20), Int32(21)])),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                model: parts.model,
                cache: nil,
                tokenizer: parts.tokenizer,
                configuration: parts.configuration,
                wiredMemoryTicket: ticket3
            )
            await thirdReturned.set()
            for await _ in stream3 {}
        }
        defer { thirdTask.cancel() }

        try? await Task.sleep(nanoseconds: 50_000_000)
        let didThirdReturnBeforeDrain = await thirdReturned.get()
        XCTAssertFalse(didThirdReturnBeforeDrain)

        async let consume1: Void = { for await _ in stream1 {} }()
        async let consume2: Void = { for await _ in stream2 {} }()
        _ = await (consume1, consume2)

        _ = await blockerTicket.end()
        blockerReleased = true
        _ = try await thirdTask.value
        await settleEvents()

        let events = await recorder.snapshot()
        XCTAssertFalse(ticketEvents(events, ticket: ticket3, kind: .admissionWait).isEmpty)
        XCTAssertEqual(ticketEvents(events, ticket: ticket3, kind: .ticketStarted).count, 1)
        XCTAssertEqual(ticketEvents(events, ticket: ticket3, kind: .ticketEnded).count, 1)

        let firstEnd = try XCTUnwrap(
            ticketEvents(events, ticket: ticket1, kind: .ticketEnded).first)
        let secondEnd = try XCTUnwrap(
            ticketEvents(events, ticket: ticket2, kind: .ticketEnded).first)
        let thirdStart = try XCTUnwrap(
            ticketEvents(events, ticket: ticket3, kind: .ticketStarted).first)
        XCTAssertLessThan(max(firstEnd.sequence, secondEnd.sequence), thirdStart.sequence)
    }
}
