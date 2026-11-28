// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class ToolCallBatchTests: XCTestCase {
    private struct ToolFailure: Error, Equatable {}

    /// Records tool events in the order they happen.
    private actor EventLog {
        private(set) var events: [String] = []
        private var waiters: [(String, CheckedContinuation<Void, Never>)] = []

        func record(_ event: String) {
            events.append(event)
            waiters.removeAll { waiter in
                guard waiter.0 == event else { return false }
                waiter.1.resume()
                return true
            }
        }

        func wait(for event: String) async {
            guard !events.contains(event) else { return }
            await withCheckedContinuation { waiters.append((event, $0)) }
        }
    }

    private static func call(_ name: String) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: [String: JSONValue]()), id: name)
    }

    /// Runs `body` for the named call and logs when it starts, ends, or is cancelled.
    private static func dispatch(
        _ log: EventLog, _ body: @escaping @Sendable (String) async throws -> Void = { _ in }
    ) -> ToolCallBatch.Dispatch {
        { call in
            let name = call.function.name
            await log.record("start \(name)")
            do {
                try await body(name)
            } catch is CancellationError {
                await log.record("cancel \(name)")
                throw CancellationError()
            }
            await log.record("end \(name)")
            return "result \(name)"
        }
    }

    private static func results(_ batch: consuming ToolCallBatch) async throws -> [String] {
        var results: [String] = []
        try await batch.results { results.append($0.content) }
        return results
    }

    func testDefaultPolicyRunsCallsOneAtATimeAfterTheStep() {
        XCTAssertEqual(ToolCallPolicy().dispatch, [])
    }

    func testAfterStepStartsCallsOnlyWhenResultsAreCollected() async throws {
        let log = EventLog()
        var batch = ToolCallBatch(options: [], dispatch: Self.dispatch(log))
        batch.append(Self.call("a"))
        for _ in 0 ..< 10 { await Task.yield() }
        let eventsBeforeResults = await log.events
        XCTAssertEqual(eventsBeforeResults, [])

        let results = try await Self.results(batch)
        XCTAssertEqual(results, ["result a"])
    }

    func testWhenWrittenStartsEachCallAsItArrives() async throws {
        let log = EventLog()
        var batch = ToolCallBatch(
            options: [.whenWritten], dispatch: Self.dispatch(log))
        batch.append(Self.call("a"))
        await log.wait(for: "end a")
        batch.append(Self.call("b"))
        await log.wait(for: "end b")

        let results = try await Self.results(batch)
        XCTAssertEqual(results, ["result a", "result b"])
    }

    func testSequentialCallsRunInCallOrder() async throws {
        let log = EventLog()
        var batch = ToolCallBatch(
            options: [.whenWritten],
            dispatch: Self.dispatch(log) { name in
                if name == "a" { try await Task.sleep(for: .milliseconds(20)) }
            })
        batch.append(Self.call("a"))
        batch.append(Self.call("b"))

        let results = try await Self.results(batch)
        XCTAssertEqual(results, ["result a", "result b"])
        let events = await log.events
        XCTAssertEqual(events, ["start a", "end a", "start b", "end b"])
    }

    func testConcurrentCallsOverlapAndDeliverResultsInCallOrder() async throws {
        let log = EventLog()
        // `a` finishes only after `b` has started, which needs both to run at once.
        var batch = ToolCallBatch(
            options: [.concurrent],
            dispatch: Self.dispatch(log) { name in
                if name == "a" { await log.wait(for: "end b") }
            })
        batch.append(Self.call("a"))
        batch.append(Self.call("b"))

        let results = try await Self.results(batch)
        XCTAssertEqual(results, ["result a", "result b"])
        let events = await log.events
        XCTAssertEqual(events.last, "end a")
    }

    func testFirstFailureKeepsEarlierResultsAndCancelsLaterCalls() async throws {
        let log = EventLog()
        let batch = {
            var batch = ToolCallBatch(
                options: [.concurrent],
                dispatch: Self.dispatch(log) { name in
                    switch name {
                    case "b": throw ToolFailure()
                    case "c": try await Task.sleep(for: .seconds(60))
                    default: break
                    }
                })
            for name in ["a", "b", "c"] { batch.append(Self.call(name)) }
            return batch
        }()

        var results: [String] = []
        do {
            try await batch.results { results.append($0.content) }
            XCTFail("expected the failing call to end the batch")
        } catch {
            XCTAssertEqual(error as? ToolFailure, ToolFailure())
        }
        XCTAssertEqual(results, ["result a"])
        await log.wait(for: "cancel c")
    }

    func testDroppingTheBatchCancelsRunningCalls() async {
        let log = EventLog()
        do {
            var batch = ToolCallBatch(
                options: [.whenWritten],
                dispatch: Self.dispatch(log) { _ in try await Task.sleep(for: .seconds(60)) })
            batch.append(Self.call("a"))
            await log.wait(for: "start a")
        }
        await log.wait(for: "cancel a")
    }

    func testCancelStopsRunningCallsAndStartsNoMore() async {
        let log = EventLog()
        var batch = ToolCallBatch(
            options: [.whenWritten, .concurrent],
            dispatch: Self.dispatch(log) { _ in try await Task.sleep(for: .seconds(60)) })
        batch.append(Self.call("a"))
        await log.wait(for: "start a")

        batch.cancel()
        batch.append(Self.call("b"))
        await log.wait(for: "cancel a")
        for _ in 0 ..< 10 { await Task.yield() }
        let events = await log.events
        XCTAssertEqual(events, ["start a", "cancel a"])

        do {
            _ = try await Self.results(batch)
            XCTFail("a cancelled batch has no results")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testCancellingTheCollectorCancelsTheCalls() async {
        let log = EventLog()
        let collector = Task {
            var batch = ToolCallBatch(
                options: [.whenWritten],
                dispatch: Self.dispatch(log) { _ in try await Task.sleep(for: .seconds(60)) })
            batch.append(Self.call("a"))
            await log.wait(for: "start a")
            return try await Self.results(batch)
        }
        await log.wait(for: "start a")
        collector.cancel()

        await log.wait(for: "cancel a")
        let result = await collector.result
        XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is CancellationError) }
    }
}
