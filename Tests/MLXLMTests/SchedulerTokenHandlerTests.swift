// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Tokenizers
import XCTest

@testable import MLXLMCommon

// MARK: - SchedulerTokenHandler Unit Tests

/// Unit tests for `SchedulerTokenHandler` — verifies both text and raw-token
/// factory methods without requiring GPU/Metal.
class SchedulerTokenHandlerTests: XCTestCase {

    // MARK: - Text Handler

    func testTextHandlerEmitsChunks() async {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let tokenizer = TestTokenizer()

        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: tokenizer,
            toolCallFormat: .json
        )

        XCTAssertTrue(handler.processToken(5))
        XCTAssertTrue(handler.processToken(10))

        let info = GenerateCompletionInfo(
            promptTokenCount: 1,
            generationTokenCount: 2,
            promptTime: 0.01,
            generationTime: 0.02,
            stopReason: .stop
        )
        handler.yieldInfo(info)
        handler.finish()

        var chunks = [String]()
        var gotInfo = false
        for await gen in stream {
            switch gen {
            case .chunk(let text): chunks.append(text)
            case .info: gotInfo = true
            case .toolCall: break
            }
        }

        XCTAssertTrue(gotInfo, "Should receive .info event")
        // Chunks may or may not appear depending on detokenizer buffering,
        // but the stream should complete without hanging.
    }

    func testTextHandlerProcessEndOfSequenceFlushesToolCalls() async {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let tokenizer = TestTokenizer()

        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: tokenizer,
            toolCallFormat: .json
        )

        // processEndOfSequence should not crash even with no pending tool calls
        handler.processEndOfSequence()
        handler.finish()

        var events = [Generation]()
        for await gen in stream {
            events.append(gen)
        }
        // Stream should terminate cleanly
    }

    func testTextHandlerProcessStopTokenIsNoOp() {
        let (_, continuation) = AsyncStream<Generation>.makeStream()
        let tokenizer = TestTokenizer()

        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: tokenizer,
            toolCallFormat: .json
        )

        // Stop token processing should be a no-op for text mode
        XCTAssertTrue(handler.processStopToken(0))
    }

    func testTextHandlerMode() {
        let (_, continuation) = AsyncStream<Generation>.makeStream()
        let tokenizer = TestTokenizer()

        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: tokenizer,
            toolCallFormat: .json
        )

        if case .decoded = handler.mode {
            // Expected
        } else {
            XCTFail("Text handler should have .decoded mode")
        }
    }

    // MARK: - Raw Token Handler

    func testRawTokenHandlerEmitsTokens() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        XCTAssertTrue(handler.processToken(42))
        XCTAssertTrue(handler.processToken(99))

        let info = GenerateCompletionInfo(
            promptTokenCount: 1,
            generationTokenCount: 2,
            promptTime: 0.01,
            generationTime: 0.02,
            stopReason: .stop
        )
        handler.yieldInfo(info)
        handler.finish()

        var tokenIDs = [Int]()
        var gotInfo = false
        for await gen in stream {
            switch gen {
            case .token(let id): tokenIDs.append(id)
            case .info: gotInfo = true
            }
        }

        XCTAssertEqual(tokenIDs, [42, 99])
        XCTAssertTrue(gotInfo)
    }

    func testRawTokenHandlerIncludeStopTokenTrue() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: true
        )

        XCTAssertTrue(handler.processToken(10))
        // Stop token should be emitted when includeStopToken is true
        XCTAssertTrue(handler.processStopToken(0))
        handler.finish()

        var tokenIDs = [Int]()
        for await gen in stream {
            if case .token(let id) = gen {
                tokenIDs.append(id)
            }
        }

        XCTAssertEqual(tokenIDs, [10, 0], "Stop token should be included")
    }

    func testRawTokenHandlerIncludeStopTokenFalse() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        XCTAssertTrue(handler.processToken(10))
        // Stop token should NOT be emitted
        XCTAssertTrue(handler.processStopToken(0))
        handler.finish()

        var tokenIDs = [Int]()
        for await gen in stream {
            if case .token(let id) = gen {
                tokenIDs.append(id)
            }
        }

        XCTAssertEqual(tokenIDs, [10], "Stop token should NOT be included")
    }

    func testRawTokenHandlerProcessEndOfSequenceIsNoOp() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        handler.processEndOfSequence()  // Should not crash
        handler.finish()

        var events = [TokenGeneration]()
        for await gen in stream {
            events.append(gen)
        }
        XCTAssertTrue(events.isEmpty, "No events should be emitted from processEndOfSequence")
    }

    func testRawTokenHandlerMode() {
        let (_, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: true
        )

        if case .rawTokens(let includeStop) = handler.mode {
            XCTAssertTrue(includeStop)
        } else {
            XCTFail("Raw token handler should have .rawTokens mode")
        }
    }

    // MARK: - Stop Token Accounting

    /// Verifies that when `includeStopToken: true`, the stop token is included
    /// in the stream output count — matching the accounting fix in
    /// InferenceScheduler where tokenCount/generatedTokenIds must include it.
    func testRawTokenHandlerIncludeStopTokenCountsInOutput() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: true
        )

        // Verify mode allows the scheduler to gate on it
        if case .rawTokens(let includeStop) = handler.mode {
            XCTAssertTrue(includeStop)
        } else {
            XCTFail("Expected .rawTokens mode")
        }

        XCTAssertTrue(handler.processToken(10))
        XCTAssertTrue(handler.processToken(20))
        // Stop token should be emitted and counted
        XCTAssertTrue(handler.processStopToken(0))
        handler.finish()

        var allTokens = [Int]()
        for await gen in stream {
            if case .token(let id) = gen {
                allTokens.append(id)
            }
        }

        // 2 regular tokens + 1 stop token = 3 total
        XCTAssertEqual(allTokens, [10, 20, 0])
        XCTAssertEqual(allTokens.count, 3, "Stop token must be counted in output")
    }

    /// Verifies that when `includeStopToken: false`, the stop token is NOT in
    /// the stream — the scheduler should not count it in tokenCount either.
    func testRawTokenHandlerExcludeStopTokenOmitsFromOutput() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        if case .rawTokens(let includeStop) = handler.mode {
            XCTAssertFalse(includeStop)
        } else {
            XCTFail("Expected .rawTokens mode")
        }

        XCTAssertTrue(handler.processToken(10))
        XCTAssertTrue(handler.processToken(20))
        XCTAssertTrue(handler.processStopToken(0))
        handler.finish()

        var allTokens = [Int]()
        for await gen in stream {
            if case .token(let id) = gen {
                allTokens.append(id)
            }
        }

        // Only 2 regular tokens, stop token omitted
        XCTAssertEqual(allTokens, [10, 20])
        XCTAssertEqual(allTokens.count, 2, "Stop token must NOT be counted in output")
    }

    // MARK: - Cancellation

    func testOnCancellationCallbackFires() async {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        let expectation = XCTestExpectation(description: "Cancellation callback fired")

        handler.onCancellation {
            expectation.fulfill()
        }

        // Start a consumer task then cancel it — this triggers .cancelled
        let task = Task {
            for await _ in stream {}
        }
        task.cancel()

        await fulfillment(of: [expectation], timeout: 2.0)
    }
}
