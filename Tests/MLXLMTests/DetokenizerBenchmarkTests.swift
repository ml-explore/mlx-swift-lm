import BenchmarkHelpers
import Foundation
import XCTest

@testable import MLXLMCommon

final class DetokenizerBenchmarkTests: XCTestCase {
    func testUnboundedDecodeWorkIsTriangularWithoutNewlines() {
        let tokenCount = 128
        let result = benchmarkStreamingDetokenization(
            tokenCounts: [tokenCount],
            newlineCadences: [nil],
            strategies: [.unbounded],
            runs: 1,
            warmupRuns: 0
        ).first!

        XCTAssertEqual(result.decodedTokenVisits, tokenCount * (tokenCount + 1) / 2)
        XCTAssertEqual(result.emittedCharacterCount, tokenCount)
    }

    func testFixedNewlineCadenceBoundsHistoricalDecodeWork() {
        let tokenCount = 128
        let cadence = 16
        let segmentCount = tokenCount / cadence
        let result = benchmarkStreamingDetokenization(
            tokenCounts: [tokenCount],
            newlineCadences: [cadence],
            strategies: [.unbounded],
            runs: 1,
            warmupRuns: 0
        ).first!

        let firstSegmentVisits = cadence * (cadence + 1) / 2 + 1
        let subsequentSegmentVisits = (cadence + 1) * (cadence + 2) / 2
        let expectedVisits =
            firstSegmentVisits + (segmentCount - 1) * subsequentSegmentVisits

        XCTAssertEqual(result.decodedTokenVisits, expectedVisits)
        XCTAssertEqual(result.emittedCharacterCount, tokenCount)
    }

    func testBoundedContextMakesDecodeWorkLinearWithoutChangingOutput() {
        let tokenCount = 512
        let contextTokens = 16
        let results = benchmarkStreamingDetokenization(
            tokenCounts: [tokenCount],
            newlineCadences: [nil],
            strategies: [.unbounded, .bounded(contextTokens: contextTokens)],
            runs: 1,
            warmupRuns: 0
        )

        let unbounded = results[0]
        let bounded = results[1]
        let expectedBoundedVisits =
            contextTokens * (contextTokens + 1) / 2
            + (tokenCount - contextTokens) * (2 * contextTokens + 1)

        XCTAssertEqual(bounded.decodedTokenVisits, expectedBoundedVisits)
        XCTAssertEqual(bounded.emittedCharacterCount, unbounded.emittedCharacterCount)
        XCTAssertLessThan(bounded.decodedTokenVisits, unbounded.decodedTokenVisits)
    }

    func testBoundedContextPreservesStreamingChunksForLocalDecoder() {
        let tokenizer = LocalDecoderTokenizer(contextSize: 2)
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var output = ""

        for token in 1 ... 100 {
            detokenizer.append(token: token)
            output += detokenizer.next() ?? ""
        }

        XCTAssertEqual(output, String(repeating: "x", count: 100))
    }

    func testHistoricalTokenizerWithoutCapabilityKeepsUnboundedFallback() {
        let recorder = HistoricalDecodeRecorder()
        let tokenizer: any Tokenizer = HistoricalTokenizer(recorder: recorder)
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        XCTAssertFalse(tokenizer is any BoundedStreamingDecodeTokenizer)

        for token in 1 ... 32 {
            detokenizer.append(token: token)
            _ = detokenizer.next()
        }

        XCTAssertEqual(recorder.maximumTokenCount, 32)
    }

    func testIncompleteUnicodeSequenceIsBufferedBeforeContextCompaction() {
        let tokenizer = IncompleteUnicodeTokenizer()
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        detokenizer.append(token: 1)
        XCTAssertNil(detokenizer.next())

        detokenizer.append(token: 2)
        XCTAssertEqual(detokenizer.next(), "é")
    }
}

private final class HistoricalDecodeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    var maximumTokenCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func record(_ count: Int) {
        lock.lock()
        value = max(value, count)
        lock.unlock()
    }
}

private struct HistoricalTokenizer: Tokenizer {
    let recorder: HistoricalDecodeRecorder

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        recorder.record(tokenIds.count)
        return String(repeating: "x", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "x" }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct LocalDecoderTokenizer: BoundedStreamingDecodeTokenizer {
    let contextSize: Int?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(repeating: "x", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "x" }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var streamingDecodeContextSize: Int? { contextSize }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct IncompleteUnicodeTokenizer: BoundedStreamingDecodeTokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds == [1] ? "\u{fffd}" : "é"
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var streamingDecodeContextSize: Int? { 1 }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
