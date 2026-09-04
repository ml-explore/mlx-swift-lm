// Copyright © 2026 Apple Inc.

import MLX
import XCTest

@testable import MLXLMCommon

public class SlidingWindowNoRepeatNGramProcessorTests: XCTestCase {

    func testBannedTokensWhenPrefixRepeats() {
        let processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 100)
        // History ends with prefix [1,2,3]; earlier n-gram [1,2,3,4] → ban 4.
        let banned = processor.bannedTokens(in: [1, 2, 3, 4, 1, 2, 3])
        XCTAssertEqual(banned, [4])
    }

    func testWindowLimitsSearch() {
        let processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 3)
        // window=3 → searchStart = max(0, 7-3)=4, searchEnd=4 → empty.
        let banned = processor.bannedTokens(in: [1, 2, 3, 4, 1, 2, 3])
        XCTAssertTrue(banned.isEmpty)
    }

    func testTooShortHistoryIsNoOp() {
        let processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 100)
        XCTAssertTrue(processor.bannedTokens(in: [1, 2, 3]).isEmpty)
    }

    func testWhitelistExemptsBannedToken() {
        let processor = SlidingWindowNoRepeatNGramProcessor(
            ngramSize: 4, windowSize: 100, whitelistTokenIds: [4])
        let banned = processor.bannedTokens(in: [1, 2, 3, 4, 1, 2, 3])
        XCTAssertTrue(banned.isEmpty)
    }

    func testProcessSetsBannedLogitsToNegInf() {
        var processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 100)
        processor.prompt(MLXArray([1, 2, 3, 4, 1, 2, 3]))

        let logits = MLXArray([
            0.5 as Float, 1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float,
        ])[
            .newAxis, .ellipsis
        ]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 2.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.0, accuracy: 1e-6)
        XCTAssertTrue(values[4].isInfinite && values[4] < 0)
    }

    func testDidSampleExtendsHistoryForNextBan() {
        var processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 3, windowSize: 100)
        // After prompt [0,1,2] and sampling 0, history is [0,1,2,0].
        // Current prefix [2,0]; past n-gram [0,1,2] does not match.
        // Sample 1 → [0,1,2,0,1], prefix [0,1]; past [0,1,2] matches → ban 2.
        processor.prompt(MLXArray([0, 1, 2]))
        processor.didSample(token: MLXArray(0))
        processor.didSample(token: MLXArray(1))

        let logits = MLXArray([1.0 as Float, 1.0 as Float, 5.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertTrue(values[2].isInfinite && values[2] < 0)
    }

    // MARK: - Pipelining
    //
    // `TokenIterator` hands `didSample` a token that has not been evaluated yet and
    // only then calls `asyncEval`. Reading it back there (or in the next `process`)
    // would block on the in-flight forward pass every step, so both must stay lazy.

    func testDidSampleAndProcessKeepTheSampledTokenInFlight() {
        var processor = SlidingWindowNoRepeatNGramProcessor(ngramSize: 3, windowSize: 100)
        processor.prompt(MLXArray([0, 1, 2, 0]))

        // A token whose evaluation materializes a 32 MB intermediate that the test
        // keeps alive, so `activeMemory` reveals whether the token was evaluated.
        let elements = 8 * 1024 * 1024
        let probeBytes = elements * MemoryLayout<Int32>.size
        let probe = MLXArray.arange(elements, dtype: .int32)
        let token = probe.max() - Int32(elements - 2)  // == 1
        let before = Memory.activeMemory

        processor.didSample(token: token)
        let logits = MLXArray([1.0 as Float, 1.0 as Float, 5.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)

        XCTAssertLessThan(
            Memory.activeMemory - before, probeBytes,
            "didSample/process must not evaluate the sampled token")

        // Positive control: consuming the logits evaluates the token and the probe.
        eval(processed)
        XCTAssertGreaterThanOrEqual(Memory.activeMemory - before, probeBytes)
        // History [0,1,2,0,1]: prefix [0,1] completed [0,1,2] before, so 2 is banned.
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertTrue(values[2].isInfinite && values[2] < 0)
        withExtendedLifetime(probe) {}
    }

    /// Regression coverage, not a discriminating test: the GPU scan must ban exactly
    /// what the CPU reference `bannedTokens(in:)` bans, including after the window
    /// wraps and with a whitelist. A CPU implementation passes this too.
    func testGPUBanSetMatchesCPUReferenceOverLongStreams() {
        let vocabulary = 12
        let cases: [(ngram: Int, window: Int, whitelist: Set<Int>)] = [
            (2, 5, []), (3, 8, []), (4, 16, [7]), (3, 64, [1, 2]),
        ]
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextToken() -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((seed >> 33) % UInt64(vocabulary))
        }

        for testCase in cases {
            var processor = SlidingWindowNoRepeatNGramProcessor(
                ngramSize: testCase.ngram, windowSize: testCase.window,
                whitelistTokenIds: testCase.whitelist)
            var history = (0 ..< 20).map { _ in nextToken() }
            processor.prompt(MLXArray(history.map { Int32($0) }))

            for step in 0 ..< 200 {
                let logits = MLXArray.zeros([1, vocabulary])
                let values = processor.process(logits: logits)[0].asArray(Float.self)
                let bannedOnGPU = Set(values.indices.filter { values[$0] == -.infinity })
                XCTAssertEqual(
                    bannedOnGPU, processor.bannedTokens(in: history),
                    "n=\(testCase.ngram) window=\(testCase.window) step=\(step)")

                let token = nextToken()
                processor.didSample(token: MLXArray(Int32(token)))
                history.append(token)
            }
        }
    }

    func testComponentsAttachNgramProcessor() {
        let components = GenerationComponents(
            logitProcessorFactory: {
                SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 100)
            }
        )
        var processor = components.logitProcessor(parameters: GenerateParameters())
        XCTAssertNotNil(processor)

        processor?.prompt(MLXArray([1, 2, 3, 4, 1, 2, 3]))
        let logits = MLXArray([
            0.0 as Float, 0.0 as Float, 0.0 as Float, 0.0 as Float, 9.0 as Float,
        ])[
            .newAxis, .ellipsis
        ]
        let values = processor!.process(logits: logits)[0].asArray(Float.self)
        XCTAssertTrue(values[4].isInfinite && values[4] < 0)
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-6)
    }

    func testComposesWithRepetitionPenalty() {
        let components = GenerationComponents(
            logitProcessorFactory: {
                SlidingWindowNoRepeatNGramProcessor(ngramSize: 4, windowSize: 100)
            }
        )
        var processor = components.logitProcessor(
            parameters: GenerateParameters(
                repetitionPenalty: 1.5,
                repetitionContextSize: 10
            ))
        XCTAssertNotNil(processor)
        XCTAssertTrue(processor is ChainedLogitProcessor)

        processor?.prompt(MLXArray([1, 2, 3, 4, 1, 2, 3]))
        let logits = MLXArray([
            0.0 as Float, 1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float,
        ])[
            .newAxis, .ellipsis
        ]
        let values = processor!.process(logits: logits)[0].asArray(Float.self)
        // Token 4 banned by n-gram regardless of penalty.
        XCTAssertTrue(values[4].isInfinite && values[4] < 0)
    }
}
