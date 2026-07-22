// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

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

    func testGenerateParametersDisabledByDefault() {
        XCTAssertNil(GenerateParameters().processor())
        XCTAssertNil(GenerateParameters(noRepeatNgramSize: 0).processor())
        XCTAssertNil(GenerateParameters(noRepeatNgramSize: nil).processor())
    }

    func testGenerateParametersEnablesNgramProcessor() {
        var processor = GenerateParameters(
            noRepeatNgramSize: 4,
            noRepeatNgramWindowSize: 100
        ).processor()
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

    func testDisabledPathLeavesLogitsUnchangedWhenOnlyTemperature() {
        // Default generate params with temperature only → no processor → unchanged path.
        let parameters = GenerateParameters(temperature: 0)
        XCTAssertNil(parameters.processor())
        XCTAssertTrue(parameters.sampler() is ArgMaxSampler)
    }

    func testComposesWithRepetitionPenalty() {
        var processor = GenerateParameters(
            repetitionPenalty: 1.5,
            repetitionContextSize: 10,
            noRepeatNgramSize: 4,
            noRepeatNgramWindowSize: 100
        ).processor()
        XCTAssertNotNil(processor)
        XCTAssertTrue(processor is SequentialLogitProcessor)

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
