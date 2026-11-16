// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private func sequentialGreedyVerification(
    logits: MLXArray, draftTokens: MLXArray
) -> (tokens: [Int], accepted: Int, finalToken: MLXArray) {
    eval(draftTokens)
    let drafts = draftTokens.asArray(Int.self)
    let sampler = ArgMaxSampler()
    var tokens = [Int]()
    var accepted = 0
    for i in 0 ... drafts.count {
        let token = sampler.sample(logits: logits[i ..< (i + 1)])
        eval(token)
        let value = token.item(Int.self)
        tokens.append(value)
        if i == drafts.count || value != drafts[i] {
            return (tokens, accepted, token)
        }
        accepted += 1
    }
    preconditionFailure("verification must produce a correction or bonus")
}

@Suite(.serialized)
struct MTPGreedyVerificationTests {
    @Test(arguments: [2, 4, 8], [DType.float32, .float16, .bfloat16])
    func matchesSequentialVerification(blockSize: Int, dtype: DType) {
        for accepted in 0 ..< blockSize {
            var data = [Float](repeating: -10, count: blockSize * 32)
            for row in 0 ..< blockSize {
                let token = row < accepted ? 7 : 9
                data[row * 32 + token] = 10
                // Equal maxima must keep the sequential argmax's tie breaking.
                data[row * 32 + token + 1] = 10
            }
            let logits = MLXArray(data, [blockSize, 32]).asType(dtype)
            let drafts = MLXArray(Array(repeating: Int32(7), count: blockSize - 1))
            let reference = sequentialGreedyVerification(logits: logits, draftTokens: drafts)
            let batched = batchedGreedyMTPVerification(logits: logits, draftTokens: drafts)

            #expect(batched.accepted == accepted)
            #expect(batched.accepted == reference.accepted)
            #expect(batched.tokens == reference.tokens)
            #expect(batched.finalToken.shape == reference.finalToken.shape)
            #expect(batched.finalToken.dtype == reference.finalToken.dtype)
            #expect(batched.finalToken.item(Int.self) == reference.finalToken.item(Int.self))
        }
    }

    @Test
    func suppressionMatchesSequentialVerification() throws {
        let suppressor = try #require(SuppressTokensProcessor(tokenIds: [7, 1000]))
        let logits = MLXArray(
            [
                Float(0), 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
            ], [4, 8])
        let drafts = MLXArray([Int32(6), 6, 7])
        let batched = batchedGreedyMTPVerification(
            logits: suppressor.process(logits: logits), draftTokens: drafts)
        let reference = sequentialGreedyVerification(
            logits: concatenated(
                (0 ..< 4).map {
                    suppressor.process(logits: logits[$0 ..< ($0 + 1)])
                }), draftTokens: drafts)
        #expect(batched.accepted == 2)
        #expect(batched.tokens == [6, 6, 6])
        #expect(batched.tokens == reference.tokens)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MTP_GREEDY_BENCHMARK"] == "1"))
    func benchmarkVerification() {
        let clock = ContinuousClock()
        let iterations = 100
        let repetitions = 5
        for blockSize in [2, 4, 8] {
            let vocabularySize = 151_936
            var data = [Float](repeating: -1, count: blockSize * vocabularySize)
            for row in 0 ..< blockSize { data[row * vocabularySize + 7] = 1 }
            let logits = MLXArray(data, [blockSize, vocabularySize])
            eval(logits)
            for accepted in [0, (blockSize - 1) / 2, blockSize - 1] {
                var values = Array(repeating: Int32(7), count: blockSize - 1)
                if accepted < values.count { values[accepted] = 9 }
                let drafts = MLXArray(values)
                eval(drafts)
                var sequentialTimes = [Double]()
                var batchedTimes = [Double]()
                var checksum = 0
                func measure(batched: Bool, iterations: Int) -> Double {
                    let start = clock.now
                    for _ in 0 ..< iterations {
                        let result =
                            batched
                            ? batchedGreedyMTPVerification(logits: logits, draftTokens: drafts)
                            : sequentialGreedyVerification(logits: logits, draftTokens: drafts)
                        checksum += result.accepted + result.tokens.count
                    }
                    let elapsed = (clock.now - start).components
                    return (Double(elapsed.seconds) * 1e6 + Double(elapsed.attoseconds) / 1e12)
                        / Double(iterations)
                }
                _ = measure(batched: false, iterations: 20)
                _ = measure(batched: true, iterations: 20)
                for repetition in 0 ..< repetitions {
                    for batched in repetition.isMultiple(of: 2) ? [false, true] : [true, false] {
                        let elapsed = measure(batched: batched, iterations: iterations)
                        if batched {
                            batchedTimes.append(elapsed)
                        } else {
                            sequentialTimes.append(elapsed)
                        }
                    }
                }
                let sequential = sequentialTimes.sorted()[repetitions / 2]
                let batched = batchedTimes.sorted()[repetitions / 2]
                #expect(checksum > 0)
                print(
                    String(
                        format:
                            "[MTPGREEDYBENCH] block=%d accepted=%d sequential=%.1f us batched=%.1f us speedup=%.2fx",
                        blockSize, accepted, sequential, batched, sequential / batched))
            }
        }
    }
}
