import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class ContinuousBatchingTests: XCTestCase {

    func testBatchKVCacheMergeExtendFilterAndExtract() {
        let first = makeCache(keys: [1, 2], values: [11, 12])
        let second = makeCache(keys: [3, 4, 5], values: [13, 14, 15])

        let merged = BatchKVCache.merge([first])
        merged.extend(BatchKVCache.merge([second]))

        XCTAssertEqual(merged.size(), 3)
        XCTAssertEqual(merged.leftPadding.asArray(Int32.self), [1, 0])
        assertCache(merged.extract(0), keys: [1, 2], values: [11, 12])
        assertCache(merged.extract(1), keys: [3, 4, 5], values: [13, 14, 15])

        merged.filter(batchIndices: MLXArray([Int32(0)]))

        XCTAssertEqual(merged.size(), 2)
        XCTAssertEqual(merged.leftPadding.asArray(Int32.self), [0])
        assertCache(merged.extract(0), keys: [1, 2], values: [11, 12])
    }

    func testBatchRotatingKVCacheKeepsSlidingWindowRows() {
        let cache = BatchRotatingKVCache(maxSize: 3, leftPadding: [0, 1])
        let (prefillKeys, _) = cache.update(
            keys: MLXArray([1, 2, 3, 4, 5, 6, 7, 8] as [Float]).reshaped([2, 1, 4, 1]),
            values: MLXArray([11, 12, 13, 14, 15, 16, 17, 18] as [Float]).reshaped([2, 1, 4, 1])
        )

        XCTAssertEqual(prefillKeys.dim(2), 4)
        XCTAssertEqual(cache.size(), 4)

        _ = cache.update(
            keys: MLXArray([9, 10] as [Float]).reshaped([2, 1, 1, 1]),
            values: MLXArray([19, 20] as [Float]).reshaped([2, 1, 1, 1])
        )

        XCTAssertEqual(cache.size(), 3)
        XCTAssertEqual(cache.state[0].asArray(Float.self), [3, 4, 9, 7, 8, 10])

        let extracted = cache.extract(0)
        XCTAssertEqual(extracted.state[0].asArray(Float.self), [3, 4, 9])
    }

    func testArraysCachePreservesBatchMetadataThroughFilterAndExtend() {
        let first = ArraysCache(size: 1, leftPadding: [0, 2])
        first[0] = MLXArray([1, 2] as [Float]).reshaped([2, 1])
        let second = ArraysCache(size: 1, leftPadding: [1])
        second[0] = MLXArray([3] as [Float]).reshaped([1, 1])

        first.extend(other: second)
        XCTAssertEqual(
            first.makeMask(N: 3)?.asArray(Bool.self),
            [
                true, true, true,
                false, false, true,
                false, true, true,
            ])

        first.filter(batchIndices: MLXArray([Int32(1), Int32(2)]))
        XCTAssertEqual(
            first.makeMask(N: 3)?.asArray(Bool.self),
            [
                false, false, true,
                false, true, true,
            ])
    }

    func testArraysCacheAdvancesLengthsForChunkedPrefill() {
        let cache = ArraysCache(size: 1)
        cache.prepare(lengths: [3, 5])

        XCTAssertEqual(
            cache.makeMask(N: 2)?.asArray(Bool.self),
            [
                true, true,
                true, true,
            ])

        cache.advance(2)

        XCTAssertEqual(
            cache.makeMask(N: 3)?.asArray(Bool.self),
            [
                true, false, false,
                true, true, true,
            ])
    }

    func testSequenceStateMachineMatchesMultiTokenStopsAndTransitions() {
        let machine = SequenceStateMachine(
            states: [
                "normal": [(sequence: [4, 5], next: "afterMarker")],
                "afterMarker": [(sequence: [6], next: nil)],
            ]
        )

        var state = machine.makeState()

        var result = machine.match(state, 4)
        XCTAssertNil(result.matchedSequence)
        XCTAssertEqual(result.currentState, "normal")
        state = result.next

        result = machine.match(state, 5)
        XCTAssertEqual(result.matchedSequence, [4, 5])
        XCTAssertEqual(result.currentState, "afterMarker")
        state = result.next

        result = machine.match(state, 6)
        XCTAssertEqual(result.matchedSequence, [6])
        XCTAssertNil(result.currentState)
    }

    func testSequenceStateMachineMatchesOverlappingStopSequence() {
        let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 2], next: nil)]])
        var state = machine.makeState()

        var result = machine.match(state, 1)
        XCTAssertNil(result.matchedSequence)
        state = result.next

        result = machine.match(state, 1)
        XCTAssertNil(result.matchedSequence)
        state = result.next

        result = machine.match(state, 2)
        XCTAssertEqual(result.matchedSequence, [1, 2])
        XCTAssertNil(result.currentState)
    }

    func testRowSamplerTopKOneAlwaysSelectsBestToken() {
        let sampler = makeRowSampler(temperature: 1, topP: 1, topK: 1, seed: 7)
        let logprobs = MLXArray([0.1 as Float, 3.0 as Float, 2.0 as Float])[
            .newAxis, .ellipsis
        ]

        for _ in 0 ..< 5 {
            XCTAssertEqual(sampler(logprobs).item(Int.self), 1)
        }
    }

    func testBatchGeneratorAdmitsQueuedRowsAndReportsFinishReasons() {
        let generator = BatchGenerator(
            model: IncrementingLanguageModel(),
            eosTokens: [[5]],
            defaultMaxTokens: 4,
            prefillBatchSize: 1,
            completionBatchSize: 2
        )

        let uids = generator.insert(prompts: [[1, 2], [8]], maxTokens: [4, 2])
        XCTAssertEqual(uids, [0, 1])

        var tokensByUID: [Int: [Int]] = [:]
        var finishReasonByUID: [Int: String] = [:]
        var steps = 0

        while generator.hasWork {
            steps += 1
            XCTAssertLessThan(steps, 10)

            for response in generator.next() {
                tokensByUID[response.uid, default: []].append(response.token)
                if let finishReason = response.finishReason {
                    finishReasonByUID[response.uid] = finishReason
                }
            }
        }

        XCTAssertEqual(tokensByUID[0], [3, 4, 5])
        XCTAssertEqual(tokensByUID[1], [9, 10])
        XCTAssertEqual(finishReasonByUID[0], "stop")
        XCTAssertEqual(finishReasonByUID[1], "length")
        XCTAssertEqual(generator.promptTokensProcessed, 3)
        XCTAssertFalse(generator.hasWork)
    }

    func testBatchGeneratorCancelRemovesQueuedRequest() {
        let generator = BatchGenerator(
            model: IncrementingLanguageModel(),
            defaultMaxTokens: 3,
            prefillBatchSize: 1,
            completionBatchSize: 1
        )

        let uids = generator.insert(prompts: [[1], [8]], maxTokens: [3, 3])
        XCTAssertTrue(generator.cancel(uid: uids[1]))
        XCTAssertFalse(generator.cancel(uid: 999))

        var seenUIDs = Set<Int>()
        var steps = 0
        while generator.hasWork {
            steps += 1
            XCTAssertLessThan(steps, 10)
            for response in generator.next() {
                seenUIDs.insert(response.uid)
            }
        }

        XCTAssertEqual(seenUIDs, [uids[0]])
    }

    func testBatchGeneratorCancelRemovesActiveRequest() {
        let generator = BatchGenerator(
            model: IncrementingLanguageModel(),
            defaultMaxTokens: 4,
            prefillBatchSize: 2,
            completionBatchSize: 2
        )

        let uids = generator.insert(prompts: [[1], [8]], maxTokens: [4, 4])
        let firstStep = generator.next()
        XCTAssertEqual(Set(firstStep.map(\.uid)), Set(uids))

        XCTAssertTrue(generator.cancel(uid: uids[0]))

        var laterUIDs = Set<Int>()
        var steps = 0
        while generator.hasWork {
            steps += 1
            XCTAssertLessThan(steps, 10)
            for response in generator.next() {
                laterUIDs.insert(response.uid)
            }
        }

        XCTAssertFalse(laterUIDs.contains(uids[0]))
        XCTAssertTrue(laterUIDs.contains(uids[1]))
    }
}

private func makeCache(keys: [Float], values: [Float]) -> KVCacheSimple {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray(keys).reshaped([1, 1, keys.count, 1]),
        values: MLXArray(values).reshaped([1, 1, values.count, 1])
    )
    return cache
}

private func assertCache(
    _ cache: KVCacheSimple,
    keys expectedKeys: [Float],
    values expectedValues: [Float],
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let state = cache.state
    XCTAssertEqual(state.count, 2, file: file, line: line)
    XCTAssertEqual(state[0].asArray(Float.self), expectedKeys, file: file, line: line)
    XCTAssertEqual(state[1].asArray(Float.self), expectedValues, file: file, line: line)
}

private final class IncrementingLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let batchSize = inputs.dim(0)
        let sequenceLength = inputs.dim(1)

        if let cache {
            let keys = MLXArray.ones([batchSize, 1, sequenceLength, 1], dtype: .float32)
            let values = keys * 2
            for layerCache in cache {
                _ = layerCache.update(keys: keys, values: values)
            }
        }

        let inputTokens = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(
            repeating: Float(-1_000),
            count: batchSize * sequenceLength * vocabularySize
        )

        for batchIndex in 0 ..< batchSize {
            for tokenIndex in 0 ..< sequenceLength {
                let inputIndex = batchIndex * sequenceLength + tokenIndex
                let nextToken = (inputTokens[inputIndex] + 1) % vocabularySize
                let logitIndex =
                    (batchIndex * sequenceLength + tokenIndex) * vocabularySize + nextToken
                logits[logitIndex] = 0
            }
        }

        return MLXArray(logits).reshaped([batchSize, sequenceLength, vocabularySize])
    }
}
