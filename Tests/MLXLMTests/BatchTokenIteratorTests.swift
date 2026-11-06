// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon

/// Batched generation must give every prompt the tokens it gets generated alone.
@Suite(.serialized)
struct BatchTokenIteratorTests {

    static let prompts: [[Int]] = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]

    @Test(
        "Admission drains prepare and remainder autoreleases before next returns",
        arguments: BatchTextOnlyPreparation.allCases)
    func admissionAutoreleases(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            try autoreleasepool {
                let log = BatchAdmissionAutoreleaseLog()
                let model = BatchTextOnlyModel(
                    preparation: preparation,
                    prepareObserver: { _ in log.enqueue("prepare") },
                    forwardObserver: { log.enqueue("forward") })
                var iterator = BatchTokenIterator(model: model, maxTokens: 2)
                let uids = iterator.insert(prompts: [[1], [3]])

                let responses = try iterator.next()
                #expect(responses.map(\.uid) == uids)
                #expect(responses.map(\.token) == [2, 4])
                #expect(log.created.filter { $0 == "prepare" }.count == 2)
                #expect(log.created.filter { $0 == "forward" }.count == 3)
                #expect(log.released.sorted() == log.created.sorted())
            }
        }
    }

    @Test("Throwing admission drains its autoreleases while leaving the request queued")
    func rejectedAdmissionAutoreleases() throws {
        try Device.withDefaultDevice(.cpu) {
            try autoreleasepool {
                let log = BatchAdmissionAutoreleaseLog()
                let model = BatchTextOnlyModel(
                    preparation: .tokens,
                    prepareObserver: { _ in
                        log.enqueue("prepare")
                        throw BatchAdmissionProbeError.rejected
                    })
                var iterator = BatchTokenIterator(model: model)
                let uids = iterator.insert(prompts: [[1]])

                do {
                    _ = try iterator.next()
                    Issue.record("Expected prompt preparation to throw")
                } catch BatchAdmissionProbeError.rejected {
                }
                #expect(iterator.uids == uids)
                #expect(log.created == ["prepare"])
                #expect(log.released == log.created)
            }
        }
    }

    @Test(
        "Recurrent admission preserves the longest legacy prefix regardless of row order",
        arguments: [[3, 7], [7, 3]])
    func recurrentAdmissionLegacyOffsets(offsets: [Int]) throws {
        try Device.withDefaultDevice(.cpu) {
            let prefixes = offsets.map { offset in
                let cache = ArraysCache(size: 2)
                cache.offset = offset
                cache[0] = MLXArray([Float(offset)]).reshaped([1, 1])
                return cache
            }
            let model = BatchLegacyOffsetModel()
            var iterator = BatchTokenIterator(model: model, maxTokens: 2)
            let uids = iterator.insert(prompts: [[1], [2]], caches: prefixes.map { [$0] })

            let first = try iterator.next()
            let second = try iterator.next()
            #expect(first.map(\.uid) == uids)
            #expect(first.map(\.token) == [0, 0])
            #expect(second.map(\.token) == [7, 7])
            #expect(model.decodeOffsets == [7, 7])
            #expect(prefixes.map(\.offset) == offsets)
            for (prefix, offset) in zip(prefixes, offsets) {
                #expect(try #require(prefix[0]).asArray(Float.self) == [Float(offset)])
                #expect(prefix[1] == nil)
            }
        }
    }

    @Test(
        "Incompatible admission preserves active caches, queued uids and the seeded stream",
        arguments: CacheAdmissionMismatch.allCases)
    func incompatibleAdmission(mismatch: CacheAdmissionMismatch) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .tokens, cacheFactory: mismatch.active)
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler, maxTokens: 6)
            let active = iterator.insert(prompts: [[1, 3]])
            var tokens = try iterator.next().map(\.token)
            let activeCaches = model.lastCache
            let saved = activeCaches.map { cache in
                cache.state.map { $0.asType(.float32).asArray(Float.self) }
            }
            let calls = sampler.calls
            let prepares = model.prepareCacheOffsets.count
            let queued = iterator.insert(prompts: [[7]], caches: [mismatch.incoming()])

            #expect(throws: KVCacheError.self) { _ = try iterator.next() }
            #expect(iterator.uids == active + queued)
            #expect(sampler.calls == calls, "rejection must not draw from the shared random state")
            #expect(
                activeCaches.map { cache in
                    cache.state.map { $0.asType(.float32).asArray(Float.self) }
                } == saved)
            if mismatch == .layerCount {
                #expect(model.prepareCacheOffsets.count == prepares)
            }

            iterator.remove(queued)
            tokens += try collect(&iterator, uids: active)[0]
            let expected = try batchGenerate(
                model: BatchTextOnlyModel(preparation: .tokens, cacheFactory: mismatch.active),
                prompts: [[1, 3]], maxTokens: 6, sampler: AdmissionCountingSampler())
            #expect(tokens == expected[0])
        }
    }

    @Test("Queued rows with different layer counts fail before either prompt is prefilled")
    func pendingLayerCounts() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .tokens)
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler)
            let uids = iterator.insert(
                prompts: [[1], [2]],
                caches: [[KVCacheSimple()], [KVCacheSimple(), KVCacheSimple()]])
            #expect(throws: KVCacheError.self) { _ = try iterator.next() }
            #expect(iterator.uids == uids)
            #expect(model.prepareCacheOffsets.isEmpty)
            #expect(sampler.calls == 0)
            iterator.remove([uids[1]])
            let responses = try iterator.next()
            #expect(responses.map(\.uid) == [uids[0]])
        }
    }

    @Test(
        "Queued rows that cannot join each other fail before sampling or touching active rows",
        arguments: [false, true])
    func incompatibleQueuedRows(withActiveRows: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(
                preparation: .tokens, cacheFactory: { [RotatingKVCache(maxSize: 16)] })
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler, maxTokens: 6)
            var active: [Int] = []
            if withActiveRows {
                active = iterator.insert(prompts: [[1, 3]])
                _ = try iterator.next()
            }
            let calls = sampler.calls
            // The first queued row matches the active rows; the second's window matches neither.
            let queued = iterator.insert(
                prompts: [[7], [9]],
                caches: [[RotatingKVCache(maxSize: 16)], [RotatingKVCache(maxSize: 8)]])

            #expect(throws: KVCacheError.self) { _ = try iterator.next() }
            #expect(iterator.uids == active + queued)
            #expect(sampler.calls == calls, "rejection must not draw from the shared random state")

            iterator.remove([queued[1]])
            let responses = try iterator.next()
            #expect(responses.map(\.uid) == active + [queued[0]])
        }
    }

    @Test("Supplied cache copies cannot erase custom types, including nested children")
    func suppliedCacheCopyTypes() throws {
        try Device.withDefaultDevice(.cpu) {
            class CustomArraysCache: ArraysCache {}
            let caches: [KVCache] = [
                CustomArraysCache(size: 2), CacheList(KVCacheSimple(), CustomArraysCache(size: 2)),
            ]
            for cache in caches {
                let model = BatchTextOnlyModel(preparation: .tokens)
                let sampler = AdmissionCountingSampler()
                var iterator = BatchTokenIterator(model: model, sampler: sampler)
                let uids = iterator.insert(prompts: [[1]], caches: [[cache]])
                #expect(throws: KVCacheError.self) { _ = try iterator.next() }
                #expect(iterator.uids == uids)
                #expect(model.prepareCacheOffsets.isEmpty)
                #expect(sampler.calls == 0)
            }
        }
    }

    @Test(
        "Text-input-only models batch through nextTokenLogits",
        arguments: BatchTextOnlyPreparation.allCases)
    func textInputEntryPoint(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: preparation)
            let expected = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 4)
            }
            let callsBefore = model.nextTokenCalls
            #expect(
                try batchGenerate(model: model, prompts: Self.prompts, maxTokens: 4) == expected)
            #expect(model.nextTokenCalls > callsBefore)
        }
    }

    @Test(
        "A single row carries arbitrary model state from prefill into decode",
        arguments: BatchTextOnlyPreparation.allCases)
    func singleRowModelState(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: preparation, stateful: true)
            let actual = try batchGenerate(model: model, prompts: [[1, 3]], maxTokens: 4)
            #expect(actual == [[4, 6, 9, 13]])
            let expected = try generateAlone(
                model, prompt: [1, 3], sampler: ArgMaxSampler(), maxTokens: 4)
            #expect(actual[0] == expected)
        }
    }

    @Test("Rejected stateful admission leaves the active row and queued uid intact")
    func statefulAdmissionPreservesActiveRow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .logits, stateful: true)
            var iterator = BatchTokenIterator(model: model, maxTokens: 5)
            let active = iterator.insert(prompts: [[1, 3]])
            var tokens = try iterator.next().map(\.token)
            let queued = iterator.insert(prompts: [[7]])
            #expect(throws: BatchTokenIterator.StateError.modelStateRequiresSingleRow) {
                _ = try iterator.next()
            }
            #expect(iterator.uids == active + queued)
            iterator.remove(queued)
            tokens += try collect(&iterator, uids: active)[0]
            #expect(tokens == [4, 6, 9, 13, 2])

            let removed = iterator.insert(prompts: [[5]])
            #expect(try iterator.next().map(\.token) == [6])
            iterator.remove(removed)
            _ = iterator.insert(prompts: [[8]])
            #expect(try iterator.next().map(\.token) == [9])
        }
    }

    @Test(
        "Stateful multirow prefill is rejected without discarding queued rows",
        arguments: BatchTextOnlyPreparation.allCases)
    func statefulPrefillRefusesMultipleRows(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: preparation, stateful: true)
            var iterator = BatchTokenIterator(model: model, maxTokens: 3)
            let uids = iterator.insert(prompts: [[1, 3], [7]])
            #expect(throws: BatchTokenIterator.StateError.modelStateRequiresSingleRow) {
                _ = try iterator.next()
            }
            #expect(iterator.uids == uids)
            iterator.remove([uids[1]])
            let tokens = try collect(&iterator, uids: [uids[0]])
            #expect(tokens == [[4, 6, 9]])
        }
    }

    @Test(
        "Rejected admission retries a supplied prefix without prefilling it twice",
        arguments: BatchTextOnlyPreparation.allCases)
    func rejectedAdmissionPreservesSuppliedCache(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let prefix = KVCacheSimple()
            let keys = MLXArray([Float(4), 5]).reshaped([1, 1, 2, 1])
            _ = prefix.update(keys: keys, values: keys)
            let model = BatchTextOnlyModel(preparation: preparation, stateful: true)
            let log = BatchPrefillEventLog()
            var prefill = PrefillParameters()
            prefill.prefix = { log.events.append(.prefix($0)) }
            var iterator = BatchTokenIterator(model: model, maxTokens: 3, prefill: prefill)
            let uids = iterator.insert(prompts: [[1, 3], [7]], caches: [[prefix], nil])
            #expect(throws: BatchTokenIterator.StateError.modelStateRequiresSingleRow) {
                _ = try iterator.next()
            }
            #expect(prefix.offset == 2)
            #expect(maxAbsDifference(try #require(prefix.state.first), keys) == 0)
            #expect(model.prepareCacheOffsets == [2])
            #expect(log.events == [.prefix(2)])

            iterator.remove([uids[1]])
            let tokens = try collect(&iterator, uids: [uids[0]])
            #expect(tokens == [[4, 6, 9]])
            #expect(model.prepareCacheOffsets == [2, 2])
            #expect(prefix.offset == 2)
            #expect(log.events == [.prefix(2), .prefix(2)])
        }
    }

    @Test("State returned during a batched forward invalidates the batch until all rows leave")
    func statefulBatchOutputRequiresReset() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .tokens, emitsBatchState: true)
            var iterator = BatchTokenIterator(model: model)
            let uids = iterator.insert(prompts: [[1, 3], [7]])
            #expect(throws: BatchTokenIterator.StateError.invalidatedByModelState) {
                _ = try iterator.next()
            }
            iterator.remove([uids[0]])
            #expect(throws: BatchTokenIterator.StateError.invalidatedByModelState) {
                _ = try iterator.next()
            }
            iterator.remove(iterator.uids)
            _ = iterator.insert(prompts: [[7]])
            let tokens = try iterator.next().map(\.token)
            #expect(tokens == [8])
        }
    }

    @Test("Greedy batched generation matches each prompt generated alone")
    func greedy() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let expected = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 8)
            }
            let actual = try batchGenerate(model: model, prompts: Self.prompts, maxTokens: 8)
            #expect(actual == expected)
        }
    }

    @Test("Rows with their own seeded samplers draw the tokens they draw alone")
    func seededPerRow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            func sampler(_ row: Int) -> TopPSampler {
                TopPSampler(temperature: 0.8, topP: 0.9, seed: UInt64(7 + row))
            }
            let expected = try Self.prompts.indices.map { row in
                try generateAlone(
                    model, prompt: Self.prompts[row], sampler: sampler(row), maxTokens: 8)
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 8)
            let uids = iterator.insert(
                prompts: Self.prompts, samplers: Self.prompts.indices.map { sampler($0) })
            let actual = try collect(&iterator, uids: uids)
            #expect(actual == expected)
        }
    }

    @Test("The controlled top-p boundary distinguishes float16 from promoted logits")
    func topPPrecisionBoundary() {
        Device.withDefaultDevice(.cpu) {
            let logits = zeros([3], dtype: .float16)
            let native = cumsum(exp(logSoftmax(logits)), axis: -1).asArray(Float.self)
            let promoted = cumsum(exp(logSoftmax(logits.asType(.float32))), axis: -1)
                .asArray(Float.self)
            // At topP 0.3334, native float16 keeps one token; float32 keeps two.
            #expect(native[1] < Float(0.6666))
            #expect(promoted[1] > Float(0.6666))
        }
    }

    @Test(
        "Per-row seeded top-p preserves standalone precision for ragged prompts",
        arguments: [DType.float16, .float32, .bfloat16])
    func seededPerRowPreservesPrecision(dtype: DType) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchUniformLogitModel(dtype: dtype)
            let expected = try Self.prompts.indices.map { row in
                try generateAlone(
                    model, prompt: Self.prompts[row], sampler: boundarySampler(row), maxTokens: 24)
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 24)
            let uids = iterator.insert(
                prompts: Self.prompts, samplers: Self.prompts.indices.map { boundarySampler($0) })
            let actual = try collect(&iterator, uids: uids)
            #expect(actual == expected)
        }
    }

    @Test(
        "Admitted rows preserve each seeded top-p stream across a batch size change",
        arguments: [DType.float16, .float32, .bfloat16])
    func seededAdmissionPreservesPrecision(dtype: DType) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchUniformLogitModel(dtype: dtype)
            let expected = try Self.prompts.indices.map { row in
                try generateAlone(
                    model, prompt: Self.prompts[row], sampler: boundarySampler(row), maxTokens: 24)
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 24)
            let first = iterator.insert(
                prompts: [Self.prompts[0]], samplers: [boundarySampler(0)])
            var firstTokens = try iterator.next().map(\.token)
            firstTokens += try iterator.next().map(\.token)
            let admitted = iterator.insert(
                prompts: Array(Self.prompts.dropFirst()),
                samplers: [boundarySampler(1), boundarySampler(2)])
            var actual = try collect(&iterator, uids: first + admitted)
            actual[0] = firstTokens + actual[0]
            #expect(actual == expected)
        }
    }

    @Test("One shared seeded sampler draws the whole batch, reproducibly")
    func sharedSampler() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            func run() throws -> [[Int]] {
                try batchGenerate(
                    model: model, prompts: Self.prompts, maxTokens: 6,
                    sampler: TopPSampler(temperature: 0.8, topP: 0.9, seed: 3))
            }
            let first = try run()
            #expect(first.allSatisfy { $0.count == 6 })
            #expect(try run() == first)
        }
    }

    @Test("Rows leave on a stop token or at their limit, and rows join while others decode")
    func rowsJoinAndLeave() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let alone = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 6)
            }
            // The third token of the first prompt ends that row.
            let stop = alone[0][2]
            var iterator = BatchTokenIterator(model: model, stopTokens: [stop], maxTokens: 6)
            let first = iterator.insert(prompts: [Self.prompts[0], Self.prompts[1]])
            var steps: [[BatchTokenIterator.Response]] = []
            steps.append(try iterator.next())
            steps.append(try iterator.next())
            let third = iterator.insert(prompts: [Self.prompts[2]], maxTokens: [3])
            #expect(iterator.uids == first + third)
            while true {
                let responses = try iterator.next()
                if responses.isEmpty { break }
                steps.append(responses)
            }

            func tokens(_ uid: Int) -> [Int] {
                steps.flatMap { $0.filter { $0.uid == uid } }.map(\.token)
            }
            func reasons(_ uid: Int) -> [BatchTokenIterator.FinishReason?] {
                steps.flatMap { $0.filter { $0.uid == uid } }.map(\.finishReason)
            }
            #expect(tokens(first[0]) == Array(alone[0].prefix(3)))
            #expect(reasons(first[0]) == [nil, nil, .stop])
            #expect(tokens(first[1]) == alone[1])
            #expect(reasons(first[1]) == [nil, nil, nil, nil, nil, .length])
            #expect(tokens(third[0]) == Array(alone[2].prefix(3)))
            #expect(reasons(third[0]) == [nil, nil, .length])
            #expect(steps[2].map(\.uid) == first + third)
            #expect(steps[3].map(\.uid) == [first[1], third[0]])
            #expect(iterator.uids.isEmpty)
        }
    }

    @Test("A removed row leaves the batch, queued or decoding")
    func remove() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            var iterator = BatchTokenIterator(model: model, maxTokens: 4)
            let uids = iterator.insert(prompts: Self.prompts)
            iterator.remove([uids[1]])
            #expect(iterator.uids == [uids[0], uids[2]])
            _ = try iterator.next()
            iterator.remove([uids[0]])
            #expect(iterator.uids == [uids[2]])
            let responses = try iterator.next()
            #expect(responses.map(\.uid) == [uids[2]])
        }
    }

    @Test("A row continues from its prompt cache")
    func promptCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let prefix: [Int] = [1, 2, 3, 4]
            let suffix: [Int] = [5, 6]
            let alone = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: alone))
            let expected = try generateAlone(
                model, prompt: suffix, cache: alone, sampler: ArgMaxSampler(), maxTokens: 6)

            let cache = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: cache))
            let actual = try batchGenerate(
                model: model, prompts: [suffix, Self.prompts[1]], caches: [cache, nil],
                maxTokens: 6)
            #expect(actual[0] == expected)
            let second = try generateAlone(
                model, prompt: Self.prompts[1], sampler: ArgMaxSampler(), maxTokens: 6)
            #expect(actual[1] == second)
        }
    }

    @Test("Logit processors run per row")
    func processors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let parameters = GenerateParameters(repetitionPenalty: 1.5, repetitionContextSize: 8)
            let expected = try Self.prompts.map { prompt in
                try generateAlone(
                    model, prompt: prompt, processor: parameters.processor(),
                    sampler: ArgMaxSampler(), maxTokens: 8)
            }
            let actual = try batchGenerate(
                model: model, prompts: Self.prompts, maxTokens: 8,
                processor: parameters.processor())
            #expect(actual == expected)
        }
    }

    @Test("A ragged batch of a recurrent model starts every row like the prompt alone")
    func recurrentRows() throws {
        // On the default device: the state-space scan is a Metal kernel. The first token of
        // each row comes from its own prefill, so it must match exactly; later steps run a
        // two-row scan whose arithmetic differs from the one-row scan.
        let model = try tinyMamba2()
        let prompts: [[Int]] = [[1, 2, 3], [4, 5, 6, 7, 8]]
        let expected = try prompts.map { prompt in
            try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 4)
        }
        let actual = try batchGenerate(model: model, prompts: prompts, maxTokens: 4)
        #expect(actual.map(\.count) == [4, 4])
        #expect(actual.map { $0[0] } == expected.map { $0[0] })
    }

    @Test("Ragged rows of a hybrid attention-recurrent model join while others decode")
    func hybridRowsJoinMidDecode() throws {
        // On the default device, like `recurrentRows`: each row's first token comes from its
        // own prefill and must match the prompt alone; later steps run a wider scan.
        let model = try tinyFalconH1()
        let prompts: [[Int]] = [[1, 2, 3, 4, 5, 6], [7, 8], [9, 10, 11]]
        let expected = try prompts.map { prompt in
            try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 5)
        }

        var iterator = BatchTokenIterator(model: model, maxTokens: 5)
        var uids = iterator.insert(prompts: Array(prompts[0 ..< 2]))
        var tokens = [Int: [Int]]()
        for _ in 0 ..< 2 {
            for response in try iterator.next() {
                tokens[response.uid, default: []].append(response.token)
            }
        }
        uids += iterator.insert(prompts: [prompts[2]])
        while true {
            let responses = try iterator.next()
            if responses.isEmpty { break }
            for response in responses {
                tokens[response.uid, default: []].append(response.token)
            }
        }
        let actual = uids.map { tokens[$0] ?? [] }
        #expect(actual.map(\.count) == [5, 5, 5])
        #expect(actual.map { $0[0] } == expected.map { $0[0] })
    }

    @Test("A sampler from outside the library gets each row's raw logits")
    func customSamplerGetsRawLogits() throws {
        // The first token with a positive logit: all log-probabilities are negative, so a
        // sampler handed those instead would always pick token zero.
        struct FirstPositiveSampler: LogitSampler {
            func sample(logits: MLXArray) -> MLXArray {
                argMax(logits .> 0, axis: -1)
            }
        }
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let sampler = FirstPositiveSampler()
            let expected = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: sampler, maxTokens: 6)
            }
            #expect(expected.contains { $0.contains { $0 != 0 } }, "the model has positive logits")
            // One sampler shared by every row, and one per row.
            #expect(
                try batchGenerate(
                    model: model, prompts: Self.prompts, maxTokens: 6, sampler: sampler)
                    == expected)
            var iterator = BatchTokenIterator(model: model, maxTokens: 6)
            let uids = iterator.insert(
                prompts: Self.prompts, samplers: Self.prompts.map { _ in FirstPositiveSampler() })
            #expect(try collect(&iterator, uids: uids) == expected)
        }
    }

    @Test("A one-token prompt joins a longer one on a sliding-window model")
    func mixedLengthRotatingRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
            let prompts: [[Int]] = [[1], [2, 3]]
            let expected = try prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 4)
            }
            let actual = try batchGenerate(model: model, prompts: prompts, maxTokens: 4)
            #expect(actual == expected)
        }
    }

    @Test(
        "Each queued prompt reports its cached prefix before prepare and progress",
        arguments: BatchTextOnlyPreparation.allCases)
    func prefillPrefixOrdering(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let prefix = KVCacheSimple()
            let keys = MLXArray([Float(8), 9]).reshaped([1, 1, 2, 1])
            _ = prefix.update(keys: keys, values: keys)
            let log = BatchPrefillEventLog()
            let model = BatchTextOnlyModel(
                preparation: preparation, prepareObserver: { _ in log.events.append(.prepare) })
            var prefill = PrefillParameters()
            prefill.prefix = { log.events.append(.prefix($0)) }
            prefill.progress = { log.events.append(.progress($0, $1)) }
            var iterator = BatchTokenIterator(model: model, maxTokens: 2, prefill: prefill)
            _ = iterator.insert(prompts: [[1, 2, 3], [4], [5, 6]], caches: [nil, [prefix], nil])

            _ = try iterator.next()
            let expected: [BatchPrefillEvent] = [
                .prefix(0), .prepare, .progress(3, 3),
                .prefix(2), .prepare, .progress(1, 1),
                .prefix(0), .prepare, .progress(2, 2),
            ]
            #expect(log.events == expected)
            _ = try iterator.next()
            #expect(log.events == expected, "decode must not report another prefill prefix")
            #expect(prefix.offset == 2)
        }
    }

    @Test(
        "Rotating prefixes report logical positions beyond their retained window",
        arguments: BatchTextOnlyPreparation.allCases)
    func prefillPrefixUsesLogicalRotatingOffset(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let prefix = RotatingKVCache(maxSize: 4)
            for position in 0 ..< 13 {
                let keys = MLXArray(Float(position)).reshaped([1, 1, 1, 1])
                _ = prefix.update(keys: keys, values: keys)
            }
            #expect(prefix.offset == 13)
            #expect(try #require(prefix.state.first).dim(2) == 4)
            let log = BatchPrefillEventLog()
            let model = BatchTextOnlyModel(
                preparation: preparation, prepareObserver: { _ in log.events.append(.prepare) })
            var prefill = PrefillParameters()
            prefill.prefix = { log.events.append(.prefix($0)) }
            prefill.progress = { log.events.append(.progress($0, $1)) }
            var iterator = BatchTokenIterator(model: model, maxTokens: 1, prefill: prefill)
            _ = iterator.insert(prompts: [[1, 2]], caches: [[prefix]])

            _ = try iterator.next()

            #expect(log.events == [.prefix(13), .prepare, .progress(2, 2)])
            #expect(prefix.offset == 13)
        }
    }

    @Test(
        "Recurrent and nested prefixes follow the canonical storage position inference",
        arguments: BatchPrefixCacheKind.allCases)
    func prefillPrefixUsesCanonicalInference(kind: BatchPrefixCacheKind) {
        Device.withDefaultDevice(.cpu) {
            let caches = kind.makeCaches()
            #expect(
                KVCacheStorage(caches, plan: .disabled).processedTokenCount == kind.expectedCount)
            let log = BatchPrefillEventLog()
            let model = BatchTextOnlyModel(
                preparation: .tokens,
                prepareObserver: { _ in
                    log.events.append(.prepare)
                    throw BatchPrefillObservationError.stopped
                })
            var prefill = PrefillParameters()
            prefill.prefix = { log.events.append(.prefix($0)) }
            var iterator = BatchTokenIterator(model: model, prefill: prefill)
            _ = iterator.insert(prompts: [[1]], caches: [caches])

            #expect(throws: BatchPrefillObservationError.stopped) { _ = try iterator.next() }
            #expect(log.events == [.prefix(kind.expectedCount), .prepare])
        }
    }

    @Test("Prefill progress ends at (total, total) for short and chunked prompts alike")
    func prefillProgressCompletes() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            final class Log: @unchecked Sendable { var events: [[Int]] = [] }
            let log = Log()
            var prefill = PrefillParameters(stepSize: 8)
            prefill.progress = { log.events.append([$0, $1]) }
            let prompts: [[Int]] = [
                [1, 2, 3], Array(repeating: [4, 5, 6, 7], count: 5).flatMap { $0 },
            ]

            _ = try batchGenerate(model: model, prompts: prompts, maxTokens: 1, prefill: prefill)

            // Prompts prefill in order: the short one reports only its completion, the
            // chunked one its chunks and then its completion, all against its own total.
            let short = log.events.prefix { $0[1] == 3 }
            let long = log.events.dropFirst(short.count)
            #expect(Array(short) == [[3, 3]])
            #expect(long.count > 1)
            #expect(long.allSatisfy { $0[1] == 20 })
            #expect(long.last == [20, 20])
            #expect(long.map { $0[0] } == long.map { $0[0] }.sorted())
        }
    }

    @Test(
        "Top-k filters the vocabulary axis in every supported logits shape",
        arguments: [
            (shape: [4], tokens: [1]),
            (shape: [2, 4], tokens: [1, 3]),
            (shape: [2, 3, 4], tokens: [1, 3, 0, 2, 1, 3]),
        ])
    func topKLogProbabilityShapes(shape: [Int], tokens: [Int]) {
        Device.withDefaultDevice(.cpu) {
            let rows: [[Float]] = [
                [0, 5, 1, -2], [-2, 1, 0, 5], [5, 0, -2, 1],
                [1, -2, 5, 0], [0, 5, 1, -2], [-2, 1, 0, 5],
            ]
            let logits = MLXArray(rows.prefix(tokens.count).flatMap { $0 }).reshaped(shape)
            let sampler = TopPSampler(temperature: 1, topK: 1, seed: 7)
            let normalized = sampler.sample(logProbabilities: logSoftmax(logits, axis: -1))
            let raw = sampler.sample(logits: logits)

            #expect(normalized.shape == Array(shape.dropLast()))
            #expect(normalized.asArray(Int.self) == tokens)
            #expect(raw.shape == Array(shape.dropLast()))
            #expect(raw.asArray(Int.self) == tokens)
        }
    }

    @Test("Top-p keeps the top token when its threshold would remove every token")
    func topPKeepsTopToken() {
        // 1 - 1e-9 rounds to 1 in float32, which no cumulative probability exceeds.
        let sampler = TopPSampler(temperature: 1, topP: 1e-9, seed: 1)
        let logits = MLXArray([Float(-1000), -1000, 0, -1000])[.newAxis]
        for _ in 0 ..< 4 {
            #expect(sampler.sample(logits: logits).item(Int.self) == 2)
        }
        let batch = MLXArray([Float(-1000), 0, -1000, 0, -1000, -1000]).reshaped([2, 3])
        #expect(
            sampler.sample(logProbabilities: logSoftmax(batch, axis: -1)).asArray(Int.self)
                == [1, 0])
    }
}

private func boundarySampler(_ row: Int) -> TopPSampler {
    TopPSampler(temperature: 1, topP: 0.3334, seed: UInt64(41 + row))
}

private final class BatchLegacyOffsetModel: Module, LanguageModel {
    private(set) var decodeOffsets: [Int] = []

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: zeros([1, 1, 16])))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let offset = cache?.first?.offset ?? 0
        decodeOffsets.append(offset)
        let logits = MLX.where(
            MLXArray.arange(16) .== offset, MLXArray(Float(5)), MLXArray(Float(-5)))
        return LMOutput(logits: broadcast(logits, to: [input.tokens.dim(0), 1, 16]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [ArraysCache(size: 2)] }
}

/// A rounding-sensitive nucleus without model or attention noise.
private final class BatchUniformLogitModel: Module, LanguageModel {
    private let dtype: DType

    init(dtype: DType) {
        self.dtype = dtype
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        let keys = tokens.asType(.float32).reshaped([tokens.dim(0), 1, tokens.dim(1), 1])
        _ = cache?.first?.update(keys: keys, values: keys)
        return LMOutput(logits: zeros([tokens.dim(0), tokens.dim(1), 3], dtype: dtype))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
}

private enum BatchAdmissionProbeError: Error {
    case rejected
}

private final class BatchAdmissionAutoreleaseLog {
    var created: [String] = []
    var released: [String] = []

    func enqueue(_ stage: String) {
        created.append(stage)
        let probe = BatchAdmissionAutoreleaseProbe { self.released.append(stage) }
        _ = Unmanaged.passRetained(probe).autorelease()
    }
}

private final class BatchAdmissionAutoreleaseProbe: NSObject {
    private let onRelease: () -> Void

    init(onRelease: @escaping () -> Void) {
        self.onRelease = onRelease
        super.init()
    }

    deinit { onRelease() }
}

enum BatchTextOnlyPreparation: Sendable, CaseIterable {
    case tokens
    case logits
}

private enum BatchPrefillEvent: Equatable {
    case prefix(Int)
    case prepare
    case progress(Int, Int)
}

private final class BatchPrefillEventLog: @unchecked Sendable {
    var events: [BatchPrefillEvent] = []
}

private enum BatchPrefillObservationError: Error, Equatable {
    case stopped
}

enum BatchPrefixCacheKind: Sendable, CaseIterable {
    case recurrent
    case nestedRecurrent
    case hybrid

    var expectedCount: Int {
        switch self {
        case .recurrent, .nestedRecurrent: 7
        case .hybrid: 11
        }
    }

    func makeCaches() -> [KVCache] {
        let arrays = ArraysCache(size: 2)
        let mamba = MambaCache()
        arrays.offset = 7
        mamba.offset = 7
        switch self {
        case .recurrent:
            return [arrays, mamba]
        case .nestedRecurrent:
            return [CacheList(arrays, CacheList(mamba))]
        case .hybrid:
            let attention = KVCacheSimple()
            let keys = MLXArray.arange(11).asType(.float32).reshaped([1, 1, 11, 1])
            _ = attention.update(keys: keys, values: keys)
            return [CacheList(arrays, CacheList(attention, mamba))]
        }
    }
}

/// The array convenience overload remains unimplemented, as on stateful VLM adapters.
private final class BatchTextOnlyModel: Module, LanguageModel {
    private static let counterKey = LMOutput.Key<Int>("batch-test.counter")
    private let preparation: BatchTextOnlyPreparation
    private let stateful: Bool
    private let emitsBatchState: Bool
    private let cacheFactory: (() -> [KVCache])?
    private let prepareObserver: (([KVCache]) throws -> Void)?
    private let forwardObserver: (() -> Void)?
    private(set) var nextTokenCalls = 0
    private(set) var prepareCacheOffsets: [Int] = []
    private(set) var lastCache: [KVCache] = []

    init(
        preparation: BatchTextOnlyPreparation, stateful: Bool = false,
        emitsBatchState: Bool = false, cacheFactory: (() -> [KVCache])? = nil,
        prepareObserver: (([KVCache]) throws -> Void)? = nil,
        forwardObserver: (() -> Void)? = nil
    ) {
        self.preparation = preparation
        self.stateful = stateful
        self.emitsBatchState = emitsBatchState
        self.cacheFactory = cacheFactory
        self.prepareObserver = prepareObserver
        self.forwardObserver = forwardObserver
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        prepareCacheOffsets.append(cache.first?.offset ?? 0)
        try prepareObserver?(cache)
        switch preparation {
        case .tokens:
            return .tokens(input.text)
        case .logits:
            let output = callAsFunction(input.text[text: .newAxis], cache: cache, state: state)
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
            return .logits(output)
        }
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardObserver?()
        let tokens = input.tokens
        let keys = tokens.asType(.float32).reshaped([tokens.dim(0), 1, tokens.dim(1), 1])
        func update(_ cache: KVCache) {
            if let list = cache as? CacheList {
                list.children.forEach(update)
            } else {
                _ = cache.update(keys: keys, values: keys)
            }
        }
        cache?.forEach(update)
        let counter = (state?[Self.counterKey] ?? 0) + 1
        let increment = stateful ? counter : 1
        let target = (tokens[0..., -1] + Int32(increment)) % Int32(16)
        let logits = MLX.where(
            MLXArray.arange(16)[.newAxis, .newAxis] .== target[0..., .newAxis, .newAxis],
            MLXArray(Float(5)), MLXArray(Float(-5)))
        var outputState: LMOutput.State?
        if stateful || (emitsBatchState && tokens.dim(0) > 1) {
            var next = state ?? LMOutput.State()
            next[Self.counterKey] = counter
            outputState = next
        }
        return LMOutput(logits: logits, state: outputState)
    }

    func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        nextTokenCalls += 1
        lastCache = cache ?? []
        return callAsFunction(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        cacheFactory?() ?? [KVCacheSimple()]
    }
}

enum CacheAdmissionMismatch: Sendable, CaseIterable {
    case fullToRotating
    case rotatingToFull
    case window
    case layerCount
    case laterLayer
    case composite
    case nestedComposite

    func active() -> [KVCache] {
        switch self {
        case .fullToRotating, .layerCount:
            [KVCacheSimple()]
        case .rotatingToFull, .window:
            [RotatingKVCache(maxSize: 16)]
        case .laterLayer:
            [KVCacheSimple(), KVCacheSimple()]
        case .composite:
            [CacheList(KVCacheSimple(), KVCacheSimple())]
        case .nestedComposite:
            [CacheList(KVCacheSimple(), CacheList(KVCacheSimple(), KVCacheSimple()))]
        }
    }

    func incoming() -> [KVCache] {
        switch self {
        case .fullToRotating:
            [RotatingKVCache(maxSize: 16)]
        case .rotatingToFull:
            [KVCacheSimple()]
        case .window:
            [RotatingKVCache(maxSize: 8)]
        case .layerCount:
            [KVCacheSimple(), KVCacheSimple()]
        case .laterLayer:
            [KVCacheSimple(), RotatingKVCache(maxSize: 16)]
        case .composite:
            [CacheList(KVCacheSimple())]
        case .nestedComposite:
            [CacheList(KVCacheSimple(), CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16)))]
        }
    }
}

private final class AdmissionCountingSampler: LogitSampler {
    private let sampler = CategoricalSampler(temperature: 20, seed: 43)
    private(set) var calls = 0

    func sample(logits: MLXArray) -> MLXArray {
        calls += 1
        return sampler.sample(logits: logits)
    }
}

private func tinyMamba2() throws -> Mamba2Model {
    let configuration = try JSONDecoder().decode(
        Mamba2Configuration.self,
        from: Data(
            """
            {
              "model_type": "mamba2", "num_heads": 4, "head_dim": 4, "vocab_size": 32,
              "hidden_size": 16, "state_size": 8, "num_hidden_layers": 2,
              "layer_norm_epsilon": 1e-5, "conv_kernel": 4, "n_groups": 1, "use_bias": false,
              "use_conv_bias": true, "tie_word_embeddings": false, "time_step_limit": [0.0, 100.0]
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 601)) {
        let model = Mamba2Model(configuration)
        eval(model)
        return model
    }
}

private func tinyFalconH1() throws -> FalconH1Model {
    let configuration = try JSONDecoder().decode(
        FalconH1Configuration.self,
        from: Data(
            """
            {
              "model_type": "falcon_h1", "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
              "vocab_size": 100, "mamba_d_ssm": 16, "mamba_d_state": 8, "mamba_d_head": 8,
              "mamba_n_heads": 2, "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_chunk_size": 64,
              "attention_in_multiplier": 2.0, "key_multiplier": 0.5
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 602)) {
        let model = FalconH1Model(configuration)
        eval(model)
        return model
    }
}

/// The tokens `TokenIterator` generates for `prompt`.
private func generateAlone(
    _ model: any LanguageModel, prompt: [Int], cache: [KVCache]? = nil,
    processor: LogitProcessor? = nil, sampler: LogitSampler, maxTokens: Int
) throws -> [Int] {
    let iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model, cache: cache,
        processor: processor, sampler: sampler, maxTokens: maxTokens)
    return Array(IteratorSequence(iterator))
}

/// Every row's tokens, in uid order, until the batch is empty.
private func collect(_ iterator: inout BatchTokenIterator, uids: [Int]) throws -> [[Int]] {
    var results = [Int: [Int]]()
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses {
            results[response.uid, default: []].append(response.token)
        }
    }
    return uids.map { results[$0] ?? [] }
}

extension BatchTokenIteratorTests {

    @Test(
        "Text-only Idefics3 iterator streams match solo through admission and removal",
        arguments: [2, 3])
    func ideficsIteratorParity(batchSize: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyIdefics3ForBatchPositions()
            let prompts = Array(
                [[1, 2, 3], Array(4 ..< 11), Array(11 ..< 20)].prefix(batchSize))
            func sampler(_ row: Int) -> TopPSampler {
                TopPSampler(temperature: 0.8, topP: 0.9, seed: UInt64(71 + row))
            }
            let expected = try prompts.indices.map { row in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompts[row].map { Int32($0) })), model: model,
                    cache: try model.newCache(parameters: nil), processor: nil,
                    sampler: sampler(row), maxTokens: 8)
                return Array(IteratorSequence(iterator))
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 8)
            var uids = iterator.insert(prompts: [prompts[0]], samplers: [sampler(0)])
            var output = [Int: [Int]]()
            for _ in 0 ..< 2 {
                for response in try iterator.next() {
                    output[response.uid, default: []].append(response.token)
                }
            }
            uids += iterator.insert(
                prompts: Array(prompts.dropFirst()),
                samplers: prompts.indices.dropFirst().map { sampler($0) })
            for _ in 0 ..< 2 {
                for response in try iterator.next() {
                    output[response.uid, default: []].append(response.token)
                }
            }
            iterator.remove([uids[0]])
            #expect(iterator.uids == Array(uids.dropFirst()))
            while true {
                let responses = try iterator.next()
                if responses.isEmpty { break }
                for response in responses {
                    output[response.uid, default: []].append(response.token)
                }
            }
            #expect(output[uids[0]] == Array(expected[0].prefix(4)))
            for row in prompts.indices.dropFirst() {
                #expect(output[uids[row]] == expected[row])
            }
        }
    }
}

extension QwenVLBatchedPositionTests {

    @Test(
        "Cold text-only Qwen VL batches match single iterators",
        arguments: QwenVLTextFamily.allCases)
    func coldTextIterator(family: QwenVLTextFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyQwenVLText(family)
            let prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
            let expected = try prompts.map { prompt in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })),
                    model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 6)
                return Array(IteratorSequence(iterator))
            }
            let actual = try batchGenerate(model: model, prompts: prompts, maxTokens: 6)
            #expect(actual == expected)
        }
    }
}
