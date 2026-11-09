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

                let events = try iterator.next()
                #expect(tokenIDs(events) == uids)
                #expect(tokens(events) == [2, 4])
                #expect(log.created.filter { $0 == "prepare" }.count == 2)
                #expect(log.created.filter { $0 == "forward" }.count == 3)
                #expect(log.released.sorted() == log.created.sorted())
            }
        }
    }

    @Test("Throwing admission drains its autoreleases and fails the request")
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

                let events = try #require(try iterator.next())
                #expect(events.count == 1)
                let failure = try #require(failures(events).first)
                #expect(failure.id == uids[0])
                #expect(failure.error as? BatchAdmissionProbeError == .rejected)
                #expect(iterator.ids.isEmpty)
                #expect(log.created == ["prepare"])
                #expect(log.released == log.created)
                #expect(try iterator.next() == nil)
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
            #expect(tokenIDs(first) == uids)
            #expect(tokens(first) == [0, 0])
            #expect(tokens(second) == [7, 7])
            #expect(model.decodeOffsets == [7, 7])
            #expect(prefixes.map(\.offset) == offsets)
            for (prefix, offset) in zip(prefixes, offsets) {
                #expect(try #require(prefix[0]).asArray(Float.self) == [Float(offset)])
                #expect(prefix[1] == nil)
            }
        }
    }

    @Test("A row with a longer recurrent prefix joining a running batch moves its offset")
    func recurrentAdmissionIntoRunningBatch() throws {
        try Device.withDefaultDevice(.cpu) {
            func prefix(_ offset: Int) -> ArraysCache {
                let cache = ArraysCache(size: 2)
                cache.offset = offset
                cache[0] = MLXArray([Float(offset)]).reshaped([1, 1])
                return cache
            }
            let model = BatchLegacyOffsetModel()
            var iterator = BatchTokenIterator(model: model, maxTokens: 4)
            let running = iterator.insert(prompts: [[1]], caches: [[prefix(3)]])
            let first = try iterator.next()
            let joining = iterator.insert(prompts: [[2]], caches: [[prefix(7)]])
            let second = try iterator.next()
            let third = try iterator.next()

            #expect(tokenIDs(first) == running)
            #expect(tokenIDs(second) == running + joining)
            #expect(tokens(second) == [3, 0])
            #expect(tokens(third) == [7, 7])
            // The batch decodes at the longest row's offset once the longer row joins.
            #expect(model.decodeOffsets == [3, 7, 7])
        }
    }

    @Test(
        "Incompatible admission fails the queued row and leaves active caches and the seeded stream",
        arguments: CacheAdmissionMismatch.allCases)
    func incompatibleAdmission(mismatch: CacheAdmissionMismatch) throws {
        try Device.withDefaultDevice(.cpu) {
            func cacheValues(_ caches: [KVCache]) -> [[[Float]]] {
                caches.map { cache in cache.state.map { $0.asType(.float32).asArray(Float.self) } }
            }
            // The same active row without the incompatible one, for comparison.
            let referenceModel = BatchTextOnlyModel(
                preparation: .tokens, cacheFactory: mismatch.active)
            let referenceSampler = AdmissionCountingSampler()
            var reference = BatchTokenIterator(
                model: referenceModel, sampler: referenceSampler, maxTokens: 6)
            _ = reference.insert(prompts: [[1, 3]])
            let referenceTokens = tokens(try reference.next()) + tokens(try reference.next())

            let model = BatchTextOnlyModel(preparation: .tokens, cacheFactory: mismatch.active)
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler, maxTokens: 6)
            let active = iterator.insert(prompts: [[1, 3]])
            var generated = tokens(try iterator.next())
            let prepares = model.prepareCacheOffsets.count
            let queued = iterator.insert(prompts: [[7]], caches: [mismatch.incoming()])

            // The queued row fails and the active row decodes in the same step.
            let events = try #require(try iterator.next())
            let failure = try #require(failures(events).first)
            #expect(failures(events).count == 1)
            #expect(failure.id == queued[0])
            #expect(failure.error is KVCacheError)
            #expect(tokenIDs(events) == active)
            #expect(iterator.ids == active)
            generated += tokens(events)
            #expect(generated == referenceTokens)
            #expect(
                sampler.calls == referenceSampler.calls,
                "rejection must not draw from the shared random state")
            #expect(cacheValues(model.lastCache) == cacheValues(referenceModel.lastCache))
            if mismatch == .layerCount {
                #expect(model.prepareCacheOffsets.count == prepares)
            }

            generated += try collect(&iterator, uids: active)[0]
            let expected = try batchGenerate(
                model: BatchTextOnlyModel(preparation: .tokens, cacheFactory: mismatch.active),
                prompts: [[1, 3]], maxTokens: 6, sampler: AdmissionCountingSampler())
            #expect(generated == expected[0])
        }
    }

    @Test("A queued row with a different layer count fails before its prompt is prefilled")
    func pendingLayerCounts() throws {
        try Device.withDefaultDevice(.cpu) {
            let referenceSampler = AdmissionCountingSampler()
            var reference = BatchTokenIterator(
                model: BatchTextOnlyModel(preparation: .tokens), sampler: referenceSampler)
            _ = reference.insert(prompts: [[1]], caches: [[KVCacheSimple()]])
            _ = try reference.next()

            let model = BatchTextOnlyModel(preparation: .tokens)
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler)
            let uids = iterator.insert(
                prompts: [[1], [2]],
                caches: [[KVCacheSimple()], [KVCacheSimple(), KVCacheSimple()]])
            let events = try iterator.next()
            #expect(failures(events).map(\.id) == [uids[1]])
            #expect(failures(events).allSatisfy { $0.error is KVCacheError })
            #expect(tokenIDs(events) == [uids[0]])
            #expect(iterator.ids == [uids[0]])
            #expect(model.prepareCacheOffsets == [0], "only the first row is prefilled")
            #expect(
                sampler.calls == referenceSampler.calls,
                "rejection must not draw from the shared random state")
        }
    }

    @Test(
        "A queued row that cannot join the others fails before sampling or touching active rows",
        arguments: [false, true])
    func incompatibleQueuedRows(withActiveRows: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            func run(includingMismatch: Bool) throws -> (
                iterator: BatchTokenIterator, sampler: AdmissionCountingSampler, active: [Int],
                queued: [Int], events: [BatchEvent]?
            ) {
                let model = BatchTextOnlyModel(
                    preparation: .tokens, cacheFactory: { [RotatingKVCache(maxSize: 16)] })
                let sampler = AdmissionCountingSampler()
                var iterator = BatchTokenIterator(model: model, sampler: sampler, maxTokens: 6)
                var active: [Int] = []
                if withActiveRows {
                    active = iterator.insert(prompts: [[1, 3]])
                    _ = try iterator.next()
                }
                // The first queued row matches the active rows; the second's window matches
                // neither.
                var prompts: [[Int]] = [[7]]
                var caches: [[KVCache]?] = [[RotatingKVCache(maxSize: 16)]]
                if includingMismatch {
                    prompts.append([9])
                    caches.append([RotatingKVCache(maxSize: 8)])
                }
                let queued = iterator.insert(prompts: prompts, caches: caches)
                let events = try iterator.next()
                return (iterator, sampler, active, queued, events)
            }
            let reference = try run(includingMismatch: false)
            let actual = try run(includingMismatch: true)

            #expect(failures(actual.events).map(\.id) == [actual.queued[1]])
            #expect(failures(actual.events).allSatisfy { $0.error is KVCacheError })
            #expect(tokenIDs(actual.events) == actual.active + [actual.queued[0]])
            #expect(tokens(actual.events) == tokens(reference.events))
            #expect(actual.iterator.ids == actual.active + [actual.queued[0]])
            #expect(
                actual.sampler.calls == reference.sampler.calls,
                "rejection must not draw from the shared random state")
        }
    }

    @Test(
        "A model whose cache layer has no batched form is refused at admission with failures",
        arguments: [false, true])
    func layerWithoutBatchedForm(nested: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(
                preparation: .tokens,
                cacheFactory: {
                    let chunked = ChunkedKVCache(chunkSize: 4)
                    return [
                        KVCacheSimple(), nested ? CacheList(KVCacheSimple(), chunked) : chunked,
                    ]
                })
            let sampler = AdmissionCountingSampler()
            var iterator = BatchTokenIterator(model: model, sampler: sampler)
            let uids = iterator.insert(prompts: [[1, 2], [3]])
            let events = try iterator.next()
            #expect(failures(events).map(\.id) == uids)
            #expect(failures(events).allSatisfy { $0.error is KVCacheError })
            #expect(events?.count == uids.count)
            #expect(iterator.ids.isEmpty)
            #expect(sampler.calls == 0)
            #expect(try iterator.next() == nil)
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
                let events = try iterator.next()
                #expect(failures(events).map(\.id) == uids)
                #expect(failures(events).allSatisfy { $0.error is KVCacheError })
                #expect(iterator.ids.isEmpty)
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

    @Test("Rejected stateful admission fails the queued row and leaves the active row intact")
    func statefulAdmissionPreservesActiveRow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .logits, stateful: true)
            var iterator = BatchTokenIterator(model: model, maxTokens: 5)
            let active = iterator.insert(prompts: [[1, 3]])
            var generated = tokens(try iterator.next())
            let queued = iterator.insert(prompts: [[7]])
            let events = try iterator.next()
            #expect(failures(events).map(\.id) == queued)
            #expect(
                failures(events).allSatisfy {
                    $0.error as? BatchTokenIterator.StateError == .modelStateRequiresSingleRow
                })
            #expect(tokenIDs(events) == active)
            #expect(iterator.ids == active)
            generated += tokens(events)
            generated += try collect(&iterator, uids: active)[0]
            #expect(generated == [4, 6, 9, 13, 2])

            let removed = iterator.insert(prompts: [[5]])
            #expect(tokens(try iterator.next()) == [6])
            iterator.remove(removed)
            _ = iterator.insert(prompts: [[8]])
            #expect(tokens(try iterator.next()) == [9])
        }
    }

    @Test(
        "A row cannot join a queued row whose prefill returned model state",
        arguments: BatchTextOnlyPreparation.allCases)
    func statefulPrefillRefusesMultipleRows(preparation: BatchTextOnlyPreparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: preparation, stateful: true)
            var iterator = BatchTokenIterator(model: model, maxTokens: 3)
            let uids = iterator.insert(prompts: [[1, 3], [7]])
            let first = try iterator.next()
            #expect(failures(first).map(\.id) == [uids[1]])
            #expect(
                failures(first).allSatisfy {
                    $0.error as? BatchTokenIterator.StateError == .modelStateRequiresSingleRow
                })
            #expect(model.prepareCacheOffsets == [0], "the refused row is not prefilled")
            #expect(iterator.ids == [uids[0]])
            let rest = try collect(&iterator, uids: [uids[0]])
            #expect([tokens(first) + rest[0]] == [[4, 6, 9]])
        }
    }

    @Test(
        "A refused row leaves its supplied prefix to retry without prefilling it twice",
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
            // The first row's prefill returns state, so the row with the prefix is refused.
            let uids = iterator.insert(prompts: [[7], [1, 3]], caches: [nil, [prefix]])
            let first = try iterator.next()
            #expect(failures(first).map(\.id) == [uids[1]])
            #expect(
                failures(first).allSatisfy {
                    $0.error as? BatchTokenIterator.StateError == .modelStateRequiresSingleRow
                })
            #expect(prefix.offset == 2)
            #expect(maxAbsDifference(try #require(prefix.state.first), keys) == 0)
            #expect(model.prepareCacheOffsets == [0])
            #expect(log.events == [.prefix(0)])
            let running = try collect(&iterator, uids: [uids[0]])
            #expect([tokens(first) + running[0]] == [[8, 10, 13]])

            let retry = iterator.insert(prompt: [1, 3], cache: [prefix])
            let retried = try collect(&iterator, uids: [retry])
            #expect(retried == [[4, 6, 9]])
            #expect(model.prepareCacheOffsets == [0, 2])
            #expect(prefix.offset == 2)
            #expect(log.events == [.prefix(0), .prefix(2)])
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
            iterator.remove(iterator.ids)
            _ = iterator.insert(prompts: [[7]])
            let events = try iterator.next()
            #expect(tokens(events) == [8])
        }
    }

    @Test("Each row ends with one terminal event after its last token, and nil means idle")
    func terminalEvents() throws {
        try Device.withDefaultDevice(.cpu) {
            // Each step samples the fed token plus one: [1] samples the stop token 2 first,
            // [5] runs to its limit and [9] stops on 11 after one token.
            let model = BatchTextOnlyModel(preparation: .tokens)
            var iterator = BatchTokenIterator(model: model, stopTokens: [2, 11])
            #expect(try iterator.next() == nil)
            let ids = iterator.insert(prompts: [[1], [5], [9]], maxTokens: [4, 3, 4])
            #expect(iterator.pendingIDs == ids)
            #expect(iterator.ids == ids)

            var steps: [[String]] = []
            while let events = try iterator.next() {
                #expect(!events.isEmpty)
                #expect(iterator.pendingIDs.isEmpty)
                steps.append(events.map(describe))
            }
            let (a, b, c) = (ids[0], ids[1], ids[2])
            #expect(
                steps == [
                    ["finished(\(a), stop, 2)", "token(\(b), 6)", "token(\(c), 10)"],
                    ["token(\(b), 7)", "finished(\(c), stop, 11)"],
                    ["token(\(b), 8)", "finished(\(b), length, nil)"],
                ])
            #expect(iterator.ids.isEmpty)
            #expect(try iterator.next() == nil)

            // Inserting after idle starts the iterator again.
            let d = iterator.insert(prompt: [3], maxTokens: 1)
            #expect(d == c + 1)
            #expect(iterator.pendingIDs == [d])
            #expect(
                try iterator.next()?.map(describe) == [
                    "token(\(d), 4)", "finished(\(d), length, nil)",
                ])
            #expect(try iterator.next() == nil)
        }
    }

    @Test("The single-prompt insert forwards its cache, limit, sampler and processor")
    func singlePromptInsert() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let parameters = GenerateParameters(repetitionPenalty: 1.5, repetitionContextSize: 8)
            let sampler = TopPSampler(temperature: 0.8, topP: 0.9, seed: 5)
            let prefix: [Int] = [1, 2, 3, 4]
            let alone = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: alone))
            let expected = try generateAlone(
                model, prompt: [5, 6], cache: alone, processor: parameters.processor(),
                sampler: TopPSampler(temperature: 0.8, topP: 0.9, seed: 5), maxTokens: 5)

            let cache = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: cache))
            var iterator = BatchTokenIterator(model: model, maxTokens: 8)
            let id = iterator.insert(
                prompt: [5, 6], maxTokens: 5, cache: cache, sampler: sampler,
                processor: parameters.processor())
            #expect(try collect(&iterator, uids: [id]) == [expected])
        }
    }

    @Test("batchGenerate records each token it returns, then finalizes every row once")
    func batchGenerateRecordsReturnedTokens() throws {
        try Device.withDefaultDevice(.cpu) {
            // Each step samples the fed token plus one: [9] stops on 11 after one token and
            // [5] runs to the limit of 3.
            let log = RecordingProcessorLog()
            let tokens = try batchGenerate(
                model: BatchTextOnlyModel(preparation: .tokens), prompts: [[9], [5]],
                maxTokens: 3, stopTokens: [11],
                processor: RecordingProcessor(name: "row", log: log))
            #expect(tokens == [[10], [6, 7, 8]])
            // The stop token 11 is not returned, so it is not recorded.
            #expect(
                log.entries == [
                    "emit row 10", "emit row 6", "emit row 7", "finalize row", "emit row 8",
                    "finalize row",
                ])

            // A row that fails to join still finalizes every processor once.
            let failing = RecordingProcessorLog()
            #expect(throws: KVCacheError.self) {
                _ = try batchGenerate(
                    model: BatchTextOnlyModel(preparation: .tokens), prompts: [[9], [5]],
                    caches: [nil, [KVCacheSimple(), KVCacheSimple()]],
                    processor: RecordingProcessor(name: "row", log: failing))
            }
            #expect(failing.entries == ["finalize row", "finalize row"])
        }
    }

    @Test("batchGenerate finalizes every row once when the batch throws")
    func batchGenerateFinalizesOnThrow() throws {
        try Device.withDefaultDevice(.cpu) {
            // Two rows decoding together make the model return state, which invalidates the
            // batch and makes next() throw.
            let log = RecordingProcessorLog()
            #expect(throws: BatchTokenIterator.StateError.invalidatedByModelState) {
                _ = try batchGenerate(
                    model: BatchTextOnlyModel(preparation: .tokens, emitsBatchState: true),
                    prompts: [[1, 3], [7]],
                    processor: RecordingProcessor(name: "row", log: log))
            }
            #expect(log.entries == ["finalize row", "finalize row"])
        }
    }

    @Test("Recorded tokens reach a row's processor before its one finalization")
    func emissionAndFinalization() throws {
        try Device.withDefaultDevice(.cpu) {
            // Each step samples the fed token plus one: a stops on 11 after one token, b runs
            // to its two-token limit, c is removed while decoding and d cannot join.
            let log = RecordingProcessorLog()
            let model = BatchTextOnlyModel(preparation: .tokens)
            var iterator = BatchTokenIterator(model: model, stopTokens: [11])
            let ids = iterator.insert(
                prompts: [[9], [5], [3], [7]], maxTokens: [8, 2, 8, 8],
                caches: [nil, nil, nil, [KVCacheSimple(), KVCacheSimple()]],
                processors: ["a", "b", "c", "d"].map {
                    RecordingProcessor(name: $0, log: log) as LogitProcessor?
                })
            let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3])

            let first = try iterator.next()
            #expect(failures(first).map(\.id) == [d])
            #expect(tokens(first) == [10, 6, 4])
            for case .token(let id, let token) in first ?? [] {
                iterator.recordEmittedToken(id: id, token: token)
            }
            #expect(log.entries == ["emit a 10", "emit b 6", "emit c 4"])
            iterator.remove([c])
            #expect(log.entries.last == "finalize c", "remove finalizes at once")
            iterator.recordEmittedToken(id: c, token: 5)
            #expect(log.entries.last == "finalize c", "a removed row records nothing")

            // d failed in the previous step, so this call finalizes it first.
            let second = try iterator.next()
            #expect(log.entries.last == "finalize d")
            #expect(
                finishes(second) == [
                    BatchFinish(id: a, reason: .stop, stopToken: 11),
                    BatchFinish(id: b, reason: .length, stopToken: nil),
                ])
            #expect(tokens(second) == [7])
            // The caller delivers b's last token, but not a's stop token.
            iterator.recordEmittedToken(id: b, token: 7)
            #expect(log.entries.last == "emit b 7", "an ended row records until the next call")

            #expect(try iterator.next() == nil)
            #expect(
                log.entries == [
                    "emit a 10", "emit b 6", "emit c 4", "finalize c", "finalize d", "emit b 7",
                    "finalize a", "finalize b",
                ])
            #expect(try iterator.next() == nil)
            iterator.remove(ids)
            #expect(log.entries.count == 8, "each row is finalized once")
        }
    }

    @Test("The GenerateParameters initializer matches single runs with the same parameters")
    func generateParametersParity() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            // A capacity smaller than prompt plus output, so every row's window rotates.
            let parameters = GenerateParameters(
                maxTokens: 10,
                kvCache: KVCacheConfiguration(
                    capacity: try .init(maxTokens: 8, preservedPrefixTokens: 0)),
                temperature: 0, repetitionPenalty: 1.5, repetitionContextSize: 8)
            let expected = try Self.prompts.map { prompt in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model,
                    parameters: parameters)
                return Array(IteratorSequence(iterator))
            }
            #expect(expected.allSatisfy { $0.count == 10 })

            var iterator = try BatchTokenIterator(model: model, parameters: parameters)
            let ids = iterator.insert(prompts: Self.prompts)
            #expect(try collect(&iterator, uids: ids) == expected)
        }
    }

    @Test("A GenerateParameters token limit of nil leaves rows unlimited")
    func generateParametersWithoutTokenLimit() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .tokens)
            var iterator = try BatchTokenIterator(
                model: model, parameters: GenerateParameters(maxTokens: nil, temperature: 0))
            let id = iterator.insert(prompt: [1])
            // Past the plain initializer's default limit of 128.
            for _ in 0 ..< 200 {
                let events = try #require(try iterator.next())
                #expect(tokenIDs(events) == [id])
                #expect(finishes(events).isEmpty)
            }
            iterator.remove([id])
            #expect(try iterator.next() == nil)
        }
    }

    @Test("The GenerateParameters initializer refuses settings batching does not support")
    func generateParametersRefusals() throws {
        let model = BatchTextOnlyModel(preparation: .tokens)
        let refused: [(GenerateParameters, BatchTokenIterator.ParameterError)] = [
            (GenerateParameters(maxKVSize: 64), .maxKVSize),
            (
                GenerateParameters(
                    kvCache: KVCacheConfiguration(capacity: try .init(maxTokens: 64))),
                .pinnedPrefix(4)
            ),
            (
                GenerateParameters(
                    kvCache: KVCacheConfiguration(
                        capacity: try .init(maxTokens: 64, preservedPrefixTokens: 0),
                        rewind: try .init(maxTokens: 8))),
                .rewindReserve
            ),
            (GenerateParameters(kvBits: 4), .strategy(.affine)),
            (
                GenerateParameters(
                    kvCache: KVCacheConfiguration(strategy: .affine(try .init(bits: 8)))),
                .strategy(.affine)
            ),
            (GenerateParameters(kvScheme: "turbo8v3"), .strategy(.turboQuant)),
            (GenerateParameters(logProbabilities: 5), .logProbabilities),
        ]
        for (parameters, error) in refused {
            #expect(throws: error) {
                _ = try BatchTokenIterator(model: model, parameters: parameters)
            }
        }
        #expect(throws: KVCacheConfigurationError.conflictingLegacyConfiguration) {
            _ = try BatchTokenIterator(
                model: model,
                parameters: GenerateParameters(
                    kvCache: KVCacheConfiguration(
                        capacity: try .init(maxTokens: 64, preservedPrefixTokens: 0)),
                    kvBits: 4))
        }
        _ = try BatchTokenIterator(model: model, parameters: GenerateParameters())
    }

    @Test("A prefill stopped by task cancellation stops admission before any sampling")
    func cancelledAdmissionStopsBeforeSampling() async throws {
        try await Task {
            try Device.withDefaultDevice(.cpu) {
                // The first row prefills, the second fails to join before its prefill, and the
                // third row's prefill sees the task cancelled.
                final class Count { var value = 0 }
                let prepares = Count()
                let model = BatchTextOnlyModel(
                    preparation: .tokens,
                    prepareObserver: { _ in
                        prepares.value += 1
                        guard prepares.value == 2 else { return }
                        withUnsafeCurrentTask { $0?.cancel() }
                        throw CancellationError()
                    })
                let sampler = AdmissionCountingSampler()
                var iterator = BatchTokenIterator(model: model, sampler: sampler)
                let ids = iterator.insert(
                    prompts: [[1], [2], [3], [4]],
                    caches: [nil, [KVCacheSimple(), KVCacheSimple()], nil, nil])
                #expect(throws: CancellationError.self) { _ = try iterator.next() }
                #expect(sampler.calls == 0, "nothing is sampled after the cancellation")
                #expect(model.nextTokenCalls == 1, "only the first row's prefill ran a forward")
                #expect(prepares.value == 2, "the fourth row is not prefilled")
                #expect(iterator.pendingIDs == [ids[0], ids[2], ids[3]])
                // The failure from the cancelled call arrives with the next one.
                let delivered = try iterator.next()
                #expect(failures(delivered).map(\.id) == [ids[1]])
                #expect(delivered?.count == 1)
            }
        }.value
    }

    @Test("Failures from a step that then throws are returned by the next call")
    func failuresSurviveAThrowingStep() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BatchTextOnlyModel(preparation: .tokens, emitsBatchState: true)
            var iterator = BatchTokenIterator(model: model)
            let ids = iterator.insert(
                prompts: [[1, 3], [7], [9]],
                caches: [nil, nil, [KVCacheSimple(), KVCacheSimple()]])
            #expect(throws: BatchTokenIterator.StateError.invalidatedByModelState) {
                _ = try iterator.next()
            }
            let delivered = try iterator.next()
            #expect(delivered?.count == 1)
            #expect(failures(delivered).map(\.id) == [ids[2]])
            #expect(failures(delivered).allSatisfy { $0.error is KVCacheError })
            #expect(throws: BatchTokenIterator.StateError.invalidatedByModelState) {
                _ = try iterator.next()
            }
            #expect(iterator.ids == [ids[0], ids[1]])
            iterator.remove(iterator.ids)
            #expect(try iterator.next() == nil)
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
            var firstTokens = tokens(try iterator.next())
            firstTokens += tokens(try iterator.next())
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
            var steps: [[BatchEvent]] = []
            steps.append(try #require(try iterator.next()))
            steps.append(try #require(try iterator.next()))
            let third = iterator.insert(prompts: [Self.prompts[2]], maxTokens: [3])
            #expect(iterator.ids == first + third)
            while let events = try iterator.next() {
                steps.append(events)
            }

            func rowTokens(_ uid: Int) -> [Int] {
                steps.flatMap { tokens($0.filter { $0.id == uid }) }
            }
            func rowFinishes(_ uid: Int) -> [BatchFinish] {
                steps.flatMap { finishes($0.filter { $0.id == uid }) }
            }
            /// Whether the row's terminal event is its last event.
            func endsWithFinish(_ uid: Int) -> Bool {
                guard case .finished = steps.flatMap({ $0.filter { $0.id == uid } }).last else {
                    return false
                }
                return true
            }
            // The stop token ends the row without being reported as a token.
            #expect(rowTokens(first[0]) == Array(alone[0].prefix(2)))
            #expect(
                rowFinishes(first[0]) == [BatchFinish(id: first[0], reason: .stop, stopToken: stop)]
            )
            #expect(rowTokens(first[1]) == alone[1])
            #expect(
                rowFinishes(first[1]) == [
                    BatchFinish(id: first[1], reason: .length, stopToken: nil)
                ])
            #expect(rowTokens(third[0]) == Array(alone[2].prefix(3)))
            #expect(
                rowFinishes(third[0]) == [
                    BatchFinish(id: third[0], reason: .length, stopToken: nil)
                ])
            #expect((first + third).allSatisfy(endsWithFinish))
            // Llama prefills resumably: each step decodes the joined rows and then admits one
            // queued row, so the first row decodes from step 1 and the third joins in step 2.
            #expect(steps[0].map(\.id) == [first[0]])
            #expect(steps[1].map(\.id) == first)
            #expect(steps[2].map(\.id) == first + third)
            // The first row's stop token ends it in step 3; it is gone from step 4.
            #expect(steps[3].map(\.id) == first + third)
            #expect(steps[4].map(\.id) == [first[1], third[0]])
            #expect(iterator.ids.isEmpty)
        }
    }

    @Test("A resumable model prefills one chunk per step while other rows decode")
    func resumablePrefillChunks() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            #expect(model is any ResumablePrefillModel)
            let prefill = PrefillParameters(stepSize: 4)
            let short: [Int] = [1, 2]
            // A warm row: a cached 4-token prefix and an 11-token suffix to prefill in chunks.
            let prefix: [Int] = [3, 4, 5, 6]
            let suffix = Array(7 ..< 18)
            func prefixCache() throws -> [KVCache] {
                let cache = try model.newCache(parameters: nil)
                eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: cache))
                return cache
            }
            func alone(_ prompt: [Int], cache: [KVCache]?) throws -> [Int] {
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model,
                    cache: cache, processor: nil, sampler: ArgMaxSampler(), prefill: prefill,
                    maxTokens: 8)
                return Array(IteratorSequence(iterator))
            }
            let expected = [
                try alone(short, cache: nil), try alone(suffix, cache: try prefixCache()),
            ]
            let chunks = model.prefillChunks(
                for: LMInput.Text(tokens: MLXArray(suffix.map { Int32($0) })), prefill: prefill)
            #expect(chunks == [0 ..< 4, 4 ..< 8, 8 ..< 10])

            var iterator = BatchTokenIterator(model: model, maxTokens: 8, prefill: prefill)
            let first = iterator.insert(prompt: short)
            let warm = iterator.insert(prompt: suffix, cache: try prefixCache())
            var steps: [[String]] = []
            var output = [Int: [Int]]()
            while let events = try iterator.next() {
                steps.append(events.map(describe))
                for case .token(let id, let token) in events {
                    output[id, default: []].append(token)
                }
            }
            #expect([output[first] ?? [], output[warm] ?? []] == expected)
            // The short row joins first; then every step decodes it and feeds one chunk of the
            // warm row, whose final forward completes its prompt before it decodes too.
            let a = expected[0]
            #expect(
                Array(steps.prefix(6)) == [
                    ["prefill(\(first), 2/2)"],
                    ["token(\(first), \(a[0]))", "prefill(\(warm), 4/11)"],
                    ["token(\(first), \(a[1]))", "prefill(\(warm), 8/11)"],
                    ["token(\(first), \(a[2]))", "prefill(\(warm), 10/11)"],
                    ["token(\(first), \(a[3]))", "prefill(\(warm), 11/11)"],
                    ["token(\(first), \(a[4]))", "token(\(warm), \(expected[1][0]))"],
                ])
        }
    }

    @Test(
        "Each model family that opts in prefills resumably and matches single runs",
        arguments: ResumablePrefillFamily.allCases)
    func resumablePrefillFamilies(family: ResumablePrefillFamily) async throws {
        // Fixed weights per family, so a near tie between the batched and single-row
        // arithmetic cannot flip a greedy token on some runs and not others.
        let seed = UInt64(ResumablePrefillFamily.allCases.firstIndex(of: family)! + 1100)
        let model = try await withRandomState(MLXRandom.RandomState(seed: seed)) {
            try await LLMTypeRegistry.shared.createModel(
                configuration: Data(family.configuration.utf8), modelType: family.rawValue)
        }
        let resumable = try #require(model as? any ResumablePrefillModel, "\(family) opts in")
        // On the default device: some families use Metal kernels. Both runs use the same ones.
        do {
            eval(model)
            let prefill = PrefillParameters(stepSize: 4)
            // A short row decodes while the long one is fed in chunks of 4, 4 and 2 tokens
            // and then its final forward.
            let prompts: [[Int]] = [[1, 2], Array(3 ..< 14)]
            #expect(
                resumable.prefillChunks(
                    for: LMInput.Text(tokens: MLXArray(prompts[1].map { Int32($0) })),
                    prefill: prefill) == [0 ..< 4, 4 ..< 8, 8 ..< 10])
            let expected = try prompts.map { prompt in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model,
                    processor: nil, sampler: ArgMaxSampler(), prefill: prefill, maxTokens: 6)
                return Array(IteratorSequence(iterator))
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 6, prefill: prefill)
            let ids = iterator.insert(prompts: prompts)
            var progress: [String] = []
            var output = [Int: [Int]]()
            while let events = try iterator.next() {
                for event in events {
                    switch event {
                    case .token(let id, let token): output[id, default: []].append(token)
                    case .prefill(let id, let processed, let total):
                        progress.append("\(id): \(processed)/\(total)")
                    case .failed(_, let error): Issue.record("\(family): \(error)")
                    case .finished: break
                    }
                }
            }
            #expect(ids.map { output[$0] ?? [] } == expected, "\(family)")
            #expect(
                progress == [
                    "\(ids[0]): 2/2", "\(ids[1]): 4/11", "\(ids[1]): 8/11", "\(ids[1]): 10/11",
                    "\(ids[1]): 11/11",
                ], "\(family)")
        }
    }

    @Test("Cancellation between resumable prefill chunks leaves the row queued")
    func resumablePrefillCancellation() async throws {
        try await Task {
            try Device.withDefaultDevice(.cpu) {
                let model = tinyLlama()
                var iterator = BatchTokenIterator(
                    model: model, prefill: PrefillParameters(stepSize: 2))
                let id = iterator.insert(prompt: Array(1 ..< 8))
                let first = try iterator.next()
                #expect(first?.map(describe) == ["prefill(\(id), 2/7)"])
                withUnsafeCurrentTask { $0?.cancel() }
                #expect(throws: CancellationError.self) { _ = try iterator.next() }
                #expect(iterator.pendingIDs == [id])
                iterator.remove([id])
                let idle = try iterator.next()
                #expect(idle == nil)
            }
        }.value
    }

    @Test("Chunks written in place keep the cache's buffer: the chunk markers hold no view of it")
    func resumablePrefillReusesCacheBuffer() throws {
        let model = InPlaceChunkModel()
        // Chunks of 2 into one 256-position buffer, with no reallocation.
        let prefill = PrefillParameters(stepSize: 2)
        let prompt = Array(1 ..< 11)
        let chunks = model.prefillChunks(
            for: LMInput.Text(tokens: MLXArray(prompt.map { Int32($0) })), prefill: prefill)
        var iterator = BatchTokenIterator(model: model, maxTokens: 1, prefill: prefill)
        iterator.insert(prompt: prompt)
        while try iterator.next() != nil {}
        // The buffer before every chunk after the first: one allocation, written in place.
        #expect(chunks.count > 2)
        #expect(model.keyBuffers.count == chunks.count - 1)
        #expect(Set(model.keyBuffers).count == 1, "\(model.keyBuffers)")
    }

    @Test("A model whose chunks leave cache writes lazy resumes after a cancelled step")
    func resumablePrefillLazyChunks() async throws {
        // Carries the iterator from the cancelled task to a fresh one; they run in turn.
        final class Carrier: @unchecked Sendable {
            var iterator: BatchTokenIterator?
            var id = -1
        }
        let prompt = Array(1 ..< 8)
        let carrier = Carrier()
        try await Task {
            try Device.withDefaultDevice(.cpu) {
                var iterator = BatchTokenIterator(
                    model: LazyChunkModel(tinyLlama()), maxTokens: 6,
                    prefill: PrefillParameters(stepSize: 2))
                let id = iterator.insert(prompt: prompt)
                let first = try iterator.next()
                #expect(first?.map(describe) == ["prefill(\(id), 2/7)"])
                withUnsafeCurrentTask { $0?.cancel() }
                #expect(throws: CancellationError.self) { _ = try iterator.next() }
                #expect(iterator.pendingIDs == [id])
                carrier.iterator = iterator
                carrier.id = id
            }
        }.value
        try await Task {
            try Device.withDefaultDevice(.cpu) {
                let expected = Array(
                    IteratorSequence(
                        try TokenIterator(
                            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })),
                            model: tinyLlama(), processor: nil, sampler: ArgMaxSampler(),
                            prefill: PrefillParameters(stepSize: 2), maxTokens: 6)))
                var iterator = try #require(carrier.iterator)
                var steps: [String] = []
                var output: [Int] = []
                while let events = try iterator.next() {
                    steps += events.map(describe)
                    for case .token(_, let token) in events { output.append(token) }
                }
                let id = try #require(carrier.id >= 0 ? carrier.id : nil, "the first task ran")
                #expect(output == expected)
                #expect(
                    Array(steps.prefix(3)) == [
                        "prefill(\(id), 4/7)", "prefill(\(id), 6/7)", "prefill(\(id), 7/7)",
                    ])
            }
        }.value
    }

    @Test("A removed row leaves the batch, queued or decoding")
    func remove() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            var iterator = BatchTokenIterator(model: model, maxTokens: 4)
            let uids = iterator.insert(prompts: Self.prompts)
            iterator.remove([uids[1]])
            #expect(iterator.ids == [uids[0], uids[2]])
            _ = try iterator.next()
            iterator.remove([uids[0]])
            #expect(iterator.ids == [uids[2]])
            // Only the remaining row appears from here on, and it runs to its limit.
            var seen = Set<Int>()
            var generated = 0
            while let events = try iterator.next() {
                seen.formUnion(events.map(\.id))
                generated += tokens(events).count
            }
            #expect(seen == [uids[2]])
            #expect(generated == 4)
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
            for case .token(let id, let token) in try iterator.next() ?? [] {
                tokens[id, default: []].append(token)
            }
        }
        uids += iterator.insert(prompts: [prompts[2]])
        while let events = try iterator.next() {
            for case .token(let id, let token) in events {
                tokens[id, default: []].append(token)
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
            let uids = iterator.insert(prompts: [[1]], caches: [caches])

            let failed = failures(try? iterator.next())
            #expect(failed.map(\.id) == uids)
            #expect(failed.allSatisfy { $0.error as? BatchPrefillObservationError == .stopped })
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

/// The model families besides Llama that opt into resumable prefill, by their `model_type`,
/// each with a tiny configuration for random weights.
enum ResumablePrefillFamily: String, CaseIterable, Sendable {
    case qwen2, qwen3
    case qwen3MoE = "qwen3_moe"
    case phi, phi3, gemma, gemma2, starcoder2, cohere
    case internlm2, granite, glm4, smollm3, olmo2, mixtral

    var configuration: String {
        // Keys each family does not read are ignored.
        let common = """
            "model_type": "\(rawValue)", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "rms_norm_eps": 1e-5, "layer_norm_eps": 1e-5, "norm_epsilon": 1e-5,
            "vocab_size": 64, "rope_theta": 10000.0, "max_position_embeddings": 128,
            "tie_word_embeddings": false, "attention_bias": false, "mlp_bias": false
            """
        let extra: String
        switch self {
        case .qwen3:
            extra = #""architectures": ["Qwen3ForCausalLM"]"#
        case .qwen3MoE:
            extra = """
                "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                "mlp_only_layers": [], "moe_intermediate_size": 32, "norm_topk_prob": true
                """
        case .mixtral:
            extra = #""num_local_experts": 4, "num_experts_per_tok": 2"#
        case .granite:
            extra = """
                "logits_scaling": 1.0, "attention_multiplier": 0.125, "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0
                """
        case .cohere:
            extra = #""logit_scale": 0.0625, "use_qk_norm": false"#
        case .starcoder2:
            extra = #""use_bias": true"#
        case .glm4, .phi:
            extra = #""partial_rotary_factor": 0.5"#
        case .internlm2:
            extra = #""bias": false"#
        case .gemma2:
            extra = """
                "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
                "query_pre_attn_scalar": 8.0
                """
        case .phi3:
            extra = #""original_max_position_embeddings": 128"#
        case .qwen2, .gemma, .smollm3, .olmo2:
            extra = #""rope_traditional": false"#
        }
        return "{\(common), \(extra)}"
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

/// A resumable model with one attention layer, written in place by each chunk. Before every
/// chunk after the first it records where the cache's key buffer lives.
private final class InPlaceChunkModel: Module, ResumablePrefillModel {
    private(set) var keyBuffers: [Int] = []

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let length = input.tokens.dim(-1)
        if let cache { write(cache, count: length) }
        return LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), length, 8]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }

    func prefillChunks(for prompt: LMInput.Text, prefill: PrefillParameters) -> [Range<Int>] {
        prefill.chunkRanges(total: prompt.tokens.size)
    }

    func prefillChunk(_ prompt: LMInput.Text, range: Range<Int>, cache: [KVCache]) {
        if let keys = cache[0].innerState().first {
            eval(keys)
            keyBuffers.append(
                keys.asData(access: .noCopy).data.withUnsafeBytes {
                    Int(bitPattern: $0.baseAddress)
                })
        }
        write(cache, count: range.count)
    }

    private func write(_ cache: [KVCache], count: Int) {
        let keys = MLXArray.ones([1, 2, count, 64])
        _ = cache[0].update(keys: keys, values: keys)
    }
}

/// A resumable model whose chunks leave their cache writes lazy: unlike `LLMModel`'s, its
/// `prefillChunk` neither evaluates nor submits the cache, so the iterator must.
private final class LazyChunkModel: Module, ResumablePrefillModel {
    let inner: LlamaModel

    init(_ inner: LlamaModel) {
        self.inner = inner
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        inner(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func prefillChunks(for prompt: LMInput.Text, prefill: PrefillParameters) -> [Range<Int>] {
        inner.prefillChunks(for: prompt, prefill: prefill)
    }

    func prefillChunk(_ prompt: LMInput.Text, range: Range<Int>, cache: [KVCache]) {
        _ = inner(prompt[.newAxis, range], cache: cache, state: nil)
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

private final class RecordingProcessorLog: @unchecked Sendable {
    var entries: [String] = []
}

/// Logs the emission hooks of one row; sampling passes through.
private struct RecordingProcessor: LogitProcessor {
    let name: String
    let log: RecordingProcessorLog

    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {}
    mutating func didEmit(token: Int) { log.entries.append("emit \(name) \(token)") }
    mutating func finalizeGeneration() { log.entries.append("finalize \(name)") }
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

/// Every row's tokens, in uid order, until the iterator is idle.
private func collect(_ iterator: inout BatchTokenIterator, uids: [Int]) throws -> [[Int]] {
    var results = [Int: [Int]]()
    while let events = try iterator.next() {
        for case .token(let id, let token) in events {
            results[id, default: []].append(token)
        }
    }
    return uids.map { results[$0] ?? [] }
}

/// The step's `.token` values, in batch order; empty for `nil`.
private func tokens(_ events: [BatchEvent]?) -> [Int] {
    (events ?? []).compactMap { if case .token(_, let token) = $0 { token } else { nil } }
}

/// The ids of the step's `.token` events, in batch order; empty for `nil`.
private func tokenIDs(_ events: [BatchEvent]?) -> [Int] {
    (events ?? []).compactMap { if case .token(let id, _) = $0 { id } else { nil } }
}

/// The step's `.failed` events, in order.
private func failures(_ events: [BatchEvent]?) -> [(id: Int, error: any Error)] {
    (events ?? []).compactMap {
        if case .failed(let id, let error) = $0 { (id, error) } else { nil }
    }
}

/// The step's `.finished` events, in order.
private func finishes(_ events: [BatchEvent]?) -> [BatchFinish] {
    (events ?? []).compactMap {
        if case .finished(let id, let reason, let stopToken) = $0 {
            BatchFinish(id: id, reason: reason, stopToken: stopToken)
        } else {
            nil
        }
    }
}

/// An event as text, to compare whole steps.
private func describe(_ event: BatchEvent) -> String {
    switch event {
    case .prefill(let id, let processed, let total): "prefill(\(id), \(processed)/\(total))"
    case .token(let id, let token): "token(\(id), \(token))"
    case .finished(let id, let reason, let stopToken):
        "finished(\(id), \(reason), \(stopToken.map(String.init) ?? "nil"))"
    case .failed(let id, let error): "failed(\(id), \(error))"
    }
}

private struct BatchFinish: Equatable {
    let id: Int
    let reason: GenerateStopReason
    let stopToken: Int?
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
                for case .token(let id, let token) in try iterator.next() ?? [] {
                    output[id, default: []].append(token)
                }
            }
            uids += iterator.insert(
                prompts: Array(prompts.dropFirst()),
                samplers: prompts.indices.dropFirst().map { sampler($0) })
            for _ in 0 ..< 2 {
                for case .token(let id, let token) in try iterator.next() ?? [] {
                    output[id, default: []].append(token)
                }
            }
            iterator.remove([uids[0]])
            #expect(iterator.ids == Array(uids.dropFirst()))
            while let events = try iterator.next() {
                for case .token(let id, let token) in events {
                    output[id, default: []].append(token)
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
