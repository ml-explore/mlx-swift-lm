// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct BatchKVCacheTests {

    @Test("Text-only Idefics3 uses each ragged row's rotary positions", arguments: [2, 3])
    func ideficsRaggedPositions(batchSize: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyIdefics3ForBatchPositions()
            let prompts = Array(
                [[1, 2, 3], Array(4 ..< 11), Array(11 ..< 20)].prefix(batchSize))
            var singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                let logits = model(MLXArray(prompt.map { Int32($0) })[.newAxis], cache: cache)
                eval(logits)
                eval(cache)
                return cache
            }
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer] })
            }
            let first = try #require(batch.first as? BatchKVCache)
            #expect(first.batchOffset.asArray(Int32.self) == prompts.map { Int32($0.count) })

            var generated = 0
            for length in [1, 2] {
                let tokens = MLXArray(
                    singles.indices.flatMap { row in
                        (0 ..< length).map { Int32(21 + row * 7 + generated + $0) }
                    }
                ).reshaped([singles.count, length])
                try expectIdeficsChunk(model, tokens: tokens, batch: batch, singles: singles)
                generated += length
            }

            let keep = batchSize == 2 ? [1] : [2, 0]
            let indices = MLXArray(keep.map { Int32($0) })
            for layer in batch { layer.filter(batchIndices: indices) }
            singles = keep.map { singles[$0] }
            if batchSize == 2 {
                guard case .scalar(let position) = first.ropeOffset else {
                    Issue.record("the surviving single row must retain its scalar RoPE path")
                    return
                }
                #expect(position == singles[0][0].offset)
            }
            let afterFilter = MLXArray(
                singles.indices.flatMap { row in [Int32(41 + row), Int32(44 + row)] }
            ).reshaped([singles.count, 2])
            try expectIdeficsChunk(model, tokens: afterFilter, batch: batch, singles: singles)

            let admitted = try model.newCache(parameters: nil)
            let prefix = MLXArray([Int32(30), 31, 32, 33, 34])[.newAxis]
            eval(model(prefix, cache: admitted))
            eval(admitted)
            for layer in batch.indices {
                let added = try mergeCaches([admitted[layer]])
                try batch[layer].validateExtension(other: added)
                batch[layer].extend(other: added)
            }
            singles.append(admitted)
            let afterAdmission = MLXArray(
                singles.indices.flatMap { row in [Int32(47 + row), Int32(50 + row)] }
            ).reshaped([singles.count, 2])
            try expectIdeficsChunk(model, tokens: afterAdmission, batch: batch, singles: singles)
        }
    }

    @Test("The row methods and merge dispatch through the BatchableKVCache type")
    func protocolSurface() throws {
        let rows: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let batch = try mergeCaches(rows)
        #expect(batch is BatchKVCache)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = batch.update(keys: keys, values: values)
        batch.filter(batchIndices: batchRows(1))
        #expect((batch as? BatchKVCache)?.batchSize == 1)
        let added = try mergeCaches([KVCacheSimple()])
        _ = added.update(keys: keys[0 ..< 1], values: values[0 ..< 1])
        batch.extend(other: added)
        #expect((batch as? BatchKVCache)?.batchSize == 2)
        let row = try #require(batch.extract(0) as? KVCacheSimple)
        let rowKeys = try #require(row.state.first)
        #expect(maxAbsDifference(rowKeys, MLXArray.ones(rowKeys.shape) * 2) == 0)
    }

    @Test("Extension checks refuse incompatible full-precision storage")
    func extensionValidation() throws {
        try Device.withDefaultDevice(.cpu) {
            let full = BatchKVCache(leftPadding: [0])
            let kv = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4)
            _ = full.update(keys: kv.0, values: kv.1)
            let compatible = BatchKVCache(leftPadding: [0])
            _ = compatible.update(keys: kv.0, values: kv.1)
            try full.validateExtension(other: compatible)
            #expect(full.batchSize == 1)
            full.extend(other: compatible)
            #expect(full.batchSize == 2)

            let wrongWidth = BatchKVCache(leftPadding: [0])
            let wider = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 8)
            _ = wrongWidth.update(keys: wider.0, values: wider.1)
            #expect(throws: KVCacheError.self) { try full.validateExtension(other: wrongWidth) }
            let prepared = BatchKVCache(leftPadding: [0, 0])
            prepared.prepare(lengths: [1, 2])
            #expect(throws: KVCacheError.self) { try full.validateExtension(other: prepared) }

            // A cache without a batched form has no extension check to call; admission
            // refuses it when it merges the rows.
            #expect(!(BaseKVCache() as any KVCache is any BatchableKVCache))
            #expect(throws: KVCacheError.self) { _ = try mergeCaches([BaseKVCache()]) }
        }
    }

    @Test("Only batched caches take row operations; merging any other cache throws")
    func batchableConformance() throws {
        let singles: [any KVCache] = [KVCacheSimple(), ChunkedKVCache(chunkSize: 4)]
        for cache in singles {
            #expect(!(cache is any BatchableKVCache), "\(type(of: cache))")
        }
        #expect(BatchKVCache(leftPadding: [0]) as any KVCache is any BatchableKVCache)
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([ChunkedKVCache(chunkSize: 4)]) }
        #expect(throws: KVCacheError.self) { _ = try BatchKVCache.merge([any KVCache]()) }
        // A batch is not a row: merging it again throws instead of nesting batches.
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([BatchKVCache(leftPadding: [0])])
        }
    }

    @Test("Update, filter, extend and extract keep each row's tokens")
    func lifecycle() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: batchRows(0))
        #expect(cache.batchSize == 1)
        #expect(cache.leftPadding.asArray(Int32.self) == [0])

        let added = BatchKVCache(leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)

        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(cache.batchOffset.asArray(Int32.self) == [3, 2])
        #expect(cache.extract(0).offset == 3)

        let second = cache.extract(1)
        #expect(second.offset == 2)
        #expect(maxAbsDifference(try #require(second.state.first), addedKV.0) == 0)
        #expect(maxAbsDifference(try #require(second.state.last), addedKV.1) == 0)
    }

    @Test("Filtering during a ragged prefill keeps the survivors' right padding")
    func filterDuringRaggedPrefill() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        cache.filter(batchIndices: batchRows(1))
        cache.finalize()

        let row = cache.extract(0)
        #expect(row.offset == 5)
        #expect(cache.leftPadding.asArray(Int32.self) == [0])
        let keys = try #require(row.state.first)
        #expect(maxAbsDifference(keys, MLXArray.ones(keys.shape) * 2) == 0)
        let values = try #require(row.state.last)
        #expect(maxAbsDifference(values, MLXArray.ones(values.shape) * 20) == 0)
    }

    @Test("Filtering drops no more shared left padding than the buffer holds")
    func filterBeforePaddingIsWritten() throws {
        // A left-padded prompt fed in chunks: the first two positions are padding in both rows.
        let cache = BatchKVCache(leftPadding: [3, 4])
        let padding = makeKV(batchSize: 2, heads: 2, seqLen: 2, headDim: 4, value: 0)
        _ = cache.update(keys: padding.0, values: padding.1)

        cache.filter(batchIndices: batchRows(0, 1))
        try #require(cache.offset == 0)
        #expect(cache.leftPadding.asArray(Int32.self) == [1, 2])
        #expect(cache.batchOffset.asArray(Int32.self) == [-1, -2])

        let rest = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: rest.0, values: rest.1)
        #expect(cache.batchOffset.asArray(Int32.self) == [2, 1])
        let row = cache.extract(0)
        #expect(row.offset == 2)
        let keys = try #require(row.state.first)
        #expect(maxAbsDifference(keys, rest.0[0 ..< 1, 0..., 1..., 0...]) == 0)
    }

    @Test("extract refuses a row index past the batch")
    func extractOutOfRange() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let cache = BatchKVCache(leftPadding: [0, 1])
                let kv = makeKV(batchSize: 2, heads: 1, seqLen: 3, headDim: 1)
                _ = cache.update(keys: kv.0, values: kv.1)
                _ = cache.extract(2)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "BatchKVCache row index out of range"))
    }

    @Test("extract returns an empty row when its padding covers every written position")
    func extractPaddingOnlyRow() throws {
        let cache = BatchKVCache(leftPadding: [4, 0])
        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 1, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        let padded = try #require(cache.extract(0) as? KVCacheSimple)
        #expect(padded.offset == 0)
        #expect(padded.keys == nil)
        #expect(padded.values == nil)
        let written = cache.extract(1)
        #expect(written.offset == 1)
        #expect(maxAbsDifference(try #require(written.state.first), kv.0[1 ..< 2]) == 0)
    }

    @Test("extract refuses a ragged prefill before finalize")
    func extractBeforeFinalize() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let cache = BatchKVCache(leftPadding: [0, 0])
                cache.prepare(lengths: [1, 3])
                let kv = makeKV(batchSize: 2, heads: 1, seqLen: 3, headDim: 1)
                _ = cache.update(keys: kv.0, values: kv.1)
                _ = cache.extract(0)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "BatchKVCache.extract takes finalized rows"))
    }

    @Test("merge throws for rows it cannot stack, also a single row")
    func mergeRefusesMismatchedRows() throws {
        func row(
            heads: Int = 2, headDim: Int = 4, rows: Int = 1, dtype: DType = .float32
        ) -> KVCacheSimple {
            let cache = KVCacheSimple()
            let kv = makeKV(batchSize: rows, heads: heads, seqLen: 3, headDim: headDim)
            _ = cache.update(keys: kv.0.asType(dtype), values: kv.1.asType(dtype))
            return cache
        }
        // Rows of one class that disagree on layout, also behind an empty first row.
        let mismatched: [[any KVCache]] = [
            [row(), row(headDim: 8)],
            [KVCacheSimple(), row(), row(headDim: 8)],
            [row(), row(heads: 4)],
            [row(), row(dtype: .float16)],
        ]
        for rows in mismatched {
            #expect(throws: KVCacheError.self) { _ = try mergeCaches(rows) }
        }
        // A cache holding two rows is not a row, also alone: admission merges one at a time.
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([row(rows: 2)]) }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([row(), row(rows: 2)]) }

        let merged = try mergeCaches([KVCacheSimple(), row(), row()])
        #expect((merged as? BatchKVCache)?.batchSize == 3)
    }

    @Test("finalize keeps every row's tokens when the buffer has spare capacity")
    func finalizeWithSpareCapacity() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        // Row 0 holds 3 tokens and 2 right-padding slots, row 1 holds 5 tokens.
        var stamps = [Float](repeating: 0, count: 10)
        for row in 0 ..< 2 {
            for position in 0 ..< (row == 0 ? 3 : 5) {
                stamps[row * 5 + position] = Float(10 * (row + 1) + position)
            }
        }
        let keys = MLXArray(stamps, [2, 1, 5, 1])
        _ = cache.update(keys: keys, values: keys * 2)

        cache.finalize()

        let first = try #require(cache.extract(0) as? KVCacheSimple)
        #expect(first.offset == 3)
        #expect(try #require(first.keys).asArray(Float.self) == [10, 11, 12])
        #expect(try #require(first.values).asArray(Float.self) == [20, 22, 24])
        let second = try #require(cache.extract(1) as? KVCacheSimple)
        #expect(second.offset == 5)
        #expect(try #require(second.keys).asArray(Float.self) == [20, 21, 22, 23, 24])
        #expect(try #require(second.values).asArray(Float.self) == [40, 42, 44, 46, 48])
    }

    @Test("merge of one cache and extract round-trip a sequence")
    func mergeRoundTrip() throws {
        let single = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 4, heads: 2, headDim: 4)
        _ = single.update(keys: kv.0, values: kv.1)

        let batch = KVCacheSimple.merge([single])
        #expect(batch.batchSize == 1)

        let restored = batch.extract(0)
        #expect(restored.offset == single.offset)
        #expect(
            maxAbsDifference(try #require(restored.state.first), try #require(single.state.first))
                == 0)
        #expect(
            maxAbsDifference(try #require(restored.state.last), try #require(single.state.last))
                == 0)

        // Writing to the batch leaves the source unchanged.
        let next = makePositionKV(positions: 4 ..< 5, heads: 2, headDim: 4)
        _ = batch.update(keys: next.0, values: next.1)
        #expect(single.offset == 4)
        #expect(maxAbsDifference(try #require(single.state.first), kv.0) == 0)
    }

    @Test("merge left-pads ragged rows and takes empty caches as empty rows")
    func mergeRaggedRows() throws {
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let merged = KVCacheSimple.merge([longer, KVCacheSimple()])
        #expect(merged.leftPadding.asArray(Int32.self) == [0, 2])
        #expect(merged.batchOffset.asArray(Int32.self) == [2, 0])
        #expect(merged.offset == 2)
        #expect(merged.extract(0).offset == 2)
        #expect(merged.extract(1).offset == 0)

        // Two empty caches merge into an empty two-row batch.
        let empty = KVCacheSimple.merge([KVCacheSimple(), KVCacheSimple()])
        #expect(empty.batchSize == 2)
        #expect(empty.offset == 0)
    }

    @Test("trim drops the newest tokens, no more than the shortest row holds")
    func trimClampsToShortestRow() throws {
        let cache = BatchKVCache(leftPadding: [2, 0])
        let kv = makePositionKV(
            positions: 0 ..< 3, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.trim(2) == 1)
        #expect(cache.batchOffset.asArray(Int32.self) == [0, 2])
        #expect(cache.extract(0).offset == 0)
        let second = cache.extract(1)
        #expect(second.offset == 2)
        #expect(try #require(second.state.first)[0, 0, 0..., 0].asArray(Float.self) == [10, 11])
        #expect(try #require(second.state.last)[0, 0, 0..., 0].asArray(Float.self) == [110, 111])
    }

    @Test("maxTrimCount reports only what trim can remove")
    func maxTrimCountMatchesTrim() {
        let cache = BatchKVCache(leftPadding: [2, 0])
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.maxTrimCount == 1)
        #expect(!canTrimPromptCache([cache], numTokens: 2))
        #expect(canTrimPromptCache([cache], numTokens: 1))
        #expect(cache.trim(1) == 1)
    }

    @Test(
        "Staged rounds refuse writes that cannot rewind every row's padding",
        arguments: [0, 2], [1, 2, 5])
    func paddedRoundIsRefused(written: Int, width: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchKVCache(leftPadding: [3, 0])
            if written > 0 {
                let kv = makePositionKV(
                    positions: 0 ..< written, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
                _ = cache.update(keys: kv.0, values: kv.1)
            }
            let state = cache.state.map { $0.asArray(Int32.self) }
            let storage = KVCacheStorage([cache], plan: .disabled)
            #expect(cache.isTrimmable)
            #expect(!cache.isTrimmable(after: width))
            let refused = storage.beginRound(maximumPositions: width)
            #expect(refused == nil)
            // Old code admits the round. Keep the negative regression safe by writing nothing.
            if let refused { storage.rollback(refused) }
            #expect(cache.offset == written)
            #expect(cache.state.map { $0.asArray(Int32.self) } == state)
            #expect(storage.processedTokenCount == written)
            #expect(storage.nativeAttentionOffsetsAreAligned)

            #expect(cache.isTrimmable(after: 0))
            let empty = try #require(storage.beginRound(maximumPositions: 0))
            storage.rollback(empty)
            #expect(cache.state.map { $0.asArray(Int32.self) } == state)
            #expect(storage.nativeAttentionOffsetsAreAligned)
        }
    }

    @Test("A zero-row batch refuses positive staged rounds", arguments: [false, true])
    func zeroRowRoundIsRefused(filtered: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchKVCache(leftPadding: filtered ? [0] : [])
            if filtered {
                let kv = makeKV(batchSize: 1, heads: 1, seqLen: 3, headDim: 1)
                _ = cache.update(keys: kv.0, values: kv.1)
                cache.filter(batchIndices: batchRows())
            }
            let storage = KVCacheStorage([cache], plan: .disabled)
            #expect(!cache.isTrimmable(after: 2))
            let refused = storage.beginRound(maximumPositions: 2)
            #expect(refused == nil)
            if let refused { storage.rollback(refused) }
            #expect(cache.batchSize == 0)
            #expect(cache.offset == 0)
            #expect(storage.processedTokenCount == 0)
            #expect(storage.nativeAttentionOffsetsAreAligned)

            #expect(cache.isTrimmable(after: 0))
            let empty = try #require(storage.beginRound(maximumPositions: 0))
            storage.rollback(empty)
            #expect(cache.offset == 0)
            #expect(storage.nativeAttentionOffsetsAreAligned)
        }
    }

    @Test(
        "A staged round rewinds exactly after every row has consumed its padding",
        arguments: [0, 1])
    func roundAfterPaddingCommitsExactly(retaining: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchKVCache(leftPadding: [3, 0])
            let prefix = makePositionKV(
                positions: 0 ..< 3, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
            _ = cache.update(keys: prefix.0, values: prefix.1)
            let storage = KVCacheStorage([cache], plan: .disabled)
            #expect(cache.isTrimmable(after: 2))
            #expect(!cache.isTrimmable(after: -1))
            let round = try #require(storage.beginRound(maximumPositions: 2))
            let proposed = makePositionKV(
                positions: 3 ..< 5, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
            _ = round.caches[0].update(keys: proposed.0, values: proposed.1)
            let result = storage.commit(round, retaining: retaining)

            #expect(result.committedPositions == retaining)
            #expect(result.discardedPositions == 2 - retaining)
            #expect(cache.offset == 3 + retaining)
            #expect(
                cache.batchOffset.asArray(Int32.self) == [Int32(retaining), Int32(3 + retaining)])
            #expect(storage.processedTokenCount == 3 + retaining)
            #expect(storage.nativeAttentionOffsetsAreAligned)
            let expected = makePositionKV(
                positions: 0 ..< 3 + retaining, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
            let state = cache.state
            #expect(maxAbsDifference(state[0], expected.0) == 0)
            #expect(maxAbsDifference(state[1], expected.1) == 0)
        }
    }

    @Test("Empty rows join empty rows, and the batch fills on its first update")
    func extendEmptyRows() throws {
        let cache = BatchKVCache(leftPadding: [0])
        cache.extend(other: BatchKVCache(leftPadding: [1]))
        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(cache.batchOffset.asArray(Int32.self) == [0, -1])

        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache.batchOffset.asArray(Int32.self) == [3, 2])
        #expect(cache.extract(1).offset == 2)
    }

    @Test("Prefilled rows join empty rows as padding, and a zero-row batch changes nothing")
    func extendMixedRows() throws {
        let cache = BatchKVCache(leftPadding: [0])
        let added = BatchKVCache(leftPadding: [0])
        let kv = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = added.update(keys: kv.0, values: kv.1)
        cache.extend(other: added)
        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [3, 0])
        #expect(cache.batchOffset.asArray(Int32.self) == [0, 3])
        #expect(cache.extract(0).offset == 0)
        #expect(maxAbsDifference(try #require(cache.extract(1).state.first), kv.0) == 0)

        // The empty row's first token lands after its padding, the other row's after its history.
        let next = makePositionKV(positions: 10 ..< 11, heads: 2, headDim: 4, batchSize: 2)
        _ = cache.update(keys: next.0, values: next.1)
        #expect(cache.batchOffset.asArray(Int32.self) == [1, 4])
        #expect(
            try #require(cache.extract(0).state.first)[0, 0, 0..., 0].asArray(Float.self) == [10])
        #expect(cache.extract(1).offset == 4)

        // The other way round, and a batch with no rows.
        let populated = BatchKVCache(leftPadding: [0])
        _ = populated.update(keys: kv.0, values: kv.1)
        populated.extend(other: BatchKVCache(leftPadding: [0]))
        populated.extend(other: BatchKVCache(leftPadding: []))
        #expect(populated.batchSize == 2)
        #expect(populated.leftPadding.asArray(Int32.self) == [0, 3])
        #expect(populated.batchOffset.asArray(Int32.self) == [3, 0])
    }

    @Test("An empty cache extended from another does not share its arrays")
    func extendEmptyCacheCopies() throws {
        let source = BatchKVCache(leftPadding: [0])
        let prompt = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = source.update(keys: prompt.0, values: prompt.1)
        let cache = BatchKVCache(leftPadding: [])
        cache.extend(other: source)

        // Both writes land inside the buffer the source allocated.
        let next = makePositionKV(positions: 50 ..< 51, heads: 2, headDim: 4)
        _ = cache.update(keys: next.0, values: next.1)
        let sourceNext = makePositionKV(positions: 70 ..< 71, heads: 2, headDim: 4)
        _ = source.update(keys: sourceNext.0, values: sourceNext.1)

        let keys = try #require(cache.extract(0).state.first)
        #expect(keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 50])
        let sourceKeys = try #require(source.extract(0).state.first)
        #expect(sourceKeys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 70])
    }

    @Test("One token with no left padding needs no mask; padding brings it back")
    func oneTokenWithoutPaddingHasNoMask() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        guard case .none = cache.makeMask(n: 1, windowSize: nil, returnArray: false) else {
            Issue.record("unpadded rows must decode one token without a mask")
            return
        }
        guard case .array = cache.makeMask(n: 2, windowSize: nil, returnArray: false) else {
            Issue.record("more than one token keeps the explicit mask")
            return
        }

        // Joining a shorter row pads it, so the one-token mask returns.
        let added = BatchKVCache(leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("a padded batch must mask its padding")
            return
        }
        #expect(mask[2, 0, 0, 0].item(Bool.self) == false)

        // Keeping only the padded row makes its padding shared, which filter drops.
        cache.filter(batchIndices: batchRows(2))
        guard case .none = cache.makeMask(n: 1, windowSize: nil, returnArray: false) else {
            Issue.record("padding dropped by filter must not keep the mask")
            return
        }
        #expect(cache.batchOffset.asArray(Int32.self) == [2])
        #expect(cache.offset == 2)
    }

    @Test("A window smaller than the history keeps the one-token mask")
    func windowKeepsOneTokenMask() throws {
        let cache = BatchKVCache(leftPadding: [0])
        let (keys, values) = makeDistinctKV(batchSize: 1, heads: 2, seqLen: 8, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        guard case .none = cache.makeMask(n: 1, windowSize: 9, returnArray: false) else {
            Issue.record("a window covering every key needs no mask")
            return
        }
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 4, returnArray: false)
        else {
            Issue.record("a window smaller than the history must keep the mask")
            return
        }
        try #require(mask.shape.last == 9)
        // The query at position 8 sees positions 5...8 only.
        #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
        #expect(mask[0, 0, 0, 5].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 8].item(Bool.self) == true)
    }

    @Test("The mask excludes each row's left padding, also for one token")
    func maskExcludesLeftPadding() throws {
        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        for n in [1, 2] {
            guard case .array(let mask) = cache.makeMask(n: n, windowSize: nil, returnArray: false)
            else {
                Issue.record("BatchKVCache must return an array mask")
                return
            }
            try #require(mask.dim(0) == 3)
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[2, 0, 0, 0].item(Bool.self) == true)
        }
        guard case .array(let mask) = cache.makeMask(n: 2, windowSize: nil, returnArray: false)
        else { return }
        // Row 0 (padding 1) attends position 1; row 1 (padding 3) does not.
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
    }

    @Test("ropeOffset is per row through the KVCache type")
    func ropeOffsetThroughKVCache() {
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        let batch = BatchKVCache(leftPadding: [1, 0])
        _ = batch.update(keys: kv.0, values: kv.1)
        let cache: any KVCache = batch
        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record("BatchKVCache.ropeOffset must be .batch")
            return
        }
        #expect(offsets.asArray(Int32.self) == [2, 3])
    }

    @Test("ropeOffset is scalar while every row shares one position")
    func ropeOffsetScalarForSharedPosition() {
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        let batch = BatchKVCache(leftPadding: [0, 0])
        _ = batch.update(keys: kv.0, values: kv.1)
        guard case .scalar(let offset) = batch.ropeOffset else {
            Issue.record("rows at one position must use the scalar offset")
            return
        }
        #expect(offset == 3)
        #expect(batch.batchOffset.asArray(Int32.self) == [3, 3])

        // A shorter row joining the batch gives the rows different positions.
        let added = BatchKVCache(leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        batch.extend(other: added)
        guard case .batch(let offsets) = batch.ropeOffset else {
            Issue.record("rows at different positions must use per-row offsets")
            return
        }
        #expect(offsets.asArray(Int32.self) == [3, 3, 2])
    }

    @Test("createCausalMask masks each row's left padding")
    func causalMaskWithLeftPadding() throws {
        let mask = createCausalMask(n: 4, offset: 0, leftPadding: MLXArray([Int32(1), 2]))
        try #require(mask.shape == [2, 1, 4, 4])
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 3, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 2, 2].item(Bool.self) == true)
    }

    @Test("The cache status counts every row's bytes as one full-precision attention layer")
    func statusCountsRows() throws {
        let rows = (0 ..< 3).map { _ in KVCacheSimple() }
        for row in rows {
            let kv = makeKV(batchSize: 1, heads: 2, seqLen: 5, headDim: 8)
            _ = row.update(keys: kv.0, values: kv.1)
        }
        let batch = KVCacheSimple.merge(rows)

        let status = KVCacheStatus(cache: [batch])
        try #require(status.layers.count == 1)
        #expect(status.layers[0].kind == .attention(maxSize: nil))
        #expect(status.layers[0].state == .active)
        #expect(status.layers[0].resolvedStrategy == .fullPrecision)
        let rowBytes = (0 ..< 3).map { batch.extract($0) }.reduce(0) { bytes, row in
            bytes + row.innerState().reduce(0) { $0 + $1.nbytes }
        }
        #expect(status.memoryBytes == rowBytes)

        // A batch holds full-precision rows only; compression skips it rather than rewriting it.
        let affine = kvCacheLayerStatuses(
            cache: [batch], configuration: KVCacheConfiguration(strategy: .affine(.fourBit)))
        #expect(affine[0].state == .skipped)
        #expect(affine[0].reason == .unsupportedShape)
    }

    @Test("A batched cache cannot be saved as a prompt cache")
    func savePromptCacheRefuses() {
        let cache = BatchKVCache(leftPadding: [0, 1])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("batch-cache-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [cache])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

/// Batched caches must give every row the logits and the keys and values it gets on its own.
@Suite(.serialized)
struct BatchedDecodeParityTests {

    @Test(
        "Mistral 3 text preserves each row's scaling with and without llama4 scaling",
        arguments: [Optional<Float>.none, .some(0), .some(1)])
    func mistral3Scaling(beta: Float?) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyMistral3Text(beta: beta)
            try expectBatchedRowsMatchSingles(model)
            try expectMergedRowsMatchSingles(model)
        }
    }

    @Test("Mistral 3 scaling follows ragged rows through chunks, admission and removal")
    func mistral3ScalingLifecycle() throws {
        try Device.withDefaultDevice(.cpu) {
            try expectScalingLifecycle(tinyMistral3Text(beta: 1))
        }
    }

    @Test("Nemotron Labs Diffusion AR scaling follows each ragged row")
    func nemotronLabsScaling() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                NemotronLabsDiffusionConfiguration.self,
                from: Data(
                    """
                    {
                      "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                      "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                      "rms_norm_eps": 1e-5, "vocab_size": 64, "rope_theta": 10000.0,
                      "rope_parameters": {
                        "llama_4_scaling_beta": 1.0, "original_max_position_embeddings": 4
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 706)) {
                let model = NemotronLabsDiffusionModel(configuration)
                eval(model)
                return model
            }
            try expectScalingLifecycle(model)
        }
    }

    @Test("Batched llama4 scaling keeps negative padding finite and scales the batch axis")
    func negativePaddingScale() {
        Device.withDefaultDevice(.cpu) {
            let scale = batchedLlama4AttentionScale(
                offsets: MLXArray([Int32(-3), 1]), length: 5, beta: 1,
                maxPositionEmbeddings: 4)
            #expect(scale.shape == [2, 1, 5, 1])
            let actual = scale.asArray(Float.self)
            let allFinite = actual.allSatisfy(\.isFinite)
            #expect(allFinite)
            #expect(Array(actual.prefix(5)) == Array(repeating: Float(1), count: 5))
            #expect(abs(actual[8] - (1 + log(Float(2)))) < 1e-6)
            let scaledQueries = MLXArray.ones([2, 4, 5, 8]) * scale
            #expect(scaledQueries.shape == [2, 4, 5, 8])
            #expect(scaledQueries[0].max().item(Float.self) == 1)
            #expect(abs(scaledQueries[1].max().item(Float.self) - (1 + log(Float(2)))) < 1e-6)
        }
    }

    @Test("A ragged batch of Llama prompts decodes like separate sequences")
    func llama() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            try expectBatchedRowsMatchSingles(model)
            try expectMergedRowsMatchSingles(model)
        }
    }

    @Test("A ragged batch of Gemma 2 prompts decodes like separate sequences")
    func gemma2() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma2()
            try expectBatchedRowsMatchSingles(model)
            try expectMergedRowsMatchSingles(model)
        }
    }

    @Test("Rows at one position behind a wider buffer decode at that position, not the buffer's")
    func logicalPositionBehindPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let prompts: [[Int32]] = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
            let singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: cache))
                return cache
            }
            // Two positions of left padding in both rows: the buffer is eight wide, the rows six.
            let widths: [IntOrPair] = [0, 0, .init((2, 0)), 0]
            let batch = singles[0].indices.map { layer in
                let cache = BatchKVCache(leftPadding: [2, 2])
                let rows = singles.map { $0[layer].state }
                _ = cache.update(
                    keys: concatenated(rows.map { MLX.padded($0[0], widths: widths) }),
                    values: concatenated(rows.map { MLX.padded($0[1], widths: widths) }))
                return cache
            }
            // `offset` counts the rows' six tokens, not the eight buffer positions, so cache
            // storage adopts the batch at the position a single-row cache would report.
            #expect(batch[0].offset == 6)
            #expect(try #require(batch[0].state.first).dim(2) == 8)
            #expect(batch[0].batchOffset.asArray(Int32.self) == [6, 6])
            let storage = KVCacheStorage(batch, plan: .disabled)
            #expect(storage.processedTokenCount == 6)
            #expect(storage.nativeAttentionOffsetsAreAligned)
            guard case .scalar(let position) = batch[0].ropeOffset else {
                Issue.record("rows at one position must use the scalar offset")
                return
            }
            #expect(position == 6)

            for step in 0 ..< 3 {
                let token = Int32(13 + step)
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batch)
                for row in prompts.indices {
                    let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                        "step \(step), row \(row)")
                }
            }
            #expect(batch[0].batchOffset.asArray(Int32.self) == [9, 9])
        }
    }
}

private func tinyMistral3Text(beta: Float?) -> Mistral3TextModel {
    let configuration = Mistral3TextConfiguration(
        hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64, attentionHeads: 4,
        rmsNormEps: 1e-5, vocabularySize: 64, headDimensions: 8, kvHeads: 2,
        ropeParameters: beta.map {
            ["llama_4_scaling_beta": .float($0), "original_max_position_embeddings": .int(4)]
        })
    return withRandomState(MLXRandom.RandomState(seed: 705)) {
        let model = Mistral3TextModel(configuration)
        eval(model)
        return model
    }
}

/// Ragged positions cross scaling bands in batches both smaller than and equal to the head count.
private func expectScalingLifecycle(_ model: any LanguageModel) throws {
    func prefill(_ prompt: [Int32]) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt)[.newAxis], cache: cache))
        return cache
    }
    var singles = try [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10]].map(prefill)
    let batch = try singles[0].indices.map { layer in
        try mergeCaches(singles.map { $0[layer].copy() })
    }
    func decode(_ tokens: [[Int32]]) {
        let length = tokens[0].count
        let output = model(
            MLXArray(tokens.flatMap { $0 }).reshaped([tokens.count, length]), cache: batch)
        for row in tokens.indices {
            let expected = model(MLXArray(tokens[row])[.newAxis], cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                "row \(row), chunk length \(length)")
        }
    }
    decode([[11, 12], [13, 14]])

    let added = try [[Int32(15), 16], [17, 18, 19, 20, 21, 22]].map(prefill)
    let incoming = try added[0].indices.map { layer in
        try mergeCaches(added.map { $0[layer].copy() })
    }
    for layer in batch.indices {
        try batch[layer].validateExtension(other: incoming[layer])
    }
    for layer in batch.indices {
        batch[layer].extend(other: incoming[layer])
    }
    singles += added
    decode([[23, 24], [25, 26], [27, 28], [29, 30]])

    batch.forEach { $0.filter(batchIndices: batchRows(3, 0)) }
    singles = [singles[3], singles[0]]
    decode([[31, 32, 33], [34, 35, 36]])
    batch.forEach { $0.filter(batchIndices: batchRows(1)) }
    singles = [singles[1]]
    decode([[37]])
}

func tinyLlama() -> LlamaModel {
    withRandomState(MLXRandom.RandomState(seed: 7)) {
        let model = LlamaModel(
            LlamaConfiguration(
                hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
                attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
        eval(model)
        return model
    }
}

/// The caches of `model`, as the single-sequence caches a batch is built from.
private func singleCaches(_ model: any LanguageModel) throws -> [KVCacheSimple] {
    try model.newCache(parameters: nil).map { cache in
        try #require(cache as? KVCacheSimple, "the parity models use plain caches")
    }
}

/// Merges fresh caches, prefills two prompts of different lengths as one right-padded batch,
/// decodes four tokens and drops a row, comparing every row's logits and extracted keys and
/// values with the same prompt run alone.
private func expectBatchedRowsMatchSingles(_ model: any LanguageModel) throws {
    let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8]]
    let longest = prompts.map(\.count).max()!

    var singles: [[any KVCache]] = []
    var expected: [MLXArray] = []
    for prompt in prompts {
        let cache = try model.newCache(parameters: nil)
        let logits = model(MLXArray(prompt).reshaped([1, prompt.count]), cache: cache)
        singles.append(cache)
        expected.append(logits[0..., (prompt.count - 1) ..< prompt.count, 0...])
    }

    // One fresh single cache per row, merged per layer into the empty batch.
    let fresh = try prompts.map { _ in try singleCaches(model) }
    let batched = fresh[0].indices.map { layer in KVCacheSimple.merge(fresh.map { $0[layer] }) }
    let caches = batched.map { $0 as any KVCache }
    batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
    let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
    let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: caches)
    batched.forEach { $0.finalize() }
    for (row, prompt) in prompts.enumerated() {
        let actual = logits[row ..< row + 1, (prompt.count - 1) ..< prompt.count, 0...]
        #expect(maxAbsDifference(actual, expected[row]) < 1e-4, "prefill, row \(row)")
    }
    try expectExtractedRowsMatch(batched, singles)

    for step in 0 ..< 4 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                "step \(step), row \(row)")
        }
    }
    try expectExtractedRowsMatch(batched, singles)

    batched.forEach { $0.filter(batchIndices: batchRows(1)) }
    let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: caches)
    let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
    try expectExtractedRowsMatch(batched, [singles[1]])
}

/// Prefills two prompts alone, merges their caches and decodes three tokens, comparing
/// every row with the same sequence continued alone.
private func expectMergedRowsMatchSingles(_ model: any LanguageModel) throws {
    let prompts: [[Int32]] = [[1, 2, 3, 4, 5, 6], [7, 8]]
    var singles: [[any KVCache]] = []
    var rows: [[KVCacheSimple]] = []
    for prompt in prompts {
        let single = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: single))
        singles.append(single)
        let row = try singleCaches(model)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: row))
        rows.append(row)
    }

    let batched = rows[0].indices.map { layer in KVCacheSimple.merge(rows.map { $0[layer] }) }
    #expect(batched[0].batchOffset.asArray(Int.self) == prompts.map(\.count))
    #expect(batched[0].leftPadding.asArray(Int.self) == [0, 4])
    let caches = batched.map { $0 as any KVCache }
    try expectExtractedRowsMatch(batched, singles)

    for step in 0 ..< 3 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                "step \(step), row \(row)")
        }
    }
    try expectExtractedRowsMatch(batched, singles)
}

/// Every row extracted from `batch` holds the offset, keys and values of its single cache.
private func expectExtractedRowsMatch(_ batch: [BatchKVCache], _ singles: [[any KVCache]])
    throws
{
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = batch[layer].extract(row)
            #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            let expected = caches[layer].state
            try #require(actual.state.count == expected.count)
            for (actualArray, expectedArray) in zip(actual.state, expected) {
                try #require(actualArray.shape == expectedArray.shape)
                #expect(
                    maxAbsDifference(actualArray, expectedArray) < 1e-4,
                    "row \(row), layer \(layer)")
            }
        }
    }
}

/// Keys and values with the same value at every position. Use this only for shape and
/// counter checks: it cannot show which positions an operation kept or their order.
func makeKV(
    batchSize: Int, heads: Int, seqLen: Int, headDim: Int, value: Float = 1
) -> (MLXArray, MLXArray) {
    let keys = MLXArray.ones([batchSize, heads, seqLen, headDim]) * value
    let values = MLXArray.ones([batchSize, heads, seqLen, headDim]) * (value + 1)
    return (keys, values)
}

/// Keys of `i + 1` and values of `(i + 1) * 10` in row `i`.
func makeDistinctKV(batchSize: Int, heads: Int, seqLen: Int, headDim: Int) -> (MLXArray, MLXArray) {
    let rows = MLXArray((0 ..< batchSize).map { Float($0 + 1) }).reshaped([batchSize, 1, 1, 1])
    let ones = MLXArray.ones([batchSize, heads, seqLen, headDim])
    return (ones * rows, ones * rows * 10)
}

/// Keys stamped with their position (plus `row * rowStride`), and values with the
/// key plus 100, so tests can check which positions survive and in what order.
func makePositionKV(
    positions: Range<Int>, heads: Int, headDim: Int, batchSize: Int = 1, rowStride: Float = 0
) -> (MLXArray, MLXArray) {
    var stamps = [Float]()
    for row in 0 ..< batchSize {
        stamps.append(contentsOf: positions.map { Float($0) + Float(row) * rowStride })
    }
    let stamp = MLXArray(stamps).reshaped([batchSize, 1, positions.count, 1])
    let ones = MLXArray.ones([batchSize, heads, positions.count, headDim])
    return (ones * stamp, ones * (stamp + 100))
}

func batchRows(_ indices: Int...) -> MLXArray {
    MLXArray(indices.map { Int32($0) })
}

/// Two layers of grouped-query attention: four heads over two key-value heads.
func tinyGemma2Configuration() throws -> Gemma2Configuration {
    try JSONDecoder().decode(
        Gemma2Configuration.self,
        from: Data(
            """
            {
              "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
              "num_attention_heads": 4, "head_dim": 8, "rms_norm_eps": 1e-6,
              "vocab_size": 32, "num_key_value_heads": 2, "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 144.0
            }
            """.utf8))
}

func tinyGemma2() throws -> Gemma2Model {
    let configuration = try tinyGemma2Configuration()
    return withRandomState(MLXRandom.RandomState(seed: 13)) {
        let model = Gemma2Model(configuration)
        eval(model)
        return model
    }
}

/// A text-only fixture with 64-wide KV heads, shared with packed-cache parity tests.
func tinyIdefics3ForBatchPositions() throws -> Idefics3 {
    let json = """
        {
            "model_type": "idefics3", "vocab_size": 64, "scale_factor": 1,
            "image_token_id": 63, "image_token_index": 63,
            "text_config": {
                "model_type": "llama", "hidden_size": 128, "num_hidden_layers": 2,
                "intermediate_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
                "rms_norm_eps": 1e-5, "vocab_size": 64, "rope_theta": 10000.0,
                "tie_word_embeddings": false
            },
            "vision_config": {
                "model_type": "siglip_vision_model", "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 32, "patch_size": 8,
                "image_size": 16, "num_channels": 3
            }
        }
        """
    let configuration = try JSONDecoder().decode(
        Idefics3Configuration.self, from: Data(json.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 913)) {
        let model = Idefics3(configuration)
        model.train(false)
        eval(model)
        return model
    }
}

private func expectIdeficsChunk(
    _ model: Idefics3, tokens: MLXArray, batch: [KVCache], singles: [[KVCache]],
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let actual = model(tokens, cache: batch)
    eval(actual)
    eval(batch)
    try #require(
        actual.shape == [singles.count, tokens.dim(1), 64], sourceLocation: sourceLocation)
    for row in singles.indices {
        let expected = model(tokens[row ..< row + 1], cache: singles[row])
        eval(expected)
        eval(singles[row])
        #expect(
            maxAbsDifference(actual[row ..< row + 1], expected) < 1e-3,
            "row \(row) must use its own rotary positions", sourceLocation: sourceLocation)
    }
}
