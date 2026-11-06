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

    @Test("The row methods and merge dispatch through the KVCache type")
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

            #expect(throws: KVCacheError.self) {
                try BaseKVCache().validateExtension(other: full)
            }
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
            #expect(batch[0].offset == 8)
            #expect(batch[0].batchOffset.asArray(Int32.self) == [6, 6])
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

@Suite(.serialized)
struct BatchedHybridCacheTests {

    @Test("Recurrent merge refuses populated rows with different slot counts")
    func recurrentMergeRejectsSlotCountMismatch() {
        Device.withDefaultDevice(.cpu) {
            let two = ArraysCache(size: 2)
            two[0] = MLXArray([Float(1), 2, 3, 4])[.newAxis]
            two[1] = MLXArray([Float(5), 6, 7, 8])[.newAxis]
            let one = ArraysCache(size: 1)
            one[0] = MLXArray([Float(9), 10, 11, 12])[.newAxis]
            expectRejectedMerge([two, one])
            expectRejectedMerge([one, two], nested: true)
        }
    }

    @Test(
        "Recurrent merge validates sparse slots and each row's metadata",
        arguments: MergeLayoutIssue.allCases)
    func recurrentMergeRejectsMalformedRows(issue: MergeLayoutIssue) {
        Device.withDefaultDevice(.cpu) {
            expectRejectedMerge(malformedRows(issue))
        }
    }

    @Test("Typed and erased recurrent merges preserve prepared sparse rows")
    func recurrentMergePreservesPreparedSparseRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let first = MambaCache(leftPadding: [2, 4])
            first[1] = MLXArray([Float(5), 6, 7, 8, 9, 10]).reshaped([2, 3])
            first.prepare(lengths: [5, 7])
            first.offset = 7
            let second = MambaCache()
            second[0] = MLXArray([Float(1), 2, 3, 4])[.newAxis]
            second.offset = 3
            let typed = ArraysCache.merge([first, second])
            let erased = try #require(try mergeCaches([first, second]) as? MambaCache)
            for merged in [typed, erased] {
                #expect(merged.offset == 7)
                #expect(merged.leftPaddingValues == [2, 4, 0])
                #expect(merged.lengthsValues == [5, 7, 0])
                #expect(try #require(merged[0]).shape == [3, 4])
                #expect(try #require(merged[1]).shape == [3, 3])
                #expect(
                    try #require(merged[0]).asArray(Float.self)
                        == [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
                #expect(
                    try #require(merged[1]).asArray(Float.self) == [5, 6, 7, 8, 9, 10, 0, 0, 0])
            }
            #expect(first.offset == 7)
            #expect(first.leftPaddingValues == [2, 4])
            #expect(first.lengthsValues == [5, 7])
            #expect(second.offset == 3)
            #expect(second[1] == nil)
            #expect(throws: KVCacheError.self) {
                _ = try ArraysCache.merge([any KVCache]())
            }
        }
    }

    enum MergeLayoutIssue: CaseIterable, Sendable {
        case emptySlotCounts
        case emptyTemplateSlotCounts
        case sparseShapes
        case sparseDtypes
        case scalarState
        case slotRowCounts
        case paddingRowCounts
        case lengthRowCounts
        case lengthRank
    }

    private func malformedRows(_ issue: MergeLayoutIssue) -> [ArraysCache] {
        let first = ArraysCache(size: 2)
        switch issue {
        case .emptySlotCounts:
            return [first, ArraysCache(size: 1)]
        case .emptyTemplateSlotCounts:
            let second = ArraysCache(size: 1)
            second[0] = MLXArray.ones([1, 4])
            return [first, second]
        case .sparseShapes, .sparseDtypes:
            let second = ArraysCache(size: 2)
            let third = ArraysCache(size: 2)
            second[1] = MLXArray.ones([1, 4])
            third[1] =
                issue == .sparseShapes
                ? MLXArray.ones([1, 8]) : MLXArray.ones([1, 4], dtype: .float16)
            return [first, second, third]
        case .scalarState:
            first[0] = MLXArray(Float(1))
        case .slotRowCounts:
            first[0] = MLXArray.ones([2, 4])
            first[1] = MLXArray.ones([1, 4])
        case .paddingRowCounts:
            let padded = ArraysCache(size: 2, leftPadding: [0])
            padded[0] = MLXArray.ones([2, 4])
            return [padded]
        case .lengthRowCounts:
            first[0] = MLXArray.ones([2, 4])
            first.prepare(lengths: [1])
        case .lengthRank:
            first[0] = MLXArray.ones([2, 4])
            first.prepare(lengths: MLXArray([Int32(1), 2]).reshaped([1, 2]))
        }
        return [first]
    }

    private func expectRejectedMerge(_ rows: [ArraysCache], nested: Bool = false) {
        let before = rows.map { row in
            (
                offset: row.offset, metadata: row.metaState,
                shapes: row.state.map(\.shape), dtypes: row.state.map(\.dtype),
                values: row.state.map { $0.asType(.float32).asArray(Float.self) }
            )
        }
        let metadataShapes = rows.map { [$0.leftPadding?.shape, $0.lengths?.shape] }
        let caches: [KVCache] = rows.map { row -> KVCache in
            if nested { return CacheList(KVCacheSimple(), CacheList(row)) }
            return row
        }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches(caches) }
        for (row, snapshot) in zip(rows, before) {
            #expect(row.offset == snapshot.offset)
            #expect(row.metaState == snapshot.metadata)
            #expect(row.state.map(\.shape) == snapshot.shapes)
            #expect(row.state.map(\.dtype) == snapshot.dtypes)
            #expect(row.state.map { $0.asType(.float32).asArray(Float.self) } == snapshot.values)
        }
        #expect(rows.map { [$0.leftPadding?.shape, $0.lengths?.shape] } == metadataShapes)
    }

    @Test("Recurrent extension checks refuse incompatible state and custom subclasses")
    func extensionValidation() throws {
        try Device.withDefaultDevice(.cpu) {
            let recurrent = ArraysCache(size: 2)
            recurrent[0] = MLXArray.zeros([1, 4])
            let wrongSlots = ArraysCache(size: 1)
            #expect(throws: KVCacheError.self) {
                try recurrent.validateExtension(other: wrongSlots)
            }
            let wrongState = ArraysCache(size: 2)
            wrongState[0] = MLXArray.zeros([1, 8])
            #expect(throws: KVCacheError.self) {
                try recurrent.validateExtension(other: wrongState)
            }
            let empty = ArraysCache(size: 2)
            try recurrent.validateExtension(other: empty)
            recurrent.extend(other: empty)
            #expect(recurrent[0]?.shape == [2, 4])

            class CustomArraysCache: ArraysCache {}
            let custom = CustomArraysCache(size: 2)
            #expect(throws: KVCacheError.self) { try custom.validateExtension(other: custom) }
            #expect(throws: KVCacheError.self) { _ = try mergeCaches([custom]) }
        }
    }

    @Test(
        "Zero-row recurrent caches cannot be saved, including nested and metadata-only batches",
        arguments: [false, true], [false, true])
    func zeroRowsAreRefused(mamba: Bool, populated: Bool) {
        Device.withDefaultDevice(.cpu) {
            let recurrent: ArraysCache = mamba ? MambaCache() : ArraysCache(size: 2)
            if populated { recurrent[0] = MLXArray.ones([2, 4]) }
            recurrent.filter(batchIndices: MLXArray([Int32]()))
            #expect(recurrent.batchSize == 0)
            let caches: [KVCache] = [
                recurrent, CacheList(recurrent), CacheList(KVCacheSimple(), CacheList(recurrent)),
            ]
            for cache in caches {
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("zero-rows-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: url) }
                #expect(throws: KVCacheError.self) {
                    try savePromptCache(url: url, cache: [cache])
                }
                #expect(FileManager.default.fileExists(atPath: url.path) == false)
            }
        }
    }

    @Test("An unpopulated single-row recurrent cache still saves and restores")
    func emptySingleRowCanBeSaved() throws {
        try Device.withDefaultDevice(.cpu) {
            for cache in [ArraysCache(size: 2), MambaCache()] {
                #expect(cache.batchSize == 1)
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("empty-row-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: url) }
                try savePromptCache(url: url, cache: [cache])
                let loaded = try loadPromptCache(url: url)
                let restored = try #require(loaded.0.first as? ArraysCache)
                #expect(type(of: restored) == type(of: cache))
                #expect(restored.batchSize == 1)
            }
        }
    }

    @Test(
        "Saving refuses only a recurrent batch whose rows nothing records, also in a list",
        arguments: [false, true])
    func unrecordedRecurrentRowsAreRefused(mamba: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            func row(leftPadding: [Int]? = nil) -> ArraysCache {
                mamba
                    ? MambaCache(leftPadding: leftPadding)
                    : ArraysCache(size: 2, leftPadding: leftPadding)
            }
            func roundTrip(_ cache: KVCache) throws -> KVCache? {
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("recurrent-rows-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: url) }
                try savePromptCache(url: url, cache: [cache])
                return try loadPromptCache(url: url).0.first
            }

            // Two empty rows: no array, padding or length in the file would record the second.
            let empty = ArraysCache.merge([row(), row()])
            #expect(empty.batchSize == 2)
            let holders: [KVCache] = [
                empty, CacheList(KVCacheSimple(), empty), CacheList(CacheList(empty)),
            ]
            for cache in holders {
                #expect(throws: KVCacheError.self) { _ = try roundTrip(cache) }
            }

            // Padding records the rows of a batch without state.
            let padded = try #require(try roundTrip(row(leftPadding: [0, 1])) as? ArraysCache)
            #expect(padded.batchSize == 2)
            #expect(padded.leftPadding?.asArray(Int.self) == [0, 1])

            // State arrays record the rows of a populated batch.
            let populated = row()
            populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
            let restored = try #require(try roundTrip(populated) as? ArraysCache)
            #expect(type(of: restored) == type(of: populated))
            #expect(restored.batchSize == 2)
            #expect(restored[0]?.asArray(Float.self) == (0 ..< 8).map(Float.init))
        }
    }

    @Test("A file with four-field recurrent metadata loads", arguments: [false, true])
    func fourFieldRecurrentMetadataLoads(mamba: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache: ArraysCache =
                mamba
                ? MambaCache(leftPadding: [1, 0]) : ArraysCache(size: 2, leftPadding: [1, 0])
            cache[1] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
            cache.prepare(lengths: [2, 3])
            // [slotCount, presentSlots, leftPadding, lengths], the format `main` writes.
            #expect(cache.metaState == ["2", "1", "1,0", "2,3"])
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("four-field-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: [cache])
            let restored = try #require(try loadPromptCache(url: url).0.first as? ArraysCache)
            #expect(type(of: restored) == type(of: cache))
            #expect(restored.metaState == cache.metaState)
            #expect(restored.batchSize == 2)
            #expect(restored[0] == nil)
            #expect(restored[1]?.asArray(Float.self) == (0 ..< 8).map(Float.init))
            #expect(restored.leftPadding?.asArray(Int.self) == [1, 0])
            #expect(restored.lengthsValues == [2, 3])
        }
    }

    @Test(
        "Empty recurrent batches keep every row through later joins, and refuse to save",
        arguments: [false, true], [2, 3])
    func emptyBatchRowsSurviveJoinsButNotSaving(mamba: Bool, rowCount: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            func emptyRow() -> ArraysCache { mamba ? MambaCache() : ArraysCache(size: 2) }
            for finalized in [false, true] {
                let batch = ArraysCache.merge((0 ..< rowCount).map { _ in emptyRow() })
                if finalized {
                    batch.prepare(lengths: Array(1 ... rowCount))
                    batch.finalize()
                }
                // No array, padding or length records the rows, so no file could restore them.
                #expect(batch.metaState == ["2", ""])
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("empty-batch-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: url) }
                #expect(throws: KVCacheError.self) {
                    try savePromptCache(url: url, cache: [batch])
                }
                #expect(FileManager.default.fileExists(atPath: url.path) == false)
                #expect(batch.batchSize == rowCount)
                #expect(batch.slotCount == 2)
                #expect(batch.state.isEmpty)
                #expect(batch.leftPadding == nil)
                #expect(batch.lengths == nil)

                let populated = emptyRow()
                populated[0] = MLXArray.ones([1, 4])
                let joined = try #require(try mergeCaches([batch, populated]) as? ArraysCache)
                #expect(joined[0]?.shape == [rowCount + 1, 4])
                #expect(
                    joined[0]?.asArray(Float.self)
                        == Array(repeating: Float(0), count: rowCount * 4)
                        + Array(repeating: 1, count: 4))
                batch.extend(other: emptyRow())
                #expect(batch.batchSize == rowCount + 1)
                batch.prepare(lengths: Array(1 ... rowCount + 1))
                #expect(batch.leftPadding == nil)
                let mask = try #require(batch.makeMask(N: rowCount + 1))
                #expect(mask.shape == [rowCount + 1, rowCount + 1])
                #expect(
                    mask.asArray(Bool.self)
                        == (1 ... rowCount + 1).flatMap { length in
                            (0 ... rowCount).map { $0 < length }
                        })
            }
        }
    }

    @Test(
        "Nested lists keep empty recurrent rows through joins, and refuse to save",
        arguments: [2, 3])
    func nestedEmptyRecurrentRowsSurviveJoinsButNotSaving(rowCount: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            func emptyRow() -> CacheList {
                CacheList(ArraysCache(size: 3), CacheList(MambaCache()))
            }
            let batch = try CacheList.merge((0 ..< rowCount).map { _ in emptyRow() })
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("nested-empty-batch-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: KVCacheError.self) {
                try savePromptCache(url: url, cache: [batch])
            }
            #expect(FileManager.default.fileExists(atPath: url.path) == false)
            let arrays = try #require(batch[0] as? ArraysCache)
            let nested = try #require(batch[1] as? CacheList)
            let mamba = try #require(nested[0] as? MambaCache)
            #expect(arrays.batchSize == rowCount)
            #expect(arrays.slotCount == 3)
            #expect(mamba.batchSize == rowCount)
            #expect(mamba.slotCount == 2)

            let populated = emptyRow()
            let populatedArrays = try #require(populated[0] as? ArraysCache)
            let populatedNested = try #require(populated[1] as? CacheList)
            let populatedMamba = try #require(populatedNested[0] as? MambaCache)
            populatedArrays[1] = MLXArray.ones([1, 4])
            populatedMamba[0] = MLXArray.ones([1, 8])
            let joined = try CacheList.merge([batch, populated])
            let joinedArrays = try #require(joined[0] as? ArraysCache)
            let joinedNested = try #require(joined[1] as? CacheList)
            let joinedMamba = try #require(joinedNested[0] as? MambaCache)
            #expect(joinedArrays[1]?.shape == [rowCount + 1, 4])
            #expect(joinedMamba[0]?.shape == [rowCount + 1, 8])
            batch.extend(other: emptyRow())
            #expect(arrays.batchSize == rowCount + 1)
            #expect(mamba.batchSize == rowCount + 1)
        }
    }

    @Test("Recurrent restore keeps legacy metadata and signed masks", arguments: [false, true])
    func recurrentRowCountMetadataCompatibility(mamba: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cacheType: ArraysCache.Type = mamba ? MambaCache.self : ArraysCache.self
            let legacy = try #require(
                try cacheType.restore(state: [MLXArray.ones([2, 4])], metaState: [""])
                    as? ArraysCache)
            #expect(type(of: legacy) == cacheType)
            #expect(legacy.batchSize == 2)
            #expect(legacy.slotCount == 1)
            for metadata in [
                ["2", ""], ["2", "", "-1,1"], ["2", "", "", "1,3"],
                ["2", "", "-1,1", "1,3"],
            ] {
                let restored = try #require(
                    try cacheType.restore(state: [], metaState: metadata) as? ArraysCache)
                #expect(restored.metaState == metadata)
                #expect(restored.batchSize == (metadata.count == 2 ? 1 : 2))
            }
            let populated = try #require(
                try cacheType.restore(state: [MLXArray.ones([2, 4])], metaState: ["2", "0"])
                    as? ArraysCache)
            #expect(populated.metaState == ["2", "0"])
            #expect(populated.batchSize == 2)
            let lengthsOnly = try #require(
                try cacheType.restore(state: [], metaState: ["2", "", "", "-1,2"])
                    as? ArraysCache)
            #expect(lengthsOnly.leftPadding == nil)
            #expect(lengthsOnly.lengthsValues == [-1, 2])
            #expect(
                lengthsOnly.makeMask(N: 3)?.asArray(Bool.self) == [
                    false, false, false, true, true, false,
                ])
            lengthsOnly.finalize()
            // Finalized, the rows are kept in memory but no saved field records them.
            #expect(lengthsOnly.metaState == ["2", ""])
            #expect(lengthsOnly.batchSize == 2)
            let withPadding = try #require(
                try cacheType.restore(
                    state: [MLXArray.ones([2, 4])],
                    metaState: ["2", "0", "-1,1", "-1,2"]) as? ArraysCache)
            #expect(
                withPadding.makeMask(N: 3)?.asArray(Bool.self) == [
                    true, true, true, false, true, true,
                ])
        }
    }

    @Test(
        "Recurrent restore rejects invalid encoded row counts and layouts",
        arguments: [false, true])
    func corruptRecurrentRowCountMetadataIsRejected(mamba: Bool) throws {
        try Device.withDefaultDevice(.cpu) { () throws -> Void in
            let cacheType: ArraysCache.Type = mamba ? MambaCache.self : ArraysCache.self
            for count in ["", "0", "-1", "two", "999999999999999999999999"] {
                #expect(throws: KVCacheError.self) {
                    try cacheType.restore(state: [], metaState: ["2", "", "", "", count])
                }
            }
            let malformed: [([MLXArray], [String])] = [
                ([], ["2", "", "", "", "2", "extra"]),
                ([], ["-1", "", "", "", "2"]),
                ([], ["invalid", "", "", "", "2"]),
                ([], ["2", "", "0", "", "2"]),
                ([], ["2", "", "", "1,2,3", "2"]),
                ([], ["2", "", "", "1,bad", "2"]),
                ([MLXArray.ones([1, 4])], ["2", "0", "", "", "2"]),
                ([MLXArray(Float(1))], ["2", "0", "", "", "2"]),
                ([MLXArray.ones([2, 4])], ["2", "-1", "", "", "2"]),
                ([MLXArray.ones([2, 4])], ["2", "2", "", "", "2"]),
                ([MLXArray.ones([2, 4])], ["2", "bad", "", "", "2"]),
                ([MLXArray.ones([2, 4])], ["2", "0,0", "", "", "2"]),
                ([], ["2", "0", "", "", "2"]),
            ]
            for (state, metadata) in malformed {
                #expect(throws: KVCacheError.self) {
                    try cacheType.restore(state: state, metaState: metadata)
                }
            }
        }
    }

    @Test("A recurrent cache prepares, finalizes and filters its rows")
    func recurrentLifecycle() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        mamba.prepare(lengths: [3, 5])
        #expect(mamba.lengths != nil)
        mamba.finalize()
        #expect(mamba.lengths == nil)

        mamba.filter(batchIndices: batchRows(0))
        #expect(mamba.batchSize == 1)
        #expect(
            maxAbsDifference(
                try #require(mamba[0]), MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])) == 0)
        #expect(
            maxAbsDifference(
                try #require(mamba[1]), MLXArray(8 ..< 12).asType(.float32).reshaped([1, 4])) == 0)
    }

    @Test("A merged empty recurrent batch masks by the prepared lengths")
    func mergedRecurrentMasksByLengths() throws {
        let cache = try #require(ArraysCache.merge([MambaCache(), MambaCache()]) as? MambaCache)
        #expect(cache.leftPadding == nil, "empty rows carry no padding that would hide lengths")
        #expect(cache.batchSize == 2)
        cache.prepare(lengths: [1, 2])
        #expect(cache.batchSize == 2)

        let mask = try #require(cache.makeMask(N: 2))
        #expect(mask.shape == [2, 2])
        #expect(mask.asArray(Bool.self) == [true, false, true, true])
    }

    @Test("merge keeps every empty recurrent row ahead of a populated one")
    func mergeKeepsEmptyRecurrentRows() throws {
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        populated[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        let merged = try #require(
            ArraysCache.merge([MambaCache(), MambaCache(), populated]) as? MambaCache)
        #expect(merged.batchSize == 4)
        let first = try #require(merged[0])
        #expect(first.shape == [4, 4])
        #expect(first[0 ..< 2].sum().item(Float.self) == 0, "empty rows hold zero state")
        #expect(maxAbsDifference(first[2 ..< 4], try #require(populated[0])) == 0)
        #expect(maxAbsDifference(try #require(merged[1])[2 ..< 4], try #require(populated[1])) == 0)

        // A hybrid list merged the same way keeps its children's rows aligned:
        // two empty sequences and one with recurrent state make three rows.
        let one = MambaCache()
        one[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4])
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
            CacheList(KVCacheSimple(), one),
        ])
        #expect((list[0] as? BatchKVCache)?.batchSize == 3)
        #expect((list[1] as? MambaCache)?.batchSize == 3)
    }

    @Test(
        "Extending a recurrent batch keeps the longest row's offset, as merge does",
        arguments: [(3, 7), (7, 3)])
    func recurrentExtensionKeepsLongestOffset(offsets: (Int, Int)) throws {
        try Device.withDefaultDevice(.cpu) {
            func row(_ offset: Int) -> ArraysCache {
                let cache = ArraysCache(size: 2)
                cache[0] = MLXArray([Float(offset)]).reshaped([1, 1])
                cache.offset = offset
                return cache
            }
            let active = row(offsets.0)
            let joined = row(offsets.1)
            try active.validateExtension(other: joined)
            active.extend(other: joined)
            #expect(active.batchSize == 2)
            #expect(active.offset == 7)
            #expect(joined.offset == offsets.1)

            // Through the protocol, as admission extends an active layer.
            let erased: any KVCache = row(offsets.0)
            erased.extend(other: row(offsets.1))
            #expect(erased.offset == 7)

            // A side with no rows contributes no position.
            func noRows() -> ArraysCache {
                let cache = ArraysCache(size: 2)
                cache.filter(batchIndices: MLXArray([Int32]()))
                cache.offset = 9
                return cache
            }
            let fromEmpty = noRows()
            fromEmpty.extend(other: row(offsets.1))
            #expect(fromEmpty.offset == offsets.1)
            let withEmpty = row(offsets.0)
            withEmpty.extend(other: noRows())
            #expect(withEmpty.offset == offsets.0)
        }
    }

    @Test("A recurrent batch with zero rows stays empty until extended")
    func recurrentZeroRows() throws {
        let empty = MambaCache(leftPadding: [])
        #expect(empty.batchSize == 0)
        #expect(empty.leftPaddingValues == [])
        #expect(empty.lengths == nil)

        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        empty.extend(other: populated)
        #expect(empty.batchSize == 2)
        #expect(maxAbsDifference(try #require(empty[0]), try #require(populated[0])) == 0)
    }

    @Test("extract returns the requested row and keeps the MambaCache type")
    func extractKeepsMambaCache() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        for row in 0 ..< 2 {
            let extracted = try #require(mamba.extract(row) as? MambaCache)
            #expect(
                maxAbsDifference(
                    try #require(extracted[0]),
                    MLXArray((row * 4) ..< (row * 4 + 4)).asType(.float32).reshaped([1, 4]))
                    == 0)
            #expect(
                maxAbsDifference(
                    try #require(extracted[1]),
                    MLXArray((8 + row * 4) ..< (12 + row * 4)).asType(.float32).reshaped([1, 4]))
                    == 0)
        }
    }

    @Test("CacheList.merge batches each child position on its own type")
    func compositeMerge() throws {
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache(), RotatingKVCache(maxSize: 16)),
            CacheList(KVCacheSimple(), MambaCache(), RotatingKVCache(maxSize: 16)),
        ])
        try #require(list.children.count == 3)
        #expect(list[0] is BatchKVCache)
        #expect(list[1] is MambaCache)
        #expect(list[2] is BatchRotatingKVCache)
        #expect((list[0] as? BatchKVCache)?.batchSize == 2)
        #expect((list[0] as? BatchKVCache)?.offset == 0)
        #expect((list[2] as? BatchRotatingKVCache)?.batchSize == 2)
        #expect((list[2] as? BatchRotatingKVCache)?.maxSize == 16)

        #expect(throws: KVCacheError.self) {
            _ = try CacheList.merge([
                CacheList(KVCacheSimple(), TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4))
            ])
        }
        #expect(throws: KVCacheError.self) {
            _ = try CacheList.merge([
                CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple()),
            ])
        }
        for cache in [
            RotatingKVCache(maxSize: 16, keep: 4), RotatingKVCache(maxSize: 16, rewindCapacity: 4),
        ] {
            #expect(throws: KVCacheError.self) {
                _ = try CacheList.merge([CacheList(KVCacheSimple(), cache)])
            }
        }
    }

    @Test("mergeCaches refuses caches without a batched form and rows of mixed types")
    func mergeCachesRefusals() throws {
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([ChunkedKVCache(chunkSize: 16)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([KVCacheSimple(), RotatingKVCache(maxSize: 4)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([])
        }
        #expect(try mergeCaches([KVCacheSimple(), KVCacheSimple()]) is BatchKVCache)
        #expect(try mergeCaches([MambaCache(), MambaCache()]) is MambaCache)
    }

    @Test("A merged CacheList filters, extends and extracts every child")
    func compositeLifecycle() throws {
        func composite() -> CacheList { CacheList(KVCacheSimple(), MambaCache()) }
        let list = try CacheList.merge([composite(), composite()])
        #expect(list.children.count == 2)
        let attention = try #require(list[0] as? BatchKVCache)
        #expect(attention.batchSize == 2)
        let recurrent = try #require(list[1] as? MambaCache)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = attention.update(keys: keys, values: values)
        recurrent[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])

        list.filter(batchIndices: batchRows(1))
        #expect(attention.batchSize == 1)
        #expect(recurrent.batchSize == 1)

        let added = try CacheList.merge([composite()])
        #expect(added.children.count == 2)
        let addedKV = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: addedKV.0, values: addedKV.1)
        let addedRecurrent = try #require(added[1] as? MambaCache)
        addedRecurrent[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4])
        list.extend(other: added)
        #expect(attention.batchSize == 2)
        #expect(recurrent.batchSize == 2)

        let first = try #require(list.extract(0) as? CacheList)
        let firstAttention = try #require(first[0] as? KVCacheSimple)
        let firstKeys = try #require(firstAttention.state.first)
        #expect(maxAbsDifference(firstKeys, MLXArray.ones(firstKeys.shape) * 2) == 0)
        let second = try #require(list.extract(1) as? CacheList)
        let secondAttention = try #require(second[0] as? KVCacheSimple)
        let secondKeys = try #require(secondAttention.state.first)
        #expect(maxAbsDifference(secondKeys, addedKV.0) == 0)
        let secondRecurrent = try #require(second[1] as? MambaCache)
        let secondState = try #require(secondRecurrent[0])
        #expect(
            maxAbsDifference(secondState, MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4]))
                == 0)
    }

    @Test("An all-empty merge keeps its row count through a later merge and extension")
    func allEmptyMergeKeepsRows() throws {
        let empty = try #require(ArraysCache.merge([MambaCache(), MambaCache()]) as? MambaCache)
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])
        let merged = try #require(ArraysCache.merge([empty, populated]) as? MambaCache)
        #expect(merged.batchSize == 3)
        #expect(try #require(merged[0]).shape == [3, 4])

        // An all-empty composite batch takes prefilled rows with its children aligned.
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
        ])
        let added = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: kv.0, values: kv.1)
        try #require(added[1] as? MambaCache)[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([
            1, 4,
        ])
        list.extend(other: added)
        #expect((list[0] as? BatchKVCache)?.batchSize == 3)
        #expect((list[1] as? MambaCache)?.batchSize == 3)
        #expect(try #require((list[1] as? MambaCache)?[0]).shape == [3, 4])
    }

    @Test(
        "Late padding or lengths preserve every preceding empty recurrent row",
        arguments: [false, true])
    func lateRecurrentMetadataKeepsRows(usePadding: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let last = MambaCache(leftPadding: usePadding ? [0] : nil)
            if !usePadding { last.prepare(lengths: [5]) }
            let merged = try #require(
                ArraysCache.merge([MambaCache(), MambaCache(), last]) as? MambaCache)
            try #require(merged.batchSize == 3)
            #expect(merged.leftPaddingValues == (usePadding ? [0, 0, 0] : nil))
            #expect(merged.lengthsValues == (usePadding ? nil : [0, 0, 5]))

            merged.finalize()
            #expect(merged.batchSize == 3)
            #expect(merged.makeMask(N: 2) == nil)
            let nested = try #require(ArraysCache.merge([merged, MambaCache()]) as? MambaCache)
            try #require(nested.batchSize == 4)
            #expect(nested.leftPadding == nil)

            let populated = MambaCache()
            populated[0] = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 4])
            let admitted = try #require(ArraysCache.merge([nested, populated]) as? MambaCache)
            try #require(admitted.batchSize == 5)
            let state = try #require(admitted[0])
            #expect(state.shape == [5, 4])
            #expect(state[0 ..< 4].sum().item(Float.self) == 0)
            #expect(maxAbsDifference(state[4 ..< 5], try #require(populated[0])) == 0)

            admitted.filter(batchIndices: batchRows(4, 2))
            #expect(admitted.batchSize == 2)
            #expect(
                maxAbsDifference(try #require(admitted[0])[0 ..< 1], try #require(populated[0]))
                    == 0)
            #expect(try #require(admitted[0])[1 ..< 2].sum().item(Float.self) == 0)
        }
    }

    @Test(
        "Recurrent merge and generic extend preserve mixed metadata in every order",
        arguments: 0 ..< 3)
    func recurrentMetadataOrderKeepsRows(metadataIndex: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let metadata = MambaCache(leftPadding: [2, 4])
            metadata.prepare(lengths: [5, 7])
            var rows = [MambaCache(), MambaCache()]
            rows.insert(metadata, at: metadataIndex)
            let padding =
                Array(repeating: 0, count: metadataIndex) + [2, 4]
                + Array(repeating: 0, count: 2 - metadataIndex)
            let lengths =
                Array(repeating: 0, count: metadataIndex) + [5, 7]
                + Array(repeating: 0, count: 2 - metadataIndex)
            let merged = ArraysCache.merge(rows)
            let extended: any KVCache = rows[0].copy()
            for row in rows.dropFirst() { extended.extend(other: row) }

            for cache in [merged, try #require(extended as? ArraysCache)] {
                try #require(cache.batchSize == 4)
                #expect(cache.leftPaddingValues == padding)
                #expect(cache.lengthsValues == lengths)
                let mask = try #require(cache.makeMask(N: 3))
                #expect(
                    mask.asArray(Bool.self)
                        == padding.flatMap { start in
                            (0 ..< 3).map { $0 >= start }
                        })
                cache.filter(batchIndices: batchRows(3, 1))
                #expect(cache.leftPaddingValues == [padding[3], padding[1]])
                #expect(cache.lengthsValues == [lengths[3], lengths[1]])
                cache.finalize()
                #expect(cache.batchSize == 2)
                #expect(cache.makeMask(N: 3) == nil)
            }
        }
    }

    @Test("Empty recurrent filtering and composite admission keep their rows aligned")
    func emptyRecurrentFilterAndCompositeAdmissionKeepRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let unpadded: any KVCache = MambaCache()
            unpadded.extend(other: MambaCache())
            let recurrent = try #require(unpadded as? MambaCache)
            #expect(recurrent.batchSize == 2)
            #expect(recurrent.leftPadding == nil)
            #expect(recurrent.makeMask(N: 2) == nil)
            recurrent.prepare(lengths: [1, 2])
            #expect(recurrent.batchSize == 2)
            #expect(
                try #require(recurrent.makeMask(N: 2)).asArray(Bool.self) == [
                    true, false, true, true,
                ])
            recurrent.finalize()
            #expect(recurrent.batchSize == 2)
            #expect(recurrent.makeMask(N: 2) == nil)
            recurrent.filter(batchIndices: batchRows())
            #expect(recurrent.batchSize == 0)
            recurrent.extend(other: MambaCache())
            #expect(recurrent.batchSize == 1)

            let untouched = MambaCache()
            untouched.filter(batchIndices: batchRows())
            #expect(untouched.batchSize == 0)

            let list = try CacheList.merge([
                CacheList(KVCacheSimple(), MambaCache()),
                CacheList(KVCacheSimple(), MambaCache()),
                CacheList(KVCacheSimple(), MambaCache(leftPadding: [0])),
            ])
            let attention = try #require(list[0] as? BatchKVCache)
            let state = try #require(list[1] as? MambaCache)
            try #require(attention.batchSize == 3)
            try #require(state.batchSize == 3)
            list.filter(batchIndices: batchRows(2, 0))
            list.finalize()
            #expect(state.batchSize == 2)
            #expect(state.leftPadding == nil)

            let populated = MambaCache()
            populated[0] = MLXArray([Float(10), 11, 12, 13]).reshaped([1, 4])
            let kv = KVCacheSimple()
            let values = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
            _ = kv.update(keys: values.0, values: values.1)
            let added = try CacheList.merge([CacheList(kv, populated)])
            list.extend(other: added)
            #expect(attention.batchSize == 3)
            #expect(state.batchSize == 3)
            let arrays = try #require(state[0])
            #expect(arrays.shape == [3, 4])
            #expect(arrays[0 ..< 2].sum().item(Float.self) == 0)
            #expect(maxAbsDifference(arrays[2 ..< 3], try #require(populated[0])) == 0)
        }
    }

    @Test("merge gives an empty input's rows zero state, however many it stands for")
    func mergeKeepsEmptyInputRows() throws {
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])
        let merged = try #require(
            ArraysCache.merge([MambaCache(leftPadding: [0, 0]), populated]) as? MambaCache)
        #expect(merged.batchSize == 3)
        #expect(try #require(merged[0]).shape == [3, 4])
        #expect(merged.leftPadding?.shape == [3])
        #expect(maxAbsDifference(try #require(merged[0])[2 ..< 3], try #require(populated[0])) == 0)

        let none = try #require(
            ArraysCache.merge([MambaCache(leftPadding: []), populated]) as? MambaCache)
        #expect(none.batchSize == 1)
        #expect(try #require(none[0]).shape == [1, 4])
    }

    @Test("Extending an empty composite batch with prefilled rows keeps its children aligned")
    func extendEmptyCompositeRows() throws {
        let list = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let added = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let kv = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: kv.0, values: kv.1)
        try #require(added[1] as? MambaCache)[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([
            1, 4,
        ])
        list.extend(other: added)

        let attention = try #require(list[0] as? BatchKVCache)
        let recurrent = try #require(list[1] as? MambaCache)
        #expect(attention.batchSize == 2)
        #expect(recurrent.batchSize == 2)
        #expect(attention.leftPadding.asArray(Int32.self) == [3, 0])
        #expect(try #require(recurrent[0]).shape == [2, 4])
        #expect(try #require(recurrent[0])[0 ..< 1].sum().item(Float.self) == 0)
        list.filter(batchIndices: batchRows(1))
        #expect(attention.batchSize == 1)
        #expect(recurrent.batchSize == 1)
        #expect(
            maxAbsDifference(
                try #require(recurrent[0]), MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4]))
                == 0)
    }

    @Test("A nested CacheList masks with its batched attention descendant")
    func nestedCompositeMask() throws {
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 1, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let list = try CacheList.merge([
            CacheList(MambaCache(), CacheList(KVCacheSimple())),
            CacheList(MambaCache(), CacheList(longer)),
        ])
        guard case .array(let mask) = list.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("A nested batched CacheList must use its attention descendant's mask")
            return
        }
        #expect(mask.dim(0) == 2)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
    }

    @Test("A merged CacheList masks with its attention child's left padding")
    func compositeMask() throws {
        // Row 1 holds one token, row 0 none, so the merge pads row 0 by one.
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 1, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let list = try CacheList.merge([
            CacheList(MambaCache(), KVCacheSimple()), CacheList(MambaCache(), longer),
        ])
        #expect((list[1] as? BatchKVCache)?.leftPadding.asArray(Int32.self) == [1, 0])
        guard case .array(let mask) = list.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("A batched CacheList must use its attention child's mask")
            return
        }
        #expect(mask.dim(0) == 2)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
    }

    @Test("The row methods and merge dispatch through the KVCache type")
    func protocolSurface() throws {
        #expect(try mergeCaches([MambaCache()]) is MambaCache)
        #expect(try mergeCaches([CacheList(KVCacheSimple(), MambaCache())]) is CacheList)
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([QuantizedKVCache()]) }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([ChunkedKVCache(chunkSize: 4)]) }
        #expect(throws: KVCacheError.self) { _ = try QuantizedKVCache.merge([QuantizedKVCache()]) }
    }
}

/// Recurrent and composite batched caches must give every row the logits it gets on its own.
///
/// The rows are prefilled one at a time and then merged: a right-padded prefill of these
/// models is not exact yet, because their state-space scans do not honour the prepared
/// lengths. The scans run as Metal kernels, so the tests stay on the default device, where a
/// one-row and a two-row forward differ by the matmul precision (see ``MatmulPrecision``).
@Suite(.serialized)
struct BatchedRecurrentParityTests {

    @Test("Merged Mamba 2 prompts decode like separate sequences")
    func mamba2() throws {
        let model = try tinyMamba2()
        try expectMergedRowsMatchSingles(model, layers: { $0 is MambaCache })
    }

    @Test("Merged Falcon H1 prompts decode like separate sequences")
    func falconH1() throws {
        let model = try tinyFalconH1()
        try expectMergedRowsMatchSingles(model, layers: { $0 is CacheList })
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

/// Prefills two prompts of different lengths alone, merges the prefilled caches, decodes four
/// tokens and drops a row, comparing every row's logits with the prompt continued alone and
/// every extracted row's state with its single cache.
private func expectMergedRowsMatchSingles(
    _ model: any LanguageModel, layers: (any KVCache) -> Bool
) throws {
    let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8]]

    var singles: [[any KVCache]] = []
    var rows: [[any KVCache]] = []
    for prompt in prompts {
        let single = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: single))
        singles.append(single)
        let row = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: row))
        rows.append(row)
    }
    let hasExpectedLayers = singles[0].contains(where: layers)
    #expect(hasExpectedLayers)

    let batched = try rows[0].indices.map { layer in
        try mergeCaches(rows.map { $0[layer] })
    }
    for cache in batched {
        if let attention = cache as? BatchKVCache {
            #expect(attention.batchOffset.asArray(Int.self) == prompts.map(\.count))
        }
        if let list = cache as? CacheList, let attention = list[1] as? BatchKVCache {
            #expect(attention.batchOffset.asArray(Int.self) == prompts.map(\.count))
        }
    }
    try expectExtractedRowsMatch(batched, singles)

    for step in 0 ..< 4 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < MatmulPrecision.splitTolerance,
                "step \(step), row \(row)")
        }
    }
    try expectExtractedRowsContinueLikeSingles(model, batched, singles)

    batched.forEach { filterRows($0, keeping: batchRows(1)) }
    let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: batched)
    let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < MatmulPrecision.splitTolerance, "after filtering")
    try expectExtractedRowsContinueLikeSingles(model, batched, [singles[1]])
}

/// Every row extracted from `batch` decodes the next token like its single cache. The
/// recurrent state arrays themselves are not compared after a decode: a one-row and a two-row
/// scan leave them in different numerical form even for copies of one row.
private func expectExtractedRowsContinueLikeSingles(
    _ model: any LanguageModel, _ batch: [any KVCache], _ singles: [[any KVCache]]
) throws {
    for (row, caches) in singles.enumerated() {
        let extracted = try batch.map { try extractRow($0, row) }
        for (layer, cache) in extracted.enumerated() where !(cache is ArraysCache) {
            #expect(cache.offset == caches[layer].offset, "row \(row), layer \(layer)")
        }
        let token = MLXArray([Int32(20 + row)]).reshaped([1, 1])
        let actual = model(token, cache: extracted)
        let expected = model(token, cache: caches.map { $0.copy() })
        #expect(
            maxAbsDifference(actual, expected) < MatmulPrecision.splitTolerance,
            "extracted row \(row)")
    }
}

private func filterRows(_ cache: any KVCache, keeping indices: MLXArray) {
    switch cache {
    case let batch as CacheList: batch.filter(batchIndices: indices)
    case let batch as ArraysCache: batch.filter(batchIndices: indices)
    case let batch as BatchKVCache: batch.filter(batchIndices: indices)
    default: Issue.record("not a batched cache: \(type(of: cache))")
    }
}

private func extractRow(_ cache: any KVCache, _ index: Int) throws -> any KVCache {
    switch cache {
    case let batch as CacheList: return batch.extract(index)
    case let batch as ArraysCache: return batch.extract(index)
    case let batch as BatchKVCache: return batch.extract(index)
    default: throw KVCacheError(message: "not a batched cache: \(type(of: cache))")
    }
}

/// Every row extracted from `batch` holds the state of its single cache, and the offset of
/// an attention cache. Recurrent state keeps one offset for the whole batch.
private func expectExtractedRowsMatch(_ batch: [any KVCache], _ singles: [[any KVCache]])
    throws
{
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = try extractRow(batch[layer], row)
            if !(actual is ArraysCache) {
                #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            }
            let a = actual.state
            let b = caches[layer].state
            try #require(a.count == b.count, "row \(row), layer \(layer)")
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape, "row \(row), layer \(layer)")
                // Recurrent state grows with the sequence, so the bound is relative.
                let mismatch = MatmulPrecision.mismatch(
                    actualArray, expectedArray, rtol: 1e-2, atol: 1e-4, reducedAtol: 2e-2)
                #expect(mismatch == nil, "row \(row), layer \(layer): \(mismatch ?? "")")
            }
        }
    }
}
