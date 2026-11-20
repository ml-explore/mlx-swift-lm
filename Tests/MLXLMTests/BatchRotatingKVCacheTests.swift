// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct BatchRotatingKVCacheTests {

    @Test("Rotating merge dispatches through the BatchableKVCache type")
    func protocolSurface() throws {
        #expect(try mergeCaches([RotatingKVCache(maxSize: 8)]) is BatchRotatingKVCache)
    }

    @Test("A rotating row takes no row operations; its batch does")
    func batchableConformance() throws {
        #expect(!(RotatingKVCache(maxSize: 8) as any KVCache is any BatchableKVCache))
        let batch = try BatchRotatingKVCache.merge([RotatingKVCache(maxSize: 8)])
        #expect(batch is BatchRotatingKVCache)
        #expect(throws: KVCacheError.self) { _ = try BatchRotatingKVCache.merge([any KVCache]()) }
        #expect(throws: KVCacheError.self) {
            _ = try BatchRotatingKVCache.merge([KVCacheSimple()])
        }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([batch]) }
    }

    @Test(
        "Text-only Mistral3 VLM batches preserve per-row RoPE and attention-scale bands",
        arguments: [Float(0), 1], [2, 3])
    func mistralVLMRaggedPositions(beta: Float, batchSize: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyMistralVLMForBatchScaling(beta: beta)
            let prompts = Array(
                [
                    [1, 2, 3], Array(4 ..< 11), Array(11 ..< 20),
                ].prefix(batchSize))
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
            #expect(batch[0] is BatchRotatingKVCache)
            #expect(batch[1] is BatchKVCache)
            let offsets = try #require(batch[0] as? BatchRotatingKVCache).batchOffset
            #expect(offsets.asArray(Int32.self) == prompts.map { Int32($0.count) })

            var generated = 0
            for length in [1, 2, 1, 2] {
                let tokens = MLXArray(
                    singles.indices.flatMap { row in
                        (0 ..< length).map { Int32(21 + row * 7 + generated + $0) }
                    }
                ).reshaped([singles.count, length])
                try expectMistralVLMChunk(model, tokens: tokens, batch: batch, singles: singles)
                generated += length
            }

            let keep = batchSize == 2 ? [1] : [2, 0]
            let indices = MLXArray(keep.map { Int32($0) })
            for layer in batch { layer.filter(batchIndices: indices) }
            singles = keep.map { singles[$0] }
            let tokens = MLXArray(singles.indices.map { Int32(41 + $0) })[0..., .newAxis]
            try expectMistralVLMChunk(model, tokens: tokens, batch: batch, singles: singles)
        }
    }

    @Test("Extension checks reject window settings, unfinished prefill and empty/prefilled mixes")
    func extensionValidation() throws {
        try Device.withDefaultDevice(.cpu) {
            let full = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
            let kv = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4)
            _ = full.update(keys: kv.0, values: kv.1)
            let compatible = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
            _ = compatible.update(keys: kv.0, values: kv.1)
            try full.validateExtension(other: compatible)
            #expect(full.batchSize == 1)
            for mismatch in 0 ..< 3 {
                let added = BatchRotatingKVCache(maxSize: mismatch == 0 ? 8 : 16, leftPadding: [0])
                if mismatch == 1 { added.step = 128 }
                if mismatch == 2 { added.capacityOrigin = .requested }
                #expect(throws: KVCacheError.self) { try full.validateExtension(other: added) }
            }
            let empty = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
            #expect(throws: KVCacheError.self) { try full.validateExtension(other: empty) }
            #expect(throws: KVCacheError.self) { try empty.validateExtension(other: full) }
            let prepared = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
            prepared.prepare(lengths: [1, 2])
            #expect(throws: KVCacheError.self) { try full.validateExtension(other: prepared) }
            let zeroRows = BatchRotatingKVCache(maxSize: 16, leftPadding: [])
            try full.validateExtension(other: zeroRows)
            try zeroRows.validateExtension(other: full)
        }
    }

    @Test("Update, filter, extend and extract keep each row's tokens")
    func lifecycle() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: batchRows(0))
        #expect(cache.batchSize == 1)

        let added = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)

        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(try row(cache).offset == 3)
        let second = try row(cache, 1)
        #expect(second.offset == 2)
        #expect(maxAbsDifference(try #require(second.state.first), addedKV.0) == 0)
        #expect(maxAbsDifference(try #require(second.state.last), addedKV.1) == 0)
    }

    @Test("isTrimmable(after:) predicts when the window fills")
    func trimmability() {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache.isTrimmable)
        #expect(cache.isTrimmable(after: 3))
        #expect(!cache.isTrimmable(after: 4))
    }

    @Test(
        "Staged rounds refuse writes that cannot rewind every row's padding",
        arguments: [0, 2], [1, 2, 5])
    func paddedRoundIsRefused(written: Int, width: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [3, 0])
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
            let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: filtered ? [0] : [])
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
            let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [3, 0])
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

    @Test("trim drops the newest tokens, no more than the shortest row holds")
    func trimClampsToShortestRow() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let kv = makePositionKV(
            positions: 0 ..< 3, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache.trim(2) == 1)
        #expect(cache.batchOffset.asArray(Int32.self) == [0, 2])
        #expect(try row(cache).offset == 0)
        let second = try row(cache, 1)
        #expect(second.offset == 2)
        #expect(try #require(second.state.first)[0, 0, 0..., 0].asArray(Float.self) == [10, 11])
        #expect(try #require(second.state.last)[0, 0, 0..., 0].asArray(Float.self) == [110, 111])
    }

    @Test("trim removes nothing once the window has filled")
    func trimAfterWrap() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0])
        append(0 ..< 4, to: cache)
        append(4 ..< 5, to: cache)
        #expect(cache.maxTrimCount == 0)
        #expect(cache.trim(2) == 0)
        #expect(cache.batchOffset.asArray(Int.self) == [5])
        append(5 ..< 6, to: cache)
        #expect(try view(try row(cache)).0.asArray(Float.self) == [2, 3, 4, 5])
    }

    @Test("A ragged prefill keeps a completed row's full window")
    func raggedPrefillKeepsCompletedRows() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0, 0])
        cache.prepare(lengths: [8, 4])
        let first = makePositionKV(
            positions: 0 ..< 4, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: first.0, values: first.1)
        // Row 1 is complete; its entries in this chunk are padding.
        let second = MLXArray([Float(4), 5, 6, 7, 99, 99, 99, 99]).reshaped([2, 1, 4, 1])
        guard case .array(let mask) = cache.makeMask(n: 4, windowSize: nil, returnArray: true)
        else {
            Issue.record("Expected an array mask")
            return
        }
        #expect(mask.shape == [2, 1, 4, 8])
        _ = cache.update(keys: second, values: second + 100)
        cache.finalize()

        #expect(try view(try row(cache, 0)).0.asArray(Float.self) == [4, 5, 6, 7])
        #expect(try view(try row(cache, 1)).0.asArray(Float.self) == [10, 11, 12, 13])
        #expect(cache.batchOffset.asArray(Int.self) == [8, 4])

        let decode = MLXArray([Float(8), 14]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(try view(try row(cache, 0)).0.asArray(Float.self) == [5, 6, 7, 8])
        #expect(try view(try row(cache, 1)).0.asArray(Float.self) == [11, 12, 13, 14])
    }

    @Test("A merged batch reports the tokens its rows have evicted")
    func evictedTokens() throws {
        let single = RotatingKVCache(maxSize: 8)
        for position in 0 ..< 11 { append(position ..< position + 1, to: single) }
        #expect(single.evictedTokenCount == 3)
        let batch = try RotatingKVCache.merge([single])
        #expect(batch.evictedTokenCount == 3)
        // The offset counts every position like the row's, as the global layers' caches do.
        #expect(batch.offset == single.offset)
        append(11 ..< 12, to: batch)
        #expect(batch.evictedTokenCount == 4)
        #expect(batch.offset == 12)
        #expect(BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 2]).evictedTokenCount == 0)
    }

    @Test("Evictions count each row's padding against the buffer it shares")
    func evictedTokensWithPadding() throws {
        // A wrapped row that a two-token write widened to nine positions, ten evicted...
        let wrapped = RotatingKVCache(maxSize: 8)
        for position in 0 ..< 17 { append(position ..< position + 1, to: wrapped) }
        let narrow = try RotatingKVCache.merge([wrapped])
        append(17 ..< 19, to: narrow)
        #expect(narrow.evictedTokenCount == 10)
        // ...joins a ten-wide row and takes one position of padding.
        let batch = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        append(0 ..< 10, to: batch)
        batch.extend(other: narrow)
        #expect(batch.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(batch.evictedTokenCount == 10)
        #expect(batch.offset == 19)
        #expect(
            try view(try row(batch, 1)).0.asArray(Float.self) == Array(11 ... 18).map(Float.init))
    }

    @Test("A ragged prefill after a wrap pads a completed row by the full roll")
    func raggedPrefillAfterWrap() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0, 0])
        let first = makePositionKV(
            positions: 0 ..< 4, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: first.0, values: first.1)
        let decode = MLXArray([Float(4), 14]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 0])

        // Row 1 is complete; its two positions in the chunk are padding.
        cache.prepare(lengths: [2, 0])
        let chunk = MLXArray([Float(5), 6, 99, 99]).reshaped([2, 1, 2, 1])
        _ = cache.update(keys: chunk, values: chunk + 100)
        cache.finalize()
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 2])
        #expect(cache.batchOffset.asArray(Int.self) == [7, 5])

        // The extracted row holds its history only, no padding ahead of it.
        let extracted = try row(cache, 1)
        let keys = try #require(extracted.state.first)
        #expect(keys[0, 0, 0..., 0].asArray(Float.self) == [11, 12, 13, 14])
        #expect(extracted.offset == 5)

        let next = MLXArray([Float(7), 15]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: next, values: next + 100)
        #expect(try view(try row(cache, 0)).0.asArray(Float.self) == [4, 5, 6, 7])
        #expect(try view(try row(cache, 1)).0.asArray(Float.self) == [12, 13, 14, 15])
    }

    @Test("maxTrimCount reports only what trim can remove")
    func maxTrimCountMatchesTrim() {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.maxTrimCount == 1)
        #expect(!canTrimPromptCache([cache], numTokens: 2))
        #expect(canTrimPromptCache([cache], numTokens: 1))
        #expect(cache.trim(1) == 1)
    }

    @Test("Empty rows join empty rows, and the batch fills on its first update")
    func extendEmptyRows() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        cache.extend(other: BatchRotatingKVCache(maxSize: 8, leftPadding: [1]))
        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(cache.batchOffset.asArray(Int32.self) == [0, -1])

        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache.batchOffset.asArray(Int32.self) == [3, 2])
        #expect(try row(cache, 1).offset == 2)

        // A batch with no rows changes nothing, prefilled or not.
        cache.extend(other: BatchRotatingKVCache(maxSize: 8, leftPadding: []))
        #expect(cache.batchSize == 2)
        #expect(cache.batchOffset.asArray(Int32.self) == [3, 2])
    }

    @Test("An empty cache extended from another does not share its arrays")
    func extendEmptyCacheCopies() throws {
        // One token at a time, so the source's buffer has room past its tokens.
        let source = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        append(0 ..< 1, to: source)
        append(1 ..< 2, to: source)
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [])
        cache.extend(other: source)

        // Both writes land inside the buffer the source allocated.
        append(50 ..< 51, to: cache)
        append(70 ..< 71, to: source)

        #expect(try view(try row(cache)).0.asArray(Float.self) == [0, 1, 50])
        #expect(try view(try row(source)).0.asArray(Float.self) == [0, 1, 70])
    }

    @Test("ropeOffset is per row through the KVCache type")
    func ropeOffsetThroughKVCache() {
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        let batch = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 0])
        _ = batch.update(keys: kv.0, values: kv.1)
        let cache: any KVCache = batch
        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record("BatchRotatingKVCache.ropeOffset must be .batch")
            return
        }
        #expect(offsets.asArray(Int32.self) == [2, 3])
    }

    @Test("A ragged prefill keeps each row's tokens after finalize")
    func raggedPrefill() throws {
        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        let kv = makePositionKV(
            positions: 0 ..< 5, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)
        cache.finalize()

        let first = try row(cache)
        #expect(first.offset == 3)
        #expect(try #require(first.state.first)[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
        #expect(
            try #require(first.state.last)[0, 0, 0..., 0].asArray(Float.self) == [100, 101, 102])
        let second = try row(cache, 1)
        #expect(second.offset == 5)
        #expect(
            try #require(second.state.first)[0, 0, 0..., 0].asArray(Float.self)
                == [10, 11, 12, 13, 14])
    }

    @Test("A one-token ragged prefill finalizes before decode")
    func oneTokenRaggedPrefill() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(lengths: [1, 0])
        let prefill = MLXArray([Float(10), 99]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: prefill, values: prefill + 100)
        cache.finalize()

        #expect(try row(cache).offset == 1)
        #expect(try #require(try row(cache).state.first).asArray(Float.self) == [10])
        #expect(try row(cache, 1).offset == 0)
        #expect(try #require(try row(cache, 1).state.first).size == 0)

        let decode = MLXArray([Float(20), 30]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(try #require(try row(cache).state.first).asArray(Float.self) == [10, 20])
        #expect(try #require(try row(cache).state.last).asArray(Float.self) == [110, 120])
        #expect(try row(cache, 1).offset == 1)
        #expect(try #require(try row(cache, 1).state.first).asArray(Float.self) == [30])
    }

    @Test("A ragged prefill accepts a trailing one-token chunk")
    func trailingOneTokenChunk() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(lengths: [2, 3])
        let first = makePositionKV(
            positions: 0 ..< 2, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: first.0, values: first.1)
        // Row 0 is already complete, so its final chunk entry is padding.
        let last = MLXArray([Float(99), 12]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: last, values: last + 100)
        cache.finalize()

        #expect(try row(cache).offset == 2)
        #expect(try #require(try row(cache).state.first).asArray(Float.self) == [0, 1])
        #expect(try row(cache, 1).offset == 3)
        #expect(try #require(try row(cache, 1).state.first).asArray(Float.self) == [10, 11, 12])

        let decode = MLXArray([Float(2), 13]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(try #require(try row(cache).state.first).asArray(Float.self) == [0, 1, 2])
        #expect(
            try #require(try row(cache, 1).state.first).asArray(Float.self) == [10, 11, 12, 13])
    }

    @Test("merge keeps the retained window, below and at capacity")
    func mergeKeepsWindow() throws {
        for length in [5, 8] {
            let single = RotatingKVCache(maxSize: 8)
            append(0 ..< length, to: single)
            let batch = try RotatingKVCache.merge([single])
            #expect(batch.batchSize == 1)
            #expect(batch.batchOffset.asArray(Int.self) == [single.offset])
            try expectHistory(try row(batch), single)
        }
    }

    @Test("merge refuses a pinned prefix, a rewind reserve, and mixed empty and prefilled rows")
    func mergeRefusesUnbatchableCaches() {
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([RotatingKVCache(maxSize: 8, keep: 2)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([RotatingKVCache(maxSize: 8, rewindCapacity: 4)])
        }
        let prefilled = RotatingKVCache(maxSize: 8)
        append(0 ..< 3, to: prefilled)
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([prefilled, RotatingKVCache(maxSize: 8)])
        }
        // Rows must share one window, empty or not, and one allocation step.
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([
                RotatingKVCache(maxSize: 8), RotatingKVCache(maxSize: 16),
            ])
        }
        let wider = RotatingKVCache(maxSize: 16)
        append(0 ..< 3, to: wider)
        #expect(throws: KVCacheError.self) { _ = try RotatingKVCache.merge([prefilled, wider]) }
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([
                RotatingKVCache(maxSize: 8, step: 2), RotatingKVCache(maxSize: 8, step: 4),
            ])
        }
    }

    @Test(
        "A ragged prefill keeps at most a window and a chunk per row",
        arguments: [[12, 4], [10, 6]])
    func raggedPrefillBoundsStorage(lengths: [Int]) throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0, 0])
        let longest = try #require(lengths.max())
        let all = makePositionKV(
            positions: 0 ..< longest, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        cache.prepare(lengths: lengths)
        for start in stride(from: 0, to: longest, by: 4) {
            let end = min(start + 4, longest)
            _ = cache.update(
                keys: all.0[.ellipsis, start ..< end, 0...],
                values: all.1[.ellipsis, start ..< end, 0...])
            #expect(try #require(cache.state.first).dim(2) <= 8, "a window and a chunk at most")
        }
        cache.finalize()
        #expect(cache.batchOffset.asArray(Int.self) == lengths)

        let decode = MLXArray([Float(longest), Float(10 + lengths[1])]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(
            try view(try row(cache, 0)).0.asArray(Float.self)
                == Array(longest - 3 ... longest).map(Float.init))
        #expect(
            try view(try row(cache, 1)).0.asArray(Float.self)
                == Array(10 + lengths[1] - 3 ... 10 + lengths[1]).map(Float.init))
    }

    @Test("A merged row that has evicted history cannot be rewound")
    func mergedEvictedRowIsNotTrimmable() throws {
        let single = RotatingKVCache(maxSize: 8)
        for position in 0 ..< 17 { append(position ..< position + 1, to: single) }
        #expect(single.trim(2) == 2)
        #expect(single.maxTrimCount == 0)

        // The batch holds six positions of a window of eight, but nine are evicted.
        let batch = try RotatingKVCache.merge([single])
        #expect(batch.evictedTokenCount == 9)
        #expect(!batch.isTrimmable)
        #expect(batch.maxTrimCount == 0)
        #expect(!canTrimPromptCache([batch], numTokens: 1))
        #expect(trimPromptCache([batch], numTokens: 1) == 0)
        #expect(batch.batchOffset.asArray(Int.self) == [15])

        // A merged row that has evicted nothing still rewinds.
        let short = RotatingKVCache(maxSize: 8)
        append(0 ..< 5, to: short)
        let fresh = try RotatingKVCache.merge([short])
        #expect(fresh.maxTrimCount == 5)
        #expect(canTrimPromptCache([fresh], numTokens: 2))
    }

    @Test("A trimmed wrapped source keeps its rows through batching and decode")
    func trimmedWrappedSource() throws {
        let single = RotatingKVCache(maxSize: 8)
        for position in 0 ..< 17 { append(position ..< position + 1, to: single) }
        #expect(single.trim(2) == 2)
        #expect(single.offset == 15)

        let batch = try RotatingKVCache.merge([single])
        #expect(batch.batchOffset.asArray(Int.self) == [15])
        try expectHistory(try row(batch), single)
        for position in 15 ..< 20 {
            append(position ..< position + 1, to: single)
            append(position ..< position + 1, to: batch)
            #expect(batch.batchOffset.asArray(Int.self) == [position + 1])
            try expectHistory(try row(batch), single)
        }
    }

    @Test("Wrapped and over-long histories survive conversion and further writes")
    func conversion() throws {
        try Device.withDefaultDevice(.cpu) {
            for overlong in [false, true] {
                let source = RotatingKVCache(maxSize: 8, step: 3)
                if overlong {
                    append(0 ..< 11, to: source)
                } else {
                    append(0 ..< 8, to: source)
                    for position in 8 ..< 11 { append(position ..< position + 1, to: source) }
                }
                let original = source.state.map { $0.asArray(Float.self) }
                let reference = try #require(source.copy() as? RotatingKVCache)
                let batch = try RotatingKVCache.merge([source])
                #expect(batch.batchSize == 1)
                #expect(batch.batchOffset.asArray(Int.self) == [11])
                #expect(batch.step == 3)
                #expect(
                    try view(try row(batch)).0.asArray(Float.self)
                        == Array(3 ... 10).map(Float.init))

                var position = 11
                for count in [1, 2, 1, 1, 1, 1, 1, 1, 1, 1] {
                    append(position ..< position + count, to: batch)
                    append(position ..< position + count, to: reference)
                    position += count
                    let extracted = try row(batch)
                    try expectHistory(extracted, reference)
                    #expect(extracted.allocationStep == 3)

                    // A multi-token write continues from the restored write position.
                    let continued = try #require(reference.copy() as? RotatingKVCache)
                    append(position ..< position + 2, to: extracted)
                    append(position ..< position + 2, to: continued)
                    try expectHistory(extracted, continued)
                }
                #expect(source.state.map { $0.asArray(Float.self) } == original)
            }
        }
    }

    @Test("Converting a full window leaves the source independently usable")
    func sourceIsolation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 8)
            append(0 ..< 8, to: source)
            let original = source.state.map { $0.asArray(Float.self) }
            let metadata = source.metaState
            let batch = try RotatingKVCache.merge([source])
            #expect(batch.batchOffset.asArray(Int.self) == [8])
            append(8 ..< 9, to: batch)
            #expect(source.metaState == metadata)
            #expect(source.state.map { $0.asArray(Float.self) } == original)
            append(20 ..< 21, to: source)
            #expect(try view(source).0.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 20])
            #expect(try view(try row(batch)).0.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 8])
        }
    }

    @Test("Extraction after a concatenating write continues with another one")
    func extractedConcatContinuation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 4)
            append(0 ..< 4, to: source)
            let batch = try RotatingKVCache.merge([source])
            #expect(batch.batchOffset.asArray(Int.self) == [4])
            append(4 ..< 6, to: batch)
            append(4 ..< 6, to: source)
            let extracted = try row(batch)
            try expectHistory(extracted, source)
            append(6 ..< 8, to: extracted)
            append(6 ..< 8, to: source)
            try expectHistory(extracted, source)
        }
    }

    @Test("Converted rows extend with shorter histories, reorder and continue")
    func raggedExtension() throws {
        try Device.withDefaultDevice(.cpu) {
            let sources = [
                RotatingKVCache(maxSize: 8), RotatingKVCache(maxSize: 8),
                RotatingKVCache(maxSize: 8),
            ]
            append(0 ..< 8, to: sources[0])
            for position in 8 ..< 11 { append(position ..< position + 1, to: sources[0]) }
            append(100 ..< 113, to: sources[1])
            append(200 ..< 203, to: sources[2])
            let batch = try RotatingKVCache.merge(sources)
            #expect(batch.batchSize == 3)
            #expect(batch.batchOffset.asArray(Int.self) == sources.map(\.offset))
            #expect(batch.leftPadding.asArray(Int32.self).count == 3)
            for index in sources.indices {
                try expectHistory(try row(batch, index), sources[index])
            }

            batch.filter(batchIndices: batchRows(2, 0))
            let survivors = [sources[2], sources[0]]
            for step in 0 ..< 10 {
                let next = MLXArray([Float(203 + step), Float(11 + step)]).reshaped([2, 1, 1, 1])
                _ = batch.update(keys: next, values: next + 100)
                append(203 + step ..< 204 + step, to: survivors[0])
                append(11 + step ..< 12 + step, to: survivors[1])
                for index in survivors.indices {
                    try expectHistory(try row(batch, index), survivors[index])
                }
            }
        }
    }

    @Test("One-token prepared masks match the returned keys at capacity", arguments: [false, true])
    func preparedMask(wrapped: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
            let initial = MLXArray([Float(90), 91, 0, 1, 10, 11, 12, 13]).reshaped([2, 1, 4, 1])
            _ = cache.update(keys: initial, values: initial + 100)
            if wrapped {
                let token = MLXArray([Float(2), 14]).reshaped([2, 1, 1, 1])
                _ = cache.update(keys: token, values: token + 100)
            }
            cache.prepare(lengths: [1, 0])
            guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 3, returnArray: true)
            else {
                Issue.record("Expected an array mask")
                return
            }
            let next = MLXArray([Float(wrapped ? 3 : 2), 99]).reshaped([2, 1, 1, 1])
            let (keys, values) = cache.update(keys: next, values: next + 100)
            let latest = Float(wrapped ? 3 : 2)
            let expectedMask = keys[0, 0, 0..., 0].asArray(Float.self).map {
                $0 >= latest - 2 && $0 <= latest
            }
            #expect(mask[0, 0, 0].asArray(Bool.self) == expectedMask)
            let scores = MLX.where(mask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let attended = matmul(softmax(scores, axis: -1), values)[0].item(Float.self)
            #expect(abs(attended - (latest - 1 + 100)) < 0.0001)

            cache.finalize()
            #expect(cache.batchOffset.asArray(Int.self) == [wrapped ? 4 : 3, wrapped ? 5 : 4])
            let decode = MLXArray([latest + 1, Float(wrapped ? 15 : 14)]).reshaped([2, 1, 1, 1])
            _ = cache.update(keys: decode, values: decode + 100)
            let first = try view(try row(cache)).0.asArray(Float.self)
            #expect(first == (wrapped ? [1, 2, 3, 4] : [0, 1, 2, 3]))
            let second = try view(try row(cache, 1)).0.asArray(Float.self)
            #expect(second == (wrapped ? [12, 13, 14, 15] : [11, 12, 13, 14]))
        }
    }

    @Test("A multi-token mask after a wrap predicts the trim of the matching update")
    func multiTokenMaskAfterWrap() {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
        let (k4, v4) = makeKV(batchSize: 2, heads: 2, seqLen: 4, headDim: 4)
        _ = cache.update(keys: k4, values: v4)
        let (k1, v1) = makeKV(batchSize: 2, heads: 2, seqLen: 1, headDim: 4)
        _ = cache.update(keys: k1, values: v1)

        guard case .array(let mask) = cache.makeMask(n: 2, windowSize: nil, returnArray: true)
        else {
            Issue.record("Expected an array mask")
            return
        }
        #expect(mask.dim(-1) == 5)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
    }

    @Test("Extraction keeps the window, step and capacity origin, also for an empty row")
    func extractionMetadata() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 2])
        cache.capacityOrigin = .requested
        cache.step = 3
        for index in 0 ..< 2 {
            let extracted = try row(cache, index)
            #expect(extracted.maxSize == 16)
            #expect(extracted.allocationStep == 3)
            #expect(extracted.capacityOrigin == .requested)
            #expect(extracted.offset == 0)
        }
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(try row(cache).capacityOrigin == .requested)
    }

    @Test("merge carries the window, step and capacity origin")
    func mergeCarriesConfiguration() throws {
        let probe = RotatingKVCache(maxSize: 16, step: 3)
        probe.capacityOrigin = .requested
        let cache = try RotatingKVCache.merge([probe])
        #expect(cache.batchSize == 1)
        #expect(cache.maxSize == 16)
        #expect(cache.step == 3)
        #expect(cache.capacityOrigin == .requested)
    }

    @Test("The cache status reports a batch as one sliding window of full-precision rows")
    func status() throws {
        let rows = (0 ..< 2).map { _ in RotatingKVCache(maxSize: 16) }
        for row in rows { append(0 ..< 5, to: row) }
        let batch = try RotatingKVCache.merge(rows)

        let status = KVCacheStatus(cache: [batch])
        try #require(status.layers.count == 1)
        #expect(status.layers[0].kind == .rotatingAttention(maxSize: 16, keep: 0))
        #expect(status.layers[0].capacitySource == .modelDefined)
        #expect(status.layers[0].state == .active)
        let rowBytes = (0 ..< 2).map { batch.extract($0) }.reduce(0) { bytes, row in
            bytes + row.innerState().reduce(0) { $0 + $1.nbytes }
        }
        #expect(status.memoryBytes == rowBytes)

        let affine = kvCacheLayerStatuses(
            cache: [batch], configuration: KVCacheConfiguration(strategy: .affine(.fourBit)))
        #expect(affine[0].state == .skipped)
        #expect(affine[0].reason == .slidingWindow)
    }

    @Test("Extending rejects a different allocation step")
    func mismatchedSteps() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let receiver = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
                let other = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
                other.step = 3
                let kv = MLXArray.ones([1, 1, 1, 1])
                _ = receiver.update(keys: kv, values: kv)
                _ = other.update(keys: kv, values: kv)
                receiver.extend(other: other)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "requires matching maxSize, step and capacityOrigin"))
    }

    @Test("extract refuses a row index past the batch")
    func extractOutOfRange() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 1])
                let kv = MLXArray.ones([2, 1, 3, 1])
                _ = cache.update(keys: kv, values: kv)
                _ = cache.extract(2)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "BatchRotatingKVCache row index out of range"))
    }

    @Test("extract returns an empty row when its padding covers every written position")
    func extractPaddingOnlyRow() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [4, 0])
        cache.capacityOrigin = .requested
        cache.step = 3
        let keys = MLXArray([Float(1), 2]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: keys, values: keys + 100)

        let padded = try row(cache, 0)
        #expect(padded.offset == 0)
        #expect(try #require(padded.state.first).size == 0)
        #expect(padded.maxSize == 8)
        #expect(padded.allocationStep == 3)
        #expect(padded.capacityOrigin == .requested)
        let written = try row(cache, 1)
        #expect(written.offset == 1)
        #expect(try view(written).0.asArray(Float.self) == [2])
    }

    @Test("extract refuses a ragged prefill before finalize")
    func extractBeforeFinalize() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
                cache.prepare(lengths: [1, 3])
                let kv = MLXArray.ones([2, 1, 3, 1])
                _ = cache.update(keys: kv, values: kv)
                _ = cache.extract(0)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "BatchRotatingKVCache.extract takes finalized rows"))
    }

    @Test("merge throws for rows of another layout or with more than one row")
    func mergeRefusesMismatchedRows() throws {
        func filled(
            heads: Int = 1, headDim: Int = 1, rows: Int = 1, dtype: DType = .float32
        ) -> RotatingKVCache {
            let cache = RotatingKVCache(maxSize: 8)
            let kv = MLXArray.ones([rows, heads, 3, headDim]).asType(dtype)
            _ = cache.update(keys: kv, values: kv)
            return cache
        }
        for other in [filled(headDim: 2), filled(heads: 2), filled(dtype: .float16)] {
            #expect(throws: KVCacheError.self) { _ = try RotatingKVCache.merge([filled(), other]) }
        }
        // A cache holding two rows is not a row, also alone: admission merges one at a time.
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([filled(rows: 2)]) }
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache.merge([filled(), filled(rows: 2)])
        }
        #expect(try RotatingKVCache.merge([filled(), filled()]).batchSize == 2)
    }

    @Test("A batched rotating cache cannot be saved as a prompt cache")
    func savePromptCacheRefuses() {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 1])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("batch-rotating-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [cache])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

private func tinyMistralVLMForBatchScaling(beta: Float) throws -> Mistral3VLM {
    let json = """
        {
            "model_type": "mistral3", "vocab_size": 64, "spatial_merge_size": 1,
            "text_config": {
                "model_type": "ministral3", "hidden_size": 16, "num_hidden_layers": 2,
                "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "rms_norm_eps": 1e-5, "vocab_size": 64,
                "max_position_embeddings": 64, "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "rope_theta": 10000.0, "llama_4_scaling_beta": \(beta),
                    "original_max_position_embeddings": 4
                }
            },
            "vision_config": {
                "model_type": "pixtral", "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 32, "patch_size": 8,
                "image_size": 16
            }
        }
        """
    let configuration = try JSONDecoder().decode(
        Mistral3VLMConfiguration.self, from: Data(json.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 906)) {
        let model = Mistral3VLM(configuration)
        model.train(false)
        eval(model)
        return model
    }
}

private func expectMistralVLMChunk(
    _ model: Mistral3VLM, tokens: MLXArray, batch: [KVCache], singles: [[KVCache]],
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
            "row \(row) must use its own positions for RoPE and query scaling",
            sourceLocation: sourceLocation)
    }
}

/// The row at `index`, as the `RotatingKVCache` a batch gives back.
private func row(_ cache: BatchRotatingKVCache, _ index: Int = 0) throws -> RotatingKVCache {
    try #require(cache.extract(index) as? RotatingKVCache)
}

private func append(_ positions: Range<Int>, to cache: any KVCache) {
    let keys = MLXArray(positions.map(Float.init)).reshaped([1, 1, positions.count, 1])
    _ = cache.update(keys: keys, values: keys + 100)
}

private func view(_ cache: RotatingKVCache) throws -> (MLXArray, MLXArray) {
    let window = try #require(cache.maxSize)
    return try #require(cache.logicalView(tail: window))
}

private func expectHistory(_ actual: RotatingKVCache, _ expected: RotatingKVCache) throws {
    #expect(actual.offset == expected.offset)
    #expect(actual.maxSize == expected.maxSize)
    let a = try view(actual)
    let b = try view(expected)
    #expect(a.0.asArray(Float.self) == b.0.asArray(Float.self))
    #expect(a.1.asArray(Float.self) == b.1.asArray(Float.self))
}

/// A sliding-window model's batched caches must give every row the logits it gets on its own,
/// also once rows have wrapped their windows.
@Suite(.serialized)
struct BatchedSlidingWindowParityTests {

    @Test("A ragged batch of Gemma 3 prompts decodes past its window like separate sequences")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
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

            let fresh = try prompts.map { _ in try model.newCache(parameters: nil) }
            let batched = try fresh[0].indices.map { layer in
                try mergeRows(fresh.map { $0[layer] })
            }
            #expect(batched.contains { $0 is BatchRotatingKVCache })
            #expect(batched.contains { $0 is BatchKVCache })
            batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
            let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: batched)
            batched.forEach { $0.finalize() }
            for (row, prompt) in prompts.enumerated() {
                let actual = logits[row ..< row + 1, (prompt.count - 1) ..< prompt.count, 0...]
                #expect(maxAbsDifference(actual, expected[row]) < 1e-4, "prefill, row \(row)")
            }

            // Twelve steps take both rows past the window of eight.
            for step in 0 ..< 12 {
                let token = Int32(10 + step % 6)
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                for row in prompts.indices {
                    let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                        "step \(step), row \(row)")
                }
            }
            try expectExtractedRowsMatch(batched, singles)

            batched.forEach { filterRows($0, keeping: batchRows(1)) }
            let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: batched)
            let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
            #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
        }
    }

    @Test(
        "Prefilled rows, wrapped or not, merge and decode like separate sequences",
        arguments: [false, true])
    func mergedRows(overlong: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            if overlong {
                eval(model(MLXArray(Array(1 ... 11)).reshaped([1, 11]), cache: singles[0]))
            } else {
                eval(model(MLXArray(Array(1 ... 8)).reshaped([1, 8]), cache: singles[0]))
                for token in 9 ... 11 {
                    eval(model(MLXArray([token]).reshaped([1, 1]), cache: singles[0]))
                }
            }
            eval(model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: singles[1]))

            // The batch is built from copies, so the single caches continue on their own.
            let batched = try singles[0].indices.map { layer in
                try mergeRows(singles.map { $0[layer].copy() })
            }
            for cache in batched {
                let positioned = try #require(cache as? any BatchPositionedKVCache)
                #expect(positioned.batchOffset.asArray(Int.self) == [11, 3])
            }
            try expectExtractedRowsMatch(batched, singles)

            for step in 0 ..< 6 {
                let token = 4 + step
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                        "step \(step), row \(row)")
                }
            }
            try expectExtractedRowsMatch(batched, singles)

            batched.forEach { filterRows($0, keeping: batchRows(0)) }
            let continuation = MLXArray([4, 5]).reshaped([1, 2])
            let expected = model(continuation, cache: singles[0])
            #expect(maxAbsDifference(model(continuation, cache: batched), expected) < 1e-4)

            // Two prefilled rows join the live one: every layer holds the same rows in
            // the same order.
            let joining = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            eval(model(MLXArray(Array(1 ... 5)).reshaped([1, 5]), cache: joining[0]))
            eval(model(MLXArray([1, 2]).reshaped([1, 2]), cache: joining[1]))
            let layers = try batched.map { try #require($0 as? any BatchableKVCache) }
            let added = try layers.indices.map { layer in
                try mergeRows(joining.map { $0[layer].copy() })
            }
            for (layer, other) in zip(layers, added) {
                try layer.validateExtension(other: other)
            }
            for (layer, other) in zip(layers, added) { layer.extend(other: other) }
            func positions(_ cache: any KVCache) throws -> [Int] {
                try #require(cache as? any BatchPositionedKVCache).batchOffset.asArray(Int.self)
            }
            for cache in batched { #expect(try positions(cache) == [19, 5, 2]) }

            // The longest row leaves: the survivors keep their order and positions on
            // every layer.
            layers.forEach { $0.filter(batchIndices: batchRows(1, 2)) }
            for cache in batched {
                #expect(try positions(cache) == [5, 2])
                #expect(cache.offset == 5)
            }
            try expectExtractedRowsMatch(batched, joining)

            let output = model(MLXArray([6, 6]).reshaped([2, 1]), cache: batched)
            for row in 0 ..< 2 {
                let single = model(MLXArray([6]).reshaped([1, 1]), cache: joining[row])
                #expect(
                    maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                    "after removal, row \(row)")
            }
        }
    }
}

func tinyGemma3() -> Gemma3TextModel {
    let configuration = MLXLLM.Gemma3TextConfiguration(
        modelType: "gemma3_text", hiddenSize: 16, hiddenLayers: 4, intermediateSize: 32,
        attentionHeads: 2, headDim: 8, rmsNormEps: 1e-5, vocabularySize: 16, kvHeads: 1,
        ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
        queryPreAttnScalar: 8, slidingWindow: 8, slidingWindowPattern: 2,
        maxPositionEmbeddings: 128)
    return withRandomState(MLXRandom.RandomState(seed: 503)) {
        let model = Gemma3TextModel(configuration)
        eval(model)
        return model
    }
}

/// One layer's rows as a batch, on the row type.
private func mergeRows(_ caches: [any KVCache]) throws -> any KVCache {
    if let rows = caches as? [RotatingKVCache] {
        return try RotatingKVCache.merge(rows)
    }
    return KVCacheSimple.merge(try caches.map { try #require($0 as? KVCacheSimple) })
}

private func filterRows(_ cache: any KVCache, keeping indices: MLXArray) {
    switch cache {
    case let batch as BatchRotatingKVCache: batch.filter(batchIndices: indices)
    case let batch as BatchKVCache: batch.filter(batchIndices: indices)
    default: Issue.record("not a batched cache: \(type(of: cache))")
    }
}

private func extractRow(_ cache: any KVCache, _ index: Int) throws -> any KVCache {
    switch cache {
    case let batch as BatchRotatingKVCache: return batch.extract(index)
    case let batch as BatchKVCache: return batch.extract(index)
    default: throw KVCacheError(message: "not a batched cache: \(type(of: cache))")
    }
}

/// Every row extracted from `batch` holds the offset and the retained window of its single
/// cache.
private func expectExtractedRowsMatch(_ batch: [any KVCache], _ singles: [[any KVCache]])
    throws
{
    func retained(_ cache: any KVCache) throws -> [MLXArray] {
        guard let rotating = cache as? RotatingKVCache else { return cache.state }
        let window = try #require(rotating.maxSize)
        let view = try #require(rotating.logicalView(tail: window))
        return [view.0, view.1]
    }
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = try extractRow(batch[layer], row)
            #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            let a = try retained(actual)
            let b = try retained(caches[layer])
            try #require(a.count == b.count)
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape, "row \(row), layer \(layer)")
                #expect(
                    maxAbsDifference(actualArray, expectedArray) < 1e-4,
                    "row \(row), layer \(layer)")
            }
        }
    }
}
