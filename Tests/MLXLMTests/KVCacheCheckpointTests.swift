// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

final class KVCacheCheckpointTests: XCTestCase {
    private final class CountingCache: KVCacheSimple {
        var copies = 0
        override func copy() -> any KVCache {
            copies += 1
            return super.copy()
        }
    }

    private func rows(_ positions: Range<Int>) -> MLXArray {
        MLXArray(positions.map(Float.init)).reshaped(1, 1, positions.count, 1)
    }

    /// A storage over a sliding window, a global layer, and recurrent state that has seen
    /// `positions`. The recurrent state holds the last position it saw.
    private func storage(_ positions: Range<Int>) -> KVCacheStorage {
        let storage = KVCacheStorage(
            [RotatingKVCache(maxSize: 8), KVCacheSimple(), MambaCache()], plan: .disabled)
        write(positions, into: storage)
        return storage
    }

    private func write(_ positions: Range<Int>, into storage: KVCacheStorage) {
        for position in positions {
            let row = rows(position ..< position + 1)
            for entry in KVCacheTree.leaves(in: storage.cache).map(\.cache) {
                if let recurrent = entry as? MambaCache {
                    recurrent[0] = MLXArray(Float(position))
                    recurrent.offset += 1
                } else {
                    _ = entry.update(keys: row, values: row)
                }
            }
        }
        storage.commitProcessedTokens(positions.count)
    }

    private func contents(_ storage: KVCacheStorage) -> [[[Float]]] {
        storage.cache.map { $0.state.map { $0.asArray(Float.self) } }
    }

    private func assertSameContents(
        _ actual: KVCacheStorage, _ expected: KVCacheStorage, line: UInt = #line
    ) {
        XCTAssertEqual(actual.processedTokenCount, expected.processedTokenCount, line: line)
        XCTAssertEqual(actual.cache.map(\.offset), expected.cache.map(\.offset), line: line)
        XCTAssertEqual(contents(actual), contents(expected), line: line)
    }

    func testRestoreReturnsAWrappedWindowAndRecurrentStateToTheCheckpoint() throws {
        let live = storage(0 ..< 12)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        write(12 ..< 20, into: live)
        XCTAssertFalse(canTrimPromptCache(live.cache, numTokens: 8))

        XCTAssertTrue(live.restore(checkpoint))

        let reference = storage(0 ..< 12)
        assertSameContents(live, reference)
        write(12 ..< 13, into: live)
        write(12 ..< 13, into: reference)
        assertSameContents(live, reference)
    }

    func testCheckpointCanBeRestoredAgain() throws {
        let live = storage(0 ..< 12)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        write(12 ..< 20, into: live)
        XCTAssertTrue(live.restore(checkpoint))
        write(30 ..< 40, into: live)

        XCTAssertTrue(live.restore(checkpoint))
        assertSameContents(live, storage(0 ..< 12))
    }

    func testRestoreRefusesACheckpointAheadOfTheTimelineAndChangesNothing() throws {
        let live = storage(0 ..< 4)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        let earlier = KVCacheStorage(
            [RotatingKVCache(maxSize: 8), KVCacheSimple(), MambaCache()], plan: .disabled)
        write(0 ..< 2, into: earlier)
        let before = contents(earlier)

        XCTAssertFalse(earlier.canRestore(checkpoint))
        XCTAssertFalse(earlier.restore(checkpoint))
        XCTAssertEqual(earlier.processedTokenCount, 2)
        XCTAssertEqual(contents(earlier), before)
    }

    func testRestoreRefusesAnotherLayout() throws {
        let checkpoint = try XCTUnwrap(storage(0 ..< 4).checkpoint())
        let other = KVCacheStorage([KVCacheSimple(), MambaCache()], plan: .disabled)
        write(0 ..< 6, into: other)

        XCTAssertFalse(other.restore(checkpoint))
        XCTAssertEqual(other.processedTokenCount, 6)
    }

    func testTrimmableEntriesNeedNoCheckpoint() throws {
        let live = KVCacheStorage([KVCacheSimple(), KVCacheSimple()], plan: .disabled)
        write(0 ..< 6, into: live)
        XCTAssertFalse(live.needsCheckpoint)
        XCTAssertTrue(storage(0 ..< 6).needsCheckpoint)

        let checkpoint = try XCTUnwrap(live.checkpoint())
        write(6 ..< 10, into: live)
        XCTAssertTrue(live.restore(checkpoint))
        XCTAssertEqual(live.cache.map(\.offset), [6, 6])
    }

    func testNestedCheckpointNeverCopiesGlobalHistory() throws {
        let global = CountingCache()
        let transcript = TranscriptKVCache(tokens: Array(0 ..< 12))
        let live = KVCacheStorage(
            [CacheList(CacheList(MambaCache(), global), RotatingKVCache(maxSize: 8)), transcript],
            plan: .disabled, processedTokenCount: 0)
        write(0 ..< 12, into: live)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        XCTAssertEqual(global.copies, 0)
        write(12 ..< 20, into: live)
        transcript.append(Array(12 ..< 20))

        XCTAssertTrue(live.restore(checkpoint))
        XCTAssertEqual(global.offset, 12)
        XCTAssertEqual(transcript.tokens, Array(0 ..< 12))
        let leaves = KVCacheTree.leaves(in: live.cache)
        XCTAssertEqual(leaves.map(\.path), [[0, 0, 0], [0, 0, 1], [0, 1]])
        XCTAssertEqual(leaves.map { $0.cache.offset }, [12, 12, 12])
        XCTAssertEqual(global.copies, 0)
    }

    func testUnknownLeafDeclinesWithoutCallingCopy() {
        let live = KVCacheStorage([BaseKVCache()], plan: .disabled)
        XCTAssertFalse(live.needsCheckpoint)
        XCTAssertNil(live.checkpoint())
    }

    func testRotatingCheckpointExcludesTemporaryPrefillRows() throws {
        let entry = RotatingKVCache(maxSize: 8, keep: 2, rewindCapacity: 4)
        let live = KVCacheStorage([entry], plan: .disabled)
        let prefill = rows(0 ..< 128)
        _ = entry.update(keys: prefill, values: prefill)
        live.commitProcessedTokens(128)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        XCTAssertEqual(checkpoint.retainedBytes, 2 * 12 * MemoryLayout<Float>.size)
        let reference = entry.copy()
        write(128 ..< 140, into: live)
        XCTAssertTrue(live.restore(checkpoint))
        for position in 128 ..< 132 {
            let row = rows(position ..< position + 1)
            let actual = live.cache[0].update(keys: row, values: row)
            let expected = reference.update(keys: row, values: row)
            XCTAssertTrue(all(actual.0 .== expected.0).item(Bool.self))
            XCTAssertTrue(all(actual.1 .== expected.1).item(Bool.self))
        }
    }

    func testEqualLeafCountsWithDifferentPathsCannotRestore() throws {
        let live = storage(0 ..< 4)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        live.replace(with: [CacheList(live.cache[0], live.cache[1]), live.cache[2]])
        XCTAssertFalse(live.restore(checkpoint))
        XCTAssertEqual(live.processedTokenCount, 4)
    }

    private func append(_ count: Int, to cache: VarianceNormalizedKVCache) {
        let row = MLXArray((0 ..< count * 32).map { Float(($0 * 13) % 101) / 101 })
            .reshaped(1, 1, count, 32)
        cache.updateForAttention(keys: row, values: row)
        eval(cache)
    }

    func testTileCheckpointRetainsOnlyRawTailAcrossSlabCompaction() throws {
        let entry = VarianceNormalizedKVCache(tileSize: 32)
        append(165, to: entry)
        let tail = entry.checkpoint()
        XCTAssertEqual(tail.retainedBytes, 2 * 5 * 32 * MemoryLayout<Float>.size)
        XCTAssertEqual(entry.maxTrimCount, 5)
        XCTAssertFalse(entry.isTrimmable(after: .max))
        let live = KVCacheStorage([entry], plan: .disabled)
        XCTAssertTrue(live.needsCheckpoint)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        let reference = entry.state.map { $0[.ellipsis] }

        for _ in 0 ..< 2 {
            append(32 * 35, to: entry)
            live.commitProcessedTokens(32 * 35)
            XCTAssertTrue(live.restore(checkpoint))
            XCTAssertEqual(entry.offset, 165)
            XCTAssertEqual(entry.state.count, reference.count)
            for (actual, expected) in zip(entry.state, reference) {
                XCTAssertTrue(all(actual .== expected).item(Bool.self))
            }
        }
        append(32 * 32, to: entry)
        XCTAssertEqual(entry.checkpoint().retainedBytes, tail.retainedBytes)
    }

    func testConversionToTileCacheRefusesAnUnsavedRawTail() throws {
        let simple = KVCacheSimple()
        let row = MLXArray.zeros([1, 1, 5, 32])
        _ = simple.update(keys: row, values: row)
        let live = KVCacheStorage([simple], plan: .disabled)
        let checkpoint = try XCTUnwrap(live.checkpoint())
        live.replace(with: [simple.toVarianceNormalized(tileSize: 32)])
        XCTAssertFalse(live.restore(checkpoint))
        XCTAssertEqual(live.processedTokenCount, 5)
    }

    func testPlannedTileCompressionRequiresACheckpointForGlobalCaches() throws {
        let configuration = KVCacheConfiguration(strategy: .varianceNormalized(.memoryFirst))
        let live = KVCacheStorage(
            [KVCacheSimple()], plan: KVCachePlan(configuration: configuration))
        XCTAssertTrue(live.needsCheckpoint)
    }
}
