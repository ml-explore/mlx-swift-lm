// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class KVCacheCheckpointTests: XCTestCase {
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
            for entry in storage.cache {
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

    func testRestoreReturnsAWrappedWindowAndRecurrentStateToTheCheckpoint() {
        let live = storage(0 ..< 12)
        let checkpoint = live.checkpoint()
        write(12 ..< 20, into: live)
        XCTAssertFalse(canTrimPromptCache(live.cache, numTokens: 8))

        XCTAssertTrue(live.restore(checkpoint))

        let reference = storage(0 ..< 12)
        assertSameContents(live, reference)
        write(12 ..< 13, into: live)
        write(12 ..< 13, into: reference)
        assertSameContents(live, reference)
    }

    func testCheckpointCanBeRestoredAgain() {
        let live = storage(0 ..< 12)
        let checkpoint = live.checkpoint()
        write(12 ..< 20, into: live)
        XCTAssertTrue(live.restore(checkpoint))
        write(30 ..< 40, into: live)

        XCTAssertTrue(live.restore(checkpoint))
        assertSameContents(live, storage(0 ..< 12))
    }

    func testRestoreRefusesACheckpointAheadOfTheTimelineAndChangesNothing() {
        let live = storage(0 ..< 4)
        let checkpoint = live.checkpoint()
        let earlier = KVCacheStorage(
            [RotatingKVCache(maxSize: 8), KVCacheSimple(), MambaCache()], plan: .disabled)
        write(0 ..< 2, into: earlier)
        let before = contents(earlier)

        XCTAssertFalse(earlier.canRestore(checkpoint))
        XCTAssertFalse(earlier.restore(checkpoint))
        XCTAssertEqual(earlier.processedTokenCount, 2)
        XCTAssertEqual(contents(earlier), before)
    }

    func testRestoreRefusesAnotherLayout() {
        let checkpoint = storage(0 ..< 4).checkpoint()
        let other = KVCacheStorage([KVCacheSimple(), MambaCache()], plan: .disabled)
        write(0 ..< 6, into: other)

        XCTAssertFalse(other.restore(checkpoint))
        XCTAssertEqual(other.processedTokenCount, 6)
    }

    func testTrimmableEntriesNeedNoCheckpoint() {
        let live = KVCacheStorage([KVCacheSimple(), KVCacheSimple()], plan: .disabled)
        write(0 ..< 6, into: live)
        XCTAssertTrue(live.canRewindToAnyPosition)
        XCTAssertFalse(storage(0 ..< 6).canRewindToAnyPosition)

        let checkpoint = live.checkpoint()
        write(6 ..< 10, into: live)
        XCTAssertTrue(live.restore(checkpoint))
        XCTAssertEqual(live.cache.map(\.offset), [6, 6])
    }
}
