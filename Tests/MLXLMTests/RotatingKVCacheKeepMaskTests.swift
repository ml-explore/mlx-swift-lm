// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Token ids each query of an `n`-token write can see, read from the mask and the returned keys.
private func visibleTokens(
    _ cache: RotatingKVCache, history: Int, n: Int, windowSize: Int?, returnArray: Bool
) -> [[Int]] {
    let mask = cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    let kv = MLXArray(Int32(history) ..< Int32(history + n)).reshaped(1, 1, n, 1)
    let (keys, _) = cache.update(keys: kv, values: kv)
    let ids = keys[0, 0, 0..., 0].asArray(Int32.self).map(Int.init)
    let rows: [[Bool]]? =
        switch mask {
        case .array(let array):
            (0 ..< n).map { array[$0].asArray(Bool.self) }
        default:
            nil
        }
    return (0 ..< n).map { row in
        let end = history + row + 1
        let visible = ids.indices.filter { col in
            if let rows { rows[row][col] } else { ids[col] < end }
        }
        return visible.map { ids[$0] }.sorted()
    }
}

@Suite struct RotatingKVCacheKeepMaskTests {

    @Test func multiTokenWritesSeeThePinnedPrefixLikeOneByOneDecoding() {
        for size in [2, 4, 8] {
            for keep in Set([0, 1, size - 1]).sorted() {
                for history in [0, 1, size - 1, size, 2 * size + 3] {
                    for step in [1, size + 1] {
                        for n in [2, 3, size + 2] {
                            for windowSize in [nil, size] {
                                for returnArray in [false, true] {
                                    let cache = RotatingKVCache(maxSize: size, keep: keep)
                                    for start in stride(from: 0, to: history, by: step) {
                                        let end = Swift.min(start + step, history)
                                        let kv = MLXArray(Int32(start) ..< Int32(end))
                                            .reshaped(1, 1, end - start, 1)
                                        _ = cache.update(keys: kv, values: kv)
                                    }
                                    let visible = visibleTokens(
                                        cache, history: history, n: n, windowSize: windowSize,
                                        returnArray: returnArray)
                                    for row in 0 ..< n {
                                        let end = history + row + 1
                                        let recent = Swift.min(
                                            Swift.max(keep, end - (size - keep)), end)
                                        let expected =
                                            Array(0 ..< Swift.min(keep, end))
                                            + Array(recent ..< end)
                                        #expect(
                                            visible[row] == expected,
                                            "size \(size) keep \(keep) history \(history) step \(step) n \(n) window \(String(describing: windowSize)) array \(returnArray) row \(row)"
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test func smallerWindowDoesNotForceThePinnedPrefixVisible() {
        let cache = RotatingKVCache(maxSize: 4, keep: 1)
        for returnArray in [false, true] {
            guard
                case .array(let mask) = cache.makeMask(
                    n: 3, windowSize: 2, returnArray: returnArray)
            else {
                Issue.record("expected an array mask")
                continue
            }
            #expect(mask[2].asArray(Bool.self) == [false, true, true])
        }
    }
}
