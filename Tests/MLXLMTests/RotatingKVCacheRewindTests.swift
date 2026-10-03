// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import XCTest

@testable import MLXLMCommon

final class RotatingKVCacheRewindTests: XCTestCase {
    private func rows(_ positions: [Int]) -> MLXArray {
        MLXArray(positions.map(Float.init)).reshaped(1, 1, positions.count, 1)
    }

    private func fill(_ caches: [KVCache], positions: Range<Int>) {
        for position in positions {
            let row = rows([position])
            for cache in caches { _ = cache.update(keys: row, values: row) }
        }
    }

    func testMixedCacheRewindsAfterRotationWithReserve() throws {
        let rotating = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        let caches: [KVCache] = [rotating, KVCacheSimple()]
        fill(caches, positions: 0 ..< 20)

        XCTAssertEqual(rotating.maxTrimCount, 4)
        XCTAssertTrue(canTrimPromptCache(caches, numTokens: 3))
        XCTAssertEqual(trimPromptCache(caches, numTokens: 3), 3)
        XCTAssertEqual(caches.map(\.offset), [17, 17])
        let view = try XCTUnwrap(rotating.logicalView(tail: 8))
        XCTAssertEqual(view.0.asArray(Float.self), Array(9 ..< 17).map(Float.init))

        let reference = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        fill([reference], positions: 0 ..< 17)
        let replacement = rows([100])
        let actual = rotating.update(keys: replacement, values: replacement)
        let expected = reference.update(keys: replacement, values: replacement)
        XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
        XCTAssertEqual(actual.1.asArray(Float.self), expected.1.asArray(Float.self))
    }

    func testRewindBeyondReserveLeavesEveryLayerUnchanged() {
        let caches: [KVCache] = [
            KVCacheSimple(), RotatingKVCache(maxSize: 8, rewindCapacity: 4),
        ]
        fill(caches, positions: 0 ..< 20)
        let states = caches.map { $0.state.map { $0.asArray(Float.self) } }
        let metadata = caches.map(\.metaState)

        XCTAssertFalse(canTrimPromptCache(caches, numTokens: 5))
        XCTAssertEqual(trimPromptCache(caches, numTokens: 5), 0)
        XCTAssertEqual(caches.map(\.offset), [20, 20])
        XCTAssertEqual(caches.map { $0.state.map { $0.asArray(Float.self) } }, states)
        XCTAssertEqual(caches.map(\.metaState), metadata)
    }

    func testDefaultCacheStillRefusesRewindAfterEviction() {
        let caches: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 8)]
        fill(caches, positions: 0 ..< 20)
        XCTAssertEqual(caches[1].maxTrimCount, 0)
        XCTAssertEqual(trimPromptCache(caches, numTokens: 3), 0)
        XCTAssertEqual(caches.map(\.offset), [20, 20])
    }

    func testReservePreservesAttentionAcrossWriteShapesAndWraps() {
        for keep in [0, 2] {
            for step in [1, 4, 256] {
                let windows: [Int?] = keep == 0 ? [nil, 4, 8] : [nil, 8]
                for window in windows {
                    let cache = RotatingKVCache(
                        maxSize: 8, keep: keep, step: step, rewindCapacity: 5)
                    let reference = RotatingKVCache(maxSize: 8, keep: keep, step: step)
                    var offset = 0
                    for count in [1, 1, 9, 1, 3, 1, 1, 16, 1, 2, 1, 1] {
                        let input = rows(Array(offset ..< (offset + count)))
                        let query = MLXArray.zeros([1, 1, count, 1])
                        let mask = cache.makeMask(n: count, windowSize: window, returnArray: true)
                        let referenceMask = reference.makeMask(
                            n: count, windowSize: window, returnArray: true)
                        let actual = cache.update(keys: input, values: input)
                        let expected = reference.update(keys: input, values: input)
                        let output = MLXFast.scaledDotProductAttention(
                            queries: query, keys: actual.0, values: actual.1, scale: 1, mask: mask)
                        let expectedOutput = MLXFast.scaledDotProductAttention(
                            queries: query, keys: expected.0, values: expected.1, scale: 1,
                            mask: referenceMask)
                        XCTAssertTrue(
                            allClose(output, expectedOutput, atol: 1e-5).item(Bool.self),
                            "keep=\(keep), step=\(step), window=\(String(describing: window)), offset=\(offset), count=\(count)"
                        )
                        XCTAssertEqual(actual.0.shape, expected.0.shape)
                        XCTAssertEqual(cache.maxSize, 8)
                        if count == 1 {
                            XCTAssertLessThanOrEqual(cache.innerState()[0].dim(2), 13)
                        }
                        offset += count
                    }
                }
            }
        }
    }

    func testViewsSelectOnlyRequestedRowsAcrossEveryRingPosition() throws {
        for keep in [0, 2, 7] {
            for reserve in [0, 1, 4, 16] {
                let cache = RotatingKVCache(maxSize: 8, keep: keep, rewindCapacity: reserve)
                for position in 0 ..< 70 {
                    _ = cache.update(keys: rows([position]), values: rows([position + 100]))
                    for tail in [-1, 0, 1, 6, 8, 20] {
                        let available = min(position + 1, 8 + reserve)
                        let prefix = min(keep, available)
                        let limit = reserve == 0 ? tail : min(tail, 8)
                        let count = max(prefix, min(max(limit, 0), available))
                        let expected =
                            Array(0 ..< prefix)
                            + Array((position + 1 - (count - prefix)) ..< (position + 1))
                        let view = try XCTUnwrap(cache.logicalView(tail: tail))
                        XCTAssertEqual(view.0.shape, [1, 1, count, 1])
                        XCTAssertEqual(view.1.shape, [1, 1, count, 1])
                        if count > 0 {
                            XCTAssertEqual(view.0.asArray(Float.self), expected.map(Float.init))
                            XCTAssertEqual(
                                view.1.asArray(Float.self), expected.map { Float($0 + 100) })
                        }
                    }
                }
            }
        }
    }

    func testRepeatedRewindsConsumeAndRefillTheReserve() throws {
        for keep in [0, 2] {
            let cache = RotatingKVCache(maxSize: 8, keep: keep, step: 2, rewindCapacity: 4)
            fill([cache], positions: 0 ..< 30)
            XCTAssertEqual(trimPromptCache([cache], numTokens: 3), 3)
            XCTAssertEqual(cache.maxTrimCount, 1)
            XCTAssertEqual(trimPromptCache([cache], numTokens: 2), 0)
            XCTAssertEqual(trimPromptCache([cache], numTokens: 1), 1)
            XCTAssertEqual(cache.maxTrimCount, 0)
            fill([cache], positions: 26 ..< 30)
            XCTAssertEqual(cache.maxTrimCount, 4)
            XCTAssertEqual(trimPromptCache([cache], numTokens: 4), 4)
            let reference = RotatingKVCache(maxSize: 8, keep: keep, rewindCapacity: 4)
            fill([reference], positions: 0 ..< 26)
            let actual = try XCTUnwrap(cache.logicalView(tail: 8))
            let expected = try XCTUnwrap(reference.logicalView(tail: 8))
            XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
        }
    }

    func testPrefillTailCanRewindWithoutAnExplicitReserve() throws {
        let cache = RotatingKVCache(maxSize: 8)
        fill([cache], positions: 0 ..< 20)
        let next = rows(Array(20 ..< 25))
        _ = cache.update(keys: next, values: next)
        XCTAssertEqual(cache.maxTrimCount, 4)
        XCTAssertEqual(trimPromptCache([cache], numTokens: 4), 4)
        XCTAssertEqual(cache.offset, 21)
        let view = try XCTUnwrap(cache.logicalView(tail: 8))
        XCTAssertEqual(view.0.asArray(Float.self), Array(13 ..< 21).map(Float.init))
    }

    func testEmptyWritesAndNonpositiveTrimsLeaveReserveUntouched() {
        let cache = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        fill([cache], positions: 0 ..< 20)
        let state = cache.state.map { $0.asArray(Float.self) }
        let metadata = cache.metaState
        let empty = rows([])
        _ = cache.update(keys: empty, values: empty)
        XCTAssertEqual(cache.trim(0), 0)
        XCTAssertEqual(cache.trim(-1), 0)
        XCTAssertEqual(trimPromptCache([cache], numTokens: -1), 0)
        XCTAssertEqual(cache.state.map { $0.asArray(Float.self) }, state)
        XCTAssertEqual(cache.metaState, metadata)
        XCTAssertFalse(cache.isTrimmable(after: Int.max))
        XCTAssertFalse(cache.isTrimmable(after: -1))
    }

    func testOversizedTrimClampsWhenTheEntirePrefixIsRetained() {
        for count in [0, 1, 7, 8, 12] {
            let cache = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
            fill([cache], positions: 0 ..< count)
            XCTAssertEqual(trimPromptCache([cache], numTokens: Int.max), count)
            XCTAssertEqual(cache.offset, 0)
            fill([cache], positions: 0 ..< 3)
            XCTAssertEqual(cache.maxTrimCount, 3)
        }
    }

    func testNestedCachesUseTheSmallestRewindLimit() {
        let small = RotatingKVCache(maxSize: 8, rewindCapacity: 2)
        let large = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        let simple = KVCacheSimple()
        fill([small, large, simple], positions: 0 ..< 20)
        let composite = CacheList(simple, CacheList(small, large))
        XCTAssertEqual(composite.maxTrimCount, 2)
        XCTAssertEqual(trimPromptCache([composite], numTokens: 3), 0)
        XCTAssertEqual(trimPromptCache([composite], numTokens: 2), 2)
        XCTAssertEqual([small.offset, large.offset, simple.offset], [18, 18, 18])
        XCTAssertEqual(CacheList(small, MambaCache()).maxTrimCount, 0)
    }

    func testReserveSurvivesCopyAndPersistenceAfterRewind() throws {
        for keep in [0, 2] {
            for trim in [0, 2, 4] {
                let cache = RotatingKVCache(maxSize: 8, keep: keep, rewindCapacity: 4)
                fill([cache], positions: 0 ..< 30)
                XCTAssertEqual(trimPromptCache([cache], numTokens: trim), trim)
                let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString + ".safetensors")
                defer { try? FileManager.default.removeItem(at: url) }
                try savePromptCache(url: url, cache: [cache])
                let (loaded, _) = try loadPromptCache(url: url)
                let copied = try XCTUnwrap(cache.copy() as? RotatingKVCache)
                let restored = try XCTUnwrap(loaded.first as? RotatingKVCache)
                for candidate in [copied, restored] {
                    XCTAssertEqual(candidate.rewindCapacity, 4)
                    XCTAssertEqual(candidate.maxTrimCount, cache.maxTrimCount)
                    XCTAssertEqual(candidate.metaState, cache.metaState)
                    let expected = try XCTUnwrap(cache.logicalView(tail: 8))
                    let actual = try XCTUnwrap(candidate.logicalView(tail: 8))
                    XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
                    XCTAssertEqual(trimPromptCache([candidate], numTokens: 4 - trim), 4 - trim)
                    fill([candidate], positions: 26 ..< 40)
                }
                XCTAssertEqual(cache.offset, 30 - trim)
            }
        }
    }

    func testReserveMetadataCanBeRestoredBeforeOrAfterArrays() throws {
        let source = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        fill([source], positions: 0 ..< 30)
        for metadataFirst in [false, true] {
            let restored = RotatingKVCache(maxSize: 1)
            if metadataFirst { restored.metaState = source.metaState }
            restored.state = source.state.map { $0[.ellipsis] }
            if !metadataFirst { restored.metaState = source.metaState }
            XCTAssertEqual(trimPromptCache([restored], numTokens: 4), 4)
            XCTAssertEqual(restored.offset, 26)
            let view = try XCTUnwrap(restored.logicalView(tail: 8))
            XCTAssertEqual(view.0.asArray(Float.self), Array(18 ..< 26).map(Float.init))
        }
    }

    func testReusePolicyChecksTheRequestedRewindLength() {
        let tokens = Array(0 ..< 20)
        let state = PromptCacheState(
            cachedTokens: tokens, processedTokenCount: 20,
            mainCacheIsAligned: true, maxTrimCount: 4)
        let policy = PromptCacheReusePolicy()
        XCTAssertEqual(
            policy.decide(
                turn: .init(promptTokens: Array(tokens.prefix(17)) + [100]), cache: state),
            .trimToCommonPrefix(commonPrefixLength: 17, trimCount: 3))
        XCTAssertEqual(
            policy.decide(
                turn: .init(promptTokens: Array(tokens.prefix(15)) + [100]), cache: state), .rebuild
        )
    }

    func testGPTOSSLogitsAfterRewindMatchColdPrefix() throws {
        let json = """
            {"model_type":"gpt_oss","num_hidden_layers":2,"num_local_experts":4,
             "num_experts_per_tok":2,"vocab_size":32,"rms_norm_eps":1e-6,
             "hidden_size":16,"intermediate_size":16,"head_dim":8,
             "num_attention_heads":2,"num_key_value_heads":1,"sliding_window":8,
             "rope_theta":10000.0,"layer_types":["sliding_attention","full_attention"]}
            """
        let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 618)) { GPTOSSModel(config) }
        let configuration = KVCacheConfiguration(rewind: try .init(maxTokens: 4))
        let cache = try model.newCache(parameters: GenerateParameters(kvCache: configuration))
        let reference = try model.newCache(parameters: nil)
        try validateKVCacheCompatibility(cache, configuration: configuration)
        XCTAssertEqual((cache[0] as? RotatingKVCache)?.rewindCapacity, 4)
        for position in 0 ..< 20 {
            let token = MLXArray([position]).reshaped(1, 1)
            eval(model(token, cache: cache))
            if position < 17 { eval(model(token, cache: reference)) }
        }
        XCTAssertEqual(trimPromptCache(cache, numTokens: 3), 3)
        for tokens in [[24], [25, 26, 27], [28]] {
            let input = MLXArray(tokens).reshaped(1, tokens.count)
            let actual = model(input, cache: cache)
            let expected = model(input, cache: reference)
            XCTAssertTrue(allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        }
    }

    func testRewindConfigurationRejectsInvalidAndUnrealizedReserves() throws {
        for count in [-1, 0] {
            XCTAssertThrowsError(try KVCacheConfiguration.Rewind(maxTokens: count))
        }
        let overflow = KVCacheConfiguration(rewind: try .init(maxTokens: Int.max))
        XCTAssertThrowsError(
            try makeSlidingWindowKVCache(parameters: .init(kvCache: overflow), window: 8))
        XCTAssertThrowsError(
            try slidingWindowCacheKind(parameters: .init(kvCache: overflow), window: 8))
        let boundedOverflow = KVCacheConfiguration(
            capacity: try .init(maxTokens: 8), rewind: overflow.rewind)
        XCTAssertThrowsError(try attentionCacheKind(parameters: .init(kvCache: boundedOverflow)))
        let configuration = KVCacheConfiguration(rewind: try .init(maxTokens: 4))
        XCTAssertThrowsError(
            try validateKVCacheCompatibility(
                [RotatingKVCache(maxSize: 8)], configuration: configuration))
        XCTAssertNoThrow(
            try validateKVCacheCompatibility([KVCacheSimple()], configuration: configuration))
        let bounded = KVCacheConfiguration(
            capacity: try .init(maxTokens: 8, preservedPrefixTokens: 2),
            rewind: try .init(maxTokens: 4))
        let cache = try XCTUnwrap(
            try makeAttentionKVCache(parameters: .init(kvCache: bounded)) as? RotatingKVCache)
        XCTAssertEqual(cache.rewindCapacity, 4)
        XCTAssertEqual(cache.maxSize, 8)
        XCTAssertEqual(cache.keepCount, 2)
    }

    func testSpeculativeCommitAndRestorePreserveTheReserve() throws {
        for start in [6, 10, 20] {
            let rotating = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
            let caches: [KVCache] = [rotating, KVCacheSimple()]
            fill(caches, positions: 0 ..< start)
            let storage = KVCacheStorage(caches, plan: .disabled)
            let round = try XCTUnwrap(storage.beginRound(maximumPositions: 4))
            let input = rows(Array(start ..< (start + 4)))
            let mask = round.caches[0].makeMask(n: 4, windowSize: 8, returnArray: true)
            for (index, cache) in round.caches.enumerated() {
                let presented = cache.update(keys: input, values: input)
                if index == 0, case .array(let array) = mask {
                    XCTAssertEqual(array.dim(-1), presented.0.dim(2))
                }
            }
            XCTAssertEqual(rotating.offset, start)
            storage.commit(round, retaining: 3)
            XCTAssertEqual(storage.rewindLastRound(2), 2)
            XCTAssertEqual(storage.processedTokenCount, start + 1)
            XCTAssertEqual(caches.map(\.offset), [start + 1, start + 1])
            XCTAssertEqual(rotating.rewindCapacity, 4)
            let reference = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
            fill([reference], positions: 0 ..< (start + 1))
            let actual = try XCTUnwrap(rotating.logicalView(tail: 8))
            let expected = try XCTUnwrap(reference.logicalView(tail: 8))
            XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
            XCTAssertGreaterThanOrEqual(rotating.maxTrimCount, reference.maxTrimCount)
        }
    }

    func testSmallWindowsAndLongPinnedPrefixesRewindExactly() throws {
        for (window, keep) in [(1, 0), (2, 1), (8, 7)] {
            for reserve in [1, 4] {
                let cache = RotatingKVCache(
                    maxSize: window, keep: keep, step: 1, rewindCapacity: reserve)
                fill([cache], positions: 0 ..< 30)
                XCTAssertEqual(cache.maxTrimCount, reserve)
                XCTAssertEqual(trimPromptCache([cache], numTokens: reserve), reserve)
                let reference = RotatingKVCache(
                    maxSize: window, keep: keep, rewindCapacity: reserve)
                fill([reference], positions: 0 ..< (30 - reserve))
                let actual = try XCTUnwrap(cache.logicalView(tail: window))
                let expected = try XCTUnwrap(reference.logicalView(tail: window))
                XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
                fill([cache], positions: (30 - reserve) ..< 40)
                XCTAssertEqual(cache.maxTrimCount, reserve)
            }
        }
    }

    func testDirectInexactTrimCannotClaimTheMissingPrefixIsRewindable() {
        let cache = RotatingKVCache(maxSize: 8, rewindCapacity: 4)
        fill([cache], positions: 0 ..< 20)
        XCTAssertEqual(cache.trim(12), 12)
        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(cache.maxTrimCount, 0)
        XCTAssertFalse(cache.isTrimmable)
        XCTAssertFalse(cache.isTrimmable(after: 1))
    }

    func testMalformedRewindMetadataThrowsWithoutRestoring() throws {
        for reserve in ["-1", "invalid", String(Int.max)] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            let metadata = ["0", "8", "256", "0", "0", "modelNative", "false", reserve]
            var entries = Dictionary(
                uniqueKeysWithValues: metadata.enumerated().map { ("0.0.\($0.offset)", $0.element) }
            )
            entries["2.0"] = "RotatingKVCache"
            try save(arrays: [String: MLXArray](), metadata: entries, url: url)
            XCTAssertThrowsError(try loadPromptCache(url: url))
        }
    }

    func testMixedCacheRewindsOversizedPrefillWithoutEvictedHistory() {
        let caches: [KVCache] = [RotatingKVCache(maxSize: 8), KVCacheSimple()]
        let original = rows(Array(0 ..< 20))
        for cache in caches {
            _ = cache.update(keys: original, values: original)
        }

        // A single prefill still holds the whole prefix, despite exceeding the window.
        XCTAssertEqual(trimPromptCache(caches, numTokens: 3), 3)
        XCTAssertEqual(caches.map(\.offset), [17, 17])

        let references: [KVCache] = [RotatingKVCache(maxSize: 8), KVCacheSimple()]
        let prefix = rows(Array(0 ..< 17))
        let replacement = rows([100])
        for (cache, reference) in zip(caches, references) {
            _ = reference.update(keys: prefix, values: prefix)
            let expected = reference.update(keys: replacement, values: replacement)
            let actual = cache.update(keys: replacement, values: replacement)
            XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
            XCTAssertEqual(actual.1.asArray(Float.self), expected.1.asArray(Float.self))
        }
    }
}
