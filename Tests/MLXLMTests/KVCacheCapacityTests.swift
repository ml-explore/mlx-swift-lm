// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct KVCacheCapacityTests {
    private func tensor(_ n: Int, dim: Int, salt: Int, dtype: DType) -> MLXArray {
        let shape = [2, 2, n, dim * 2]
        let data = (0 ..< shape.reduce(1, *)).map { Float(($0 * 13 + salt) % 113 - 56) / 16 }
        return MLXArray(data, shape).asType(dtype)[.ellipsis, .stride(by: 2)]
    }

    private func identical(_ a: MLXArray, _ b: MLXArray) {
        #expect(a.shape == b.shape)
        #expect(a.dtype == b.dtype)
        #expect(
            a.asType(.float32).asArray(Float.self).map(\.bitPattern)
                == b.asType(.float32).asArray(Float.self).map(\.bitPattern))
    }

    @Test func capacityBoundariesAndOverflow() {
        for (current, required, expected) in [
            (0, 1, 256), (0, 513, 768), (256, 256, 256), (256, 257, 512),
            (1024, 1025, 1280), (4096, 4097, 5120),
            (32768, 32769, 40960), (65536, 65537, 73728), (1024, 20000, 20224),
        ] {
            #expect(
                KVCacheCapacity.next(current: current, required: required, step: 256) == expected)
        }
        #expect(
            KVCacheCapacity.next(current: Int.max - 3, required: Int.max - 2, step: 256) == Int.max)
        #expect(
            KVCacheCapacity.next(current: 256, required: 128, step: 256, reservation: 8192) == 8192)
        for step in [1, 17, 256, 1000, 16384] {
            for current in stride(from: 0, through: 100000, by: 997) {
                let result = KVCacheCapacity.next(
                    current: current, required: current + 1, step: step)
                #expect(result >= current + 1)
                #expect(result % step == 0)
                #expect(result - current <= max(8192, step) + step - 1)
            }
        }
    }

    @Test func mixedAppendsTrimAndStepChangesPreserveLiveRows() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let cache = KVCacheSimple()
            var oracleK: MLXArray?
            var oracleV: MLXArray?
            for (i, n) in [1, 16, 239, 1, 511, 257, 7, 768, 1, 8193].enumerated() {
                if i == 6 { cache.step = 17 }
                let k = tensor(n, dim: 8, salt: i, dtype: dtype)
                let v = tensor(n, dim: 12, salt: i + 31, dtype: dtype)
                let result = cache.update(keys: k, values: v)
                oracleK = oracleK.map { concatenated([$0, k], axis: 2) } ?? k
                oracleV = oracleV.map { concatenated([$0, v], axis: 2) } ?? v
                identical(result.0, oracleK!)
                identical(result.1, oracleV!)
                #expect(cache.state[0].dim(2) == cache.offset)
                #expect(cache.keys!.dim(2) >= cache.offset)
            }
            #expect(cache.trim(cache.offset - 1024) > 0)
            cache.step = 256
            let n = cache.keys!.dim(2) - 1024 + 3
            let k = tensor(n, dim: 8, salt: 901, dtype: dtype)
            let v = tensor(n, dim: 12, salt: 903, dtype: dtype)
            let result = cache.update(keys: k, values: v)
            identical(result.0, concatenated([oracleK![.ellipsis, ..<1024, 0...], k], axis: 2))
            identical(result.1, concatenated([oracleV![.ellipsis, ..<1024, 0...], v], axis: 2))
            let count = cache.offset
            #expect(cache.trim(Int.max) == count)
            let restarted = cache.update(keys: k, values: v)
            identical(restarted.0, k)
            identical(restarted.1, v)
        }
    }

    @Test func coldAndWarmPromptReservationNeedOneAllocation() throws {
        for warm in [false, true] {
            let cache = KVCacheSimple()
            if warm {
                _ = cache.update(
                    keys: tensor(17, dim: 8, salt: 1, dtype: .bfloat16),
                    values: tensor(17, dim: 12, salt: 2, dtype: .bfloat16))
            }
            let prefix = cache.offset
            let required = prefix + 8192
            withReservedPromptCache([cache], additionalTokens: 8192) {
                for i in 0 ..< 16 {
                    let k = tensor(512, dim: 8, salt: i, dtype: .bfloat16)
                    let v = tensor(512, dim: 12, salt: i + 1, dtype: .bfloat16)
                    let output = cache.update(keys: k, values: v)
                    #expect(output.0.dim(2) == prefix + (i + 1) * 512)
                    #expect(cache.keys!.dim(2) == ((required + 255) / 256) * 256)
                    eval(cache)
                }
            }
            #expect(cache.offset == required)
            #expect(cache.state[0].dim(2) == required)
        }
    }

    @Test func unusedReservationIsClearedOnErrorTrimRestoreAndCopy() throws {
        enum Failure: Error { case cancelled }
        let cache = KVCacheSimple()
        #expect(throws: Failure.self) {
            try withReservedPromptCache([CacheList(cache, MambaCache())], additionalTokens: 8192) {
                throw Failure.cancelled
            }
        }
        #expect(cache.state.isEmpty)
        let row = tensor(1, dim: 8, salt: 1, dtype: .float16)
        _ = cache.update(keys: row, values: row)
        #expect(cache.keys!.dim(2) == 256)
        cache.reserveCapacity(32768)
        let copy = try #require(cache.copy() as? KVCacheSimple)
        _ = copy.update(keys: row, values: row)
        #expect(copy.keys!.dim(2) == 512)
        _ = cache.trim(0)
        _ = cache.update(keys: row, values: row)
        #expect(cache.keys!.dim(2) == 256)
        cache.reserveCapacity(32768)
        cache.state = cache.state
        _ = cache.update(keys: row, values: row)
        #expect(cache.keys!.dim(2) == 512)
        identical(copy.state[0], cache.state[0][.ellipsis, ..<2, 0...])
    }

    @Test func batchElementsDoNotDuplicateTheCacheTimeline() {
        let input = LMInput(tokens: MLXArray.zeros([2, 4096], dtype: .int32))
        let cache = KVCacheSimple()
        withReservedPromptCache([cache], additionalTokens: input.text.cacheSequenceLength) {
            let k = tensor(512, dim: 8, salt: 1, dtype: .bfloat16)
            _ = cache.update(keys: k, values: k)
            #expect(cache.keys!.dim(2) == 4096)
            #expect(cache.offset == 512)
        }
    }

    @Test func serializationQuantizationAndCopiesExcludeSpareCapacity() throws {
        let cache = KVCacheSimple()
        cache.reserveCapacity(4096)
        _ = cache.update(
            keys: tensor(129, dim: 64, salt: 3, dtype: .bfloat16),
            values: tensor(129, dim: 96, salt: 5, dtype: .bfloat16))
        #expect(cache.keys!.dim(2) == 4096)
        let saved = cache.state
        eval(saved)
        let copy = try #require(cache.copy() as? KVCacheSimple)
        #expect(copy.step == cache.step)
        _ = copy.trim(1)
        _ = copy.update(
            keys: tensor(2, dim: 64, salt: 7, dtype: .bfloat16),
            values: tensor(2, dim: 96, salt: 11, dtype: .bfloat16))
        eval(copy)
        identical(cache.state[0], saved[0])
        identical(cache.state[1], saved[1])
        let quantized = try cache.toQuantized(groupSize: 32, bits: 4)
        #expect(quantized.offset == 129)
        #expect(quantized.state.allSatisfy { $0.dim(2) == 129 })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache])
        let (restored, _) = try loadPromptCache(url: url)
        #expect(restored[0].offset == 129)
        identical(restored[0].state[0], saved[0])
        identical(restored[0].state[1], saved[1])
    }

    @Test func specialValuesSurviveLazyGrowth() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let cache = KVCacheSimple()
            let row = MLXArray(
                [Float(0), -0.0, .infinity, -.infinity, .nan, 1, -1, 0.000001],
                [1, 1, 1, 8]
            ).asType(dtype)
            let expected = row.asType(.float32).asArray(Float.self).map(\.bitPattern)
            for i in 0 ..< 1300 {
                _ = cache.update(keys: row, values: row)
                if i % 100 == 0 { eval(cache) }
            }
            let actual = cache.state[0].asType(.float32).asArray(Float.self).map(\.bitPattern)
            #expect(actual == (0 ..< 1300).flatMap { _ in expected })
        }
    }
}
