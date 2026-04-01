import Foundation
import MLX
import MLXLMCommon
import Testing

private func makeTensor(shape: [Int], scale: Float = 0.03125, bias: Float = 0) -> MLXArray {
    let count = shape.reduce(1, *)
    let values = (0 ..< count).map { index in
        bias + Float((index % 97) - 48) * scale
    }
    return MLXArray(values).reshaped(shape).asType(.float32)
}

private func meanAbsoluteError(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    let delta = abs(lhs.asType(.float32) - rhs.asType(.float32))
    return (sum(delta).item(Float.self) / Float(delta.size))
}

@Test
func testFractionalKvBitsUseTurboQuantAutomatically() async throws {
    var cache: [KVCache] = [KVCacheSimple()]
    let keys = makeTensor(shape: [1, 2, 8, 16])
    let values = makeTensor(shape: [1, 2, 8, 16], bias: 0.25)

    _ = cache[0].update(keys: keys, values: values)
    maybeQuantizeKVCache(cache: &cache, kvBits: 3.5, quantizedKVStart: 0)

    #expect(cache[0] is TurboQuantKVCache)
}

@Test
func testExplicitTurboQuantSelectsTurboCacheForIntegerBits() async throws {
    var cache: [KVCache] = [KVCacheSimple()]
    let keys = makeTensor(shape: [1, 2, 8, 16])
    let values = makeTensor(shape: [1, 2, 8, 16], bias: 0.1)

    _ = cache[0].update(keys: keys, values: values)
    maybeQuantizeKVCache(
        cache: &cache,
        kvBits: 4.0,
        kvQuantizationScheme: .turboQuant,
        quantizedKVStart: 0
    )

    #expect(cache[0] is TurboQuantKVCache)
}

@Test
func testIntegerUniformKvBitsStillSelectUniformQuantization() async throws {
    var cache: [KVCache] = [KVCacheSimple()]
    let keys = makeTensor(shape: [1, 2, 8, 64])
    let values = makeTensor(shape: [1, 2, 8, 64], bias: -0.1)

    _ = cache[0].update(keys: keys, values: values)
    maybeQuantizeKVCache(
        cache: &cache,
        kvBits: 4.0,
        kvQuantizationScheme: .uniform,
        quantizedKVStart: 0
    )

    #expect(cache[0] is QuantizedKVCache)
}

@Test
func testMaybeQuantizeRecursesIntoCacheLists() async throws {
    let nestedCache = CacheList(MambaCache(), KVCacheSimple())
    let keys = makeTensor(shape: [1, 2, 8, 16])
    let values = makeTensor(shape: [1, 2, 8, 16], bias: 0.2)

    _ = nestedCache[1].update(keys: keys, values: values)

    var cache: [KVCache] = [nestedCache]
    maybeQuantizeKVCache(cache: &cache, kvBits: 3.5, quantizedKVStart: 0)

    let quantizedList = try #require(cache[0] as? CacheList)
    #expect(quantizedList[0] is MambaCache)
    #expect(quantizedList[1] is TurboQuantKVCache)
}

@Test
func testRotatingCachesRemainUnquantized() async throws {
    var cache: [KVCache] = [RotatingKVCache(maxSize: 32)]
    let keys = makeTensor(shape: [1, 2, 8, 16])
    let values = makeTensor(shape: [1, 2, 8, 16], bias: -0.2)

    _ = cache[0].update(keys: keys, values: values)
    maybeQuantizeKVCache(cache: &cache, kvBits: 3.5, quantizedKVStart: 0)

    #expect(cache[0] is RotatingKVCache)
}

@Test
func testTurboQuantSplitBitsImproveReconstructionOverLowerIntegerBits() async throws {
    let keys = makeTensor(shape: [1, 2, 32, 16], scale: 0.02)
    let values = makeTensor(shape: [1, 2, 32, 16], scale: 0.0175, bias: 0.15)

    let cache3 = TurboQuantKVCache(bits: 3.0)
    let cache35 = TurboQuantKVCache(bits: 3.5)

    let (keys3, values3) = cache3.update(keys: keys, values: values)
    let (keys35, values35) = cache35.update(keys: keys, values: values)

    let error3 = meanAbsoluteError(keys3, keys) + meanAbsoluteError(values3, values)
    let error35 = meanAbsoluteError(keys35, keys) + meanAbsoluteError(values35, values)

    #expect(error35 <= error3)
}

@Test
func testTurboQuantPrefillAttentionMatchesReferenceShapeAndTolerance() async throws {
    let queries = makeTensor(shape: [1, 4, 8, 16], scale: 0.015)
    let keys = makeTensor(shape: [1, 2, 8, 16], scale: 0.02, bias: 0.05)
    let values = makeTensor(shape: [1, 2, 8, 16], scale: 0.018, bias: -0.03)
    let scale = Float(0.25)

    let turboCache = TurboQuantKVCache(bits: 4.0)
    let turboOutput = attentionWithCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
        cache: turboCache,
        scale: scale,
        mask: .causal
    )
    let reference = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: .causal
    )

    #expect(turboOutput.shape == reference.shape)
    #expect(meanAbsoluteError(turboOutput, reference) < 0.15)
    #expect(turboCache.nbytes < keys.nbytes + values.nbytes)
}

@Test
func testTurboQuantDecodeAttentionMatchesReferenceShapeAndTolerance() async throws {
    let prefixKeys = makeTensor(shape: [1, 2, 12, 16], scale: 0.018, bias: 0.02)
    let prefixValues = makeTensor(shape: [1, 2, 12, 16], scale: 0.016, bias: -0.02)
    let nextQueries = makeTensor(shape: [1, 4, 1, 16], scale: 0.014, bias: 0.01)
    let nextKeys = makeTensor(shape: [1, 2, 1, 16], scale: 0.02, bias: 0.08)
    let nextValues = makeTensor(shape: [1, 2, 1, 16], scale: 0.013, bias: -0.05)
    let scale = Float(0.25)

    let turboCache = TurboQuantKVCache(bits: 4.0)
    _ = turboCache.update(keys: prefixKeys, values: prefixValues)

    let turboOutput = attentionWithCacheUpdate(
        queries: nextQueries,
        keys: nextKeys,
        values: nextValues,
        cache: turboCache,
        scale: scale,
        mask: .none
    )
    let reference = MLXFast.scaledDotProductAttention(
        queries: nextQueries,
        keys: concatenated([prefixKeys, nextKeys], axis: 2),
        values: concatenated([prefixValues, nextValues], axis: 2),
        scale: scale,
        mask: .none
    )

    #expect(turboOutput.shape == reference.shape)
    #expect(meanAbsoluteError(turboOutput, reference) < 0.15)
}

@Test
func testTurboQuantPromptCacheRoundTrips() async throws {
    let cache = TurboQuantKVCache(bits: 3.5)
    let keys = makeTensor(shape: [1, 2, 16, 16], scale: 0.02)
    let values = makeTensor(shape: [1, 2, 16, 16], scale: 0.018, bias: 0.07)

    _ = cache.update(keys: keys, values: values)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    let restored = try #require(loadedCache.first as? TurboQuantKVCache)
    let originalState = try #require(cache.dequantizedState())
    let restoredState = try #require(restored.dequantizedState())

    #expect(restored.bits == cache.bits)
    #expect(meanAbsoluteError(originalState.0, restoredState.0) < 1e-5)
    #expect(meanAbsoluteError(originalState.1, restoredState.1) < 1e-5)
}
