import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
]

// MARK: - Helper

private func tempURL() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")
}

/// Assert two arrays of MLXArray are element-wise close
private func assertArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray], label: String = "") {
    #expect(lhs.count == rhs.count, "state count mismatch \(label)")
    for (i, (a, b)) in zip(lhs, rhs).enumerated() {
        #expect(a.shape == b.shape, "shape mismatch at index \(i) \(label)")
        let close = allClose(a, b).item(Bool.self)
        #expect(close, "values not close at index \(i) \(label)")
    }
}

private final class LifecycleRecordingCache: BaseKVCache {
    private(set) var preparedLengths: [Int]?
    private(set) var finalizeCallCount = 0

    override func prepare(lengths: [Int]?) {
        preparedLengths = lengths
    }

    override func finalize() {
        finalizeCallCount += 1
    }

    override func copy() -> any KVCache {
        let new = LifecycleRecordingCache()
        new.preparedLengths = preparedLengths
        new.finalizeCallCount = finalizeCallCount
        return new
    }
}

// MARK: - Original parameterized test (updated with value assertions)

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = tempURL()

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        assertArraysClose(lhs.state, rhs.state)
    }
}

@Test func testQuantizedKVCacheRestoresNonDefaultQuantizationMetadata() throws {
    let cache = QuantizedKVCache(groupSize: 64, bits: 4)
    let keys = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
    let values = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
    _ = cache.updateQuantized(keys: keys, values: values)

    #expect(cache.groupSize == 32)
    #expect(cache.bits == 4)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? QuantizedKVCache)
    #expect(restored.groupSize == 32)
    #expect(restored.bits == 4)
    #expect(restored.metaState == cache.metaState)

    let moreKeys = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
    _ = restored.updateQuantized(keys: moreKeys, values: moreValues)

    #expect(restored.groupSize == 32)
    #expect(restored.bits == 4)
}

@Test func testQuantizedKVCacheMetaStateRestoresQuantizationMetadataWithoutState() {
    let cache = QuantizedKVCache()

    cache.metaState = ["256", "11", "32", "4"]

    #expect(cache.offset == 11)
    #expect(cache.groupSize == 32)
    #expect(cache.bits == 4)
    #expect(cache.metaState == ["256", "11", "32", "4"])
}

@Test func testQuantizedKVCacheCopyPreservesRestoredQuantizationMetadata() throws {
    let cache = QuantizedKVCache()
    cache.metaState = ["256", "5", "32", "4"]

    let copied = try #require(cache.copy() as? QuantizedKVCache)

    #expect(copied.offset == 5)
    #expect(copied.groupSize == 32)
    #expect(copied.bits == 4)
    #expect(copied.metaState == cache.metaState)
}

@Test func testEmptyKVCacheSimpleToQuantizedPreservesRequestedQuantizationMetadata() throws {
    let cache = KVCacheSimple()
    cache.offset = 7

    let quantized = try cache.toQuantized(groupSize: 128, bits: 4)

    #expect(quantized.offset == 7)
    #expect(quantized.groupSize == 128)
    #expect(quantized.bits == 4)
    #expect(quantized.metaState == ["256", "7", "128", "4"])
}

// MARK: - ArraysCache sparse slot round-trip

@Test func testArraysCacheSparseSlots() throws {
    let cache = ArraysCache(size: 3)
    let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
    cache[0] = a
    // slot 1 stays nil
    cache[2] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.slotCount == 3)
    #expect(restored[0] != nil)
    #expect(restored[1] == nil)
    #expect(restored[2] != nil)
    #expect(allClose(restored[0]!, a).item(Bool.self))
    #expect(allClose(restored[2]!, b).item(Bool.self))
}

// MARK: - ArraysCache leftPadding round-trip

@Test func testArraysCacheLeftPadding() throws {
    let cache = ArraysCache(size: 2, leftPadding: [0, 5])
    let a = MLXArray.ones([2, 4], dtype: .float32)
    let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.leftPaddingValues == [0, 5])
    assertArraysClose(restored.state, cache.state)
}

@Test func testArraysCacheMaskUsesLeftPaddingAfterStateUpdate() throws {
    let cache = ArraysCache(size: 2, leftPadding: [1, 3])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    let mask = try #require(cache.makeMask(N: 4))
    #expect(
        mask.asArray(Bool.self) == [
            false, true, true, true,
            false, false, false, true,
        ])
}

@Test func testArraysCacheAdvanceUpdatesSequenceMetadataOnly() throws {
    let cache = ArraysCache(size: 2, leftPadding: [3, 5])
    cache.offset = 7
    cache.prepare(lengths: [4, 6])

    cache.advance(2)

    #expect(cache.offset == 7)
    #expect(cache.leftPaddingValues == [1, 3])
    #expect(cache.lengthsValues == [2, 4])
}

@Test func testArraysCacheMaskUsesLengthsWhenLeftPaddingIsAbsent() throws {
    let cache = ArraysCache(size: 2)
    cache.prepare(lengths: [1, 3])

    let mask = try #require(cache.makeMask(N: 4))
    #expect(
        mask.asArray(Bool.self) == [
            true, false, false, false,
            true, true, true, false,
        ])
}

@Test func testTextSequenceLengthsComeFromAttentionMask() throws {
    let tokens = MLXArray(0 ..< 8).reshaped(2, 4)
    let mask = MLXArray([1, 1, 0, 0, 1, 1, 1, 0]).reshaped(2, 4)
    let text = LMInput.Text(tokens: tokens, mask: mask)

    #expect(text.sequenceLengths == [2, 3])
}

@Test func testTextSequenceLengthsInferUniformBatches() throws {
    let text = LMInput.Text(tokens: MLXArray(0 ..< 8).reshaped(2, 4))

    #expect(text.sequenceLengths == [4, 4])
}

@Test func testCacheListForwardsPrepareAndFinalize() throws {
    let arrays = ArraysCache(size: 2)
    let cache = CacheList(arrays, KVCacheSimple())

    cache.prepare(lengths: [2, 4])
    #expect(arrays.lengthsValues == [2, 4])

    cache.finalize()
    #expect(arrays.lengthsValues == nil)
}

@Test func testCacheListForwardsLifecycleThroughKVCacheProtocol() throws {
    let lifecycle = LifecycleRecordingCache()
    let cache = CacheList(KVCacheSimple(), lifecycle)

    cache.prepare(lengths: [2, 4])
    #expect(lifecycle.preparedLengths == [2, 4])

    cache.finalize()
    #expect(lifecycle.finalizeCallCount == 1)
}

@Test func testWithPreparedCacheScopesSequenceMetadata() throws {
    let cache = ArraysCache(size: 2)

    withPreparedCache([cache], lengths: [2, 4]) {
        #expect(cache.lengthsValues == [2, 4])
    }

    #expect(cache.lengthsValues == nil)
}

@Test func testArraysCacheLengthsRoundTrip() throws {
    let cache = ArraysCache(size: 2)
    cache.prepare(lengths: [4, 2])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.currentLengths?.asArray(Int.self) == [4, 2])
    #expect(restored.lengthsValues == [4, 2])
    assertArraysClose(restored.state, cache.state)
}

@Test func testArraysCacheAdvanceUpdatesLengthsAndLeftPaddingMasks() throws {
    let cache = ArraysCache(size: 2, leftPadding: [1, 3])
    cache.prepare(lengths: [4, 2])
    cache.advance(2)

    #expect(cache.leftPaddingValues == [-1, 1])
    #expect(cache.currentLengths?.asArray(Int.self) == [2, 0])

    let mask = try #require(cache.makeMask(N: 3))
    #expect(mask.asArray(Bool.self) == [true, true, true, false, true, true])

    let lengthOnly = ArraysCache(size: 1)
    lengthOnly.prepare(lengths: [2, 0])
    let lengthMask = try #require(lengthOnly.makeMask(N: 3))
    #expect(lengthMask.asArray(Bool.self) == [true, true, false, false, false, false])

    cache.finalize()
    #expect(cache.leftPaddingValues == nil)
    #expect(cache.currentLengths == nil)
}

@Test func testArraysCacheFilterAndExtendPreserveBatchMetadata() throws {
    let first = ArraysCache(size: 1, leftPadding: [0, 2])
    first.prepare(lengths: [5, 3])
    first[0] = MLXArray.ones([2, 2], dtype: .float32)

    first.filter(batchIndices: MLXArray([1]))
    #expect(first.leftPaddingValues == [2])
    #expect(first.currentLengths?.asArray(Int.self) == [3])
    #expect(first[0]?.shape == [1, 2])

    let second = ArraysCache(size: 1, leftPadding: [1, 4])
    second.prepare(lengths: [6, 2])
    second[0] = MLXArray.ones([2, 2], dtype: .float32) * 2

    first.extend(other: second)
    #expect(first.leftPaddingValues == [2, 1, 4])
    #expect(first.currentLengths?.asArray(Int.self) == [3, 6, 2])
    #expect(first[0]?.shape == [3, 2])
}

@Test func testAttentionMaskUsesSharedCausalCachePath() throws {
    let cache = KVCacheSimple()
    let prefillInput = MLXArray.ones([1, 3, 8], dtype: .float32)

    let prefillMask = createAttentionMask(h: prefillInput, cache: cache)
    if case .causal = prefillMask {
        // Expected for multi-token prefill: Falcon H1 uses the shared symbolic causal mask path.
    } else {
        Issue.record("Expected symbolic causal attention mask for prefill")
    }

    let tokenInput = MLXArray.ones([1, 1, 8], dtype: .float32)
    let tokenMask = createAttentionMask(h: tokenInput, cache: cache)
    if case .none = tokenMask {
        // Expected for one-token decode: no materialized mask is needed.
    } else {
        Issue.record("Expected no attention mask for one-token decode")
    }

    cache.offset = 2
    let forcedMask = createAttentionMask(h: prefillInput, cache: cache, returnArray: true)
    guard case .array(let mask) = forcedMask else {
        Issue.record("Expected forced attention mask array")
        return
    }
    #expect(mask.shape == [3, 5])
    #expect(
        mask.asArray(Bool.self) == [
            true, true, true, false, false,
            true, true, true, true, false,
            true, true, true, true, true,
        ])
}

@Test func testSSMMaskUsesSharedMambaMetadataPath() throws {
    let leftPadded = MambaCache(leftPadding: [1, 3])
    let input = MLXArray.ones([2, 4, 8], dtype: .float32)

    let leftPaddingMask = try #require(createSSMMask(h: input, cache: leftPadded))
    #expect(
        leftPaddingMask.asArray(Bool.self) == [
            false, true, true, true,
            false, false, false, true,
        ])

    let lengthMasked = MambaCache()
    lengthMasked.prepare(lengths: [3, 1])
    let lengthsMask = try #require(createSSMMask(h: input, cache: lengthMasked))
    #expect(
        lengthsMask.asArray(Bool.self) == [
            true, true, true, false,
            true, false, false, false,
        ])
}

@Test func testCacheListPrepareFinalizePropagatesThroughNestedHybridCaches() throws {
    let mamba = MambaCache(leftPadding: [0, 2])
    let arrays = ArraysCache(size: 1)
    let nested = CacheList(CacheList(mamba), arrays)

    nested.prepare(lengths: [4, 1])

    #expect(mamba.currentLengths?.asArray(Int.self) == [4, 1])
    #expect(arrays.currentLengths?.asArray(Int.self) == [4, 1])

    nested.finalize()

    #expect(mamba.currentLengths == nil)
    #expect(mamba.leftPaddingValues == nil)
    #expect(arrays.currentLengths == nil)
}

@Test func testMambaCacheCopyPreservesBatchMaskMetadata() throws {
    let cache = MambaCache(leftPadding: [2, 0])
    cache.prepare(lengths: [5, 3])
    cache[0] = MLXArray.ones([2, 3, 4], dtype: .float32)
    cache[1] = MLXArray.ones([2, 1, 4, 4], dtype: .float32)

    let copied = try #require(cache.copy() as? MambaCache)

    #expect(copied.leftPaddingValues == [2, 0])
    #expect(copied.currentLengths?.asArray(Int.self) == [5, 3])
    #expect(copied[0]?.shape == [2, 3, 4])
    #expect(copied[1]?.shape == [2, 1, 4, 4])
}

@Test func testArraysCacheFilterKeepsSequenceMetadata() throws {
    let cache = ArraysCache(size: 2, leftPadding: [1, 3])
    cache.prepare(lengths: [2, 4])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    cache.filter(batchIndices: MLXArray([1]))

    #expect(cache.leftPaddingValues == [3])
    #expect(cache.lengthsValues == [4])
}

@Test func testArraysCacheExtendPadsMissingSlotsAndMetadata() throws {
    let first = ArraysCache(size: 2, leftPadding: [1, 3])
    first.prepare(lengths: [2, 4])
    first[0] = MLXArray.ones([2, 4], dtype: .float32)

    let second = ArraysCache(size: 2)
    second[1] = MLXArray.ones([1, 4], dtype: .float32) * 2

    first.extend(other: second)

    #expect(first[0]?.shape == [3, 4])
    #expect(first[1]?.shape == [3, 4])
    #expect(first.leftPaddingValues == [1, 3, 0])
    #expect(first.lengthsValues == [2, 4, 0])
}

@Test func testArraysCacheCopyPreservesSparseSlotsAndMetadata() throws {
    let cache = ArraysCache(size: 3, leftPadding: [2])
    cache.prepare(lengths: [5])
    cache[2] = MLXArray.ones([1, 4], dtype: .float32)

    let copied = try #require(cache.copy() as? ArraysCache)

    #expect(copied.slotCount == 3)
    #expect(copied[0] == nil)
    #expect(copied[1] == nil)
    #expect(copied[2] != nil)
    #expect(copied.leftPaddingValues == [2])
    #expect(copied.lengthsValues == [5])
}

// MARK: - MambaCache type preservation

@Test func testMambaCacheRoundTrip() throws {
    let cache = MambaCache()
    let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? MambaCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

// MARK: - CacheList with KV caches

@Test func testCacheListKVCaches() throws {
    let simple = KVCacheSimple()
    let rotating = RotatingKVCache(maxSize: 32)

    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)
    _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

    let cacheList = CacheList(simple, rotating)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cacheList], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? CacheList)
    let child0 = try #require(restored[0] as? KVCacheSimple)
    let child1 = try #require(restored[1] as? RotatingKVCache)

    assertArraysClose(child0.state, simple.state, label: "child0")
    assertArraysClose(child1.state, rotating.state, label: "child1")
    #expect(child1.metaState == rotating.metaState)
}

// MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

@Test func testCacheListHybrid() throws {
    let mamba = MambaCache()
    mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let simple = KVCacheSimple()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)

    let cacheList = CacheList(mamba, simple)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cacheList], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? CacheList)
    let restoredMamba = try #require(restored[0] as? MambaCache)
    let restoredSimple = try #require(restored[1] as? KVCacheSimple)

    assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
    assertArraysClose(restoredSimple.state, simple.state, label: "simple")
}

// MARK: - Simple cache round-trip with value assertions

@Test func testSimpleCacheRoundTrip() throws {
    let cache = KVCacheSimple()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: values)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)
    #expect(loaded.count == 1)
    assertArraysClose(loaded[0].state, cache.state)
}

// MARK: - ArraysCache fully populated round-trip

@Test func testArraysCacheFullyPopulated() throws {
    let cache = ArraysCache(size: 2)
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)
    cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

// MARK: - Quantized attention causal masking
// Regression for the finfo.min mistranslation (masked fill was Float.leastNormalMagnitude ≈ 0).

@Test
func testQuantizedAttentionCausalMaskMatchesFullPrecision() throws {
    MLXRandom.seed(0)

    let (B, nKVHeads, L, D) = (1, 2, 4, 64)
    let scale = 1.0 / Float(D).squareRoot()

    // nRepeats 1 = MHA, 2 = GQA (exercises the 5-D reshape + .causal path that silently corrupts).
    for nRepeats in [1, 2] {
        let nQHeads = nKVHeads * nRepeats
        let q = MLXRandom.normal([B, nQHeads, L, D])
        let k = MLXRandom.normal([B, nKVHeads, L, D])
        let v = MLXRandom.normal([B, nKVHeads, L, D])

        // Reference: full-precision causal attention.
        let reference = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)

        // Path under test: quantized cache + .causal.
        let cache = QuantizedKVCache(groupSize: 64, bits: 8)
        let (qK, qV) = cache.updateQuantized(keys: k, values: v)
        let out = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .causal,
            groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

        #expect(out.shape == reference.shape)
        // 8-bit quant error is << 0.1; the bug diverges by O(1).
        let close = allClose(out, reference, rtol: 0.05, atol: 0.1).item(Bool.self)
        #expect(
            close, "quantized causal attention diverges from full precision (nRepeats=\(nRepeats))")
    }
}

@Test("quantizedScaledDotProductAttention preserves the score dtype")
func preservesScoreDtype() {
    MLXRandom.seed(0)
    let (B, H, L, D) = (1, 2, 4, 64)
    let scale = 1.0 / Float(D).squareRoot()

    // f32 passes even with a mis-typed fill; f16/bf16 are exactly what a
    // float32 fill silently promotes — so assert the output keeps its dtype.
    for dtype in [DType.float16, .bfloat16, .float32] {
        let q = MLXRandom.normal([B, H, L, D]).asType(dtype)
        let k = MLXRandom.normal([B, H, L, D]).asType(dtype)
        let v = MLXRandom.normal([B, H, L, D]).asType(dtype)

        let cache = QuantizedKVCache(groupSize: 64, bits: 8)
        let (qK, qV) = cache.updateQuantized(keys: k, values: v)
        let out = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qK, quantizedValues: qV,
            scale: scale, mask: .causal,
            groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

        #expect(out.dtype == dtype, "output promoted to \(out.dtype) for input \(dtype)")
        #expect(out.asType(.float32).sum().item(Float.self).isFinite)  // no -inf → NaN
    }
}

// MARK: - ropeOffset overridability

/// A `BaseKVCache` subclass reporting a per-row RoPE offset, as a batched cache does.
private final class BatchOffsetProbeCache: BaseKVCache {
    override var ropeOffset: RoPEOffset { .batch(MLXArray([10, 20])) }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }
}

/// Models read `ropeOffset` through a `KVCache` reference, so a subclass override has to
/// survive that dispatch. When `BaseKVCache` did not declare `ropeOffset`, the extension
/// default was the witness and this resolved to `.scalar(0)`, silently ignoring the subclass.
@Test func testSubclassRopeOffsetOverrideIsHonoredThroughKVCacheReference() {
    let cache: any KVCache = BatchOffsetProbeCache()

    guard case .batch(let offsets) = cache.ropeOffset else {
        Issue.record(
            "subclass ropeOffset override ignored — resolved to the .scalar extension default")
        return
    }
    #expect(offsets.asArray(Int32.self) == [10, 20])
}

// MARK: - Corrupt prompt cache recovery (throws, not fatalError)

/// Writes a hand-crafted "prompt cache" file that claims to hold a `KVCacheSimple`
/// but only has one array where the format requires exactly two (keys, values),
/// simulating a truncated/corrupted cache file.
private func writeCorruptSimpleCacheFile(className: String, arrayCount: Int, metaState: [String])
    throws -> URL
{
    let url = tempURL()
    var arrays: [String: MLXArray] = [:]
    for i in 0 ..< arrayCount {
        arrays["0.\(i)"] = MLXArray.zeros([1, 1, 1, 1], dtype: .float32)
    }
    var metadata: [String: String] = [:]
    for (j, value) in metaState.enumerated() {
        metadata["0.0.\(j)"] = value
    }
    metadata["2.0"] = className
    try save(arrays: arrays, metadata: metadata, url: url)
    return url
}

@Test func testLoadPromptCacheThrowsOnCorruptKVCacheSimpleStateInsteadOfCrashing() throws {
    // KVCacheSimple requires exactly 2 arrays; provide only 1.
    let url = try writeCorruptSimpleCacheFile(
        className: "KVCacheSimple", arrayCount: 1, metaState: [""])

    #expect(throws: KVCacheError.self) {
        _ = try loadPromptCache(url: url)
    }
}

@Test func testLoadPromptCacheThrowsOnCorruptQuantizedKVCacheMetaStateInsteadOfCrashing() throws {
    // QuantizedKVCache requires exactly 4 metaState values; provide only 2.
    let url = try writeCorruptSimpleCacheFile(
        className: "QuantizedKVCache", arrayCount: 4, metaState: ["256", "0"])

    #expect(throws: KVCacheError.self) {
        _ = try loadPromptCache(url: url)
    }
}

@Test func testLoadPromptCacheThrowsOnCorruptChunkedKVCacheMetaStateInsteadOfCrashing() throws {
    // ChunkedKVCache requires exactly 2 metaState values; provide only 1.
    let url = try writeCorruptSimpleCacheFile(
        className: "ChunkedKVCache", arrayCount: 2, metaState: ["None"])

    #expect(throws: KVCacheError.self) {
        _ = try loadPromptCache(url: url)
    }
}

@Test func testRotatingKVCacheToQuantizedThrowsInsteadOfCrashing() throws {
    let cache = RotatingKVCache(maxSize: 32)

    #expect(throws: KVCacheError.self) {
        _ = try cache.toQuantized()
    }
}

@Test func testKVCacheSimpleToQuantizedThrowsOnIncompatibleHeadDim() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.zeros([1, 2, 4, 5], dtype: .float32),
        values: MLXArray.zeros([1, 2, 4, 5], dtype: .float32)
    )

    // Head dim 5 is not divisible by any of the supported group sizes (32, 64, 128).
    #expect(throws: KVCacheError.self) {
        _ = try cache.toQuantized(groupSize: 64, bits: 4)
    }
}
