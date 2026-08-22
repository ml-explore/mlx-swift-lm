import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { FixedCapacityKVCache(maxTokens: 64) },
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

private struct SerializedCacheFixture {
    let className: String
    let arrayCount: Int
    let metadata: [String]
    var arrayShape = [1, 1, 1, 1]
}

private func writePromptCacheFixture(_ fixture: SerializedCacheFixture) throws -> URL {
    let url = tempURL()
    let arrays: [String: MLXArray] = Dictionary(
        uniqueKeysWithValues: (0 ..< fixture.arrayCount).map { index in
            (
                "0.\(index)",
                MLXArray.zeros(fixture.arrayShape, dtype: .float32, stream: .cpu)
            )
        })
    var metadata: [String: String] = Dictionary(
        uniqueKeysWithValues: fixture.metadata.enumerated().map {
            ("0.0.\($0.offset)", $0.element)
        })
    metadata["2.0"] = fixture.className
    try save(arrays: arrays, metadata: metadata, url: url, stream: .cpu)
    return url
}

private func expectPromptCacheLoadToFail(_ fixtures: [SerializedCacheFixture]) throws {
    for fixture in fixtures {
        let url = try writePromptCacheFixture(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            _ = try loadPromptCache(url: url)
        }
    }
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

/// A direct protocol conformer that deliberately relies on the
/// `KVCache.isTrimmable(after:)` extension default.
private final class ProtocolDefaultTrimmabilityCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let copy = ProtocolDefaultTrimmabilityCache()
        copy.offset = offset
        return copy
    }

    func innerState() -> [MLXArray] { [] }
}

// MARK: - Predictive trimmability

@Test func testRotatingKVCachePredictsStrictTrimBoundary() {
    let cache = RotatingKVCache(maxSize: 8)
    cache.offset = 3

    #expect(cache.isTrimmable(after: 4))
    #expect(!cache.isTrimmable(after: 5))
}

@Test func testRotatingKVCacheIsNotTrimmableAtOrPastWindow() {
    let cache = RotatingKVCache(maxSize: 8)

    cache.offset = 8
    #expect(!cache.isTrimmable)
    #expect(!cache.isTrimmable(after: 0))

    cache.offset = 9
    #expect(!cache.isTrimmable)
    #expect(!cache.isTrimmable(after: 0))
}

@Test func testRotatingPredictiveTrimOverrideDispatchesThroughKVCache() {
    let rotating = RotatingKVCache(maxSize: 8)
    rotating.offset = 3
    let cache: any KVCache = rotating

    #expect(cache.isTrimmable(after: 4))
    #expect(!cache.isTrimmable(after: 5))
}

@Test func testUnboundedCacheRemainsTrimmableAfterPositiveLookAhead() {
    let cache: any KVCache = KVCacheSimple()

    #expect(cache.isTrimmable(after: 1_000))
}

@Test func testCacheListDelegatesPredictiveTrimmabilityToChildren() {
    let rotating = RotatingKVCache(maxSize: 8)
    rotating.offset = 3
    let cache: any KVCache = CacheList(KVCacheSimple(), rotating)

    #expect(cache.isTrimmable(after: 4))
    #expect(!cache.isTrimmable(after: 5))
}

@Test func testDirectKVCacheConformerUsesPredictiveTrimDefault() {
    let cache: any KVCache = ProtocolDefaultTrimmabilityCache()

    #expect(cache.isTrimmable(after: 1_000))
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

@Test func testFixedCapacityKVCacheAsymmetricStateRoundTrip() throws {
    let cache = FixedCapacityKVCache(maxTokens: 8)
    cache.state = [
        MLXArray.ones([1, 2, 3, 6], dtype: .float16),
        MLXArray.ones([1, 2, 3, 4], dtype: .float16) * 2,
    ]
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache])
    let (loaded, _) = try loadPromptCache(url: url)
    let restored = try #require(loaded.first as? FixedCapacityKVCache)

    #expect(restored.maxTokens == 8)
    #expect(restored.innerState()[0].shape == [1, 2, 8, 6])
    #expect(restored.innerState()[1].shape == [1, 2, 8, 4])
    assertArraysClose(restored.state, cache.state)
}

@Test func testPromptCacheStateRoundTrip() throws {
    let cache = KVCacheSimple()
    let keys = MLXArray.ones([1, 2, 4, 8], dtype: .bfloat16)
    let values = MLXArray.zeros([1, 2, 4, 8], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: values)

    let ropeDeltasKey = LMOutput.Key<MLXArray>("test.ropeDeltas")
    let positionsKey = LMOutput.Key<MLXArray>("test.positionIds")
    let ropeDeltas = MLXArray([Int32(3), 5, 7]).reshaped([1, 3])
    let positions = MLXArray([Int32(11), 13, 17, 19]).reshaped([2, 2])
    var state = LMOutput.State()
    state[ropeDeltasKey] = ropeDeltas
    state[positionsKey] = positions

    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }
    try savePromptCache(
        url: url, cache: [cache], metadata: ["source": "test"], state: state)

    let snapshot = try loadPromptCacheSnapshot(url: url)
    let restoredRopeDeltas = try #require(snapshot.state?[ropeDeltasKey])
    let restoredPositions = try #require(snapshot.state?[positionsKey])

    #expect(snapshot.metadata == ["source": "test"])
    #expect(restoredRopeDeltas.dtype == ropeDeltas.dtype)
    #expect(restoredRopeDeltas.shape == ropeDeltas.shape)
    #expect(allClose(restoredRopeDeltas, ropeDeltas, rtol: 0, atol: 0).item(Bool.self))
    #expect(restoredPositions.dtype == positions.dtype)
    #expect(restoredPositions.shape == positions.shape)
    #expect(allClose(restoredPositions, positions, rtol: 0, atol: 0).item(Bool.self))

    let (storedArrays, storedMetadata) = try loadArraysAndMetadata(url: url)
    #expect(storedArrays["__mlx_lm_state_tensor_0"] != nil)
    #expect(storedArrays["__mlx_lm_state_tensor_1"] != nil)
    #expect(
        storedArrays.keys.allSatisfy {
            Int($0.split(separator: ".")[0]) != nil
                || $0.hasPrefix("__mlx_lm_state_tensor_")
        })
    #expect(storedMetadata["1.__mlx_lm_state_0_key"] == "test.positionIds")
    #expect(storedMetadata["1.__mlx_lm_state_1_key"] == "test.ropeDeltas")
    #expect(storedMetadata["2.0"] == "__mlx_lm_state_v1__:KVCache")
    #expect(throws: KVCacheError.self) {
        try loadPromptCache(url: url)
    }
}

@Test func testLegacyPromptCacheLoadsWithNilState() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache])
    let snapshot = try loadPromptCacheSnapshot(url: url)
    let (legacyCache, legacyMetadata) = try loadPromptCache(url: url)
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)

    #expect(snapshot.cache.count == 1)
    #expect(snapshot.state == nil)
    #expect(legacyCache.count == 1)
    #expect(legacyMetadata.isEmpty)
    #expect(arrays.keys.allSatisfy { Int($0.split(separator: ".")[0]) != nil })
    #expect(!metadata.keys.contains { $0.contains("__mlx_lm_state_") })
    #expect(metadata["2.0"] == "KVCache")
}

@Test func testEmptyPromptCacheStateUsesTheLegacyFormat() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let legacyURL = tempURL()
    let emptyStateURL = tempURL()
    defer {
        try? FileManager.default.removeItem(at: legacyURL)
        try? FileManager.default.removeItem(at: emptyStateURL)
    }

    try savePromptCache(url: legacyURL, cache: [cache])
    try savePromptCache(url: emptyStateURL, cache: [cache], state: LMOutput.State())
    let snapshot = try loadPromptCacheSnapshot(url: emptyStateURL)
    let (legacyCache, _) = try loadPromptCache(url: emptyStateURL)
    let (legacyArrays, legacyMetadata) = try loadArraysAndMetadata(url: legacyURL)
    let (emptyStateArrays, emptyStateMetadata) = try loadArraysAndMetadata(url: emptyStateURL)

    #expect(snapshot.state == nil)
    #expect(legacyCache.count == 1)
    #expect(Set(emptyStateArrays.keys) == Set(legacyArrays.keys))
    #expect(emptyStateMetadata == legacyMetadata)
}

@Test func testPromptCacheStateRejectsReservedUserMetadata() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    #expect(throws: KVCacheError.self) {
        try savePromptCache(
            url: url, cache: [cache],
            metadata: ["__mlx_lm_state_version": "caller-owned"])
    }
}

@Test func testPromptCacheStateRejectsUnsupportedValues() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let unsupportedKey = LMOutput.Key<Bool>("test.unsupported")
    var state = LMOutput.State()
    state[unsupportedKey] = true
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    #expect(throws: LMOutput.State.SerializationError.self) {
        try savePromptCache(url: url, cache: [cache], state: state)
    }
    #expect(!FileManager.default.fileExists(atPath: url.path))
}

@Test func testPromptCacheStateRejectsMalformedEntries() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let key = LMOutput.Key<MLXArray>("test.state")
    var state = LMOutput.State()
    state[key] = MLXArray([Int32(1)])
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache], state: state)
    let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
    var malformedMetadata = savedMetadata
    malformedMetadata["1.__mlx_lm_state_count"] = "2"
    try save(arrays: arrays, metadata: malformedMetadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
    }
}

@Test func testPromptCacheStateRejectsUnknownVersions() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let key = LMOutput.Key<MLXArray>("test.state")
    var state = LMOutput.State()
    state[key] = MLXArray([Int32(1)])
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache], state: state)
    let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
    var malformedMetadata = savedMetadata
    malformedMetadata["1.__mlx_lm_state_version"] = "999"
    try save(arrays: arrays, metadata: malformedMetadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
    }
}

@Test func testPromptCacheStateRejectsDuplicateKeys() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let firstKey = LMOutput.Key<MLXArray>("test.first")
    let secondKey = LMOutput.Key<MLXArray>("test.second")
    var state = LMOutput.State()
    state[firstKey] = MLXArray([Int32(1)])
    state[secondKey] = MLXArray([Int32(2)])
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache], state: state)
    let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
    var malformedMetadata = savedMetadata
    malformedMetadata["1.__mlx_lm_state_1_key"] = "test.first"
    try save(arrays: arrays, metadata: malformedMetadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
    }
}

@Test func testPromptCacheStateRejectsUnexpectedTensors() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let key = LMOutput.Key<MLXArray>("test.state")
    var state = LMOutput.State()
    state[key] = MLXArray([Int32(1)])
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache], state: state)
    let (storedArrays, metadata) = try loadArraysAndMetadata(url: url)
    var malformedArrays = storedArrays
    malformedArrays["__mlx_lm_state_tensor_99"] = MLXArray([Int32(99)])
    try save(arrays: malformedArrays, metadata: metadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
    }
}

@Test func testPromptCacheStateRequiresCompatibilityMarker() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let key = LMOutput.Key<MLXArray>("test.state")
    var state = LMOutput.State()
    state[key] = MLXArray([Int32(1)])
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache], state: state)
    let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
    var malformedMetadata = storedMetadata
    malformedMetadata["2.0"] = "KVCache"
    try save(arrays: arrays, metadata: malformedMetadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
    }
}

@Test func testPromptCacheCompatibilityMarkerRequiresState() throws {
    let cache = KVCacheSimple()
    _ = cache.update(
        keys: MLXArray.ones([1, 1, 1, 4]),
        values: MLXArray.zeros([1, 1, 1, 4]))
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: [cache])
    let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
    var malformedMetadata = storedMetadata
    malformedMetadata["2.0"] = "__mlx_lm_state_v1__:KVCache"
    try save(arrays: arrays, metadata: malformedMetadata, url: url)

    #expect(throws: KVCacheError.self) {
        try loadPromptCacheSnapshot(url: url)
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

@Test func testDirectKVCacheQuantizationFailuresAreRecoverable() throws {
    let incompatible = KVCacheSimple()
    incompatible.state = [
        MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
        MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
    ]

    #expect(throws: KVCacheError.self) {
        _ = try incompatible.toQuantized(groupSize: 64, bits: 4)
    }
    #expect(throws: KVCacheError.self) {
        _ = try RotatingKVCache(maxSize: 32).toQuantized()
    }
}

@Test func testPromptCacheRestorationRejectsIncompleteState() throws {
    try expectPromptCacheLoadToFail([
        .init(className: "KVCacheSimple", arrayCount: 1, metadata: [""]),
        .init(
            className: "RotatingKVCache", arrayCount: 1,
            metadata: ["0", "32", "256", "0", "0", "modelNative"]),
        .init(
            className: "QuantizedKVCache", arrayCount: 3,
            metadata: ["256", "0", "64", "4"]),
        .init(className: "ChunkedKVCache", arrayCount: 1, metadata: ["None", "0"]),
    ])
}

@Test func testPromptCacheRestorationRejectsInvalidMetadata() throws {
    try expectPromptCacheLoadToFail([
        .init(className: "KVCacheSimple", arrayCount: 0, metadata: ["unexpected"]),
        .init(
            className: "RotatingKVCache", arrayCount: 0,
            metadata: ["0", "32", "invalid", "0", "0", "modelNative"]),
        .init(
            className: "RotatingKVCache", arrayCount: 0,
            metadata: ["0", "32", "256", "0", "0", "invalid-origin"]),
        .init(
            className: "QuantizedKVCache", arrayCount: 0,
            metadata: ["256", "invalid", "64", "4"]),
        .init(className: "ChunkedKVCache", arrayCount: 0, metadata: ["invalid", "0"]),
        .init(className: "FixedCapacityKVCache", arrayCount: 0, metadata: ["0"]),
        .init(
            className: "FixedCapacityKVCache", arrayCount: 2, metadata: ["1"],
            arrayShape: [1, 1, 2, 4]),
    ])
}

@Test func testPromptCacheRestorationRejectsInvalidStateRank() throws {
    try expectPromptCacheLoadToFail([
        .init(
            className: "KVCacheSimple", arrayCount: 2, metadata: [""],
            arrayShape: [1, 1])
    ])
}

@Test func testPromptCacheRestorationAcceptsLegacyRotatingMetadata() throws {
    let fixture = SerializedCacheFixture(
        className: "RotatingKVCache",
        arrayCount: 2,
        metadata: ["0", "32", "256", "1", "1"])
    let url = try writePromptCacheFixture(fixture)
    defer { try? FileManager.default.removeItem(at: url) }

    let (restored, _) = try loadPromptCache(url: url)
    let rotating = try #require(restored.first as? RotatingKVCache)

    #expect(rotating.capacityOrigin == .modelNative)
    #expect(rotating.metaState == ["0", "32", "256", "1", "1", "modelNative"])
}

@Test func testPromptCacheRoundTripPreservesEmptyCaches() throws {
    let populated = KVCacheSimple()
    populated.state = [
        MLXArray.ones([1, 1, 1, 32], stream: .cpu),
        MLXArray.ones([1, 1, 1, 32], stream: .cpu),
    ]
    let caches: [KVCache] = [
        KVCacheSimple(),
        populated,
        RotatingKVCache(maxSize: 32),
        QuantizedKVCache(),
        ChunkedKVCache(chunkSize: 16),
    ]
    let url = tempURL()
    defer { try? FileManager.default.removeItem(at: url) }

    try savePromptCache(url: url, cache: caches)
    let (restored, _) = try loadPromptCache(url: url)

    #expect(restored.count == caches.count)
    #expect(restored[0] is KVCacheSimple)
    #expect(restored[0].state.isEmpty)
    #expect(restored[1].state.count == 2)
    #expect(restored[2] is RotatingKVCache)
    #expect(restored[2].state.isEmpty)
    #expect(restored[3] is QuantizedKVCache)
    #expect(restored[3].state.isEmpty)
    #expect(restored[4] is ChunkedKVCache)
    #expect(restored[4].state.isEmpty)
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

// MARK: - RotatingKVCache.logicalView
//
// `logicalView(tail:)` is the read-only counterpart to `update(keys:values:)`: it exposes the
// ring's contents in chronological order without writing. Speculative decoding needs it to
// present committed history alongside K/V it has not committed yet.
//
// Nothing in this repo drove a `RotatingKVCache` ring past its wrap before these tests, so the
// rotation in `updateInPlace`, the linearization in `temporalOrder`, and the windowed mask that
// `makeMask` builds over the multi-token presentation were all uncovered. They are the
// foundation the view rests on, so they are pinned here too.

/// K/V whose every element encodes its own sequence position: keys hold `+p`, values `-p`.
/// Any reordering, duplication, or K/V mix-up changes the numbers.
private func positionedKV(
    _ positions: Range<Int>, headDim: Int = 2
) -> (MLXArray, MLXArray) {
    let count = positions.count
    let keys = MLXArray(
        positions.flatMap { Array(repeating: Float($0), count: headDim) },
        [1, 1, count, headDim])
    let values = MLXArray(
        positions.flatMap { Array(repeating: Float(-$0), count: headDim) },
        [1, 1, count, headDim])
    return (keys, values)
}

/// Recover the sequence positions encoded by `positionedKV` from a `[B, H, S, D]` key array.
private func encodedPositions(_ keys: MLXArray) -> [Int] {
    keys[0, 0, 0..., 0].asArray(Float.self).map { Int($0) }
}

/// Drive `cache` through `count` single-token writes, so the ring actually rotates.
private func fillOneAtATime(_ cache: RotatingKVCache, positions: Range<Int>) {
    for p in positions {
        let (k, v) = positionedKV(p ..< (p + 1))
        _ = cache.update(keys: k, values: v)
    }
}

@Test func testRotatingLogicalViewIsNilBeforeFirstWrite() {
    #expect(RotatingKVCache(maxSize: 8, keep: 0).logicalView(tail: 8) == nil)
}

@Test func testRotatingLogicalViewReturnsChronologicalTailPastWrap() throws {
    let cache = RotatingKVCache(maxSize: 8, keep: 0)
    fillOneAtATime(cache, positions: 0 ..< 20)

    // The ring has wrapped twice; physical order is rotated, chronological order is not.
    #expect(cache.offset == 20)

    let full = try #require(cache.logicalView(tail: 8))
    #expect(
        encodedPositions(full.0) == Array(12 ..< 20),
        "logicalView did not linearize the rotated ring")
    #expect(encodedPositions(full.1).map { -$0 } == Array(12 ..< 20), "values diverged from keys")

    let short = try #require(cache.logicalView(tail: 3))
    #expect(encodedPositions(short.0) == [17, 18, 19], "a short tail took the wrong end")
}

@Test func testRotatingLogicalViewClampsTailToWhatTheCacheHolds() throws {
    let cache = RotatingKVCache(maxSize: 8, keep: 0)
    fillOneAtATime(cache, positions: 0 ..< 5)

    #expect(encodedPositions(try #require(cache.logicalView(tail: 99)).0) == Array(0 ..< 5))
    #expect(encodedPositions(try #require(cache.logicalView(tail: 5)).0) == Array(0 ..< 5))
    #expect(try #require(cache.logicalView(tail: 0)).0.dim(2) == 0)
    #expect(try #require(cache.logicalView(tail: -1)).0.dim(2) == 0, "negative tail must clamp")
}

@Test func testRotatingLogicalViewLeavesTheRingUntouched() {
    let cache = RotatingKVCache(maxSize: 8, keep: 0)
    fillOneAtATime(cache, positions: 0 ..< 20)

    let beforeState = cache.innerState().map { MLXArray($0.asArray(Float.self), $0.shape) }
    let beforeOffset = cache.offset
    let beforeMeta = cache.metaState

    _ = cache.logicalView(tail: 7)
    _ = cache.logicalView(tail: 3)

    let afterState = cache.innerState()
    #expect(afterState.count == beforeState.count)
    for (index, (before, after)) in zip(beforeState, afterState).enumerated() {
        #expect(before.shape == after.shape, "innerState[\(index)] was reshaped by a read")
        #expect(
            allClose(before, after, rtol: 0, atol: 0).item(Bool.self),
            "innerState[\(index)] was mutated by a read")
    }
    #expect(cache.offset == beforeOffset, "logicalView moved the offset")
    #expect(cache.metaState == beforeMeta, "logicalView disturbed idx/offset bookkeeping")
}

@Test func testRotatingLogicalViewPreservesPinnedPrefix() throws {
    // `keep` pins the oldest entries; the write path front-trims *after* them, so a view must
    // splice around the prefix rather than take a flat tail.
    let cache = RotatingKVCache(maxSize: 8, keep: 2)
    fillOneAtATime(cache, positions: 0 ..< 20)

    let view = try #require(cache.logicalView(tail: 5))
    let positions = encodedPositions(view.0)
    #expect(positions.count == 5)
    #expect(Array(positions.prefix(2)) == [0, 1], "pinned prefix was evicted by the view")
    #expect(positions == [0, 1] + Array(positions.suffix(3)))
    #expect(positions.suffix(3).sorted() == Array(positions.suffix(3)), "tail out of order")
}

@Test func testRotatingLogicalViewFloorsAtThePinnedPrefix() throws {
    // A pinned entry is never evictable, so `keep` is a floor on the view and not just a
    // splice point: a `tail` below it still returns the prefix. The alternative -- honouring
    // `tail` exactly -- would hand back a context the ring itself can never present.
    let cache = RotatingKVCache(maxSize: 8, keep: 2)
    fillOneAtATime(cache, positions: 0 ..< 20)

    for tail in 0 ... 2 {
        let view = try #require(cache.logicalView(tail: tail))
        #expect(
            encodedPositions(view.0) == [0, 1],
            "tail \(tail) below `keep` dropped the pinned prefix")
        #expect(encodedPositions(view.1).map { -$0 } == [0, 1], "values diverged from keys")
    }

    // One past the floor is the first tail that actually selects anything.
    let positions = encodedPositions(try #require(cache.logicalView(tail: 3)).0)
    #expect(positions.count == 3)
    #expect(Array(positions.prefix(2)) == [0, 1], "the prefix moved once the tail cleared it")
}

@Test func testRotatingLogicalViewFloorIsBoundedByWhatTheCacheHolds() throws {
    // Fewer entries written than `keep` pins: the floor is what exists, not what is reserved.
    let cache = RotatingKVCache(maxSize: 8, keep: 4)
    fillOneAtATime(cache, positions: 0 ..< 2)

    #expect(encodedPositions(try #require(cache.logicalView(tail: 0)).0) == [0, 1])
    #expect(encodedPositions(try #require(cache.logicalView(tail: 9)).0) == [0, 1])
}

/// The property the speculative overlay rests on: for a multi-token write, the presentation the
/// cache *would* return equals `logicalView(tail: maxSize - 1)` followed by the new rows. An
/// adapter can therefore stage beside the ring and hand the model an identical view.
@Test func testRotatingLogicalViewPlusNewRowsEqualsTheWritePresentation() throws {
    let window = 8
    let staged = 4

    let cache = RotatingKVCache(maxSize: window, keep: 0)
    fillOneAtATime(cache, positions: 0 ..< 20)

    // Read the view and the mask *before* the write, exactly as an adapter would.
    let view = try #require(cache.logicalView(tail: window - 1))
    let mask = cache.makeMask(n: staged, windowSize: window, returnArray: false)

    let (newKeys, newValues) = positionedKV(20 ..< (20 + staged))
    let presented = cache.update(keys: newKeys, values: newValues)

    let composedKeys = concatenated([view.0, newKeys], axis: 2)
    let composedValues = concatenated([view.1, newValues], axis: 2)

    #expect(
        encodedPositions(presented.0) == encodedPositions(composedKeys),
        "staged presentation diverged from the live write path")
    #expect(
        allClose(presented.0, composedKeys, rtol: 0, atol: 0).item(Bool.self),
        "keys diverged from the live write path")
    #expect(
        allClose(presented.1, composedValues, rtol: 0, atol: 0).item(Bool.self),
        "values diverged from the live write path")

    // The mask the cache built before the write has to span exactly that presentation, or an
    // adapter could not delegate mask construction to the live cache.
    guard case .array(let maskArray) = mask else {
        Issue.record("expected an array mask once offset + n exceeds the window, got \(mask)")
        return
    }
    #expect(
        maskArray.dim(-1) == composedKeys.dim(2),
        "mask key axis \(maskArray.dim(-1)) != presented length \(composedKeys.dim(2))")
    #expect(maskArray.dim(-2) == staged)
}

/// Attention through the rotating cache's own presentation and mask, past a wrap, must equal a
/// reference that never rotated: every position kept in a plain array under an explicit
/// causal-intersect-window mask over absolute positions.
@Test func testRotatingCacheAttentionPastWrapMatchesLogicalOrderReference() {
    MLXRandom.seed(0)

    let window = 8
    let history = 20
    let queries = 4
    let heads = 2
    let headDim = 4
    let scale = 1.0 / Float(headDim).squareRoot()

    let allKeys = MLXRandom.normal([1, heads, history + queries, headDim])
    let allValues = MLXRandom.normal([1, heads, history + queries, headDim])
    let q = MLXRandom.normal([1, heads, queries, headDim])

    let cache = RotatingKVCache(maxSize: window, keep: 0)
    for p in 0 ..< history {
        _ = cache.update(
            keys: allKeys[0..., 0..., p ..< (p + 1), 0...],
            values: allValues[0..., 0..., p ..< (p + 1), 0...])
    }

    let mask = cache.makeMask(n: queries, windowSize: window, returnArray: false)
    let (cachedKeys, cachedValues) = cache.update(
        keys: allKeys[0..., 0..., history ..< (history + queries), 0...],
        values: allValues[0..., 0..., history ..< (history + queries), 0...])
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: cachedKeys, values: cachedValues, scale: scale, mask: mask)

    // Reference: all history retained, masked by absolute position.
    let queryPositions = MLXArray(Int32(history) ..< Int32(history + queries))[0..., .newAxis]
    let keyPositions = MLXArray(Int32(0) ..< Int32(history + queries))[.newAxis]
    let referenceMask =
        (queryPositions .>= keyPositions) & (queryPositions .< keyPositions + Int32(window))
    let reference = MLXFast.scaledDotProductAttention(
        queries: q, keys: allKeys, values: allValues, scale: scale, mask: .array(referenceMask))

    #expect(out.shape == reference.shape)
    #expect(
        allClose(out, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self),
        "post-wrap attention diverged from the logical-order reference")
}
