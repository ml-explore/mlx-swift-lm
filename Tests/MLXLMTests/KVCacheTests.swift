import Foundation
import MLX
import MLXLMCommon
import Testing

private typealias CacheFactory = @Sendable () -> any KVCache

@Test(
    .serialized,
    arguments: [
        ({ KVCacheSimple() } as CacheFactory),
        ({ RotatingKVCache(maxSize: 32) } as CacheFactory),
        ({ QuantizedKVCache() } as CacheFactory),
        ({ TurboQuantKVCache(bits: 3.5) } as CacheFactory),
        ({ ChunkedKVCache(chunkSize: 16) } as CacheFactory),
        ({ ArraysCache(size: 2) } as CacheFactory),
        ({ MambaCache() } as CacheFactory),
    ])
func testCacheSerialization(creator: CacheFactory) async throws {
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

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}
