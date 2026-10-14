// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Stores token ids, standing in for a cache type defined outside MLXLMCommon.
private final class TokenListCache: BaseKVCache {
    override static var serializedClassName: String? { "Tests.TokenListCache" }

    var tokens: [Int32]

    init(tokens: [Int32]) {
        self.tokens = tokens
        super.init()
        offset = tokens.count
    }

    override var state: [MLXArray] {
        get { [MLXArray(tokens)] }
        set { tokens = newValue.first?.asArray(Int32.self) ?? [] }
    }

    override var metaState: [String] {
        get { [String(offset)] }
        set { offset = newValue.first.flatMap(Int.init) ?? tokens.count }
    }

    override var isTrimmable: Bool { true }

    override func copy() -> any KVCache { TokenListCache(tokens: tokens) }

    override static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        let cache = TokenListCache(tokens: state.first?.asArray(Int32.self) ?? [])
        cache.metaState = metaState
        return cache
    }
}

private final class OtherTokenListCache: BaseKVCache {
    override static var serializedClassName: String? { "Tests.TokenListCache" }
    override func copy() -> any KVCache { OtherTokenListCache() }
    override static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        OtherTokenListCache()
    }
}

private final class BuiltInNameCache: BaseKVCache {
    override static var serializedClassName: String? { "KVCacheSimple" }
    override func copy() -> any KVCache { BuiltInNameCache() }
}

/// Never registered explicitly, so only saving registers it.
private final class SavedOnlyCache: BaseKVCache {
    override static var serializedClassName: String? { "Tests.SavedOnlyCache" }
    override func copy() -> any KVCache { SavedOnlyCache() }
    override static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        SavedOnlyCache()
    }
}

private final class UnnamedCache: BaseKVCache {
    override func copy() -> any KVCache { UnnamedCache() }
}

private final class InheritedNameCache: KVCacheSimple {}

struct KVCacheTypeRegistryTests {

    static func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "prompt-cache-\(UUID().uuidString).safetensors")
    }

    @Test func `a registered custom cache saves and restores with built-in caches`() throws {
        try registerKVCacheType(TokenListCache.self)

        let simple = KVCacheSimple()
        _ = simple.update(keys: MLXArray.ones([1, 2, 3, 8]), values: MLXArray.ones([1, 2, 3, 8]))
        let custom = TokenListCache(tokens: [7, 8, 9])

        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [simple, custom])

        let restored = try loadPromptCacheSnapshot(url: url).cache
        #expect(restored.count == 2)
        #expect(restored[0] is KVCacheSimple)
        #expect(restored[0].offset == 3)
        let tokens = try #require(restored[1] as? TokenListCache)
        #expect(tokens.tokens == [7, 8, 9])
        #expect(tokens.offset == 3)
    }

    @Test func `a custom cache inside a CacheList round-trips`() throws {
        try registerKVCacheType(TokenListCache.self)

        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(
            url: url, cache: [CacheList(KVCacheSimple(), TokenListCache(tokens: [1, 2]))])

        let list = try #require(try loadPromptCacheSnapshot(url: url).cache.first as? CacheList)
        let tokens = try #require(list.children.last as? TokenListCache)
        #expect(tokens.tokens == [1, 2])
    }

    @Test func `saving registers the type so the same process can load it`() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [SavedOnlyCache()])

        #expect(try loadPromptCacheSnapshot(url: url).cache.first is SavedOnlyCache)
    }

    @Test func `registering the same type twice is allowed`() throws {
        try registerKVCacheType(TokenListCache.self)
        try registerKVCacheType(TokenListCache.self)
    }

    @Test func `another type cannot take a registered name`() throws {
        try registerKVCacheType(TokenListCache.self)
        #expect(throws: (any Error).self) {
            try registerKVCacheType(OtherTokenListCache.self)
        }

        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [OtherTokenListCache()])
        }
    }

    @Test func `built-in names and aliases are taken`() {
        #expect(throws: (any Error).self) {
            try registerKVCacheType(BuiltInNameCache.self)
        }
        #expect(throws: (any Error).self) {
            try registerKVCacheType(InheritedNameCache.self)
        }
    }

    @Test func `a cache without a serialized class name cannot be saved`() {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [UnnamedCache()])
        }
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [CacheList(KVCacheSimple(), UnnamedCache())])
        }
    }

    @Test func `a subclass that keeps its parent's name restores as the parent`() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [InheritedNameCache()])

        let restored = try #require(try loadPromptCacheSnapshot(url: url).cache.first)
        #expect(type(of: restored) == KVCacheSimple.self)
    }
}
