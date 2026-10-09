// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Stores token ids, standing in for a cache type defined outside MLXLMCommon.
private final class TokenListCache: BaseKVCache, PromptCacheRestorable {
    static var promptCacheClassName: String { "Tests.TokenListCache" }

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

    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        let cache = TokenListCache(tokens: state.first?.asArray(Int32.self) ?? [])
        cache.metaState = metaState
        return cache
    }
}

private final class OtherTokenListCache: BaseKVCache, PromptCacheRestorable {
    static var promptCacheClassName: String { "Tests.TokenListCache" }
    override func copy() -> any KVCache { OtherTokenListCache() }
    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        OtherTokenListCache()
    }
}

private final class BuiltInNameCache: BaseKVCache, PromptCacheRestorable {
    static var promptCacheClassName: String { "KVCacheSimple" }
    override func copy() -> any KVCache { BuiltInNameCache() }
    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        BuiltInNameCache()
    }
}

private final class UnregisteredCache: BaseKVCache, PromptCacheRestorable {
    static var promptCacheClassName: String { "Tests.UnregisteredCache" }
    override func copy() -> any KVCache { UnregisteredCache() }
    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        UnregisteredCache()
    }
}

struct PromptCacheTypeRegistryTests {

    static func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "prompt-cache-\(UUID().uuidString).safetensors")
    }

    @Test func `a registered custom cache saves and restores with built-in caches`() throws {
        try registerPromptCacheType(TokenListCache.self)

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
        try registerPromptCacheType(TokenListCache.self)

        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(
            url: url, cache: [CacheList(KVCacheSimple(), TokenListCache(tokens: [1, 2]))])

        let list = try #require(try loadPromptCacheSnapshot(url: url).cache.first as? CacheList)
        let tokens = try #require(list.children.last as? TokenListCache)
        #expect(tokens.tokens == [1, 2])
    }

    @Test func `registering the same type twice is allowed`() throws {
        try registerPromptCacheType(TokenListCache.self)
        try registerPromptCacheType(TokenListCache.self)
    }

    @Test func `another type cannot take a registered name`() throws {
        try registerPromptCacheType(TokenListCache.self)
        #expect(throws: (any Error).self) {
            try registerPromptCacheType(OtherTokenListCache.self)
        }
    }

    @Test func `built-in names are reserved`() {
        #expect(throws: (any Error).self) {
            try registerPromptCacheType(BuiltInNameCache.self)
        }
    }

    @Test func `an unregistered custom cache saves under its name but does not load`() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [UnregisteredCache()])

        do {
            _ = try loadPromptCacheSnapshot(url: url)
            Issue.record("expected an unknown cache class error")
        } catch {
            #expect(error.localizedDescription.contains("Tests.UnregisteredCache"))
            #expect(error.localizedDescription.contains("registerPromptCacheType"))
        }
    }
}
