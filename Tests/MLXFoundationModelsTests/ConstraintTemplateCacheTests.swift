// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

// Extends the serialized parent declared in ModelCacheEvictionTests.swift: these
// tests read the process-global cache's compilation counter, which other
// cache suites would disturb if they ran at the same time.
extension FoundationModelsCacheTests {

    @Suite("MLXLanguageModel constraint template cache")
    struct ConstraintTemplateCaching {

        private static let integerSchema = #"{ "type": "integer" }"#
        private static let stringSchema = #"{ "type": "string" }"#

        private func makeTokenizer() throws -> GrammarTokenizer {
            let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
            return try GrammarTokenizer(vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func makeConstraint(
            modelID: String, source: String = integerSchema, kind: ConstraintKind = .json,
            fastForward: Bool = true
        ) async throws -> GrammarConstraint {
            try await MLXLanguageModel.makeConstraint(
                modelID: modelID,
                kind: kind,
                source: source,
                tokenizer: try makeTokenizer(),
                hostTokenizer: TemplateStubTokenizer(),
                fastForward: fastForward)
        }

        private func allows(_ mask: MaskResult, _ token: Int) -> Bool {
            (UInt32(bitPattern: mask.mask[token / 32]) >> UInt32(token % 32)) & 1 == 1
        }

        @Test("the same schema compiles once, later requests reuse the compiled grammar")
        func compilesOnce() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/template-\(UUID().uuidString)"
            let before = await MLXLanguageModel.constraintCompilationCount()

            _ = try await makeConstraint(modelID: id)
            _ = try await makeConstraint(modelID: id)
            _ = try await makeConstraint(modelID: id)

            let compiled = await MLXLanguageModel.constraintCompilationCount() - before
            #expect(compiled == 1, "three requests with one schema must compile once")
            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: id) == 1)
        }

        @Test("each request starts at the grammar's initial state")
        func requestsAreIndependent() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/template-independent-\(UUID().uuidString)"

            let first = try await makeConstraint(modelID: id)
            _ = try first.commitToken(45)  // "-": an integer may continue with a digit only
            let firstMask = try first.computeMask()
            #expect(!allows(firstMask, 45))

            let second = try await makeConstraint(modelID: id)
            let secondMask = try second.computeMask()
            #expect(allows(secondMask, 45), "a reused grammar must not inherit matcher state")
            #expect(allows(secondMask, 52))
        }

        @Test("kind, fast-forward setting and source each get their own template")
        func keyCoversEveryInput() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/template-key-\(UUID().uuidString)"
            let before = await MLXLanguageModel.constraintCompilationCount()

            _ = try await makeConstraint(modelID: id, fastForward: true)
            _ = try await makeConstraint(modelID: id, fastForward: false)
            _ = try await makeConstraint(modelID: id, source: Self.stringSchema)

            let compiled = await MLXLanguageModel.constraintCompilationCount() - before
            #expect(compiled == 3)
            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: id) == 3)
        }

        @Test("the cache keeps the most recently used templates and drops the rest")
        func evictsLeastRecentlyUsed() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            await MLXLanguageModel.evictAll()
            let id = "org/template-lru-\(UUID().uuidString)"
            let limit = ModelCache.maxConstraintTemplates

            // Fill the cache, then touch the first schema so it is the most recent.
            for i in 0 ..< limit {
                _ = try await makeConstraint(modelID: id, source: Self.schema(i))
            }
            _ = try await makeConstraint(modelID: id, source: Self.schema(0))

            // One more schema evicts the least recently used one: schema(1).
            _ = try await makeConstraint(modelID: id, source: Self.schema(limit))
            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: id) == limit)

            let before = await MLXLanguageModel.constraintCompilationCount()
            _ = try await makeConstraint(modelID: id, source: Self.schema(0))
            _ = try await makeConstraint(modelID: id, source: Self.schema(limit))
            #expect(await MLXLanguageModel.constraintCompilationCount() == before)

            _ = try await makeConstraint(modelID: id, source: Self.schema(1))
            #expect(await MLXLanguageModel.constraintCompilationCount() == before + 1)
        }

        @Test("evict() drops only this model's templates")
        func evictIsPerModel() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let idA = "org/template-evict-a-\(UUID().uuidString)"
            let idB = "org/template-evict-b-\(UUID().uuidString)"
            let model = MLXLanguageModel(
                configuration: ModelConfiguration(id: idA),
                capabilities: [],
                weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                load: { configuration, progress in
                    try await loadModelContainer(
                        from: TemplateStubDownloader(), using: TemplateStubTokenizerLoader(),
                        configuration: configuration, progress: progress)
                })

            _ = try await makeConstraint(modelID: idA)
            _ = try await makeConstraint(modelID: idB)

            await model.evict()

            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: idA) == 0)
            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: idB) == 1)

            let before = await MLXLanguageModel.constraintCompilationCount()
            _ = try await makeConstraint(modelID: idB)
            #expect(await MLXLanguageModel.constraintCompilationCount() == before)
        }

        @Test("evictAll() forces a recompile on the next request")
        func evictAllClearsTemplates() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let id = "org/template-evictall-\(UUID().uuidString)"
            _ = try await makeConstraint(modelID: id)

            await MLXLanguageModel.evictAll()
            #expect(await MLXLanguageModel.cachedConstraintTemplateCount(modelID: id) == 0)

            let before = await MLXLanguageModel.constraintCompilationCount()
            _ = try await makeConstraint(modelID: id)
            #expect(await MLXLanguageModel.constraintCompilationCount() == before + 1)
        }

        /// A distinct JSON schema per index: an integer with a different lower bound.
        private static func schema(_ index: Int) -> String {
            #"{ "type": "integer", "minimum": \#(index) }"#
        }
    }
}

// MARK: - Fixtures

private struct TemplateStubDownloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL { URL(fileURLWithPath: "/no/such/path") }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct TemplateStubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { TemplateStubTokenizer() }
}

private struct TemplateStubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
