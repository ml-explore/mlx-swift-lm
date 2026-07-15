// Copyright © 2026 Apple Inc.

import MLXHuggingFaceMacros
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class LanguageModelMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLanguageModel": LanguageModelMacro.self
    ]

    func testConfigurationOnly() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config)",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    weightsLocation: { id in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                            return cache.cacheDirectory
                        }
                        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                            let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                            return snapshot
                        }
                        return cache.repoDirectory(repo: repo, kind: .model)
                    },
                    load: { configuration, progressHandler in
                        try await loadModelContainer(
                            from: #hubDownloader(),
                            using: #huggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progressHandler: progressHandler)
                    })
                """,
            macros: testMacros)
    }

    func testExplicitCapabilities() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, capabilities: [.guidedGeneration, .toolCalling])",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    capabilities: [.guidedGeneration, .toolCalling],
                    weightsLocation: { id in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                            return cache.cacheDirectory
                        }
                        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                            let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                            return snapshot
                        }
                        return cache.repoDirectory(repo: repo, kind: .model)
                    },
                    load: { configuration, progressHandler in
                        try await loadModelContainer(
                            from: #hubDownloader(),
                            using: #huggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progressHandler: progressHandler)
                    })
                """,
            macros: testMacros)
    }

    func testExplicitConfigurationResolver() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, configurationResolver: MyResolver())",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    configurationResolver: MyResolver(),
                    weightsLocation: { id in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                            return cache.cacheDirectory
                        }
                        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                            let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                            return snapshot
                        }
                        return cache.repoDirectory(repo: repo, kind: .model)
                    },
                    load: { configuration, progressHandler in
                        try await loadModelContainer(
                            from: #hubDownloader(),
                            using: #huggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progressHandler: progressHandler)
                    })
                """,
            macros: testMacros)
    }

    func testExplicitCapabilitiesAndConfigurationResolver() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, capabilities: [.guidedGeneration, .toolCalling], configurationResolver: MyResolver())",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    capabilities: [.guidedGeneration, .toolCalling],
                    configurationResolver: MyResolver(),
                    weightsLocation: { id in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                            return cache.cacheDirectory
                        }
                        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                            let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                            return snapshot
                        }
                        return cache.repoDirectory(repo: repo, kind: .model)
                    },
                    load: { configuration, progressHandler in
                        try await loadModelContainer(
                            from: #hubDownloader(),
                            using: #huggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progressHandler: progressHandler)
                    })
                """,
            macros: testMacros)
    }

    func testMissingConfigurationDiagnoses() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel()",
            expandedSource: "let model = #huggingFaceLanguageModel()",
            diagnostics: [
                DiagnosticSpec(
                    message: "#huggingFaceLanguageModel requires a configuration",
                    line: 1,
                    column: 13)
            ],
            macros: testMacros)
    }
}
