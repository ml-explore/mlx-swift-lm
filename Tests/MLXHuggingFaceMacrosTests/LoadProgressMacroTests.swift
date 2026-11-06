// Copyright © 2026 Apple Inc.

import MLXHuggingFaceMacros
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class LoadProgressMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLoadModelContainer": LoadContainerMacro.self,
        "huggingFaceLoadModel": LoadContextMacro.self,
    ]

    func testContainerForwardsWeightProgress() {
        assertMacroExpansion(
            "let model = #huggingFaceLoadModelContainer(configuration: config, progress: handlers)",
            expandedSource: """
                let model = loadModelContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progress: handlers)
                """,
            macros: testMacros)
    }

    func testContainerWrapsLegacyDownloadProgress() {
        assertMacroExpansion(
            "let model = #huggingFaceLoadModelContainer(configuration: config, progressHandler: download)",
            expandedSource: """
                let model = loadModelContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progress: .download(download))
                """,
            macros: testMacros)
    }

    func testContextForwardsWeightProgress() {
        assertMacroExpansion(
            "let context = #huggingFaceLoadModel(configuration: config, progress: handlers)",
            expandedSource: """
                let context = loadModel(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progress: handlers)
                """,
            macros: testMacros)
    }
}
