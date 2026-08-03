// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Verifies `reasoningConfig` survives the `ModelConfiguration` →
/// `ResolvedModelConfiguration` propagation (sites 1–3 of the chain).
///
/// Sites 4–5 (`LLMModelFactory._load`'s inference block and its hand-rebuilt
/// `ModelConfiguration`) are verified end-to-end by the on-device reasoning
/// integration tests: reasoning only routes when the resolved config reaches
/// `ModelContext.configuration` through that reconstruction, so a dropped field
/// there surfaces as "no reasoning events on a known reasoning model".
@Suite
struct ReasoningConfigResolutionTests {

    private static let dir = URL(fileURLWithPath: "/tmp/reasoning-tests-fixture")
    private static let qwen3 = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true)

    // MARK: - ModelConfiguration field

    @Test func idInitDefaultsToNil() {
        #expect(
            ModelConfiguration(id: "mlx-community/Qwen2.5-3B-Instruct-4bit").reasoningConfig == nil)
    }

    @Test func idInitCarriesReasoningConfig() {
        let c = ModelConfiguration(id: "x", reasoningConfig: Self.qwen3)
        #expect(c.reasoningConfig == Self.qwen3)
    }

    @Test func directoryInitCarriesReasoningConfig() {
        let c = ModelConfiguration(directory: Self.dir, reasoningConfig: Self.qwen3)
        #expect(c.reasoningConfig == Self.qwen3)
    }

    // MARK: - resolved() round-trip (the field must survive into the resolved form)

    @Test func resolvedPreservesReasoningConfig() {
        let resolved = ModelConfiguration(id: "x", reasoningConfig: Self.qwen3)
            .resolved(modelDirectory: Self.dir, tokenizerDirectory: Self.dir)
        #expect(resolved.reasoningConfig == Self.qwen3)
    }

    @Test func resolvedPreservesNil() {
        let resolved = ModelConfiguration(id: "x")
            .resolved(modelDirectory: Self.dir, tokenizerDirectory: Self.dir)
        #expect(resolved.reasoningConfig == nil)
    }

    // MARK: - ResolvedModelConfiguration directory convenience

    @Test func resolvedDirectoryConvenienceDefaultsNil() {
        #expect(ResolvedModelConfiguration(directory: Self.dir).reasoningConfig == nil)
    }

    // MARK: - Equatable still holds with the new field

    @Test func equatableIncludesReasoningConfig() {
        let a = ModelConfiguration(id: "x", reasoningConfig: Self.qwen3)
        let b = ModelConfiguration(id: "x", reasoningConfig: Self.qwen3)
        let c = ModelConfiguration(id: "x", reasoningConfig: nil)
        #expect(a == b)
        #expect(a != c)
    }
}
