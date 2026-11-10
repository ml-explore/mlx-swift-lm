// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

struct ModelTypeRegistryTests {

    private static func configuration(architecture: String) -> Data {
        Data(#"{"model_type":"shared","architectures":["\#(architecture)"]}"#.utf8)
    }

    private func registry() async -> ModelTypeRegistry<String> {
        let registry = ModelTypeRegistry<String>()
        await registry.registerModelType(
            "shared",
            matches: { String(decoding: $0, as: UTF8.self).contains("Drafter") },
            creator: { _ in "drafter" })
        return registry
    }

    @Test("A shared model type is contained only for the configurations its creator accepts")
    func containsHonoursTheMatchesPredicate() async {
        let registry = await registry()
        let drafter = Self.configuration(architecture: "Drafter")
        let other = Self.configuration(architecture: "Language")

        #expect(await registry.contains("shared"))
        #expect(await registry.contains("shared", configuration: drafter))
        #expect(await !registry.contains("shared", configuration: other))
        #expect(await !registry.contains("unknown", configuration: drafter))
        await #expect(throws: ModelFactoryError.self) {
            _ = try await registry.createModel(configuration: other, modelType: "shared")
        }
    }

    @Test("A model type registered without a predicate accepts every configuration")
    func containsAcceptsEveryConfigurationWithoutAPredicate() async {
        let registry = ModelTypeRegistry<String>(creators: ["plain": { _ in "plain" }])

        #expect(await registry.contains("plain", configuration: Data("{}".utf8)))
    }
}
