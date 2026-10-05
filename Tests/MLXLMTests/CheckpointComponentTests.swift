// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

struct CheckpointComponentTests {
    private let component = CheckpointComponent(
        name: "predictor", namespaces: ["head", "wrapper.head"], destination: "predictor",
        excludedNamespaces: ["backbone", "wrapper"])

    @Test(arguments: ["head", "wrapper.head"])
    func selectionKeepsComponentProvenanceAndPrecision(namespace: String) throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "\(namespace).fc.weight": MLXArray(1), "backbone.fc.weight": MLXArray(2),
                "wrapper.backbone.weight": MLXArray(3), "unknown.weight": MLXArray(4),
            ],
            metadata: ["format": "mlx"],
            weightMetadata: ["\(namespace).fc.weight": ["component_format": "native"]],
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(namespace).fc": .quantize(.init(groupSize: 32, bits: 8)),
                    "backbone.fc": .skip,
                ]))
        let selected = try component.select(from: checkpoint)
        #expect(selected.source == .embedded(namespace: namespace))
        #expect(Set(selected.checkpoint.weights.keys) == ["predictor.fc.weight", "unknown.weight"])
        #expect(
            selected.checkpoint.metadata(forWeight: "predictor.fc.weight")["component_format"]
                == "native")
        #expect(
            selected.checkpoint.perLayerQuantization?.quantization(layer: "predictor.fc")?.bits == 8
        )
        #expect(
            selected.checkpoint.perLayerQuantization?.quantization(layer: "predictor.other")?.bits
                == 4)
        #expect(
            selected.checkpoint.perLayerQuantization?.perLayerQuantization["backbone.fc"] == nil)
    }

    @Test
    func standaloneRequiresAnExplicitDeclaration() throws {
        let checkpoint = ModelCheckpoint(weights: ["fc.weight": MLXArray(1)])
        #expect(throws: CheckpointComponent.SelectionError.self) {
            try component.select(from: checkpoint)
        }
        let selected = try component.select(from: checkpoint, standalone: true)
        #expect(selected.source == .standalone)
        #expect(selected.checkpoint.weights["predictor.fc.weight"] != nil)
        #expect(throws: CheckpointComponent.SelectionError.self) {
            try component.select(from: .init(weights: [:]), standalone: true)
        }
    }

    @Test
    func competingNamespacesAndDestinationAliasesAreRejected() {
        #expect(throws: CheckpointComponent.SelectionError.self) {
            try component.select(
                from: .init(weights: [
                    "head.fc.weight": MLXArray(1), "wrapper.head.fc.weight": MLXArray(2),
                ]))
        }
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try component.select(
                from: .init(weights: [
                    "head.fc.weight": MLXArray(1), "predictor.fc.weight": MLXArray(2),
                ]))
        }
    }

    @Test
    func excludingAComponentPreservesSimilarlyNamedModules() throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "head.fc.weight": MLXArray(1), "headless.weight": MLXArray(2),
                "backbone.head.weight": MLXArray(3), "wrapper.head.fc.weight": MLXArray(4),
            ],
            perLayerQuantization: .init(perLayerQuantization: ["head": .skip, "headless": .skip]))
        let selected = try component.excluding(from: checkpoint)
        #expect(Set(selected.weights.keys) == ["headless.weight", "backbone.head.weight"])
        #expect(selected.perLayerQuantization?.perLayerQuantization["head"] == nil)
        guard case .skip? = selected.perLayerQuantization?.perLayerQuantization["headless"] else {
            Issue.record("The unrelated module's precision declaration was removed")
            return
        }
    }

    @Test
    func orderedMappingsUseModuleBoundariesForTensorsAndSettings() throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "outer.text.fc.weight": MLXArray(1), "outer.text.lm_head.scales": MLXArray(2),
                "outer.text.lm_head_extra.weight": MLXArray(3), "outer.texture.weight": MLXArray(4),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "outer.text.fc": .quantize(.init(groupSize: 32, bits: 8)),
                "outer.text.lm_head": .skip,
            ]))
        let mapped = try checkpoint.mapNames(
            using: .init([
                .replacePrefix("outer", with: ""), .replacePrefix("text", with: "language_model"),
                .excludeModule("lm_head"),
            ]))
        #expect(
            Set(mapped.weights.keys) == [
                "language_model.fc.weight", "language_model.lm_head_extra.weight", "texture.weight",
            ])
        #expect(mapped.perLayerQuantization?.quantization(layer: "language_model.fc")?.bits == 8)
        #expect(mapped.perLayerQuantization?.perLayerQuantization["language_model.lm_head"] == nil)
    }
}
