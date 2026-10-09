// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

struct Qwen38CheckpointTests {
    @Test(arguments: [Qwen35CheckpointPolicy.TargetLayout.text, .wrappedText, .vision])
    func visualNamespacesKeepProvenanceAndPrecision(layout: Qwen35CheckpointPolicy.TargetLayout)
        throws
    {
        let checkpoint = ModelCheckpoint(
            weights: [
                "visual.merger.fc.weight": MLXArray(1),
                "model.visual.blocks.0.fc.weight": MLXArray(2),
                "vision_tower.blocks.1.fc.weight": MLXArray(3),
                "visual_extra.weight": MLXArray(4),
            ],
            weightMetadata: ["visual.merger.fc.weight": ["source": "vision"]],
            perLayerQuantization: .init(perLayerQuantization: [
                "visual.merger.fc": .quantize(.init(groupSize: 32, bits: 8)),
                "model.visual.blocks.0.fc": .skip,
                "vision_tower.blocks.1.fc": .skip,
                "visual_extra": .skip,
            ]))
        let prepared = try Qwen35CheckpointPolicy.prepareTarget(
            checkpoint, layout: layout, tiedWordEmbeddings: false)
        #expect(prepared.weights["visual_extra.weight"] != nil)
        guard case .skip? = prepared.perLayerQuantization?.perLayerQuantization["visual_extra"]
        else {
            Issue.record("The unrelated module's precision declaration was removed")
            return
        }
        #expect(prepared.weights["visual.merger.fc.weight"] == nil)
        #expect(prepared.perLayerQuantization?.perLayerQuantization["visual.merger.fc"] == nil)
        if layout == .vision {
            #expect(prepared.weights.count == 4)
            #expect(prepared.weights["vision_tower.merger.fc.weight"]?.item(Int.self) == 1)
            #expect(prepared.weights["vision_tower.blocks.0.fc.weight"]?.item(Int.self) == 2)
            #expect(prepared.weights["vision_tower.blocks.1.fc.weight"]?.item(Int.self) == 3)
            #expect(
                prepared.metadata(forWeight: "vision_tower.merger.fc.weight")["source"] == "vision")
            #expect(
                prepared.perLayerQuantization?.quantization(layer: "vision_tower.merger.fc")?.bits
                    == 8)
            guard
                case .skip? = prepared.perLayerQuantization?.perLayerQuantization[
                    "vision_tower.blocks.0.fc"]
            else {
                Issue.record("The visual module's precision declaration was changed")
                return
            }
        } else {
            #expect(Set(prepared.weights.keys) == ["visual_extra.weight"])
            #expect(prepared.perLayerQuantization?.perLayerQuantization.count == 1)
        }
    }

    @Test
    func competingVisualAliasesAreRejected() {
        for prefix in ["model.visual", "vision_tower"] {
            #expect(throws: ModelCheckpoint.MappingError.self) {
                try Qwen35CheckpointPolicy.prepareTarget(
                    .init(weights: [
                        "visual.fc.weight": MLXArray(1), "\(prefix).fc.weight": MLXArray(2),
                    ]), layout: .vision, tiedWordEmbeddings: false)
            }
        }
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try Qwen35CheckpointPolicy.prepareTarget(
                .init(
                    weights: ["visual.fc.weight": MLXArray(1)],
                    perLayerQuantization: .init(perLayerQuantization: [
                        "visual.fc": .skip, "vision_tower.fc": .skip,
                    ])), layout: .vision, tiedWordEmbeddings: false)
        }
    }
}
