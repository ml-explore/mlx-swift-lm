// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The serialized MTP component is independent of its runtime `mtp` wrapper.
package struct Qwen35CheckpointPolicy: Sendable {
    package enum Layout: Sendable {
        case embedded
        case standalone
    }

    package let layout: Layout
    package let preconvertedNorms: Bool

    package init(layout: Layout, preconvertedNorms: Bool) {
        self.layout = layout
        self.preconvertedNorms = preconvertedNorms
    }

    package static let normMetadataKey = "mlx_swift_lm.qwen_mtp.norm_convention"
    private static let mtpComponent = CheckpointComponent(
        name: "Qwen MTP", namespaces: ["mtp", "language_model.mtp"], destination: "mtp",
        excludedNamespaces: [
            "model", "language_model.model", "language_model.lm_head", "lm_head",
            "vision_tower", "visual",
        ])
    private static let normSuffixes = [
        "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
        ".input_layernorm.weight", ".post_attention_layernorm.weight",
        ".q_norm.weight", ".k_norm.weight",
    ]

    package static func targetWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapping = CheckpointNameMapping(mtpComponent.namespaces.map { .excludePrefix($0) })
        return weights.filter { name, _ in mapping.mapName(name) != nil }
    }

    package enum TargetLayout {
        case text
        case wrappedText
        case vision
    }

    package static func prepareTarget(
        _ checkpoint: ModelCheckpoint, layout: TargetLayout, tiedWordEmbeddings: Bool
    ) throws -> ModelCheckpoint {
        var rules = [CheckpointNameMapping.Rule]()
        if tiedWordEmbeddings { rules.append(.excludeModule("lm_head")) }
        if layout == .vision {
            rules.append(.replacePrefix("model.visual", with: "vision_tower"))
        } else {
            rules += [.excludePrefix("model.visual"), .excludePrefix("vision_tower")]
        }
        rules.append(.replacePrefix("model.language_model", with: "model"))
        if layout == .text {
            rules.append(.replacePrefix("language_model", with: ""))
        } else {
            rules += [
                .replacePrefix("model", with: "language_model.model"),
                .replacePrefix("lm_head", with: "language_model.lm_head"),
            ]
        }
        let selected = try mtpComponent.excluding(from: checkpoint)
        return try selected.mapNames(using: .init(rules))
    }

    package func prepare(
        _ checkpoint: ModelCheckpoint, mtpNumHiddenLayers: Int, numExperts: Int
    ) throws -> ModelCheckpoint {
        let selection = try Self.mtpComponent.select(
            from: checkpoint, standalone: layout == .standalone)
        var prepared = selection.checkpoint

        var conventions = Set<NormConvention>()
        for name in prepared.weights.keys.sorted() where Self.isNorm(name) {
            let convention = try normConvention(
                metadata: prepared.metadata(forWeight: name), source: selection.source)
            conventions.insert(convention)
            if convention == .offset, let value = prepared.weights[name], value.ndim == 1 {
                prepared.weights[name] = value + MLXArray(1, dtype: value.dtype)
            }
        }
        guard conventions.count <= 1 else { throw LoadingError.conflictingNormConventions }
        try prepareExperts(
            &prepared, layers: max(mtpNumHiddenLayers, 1), numExperts: numExperts)
        return prepared
    }

    private enum NormConvention: String {
        case offset
        case scale
    }

    private func normConvention(metadata: [String: String], source: CheckpointComponent.Source)
        throws -> NormConvention
    {
        if let declared = metadata[Self.normMetadataKey] {
            guard let convention = NormConvention(rawValue: declared) else {
                throw LoadingError.unsupportedNormConvention(declared)
            }
            return convention
        }
        if preconvertedNorms || source != .embedded(namespace: "mtp")
            || metadata["format"]?.lowercased() == "mlx"
        {
            return .scale
        }
        return .offset
    }

    private static func isNorm(_ name: String) -> Bool {
        normSuffixes.contains { name.hasSuffix($0) }
    }

    private func prepareExperts(
        _ checkpoint: inout ModelCheckpoint, layers: Int, numExperts: Int
    ) throws {
        for layer in 0 ..< layers {
            let prefix = "mtp.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let source = "\(prefix).experts"
                let destination = "\(prefix).switch_mlp.\(projection)"
                let modules = (0 ..< numExperts).map { "\(source).\($0).\(projection)" }
                guard modules.contains(where: { checkpoint.weights["\($0).weight"] != nil }) else {
                    continue
                }
                try mergeQuantization(&checkpoint, modules: modules, destination: destination)
                for parameter in ["weight", "scales", "biases"] {
                    let names = modules.map { "\($0).\(parameter)" }
                    guard names.contains(where: { checkpoint.weights[$0] != nil }) else { continue }
                    let arrays = try names.map { name in
                        guard let array = checkpoint.weights[name] else {
                            throw LoadingError.incompleteExperts(name)
                        }
                        return array
                    }
                    guard arrays.allSatisfy({ $0.shape == arrays[0].shape }) else {
                        throw LoadingError.incompleteExperts(source)
                    }
                    let name = "\(destination).\(parameter)"
                    guard checkpoint.weights[name] == nil else {
                        throw LoadingError.ambiguousLayout
                    }
                    checkpoint.weights[name] = stacked(arrays)
                    for name in names { checkpoint.weights.removeValue(forKey: name) }
                }
                for module in modules {
                    checkpoint.perLayerQuantization?.perLayerQuantization.removeValue(
                        forKey: module)
                }
            }

            let fused = "\(prefix).experts.gate_up_proj"
            if let value = checkpoint.weights[fused] {
                guard value.ndim >= 2, value.dim(-2) % 2 == 0 else {
                    throw LoadingError.incompleteExperts(fused)
                }
                let destinations = ["gate_proj", "up_proj"].map { "\(prefix).switch_mlp.\($0)" }
                for destination in destinations {
                    try mergeQuantization(&checkpoint, modules: [fused], destination: destination)
                }
                for parameter in ["weight", "scales", "biases"] {
                    let source = parameter == "weight" ? fused : "\(fused)_\(parameter)"
                    guard let array = checkpoint.weights.removeValue(forKey: source) else {
                        continue
                    }
                    guard array.ndim >= 2, array.dim(-2) == value.dim(-2) else {
                        throw LoadingError.incompleteExperts(source)
                    }
                    let mid = array.dim(-2) / 2
                    for (index, destination) in destinations.enumerated() {
                        let name = "\(destination).\(parameter)"
                        guard checkpoint.weights[name] == nil else {
                            throw LoadingError.ambiguousLayout
                        }
                        checkpoint.weights[name] =
                            index == 0
                            ? array[.ellipsis, ..<mid, 0...] : array[.ellipsis, mid..., 0...]
                    }
                }
                checkpoint.perLayerQuantization?.perLayerQuantization.removeValue(forKey: fused)
            }
            let down = "\(prefix).experts.down_proj"
            if checkpoint.weights[down] != nil {
                let destination = "\(prefix).switch_mlp.down_proj"
                try mergeQuantization(&checkpoint, modules: [down], destination: destination)
                for parameter in ["weight", "scales", "biases"] {
                    let source = parameter == "weight" ? down : "\(down)_\(parameter)"
                    guard let array = checkpoint.weights.removeValue(forKey: source) else {
                        continue
                    }
                    let name = "\(destination).\(parameter)"
                    guard checkpoint.weights[name] == nil else {
                        throw LoadingError.ambiguousLayout
                    }
                    checkpoint.weights[name] = array
                }
                checkpoint.perLayerQuantization?.perLayerQuantization.removeValue(forKey: down)
            }
        }
    }

    private func mergeQuantization(
        _ checkpoint: inout ModelCheckpoint, modules: [String], destination: String
    ) throws {
        guard var settings = checkpoint.perLayerQuantization, let first = modules.first else {
            return
        }
        let quantization = settings.quantization(layer: first)
        guard modules.allSatisfy({ settings.quantization(layer: $0) == quantization }) else {
            throw LoadingError.conflictingExpertQuantization(destination)
        }
        guard settings.perLayerQuantization[destination] == nil else {
            throw LoadingError.ambiguousLayout
        }
        settings.perLayerQuantization[destination] =
            quantization.map {
                .quantize($0)
            } ?? .skip
        checkpoint.perLayerQuantization = settings
    }

    package enum LoadingError: LocalizedError {
        case ambiguousLayout
        case unsupportedNormConvention(String)
        case conflictingNormConventions
        case incompleteExperts(String)
        case conflictingExpertQuantization(String)

        package var errorDescription: String? {
            switch self {
            case .ambiguousLayout: "Checkpoint contains competing Qwen MTP layouts or aliases."
            case .unsupportedNormConvention(let value):
                "Unsupported Qwen MTP norm convention '\(value)'."
            case .conflictingNormConventions:
                "Qwen MTP norm tensors declare conflicting conventions."
            case .incompleteExperts(let name):
                "Incomplete or incompatible Qwen MTP experts at '\(name)'."
            case .conflictingExpertQuantization(let name):
                "Qwen MTP experts at '\(name)' have different quantization settings."
            }
        }
    }
}
