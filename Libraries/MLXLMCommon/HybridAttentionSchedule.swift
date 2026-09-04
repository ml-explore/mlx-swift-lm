// Copyright © 2026 Apple Inc.

import Foundation

/// A validated hybrid decoder schedule shared by text and multimodal models.
///
/// Explicit `layer_types` metadata is authoritative. When it is absent, the
/// legacy interval pattern is expanded once so downstream model code does not
/// need to maintain two scheduling paths.
package struct HybridAttentionSchedule: Sendable, Equatable {
    package static let linearAttention = "linear_attention"
    package static let fullAttention = "full_attention"

    package let layerTypes: [String]

    package init(
        hiddenLayerCount: Int,
        fullAttentionInterval: Int,
        explicitLayerTypes: [String]?
    ) throws {
        guard hiddenLayerCount >= 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "num_hidden_layers must be non-negative, got \(hiddenLayerCount)")
        }

        if let explicitLayerTypes {
            guard explicitLayerTypes.count == hiddenLayerCount else {
                throw ModelFactoryError.invalidConfiguration(
                    "layer_types contains \(explicitLayerTypes.count) entries, but "
                        + "num_hidden_layers is \(hiddenLayerCount)")
            }

            let supported = [Self.linearAttention, Self.fullAttention]
            guard let unsupported = explicitLayerTypes.first(where: { !supported.contains($0) })
            else {
                self.layerTypes = explicitLayerTypes
                return
            }
            throw ModelFactoryError.invalidConfiguration(
                "unsupported Qwen hybrid layer type '\(unsupported)'")
        }

        guard fullAttentionInterval > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "full_attention_interval must be positive, got \(fullAttentionInterval)")
        }
        self.layerTypes = (0 ..< hiddenLayerCount).map { index in
            (index + 1).isMultiple(of: fullAttentionInterval)
                ? Self.fullAttention : Self.linearAttention
        }
    }
}
