// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5-family text MTP drafter model types.
///
/// Standalone Qwen MTP checkpoints do not identify the verifier architecture:
/// their `vision_config` is empty for both text and multimodal use. Text and
/// multimodal drafters therefore use separate registries instead of relying
/// on config-shape predicates or registration order.
public enum Qwen35TextMTPRegistration {
    public static func register() async {
        let registry = MTPDrafterTypeRegistry.shared

        for modelType in ["qwen3_5_text", "qwen3_8_text"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35TextConfiguration.self, from: data)
                    return Qwen35MTPDraftModel(config)
                })
        }

        for modelType in ["qwen3_5", "qwen3_5_moe", "qwen3_8", "qwen3_8_moe"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35Configuration.self, from: data)
                    return Qwen35MTPDraftModel(config)
                })
        }

        for modelType in ["qwen3_5_mtp", "qwen3_8_mtp"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35Configuration.self, from: data)
                    return Qwen35MTPDraftModel(config, preconvertedNorms: true)
                })
        }
    }
}
