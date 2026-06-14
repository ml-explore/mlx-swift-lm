// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5/Qwen3.6 VLM MTP drafter model types.
///
/// Callers should invoke this once before loading a Qwen VLM drafter through
/// ``MTPDrafterModelFactory``. The `"qwen3_5"` names overlap with the text
/// registration; config-shape predicates keep both registrations usable in
/// the same process.
public enum Qwen35VLMMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5",
            matches: qwen35VLMMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35VLMNextNDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_moe",
            matches: qwen35VLMMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35VLMNextNDraftModel(config)
            }
        )
    }
}

private func qwen35VLMMTPConfiguration(_ data: Data) -> Bool {
    guard let shape = try? JSONDecoder.json5().decode(Qwen35MTPConfigurationShape.self, from: data)
    else {
        return false
    }
    return shape.visionConfig != nil
}

private struct Qwen35MTPConfigurationShape: Decodable {
    var visionConfig: VisionConfig?

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
    }

    struct VisionConfig: Decodable {}
}
