// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5/Qwen3.6 text MTP drafter model types.
///
/// Callers should invoke this once before loading a Qwen text drafter through
/// ``MTPDrafterModelFactory``.
public enum Qwen35TextMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_text",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35TextConfiguration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5",
            matches: qwen35TextMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_moe",
            matches: qwen35TextMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
    }
}

private func qwen35TextMTPConfiguration(_ data: Data) -> Bool {
    guard let shape = try? JSONDecoder.json5().decode(Qwen35MTPConfigurationShape.self, from: data)
    else {
        return true
    }
    return shape.visionConfig == nil
}

private struct Qwen35MTPConfigurationShape: Decodable {
    var visionConfig: VisionConfig?

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
    }

    struct VisionConfig: Decodable {}
}
