// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5/Qwen3.6 text MTP drafter model types.
///
/// Callers should invoke this once before loading a Qwen text drafter through
/// `MTPDrafterModelFactory`.
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
            "qwen3_5_mtp",
            matches: qwen35TextMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(config, preconvertedNorms: true)
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
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let vision = root["vision_config"] as? [String: Any]
    else { return true }
    return vision.isEmpty
}
