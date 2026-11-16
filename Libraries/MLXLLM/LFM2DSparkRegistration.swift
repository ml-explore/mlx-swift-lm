// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Liquid AI's `Lfm2DSparkDraftModel` checkpoints with the shared
/// block-drafter factory.
public enum LFM2DSparkRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3",
            matches: { data in
                guard
                    let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                    let architectures = object["architectures"] as? [String]
                else { return false }
                return architectures.contains("Lfm2DSparkDraftModel")
            },
            creator: { data in
                let configuration = try JSONDecoder.json5().decode(
                    LFM2DSparkConfiguration.self, from: data)
                try configuration.validateModelConfiguration()
                return LFM2DSparkDraftModel(configuration)
            })
    }
}
