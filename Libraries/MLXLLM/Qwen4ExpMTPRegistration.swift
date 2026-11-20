// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers the Qwen3.8-Flash-Next MTP head under its checkpoint's `qwen4_exp` model type.
public enum Qwen4ExpMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen4_exp",
            creator: { data in
                try Qwen4ExpMTPDraftModel(
                    JSONDecoder.json5().decode(Qwen4ExpConfiguration.self, from: data))
            })
    }
}
