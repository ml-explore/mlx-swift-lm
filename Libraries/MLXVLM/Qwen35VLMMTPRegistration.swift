// Copyright © 2026 Apple Inc.

import MLXLMCommon

@available(*, deprecated, message: "MTPDrafterTypeRegistry includes Qwen35VLMNextNDraftModel.")
public enum Qwen35VLMMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.register(Qwen35VLMNextNDraftModel.self)
    }
}
