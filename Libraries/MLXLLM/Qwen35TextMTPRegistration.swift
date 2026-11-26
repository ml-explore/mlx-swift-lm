// Copyright © 2026 Apple Inc.

import MLXLMCommon

@available(*, deprecated, message: "MTPDrafterTypeRegistry includes Qwen35MTPDraftModel.")
public enum Qwen35TextMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.register(Qwen35MTPDraftModel.self)
    }
}
