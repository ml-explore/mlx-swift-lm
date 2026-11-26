// Copyright © 2026 Apple Inc.

import MLXLMCommon

@available(*, deprecated, message: "MTPDrafterTypeRegistry includes Gemma4AssistantDraftModel.")
public enum Gemma4AssistantRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.register(Gemma4AssistantDraftModel.self)
    }
}
