// Copyright © 2026 Apple Inc.

import MLXLMCommon

@available(*, deprecated, message: "MTPDrafterTypeRegistry includes LFM2DSparkDraftModel.")
public enum LFM2DSparkRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.register(LFM2DSparkDraftModel.self)
    }
}
