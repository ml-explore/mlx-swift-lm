// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers the `gemma4_assistant` model type with the shared MTP drafter
/// type registry.
///
/// Callers must invoke this once before loading a drafter (typically at app
/// launch). Re-registration is idempotent — it overwrites the prior creator
/// with the same one.
///
/// Async because `ModelTypeRegistry` is an `actor` — registration calls
/// require `await`. A bootstrap pattern like the synchronous
/// `LLMTypeRegistry.shared = .init(creators: [...])` is not possible here:
/// the drafter implementation (`Gemma4AssistantDraftModel`) lives in
/// MLXVLM, and importing it into MLXLMCommon's `MTPDrafterTypeRegistry.shared`
/// would form a circular dependency.
///
/// Usage:
///
/// ```swift
/// await Gemma4AssistantRegistration.register()
/// // ... later, load a drafter via MTPDrafterModelFactory.shared ...
/// ```
public enum Gemma4AssistantRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "gemma4_assistant",
            creator: { data in
                let config = try JSONDecoder().decode(
                    Gemma4AssistantConfiguration.self, from: data)
                return Gemma4AssistantDraftModel(config)
            }
        )
    }
}
