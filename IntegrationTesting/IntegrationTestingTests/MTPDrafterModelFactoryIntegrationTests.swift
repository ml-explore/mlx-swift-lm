// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Factory load (gated on checkpoint presence)

@Test
func testMTPDrafterFactoryLoadFromDirectoryWhenCheckpointPresent() async throws {
    // Look for the 31B-assistant-bf16 checkpoint in the HF cache; skip
    // gracefully when absent.
    guard
        let snapshot = hfSnapshotDir(
            modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
    else {
        Issue.record("31B-assistant-bf16 checkpoint not in HF cache; skipping factory load test")
        return
    }

    await Gemma4AssistantRegistration.register()
    let factory = MTPDrafterModelFactory.shared

    let container = try await factory.loadContainer(
        from: snapshot, using: NoOpTokenizerLoader()
    )
    let isDrafter = await container.perform { ctx in
        ctx.model is Gemma4AssistantDraftModel
    }
    #expect(isDrafter)
}

// MARK: - Helpers

/// A `TokenizerLoader` placeholder. `MTPDrafterModelFactory` ignores the
/// loader (drafters borrow their target's tokenizer), but the protocol
/// requires a non-optional argument.
private final class NoOpTokenizerLoader: TokenizerLoader {
    func load(from url: URL) async throws -> any Tokenizer {
        fatalError("MTPDrafterModelFactory must not call tokenizer load")
    }
}
