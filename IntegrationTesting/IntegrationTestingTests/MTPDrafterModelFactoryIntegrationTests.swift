// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Factory load (auto-downloads the checkpoint if not cached)

/// Pinned checkpoint revision matching the weights that were live when the
/// Rung 4 `drafter_block` fixtures were generated. Kept in sync with
/// `MTPRung4TokenParityTests`.
private let drafter31BRevision = "28e92270316e89288579ec59c17939541d9ca433"

extension ModelLoadingSuite {
    @Suite struct MTPDrafterModelFactoryIntegrationTests {}
}

extension ModelLoadingSuite.MTPDrafterModelFactoryIntegrationTests {
    @Test
    func testMTPDrafterFactoryLoadFromDirectoryWhenCheckpointPresent() async throws {
        await Gemma4AssistantRegistration.register()
        let factory = MTPDrafterModelFactory.shared

        let configuration = ModelConfiguration(
            id: "mlx-community/gemma-4-31B-it-assistant-bf16", revision: drafter31BRevision)
        let container = try await TestModelManager.shared.load(
            TestModelManager.key("mtp-drafter-container", configuration)
        ) {
            try await factory.loadContainer(
                from: #hubDownloader(),
                using: NoOpTokenizerLoader(),
                configuration: configuration
            )
        }
        let isDrafter = await container.perform { ctx in
            ctx.model is Gemma4AssistantDraftModel
        }
        #expect(isDrafter)
    }
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
