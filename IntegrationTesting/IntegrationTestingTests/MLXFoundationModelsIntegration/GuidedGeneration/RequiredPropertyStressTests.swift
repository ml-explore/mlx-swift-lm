// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing
import FoundationModels

@testable import MLXFoundationModels

// Types from the bug report: a top-level object with three required
// properties, one of which is a nested object.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct NPC: Equatable {
    let name: String
    let coffeeOrder: String
    let picture: Picture
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Picture: Equatable {
    @Guide(description: "Avoid human-like descriptions. Stick to animals, plants, or objects.")
    let imageDescription: String
}

/// End-to-end regression for the bug where guided generation dropped a required
/// object property. A Gemma-4 (SentencePiece) checkpoint could emit a token
/// whose decode was not append-only (a space collapsing before a comma), and
/// the streaming detokenizer's character-count diff silently dropped the comma,
/// producing invalid JSON that failed `Generable` decoding. See the fix in
/// `NaiveStreamingDetokenizer.next()` and the model-free regression in
/// `StreamingDetokenizerTests`.
///
/// Heavyweight (downloads a Gemma-4 checkpoint and runs many generations), so
/// gated behind `MLX_RUN_REQUIRED_PROPERTY_STRESS=1` — it must not run on every
/// integration pass. Mirrors the `MLX_RUN_VLM_INTEGRATION` / `MLX_RUN_COLD_FETCH`
/// gates elsewhere in this target.
@Suite(
    .serialized,
    .timeLimit(.minutes(30)),
    .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_REQUIRED_PROPERTY_STRESS"] == "1")
)
struct RequiredPropertyStressTests {

    private static let instructions = """
        A conversation between a user and a helpful assistant. This is a fantasy RPG game that \
        takes place at Dream Coffee, the beloved coffee shop of the dream realm. Your role is to \
        use your imagination to generate fun game characters.
        """

    private static let prompt = """
        Create an NPC customer with a fun personality suitable for the dream realm. Have the \
        customer order coffee. Here are some examples to inspire you:
        {name: "Thimblefoot", imageDescription: "A horse with a rainbow mane", coffeeOrder: "I \
        would like a coffee that's refreshing and sweet like grass of a summer meadow"}
        {name: "Spiderkid", imageDescription: "A furry spider with a cool baseball cap", \
        coffeeOrder: "An iced coffee please, that's as spooky as me!"}
        """

    @Test
    func gemma4DoesNotDropRequiredProperties() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.gemma4ModelID, capabilities: [.guidedGeneration])

        let iterations = 25
        var failures = 0

        for i in 1 ... iterations {
            let session = LanguageModelSession(model: model) { Self.instructions }
            do {
                let npc = try await session.respond(to: Self.prompt, generating: NPC.self).content
                print("[\(i)] OK: \(npc.name)")
            } catch {
                failures += 1
                print("[\(i)] FAIL: \(error)")
            }
        }

        print("Failures: \(failures)/\(iterations)")
        await releaseAllGPUMemory()
        #expect(failures == 0, "\(failures)/\(iterations) generations dropped a required property")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
