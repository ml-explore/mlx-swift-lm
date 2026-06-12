// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 VIDEO inference. Downloads gemma-4-e4b-it-4bit and
// asks it to describe a real video clip, exercising the PR #256 video tower
// end to end (frame sampling via MediaProcessing → vision tower → text).
//
// Run:
//   xcodebuild test -project IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4VideoIntegrationTests

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

// Resolve Tests/MLXLMTests/Resources relative to this source file so the suite
// runs on any clone / CI checkout, not just one developer's machine.
private let videoResources = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()  // IntegrationTestingTests/
    .deletingLastPathComponent()  // IntegrationTesting/
    .deletingLastPathComponent()  // repo root
    .appendingPathComponent("Tests/MLXLMTests/Resources")

@Suite(.serialized)
struct Gemma4VideoIntegrationTests {

    // The repo already ships a small real clip for VLM tests.
    private static let videoURL = videoResources.appendingPathComponent("1080p_30.mov")

    @Test func gemma4_e4b_describesVideo() async throws {
        let container = try await models.vlmContainer(
            for: ModelConfiguration(id: "mlx-community/gemma-4-e4b-it-4bit")
        )

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 120, temperature: 0)
        )

        let answer = try await session.respond(
            to: "Describe what happens in this video in one or two sentences.",
            images: [],
            videos: [.url(Self.videoURL)],
            audios: []
        )

        print("🎬 Gemma 4 video description:\n\(answer)")

        let lower = answer.lowercased()
        // Reject the degenerate <pad>/special-token wall failure mode.
        #expect(!lower.contains("<pad>"), "description is a <pad> wall — video tower not producing usable embeddings")
        // Must be a substantive natural-language description, not a token fragment.
        let wordCount = answer.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
        #expect(wordCount >= 8, "description too short to be a real video understanding: \(answer)")
        // The clip is a sequence of solid colour blocks; a correct description
        // should reference colour/blocks/frames. Require at least one such cue.
        let visualCues = ["color", "colour", "block", "frame", "screen", "background", "blue", "green", "yellow", "magenta", "red"]
        #expect(
            visualCues.contains(where: { lower.contains($0) }),
            "description lacks any visual cue from the clip: \(answer)"
        )
    }

    // Big Buck Bunny (Blender Foundation, CC-BY-3.0) — a real animated outdoor
    // scene. See Resources/FIXTURES_LICENSES.md.
    private static let bbbURL = videoResources.appendingPathComponent("gemma_video_bbb.mp4")

    @Test func gemma4_e4b_describesVideo_bbb() async throws {
        let container = try await models.vlmContainer(
            for: ModelConfiguration(id: "mlx-community/gemma-4-e4b-it-4bit"))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 150, temperature: 0))

        let answer = try await session.respond(
            to: "Describe what happens in this video in one or two sentences.",
            images: [], videos: [.url(Self.bbbURL)], audios: [])

        print("🎬 Gemma 4 BBB description:\n\(answer)")
        let lower = answer.lowercased()
        #expect(!lower.contains("<pad>"), "BBB description is a <pad> wall")
        #expect(
            answer.split(whereSeparator: { $0 == " " || $0 == "\n" }).count >= 8,
            "BBB description too short: \(answer)")
        // Animated outdoor nature scene (rabbit/animal, grass/trees, sky, cartoon).
        let cues = [
            "rabbit", "bunny", "animal", "creature", "animat", "cartoon", "character",
            "grass", "tree", "forest", "field", "nature", "sky", "green", "outdoor",
        ]
        #expect(
            cues.contains(where: { lower.contains($0) }),
            "BBB description lacks any scene cue: \(answer)")
    }
}
