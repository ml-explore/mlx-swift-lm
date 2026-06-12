// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 audio inference. Downloads an audio-capable Gemma 4
// VLM and asks it to transcribe real speech clips, exercising the full audio
// path: AVAssetReader PCM -> mel feature extractor -> Conformer audio tower ->
// begin/end-of-audio prompt splice -> text.
//
// Speech clips are committed under Tests/MLXLMTests/Resources/.
//
// Run:
//   xcodebuild test -project IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4AudioIntegrationTests

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

private let resources = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("Tests/MLXLMTests/Resources")
    .path

struct SpeechCase: Sendable, CustomStringConvertible {
    let file: String
    let expected: [String]
    var description: String { file }
}

private let speechCases: [SpeechCase] = [
    .init(
        file: "gemma_speech_test.wav",
        expected: ["quick", "brown", "fox", "lazy", "dog", "river"]),
    .init(
        file: "gemma_speech_long.wav",
        expected: ["weather", "rain", "forecast", "afternoon", "breeze", "evening", "sky"]),
]

@Suite(.serialized)
struct Gemma4AudioIntegrationTests {

    private func transcribe(model: String, clip: SpeechCase) async throws -> String {
        let container = try await models.vlmContainer(for: ModelConfiguration(id: model))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 120, temperature: 0))
        let url = URL(fileURLWithPath: "\(resources)/\(clip.file)")
        return try await session.respond(
            to: "Transcribe the speech in this audio clip.",
            images: [], videos: [], audios: [.url(url)])
    }

    private func assertRecovered(_ answer: String, _ clip: SpeechCase) {
        let lower = answer.lowercased()
        #expect(!lower.contains("<pad>"), "audio path regressed to a <pad> wall: \(answer)")
        let hits = clip.expected.filter { lower.contains($0) }
        #expect(
            hits.count >= 3,
            "[\(clip.file)] did not recover the spoken words (matched \(hits) in: \(answer))")
    }

    @Test(arguments: speechCases)
    func gemma4_e4b_transcribes(_ clip: SpeechCase) async throws {
        let answer = try await transcribe(model: "mlx-community/gemma-4-e4b-it-4bit", clip: clip)
        print("[e4b/\(clip.file)] \(answer)")
        assertRecovered(answer, clip)
    }

    @Test func gemma4_e4b_perceivesRealSpeech() async throws {
        let clip = SpeechCase(file: "gemma_audio_librispeech.wav", expected: [])
        let answer = try await transcribe(model: "mlx-community/gemma-4-e4b-it-4bit", clip: clip)
        print("[e4b/librispeech] \(answer)")
        let lower = answer.lowercased()
        #expect(!lower.contains("<pad>"), "audio path regressed to a <pad> wall")
        #expect(
            !lower.contains("not provided") && !lower.contains("no audio")
                && !lower.contains("haven't provided") && !lower.contains("have not provided"),
            "model claims no audio; audio not reaching the model: \(answer)")
        let contentWords = ["middle", "class", "welcome", "mr", "mister", "gospel", "apostle"]
        let hits = contentWords.filter { lower.contains($0) }
        #expect(
            hits.count >= 1,
            "did not perceive the real-speech content (matched \(hits) in: \(answer))")
    }
}
