// Copyright © 2026 Apple Inc.

import CoreImage
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

@Suite(.serialized)
struct Qwen3VLCacheIntegrationTests {
    @Test func imageContinuationAfterCacheRotation() async throws {
        let modelID = IntegrationTestModelIDs.vlm
        let configuration =
            hfSnapshotDir(modelId: modelID).map {
                ModelConfiguration(directory: $0)
            } ?? ModelConfiguration(id: modelID)
        let container = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: configuration)
        let parameters = GenerateParameters(
            maxTokens: 8,
            kvCache: KVCacheConfiguration(
                capacity: try .init(maxTokens: 32, preservedPrefixTokens: 4)),
            temperature: 0, prefill: .init(stepSize: 16))
        let session = ChatSession(container, generateParameters: parameters)

        _ = try await session.respond(to: String(repeating: "One two three. ", count: 32))
        let response = try await session.respond(
            to: "What color is this image?", image: .ciImage(VisionTestImages.solidColor(.red)))
        #expect(!response.isEmpty)

        let status = try await session.cacheStatus()
        #expect(status.capacityDisposition == .fullyApplied)
        #expect((status.processedTokenCount ?? 0) > 32)
        #expect(status.attentionMaxSizes.allSatisfy { $0 == 32 })
    }
}
