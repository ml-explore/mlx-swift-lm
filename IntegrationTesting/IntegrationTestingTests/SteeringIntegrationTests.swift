// Copyright © 2026 Apple Inc.

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

@Suite(.serialized)
struct SteeringIntegrationTests {
    @Test(
        .timeLimit(.minutes(2)),
        arguments: [
            SteeringPolicy.nextSafeBoundary, .nextStepBoundary,
        ])
    func steersRunningResponse(policy: SteeringPolicy) async throws {
        let container = try await models.vlmContainer(
            for: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit"))
        let configuration = await container.configuration
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 128, temperature: 0))
        let stream = session.streamDetails(
            to: "List 50 facts about the ocean. Write one sentence per fact.")
        var accepted: UUID?
        var applied: [UUID] = []
        var infos: [GenerateCompletionInfo] = []
        var successorText = ""
        for try await event in stream {
            switch event {
            case .chunk(let text):
                if !applied.isEmpty { successorText += text }
                if accepted == nil {
                    accepted = try session.steer(
                        "Stop the list. Reply only with STEERING_CONFIRMED.", policy: policy)
                }
            case .steering(.applied(let ids)):
                applied += ids
            case .steering(.failed(let failure)):
                Issue.record("Steering failed: \(failure.reason)")
            case .info(let info):
                infos.append(info)
            case .toolCall, .rejectedToolCall:
                Issue.record("Unexpected tool output")
            }
        }
        let id = try #require(accepted)
        #expect(applied == [id])
        #expect(infos.count == 2)
        if case .nextSafeBoundary = policy, configuration.reasoningConfig == nil {
            #expect(infos.first?.stopReason == .steered)
        } else {
            #expect(infos.first?.stopReason != .steered)
        }
        #expect(successorText.contains("STEERING_CONFIRMED"))
        #expect(!session.canSteer())

        let followup = try await session.respond(to: "What word did I ask you to reply with?")
        #expect(followup.contains("STEERING_CONFIRMED"))
    }
}
