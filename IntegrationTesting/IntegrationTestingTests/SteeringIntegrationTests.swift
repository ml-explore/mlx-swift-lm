// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLXLMCommon
import Testing

extension ModelLoadingSuite {
    @Suite(.serialized) struct SteeringIntegrationTests {}
}

extension ModelLoadingSuite.SteeringIntegrationTests {
    @Test(
        .timeLimit(.minutes(2)),
        arguments: [
            SteeringPolicy.nextSafeBoundary, .nextStepBoundary,
        ])
    func steersRunningResponse(policy: SteeringPolicy) async throws {
        let container = try await models.vlmContainer(
            for: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit"))
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
            case .reasoning:
                break
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
        if case .nextSafeBoundary = policy {
            #expect(infos.first?.stopReason == .steered)
        } else {
            #expect(infos.first?.stopReason != .steered)
        }
        #expect(successorText.contains("STEERING_CONFIRMED"))
        #expect(!session.canSteer())

        let followup = try await session.respond(to: "What word did I ask you to reply with?")
        #expect(followup.contains("STEERING_CONFIRMED"))
    }

    @Test(.timeLimit(.minutes(2)))
    func interruptKeepsTheResponseForTheNextTurn() async throws {
        let container = try await models.vlmContainer(
            for: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit"))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 128, temperature: 0))
        let stream = session.streamDetails(
            to: "List 50 facts about the ocean. Write one sentence per fact.")
        var interrupted = false
        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            if event.chunk != nil, !interrupted {
                try session.interrupt()
                interrupted = true
            }
            if let info = event.info { infos.append(info) }
        }
        let info = try #require(infos.last)
        #expect(infos.map(\.stopReason) == [.interrupted])

        var followup = ""
        var followupInfo: GenerateCompletionInfo?
        for try await event in session.streamDetails(
            to: "Which topic did I ask you about? Reply with one word.")
        {
            followup += event.chunk ?? ""
            followupInfo = event.info ?? followupInfo
        }
        #expect(followup.lowercased().contains("ocean"))
        #expect(
            followupInfo?.cachedPromptTokenCount
                == info.promptTokenCount + info.generationTokenCount)
    }
}
