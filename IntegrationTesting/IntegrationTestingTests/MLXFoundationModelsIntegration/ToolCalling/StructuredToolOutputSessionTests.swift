// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import MLX
import Testing

@testable import MLXFoundationModels

private let structuredToolOutputSentinel = "fm-structured-output-9D7E-4821"

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension SessionPropertyValues {
    @SessionPropertyEntry
    var structuredToolOutputCallCount: Int = 0
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct StructuredLookupArguments {
    @Guide(description: "The record identifier to retrieve.")
    var recordID: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct StructuredLookupResult {
    var requiredToken: String
    var summary: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct StructuredLookupTool: Tool {
    let name = "lookup_structured_record"
    let description =
        "Retrieves a record whose structured result contains the exact token for the response."

    func call(arguments: StructuredLookupArguments) async throws -> StructuredLookupResult {
        StructuredLookupResult(
            requiredToken: structuredToolOutputSentinel,
            summary: "Retrieved record \(arguments.recordID).")
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct StructuredToolOutputProfile: LanguageModelSession.DynamicProfile {
    let model: MLXLanguageModel

    @SessionProperty(\.structuredToolOutputCallCount)
    var toolCallCount

    var body: some LanguageModelSession.DynamicProfile {
        if toolCallCount == 0 {
            Profile {
                Instructions {
                    "Call the lookup tool once. After it returns, answer with the value of its requiredToken field exactly."
                }
                StructuredLookupTool()
            }
            .model(model)
            .toolCallingMode(.required)
            .onToolCall {
                toolCallCount += 1
            }
        } else {
            Profile {
                Instructions {
                    "Use the latest tool output. Return its requiredToken field exactly and no other text."
                }
            }
            .model(model)
            .toolCallingMode(.disallowed)
        }
    }
}

/// Opt-in live Foundation Models tool round trip. The two model rows load
/// multi-GB checkpoints, so default sweeps skip this suite. Set
/// `MLX_RUN_FM_TOOL_INTEGRATION=1` on macOS, or
/// `TEST_RUNNER_MLX_RUN_FM_TOOL_INTEGRATION=1` for an iOS device test runner.
@Suite(
    .serialized,
    .timeLimit(.minutes(10)),
    .enabled(
        if: ProcessInfo.processInfo.environment[
            "MLX_RUN_FM_TOOL_INTEGRATION"
        ] == "1")
)
struct StructuredToolOutputSessionTests {
    static let models = [TestFixtures.gemma4ModelID, TestFixtures.qwen3ModelID]

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeSessionModel(_ modelID: String) -> MLXLanguageModel {
        if modelID == TestFixtures.qwen3ModelID {
            return makeReasoningTestModel(modelID)
        }
        return makeTestModel(modelID)
    }

    @Test("Setup: release GPU state before live structured-output sessions")
    func clearGPUBeforeStructuredSessions() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()
    }

    @Test(arguments: StructuredToolOutputSessionTests.models)
    func languageModelSessionUsesGenerableToolOutput(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeSessionModel(modelID)
        let session = LanguageModelSession(
            profile: StructuredToolOutputProfile(model: model))

        let response = try await session.respond(
            to: "Retrieve record alpha and follow the profile instructions.")

        #expect(session.properties.structuredToolOutputCallCount == 1)

        let outputs = session.transcript.compactMap { entry -> Transcript.ToolOutput? in
            guard case .toolOutput(let output) = entry else { return nil }
            return output
        }
        #expect(outputs.count == 1)

        let structuredSegments = outputs.flatMap { $0.segments }.compactMap {
            segment -> Transcript.StructuredSegment? in
            guard case .structure(let structured) = segment else { return nil }
            return structured
        }
        let structured = try #require(structuredSegments.first)
        #expect(structured.content.jsonString.contains(structuredToolOutputSentinel))

        #expect(
            response.content.contains(structuredToolOutputSentinel),
            "The live \(modelID) response must use the token available only in the structured tool output. Got: \(response.content)"
        )
    }
}

#endif  // FoundationModelsIntegration
