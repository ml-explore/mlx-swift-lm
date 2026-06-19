// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    /// The adapter is the only place that can enforce `.vision` for labeled
    /// image attachments (the SDK's own vision guard inspects only SPI `.image`
    /// segments, which this design never touches). The gate must fire before any
    /// weight download, so this test runs with no model on disk.
    @Suite("MLXLanguageModel vision capability gate")
    struct VisionCapabilityGateTests {

        @Test("Image input without .vision throws unsupportedCapability(.vision)")
        func imageWithoutVisionThrows() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let model = makeStubModel(
                "vision/not-declared",
                capabilities: LanguageModelCapabilities(capabilities: []))
            let executor = try makeMLXExecutor(for: model)

            let attachment = Transcript.AttachmentSegment(
                content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
                label: "photo")
            let prompt = Transcript.Prompt(
                segments: [
                    .text(Transcript.TextSegment(content: "Describe this")),
                    .attachment(attachment),
                ],
                responseFormat: nil
            )
            let request = makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(prompt)]))
            let channel = LanguageModelExecutorGenerationChannel()

            do {
                try await executor.respond(
                    to: request, model: model, streamingInto: channel)
                Issue.record("Expected unsupportedCapability(.vision), but respond returned")
            } catch let error as LanguageModelError {
                guard case .unsupportedCapability(let unsupported) = error else {
                    Issue.record("Expected unsupportedCapability, got \(error)")
                    return
                }
                #expect(unsupported.capability == .vision)
            }
        }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
