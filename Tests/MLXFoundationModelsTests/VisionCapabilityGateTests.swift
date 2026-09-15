// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// The adapter is the only place that can enforce `.vision` for labeled
/// image attachments, because the SDK's own vision guard doesn't inspect
/// these public attachment segments, only its own internal image path. The
/// gate covers prompt attachments: instructions attachments are dropped
/// during conversion and never reach it. The gate must fire before any
/// weight download, so these tests run with no model on disk.
@Suite("MLXLanguageModel vision capability gate")
struct VisionCapabilityGateTests {

    @Test("Image input without .vision throws unsupportedCapability(.vision)")
    func imageWithoutVisionThrows() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeStubModel(
            "vision/not-declared",
            capabilities: [])
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
            return
        } catch let error as LanguageModelError {
            guard case .unsupportedCapability(let unsupported) = error else {
                Issue.record("Expected unsupportedCapability, got \(error)")
                return
            }
            #expect(unsupported.capability == .vision)
        } catch {
            Issue.record(
                "Expected LanguageModelError.unsupportedCapability(.vision), got: \(error)")
        }
    }

    /// An image carried only on the instructions is dropped during conversion,
    /// so it never reaches the gate and a model without `.vision` must not be
    /// rejected for it. This is deliberate: it matches FoundationModels, which
    /// also ignores images attached to a session's instructions. The request
    /// still fails, but for the unrelated reason that this stub model has no
    /// weights on disk, which is proof the gate let it through.
    @Test("Instructions-only image does not trip the vision gate")
    func instructionsOnlyImageDoesNotTripTheGate() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeStubModel(
            "vision/instructions-only-image",
            capabilities: [])
        let executor = try makeMLXExecutor(for: model)

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "reference")
        let instructions = Transcript.Instructions(
            segments: [
                .text(Transcript.TextSegment(content: "Use this reference:")),
                .attachment(attachment),
            ],
            toolDefinitions: []
        )
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Hello"))],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [.instructions(instructions), .prompt(prompt)]))
        let channel = LanguageModelExecutorGenerationChannel()

        do {
            try await executor.respond(
                to: request, model: model, streamingInto: channel)
            Issue.record("Expected the weight load to fail, but respond returned")
        } catch let error as LanguageModelError {
            // Any LanguageModelError here means the adapter rejected the
            // request itself rather than getting as far as the weights, and a
            // vision rejection specifically is the regression this guards.
            Issue.record(
                "Expected the missing weights to fail the request, got a rejection instead: \(error)"
            )
        } catch let error as ModelFactoryError {
            guard case .configurationFileError(let file, let modelName, _) = error else {
                Issue.record("Expected the missing config.json to fail the load, got \(error)")
                return
            }
            #expect(file == "config.json")
            #expect(modelName == "vision/instructions-only-image")
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
