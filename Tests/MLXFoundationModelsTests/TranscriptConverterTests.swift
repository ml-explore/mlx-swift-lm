// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct StructuredWeatherToolResult {
    var temperature: Int
    var condition: String
}

@Suite
struct TranscriptConverterTests {

    @Test
    func testConvertInstructionsToSystemMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let instructions = Transcript.Instructions(
            segments: [
                .text(Transcript.TextSegment(content: "You are a helpful assistant."))
            ],
            toolDefinitions: []
        )

        let entries: [Transcript.Entry] = [.instructions(instructions)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .system)
        #expect(message.content == "You are a helpful assistant.")
    }

    @Test
    func testConvertPromptToUserMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Hello!"))
            ],
            responseFormat: nil
        )

        let entries: [Transcript.Entry] = [.prompt(prompt)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .user)
        #expect(message.content == "Hello!")
    }

    @Test
    func testConvertResponseToAssistantMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let response = Transcript.Response(
            assetIDs: [],
            segments: [
                .text(Transcript.TextSegment(content: "Hi there!"))
            ]
        )

        let entries: [Transcript.Entry] = [.response(response)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .assistant)
        #expect(message.content == "Hi there!")
    }

    @Test
    func testMultipleSegmentsAreConcatenated() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Hello")),
                .text(Transcript.TextSegment(content: "world")),
            ],
            responseFormat: nil
        )

        let entries: [Transcript.Entry] = [.prompt(prompt)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .user)
        #expect(message.content == "Hello\nworld")
    }

    @Test
    func testMultiTurnConversation() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: "Be helpful"))],
                    toolDefinitions: []
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Hi"))],
                    responseFormat: nil
                )),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "Hello"))]
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "How are you?"))],
                    responseFormat: nil
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 4)
        #expect(messages[0].role == .system)
        #expect(messages[1].role == .user)
        #expect(messages[2].role == .assistant)
        #expect(messages[3].role == .user)
    }

    @Test
    func testEmptyTranscriptReturnsEmptyArray() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = []
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.isEmpty)
    }

    @Test
    func testUnsupportedEntryTypesAreSkipped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Only supported types here. toolCalls/toolOutput are now mapped (see
        // the tool-calling tests below); genuinely unknown future entry types
        // still hit the `default:` tripwire and are skipped.
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Test"))],
                    responseFormat: nil
                ))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)
        #expect(messages.count == 1)
    }

    @Test
    func testToolCallsBecomeAssistantMessageWithCalls() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let toolCall = Transcript.ToolCall(
            id: "call_1",
            toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location":"Paris"}"#))
        let entries: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls(id: "tc_1", [toolCall]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .assistant)

        // The structured calls are fileprivate on Chat.Message.Tool; assert via
        // the observable serialization the chat template consumes.
        let raw = DefaultMessageGenerator().generate(message: messages[0])
        let calls = try #require(raw["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 1)
        #expect(calls[0]["id"] as? String == "call_1")
        let function = try #require(calls[0]["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "get_weather")
        let arguments = try #require(function["arguments"] as? [String: any Sendable])
        #expect(arguments["location"] as? String == "Paris")
    }

    @Test
    func testToolOutputBecomesToolMessageCorrelatedByID() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let output = Transcript.ToolOutput(
            id: "call_1",
            toolName: "get_weather",
            segments: [.text(Transcript.TextSegment(content: "18C and sunny"))])
        let entries: [Transcript.Entry] = [.toolOutput(output)]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .tool)
        #expect(messages[0].content == "18C and sunny")

        // ToolOutput.id is the originating call id, so the emitted tool message
        // carries a tool_call_id that the template correlates to the call.
        let raw = DefaultMessageGenerator().generate(message: messages[0])
        #expect(raw["tool_call_id"] as? String == "call_1")
    }

    @Test
    func testStructuredToolOutputPreservesEveryGeneratedContentJSONShape() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let cases: [(id: String, content: GeneratedContent)] = [
            (
                id: "generable",
                content: StructuredWeatherToolResult(
                    temperature: 18,
                    condition: "sunny"
                ).generatedContent
            ),
            (id: "array", content: try GeneratedContent(json: #"["Paris","Tokyo"]"#)),
            (id: "scalar", content: try GeneratedContent(json: #"18"#)),
        ]

        for testCase in cases {
            let output = Transcript.ToolOutput(
                id: "call_\(testCase.id)",
                toolName: "get_weather",
                segments: [
                    .structure(
                        Transcript.StructuredSegment(
                            schemaName: "WeatherResult",
                            content: testCase.content))
                ])

            let messages = TranscriptConverter.mlxMessages(for: [.toolOutput(output)])

            #expect(messages.count == 1)
            let message = try #require(messages.first)
            #expect(message.role == .tool)
            #expect(message.content == testCase.content.jsonString)

            let raw = DefaultMessageGenerator().generate(message: message)
            #expect(raw["tool_call_id"] as? String == "call_\(testCase.id)")
        }
    }

    @Test
    func testMixedTextAndStructuredToolOutputPreservesSegmentOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let content = try GeneratedContent(
            json: #"{"temperature":18,"condition":"sunny"}"#)
        let output = Transcript.ToolOutput(
            id: "call_mixed",
            toolName: "get_weather",
            segments: [
                .text(Transcript.TextSegment(content: "Weather result:")),
                .structure(
                    Transcript.StructuredSegment(
                        schemaName: "WeatherResult",
                        content: content)),
                .text(Transcript.TextSegment(content: "Use this current reading.")),
            ])

        let messages = TranscriptConverter.mlxMessages(for: [.toolOutput(output)])

        let message = try #require(messages.first)
        #expect(
            message.content
                == [
                    "Weather result:",
                    content.jsonString,
                    "Use this current reading.",
                ].joined(separator: "\n"))

        let raw = DefaultMessageGenerator().generate(message: message)
        #expect(raw["tool_call_id"] as? String == "call_mixed")
    }

    @Test
    func testToolCallAndOutputFormCorrelatedContinuation() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let toolCall = Transcript.ToolCall(
            id: "call_1",
            toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location":"Paris"}"#))
        let output = Transcript.ToolOutput(
            id: "call_1",
            toolName: "get_weather",
            segments: [.text(Transcript.TextSegment(content: "18C and sunny"))])
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Weather in Paris?"))],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls(id: "tc_1", [toolCall])),
            .toolOutput(output),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.map(\.role) == [.user, .assistant, .tool])

        let raw = DefaultMessageGenerator().generate(messages: messages)
        let calls = try #require(raw[1]["tool_calls"] as? [[String: any Sendable]])
        #expect(calls[0]["id"] as? String == "call_1")
        #expect(raw[2]["tool_call_id"] as? String == "call_1")
    }

    @Test
    func testTwoToolRoundsProduceCorrelatedMessages() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let callA = Transcript.ToolCall(
            id: "call_a",
            toolName: "get_location",
            arguments: try GeneratedContent(json: #"{}"#))
        let outputA = Transcript.ToolOutput(
            id: "call_a",
            toolName: "get_location",
            segments: [.text(Transcript.TextSegment(content: "Paris"))])
        let callB = Transcript.ToolCall(
            id: "call_b",
            toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location":"Paris"}"#))
        let outputB = Transcript.ToolOutput(
            id: "call_b",
            toolName: "get_weather",
            segments: [.text(Transcript.TextSegment(content: "18C and sunny"))])
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "What should I wear in Paris?"))
                    ],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls(id: "tc_a", [callA])),
            .toolOutput(outputA),
            .toolCalls(Transcript.ToolCalls(id: "tc_b", [callB])),
            .toolOutput(outputB),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        // Both rounds replay in order: each assistant tool-call message is
        // immediately followed by its correlated tool result, so a third-round
        // invocation sees the full history.
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .assistant, .tool])

        let raw = DefaultMessageGenerator().generate(messages: messages)
        let callsA = try #require(raw[1]["tool_calls"] as? [[String: any Sendable]])
        #expect(callsA[0]["id"] as? String == "call_a")
        #expect(raw[2]["tool_call_id"] as? String == "call_a")
        #expect(messages[2].content == "Paris")

        let callsB = try #require(raw[3]["tool_calls"] as? [[String: any Sendable]])
        #expect(callsB[0]["id"] as? String == "call_b")
        #expect(raw[4]["tool_call_id"] as? String == "call_b")
        #expect(messages[4].content == "18C and sunny")
    }

    @Test
    func testReasoningEntryIsDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // A prior turn that contains reasoning between the prompt and response.
        // The reasoning must not be replayed into the chat history.
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "What is 2+2?"))],
                    responseFormat: nil
                )),
            .reasoning(
                Transcript.Reasoning(
                    segments: [
                        .text(Transcript.TextSegment(content: "Let me add: 2 plus 2 is 4."))
                    ]
                )),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "4"))]
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 2)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "What is 2+2?")
        #expect(messages[1].role == .assistant)
        #expect(messages[1].content == "4")
    }

    @Test
    func testMultipleReasoningEntriesAllDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = [
            .reasoning(
                Transcript.Reasoning(
                    segments: [.text(Transcript.TextSegment(content: "first thought"))]
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Hi"))],
                    responseFormat: nil
                )),
            .reasoning(
                Transcript.Reasoning(
                    segments: [.text(Transcript.TextSegment(content: "second thought"))]
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Hi")
    }

    @Test
    func testLabeledImageAttachmentBecomesUserMessageImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

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

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Describe this")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testImageOnlyPromptStillProducesMessageWithImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [.attachment(attachment)],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testInstructionsImageAttachmentBecomesSystemMessageImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

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

        let messages = TranscriptConverter.mlxMessages(for: [.instructions(instructions)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .system)
        #expect(messages[0].content == "Use this reference:")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testUrlBackedImageAttachmentYieldsDecodedCIImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let url = try makeSolidImageFileURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(imageURL: url)),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [.attachment(attachment)],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].images.count == 1)
        // The SDK eagerly decodes a URL-backed attachment at construction, so
        // the converter hands over the in-memory CIImage rather than the URL —
        // passing the URL would force a redundant, failure-prone re-decode.
        guard let image = messages[0].images.first else {
            Issue.record("Expected one image input")
            return
        }
        guard case .ciImage = image else {
            Issue.record("URL-backed attachment should yield .ciImage, not .url")
            return
        }
    }

    @Test
    func testMultipleImageAttachmentsPreserveCountAndOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Two distinguishable images (different dimensions) so order is checkable.
        let first = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage(width: 2, height: 2))),
            label: "first")
        let second = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage(width: 4, height: 4))),
            label: "second")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Compare these")),
                .attachment(first),
                .attachment(second),
            ],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].images.count == 2)
        // Segment order is preserved: the 2x2 image precedes the 4x4 image.
        let widths = messages[0].images.compactMap { image -> CGFloat? in
            guard case .ciImage(let ciImage) = image else { return nil }
            return ciImage.extent.width
        }
        #expect(widths == [2, 4])
    }

    @Test
    func testImageBeforeTextStillConcatenatesText() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Attachment position must not perturb text concatenation.
        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [
                .attachment(attachment),
                .text(Transcript.TextSegment(content: "line one")),
                .text(Transcript.TextSegment(content: "line two")),
            ],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].content == "line one\nline two")
        #expect(messages[0].images.count == 1)
    }

}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
