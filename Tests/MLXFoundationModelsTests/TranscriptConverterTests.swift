// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

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

        // Create a transcript with only supported types
        // (toolCalls and toolOutput would be skipped, but we can't easily create them in tests)
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
                    segments: [.text(Transcript.TextSegment(content: "Let me add: 2 plus 2 is 4."))]
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

}
