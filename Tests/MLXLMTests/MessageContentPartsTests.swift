// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Message content parts")
struct MessageContentPartsTests {

    let generator = DefaultMessageGenerator()

    private func image(_ label: String?) -> UserInput.Image {
        .url(URL(fileURLWithPath: "/tmp/example.png"), label: label)
    }

    @Test("GLM-OCR puts images before the recognition prompt")
    func glmOcrImagesBeforeText() {
        let message = GlmOcrMessageGenerator().generate(
            message: .user("Text Recognition:", images: [image(nil), image(nil)]))
        #expect(message["role"] as? String == "user")
        #expect(
            message["content"] as? [[String: String]]
                == [
                    ["type": "image"],
                    ["type": "image"],
                    ["type": "text", "text": "Text Recognition:"],
                ])
    }

    @Test("No images gives one text part")
    func noImages() {
        let message = Chat.Message.user("hello")
        #expect(
            generator.contentParts(for: message, layout: .imagesThenVideosThenText)
                == [["type": "text", "text": "hello"]])
    }

    @Test("A name holding a marker character is left out, and its image stays")
    func markerNameIsLeftOut() {
        let message = Chat.Message.user("hello", images: [image("<|image|>"), image("B")])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [
                    ["type": "image"],
                    ["type": "text", "text": "[B]"],
                    ["type": "image"],
                    ["type": "text", "text": "hello"],
                ])
    }

    @Test("Every marker character costs the name, on any layout")
    func everyMarkerCharacterIsRefused() {
        for label in ["<a", "a>", "a|b", "[a", "a]"] {
            let message = Chat.Message.user("hello", images: [image(label)])
            for layout: MessageContentLayout in [
                .imagesThenVideosThenText, .imagesThenText, .textThenImages,
            ] {
                let parts = generator.contentParts(for: message, layout: layout)
                #expect(
                    !parts.contains(["type": "text", "text": "[\(label)]"]),
                    "\(label) reached the parts on \(layout)")
                #expect(parts.contains(["type": "image"]), "\(label) lost its image")
            }
        }
    }

    @Test("Unlabeled images give the array generators produced before labels")
    func unlabeledImagesAreUnchanged() {
        let message = Chat.Message.user("hello", images: [image(nil), image(nil)])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenVideosThenText)
                == [
                    ["type": "image"],
                    ["type": "image"],
                    ["type": "text", "text": "hello"],
                ])
        #expect(
            generator.contentParts(for: message, layout: .textThenImages)
                == [
                    ["type": "text", "text": "hello"],
                    ["type": "image"],
                    ["type": "image"],
                ])
    }

    @Test("One labeled image is named immediately before itself")
    func oneLabeledImage() {
        let message = Chat.Message.user("hello", images: [image("A")])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "text", "text": "hello"],
                ])
    }

    @Test("Each label sits directly before its own image, with nothing between")
    func threeLabeledImages() {
        let message = Chat.Message.user(
            "which one is blue?", images: [image("A"), image("B"), image("C")])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "text", "text": "[B]"],
                    ["type": "image"],
                    ["type": "text", "text": "[C]"],
                    ["type": "image"],
                    ["type": "text", "text": "which one is blue?"],
                ])
    }

    @Test("No part carries a separator, in any layout")
    func noSeparatorAnywhere() {
        let message = Chat.Message.user(
            "which one is blue?", images: [image("A"), image("B")])
        for layout: MessageContentLayout in [
            .imagesThenVideosThenText, .imagesThenText, .textThenImages,
        ] {
            for part in generator.contentParts(for: message, layout: layout) {
                let text = part["text"] ?? ""
                #expect(!text.contains("\u{2060}"), "word joiner in \(text) for \(layout)")
                #expect(!text.contains("\u{200B}"), "zero width space in \(text) for \(layout)")
                #expect(!text.contains("\n"), "newline in \(text) for \(layout)")
            }
        }
    }

    @Test("An unlabeled image among labeled ones is a bare image part")
    func mixedLabels() {
        let message = Chat.Message.user("hello", images: [image("A"), image(nil)])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "image"],
                    ["type": "text", "text": "hello"],
                ])
    }

    @Test("Text first puts the prose ahead of the labeled images")
    func textFirstLayout() {
        let message = Chat.Message.user("hello", images: [image("A"), image("B")])
        #expect(
            generator.contentParts(for: message, layout: .textThenImages)
                == [
                    ["type": "text", "text": "hello"],
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "text", "text": "[B]"],
                    ["type": "image"],
                ])
    }

    @Test("Video parts appear only in the layout that emits them")
    func videoParts() {
        let message = Chat.Message.user(
            "hello", images: [image("A")],
            videos: [.url(URL(fileURLWithPath: "/tmp/example.mov"))])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenVideosThenText)
                == [
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "video"],
                    ["type": "text", "text": "hello"],
                ])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [
                    ["type": "text", "text": "[A]"],
                    ["type": "image"],
                    ["type": "text", "text": "hello"],
                ])
    }

    @Test("Empty text still gives a text part")
    func emptyTextStillGivesAPart() {
        let message = Chat.Message.user("", images: [image(nil)])
        #expect(
            generator.contentParts(for: message, layout: .imagesThenText)
                == [["type": "image"], ["type": "text", "text": ""]])
    }

    @Test("Empty messages without tool calls keep a text part in every layout")
    func emptyMessagesWithoutToolCallsKeepText() {
        for role: Chat.Message.Role in [.user, .assistant, .system, .tool] {
            for tool: Chat.Message.Tool? in [nil, .calls([]), .result(id: "call_1")] {
                let message = Chat.Message(role: role, content: "", tool: tool)
                for layout: MessageContentLayout in [
                    .imagesThenVideosThenText, .imagesThenText, .textThenImages,
                ] {
                    #expect(
                        generator.contentParts(for: message, layout: layout)
                            == [["type": "text", "text": ""]])
                }
            }
        }
    }

    @Test("Only assistant tool calls omit empty text, and nonempty text is preserved")
    func toolCallTextByRole() {
        let call = ToolCall(function: .init(name: "get_weather", arguments: [:]))
        for role: Chat.Message.Role in [.user, .assistant, .system, .tool] {
            for text in ["", " ", "Checking the weather."] {
                let message = Chat.Message(role: role, content: text, tool: .calls([call]))
                for layout: MessageContentLayout in [
                    .imagesThenVideosThenText, .imagesThenText, .textThenImages,
                ] {
                    let expected =
                        role == .assistant && text.isEmpty
                        ? [] : [["type": "text", "text": text]]
                    #expect(generator.contentParts(for: message, layout: layout) == expected)
                }
            }
        }
    }

    @Test("Omitting tool-call text preserves image labels and media")
    func toolCallsKeepMedia() {
        let call = ToolCall(function: .init(name: "get_weather", arguments: [:]))
        let message = Chat.Message.assistant(
            "", images: [image("A"), image(nil)],
            videos: [.url(URL(fileURLWithPath: "/tmp/example.mov"))], toolCalls: [call])
        for layout: MessageContentLayout in [
            .imagesThenVideosThenText, .imagesThenText, .textThenImages,
        ] {
            var expected = [
                ["type": "text", "text": "[A]"], ["type": "image"], ["type": "image"],
            ]
            if case .imagesThenVideosThenText = layout {
                expected.append(["type": "video"])
            }
            #expect(generator.contentParts(for: message, layout: layout) == expected)
        }
    }

    @Test("Qwen reasoning with no answer omits text only when tool calls are present")
    func reasoningOnlyAssistantContent() {
        let call = ToolCall(function: .init(name: "get_weather", arguments: [:]))
        for calls: [ToolCall]? in [nil, [], [call]] {
            let message = Qwen3VLMessageGenerator().generate(
                message: .assistant("<think>Check the weather.</think>", toolCalls: calls))
            #expect(message["reasoning_content"] as? String == "Check the weather.")
            let expected = calls?.isEmpty == false ? [] : [["type": "text", "text": ""]]
            #expect(message["content"] as? [[String: String]] == expected)
        }
    }

    @Test("Content-parts generators omit empty text only for actual tool calls")
    func toolCallWithoutTextHasEmptyContent() {
        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": .string("Rome")]))
        let generators: [any MessageGenerator] = [
            Qwen2VLMessageGenerator(), Qwen3VLMessageGenerator(), Gemma4MessageGenerator(),
            Mistral3MessageGenerator(), FastVLMMessageGenerator(), GlmOcrMessageGenerator(),
        ]
        for generator in generators {
            for input: Chat.Message in [.user(""), .assistant(""), .assistant("", toolCalls: [])] {
                let message = generator.generate(message: input)
                #expect(
                    (message["content"] as? [[String: String]])
                        == [["type": "text", "text": ""]],
                    "\(type(of: generator)) changed an empty message without tool calls")
            }
            let message = generator.generate(message: .assistant("", toolCalls: [call]))
            #expect(
                (message["content"] as? [[String: String]])?.isEmpty == true,
                "\(type(of: generator)) gave \(String(describing: message["content"]))")
            #expect(message["tool_calls"] != nil, "\(type(of: generator)) lost the tool call")
        }
    }
}
