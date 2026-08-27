// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// The helper that builds a message's content parts. A labeled image is named
/// immediately before itself, and a message with no labels produces the array
/// generators produced before labels existed.
@Suite("Message content parts")
struct MessageContentPartsTests {

    let generator = DefaultMessageGenerator()

    private func image(_ label: String?) -> UserInput.Image {
        .url(URL(fileURLWithPath: "/tmp/example.png"), label: label)
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
                == [
                    ["type": "image"],
                    ["type": "text", "text": ""],
                ])
    }
}
