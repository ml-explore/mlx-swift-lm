// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

private struct SpecialTokenStubTokenizer: MLXLMCommon.Tokenizer {
    let specials: [String]

    private static let scalarBase = 1_000_000

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while let scalar = rest.unicodeScalars.first {
            if let index = specials.firstIndex(where: { rest.hasPrefix($0) }) {
                ids.append(index)
                rest = rest.dropFirst(specials[index].count)
            } else {
                ids.append(Self.scalarBase + Int(scalar.value))
                rest = rest.dropFirst()
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { id -> String? in
            if specials.indices.contains(id) {
                return skipSpecialTokens ? nil : specials[id]
            }
            guard let scalar = Unicode.Scalar(UInt32(id - Self.scalarBase)) else { return nil }
            return String(Character(scalar))
        }
        .joined()
    }

    func convertTokenToId(_ token: String) -> Int? { specials.firstIndex(of: token) }

    func convertIdToToken(_ id: Int) -> String? {
        specials.indices.contains(id) ? specials[id] : nil
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private let mistralTokenizer = SpecialTokenStubTokenizer(specials: ["[IMG]", "[INST]"])

private func image(_ name: String, label: String?) -> UserInput.Image {
    .url(URL(fileURLWithPath: "/tmp/\(name).png"), label: label)
}

@Suite("Special-token label screen")
struct SpecialTokenLabelScreenTests {

    @Test("A label whose bracketed form is a special token is removed, and its image stays")
    func specialTokenLabelsAreRemoved() {
        let input = UserInput(chat: [
            .user(
                "which one?",
                images: [
                    image("a", label: "IMG"), image("b", label: "receipt"),
                    image("c", label: "INST"), image("d", label: "IMG2"),
                ])
        ])

        let screened = input.removingSpecialTokenLabels(using: mistralTokenizer)

        #expect(screened.images.map(\.label) == [nil, "receipt", nil, "IMG2"])
    }

    @Test("A text prompt comes back unchanged")
    func textPromptIsUnchanged() {
        // `UserInput(prompt:images:)` builds a `.chat` prompt, so set `.text` directly.
        var input = UserInput(prompt: "hello", images: [image("a", label: "IMG")])
        input.prompt = .text("hello")

        let screened = input.removingSpecialTokenLabels(using: mistralTokenizer)

        #expect(screened.images.map(\.label) == ["IMG"])
    }

    @Test("After the screen, the Mistral3 generator writes no [IMG] text part")
    func screenedLabelNeverReachesThePrompt() throws {
        let input = UserInput(chat: [
            .user("what is in this picture?", images: [image("a", label: "IMG")])
        ])

        let messages = Mistral3MessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: mistralTokenizer))

        let content = try #require(messages.first?["content"] as? [[String: String]])
        #expect(content.filter { $0["type"] == "image" }.count == 1)
        #expect(!content.contains { $0["text"] == "[IMG]" })
    }

    @Test("A configured generator gets screened labels too")
    func configuredGeneratorPathIsScreened() async throws {
        let delegate = CapturingUserInputProcessor()
        let processor = MessageGeneratorUserInputProcessor(
            processor: delegate, messageGenerator: Mistral3MessageGenerator(),
            tokenizer: mistralTokenizer)

        _ = try await processor.prepare(
            input: UserInput(chat: [
                .user("what is in this picture?", images: [image("a", label: "IMG")])
            ]))

        guard case .messages(let messages) = delegate.prompt else {
            Issue.record("expected generated messages, got \(String(describing: delegate.prompt))")
            return
        }
        let content = try #require(messages.first?["content"] as? [[String: String]])
        #expect(content.filter { $0["type"] == "image" }.count == 1)
        #expect(!content.contains { $0["text"] == "[IMG]" })
    }
}

/// `@unchecked Sendable` is safe only while `lock` guards every access to `storedPrompt`.
private final class CapturingUserInputProcessor: UserInputProcessor, @unchecked Sendable {
    private let lock = NSLock()
    private var storedPrompt: UserInput.Prompt?

    var prompt: UserInput.Prompt? {
        lock.withLock { storedPrompt }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let prompt = input.prompt
        lock.withLock { storedPrompt = prompt }
        return LMInput(tokens: MLXArray([Int32(0)]))
    }
}
