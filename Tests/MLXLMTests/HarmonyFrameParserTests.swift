// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Pure parser tests: fragmented headers, channel transitions, recipients, terminators.
struct HarmonyFrameParserTests {

    @Test("Parses analysis then final frames with explicit terminators")
    func analysisThenFinal() throws {
        let pieces = [
            "<|channel|>", "analysis", "<|message|>", "think", "<|end|>",
            "<|channel|>", "final", "<|message|>", "answer", "<|return|>",
        ]
        let frames = try collectFrames(pieces)

        #expect(frames.count == 2)
        #expect(frames[0].header.channel == .analysis)
        #expect(frames[0].terminator == .end)
        #expect(decode(frames[0].payloadTokenIds, pieces: pieces) == "think")
        #expect(frames[1].header.channel == .final)
        #expect(frames[1].terminator == .return)
        #expect(decode(frames[1].payloadTokenIds, pieces: pieces) == "answer")
    }

    @Test("Parses commentary tool recipient and constrain")
    func commentaryRecipient() throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather ",
            "<|constrain|>", "json",
            "<|message|>", #"{"city":"Paris"}"#, "<|call|>",
        ]
        let frames = try collectFrames(pieces)
        #expect(frames.count == 1)
        #expect(frames[0].header.channel == .commentary)
        #expect(frames[0].header.recipient == "functions.get_weather")
        #expect(frames[0].header.constrain == "json")
        #expect(frames[0].terminator == .call)
    }

    @Test("Parses role-header form: start assistant to=… channel commentary")
    func roleHeaderRecipient() throws {
        let pieces = [
            "<|start|>", "assistant to=functions.search",
            "<|channel|>", "commentary",
            "<|message|>", #"{"q":"x"}"#, "<|call|>",
        ]
        let frames = try collectFrames(pieces)
        #expect(frames.count == 1)
        #expect(frames[0].header.channel == .commentary)
        #expect(frames[0].header.recipient == "functions.search")
    }

    @Test("Requires to= to be a standalone header attribute")
    func recipientAttributeBoundary() throws {
        for malformed in [
            "commentary not_to=functions.erase",
            "commentary proto=functions.erase",
            "commentary xxto=functions.erase",
        ] {
            let frames = try collectFrames([
                "<|channel|>", malformed, "<|message|>", #"{"confirmed":true}"#, "<|call|>",
            ])

            #expect(frames.count == 1)
            #expect(frames[0].header.recipient == nil)
        }
    }

    @Test("Closes an open tool frame at the next channel header")
    func closesOnNextHeader() throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.a", "<|message|>", #"{"x":1}"#,
            "<|channel|>", "final", "<|message|>", "hi",
        ]
        let frames = try collectFrames(pieces)
        #expect(frames.count == 2)
        #expect(frames[0].header.channel == .commentary)
        #expect(frames[0].terminator == .incomplete)
        #expect(frames[1].header.channel == .final)
        #expect(frames[1].terminator == .incomplete)
    }

    @Test("Literal control text inside a payload is not a frame boundary")
    func literalMarkersInPayload() throws {
        // The payload is one atomic token string that contains the letters of
        // a control tag; only real control *token ids* are boundaries.
        let pieces = [
            "<|channel|>", "final", "<|message|>",
            "see <|call|> later",
            "<|end|>",
        ]
        let frames = try collectFrames(pieces)
        #expect(frames.count == 1)
        #expect(decode(frames[0].payloadTokenIds, pieces: pieces) == "see <|call|> later")
    }

    @Test("finish() flushes an open body as incomplete")
    func finishFlushesBody() throws {
        let pieces = [
            "<|channel|>", "final", "<|message|>", "partial",
        ]
        let tokenizer = DeterministicTokenizer(tokens: pieces + harmonyControlTokens)
        var parser = try #require(HarmonyFrameParser(tokenizer: tokenizer))
        for piece in pieces {
            let id = try #require(tokenizer.convertTokenToId(piece))
            _ = parser.push(id)
        }
        let steps = parser.finish()
        let frames = steps.compactMap { step -> HarmonyFrame? in
            if case .closed(let frame) = step { return frame }
            return nil
        }
        #expect(frames.count == 1)
        #expect(frames[0].terminator == .incomplete)
    }

    @Test("Requires the complete Harmony control-token set")
    func requiresCompleteControlTokens() {
        let missingCall = harmonyControlTokens.filter { $0 != "<|call|>" }
        let tokenizer = DeterministicTokenizer(tokens: missingCall)
        #expect(HarmonyFrameParser(tokenizer: tokenizer) == nil)
    }

    @Test("Resolves the Harmony call and return stop tokens")
    func resolvesStopTokens() throws {
        let tokenizer = DeterministicTokenizer(tokens: harmonyControlTokens)
        let ids = try #require(HarmonyFrameParser.stopTokenIDs(tokenizer: tokenizer))
        #expect(
            ids
                == Set(
                    [
                        tokenizer.convertTokenToId("<|call|>"),
                        tokenizer.convertTokenToId("<|return|>"),
                    ].compactMap { $0 }))
    }
}

// MARK: - Helpers

private func collectFrames(_ pieces: [String]) throws -> [HarmonyFrame] {
    let tokenizer = DeterministicTokenizer(tokens: pieces + harmonyControlTokens)
    var parser = try #require(HarmonyFrameParser(tokenizer: tokenizer))
    var frames: [HarmonyFrame] = []
    for piece in pieces {
        let id = try #require(tokenizer.convertTokenToId(piece))
        for step in parser.push(id) {
            if case .closed(let frame) = step {
                frames.append(frame)
            }
        }
    }
    for step in parser.finish() {
        if case .closed(let frame) = step {
            frames.append(frame)
        }
    }
    return frames
}

private func decode(_ ids: [Int], pieces: [String]) -> String {
    let tokenizer = DeterministicTokenizer(tokens: pieces + harmonyControlTokens)
    return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
}

private let harmonyControlTokens = [
    "<|start|>", "<|channel|>", "<|message|>", "<|end|>",
    "<|call|>", "<|return|>", "<|constrain|>",
]

private struct DeterministicTokenizer: Tokenizer {
    private let tokenToID: [String: Int]
    private let idToToken: [Int: String]

    init(tokens: [String]) {
        var tokenToID: [String: Int] = [:]
        var idToToken: [Int: String] = [:]
        var nextID = 0
        for token in tokens where tokenToID[token] == nil {
            tokenToID[token] = nextID
            idToToken[nextID] = token
            nextID += 1
        }
        self.tokenToID = tokenToID
        self.idToToken = idToToken
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenToID[text].map { [$0] } ?? []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToToken[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { tokenToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToToken[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
