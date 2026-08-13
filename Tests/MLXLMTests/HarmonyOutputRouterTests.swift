// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Router policy: analysis → private reasoning, final → response, allowlisted
/// commentary → tool call, one call per turn, strict JSON, no undeclared names.
struct HarmonyOutputRouterTests {

    @Test("final payload streams as response; analysis remains private reasoning")
    func finalVisibleAnalysisHidden() throws {
        let events = try route(
            [
                "<|channel|>", "analysis", "<|message|>", "private", "<|end|>",
                "<|channel|>", "final", "<|message|>", "public", "<|return|>",
            ],
            allowed: ["search"])

        #expect(events.compactMap(\.response).joined() == "public")
        #expect(events.compactMap(\.reasoning).joined() == "private")
        #expect(events.compactMap(\.toolCall).isEmpty)
        #expect(!events.compactMap(\.response).joined().contains("private"))
        #expect(!events.compactMap(\.response).joined().contains("<|"))
    }

    @Test("allowlisted commentary function becomes a single tool call")
    func allowlistedToolCall() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=functions.get_weather",
                "<|message|>", #"{"city":"Paris"}"#, "<|call|>",
            ],
            allowed: ["get_weather", "get_time"])

        let calls = events.compactMap(\.toolCall)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["city"] == .string("Paris"))
        #expect(events.compactMap(\.response).joined().isEmpty)
    }

    @Test("undeclared tool names are rejected")
    func undeclaredRejected() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=functions.secret",
                "<|message|>", #"{"x":1}"#, "<|call|>",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("bare recipients are not treated as function-namespace calls")
    func bareRecipientRejected() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=get_weather",
                "<|message|>", #"{"city":"Paris"}"#, "<|call|>",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("analysis recipient is never dispatched")
    func analysisRecipientIgnored() throws {
        let events = try route(
            [
                "<|channel|>", "analysis to=functions.get_weather",
                "<|message|>", #"{"city":"X"}"#, "<|end|>",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.toolCall).isEmpty)
        #expect(events.compactMap(\.response).joined().isEmpty)
        #expect(events.compactMap(\.reasoning).joined() == #"{"city":"X"}"#)
    }

    @Test("second tool call in the same turn is ignored")
    func oneCallPerTurn() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=functions.get_weather",
                "<|message|>", #"{"city":"Paris"}"#, "<|call|>",
                "<|channel|>", "commentary to=functions.get_time",
                "<|message|>", #"{"tz":"UTC"}"#, "<|call|>",
            ],
            allowed: ["get_weather", "get_time"])

        let calls = events.compactMap(\.toolCall)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
    }

    @Test("malformed argument JSON is rejected")
    func malformedJSONRejected() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=functions.get_weather",
                "<|message|>", "{not json", "<|call|>",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("tool-shaped frames require the call commit token")
    func uncommittedCallRejected() throws {
        for terminator in ["<|end|>", "<|return|>"] {
            let events = try route(
                [
                    "<|channel|>", "commentary to=functions.get_weather",
                    "<|message|>", #"{"city":"Paris"}"#, terminator,
                ],
                allowed: ["get_weather"])
            #expect(events.compactMap(\.toolCall).isEmpty)
        }

        let truncated = try route(
            [
                "<|channel|>", "commentary to=functions.get_weather",
                "<|message|>", #"{"city":"Paris"}"#,
            ],
            allowed: ["get_weather"])
        #expect(truncated.compactMap(\.toolCall).isEmpty)
    }

    @Test("code-fenced JSON is not silently unwrapped")
    func fencedJSONRejected() throws {
        let events = try route(
            [
                "<|channel|>", "commentary to=functions.get_weather",
                "<|message|>", "```json\n{\"city\":\"Boston\"}\n```", "<|call|>",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("recipient-less commentary is user-visible response text")
    func commentaryPreambleVisible() throws {
        let events = try route(
            [
                "<|channel|>", "commentary", "<|message|>", "plan steps", "<|end|>",
                "<|channel|>", "final", "<|message|>", "done",
            ],
            allowed: ["get_weather"])

        #expect(events.compactMap(\.response).joined() == "plan stepsdone")
    }
}

// MARK: - Helpers

extension HarmonyOutputRouter.Event {
    fileprivate var reasoning: String? {
        if case .reasoning(let text) = self { return text }
        return nil
    }
    fileprivate var response: String? {
        if case .response(let text) = self { return text }
        return nil
    }
    fileprivate var toolCall: ToolCall? {
        if case .toolCall(let call) = self { return call }
        return nil
    }
}

private func route(_ pieces: [String], allowed: Set<String>) throws -> [HarmonyOutputRouter.Event] {
    let tokenizer = RouterDeterministicTokenizer(tokens: pieces + routerControlTokens)
    var parser = try #require(HarmonyFrameParser(tokenizer: tokenizer))
    var router = HarmonyOutputRouter(tokenizer: tokenizer, allowedToolNames: allowed)
    var events: [HarmonyOutputRouter.Event] = []
    for piece in pieces {
        let id = try #require(tokenizer.convertTokenToId(piece))
        for step in parser.push(id) {
            events.append(contentsOf: router.route(step))
        }
    }
    for step in parser.finish() {
        events.append(contentsOf: router.route(step))
    }
    events.append(contentsOf: router.finish())
    return events
}

private let routerControlTokens = [
    "<|start|>", "<|channel|>", "<|message|>", "<|end|>",
    "<|call|>", "<|return|>", "<|constrain|>",
]

private struct RouterDeterministicTokenizer: Tokenizer {
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
