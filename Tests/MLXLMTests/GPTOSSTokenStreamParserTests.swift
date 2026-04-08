import Foundation
@testable import MLXLMCommon
import Testing
import Tokenizers

struct GPTOSSTokenStreamParserTests {
    @Test("GPT-OSS token parser preserves visible Harmony frames while extracting tool calls")
    func testGPTOSSTokenParserPreservesVisibleFrames() async throws {
        let pieces = [
            "<|channel|>", "analysis", "<|message|>", "Let me think.", "<|end|>",
            "<|channel|>", "commentary to=functions.search", "<|message|>",
            #"{"query":"swift"}"#, "<|call|>",
            "<|channel|>", "final", "<|message|>", "Done.",
        ]
        let events = try await collectEvents(for: pieces)

        let visibleText = events.compactMap(\.chunk).joined()
        let toolCalls = events.compactMap(\.toolCall)

        #expect(
            visibleText
                == "<|channel|>analysis<|message|>Let me think.<|end|><|channel|>final<|message|>Done."
        )
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "search")
        #expect(toolCalls[0].function.arguments["query"] == .string("swift"))
    }

    @Test("GPT-OSS token parser ignores literal Harmony markers inside JSON strings")
    func testGPTOSSTokenParserHandlesLiteralHarmonyMarkers() async throws {
        let literal = "literal <|call|> and <|channel|> text"
        let pieces = [
            "<|channel|>", "commentary to=functions.search", "<|message|>",
            #"{"query":"\#(literal)"}"#, "<|call|>",
        ]
        let events = try await collectEvents(for: pieces)

        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(events.compactMap(\.toolCall).count == 1)
        #expect(events.compactMap(\.toolCall).first?.function.name == "search")
        #expect(events.compactMap(\.toolCall).first?.function.arguments["query"] == .string(literal))
    }

    @Test("GPT-OSS token parser finalizes a tool call at the next Harmony header")
    func testGPTOSSTokenParserFinalizesOnNextHeader() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather", "<|message|>",
            #"{"location":"Paris"}"#,
            "<|channel|>", "final", "<|message|>", "Sunny.",
        ]
        let events = try await collectEvents(for: pieces)

        let visibleText = events.compactMap(\.chunk).joined()
        let toolCalls = events.compactMap(\.toolCall)

        #expect(visibleText == "<|channel|>final<|message|>Sunny.")
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Paris"))
    }

    @Test("GPT-OSS token parser finalizes a role-header tool call on EOS")
    func testGPTOSSTokenParserFinalizesRoleHeaderCallOnEOS() async throws {
        let pieces = [
            "<|start|>", "assistant to=functions.get_weather", "<|channel|>", "commentary",
            "<|message|>", #"{"location":"Naples"}"#,
        ]
        let events = try await collectEvents(for: pieces)

        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(events.compactMap(\.toolCall).count == 1)
        #expect(events.compactMap(\.toolCall).first?.function.name == "get_weather")
        #expect(events.compactMap(\.toolCall).first?.function.arguments["location"] == .string("Naples"))
    }

    @Test("GPT-OSS token parser finalizes a tool call on return boundary")
    func testGPTOSSTokenParserFinalizesOnReturnBoundary() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_time", "<|message|>",
            #"{"timezone":"UTC"}"#, "<|return|>",
            "<|channel|>", "final", "<|message|>", "Done.",
        ]
        let events = try await collectEvents(for: pieces)

        let visibleText = events.compactMap(\.chunk).joined()
        let toolCalls = events.compactMap(\.toolCall)

        #expect(visibleText == "<|channel|>final<|message|>Done.")
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_time")
        #expect(toolCalls[0].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("GPT-OSS token parser finalizes a tool call on end boundary")
    func testGPTOSSTokenParserFinalizesOnEndBoundary() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather", "<|message|>",
            #"{"location":"Berlin"}"#, "<|end|>",
            "<|channel|>", "final", "<|message|>", "Rainy.",
        ]
        let events = try await collectEvents(for: pieces)

        let visibleText = events.compactMap(\.chunk).joined()
        let toolCalls = events.compactMap(\.toolCall)

        #expect(visibleText == "<|channel|>final<|message|>Rainy.")
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Berlin"))
    }

    @Test("GPT-OSS token parser extracts multiple back-to-back tool calls")
    func testGPTOSSTokenParserExtractsMultipleToolCalls() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather", "<|message|>",
            #"{"location":"Paris"}"#, "<|call|>",
            "<|channel|>", "commentary to=functions.get_time", "<|message|>",
            #"{"timezone":"UTC"}"#, "<|call|>",
        ]
        let events = try await collectEvents(for: pieces)

        let toolCalls = events.compactMap(\.toolCall)

        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls[1].function.name == "get_time")
        #expect(toolCalls[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("GPT-OSS token parser keeps non-tool commentary visible")
    func testGPTOSSTokenParserPreservesCommentaryWithoutRecipient() async throws {
        let pieces = [
            "<|channel|>", "commentary", "<|message|>", "Let me think out loud.", "<|end|>",
        ]
        let events = try await collectEvents(for: pieces)

        #expect(
            events.compactMap(\.chunk).joined()
                == "<|channel|>commentary<|message|>Let me think out loud.<|end|>"
        )
        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("GPT-OSS token parser ignores non-commentary recipients")
    func testGPTOSSTokenParserIgnoresNonCommentaryRecipients() async throws {
        let pieces = [
            "<|channel|>", "analysis to=functions.search", "<|message|>", #"{"query":"swift"}"#,
            "<|end|>",
        ]
        let events = try await collectEvents(for: pieces)

        #expect(
            events.compactMap(\.chunk).joined()
                == #"<|channel|>analysis to=functions.search<|message|>{"query":"swift"}<|end|>"#
        )
        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    @Test("GPT-OSS token parser handles constrain tags and nested arguments objects")
    func testGPTOSSTokenParserHandlesConstrainAndNestedArguments() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather ", "<|constrain|>", "json",
            "<|message|>", #"{"arguments":{"location":"Austin"}}"#, "<|call|>",
        ]
        let events = try await collectEvents(for: pieces)

        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(events.compactMap(\.toolCall).count == 1)
        #expect(events.compactMap(\.toolCall).first?.function.name == "get_weather")
        #expect(events.compactMap(\.toolCall).first?.function.arguments["location"] == .string("Austin"))
    }

    @Test("GPT-OSS token parser handles fenced JSON payloads")
    func testGPTOSSTokenParserHandlesFencedJSON() async throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.get_weather", "<|message|>",
            "```json\n{\"location\":\"Boston\"}\n```", "<|call|>",
        ]
        let events = try await collectEvents(for: pieces)

        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(events.compactMap(\.toolCall).count == 1)
        #expect(events.compactMap(\.toolCall).first?.function.name == "get_weather")
        #expect(events.compactMap(\.toolCall).first?.function.arguments["location"] == .string("Boston"))
    }
}

private enum GPTOSSTokenStreamParserTestError: Error {
    case missingToken(String)
    case parserUnavailable
}

private func collectEvents(for pieces: [String]) async throws -> [Generation] {
    let tokenizer = DeterministicTokenizer(tokens: pieces + harmonyControlTokens)
    guard var parser = GPTOSSTokenStreamParser(tokenizer: tokenizer) else {
        throw GPTOSSTokenStreamParserTestError.parserUnavailable
    }

    let (stream, continuation) = AsyncStream<Generation>.makeStream()
    for piece in pieces {
        guard let token = tokenizer.convertTokenToId(piece) else {
            throw GPTOSSTokenStreamParserTestError.missingToken(piece)
        }
        _ = parser.onToken(token, emit: continuation.yield)
    }
    parser.onGenerationEnd(emit: continuation.yield)
    continuation.finish()

    var events: [Generation] = []
    for await event in stream {
        events.append(event)
    }
    return events
}

private let harmonyControlTokens = [
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|call|>",
    "<|return|>",
    "<|constrain|>",
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

    func tokenize(text: String) -> [String] {
        [text]
    }

    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        guard let tokenID = tokenToID[text] else { return [] }
        return [tokenID]
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.compactMap { idToToken[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenToID[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        idToToken[id]
    }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        []
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
