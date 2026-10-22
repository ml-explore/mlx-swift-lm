// Copyright © 2026 Apple Inc.

import Foundation
import MLXScriptedLM
import Testing

struct TranscriptTests {

    static let json = #"""
        {
          "turns": [
            { "system": "Be brief." },
            { "user": "hello" },
            { "assistant": "hi there" },
            { "user": "count" },
            { "assistant": { "text": "one two three", "terminator": "none" } }
          ]
        }
        """#

    static let transcript = Transcript([
        .system("Be brief."),
        .user("hello"),
        .assistant("hi there"),
        .user("count"),
        .assistant(.init("one two three", terminator: .none)),
    ])

    @Test func `decodes from JSON`() throws {
        let decoded = try JSONDecoder().decode(Transcript.self, from: Data(Self.json.utf8))
        #expect(decoded == Self.transcript)
    }

    @Test func `round-trips through JSON`() throws {
        let data = try JSONEncoder().encode(Self.transcript)
        #expect(try JSONDecoder().decode(Transcript.self, from: data) == Self.transcript)

        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#"{"assistant":"hi there"}"#))
    }

    @Test func `a turn with two roles fails to decode`() {
        let json = #"{ "turns": [ { "user": "a", "system": "b" } ] }"#
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(Transcript.self, from: Data(json.utf8))
        }
    }

    @Test func `scripts end with EOS unless the terminator is none`() throws {
        let tokenizer = ScriptedTokenizer(transcript: Self.transcript)
        let scripts = Self.transcript.scripts(tokenizer: tokenizer)
        let eos = try #require(tokenizer.convertTokenToId("<|end|>"))

        #expect(scripts.count == 2)
        #expect(scripts[0] == tokenizer.encode(text: "hi there", addSpecialTokens: false) + [eos])
        #expect(scripts[1] == tokenizer.encode(text: "one two three", addSpecialTokens: false))
    }

    @Test func `the turn marker is the template's generation prompt`() throws {
        let tokenizer = ScriptedTokenizer(transcript: Self.transcript, template: ChatMLTemplate())
        let scripts = try TurnScripts(transcript: Self.transcript, tokenizer: tokenizer)
        #expect(
            tokenizer.decode(tokenIds: scripts.turnMarker, skipSpecialTokens: false)
                == "<|im_start|>assistant\n")
    }
}
