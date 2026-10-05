// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM
import Testing

struct PseudoWordTokenizerTests {

    @Test func `ids always fit the vocabulary`() throws {
        let tokenizer = PseudoWordTokenizer(vocabularySize: 100)
        let texts = FuzzText.strings(count: 200, seed: 5) + FuzzText.outsideFragments

        for text in texts {
            #expect(tokenizer.encode(text: text).allSatisfy { (0 ..< 100).contains($0) })
        }
        let prompt = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": texts.joined(separator: " ")]
        ])
        #expect(!prompt.isEmpty)
        #expect(prompt.allSatisfy { (0 ..< 100).contains($0) })
    }

    @Test func `encoding is deterministic`() {
        let text = FuzzText.strings(count: 20, seed: 6).joined(separator: " ")
        #expect(
            PseudoWordTokenizer().encode(text: text, addSpecialTokens: false)
                == PseudoWordTokenizer().encode(text: text, addSpecialTokens: false))
    }

    @Test func `model output re-encodes to the same ids`() {
        let tokenizer = PseudoWordTokenizer(vocabularySize: 100)
        var rng = SeededGenerator(seed: 7)
        let ids = (0 ..< 200).map { _ in Int.random(in: 0 ..< 100, using: &rng) }

        let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(tokenizer.encode(text: text, addSpecialTokens: false) == ids)
    }

    @Test func `every id has a distinct word`() {
        let tokenizer = PseudoWordTokenizer(vocabularySize: 1000)
        let words = (0 ..< 1000).compactMap(tokenizer.convertIdToToken)
        #expect(Set(words).count == 1000)
        #expect(tokenizer.convertTokenToId(words[42]) == 42)
        #expect(tokenizer.convertTokenToId("not-a-word") == nil)
    }

    @Test func `there is no EOS or unknown token`() {
        let tokenizer: any Tokenizer = PseudoWordTokenizer()
        #expect(tokenizer.eosTokenId == nil)
        #expect(tokenizer.unknownTokenId == nil)
    }

    @Test func `a turn's prompt and random output prefix the next turn`() throws {
        let tokenizer = PseudoWordTokenizer()
        var rng = SeededGenerator(seed: 8)
        let generated = (0 ..< 12).map { _ in Int.random(in: 0 ..< 100, using: &rng) }
        let reply = tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)

        let turn1: [Chat.Message] = [.user("how are you?")]
        let generator = DefaultMessageGenerator()
        let prompt1 = try tokenizer.applyChatTemplate(messages: generator.generate(messages: turn1))
        let prompt2 = try tokenizer.applyChatTemplate(
            messages: generator.generate(messages: turn1 + [.assistant(reply), .user("good")]))

        #expect(prompt2.starts(with: prompt1 + generated))
    }

    @Test func `streaming output matches decode`() {
        let tokenizer = PseudoWordTokenizer()
        let ids = Array(0 ..< 30)
        let (chunks, held) = ScriptedTokenizerTests.stream(ids, tokenizer: tokenizer)

        #expect(chunks.joined() == tokenizer.decode(tokenIds: ids, skipSpecialTokens: false))
        #expect(held == 0)
    }
}
