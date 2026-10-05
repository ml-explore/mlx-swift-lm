// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM
import Testing

struct ScriptedTokenizerTests {

    /// Simplest example of the tokenizer functionality
    @Test func trivial() {
        let corpus = "the quick brown fox and the lazy cat"
        let t = ScriptedTokenizer(corpus: [corpus])

        let text = "welcome to the fox"
        let tokens = t.encode(text: text)

        // tokenization is deterministic with the same corpus
        #expect(tokens == [183, 165, 172, 163, 175, 173, 165, 96, 180, 175, 327, 325])

        // round trip
        #expect(t.decode(tokenIds: tokens) == text)
    }

    /// Feeds `ids` through the library's streaming detokenizer, one token at a time.
    static func stream(_ ids: [Int], tokenizer: some Tokenizer) -> (chunks: [String], held: Int) {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var chunks: [String] = []
        var held = 0
        for id in ids {
            detokenizer.append(token: id)
            if let chunk = detokenizer.next() {
                chunks.append(chunk)
            } else {
                held += 1
            }
        }
        return (chunks, held)
    }

    @Test func `encode takes the longest matching piece`() {
        let tokenizer = ScriptedTokenizer(
            corpus: ["hell", "hello"], configuration: .init(maxPieceLength: nil))
        let hell = tokenizer.convertTokenToId("hell")!
        let hello = tokenizer.convertTokenToId("hello")!
        let x = tokenizer.convertTokenToId("x")!
        let space = tokenizer.convertTokenToId(" ")!

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [hello])
        #expect(tokenizer.encode(text: "hellx", addSpecialTokens: false) == [hell, x])
        #expect(tokenizer.encode(text: " hello", addSpecialTokens: false) == [space, hello])
    }

    @Test func `specials take precedence over pieces`() {
        let tokenizer = ScriptedTokenizer(
            corpus: ["think", "thinking"], specials: ["<think>"],
            configuration: .init(maxPieceLength: nil))
        let think = tokenizer.convertTokenToId("<think>")!

        #expect(tokenizer.encode(text: "<think>", addSpecialTokens: false) == [think])
        #expect(!tokenizer.encode(text: "think", addSpecialTokens: false).contains(think))
    }

    @Test func `skipSpecialTokens drops only specials`() {
        let tokenizer = ScriptedTokenizer(corpus: ["hi there"])
        let text = "<|user|>hi there<|end|>"
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)

        #expect(tokenizer.decode(tokenIds: ids, skipSpecialTokens: false) == text)
        #expect(tokenizer.decode(tokenIds: ids, skipSpecialTokens: true) == "hi there")
    }

    @Test func `library special-token detection sees atomic markers only`() {
        let tokenizer = ScriptedTokenizer(corpus: [], textMarkers: ["<think>"])

        #expect(tokenizer.specialTokenNames(inImageLabel: "a <|end|> b") == ["<|end|>"])
        #expect(tokenizer.specialTokenNames(inImageLabel: "a <think> b") == nil)
        #expect(tokenizer.specialTokenNames(inImageLabel: "plain") == nil)
    }

    @Test func `token lookups resolve markers, bytes, and EOS`() {
        let tokenizer = ScriptedTokenizer(corpus: ["hello"], textMarkers: ["<think>"])
        let end = tokenizer.convertTokenToId("<|end|>")

        #expect(tokenizer.eosToken == "<|end|>")
        #expect(end != nil)
        #expect(tokenizer.eosTokenId == end)
        #expect(tokenizer.convertIdToToken(end!) == "<|end|>")
        #expect(tokenizer.convertTokenToId("<0x0A>") == tokenizer.convertTokenToId("\n"))
        #expect(tokenizer.convertIdToToken(tokenizer.convertTokenToId("\n")!) == "<0x0A>")
        #expect(tokenizer.convertTokenToId("<think>") == nil)
        #expect(tokenizer.unknownTokenId == nil)
    }

    @Test func `addSpecialTokens adds BOS`() {
        let tokenizer = ScriptedTokenizer(corpus: ["hello"], bosToken: "<|bos|>")
        let bos = tokenizer.convertTokenToId("<|bos|>")!

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: true).first == bos)
        #expect(!tokenizer.encode(text: "hello", addSpecialTokens: false).contains(bos))
    }

    @Test func `streaming detokenizer handles multi-token words and split UTF-8`() {
        let text = "Hello wonderful 👍🏽 world\nnext line é"
        let tokenizer = ScriptedTokenizer(corpus: [text])
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let thumbs = tokenizer.encode(text: "👍🏽", addSpecialTokens: false)
        let wonderful = tokenizer.encode(text: " wonderful", addSpecialTokens: false)

        #expect(thumbs.count == 8)
        #expect(wonderful.count > 1)

        let (chunks, held) = Self.stream(ids, tokenizer: tokenizer)
        #expect(chunks.joined() == text)
        #expect(!chunks.contains { $0.contains("\u{FFFD}") })
        #expect(held > 0, "partial UTF-8 should be held back")
    }

    @Test func `a multi-token marker decodes like an atomic one`() {
        let text = "<think>plan the reply</think>The answer."
        let atomic = ScriptedTokenizer(corpus: [text], specials: ["<think>", "</think>"])
        let multi = ScriptedTokenizer(corpus: [text], textMarkers: ["<think>", "</think>"])

        let atomicIDs = atomic.encode(text: text, addSpecialTokens: false)
        let multiIDs = multi.encode(text: text, addSpecialTokens: false)
        #expect(multiIDs.count > atomicIDs.count)

        #expect(atomic.decode(tokenIds: atomicIDs, skipSpecialTokens: false) == text)
        #expect(multi.decode(tokenIds: multiIDs, skipSpecialTokens: false) == text)
        #expect(
            Self.stream(atomicIDs, tokenizer: atomic).chunks.joined()
                == Self.stream(multiIDs, tokenizer: multi).chunks.joined())
    }

    @Test func `eos defaults to the template's end of turn`() {
        let tokenizer = ScriptedTokenizer(corpus: [])
        #expect(tokenizer.eosTokenId == tokenizer.convertTokenToId("<|end|>"))
    }

    @Test func `eos none keeps the marker but stops nothing`() {
        let tokenizer = ScriptedTokenizer(corpus: [], eos: .none)
        #expect(tokenizer.eosTokenId == nil)
        #expect(tokenizer.convertTokenToId("<|end|>") != nil)
    }

    @Test func `eos token registers that special`() {
        let tokenizer = ScriptedTokenizer(corpus: [], eos: .token("<|stop|>"))
        #expect(tokenizer.eosToken == "<|stop|>")
        #expect(tokenizer.eosTokenId == tokenizer.convertTokenToId("<|stop|>"))
        #expect(tokenizer.vocabulary.isSpecial(tokenizer.eosTokenId!))
    }
}
