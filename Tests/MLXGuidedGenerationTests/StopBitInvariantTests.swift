// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXGuidedGeneration

/// The declared stop bits of `computeMask()` match `canStop()`, which asks
/// the matcher by accepting a stop token and rolling it back.
@Suite
struct StopBitInvariantTests {

    private static let byteVocabulary = HarnessTokenizer().grammarVocabulary

    private static let twoStringSchema =
        #"{"type":"object","properties":{"name":{"type":"string"},"city":{"type":"string"}},"required":["name","city"],"additionalProperties":false}"#

    @Test func rejectedStopLeavesStateUnchanged() throws {
        let tokenizer = try GrammarTokenizer(
            vocab: Self.byteVocabulary, vocabType: .byteFallback, eosTokenId: 255)
        let constraint = try GrammarConstraint(tokenizer: tokenizer, grammar: #"root ::= "ab""#)
        _ = try constraint.commitToken(0x61)
        let before = try constraint.computeRawMask()

        #expect(try !constraint.canStop())
        #expect(try constraint.computeRawMask().mask == before.mask)
        _ = try constraint.commitToken(0x62)
        #expect(try constraint.canStop())
        #expect(try constraint.canStop())
    }

    /// Two constraints take the same tokens; only one asks `canStop()` at
    /// every step. Masks and fast-forward commits stay identical.
    @Test func acceptAndRollbackChangesNothing() throws {
        let host = HarnessTokenizer()
        let schema =
            #"{"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"}},"required":["s","n"],"additionalProperties":false}"#
        let target = Array(#"{"s":"ab","n":12}"#.utf8).map(Int.init)
        func make() throws -> GrammarConstraint {
            try GrammarConstraint(
                tokenizer: host.makeGrammarTokenizer(stopTokenIds: [0x78]), jsonSchema: schema,
                fastForward: true, hostTokenizer: host)
        }
        let asked = try make()
        let reference = try make()

        var emitted: [Int] = []
        while emitted.count < target.count {
            _ = try asked.canStop()
            _ = try asked.canStop()
            #expect(try asked.computeRawMask().mask == reference.computeRawMask().mask)
            let next = Int32(target[emitted.count])
            let askedResult = try asked.commitToken(next)
            let referenceResult = try reference.commitToken(next)
            #expect(askedResult.tokens == referenceResult.tokens)
            #expect(askedResult.isTerminated == referenceResult.isTerminated)
            emitted += [Int(next)] + askedResult.tokens.map(Int.init)
        }
        #expect(emitted == target)
        #expect(try asked.canStop())
        #expect(try asked.computeRawMask().mask == reference.computeRawMask().mask)
        #expect(try asked.commitToken(0x78).isTerminated)
        #expect(try reference.commitToken(255).isTerminated)
    }

    /// On a vocabulary large enough for xgrammar to store some masks as
    /// bitsets, the stop bits are clear inside the JSON and set once it closes.
    @Test(arguments: [
        #"{"name":"ada","city":"rome"}"#,
        #"{"name":"","city":"x y, z"}"#,
        #"{"name":"bob \"b\"","city":"abc]|abd];"}"#,
    ])
    func stopBitsMatchCanStopOnBigVocabulary(output: String) throws {
        let tokenizer = try BigVocabulary.grammarTokenizer()
        let stops = BigVocabulary.stopTokenIDs
        for tokens in [BigVocabulary.characterTokens(output), BigVocabulary.greedyTokens(output)] {
            let constraint = try GrammarConstraint(
                tokenizer: tokenizer, jsonSchema: Self.twoStringSchema)
            let stopPositions = try Self.walk(
                constraint, tokens: tokens, stops: stops, vocabSize: tokenizer.vocabSize)
            #expect(stopPositions == [tokens.count])
        }
    }

    /// The same check on the tool-calling structural tag, wrapped and bare.
    @Test(arguments: [
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Rome, IT\"}}\n</tool_call>",
        "{\"name\": \"find_place\", \"arguments\": {\"name\":\"ada\",\"city\":\"abc]|\"}}",
    ])
    func stopBitsMatchCanStopOnToolCallingTag(output: String) throws {
        let tokenizer = try BigVocabulary.grammarTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, structuralTag: BigVocabulary.toolCallingTag)
        let tokens = BigVocabulary.characterTokens(output)
        let stopPositions = try Self.walk(
            constraint, tokens: tokens, stops: BigVocabulary.stopTokenIDs,
            vocabSize: tokenizer.vocabSize)
        #expect(stopPositions == [tokens.count])
    }

    /// Commits `tokens` one by one and returns the positions where the grammar
    /// can end. At each one the mask's stop bits must match `canStop()`.
    private static func walk(
        _ constraint: GrammarConstraint, tokens: [Int32], stops: [Int], vocabSize: Int
    ) throws -> [Int] {
        var stopPositions: [Int] = []
        for position in 0 ... tokens.count {
            let canStop = try constraint.canStop()
            let bits = stopBits(try constraint.computeMask(), stops: stops, vocabSize: vocabSize)
            #expect(bits == canStop, "position \(position) of \(tokens.count)")
            let rawBits = stopBits(
                try constraint.computeRawMask(), stops: stops, vocabSize: vocabSize)
            // Upstream xgrammar can set stop bits where the grammar cannot end.
            withKnownIssue(
                "xgrammar leaves stop bits set in cached bitset masks", isIntermittent: true
            ) {
                #expect(rawBits == canStop, "raw mask, position \(position)")
            }
            if canStop { stopPositions.append(position) }
            if position < tokens.count {
                _ = try constraint.commitToken(tokens[position])
            }
        }
        return stopPositions
    }
}

extension BigVocabulary {
    /// Longest-match tokens for `text`.
    static func greedyTokens(_ text: String) -> [Int32] {
        let characters = Array(text)
        var tokens: [Int32] = []
        var start = 0
        while start < characters.count {
            var end = min(characters.count, start + 6)
            while end > start + 1, ids[String(characters[start ..< end])] == nil {
                end -= 1
            }
            tokens.append(Int32(ids[String(characters[start ..< end])]!))
            start = end
        }
        return tokens
    }
}
