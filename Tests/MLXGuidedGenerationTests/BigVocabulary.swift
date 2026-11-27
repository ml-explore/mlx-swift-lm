// Copyright © 2026 Apple Inc.

@testable import MLXGuidedGeneration

/// A deterministic raw vocabulary of about 8,300 tokens, large enough for
/// xgrammar to store masks as bitsets (it needs 1,000 tokens or more).
enum BigVocabulary {
    static let eos = "<eos>"
    static let endOfTurn = "<end_of_turn>"
    static let unknown = "<unk>"

    static let tokens: [String] = {
        let letters = (UInt8(ascii: "a") ... UInt8(ascii: "z")).map { String(UnicodeScalar($0)) }
        var tokens = (0x20 ... 0x7E).map { String(UnicodeScalar(UInt8($0))) }
        for a in letters {
            for b in letters { tokens.append(a + b) }
        }
        for a in letters.prefix(5) {
            for b in letters {
                for c in letters { tokens.append(a + b + c) }
            }
        }
        // Tokens that end a bracketed run and go on with `|` or `;`.
        for a in letters {
            for b in letters {
                tokens.append(a + b + "]|")
                tokens.append(a + b + "];")
            }
        }
        for a in letters.prefix(2) {
            for b in letters {
                for c in letters {
                    tokens.append(a + b + c + "]|")
                    tokens.append(a + b + c + "];")
                }
            }
        }
        for a in letters {
            tokens += ["\"" + a, a + "\"", a + "\",", a + "\"}"]
        }
        tokens += ["{\"", "\"}", "\":\"", "\",\"", "\":", "\",", "\":[", "]}", "},{", "\"]"]
        tokens.append("\n")
        tokens += [eos, endOfTurn, unknown]
        return tokens
    }()

    static let ids: [String: Int] = {
        var ids: [String: Int] = [:]
        for (id, token) in tokens.enumerated() where ids[token] == nil {
            ids[token] = id
        }
        return ids
    }()

    static var stopTokenIDs: [Int] { [eos, endOfTurn, unknown].map { ids[$0]! } }

    static func grammarTokenizer() throws -> GrammarTokenizer {
        try GrammarTokenizer(
            vocab: tokens, vocabType: .raw, eosTokenId: Int32(ids[eos]!),
            stopTokenIds: stopTokenIDs)
    }

    /// The tool-calling structural tag as `SchemaConverter` builds it, with
    /// two tools.
    static let toolCallingTag: String = {
        func tool(_ name: String, _ properties: [String]) -> String {
            let props = properties.map { "\"\($0)\":{\"type\":\"string\"}" }.joined(separator: ",")
            let required = properties.map { "\"\($0)\"" }.joined(separator: ",")
            return #"{"type":"tag","begin":"{\"name\": \""# + name
                + #"\", \"arguments\": ","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"#
                + props + #"},"required":["# + required + #"]}},"end":["}"]}"#
        }
        let perTool =
            #"{"type":"or","elements":["# + tool("get_weather", ["location"]) + ","
            + tool("find_place", ["name", "city"]) + "]}"
        return
            #"{"type":"structural_tag","format":{"type":"or","elements":[{"type":"tag","begin":"<tool_call>\n","content":"#
            + perTool + #","end":["\n</tool_call>"]},"# + perTool + "]}}"
    }()

    /// One token per character of `text`.
    static func characterTokens(_ text: String) -> [Int32] {
        text.map { Int32(ids[String($0)]!) }
    }
}

/// The declared stop bits of `mask`: all set, all clear, or mixed (nil).
func stopBits(_ mask: MaskResult, stops: [Int], vocabSize: Int) -> Bool? {
    let bits = stops.map {
        GuidedGenerationLoop.maskAllows(mask, tokenId: $0, vocabSize: vocabSize)
    }
    if bits.allSatisfy({ $0 }) { return true }
    if bits.allSatisfy({ !$0 }) { return false }
    return nil
}
