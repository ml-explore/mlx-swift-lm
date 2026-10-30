// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for `xg_compile_json_schema_with_whitespace`.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct WhitespaceTests {

    /// After `{`, the default grammar admits a newline; with
    /// `any_whitespace` off it admits only the opening quote of a key.
    @Test(arguments: [(Int32(1), true), (Int32(0), false)])
    func testNewlineAfterOpenBrace(anyWhitespace: Int32, acceptsNewline: Bool) throws {
        let fixture = try Self.loadGemmaFixture()
        let vocabSize = fixture.vocabSize
        let eosId = fixture.eosTokenId
        let openBraceTokenId = 2
        let newlineTokenId = 3
        let quoteTokenId = 4
        #expect(![openBraceTokenId, newlineTokenId, quoteTokenId].contains(eosId))

        var vocabStrings = Array(repeating: "<|tok|>", count: vocabSize)
        vocabStrings[eosId] = fixture.eosTokenString
        vocabStrings[openBraceTokenId] = "{"
        vocabStrings[newlineTokenId] = "\n"
        vocabStrings[quoteTokenId] = "\""

        let cStrings = vocabStrings.map { $0.utf8CString }
        var vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map { arr in
            arr.withUnsafeBufferPointer { buf in buf.baseAddress }
        }
        var info: OpaquePointer?
        let stopTokens: [Int32] = [Int32(eosId)]
        let tokenizerStatus: XGStatus = vocabPtrs.withUnsafeMutableBufferPointer { vocabBuf in
            stopTokens.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    vocabBuf.baseAddress, vocabBuf.count, XG_VOCAB_TYPE_RAW,
                    stopBuf.baseAddress, stopBuf.count, &info)
            }
        }
        #expect(tokenizerStatus == XG_OK)
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        #expect(xg_grammar_compiler_new(info, &compiler) == XG_OK)
        defer { xg_grammar_compiler_free(compiler) }

        let schema =
            #"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#
        var compiled: OpaquePointer?
        let compileStatus = schema.withCString { schemaPtr in
            xg_compile_json_schema_with_whitespace(compiler, schemaPtr, anyWhitespace, &compiled)
        }
        #expect(
            compileStatus == XG_OK,
            "xg_compile_json_schema_with_whitespace returned \(compileStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        defer { xg_compiled_grammar_free(compiled) }

        var matcher: OpaquePointer?
        #expect(xg_matcher_new(compiled, &matcher) == XG_OK)
        defer { xg_matcher_free(matcher) }

        #expect(xg_matcher_accept_token(matcher, Int32(openBraceTokenId)) == XG_OK)
        let newlineStatus = xg_matcher_accept_token(matcher, Int32(newlineTokenId))
        if acceptsNewline {
            #expect(newlineStatus == XG_OK, "a newline after `{` must be accepted by default")
        } else {
            #expect(newlineStatus == XG_ERR_INVALID_ARG, "a newline after `{` must be refused")
            #expect(xg_matcher_accept_token(matcher, Int32(quoteTokenId)) == XG_OK)
        }
        _ = cStrings.count
    }

    // MARK: - Fixture loading

    private struct GemmaFixture {
        let vocabSize: Int
        let eosTokenId: Int
        let eosTokenString: String
    }

    private static func loadGemmaFixture() throws -> GemmaFixture {
        let url = Self.goldensDirectory.appendingPathComponent("tokenizer_gemma3.json")
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let vocabSize = json["vocabSize"] as? Int,
            let eosTokenId = json["eosTokenId"] as? Int,
            let eosTokenString = json["eosTokenString"] as? String
        else {
            throw FixtureError.malformed("tokenizer_gemma3.json")
        }
        return GemmaFixture(
            vocabSize: vocabSize, eosTokenId: eosTokenId, eosTokenString: eosTokenString)
    }

    private static let goldensDirectory: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures", isDirectory: true)
            .appendingPathComponent("goldens", isDirectory: true)
    }()

    private enum FixtureError: Error {
        case malformed(String)
    }
}
