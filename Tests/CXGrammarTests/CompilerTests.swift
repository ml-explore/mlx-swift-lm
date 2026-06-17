// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for the CXGrammar shim's tokenizer-aware
// GrammarCompiler path. Distinct from SchemaErrorTests, which covers
// the tokenizer-free `xg_grammar_from_json_schema` entry point —
// GrammarCompiler is the path that binds a schema to a specific
// tokenizer so a matcher can be built from it.

import MLXCXGrammar
import Foundation
import Testing

@Suite
struct CompilerTests {

    /// Compile a minimal JSON schema against the synthetic tokenizer.
    ///
    /// Builds the same synthetic gemma-3 vocab used in
    /// TokenizerInfoTests, constructs an `XGGrammarCompiler`, then
    /// compiles `{"type":"object","properties":{"name":{"type":"string"}}}`
    /// against it. The bar is: `XG_OK` + non-null `XGCompiledGrammar`
    /// handle.
    @Test
    func testCompileSimpleSchema() throws {
        let fixture = try Self.loadGemmaFixture()

        let vocabSize = Int(fixture.vocabSize)
        let eosId = Int(fixture.eosTokenId)

        let placeholder = "<|tok|>"
        var vocabStrings = Array(repeating: placeholder, count: vocabSize)
        if eosId >= 0 && eosId < vocabSize {
            vocabStrings[eosId] = fixture.eosTokenString
        }

        let cStrings = vocabStrings.map { $0.utf8CString }
        var vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map { arr in
            arr.withUnsafeBufferPointer { buf in buf.baseAddress }
        }

        var info: OpaquePointer?
        let stopTokens: [Int32] = [Int32(eosId)]

        let tokenizerStatus: XGStatus = vocabPtrs.withUnsafeMutableBufferPointer { vocabBuf in
            stopTokens.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    vocabBuf.baseAddress,
                    vocabBuf.count,
                    XG_VOCAB_TYPE_RAW,
                    stopBuf.baseAddress,
                    stopBuf.count,
                    &info
                )
            }
        }
        #expect(tokenizerStatus == XG_OK, "precondition: xg_tokenizer_info_new should succeed")
        #expect(info != nil, "precondition: xg_tokenizer_info_new should produce a handle")
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        let compilerStatus = xg_grammar_compiler_new(info, &compiler)
        #expect(
            compilerStatus == XG_OK,
            "xg_grammar_compiler_new returned \(compilerStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(compiler != nil, "xg_grammar_compiler_new produced a null handle on success")
        defer { xg_grammar_compiler_free(compiler) }

        let schema = #"{"type":"object","properties":{"name":{"type":"string"}}}"#

        var compiled: OpaquePointer?
        let compileStatus: XGStatus = schema.withCString { schemaPtr in
            xg_compile_json_schema(compiler, schemaPtr, &compiled)
        }

        #expect(
            compileStatus == XG_OK,
            "xg_compile_json_schema returned \(compileStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(compiled != nil, "xg_compile_json_schema produced a null handle on success")
        xg_compiled_grammar_free(compiled)

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
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let json else {
            throw FixtureError.malformed("top-level not an object")
        }
        guard let vocabSize = json["vocabSize"] as? Int else {
            throw FixtureError.malformed("missing vocabSize")
        }
        guard let eosTokenId = json["eosTokenId"] as? Int else {
            throw FixtureError.malformed("missing eosTokenId")
        }
        guard let eosTokenString = json["eosTokenString"] as? String else {
            throw FixtureError.malformed("missing eosTokenString")
        }
        return GemmaFixture(
            vocabSize: vocabSize,
            eosTokenId: eosTokenId,
            eosTokenString: eosTokenString
        )
    }

    private static let goldensDirectory: URL = {
        let thisFile = URL(fileURLWithPath: #filePath)
        return
            thisFile
            .deletingLastPathComponent()  // Tests/CXGrammarTests
            .appendingPathComponent("Fixtures", isDirectory: true)
            .appendingPathComponent("goldens", isDirectory: true)
    }()

    private enum FixtureError: Error {
        case malformed(String)
    }
}
