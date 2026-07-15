// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for the CXGrammar shim's `xg_tokenizer_info_*` surface.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct TokenizerInfoTests {

    /// TokenizerInfo construction.
    ///
    /// Loads the gemma-3 golden fixture to derive the vocab shape
    /// (vocabSize, eosTokenId, eosTokenString), builds a synthetic
    /// vocab of that length with the EOS string at its declared
    /// position, and asserts that `xg_tokenizer_info_new` returns
    /// `XG_OK` with a non-null handle. This only certifies that the
    /// C++ constructor is reachable from Swift and doesn't throw on
    /// RAW vocab.
    @Test
    func testTokenizerInfoConstruction() throws {
        let fixture = try Self.loadGemmaFixture()

        // Build a synthetic vocab of the declared size with placeholder
        // entries everywhere except the EOS slot. RAW vocab_type means
        // xgrammar treats each string as its literal UTF-8 byte sequence,
        // so placeholder strings don't trip byte-fallback parsing.
        let vocabSize = Int(fixture.vocabSize)
        let eosId = Int(fixture.eosTokenId)

        let placeholder = "<|tok|>"
        var vocabStrings = Array(repeating: placeholder, count: vocabSize)
        if eosId >= 0 && eosId < vocabSize {
            vocabStrings[eosId] = fixture.eosTokenString
        }

        // C strings must outlive the call; hold onto the CStrings so
        // the `const char *` pointers we hand xgrammar remain valid.
        let cStrings = vocabStrings.map { $0.utf8CString }
        var vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map { arr in
            arr.withUnsafeBufferPointer { buf in buf.baseAddress }
        }

        var info: OpaquePointer?
        let stopTokens: [Int32] = [Int32(eosId)]

        let status: XGStatus = vocabPtrs.withUnsafeMutableBufferPointer { vocabBuf in
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

        #expect(status == XG_OK, "xg_tokenizer_info_new returned status \(status)")
        #expect(info != nil, "xg_tokenizer_info_new produced a null handle on success")

        xg_tokenizer_info_free(info)

        // Keep `cStrings` alive until after the shim call returns.
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

    /// Resolves `Tests/CXGrammarTests/Fixtures/goldens/` relative to this
    /// source file on disk, without Bundle wiring. This target owns its
    /// fixture copy (tiny tokenizer metadata) so it stays self-contained
    /// rather than reaching into a sibling test target's directory.
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
