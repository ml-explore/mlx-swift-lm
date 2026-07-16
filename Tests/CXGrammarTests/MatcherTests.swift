// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for the CXGrammar shim's GrammarMatcher path.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct MatcherTests {

    /// Initial mask allows `{` and has correct length.
    ///
    /// Constructs a tokenizer whose vocab is a mostly-placeholder array
    /// of `vocabSize` entries, with:
    ///   - eos string at `eosTokenId`
    ///   - `{` at a chosen low-index position (`openBraceTokenId`)
    ///
    /// Compiles the minimal object schema and builds a
    /// matcher. In the initial state, JSON must begin with `{`, so:
    ///   - bitmask word count == `(vocabSize + 31) / 32`
    ///   - bit for `{` (index `openBraceTokenId`) is SET in word 0
    ///   - LSB-first ordering: the mirrored MSB-first position is NOT set
    ///   - eos bit is NOT set (grammar has not yet completed)
    @Test
    func testInitialMaskShape() throws {
        let fixture = try Self.loadGemmaFixture()

        let vocabSize = Int(fixture.vocabSize)
        let eosId = Int(fixture.eosTokenId)
        let openBraceTokenId = 2  // Placed at word 0, bit 2; must not collide with eos.
        #expect(openBraceTokenId != eosId, "brace token must not collide with eos")

        let placeholder = "<|tok|>"
        var vocabStrings = Array(repeating: placeholder, count: vocabSize)
        if eosId >= 0 && eosId < vocabSize {
            vocabStrings[eosId] = fixture.eosTokenString
        }
        vocabStrings[openBraceTokenId] = "{"

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
        #expect(tokenizerStatus == XG_OK)
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        #expect(xg_grammar_compiler_new(info, &compiler) == XG_OK)
        defer { xg_grammar_compiler_free(compiler) }

        let schema = #"{"type":"object","properties":{"name":{"type":"string"}}}"#
        var compiled: OpaquePointer?
        let compileStatus = schema.withCString { schemaPtr in
            xg_compile_json_schema(compiler, schemaPtr, &compiled)
        }
        #expect(compileStatus == XG_OK)
        defer { xg_compiled_grammar_free(compiled) }

        var matcher: OpaquePointer?
        let matcherStatus = xg_matcher_new(compiled, &matcher)
        #expect(
            matcherStatus == XG_OK,
            "xg_matcher_new returned \(matcherStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(matcher != nil)
        defer { xg_matcher_free(matcher) }

        // Length check — shim's helper must agree with the formula.
        let expectedWords = Int((vocabSize + 31) / 32)
        let reportedWords = Int(xg_bitmask_size(Int32(vocabSize)))
        #expect(reportedWords == expectedWords, "xg_bitmask_size must equal (vocabSize + 31) / 32")

        var bitmask = [Int32](repeating: 0, count: expectedWords)
        var needsApply: Int32 = -1

        let fillStatus = bitmask.withUnsafeMutableBufferPointer { buf in
            xg_matcher_fill_next_token_bitmask(
                matcher,
                buf.baseAddress,
                buf.count,
                Int32(vocabSize),
                &needsApply
            )
        }
        #expect(
            fillStatus == XG_OK,
            "xg_matcher_fill_next_token_bitmask returned \(fillStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(
            needsApply == 1,
            "initial mask must be a proper subset of the vocab; needs_apply should be true")

        // Bit for { must be set; mirrored MSB-first position must NOT be set
        // (asserts LSB-first word ordering).
        let word0 = bitmask[0]
        let braceBit = Int32(1) << Int32(openBraceTokenId)
        let mirroredBit = Int32(1) << Int32(31 - openBraceTokenId)
        #expect(
            (word0 & braceBit) != 0,
            "bit \(openBraceTokenId) (token `{`) must be set in word 0; got word0=0x\(String(word0, radix: 16))"
        )
        #expect(
            (word0 & mirroredBit) == 0,
            "MSB-first mirrored bit \(31 - openBraceTokenId) must NOT be set (LSB-first ordering check); got word0=0x\(String(word0, radix: 16))"
        )

        // eos (token 1) must not be initially acceptable.
        let eosBit = Int32(1) << Int32(eosId)
        #expect(
            (word0 & eosBit) == 0,
            "eos bit \(eosId) must NOT be set in the initial mask (grammar has not completed yet)"
        )

        _ = cStrings.count
    }

    /// Committing a token advances matcher state.
    ///
    /// Build the same matcher used by testInitialMaskShape, capture
    /// the initial bitmask, commit the `{` token, then capture the
    /// next bitmask. The masks must differ: after opening the object
    /// the next acceptable tokens are `"` (for the property key) or
    /// `}` (for an empty object), not `{` alone.
    @Test
    func testCommitAdvancesState() throws {
        let context = try Self.makeConstraintContext()
        defer { context.tearDown() }

        let expectedWords = Int(xg_bitmask_size(Int32(context.vocabSize)))
        var initialMask = [Int32](repeating: 0, count: expectedWords)
        let initialFillStatus = initialMask.withUnsafeMutableBufferPointer { buf in
            xg_matcher_fill_next_token_bitmask(
                context.matcher,
                buf.baseAddress,
                buf.count,
                Int32(context.vocabSize),
                nil
            )
        }
        #expect(initialFillStatus == XG_OK)

        let acceptStatus = xg_matcher_accept_token(context.matcher, Int32(context.openBraceTokenId))
        #expect(
            acceptStatus == XG_OK,
            "xg_matcher_accept_token on `{` must succeed; got \(acceptStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )

        var nextMask = [Int32](repeating: 0, count: expectedWords)
        let nextFillStatus = nextMask.withUnsafeMutableBufferPointer { buf in
            xg_matcher_fill_next_token_bitmask(
                context.matcher,
                buf.baseAddress,
                buf.count,
                Int32(context.vocabSize),
                nil
            )
        }
        #expect(nextFillStatus == XG_OK)

        #expect(initialMask != nextMask, "matcher state must advance after committing a token")
    }

    /// Accepting a grammar-disallowed token returns
    /// `XG_ERR_INVALID_ARG`.
    ///
    /// Token 0 in the synthetic vocab decodes to `<|tok|>`, which
    /// starts with `<` — not a valid first byte for a strict-mode
    /// JSON object. xgrammar's AcceptToken returns `false` for that,
    /// which the shim maps to `XG_ERR_INVALID_ARG` (a caller-argument
    /// error, distinct from internal failure).
    @Test
    func testRejectsInvalidToken() throws {
        let context = try Self.makeConstraintContext()
        defer { context.tearDown() }

        let invalidTokenId: Int32 = 0  // "<|tok|>" placeholder

        let acceptStatus = xg_matcher_accept_token(context.matcher, invalidTokenId)
        #expect(
            acceptStatus == XG_ERR_INVALID_ARG,
            "xg_matcher_accept_token on a grammar-disallowed token must return XG_ERR_INVALID_ARG; got \(acceptStatus)"
        )
    }

    // MARK: - Shared matcher setup

    /// Groups the handles a matcher test needs. Ordered-destroy in
    /// tearDown: matcher → compiled → compiler → tokenizer, matching
    /// construction order.
    private struct ConstraintContext {
        let vocabSize: Int
        let openBraceTokenId: Int
        let info: OpaquePointer?
        let compiler: OpaquePointer?
        let compiled: OpaquePointer?
        let matcher: OpaquePointer?
        let cStrings: [[CChar]]  // Keep backing storage alive.

        func tearDown() {
            xg_matcher_free(matcher)
            xg_compiled_grammar_free(compiled)
            xg_grammar_compiler_free(compiler)
            xg_tokenizer_info_free(info)
            _ = cStrings.count
        }
    }

    private static func makeConstraintContext() throws -> ConstraintContext {
        let fixture = try loadGemmaFixture()
        let vocabSize = Int(fixture.vocabSize)
        let eosId = Int(fixture.eosTokenId)
        let openBraceTokenId = 2

        let placeholder = "<|tok|>"
        var vocabStrings = Array(repeating: placeholder, count: vocabSize)
        if eosId >= 0 && eosId < vocabSize {
            vocabStrings[eosId] = fixture.eosTokenString
        }
        vocabStrings[openBraceTokenId] = "{"

        let cStrings = vocabStrings.map { Array($0.utf8CString) }
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
        precondition(tokenizerStatus == XG_OK, "tokenizer construction failed: \(tokenizerStatus)")

        var compiler: OpaquePointer?
        let compilerStatus = xg_grammar_compiler_new(info, &compiler)
        precondition(compilerStatus == XG_OK, "compiler construction failed: \(compilerStatus)")

        let schema = #"{"type":"object","properties":{"name":{"type":"string"}}}"#
        var compiled: OpaquePointer?
        let compileStatus = schema.withCString { ptr in
            xg_compile_json_schema(compiler, ptr, &compiled)
        }
        precondition(compileStatus == XG_OK, "schema compile failed: \(compileStatus)")

        var matcher: OpaquePointer?
        let matcherStatus = xg_matcher_new(compiled, &matcher)
        precondition(matcherStatus == XG_OK, "matcher construction failed: \(matcherStatus)")

        return ConstraintContext(
            vocabSize: vocabSize,
            openBraceTokenId: openBraceTokenId,
            info: info,
            compiler: compiler,
            compiled: compiled,
            matcher: matcher,
            cStrings: cStrings
        )
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
