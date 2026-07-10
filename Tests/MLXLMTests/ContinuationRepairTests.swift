// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Deterministic tokenizer for `repairContinuation` tests: known BOS id and
/// a fixed encoding for the turn-closure text.
private struct RepairTestTokenizer: MLXLMCommon.Tokenizer {

    static let bosId = 2
    static let closureText = "<turn|>\n"
    static let closureIds = [77, 78]

    var bosToken: String? = "<bos>"
    var eosToken: String? = nil
    var unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text == Self.closureText ? Self.closureIds : [9]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        token == bosToken ? Self.bosId : nil
    }

    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [Self.bosId, 10, 11]
    }
}

public class ContinuationRepairTests: XCTestCase {

    private let tokenizer = RepairTestTokenizer()

    private func tokenList(_ tokens: MLXArray) -> [Int] {
        tokens.asArray(Int.self)
    }

    // MARK: - rank-1 (default text-only LLMUserInputProcessor shape)

    func testStripsBOSFromRank1Tokens() {
        let input = LMInput(tokens: MLXArray([2, 10, 11]))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer, closure: nil, stripBOS: true)

        XCTAssertEqual(repaired.text.tokens.ndim, 1)
        XCTAssertEqual(tokenList(repaired.text.tokens), [10, 11])
        XCTAssertNil(repaired.text.mask)
    }

    func testPrependsClosureToRank1Tokens() {
        let input = LMInput(tokens: MLXArray([2, 10, 11]))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer,
            closure: RepairTestTokenizer.closureText, stripBOS: true)

        XCTAssertEqual(repaired.text.tokens.ndim, 1)
        XCTAssertEqual(tokenList(repaired.text.tokens), [77, 78, 10, 11])
    }

    func testNilClosureAndDisabledStripReturnInputUnchanged() {
        let input = LMInput(tokens: MLXArray([2, 10, 11]))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer, closure: nil, stripBOS: false)

        XCTAssertEqual(tokenList(repaired.text.tokens), [2, 10, 11])
    }

    func testLoneBOSTokenIsNotStripped() {
        // A single-token prompt equal to BOS must survive (dim(1) > 1 guard).
        let input = LMInput(tokens: MLXArray([2]))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer, closure: nil, stripBOS: true)

        XCTAssertEqual(tokenList(repaired.text.tokens), [2])
    }

    func testNonBOSFirstTokenIsNotStripped() {
        let input = LMInput(tokens: MLXArray([10, 11]))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer, closure: nil, stripBOS: true)

        XCTAssertEqual(tokenList(repaired.text.tokens), [10, 11])
    }

    // MARK: - rank-2 (VLM processor shape)

    func testRepairsRank2TokensPreservingRank() {
        let input = LMInput(
            text: .init(tokens: MLXArray([2, 10, 11]).expandedDimensions(axis: 0)))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer,
            closure: RepairTestTokenizer.closureText, stripBOS: true)

        XCTAssertEqual(repaired.text.tokens.ndim, 2)
        XCTAssertEqual(repaired.text.tokens.dim(0), 1)
        XCTAssertEqual(tokenList(repaired.text.tokens.squeezed(axis: 0)), [77, 78, 10, 11])
    }

    // MARK: - mask handling

    func testNilMaskStaysNil() {
        let input = LMInput(
            text: .init(tokens: MLXArray([2, 10, 11]).expandedDimensions(axis: 0), mask: nil))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer, closure: nil, stripBOS: true)

        XCTAssertNil(repaired.text.mask)
    }

    func testExistingMaskIsRebuiltToNewLength() throws {
        let tokens = MLXArray([2, 10, 11]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)))

        let repaired = repairContinuation(
            input, tokenizer: tokenizer,
            closure: RepairTestTokenizer.closureText, stripBOS: true)

        let mask = try XCTUnwrap(repaired.text.mask)
        XCTAssertEqual(mask.dim(-1), repaired.text.tokens.dim(-1))
    }
}
