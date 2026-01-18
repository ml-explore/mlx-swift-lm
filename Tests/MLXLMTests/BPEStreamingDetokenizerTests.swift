// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon
import Tokenizers
import XCTest

final class BPEStreamingDetokenizerTests: XCTestCase {

    func testByteLevelDetokenizerDecodesBytes() {
        let text = "Hello, world!"
        let byteToUnicode = Self.byteToUnicode()
        let tokens: [String] = text.utf8.compactMap { byte in
            byteToUnicode[byte].map(String.init)
        }
        var vocabulary: [Int: String] = [:]
        for (index, token) in tokens.enumerated() {
            vocabulary[index] = token
        }
        let tokenizer = ByteLevelTestTokenizer(vocabulary: vocabulary)
        var detokenizer = BPEStreamingDetokenizer(tokenizer: tokenizer)

        var output = ""
        for index in 0 ..< tokens.count {
            detokenizer.append(token: index)
            if let chunk = detokenizer.next() {
                output += chunk
            }
        }

        XCTAssertEqual(output, text)
    }

    private static func byteToUnicode() -> [UInt8: UnicodeScalar] {
        var byteSet = Set<UInt8>()
        var mapping: [UInt8: UnicodeScalar] = [:]

        func addRange(_ range: ClosedRange<Int>) {
            for value in range {
                byteSet.insert(UInt8(value))
            }
        }

        addRange(33 ... 126)
        addRange(161 ... 172)
        addRange(174 ... 255)

        var n = 0
        for value in 0 ... 255 {
            let byte = UInt8(value)
            if byteSet.contains(byte) {
                if let scalar = UnicodeScalar(value) {
                    mapping[byte] = scalar
                }
            } else {
                if let scalar = UnicodeScalar(256 + n) {
                    mapping[byte] = scalar
                }
                n += 1
            }
        }

        return mapping
    }
}

private struct ByteLevelTestTokenizer: Tokenizer {

    let vocabulary: [Int: String]

    init(vocabulary: [Int: String]) {
        self.vocabulary = vocabulary
    }

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map { String($0) }
    }

    func encode(text: String) -> [Int] {
        Array(vocabulary.keys).sorted()
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.compactMap { convertIdToToken($0) }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        vocabulary.first(where: { $0.value == token })?.key
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocabulary[id]
    }

    var bosToken: String? = nil

    var bosTokenId: Int? = nil

    var eosToken: String? = nil

    var eosTokenId: Int? = nil

    var unknownToken: String? = nil

    var unknownTokenId: Int? = nil

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        []
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
