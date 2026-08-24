// Copyright © 2026 Apple Inc.

import Foundation

/// Parses the JSON-compatible subset of Python literals used in tool arguments.
///
/// Pythonic tool-call formats commonly use single-quoted strings and Python's
/// `True`, `False`, and `None` spellings. JSONSerialization cannot read those,
/// so this parser supplies the small recursive grammar the dialect needs while
/// deliberately rejecting executable Python expressions.
struct PythonLiteralParser {
    private let source: Substring
    private var index: String.Index

    private init(_ source: Substring) {
        self.source = source
        self.index = source.startIndex
    }

    static func parse(_ source: Substring) -> (any Sendable)? {
        var parser = Self(source)
        guard let value = parser.parseValue() else { return nil }
        parser.skipWhitespace()
        return parser.index == source.endIndex ? value : nil
    }

    private var current: Character? {
        index < source.endIndex ? source[index] : nil
    }

    private mutating func parseValue() -> (any Sendable)? {
        skipWhitespace()

        switch current {
        case "'", "\"":
            return parseString()
        case "[":
            return parseArray()
        case "{":
            return parseObject()
        case nil:
            return nil
        default:
            return parseAtom()
        }
    }

    private mutating func parseArray() -> (any Sendable)? {
        guard consume("[") else { return nil }
        skipWhitespace()

        var values: [any Sendable] = []
        if consume("]") { return values }

        while true {
            guard let value = parseValue() else { return nil }
            values.append(value)
            skipWhitespace()

            if consume("]") { return values }
            guard consume(",") else { return nil }
            skipWhitespace()
            if consume("]") { return values }
        }
    }

    private mutating func parseObject() -> (any Sendable)? {
        guard consume("{") else { return nil }
        skipWhitespace()

        var object: [String: any Sendable] = [:]
        if consume("}") { return object }

        while true {
            guard let key = parseString() else { return nil }
            skipWhitespace()
            guard consume(":"), let value = parseValue() else { return nil }
            object[key] = value
            skipWhitespace()

            if consume("}") { return object }
            guard consume(",") else { return nil }
            skipWhitespace()
            if consume("}") { return object }
        }
    }

    private mutating func parseString() -> String? {
        guard let quote = current, quote == "'" || quote == "\"" else { return nil }
        advance()

        var result = ""
        while let character = current {
            advance()

            if character == quote { return result }
            guard character == "\\" else {
                result.append(character)
                continue
            }

            guard let escaped = current else { return nil }
            advance()

            switch escaped {
            case "\\": result.append("\\")
            case "'": result.append("'")
            case "\"": result.append("\"")
            case "n": result.append("\n")
            case "r": result.append("\r")
            case "t": result.append("\t")
            case "b": result.append("\u{8}")
            case "f": result.append("\u{c}")
            case "u":
                guard let scalar = parseUnicodeScalar(digitCount: 4) else { return nil }
                result.unicodeScalars.append(scalar)
            case "U":
                guard let scalar = parseUnicodeScalar(digitCount: 8) else { return nil }
                result.unicodeScalars.append(scalar)
            default:
                // Python preserves an unrecognized escape in the string value.
                result.append("\\")
                result.append(escaped)
            }
        }
        return nil
    }

    private mutating func parseUnicodeScalar(digitCount: Int) -> Unicode.Scalar? {
        var digits = ""
        digits.reserveCapacity(digitCount)

        for _ in 0 ..< digitCount {
            guard let character = current,
                UInt32(String(character), radix: 16) != nil
            else { return nil }
            digits.append(character)
            advance()
        }

        guard let value = UInt32(digits, radix: 16) else { return nil }
        return Unicode.Scalar(value)
    }

    private mutating func parseAtom() -> (any Sendable)? {
        let start = index
        while let character = current,
            !character.isWhitespace,
            ![",", "]", "}"].contains(character)
        {
            advance()
        }

        guard start < index else { return nil }
        let token = String(source[start ..< index])

        switch token {
        case "True", "true": return true
        case "False", "false": return false
        case "None", "null", "nil": return NSNull()
        default: break
        }

        let normalizedNumber = token.replacingOccurrences(of: "_", with: "")
        if let integer = Int(normalizedNumber) { return integer }
        if let number = Double(normalizedNumber), number.isFinite { return number }
        return nil
    }

    private mutating func skipWhitespace() {
        while current?.isWhitespace == true { advance() }
    }

    @discardableResult
    private mutating func consume(_ character: Character) -> Bool {
        guard current == character else { return false }
        advance()
        return true
    }

    private mutating func advance() {
        index = source.index(after: index)
    }
}

func tryParsePythonLiteral(_ value: String) -> (any Sendable)? {
    PythonLiteralParser.parse(value[...])
}
