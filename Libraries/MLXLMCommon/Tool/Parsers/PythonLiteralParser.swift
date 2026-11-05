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
    private var depth = 0

    private init(_ source: Substring) {
        self.source = source
        self.index = source.startIndex
    }

    static func parse(_ source: Substring) -> (any Sendable)? {
        var parser = Self(source)
        guard let value = parser.parseValue() else { return nil }
        parser.skipWhitespace()
        if parser.consume(",") {
            var values = [value]
            repeat {
                parser.skipWhitespace()
                if parser.current == nil { return values }
                guard let next = parser.parseValue() else { return nil }
                values.append(next)
                parser.skipWhitespace()
            } while parser.consume(",")
            return parser.current == nil ? values : nil
        }
        return parser.index == source.endIndex ? value : nil
    }

    private var current: Character? {
        index < source.endIndex ? source[index] : nil
    }

    private mutating func parseValue() -> (any Sendable)? {
        guard depth < 32 else { return nil }
        depth += 1
        defer { depth -= 1 }
        skipWhitespace()

        switch current {
        case "'", "\"":
            return parseString()
        case "[", "(":
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
        let tuple = current == "("
        guard consume(tuple ? "(" : "[") else { return nil }
        let close: Character = tuple ? ")" : "]"
        var hasComma = false
        skipWhitespace()

        var values: [any Sendable] = []
        if consume(close) { return values }

        while true {
            guard let value = parseValue() else { return nil }
            values.append(value)
            skipWhitespace()

            if consume(close) { return tuple && !hasComma ? value : values }
            guard consume(",") else { return nil }
            hasComma = true
            skipWhitespace()
            if consume(close) { return values }
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
            case "a": result.append("\u{7}")
            case "b": result.append("\u{8}")
            case "f": result.append("\u{c}")
            case "v": result.append("\u{b}")
            case "\n", "\r\n", "\r":
                // A backslash at the end of a line continues the string.
                break
            case "0", "1", "2", "3", "4", "5", "6", "7":
                guard let scalar = parseOctalEscape(startingWith: escaped) else { return nil }
                result.unicodeScalars.append(scalar)
            case "x":
                guard let scalar = parseUnicodeScalar(digitCount: 2) else { return nil }
                result.unicodeScalars.append(scalar)
            case "u":
                guard let scalar = parseUTF16Escape() else { return nil }
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

    /// Joins a surrogate pair written as two `\u` escapes, as JSON does.
    private mutating func parseUTF16Escape() -> Unicode.Scalar? {
        guard let high = parseHexValue(digitCount: 4) else { return nil }
        guard (0xD800 ... 0xDBFF).contains(high) else { return Unicode.Scalar(high) }

        guard consume("\\"), consume("u"),
            let low = parseHexValue(digitCount: 4),
            (0xDC00 ... 0xDFFF).contains(low)
        else { return nil }
        return Unicode.Scalar(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    }

    /// Reads up to three octal digits, as Python does.
    private mutating func parseOctalEscape(startingWith first: Character) -> Unicode.Scalar? {
        var digits = String(first)
        while digits.count < 3, let character = current,
            UInt32(String(character), radix: 8) != nil
        {
            digits.append(character)
            advance()
        }
        return UInt32(digits, radix: 8).flatMap { Unicode.Scalar($0) }
    }

    private mutating func parseUnicodeScalar(digitCount: Int) -> Unicode.Scalar? {
        parseHexValue(digitCount: digitCount).flatMap { Unicode.Scalar($0) }
    }

    private mutating func parseHexValue(digitCount: Int) -> UInt32? {
        var digits = ""
        digits.reserveCapacity(digitCount)

        for _ in 0 ..< digitCount {
            guard let character = current,
                UInt32(String(character), radix: 16) != nil
            else { return nil }
            digits.append(character)
            advance()
        }

        return UInt32(digits, radix: 16)
    }

    private mutating func parseAtom() -> (any Sendable)? {
        let start = index
        while let character = current,
            !character.isWhitespace,
            ![",", "]", "}", ")"].contains(character)
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
