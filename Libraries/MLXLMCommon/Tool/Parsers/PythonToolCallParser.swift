// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Python-style function calls: `<tag>[func(arg="value", arg2=123)]</tag>`
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public struct PythonToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip tags if present
        if let start = startTag, let range = text.range(of: start) {
            text = String(text[range.upperBound...])
        }
        if let end = endTag, let range = text.range(of: end) {
            text = String(text[..<range.lowerBound])
        }

        // Parse Python-style function call
        var buffer = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return parseFunctionCall(&buffer)
    }

    // MARK: - Python Parsing Logic

    private func parseFunctionCall(_ buffer: inout String) -> ToolCall? {
        // Skip leading brackets/whitespace
        var i = buffer.startIndex
        while i < buffer.endIndex {
            let ch = buffer[i]
            if ch == "[" || ch == "]" || ch == "," || ch.isWhitespace {
                i = buffer.index(after: i)
                continue
            }
            break
        }

        if i >= buffer.endIndex {
            buffer = ""
            return nil
        }

        // Read function name
        guard let nameEnd = readIdentifier(buffer, from: i) else {
            return nil
        }
        let name = String(buffer[i ..< nameEnd])

        // Find opening paren
        var j = nameEnd
        skipWhitespace(buffer, &j)
        guard j < buffer.endIndex, buffer[j] == "(" else { return nil }

        // Find matching closing paren
        guard let closeIdx = findMatchingParen(in: buffer, openIndex: j) else {
            return nil
        }

        // Parse arguments
        let argsBody = String(buffer[buffer.index(after: j) ..< closeIdx])
        let arguments = parseArgs(argsBody)

        // Update buffer (consume parsed content)
        var k = buffer.index(after: closeIdx)
        skipWhitespace(buffer, &k)
        if k < buffer.endIndex, buffer[k] == "," {
            k = buffer.index(after: k)
        }
        while k < buffer.endIndex, buffer[k] == "]" || buffer[k].isWhitespace {
            k = buffer.index(after: k)
        }
        buffer = String(buffer[k...])

        return ToolCall(function: ToolCall.Function(name: name, arguments: arguments))
    }

    private func readIdentifier(_ s: String, from start: String.Index) -> String.Index? {
        var i = start
        guard i < s.endIndex, s[i].isLetter || s[i] == "_" else { return nil }
        i = s.index(after: i)
        while i < s.endIndex, s[i].isLetter || s[i].isNumber || s[i] == "_" {
            i = s.index(after: i)
        }
        return i
    }

    private func skipWhitespace(_ s: String, _ i: inout String.Index) {
        while i < s.endIndex, s[i].isWhitespace {
            i = s.index(after: i)
        }
    }

    private func findMatchingParen(in s: String, openIndex: String.Index) -> String.Index? {
        var i = s.index(after: openIndex)
        var depth = 1
        var quote: Character?
        var escape = false

        while i < s.endIndex {
            let ch = s[i]
            if let q = quote {
                if escape {
                    escape = false
                } else if ch == "\\" {
                    escape = true
                } else if ch == q {
                    quote = nil
                }
            } else {
                switch ch {
                case "'", "\"": quote = ch
                case "(": depth += 1
                case ")":
                    depth -= 1
                    if depth == 0 { return i }
                default: break
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    private func parseArgs(_ body: String) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        let parts = splitTopLevel(body, on: ",")

        for part in parts {
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                let eqIdx = indexOfTopLevelEquals(in: trimmed)
            else { continue }

            let key = String(trimmed[..<eqIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            let valueStr = String(trimmed[trimmed.index(after: eqIdx)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            result[key] = parseValue(valueStr)
        }
        return result
    }

    private func parseValue(_ s: String) -> any Sendable {
        guard let first = s.first else { return "" }

        // Quoted string
        if first == "\"" || first == "'" {
            return unquoteString(s)
        }

        // Boolean
        let lower = s.lowercased()
        if lower == "true" { return true }
        if lower == "false" { return false }
        if lower == "none" || lower == "null" { return NSNull() }

        // Number
        if let intVal = Int(s) { return intVal }
        if let dblVal = Double(s) { return dblVal }

        return s
    }

    private func unquoteString(_ s: String) -> String {
        guard let q = s.first, q == "\"" || q == "'", s.last == q else { return s }
        let inner = s.dropFirst().dropLast()
        var result = ""
        var escape = false
        for ch in inner {
            if escape {
                switch ch {
                case "n": result.append("\n")
                case "t": result.append("\t")
                case "r": result.append("\r")
                case "\\": result.append("\\")
                case "\"": result.append("\"")
                case "'": result.append("'")
                default: result.append(ch)
                }
                escape = false
            } else if ch == "\\" {
                escape = true
            } else {
                result.append(ch)
            }
        }
        return result
    }

    private func splitTopLevel(_ s: String, on sep: Character) -> [String] {
        var result: [String] = []
        var current = ""
        var depth = 0
        var quote: Character?
        var escape = false

        for ch in s {
            if let q = quote {
                current.append(ch)
                if escape {
                    escape = false
                } else if ch == "\\" {
                    escape = true
                } else if ch == q {
                    quote = nil
                }
            } else {
                switch ch {
                case "'", "\"":
                    quote = ch
                    current.append(ch)
                case "(", "[", "{":
                    depth += 1
                    current.append(ch)
                case ")", "]", "}":
                    depth = max(0, depth - 1)
                    current.append(ch)
                default:
                    if ch == sep && depth == 0 {
                        result.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                        current = ""
                    } else {
                        current.append(ch)
                    }
                }
            }
        }

        let final = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !final.isEmpty {
            result.append(final)
        }
        return result
    }

    private func indexOfTopLevelEquals(in s: String) -> String.Index? {
        var i = s.startIndex
        var depthParen = 0
        var depthBrace = 0
        var depthBracket = 0
        var quote: Character?
        var escape = false

        while i < s.endIndex {
            let ch = s[i]
            if let q = quote {
                if escape {
                    escape = false
                } else if ch == "\\" {
                    escape = true
                } else if ch == q {
                    quote = nil
                }
            } else {
                switch ch {
                case "'", "\"": quote = ch
                case "(": depthParen += 1
                case ")": if depthParen > 0 { depthParen -= 1 }
                case "[": depthBracket += 1
                case "]": if depthBracket > 0 { depthBracket -= 1 }
                case "{": depthBrace += 1
                case "}": if depthBrace > 0 { depthBrace -= 1 }
                case "=":
                    if depthParen == 0, depthBrace == 0, depthBracket == 0 {
                        return i
                    }
                default: break
                }
            }
            i = s.index(after: i)
        }
        return nil
    }
}
