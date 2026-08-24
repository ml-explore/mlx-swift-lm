// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Pythonic tool call format: [function_name(arg1='value1', arg2='value2')]
/// Used by LFM2.5 and similar models that output tool calls in Python function call syntax.
/// Reference: LiquidAI LFM2.5 chat template format
public struct PythonicToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    /// Python string literals use either quote character, and values may nest
    /// any bracket kind, so every scan of this dialect shares one configuration.
    private static let scanner = StructuredTextScanner(quotes: ["'", "\""])

    public init(startTag: String? = nil, endTag: String? = nil) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseMultiple(content: content, tools: tools).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            return
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .flatMap { parseMultiple(content: $0, tools: tools) }
        } else {
            return parseMultiple(content: toolCallBuffer, tools: tools)
        }
    }

    private func parseMultiple(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        callBodies(in: unwrapped(content)).compactMap { parseCall($0, tools: tools) }
    }

    /// Strips the protocol tags and surrounding whitespace, leaving the call list.
    private func unwrapped(_ content: String) -> Substring {
        var text = content[...]

        if let startTag, let startRange = text.range(of: startTag) {
            text = text[startRange.upperBound...]
        }
        if let endTag, let endRange = text.range(of: endTag) {
            text = text[..<endRange.lowerBound]
        }

        return text.trimmingWhitespace()
    }

    /// Splits a call list into one body per call.
    ///
    /// The list may be wrapped in `[...]`, which has to balance: a bracket left
    /// open means the payload is truncated, not that the calls inside it are
    /// ready to execute. Bodies are separated by top-level commas, so a comma
    /// inside an argument value never splits one call into two.
    private func callBodies(in text: Substring) -> [Substring] {
        var list = text

        if list.first == "[" {
            guard let end = Self.scanner.endOfGroup(in: list, openedAt: list.startIndex)
            else { return [] }
            list = list[list.index(after: list.startIndex) ..< end]
        }

        return Self.scanner.splitTopLevel(list, separator: ",")
    }

    /// Parses one `name(arguments)` body.
    ///
    /// The argument list ends at the parenthesis balancing the one that opened
    /// it, so a value containing `)`, `]`, or `)]` survives intact.
    private func parseCall(_ body: Substring, tools: [[String: any Sendable]]?) -> ToolCall? {
        let body = body.trimmingWhitespace()
        guard let open = Self.scanner.firstTopLevelIndex(of: "(", in: body),
            let close = Self.scanner.endOfGroup(in: body, openedAt: open)
        else { return nil }

        let name = identifierEnding(at: open, in: body)
        guard !name.isEmpty else { return nil }

        let funcName = String(name)
        let arguments = parseArguments(
            String(body[body.index(after: open) ..< close]), funcName: funcName, tools: tools)
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// The identifier run ending just before `index`, which skips whatever list
    /// punctuation or qualifier the model emitted ahead of the call.
    private func identifierEnding(at index: String.Index, in body: Substring) -> Substring {
        let prefix = body[..<index]
        let isIdentifier: (Character) -> Bool = { $0.isLetter || $0.isNumber || $0 == "_" }
        guard let boundary = prefix.lastIndex(where: { !isIdentifier($0) }) else { return prefix }
        return prefix[prefix.index(after: boundary)...]
    }

    /// Parse Pythonic keyword arguments: arg1='value1', arg2="value2", arg3=123
    ///
    /// Values may be JSON or Python literals, including single-quoted collections
    /// and the `True`, `False`, and `None` spellings. Splitting is bracket-, brace-,
    /// and quote-aware so commas inside a value do not truncate it.
    private func parseArguments(
        _ argsString: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]

        for pair in Self.scanner.splitTopLevel(argsString[...], separator: ",") {
            guard let eq = Self.scanner.firstTopLevelIndex(of: "=", in: pair) else { continue }
            let key = String(pair[..<eq]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            let value = pair[pair.index(after: eq)...].trimmingWhitespace()
            arguments[key] = parseArgumentValue(
                value, key: key, funcName: funcName, tools: tools)
        }

        return unwrapArgumentWrapper(arguments, funcName: funcName, tools: tools)
    }

    /// Converts one argument without letting inferred scalar types override a
    /// declared schema. Collections retain the parser's historical eager
    /// decoding behavior, now with Python-literal syntax as a fallback.
    private func parseArgumentValue(
        _ value: Substring,
        key: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        let literal = String(value)

        if value.first == "{" || value.first == "[" {
            return tryParseJSON(literal) ?? tryParsePythonLiteral(literal) ?? literal
        }

        if value.first == "'" || value.first == "\"" {
            let string = (tryParsePythonLiteral(literal) as? String) ?? literal
            return convertParameterValue(
                string, paramName: key, funcName: funcName, tools: tools)
        }

        if getParameterType(funcName: funcName, paramName: key, tools: tools) != nil {
            return convertParameterValue(
                literal, paramName: key, funcName: funcName, tools: tools)
        }

        return tryParsePythonLiteral(literal) ?? literal
    }

    /// Some models wrap all arguments in a single object under a schema key —
    /// e.g. LFM2 emits `get_weather(properties={"location": "Tokyo"})`, mirroring
    /// the JSON-schema `properties` container. When the call has exactly one
    /// argument, its value is an object, and its key is a recognized wrapper name
    /// that is not itself a declared parameter, treat the inner object as the
    /// arguments. Restricted to wrapper names so a genuine object-valued argument
    /// (e.g. `configure(settings={...})`) is preserved untouched.
    private func unwrapArgumentWrapper(
        _ arguments: [String: any Sendable],
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        guard arguments.count == 1,
            let (key, value) = arguments.first,
            let object = value as? [String: any Sendable]
        else { return arguments }

        let wrapperKeys: Set<String> = ["properties", "parameters", "arguments", "args", "kwargs"]
        guard wrapperKeys.contains(key.lowercased()) else { return arguments }

        // If the wrapper name is genuinely a declared parameter, keep as-is.
        if getParameterType(funcName: funcName, paramName: key, tools: tools) != nil {
            return arguments
        }
        return object
    }
}
