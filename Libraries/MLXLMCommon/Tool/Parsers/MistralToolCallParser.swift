// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Mistral V13 tool call format: `[TOOL_CALLS]name[ARGS]{"json_args"}`
///
/// This format is used by Mistral3/Ministral-3 2512 models and Devstral 2.
/// The special tokens `[TOOL_CALLS]` (token ID 9) and `[ARGS]` (ID 32) are used
/// as delimiters. Multiple tool calls use repeated `[TOOL_CALLS]` tokens.
///
/// Also handles the older V11 format which includes an optional `[CALL_ID]`
/// between the function name and `[ARGS]` (V13 does not use `[CALL_ID]`).
///
/// Examples:
/// - `[TOOL_CALLS]get_weather[ARGS]{"location": "Tokyo"}`
/// - `[TOOL_CALLS]fn1[ARGS]{...}[TOOL_CALLS]fn2[ARGS]{...}` (multiple calls)
///
/// The end tag is `</s>` (EOS token). Since stop tokens are intercepted at the
/// token ID level before detokenization, the EOS text never reaches the processor
/// — tool calls are extracted via `ToolCallProcessor.flush()` at generation end.
public struct MistralToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "[TOOL_CALLS]"
    public let endTag: String? = "</s>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip [TOOL_CALLS] only when it is a true prefix.
        // ToolCallProcessor.flush() already splits on this token.
        if text.hasPrefix("[TOOL_CALLS]") {
            text = String(text.dropFirst("[TOOL_CALLS]".count))
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Split on [ARGS] to get function name and arguments
        guard let argsRange = text.range(of: "[ARGS]") else {
            return nil
        }

        var namePart = String(text[..<argsRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let argsPart = String(text[argsRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Handle optional [CALL_ID] between name and [ARGS]
        if let callIdRange = namePart.range(of: "[CALL_ID]") {
            namePart = String(namePart[..<callIdRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard !namePart.isEmpty else { return nil }

        // Parse arguments as JSON using deserialize from ParserUtilities
        let arguments = deserialize(argsPart)

        guard let argsDict = arguments as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(
            function: ToolCall.Function(name: namePart, arguments: argsDict)
        )
    }
}
