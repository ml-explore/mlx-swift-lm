// Copyright © 2026 Apple Inc.

import Foundation

/// Shared structural scanner for the Qwen XML function dialect:
/// `<function=name><parameter=key>value</parameter>...</function>`.
///
/// `XMLFunctionParser` and `Qwen35ToolCallParser` validate native payloads with
/// this scanner, and the cross-dialect recovery scanner uses the same
/// implementation for candidate extents and argument extraction. There is
/// exactly one definition of a canonical payload, so a malformed or ambiguous
/// payload cannot be promoted through a looser secondary path:
///
/// - the function close is the structural close, not the first textual
///   `</function>` (a parameter value may contain that literal),
/// - every opened `<parameter=` must close with `</parameter>`,
/// - the payload must be consumed in full, apart from whitespace.
enum QwenXMLPayloadScanner {

    /// A canonically valid payload.
    struct Payload {
        let name: String
        /// Parameters in source order. Each value is the verbatim text between
        /// the parameter's `>` and its matching `</parameter>`.
        let parameters: [(name: String, value: Substring)]
        /// Index just past the payload (after `</function>`, or after the last
        /// parameter when a missing close is explicitly allowed).
        let end: String.Index
    }

    enum ScanResult {
        /// The payload is structurally complete and canonical.
        case complete(Payload)
        /// The buffer ended before completeness or malformation could be
        /// decided; more input may still complete it.
        case needMore
        /// A structural violation was found. `consumed` marks the extent that
        /// was examined, suitable for a bounded rejection diagnostic.
        case malformed(consumed: String.Index)
    }

    static let functionOpen = "<function="
    static let functionClose = "</function>"
    static let parameterOpen = "<parameter="
    static let parameterClose = "</parameter>"

    /// Scan a payload that must begin with `<function=`.
    ///
    /// - Parameter allowMissingFunctionClose: when `true`, a payload that is
    ///   canonical except for the final `</function>` also reports
    ///   `.complete`. Used only by permissive end-of-stream repair.
    static func scan(
        _ text: Substring,
        allowMissingFunctionClose: Bool = false
    ) -> ScanResult {
        var index = text.startIndex

        switch matchLiteral(functionOpen, in: text, at: index) {
        case .matched(let after): index = after
        case .needMore: return .needMore
        case .noMatch: return .malformed(consumed: index)
        }

        // Function name: everything up to `>`; non-empty, no whitespace.
        guard let nameEnd = text[index...].firstIndex(of: ">") else { return .needMore }
        let name = text[index ..< nameEnd]
        guard !name.isEmpty, !name.contains(where: \.isWhitespace) else {
            return .malformed(consumed: text.index(after: nameEnd))
        }
        index = text.index(after: nameEnd)

        var parameters: [(name: String, value: Substring)] = []

        while true {
            while index < text.endIndex, text[index].isWhitespace {
                index = text.index(after: index)
            }
            guard index < text.endIndex else {
                if allowMissingFunctionClose {
                    return .complete(
                        Payload(name: String(name), parameters: parameters, end: index))
                }
                return .needMore
            }

            switch matchLiteral(functionClose, in: text, at: index) {
            case .matched(let after):
                return .complete(
                    Payload(name: String(name), parameters: parameters, end: after))
            case .needMore:
                return .needMore
            case .noMatch:
                break
            }

            switch matchLiteral(parameterOpen, in: text, at: index) {
            case .matched(let after):
                index = after
            case .needMore:
                return .needMore
            case .noMatch:
                return .malformed(consumed: index)
            }

            guard let parameterNameEnd = text[index...].firstIndex(of: ">") else {
                return .needMore
            }
            let parameterName = text[index ..< parameterNameEnd]
            guard !parameterName.isEmpty, !parameterName.contains(where: \.isWhitespace) else {
                return .malformed(consumed: text.index(after: parameterNameEnd))
            }
            let valueStart = text.index(after: parameterNameEnd)

            // The parameter value is opaque text up to the first
            // `</parameter>`. An unterminated value may simply need more input;
            // the bounded buffer upstream decides when to give up.
            guard
                let valueEnd = text.range(
                    of: parameterClose, range: valueStart ..< text.endIndex)
            else { return .needMore }
            parameters.append(
                (name: String(parameterName), value: text[valueStart ..< valueEnd.lowerBound]))
            index = valueEnd.upperBound
        }
    }

    /// Whether `payload` is canonical in full: structurally valid with nothing
    /// but whitespace after the closing tag.
    static func isCanonical(_ payload: Substring) -> Bool {
        guard case .complete(let parsed) = scan(payload) else { return false }
        return payload[parsed.end...].allSatisfy(\.isWhitespace)
    }

    /// Parse a payload that must be canonical *in full* into a `ToolCall`.
    ///
    /// This is the only supported way to turn the Qwen XML dialect into an
    /// executable call. Validation and value extraction share one grammar, so a
    /// structurally degraded payload can never be downgraded into a call with
    /// silently missing arguments.
    ///
    /// - Parameter allowMissingFunctionClose: see ``scan(_:allowMissingFunctionClose:)``.
    ///   Only documented end-of-stream repair may set this.
    static func parseCanonical(
        _ payload: Substring,
        tools: [[String: any Sendable]]?,
        allowMissingFunctionClose: Bool = false
    ) -> ToolCall? {
        guard
            case .complete(let parsed) = scan(
                payload, allowMissingFunctionClose: allowMissingFunctionClose),
            payload[parsed.end...].allSatisfy(\.isWhitespace)
        else { return nil }
        return parsed.toolCall(tools: tools)
    }

    /// Strip the optional framing tags from `content`, returning the payload a
    /// parser must then validate in full.
    ///
    /// Returns `nil` when `endTag` appears anywhere other than the very end: the
    /// streaming processor separates trailing content before parsing, so a
    /// direct caller must never have it silently discarded. A close marker
    /// *inside* a parameter value is unaffected, because the trailing marker is
    /// removed as a suffix before that check can apply.
    static func framedPayload(
        in content: String,
        startTag: String?,
        endTag: String?
    ) -> String? {
        var payload = content.trimmingCharacters(in: .whitespacesAndNewlines)

        if let startTag, payload.hasPrefix(startTag) {
            payload.removeFirst(startTag.count)
            payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let endTag {
            if payload.hasSuffix(endTag) {
                payload.removeLast(endTag.count)
                payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
            } else if payload.contains(endTag) {
                return nil
            }
        }

        return payload
    }

    private enum LiteralMatch {
        case matched(after: String.Index)
        case needMore
        case noMatch
    }

    /// Match `literal` at `index`, distinguishing "not here" from "the buffer
    /// ended while it could still complete".
    private static func matchLiteral(
        _ literal: String,
        in text: Substring,
        at index: String.Index
    ) -> LiteralMatch {
        var literalIndex = literal.startIndex
        var textIndex = index
        while literalIndex < literal.endIndex {
            guard textIndex < text.endIndex else { return .needMore }
            guard text[textIndex] == literal[literalIndex] else { return .noMatch }
            literalIndex = literal.index(after: literalIndex)
            textIndex = text.index(after: textIndex)
        }
        return .matched(after: textIndex)
    }
}

extension QwenXMLPayloadScanner.Payload {

    /// The single definition of how a canonical payload becomes call arguments.
    ///
    /// Every caller — the native ``XMLFunctionParser`` and ``Qwen35ToolCallParser``
    /// paths as well as cross-dialect recovery — funnels through this method, so
    /// argument extraction cannot drift between them.
    func toolCall(tools: [[String: any Sendable]]?) -> ToolCall {
        var arguments: [String: any Sendable] = [:]
        for parameter in parameters {
            var value = String(parameter.value)
            // Trim a single leading/trailing newline (matching the Qwen3 Coder
            // reference implementation, which formats values on their own line).
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            arguments[parameter.name] = convertParameterValue(
                value, paramName: parameter.name, funcName: name, tools: tools)
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }
}

/// Structural framing for `<tool_call>...</tool_call>` payloads, shared by the
/// native tagged streaming path and cross-dialect recovery so both always agree
/// on where a frame ends.
///
/// A frame closes only after a structurally complete payload:
///
/// - a JSON payload must balance as one top-level value, honoring string
///   escapes — a literal `</tool_call>` inside a string argument is data and
///   cannot close the frame;
/// - a Qwen XML payload must be canonical per ``QwenXMLPayloadScanner`` — a
///   literal `</function>` inside a parameter value cannot close it.
///
/// Payloads in neither dialect fall back to the first textual close, matching
/// the historical framing the native parsers already used.
enum ToolCallFrameScanner {
    static let startTag = "<tool_call>"
    static let endTag = "</tool_call>"

    /// The index just past the frame's structural close, or `nil` when the
    /// buffer does not yet contain a complete frame. `text` must begin with
    /// `<tool_call>`.
    static func frameEnd(in text: String) -> String.Index? {
        guard text.hasPrefix(startTag) else { return nil }
        var payloadStart = text.index(text.startIndex, offsetBy: startTag.count)
        while payloadStart < text.endIndex, text[payloadStart].isWhitespace {
            payloadStart = text.index(after: payloadStart)
        }
        guard payloadStart < text.endIndex else { return nil }

        if text[payloadStart] == "{" {
            let jsonScanner = JSONLeadingObjectScanner(startCharacter: "{")
            let payloadAndSuffix = String(text[payloadStart...])
            guard let split = jsonScanner.splitLeadingObject(from: payloadAndSuffix) else {
                return nil
            }
            var afterPayload = text.index(payloadStart, offsetBy: split.object.count)
            while afterPayload < text.endIndex, text[afterPayload].isWhitespace {
                afterPayload = text.index(after: afterPayload)
            }
            guard afterPayload < text.endIndex,
                text[afterPayload...].hasPrefix(endTag)
            else { return nil }
            return text.index(afterPayload, offsetBy: endTag.count)
        }

        if text[payloadStart...].hasPrefix(QwenXMLPayloadScanner.functionOpen) {
            guard
                case .complete(let payload) = QwenXMLPayloadScanner.scan(text[payloadStart...])
            else { return nil }
            var afterPayload = payload.end
            while afterPayload < text.endIndex, text[afterPayload].isWhitespace {
                afterPayload = text.index(after: afterPayload)
            }
            guard afterPayload < text.endIndex,
                text[afterPayload...].hasPrefix(endTag)
            else { return nil }
            return text.index(afterPayload, offsetBy: endTag.count)
        }

        // A split `<function=` prefix may still complete; anything else uses
        // the first textual close.
        if QwenXMLPayloadScanner.functionOpen.hasPrefix(String(text[payloadStart...])) {
            return nil
        }
        return text.range(of: endTag).map(\.upperBound)
    }
}
