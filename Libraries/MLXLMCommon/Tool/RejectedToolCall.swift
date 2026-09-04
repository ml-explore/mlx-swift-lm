// Copyright © 2026 Apple Inc.

import Foundation

/// A tool-call-shaped model output that was deliberately not made executable.
///
/// Rejections are emitted separately from ``ToolCall`` so code handling a
/// `ToolCall` can rely on it having passed parsing and authorization. The raw
/// model output is retained only as a bounded diagnostic preview and is never
/// included in the error description.
public struct RejectedToolCall: Hashable, Codable, Sendable {
    /// Why the model output could not become an executable ``ToolCall``.
    public enum Reason: String, Hashable, Codable, Sendable, CaseIterable {
        /// A complete tool protocol payload could not be parsed.
        case malformedSyntax = "malformed_syntax"

        /// Generation ended after a recognizable tool-call prefix but before
        /// a complete call was available.
        case incompleteOutput = "incomplete_output"

        /// The payload looked like a tool call but omitted its function name.
        case missingToolName = "missing_tool_name"

        /// The call named a function that was not present in the supplied tools.
        case undeclaredTool = "undeclared_tool"

        /// The function arguments could not be represented as a tool input or
        /// definitively violated the tool's declared schema.
        case invalidArguments = "invalid_arguments"

        /// The candidate exceeded the bounded parser budget and was discarded
        /// without execution.
        case resourceLimitExceeded = "resource_limit_exceeded"
    }

    /// The default maximum number of UTF-8 bytes retained in ``rawTextPreview``.
    public static let defaultPreviewByteLimit = 32_768

    /// The semantic reason the output was rejected.
    public let reason: Reason

    /// The protocol used to interpret the model output.
    public let format: ToolCallFormat

    /// The parsed function name, when one could be recovered safely.
    public let toolName: String?

    /// The parsed call identifier, when one could be recovered safely.
    public let callID: String?

    /// A bounded prefix-and-suffix preview of the rejected model output.
    ///
    /// This may contain sensitive argument values. Libraries must not log or
    /// persist it automatically.
    public let rawTextPreview: String

    /// The number of UTF-8 bytes in the complete rejected model output.
    public let rawTextByteCount: Int

    /// Whether ``rawTextPreview`` omits bytes from the middle of the output.
    public let isPreviewTruncated: Bool

    /// A safe diagnostic summary that does not contain raw model output.
    public let detail: String?

    /// Creates a rejection while retaining a bounded diagnostic preview.
    ///
    /// - Parameters:
    ///   - reason: The semantic reason the output was rejected.
    ///   - format: The tool-call protocol used to interpret the output.
    ///   - toolName: The recovered function name, if available.
    ///   - callID: The recovered call identifier, if available.
    ///   - rawText: The complete rejected model output. Only a bounded preview
    ///     is retained by this value.
    ///   - detail: A diagnostic summary that must not contain raw model output
    ///     or sensitive argument values.
    ///   - previewByteLimit: Maximum approximate UTF-8 size of the retained
    ///     preview. The preview preserves both ends and never splits a Swift
    ///     `Character`.
    public init(
        reason: Reason,
        format: ToolCallFormat,
        toolName: String? = nil,
        callID: String? = nil,
        rawText: String,
        detail: String? = nil,
        previewByteLimit: Int = RejectedToolCall.defaultPreviewByteLimit
    ) {
        let preview = Self.makePreview(rawText, byteLimit: previewByteLimit)
        self.reason = reason
        self.format = format
        self.toolName = toolName
        self.callID = callID
        self.rawTextPreview = preview.text
        self.rawTextByteCount = preview.byteCount
        self.isPreviewTruncated = preview.isTruncated
        self.detail = detail
    }

    private static func makePreview(_ text: String, byteLimit: Int) -> (
        text: String, byteCount: Int, isTruncated: Bool
    ) {
        let byteCount = text.utf8.count
        let byteLimit = max(0, byteLimit)
        guard byteCount > byteLimit else {
            return (text, byteCount, false)
        }
        guard byteLimit > 0 else {
            return ("", byteCount, true)
        }

        let prefixBudget = byteLimit / 2
        let suffixBudget = byteLimit - prefixBudget
        let prefix = text.prefixFittingUTF8Bytes(prefixBudget)
        let suffix = text.suffixFittingUTF8Bytes(suffixBudget)
        return (String(prefix) + String(suffix), byteCount, true)
    }
}

extension RejectedToolCall.Reason {
    /// A non-sensitive explanation suitable for public diagnostics.
    var diagnosticDetail: String {
        switch self {
        case .malformedSyntax:
            "The tool-call payload could not be parsed."
        case .incompleteOutput:
            "Generation ended before the tool-call payload was complete."
        case .missingToolName:
            "The tool-call payload did not contain a function name."
        case .undeclaredTool:
            "The function was not present in the supplied tool definitions."
        case .invalidArguments:
            "The function arguments were not a JSON object."
        case .resourceLimitExceeded:
            "The tool-call payload exceeded the parser's safety limit."
        }
    }
}

/// An error used when an API cannot represent a ``RejectedToolCall`` as a
/// structured stream event.
public struct RejectedToolCallError: Error, Hashable, Sendable, LocalizedError {
    public let rejection: RejectedToolCall

    public init(_ rejection: RejectedToolCall) {
        self.rejection = rejection
    }

    public var errorDescription: String? {
        var description = "The model produced a rejected tool call (\(rejection.reason.rawValue))"
        if let detail = rejection.detail {
            description += ": \(detail)"
        } else {
            description += "."
        }
        return description
    }
}

extension String {
    fileprivate func prefixFittingUTF8Bytes(_ byteLimit: Int) -> Substring {
        var byteCount = 0
        var end = startIndex
        while end < endIndex {
            let next = index(after: end)
            let characterByteCount = self[end ..< next].utf8.count
            guard byteCount + characterByteCount <= byteLimit else { break }
            byteCount += characterByteCount
            end = next
        }
        return self[..<end]
    }

    fileprivate func suffixFittingUTF8Bytes(_ byteLimit: Int) -> Substring {
        var byteCount = 0
        var start = endIndex
        while start > startIndex {
            let previous = index(before: start)
            let characterByteCount = self[previous ..< start].utf8.count
            guard byteCount + characterByteCount <= byteLimit else { break }
            byteCount += characterByteCount
            start = previous
        }
        return self[start...]
    }
}
