// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ToolCallParser Protocol

/// Protocol for parsing tool call content from model output.
///
/// Different models use different formats for tool calls. This protocol provides
/// a common interface for parsing tool calls from model output text.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public protocol ToolCallParser: Sendable {
    /// The start tag that indicates a tool call is beginning.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var startTag: String? { get }

    /// The end tag that indicates a tool call has ended.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var endTag: String? { get }

    /// Parse the content into a `ToolCall`.
    /// - Parameters:
    ///   - content: The text content to parse (may include tags)
    ///   - tools: Optional tool schemas for type-aware parsing
    /// - Returns: A `ToolCall` if parsing succeeds, `nil` otherwise
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?
}

// MARK: - ToolCallFormat Enum

/// Supported tool call formats for different language models.
///
/// This enum defines the various tool call formats used by different LLM families.
/// Each format has its own syntax for encoding function names and arguments.
///
/// The raw string values can be used for JSON serialization or CLI parameters.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public enum ToolCallFormat: String, Sendable, Codable, CaseIterable {
    /// Default JSON format used by Llama, Qwen, and most models.
    /// Example: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
    case json

    /// LFM2 JSON format with model-specific tags.
    /// Example: `<|tool_call_start|>{"name": "func", "arguments": {...}}<|tool_call_end|>`
    case lfm2

    /// XML function format used by Qwen3 Coder.
    /// Example: `<function=name><parameter=key>value</parameter></function>`
    case xmlFunction = "xml_function"

    /// GLM4 format with arg_key/arg_value tags.
    /// Example: `func<arg_key>k</arg_key><arg_value>v</arg_value>`
    case glm4

    /// Gemma function call format.
    /// Example: `call:name{key:value,k:<escape>str<escape>}`
    case gemma

    /// Kimi K2 format with functions prefix.
    /// Example: `functions.name:0<|tool_call_argument_begin|>{"key": "value"}`
    case kimiK2 = "kimi_k2"

    /// MiniMax M2 format with invoke/parameter tags.
    /// Example: `<invoke name="f"><parameter name="k">v</parameter></invoke>`
    case minimaxM2 = "minimax_m2"

    // MARK: - Factory Methods

    /// Create the appropriate parser for this format.
    /// - Returns: A parser instance configured for this format
    public func createParser() -> any ToolCallParser {
        switch self {
        case .json:
            return JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .lfm2:
            return JSONToolCallParser(startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        case .xmlFunction:
            return XMLFunctionParser()
        case .glm4:
            return GLM4ToolCallParser()
        case .gemma:
            return GemmaFunctionParser()
        case .kimiK2:
            return KimiK2ToolCallParser()
        case .minimaxM2:
            return MiniMaxM2ToolCallParser()
        }
    }

    /// Infer the tool call format based on model type from config.json.
    ///
    /// This method maps known model types to their corresponding tool call formats,
    /// enabling automatic format detection when loading models.
    ///
    /// - Parameter modelType: The `model_type` value from config.json
    /// - Returns: The appropriate `ToolCallFormat`, or `nil` to use the default format
    public static func infer(from modelType: String) -> ToolCallFormat? {
        switch modelType.lowercased() {
        case "lfm2", "lfm2_moe":
            return .lfm2
        case "glm4", "glm4_moe", "glm4_moe_lite":
            return .glm4
        case "gemma":
            return .gemma
        default:
            return nil
        }
    }
}
