// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public enum ServerToolParserError: Error, LocalizedError, Equatable {
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let name):
            return "Unsupported tool_call_parser '\(name)'"
        }
    }
}

public enum ServerToolParser {
    public static func resolve(requested: String?, modelType: String?) throws -> ToolCallFormat {
        let normalized = requested?.lowercased().replacingOccurrences(of: "-", with: "_")

        if normalized == nil || normalized == "auto" {
            if let modelType, let inferred = ToolCallFormat.infer(from: modelType) {
                return inferred
            }
            return .json
        }

        switch normalized {
        case "json", "default":
            return .json
        case "lfm2", "lfm2_5", "lfm25":
            return .lfm2
        case "xml", "xml_function", "qwen_xml", "hermes", "nemotron":
            return .xmlFunction
        case "glm4", "glm_4":
            return .glm4
        case "gemma", "gemma4", "gemma_4":
            return .gemma
        case "kimi_k2", "kimi":
            return .kimiK2
        case "minimax_m2", "minimax":
            return .minimaxM2
        case "mistral", "mistral_v11":
            return .mistral
        case "llama3", "llama3_json", "llama_3":
            return .llama3
        case .some(let name):
            throw ServerToolParserError.unsupported(name)
        case .none:
            return .json
        }
    }
}
