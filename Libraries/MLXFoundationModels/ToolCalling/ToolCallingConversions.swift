// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import MLXLMCommon
        import FoundationModels

        /// Conversions from FoundationModels tool definitions to the OpenAI-style
        /// function-envelope dict shape that MLXLMCommon's
        /// `Tokenizer.applyChatTemplate(messages:tools:)` expects for its `tools:`
        /// parameter.
        ///
        /// MLXLMCommon's chat template surface uses `[String: any Sendable]` so the
        /// dictionaries can cross actor boundaries. These factories bridge our
        /// strongly-typed Swift representations into that form without leaking `Any`
        /// into the rest of the codebase.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        enum ToolCallingConversions {

            /// Converts a `Transcript.ToolDefinition` to the OpenAI-style function
            /// envelope that MLXLMCommon chat templates (including Qwen, Llama,
            /// Phi, Gemma) are trained to expect:
            /// ```
            /// {
            ///   "type": "function",
            ///   "function": {
            ///     "name": "<tool name>",
            ///     "description": "<tool description>",
            ///     "parameters": <JSON Schema>
            ///   }
            /// }
            /// ```
            static func makeToolSpec(from tool: Transcript.ToolDefinition) throws -> [String:
                any Sendable]
            {
                let schema: GenerationSchema = tool.parameters
                let paramsData = try JSONEncoder().encode(schema)
                guard
                    let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
                        as? [String: any Sendable]
                else {
                    throw ToolCallingConversionError.invalidParameterSchema
                }

                return [
                    "type": "function",
                    "function": [
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": paramsAny,
                    ] as [String: any Sendable],
                ]
            }

            /// Converts an array of tool definitions, preserving order. Throws on the
            /// first conversion failure (unexpected -- `GenerationSchema` is `Codable`
            /// and tool parameter schemas should always encode cleanly).
            static func makeToolSpecs(from tools: [Transcript.ToolDefinition]) throws -> [[String:
                any Sendable]]
            {
                try tools.map(makeToolSpec(from:))
            }

            enum ToolCallingConversionError: Error {
                case invalidParameterSchema
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
