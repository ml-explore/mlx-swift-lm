// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

        import Foundation
        import os
        import FoundationModels

        /// Converts FoundationModels.GenerationSchema to a JSON string for xgrammar.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        enum SchemaConverter {
            private static let logger = Logger(
                subsystem: "com.apple.FoundationModels-MLX",
                category: "SchemaConverter"
            )

            /// Encodes a GenerationSchema to a standard JSON Schema string.
            ///
            /// `GenerationSchema` is itself `Codable`, and its `encode(to:)` internally
            /// calls `jsonSchema()` and encodes the resulting JSON Schema structure.
            /// So `JSONEncoder().encode(schema)` produces the same JSON bytes as
            /// `JSONEncoder().encode(schema.jsonSchema())` would, without needing
            /// to import the framework that owns the `JSONSchema` type.
            static func encodeToJSON(_ schema: GenerationSchema) throws -> String {
                let data = try JSONEncoder().encode(schema)
                guard let jsonString = String(data: data, encoding: .utf8) else {
                    throw SchemaConversionError.encodingFailed
                }
                logger.debug("Schema JSON (\(data.count) bytes)")
                return jsonString
            }

            /// Builds the JSON Schema describing the tool-calling envelope itself:
            /// a `oneOf` over each supplied tool's `{name, arguments}` shape.
            ///
            /// Shape:
            /// ```
            /// {
            ///   "oneOf": [
            ///     {
            ///       "type": "object",
            ///       "required": ["name", "arguments"],
            ///       "additionalProperties": false,
            ///       "properties": {
            ///         "name": {"const": "<tool name>"},
            ///         "arguments": <tool's parameters schema>
            ///       }
            ///     },
            ///     ...
            ///   ]
            /// }
            /// ```
            ///
            /// This is the *inner* schema -- it describes one tool call JSON object.
            /// For end-to-end grammar generation that also encodes the model's native
            /// tool-call wrapper (e.g. Qwen's `<tool_call>...</tool_call>`), see
            /// `encodeToolCallingGrammar(tools:)`.
            ///
            /// Requires a non-empty tool list.
            static func encodeToolCallingEnvelopeJSON(
                tools: [Transcript.ToolDefinition]
            ) throws -> String {
                let envelope = try toolCallingEnvelopeObject(tools: tools)
                let data = try JSONSerialization.data(withJSONObject: envelope)
                guard let jsonString = String(data: data, encoding: .utf8) else {
                    throw SchemaConversionError.encodingFailed
                }
                logger.debug(
                    "Tool-calling envelope JSON (\(data.count) bytes, \(tools.count) tools)")
                return jsonString
            }

            /// Builds an xgrammar structural-tag JSON that constrains the model
            /// to emit a tool call either wrapped in Qwen-style
            /// `<tool_call>...</tool_call>` delimiters or as bare JSON. The
            /// inner JSON is the envelope produced by
            /// `toolCallingEnvelopeObject` (and serialized by
            /// `encodeToolCallingEnvelopeJSON`).
            ///
            /// Structural-tag shape:
            /// ```json
            /// {
            ///   "type": "structural_tag",
            ///   "format": {
            ///     "type": "or",
            ///     "elements": [
            ///       {
            ///         "type": "tag",
            ///         "begin": "<tool_call>\n",
            ///         "content": { "type": "json_schema", "json_schema": <envelope> },
            ///         "end": ["\n</tool_call>"]
            ///       },
            ///       { "type": "json_schema", "json_schema": <envelope> }
            ///     ]
            ///   }
            /// }
            /// ```
            ///
            /// Accepting both alternatives lets the model stay in its trained
            /// distribution — Qwen-family models overwhelmingly prefer the
            /// wrapped form; the bare arm is a defensive fallback for models
            /// that were trained on raw JSON and happen to share the envelope
            /// shape.
            ///
            /// **Why structural tag over hand-rolled GBNF.** The envelope is a
            /// JSON object whose shape depends on the tool's `parameters`
            /// schema, which varies per tool. Emitting GBNF would require a
            /// Swift-side JSON-schema-to-GBNF compiler — reinventing exactly
            /// what xgrammar's `Grammar::FromJSONSchema` already does in C++.
            /// Structural tag is xgrammar's first-class API for this
            /// multi-format dispatch case; we assemble the dispatch shape in
            /// Swift and let xgrammar compile the embedded JSON schema the
            /// same way the plain `jsonSchema:` path does.
            ///
            /// **Why string literals, not special-token references.** The more
            /// idiomatic structural-tag form for Qwen would use a
            /// `TokenFormat` for `<tool_call>` / `</tool_call>` (Qwen encodes
            /// them as single special tokens). That would require threading
            /// the bound `GrammarTokenizer` through to `Grammar::FromStructuralTag`
            /// for token-string resolution, which the shim entry point
            /// (`xg_compile_structural_tag`) currently declines to do. The
            /// plain-string form is equivalent at the byte level: xgrammar
            /// matches the byte sequence `<tool_call>` against the vocab
            /// mask, finds Qwen's `<tool_call>` special token (whose decoded
            /// bytes are exactly that string), and accepts it.
            ///
            /// Requires a non-empty tool list.
            static func encodeToolCallingGrammar(
                tools: [Transcript.ToolDefinition]
            ) throws -> String {
                let envelope = try toolCallingEnvelopeObject(tools: tools)

                // `json_schema` entries must embed the schema as an inline
                // JSON *object*, not a stringified schema — xgrammar's
                // structural-tag parser rejects stringified schemas outright
                // (see `StructuralTagParser::ParseJSONSchemaFormat`). The
                // envelope is already an `[String: Any]`; pass the same
                // reference into both `or.elements` arms so the emitted JSON
                // round-trips identically on the wrapped and bare sides.
                let jsonSchemaFormat: [String: Any] = [
                    "type": "json_schema",
                    "json_schema": envelope,
                ]
                let structuralTag: [String: Any] = [
                    "type": "structural_tag",
                    "format": [
                        "type": "or",
                        "elements": [
                            [
                                "type": "tag",
                                "begin": "<tool_call>\n",
                                "content": jsonSchemaFormat,
                                "end": ["\n</tool_call>"],
                            ],
                            jsonSchemaFormat,
                        ] as [Any],
                    ] as [String: Any],
                ]

                let data = try JSONSerialization.data(withJSONObject: structuralTag)
                guard let jsonString = String(data: data, encoding: .utf8) else {
                    throw SchemaConversionError.encodingFailed
                }
                logger.debug(
                    "Tool-calling structural-tag JSON (\(data.count) bytes, \(tools.count) tools)"
                )
                return jsonString
            }

            private static func toolCallingEnvelopeObject(
                tools: [Transcript.ToolDefinition]
            ) throws -> [String: Any] {
                guard !tools.isEmpty else {
                    throw SchemaConversionError.noTools
                }

                let encoder = JSONEncoder()
                let oneOf: [[String: Any]] = try tools.map { tool in
                    // Round-trip the tool's parameters through JSONSerialization so we
                    // can embed it as a nested object in the envelope we assemble via
                    // JSONSerialization.data(withJSONObject:). Cheap: schemas are small.
                    let paramsData = try encoder.encode(tool.parameters)
                    let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
                    return [
                        "type": "object",
                        "required": ["name", "arguments"],
                        "additionalProperties": false,
                        "properties": [
                            "name": ["const": tool.name],
                            "arguments": paramsAny,
                        ],
                    ]
                }
                return ["oneOf": oneOf]
            }

            enum SchemaConversionError: Error {
                case encodingFailed
                case noTools
            }
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
