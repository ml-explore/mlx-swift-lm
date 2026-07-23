// Copyright © 2026 Apple Inc.

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
    ///   ],
    ///   "$defs": { "<tool name>__<def name>": ... }
    /// }
    /// ```
    ///
    /// If a tool's parameters schema carries `$defs` (named sub-schemas such
    /// as nested `@Generable` types), they are hoisted to the envelope root
    /// under per-tool namespaced keys, with that tool's `$ref`s rewritten to
    /// match — JSON Pointers resolve from the document root, so defs left
    /// nested inside `arguments` would leave every ref dangling.
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
    /// `<tool_call>...</tool_call>` delimiters or as bare JSON.
    ///
    /// Each tool is its own `tag` whose `begin` is the literal prefix
    /// `{"name": "<tool>", "arguments": ` and whose `content` is that
    /// tool's parameter schema, closed by an `end` of `}`. Structural-tag
    /// shape:
    /// ```json
    /// {
    ///   "type": "structural_tag",
    ///   "format": {
    ///     "type": "or",
    ///     "elements": [
    ///       {
    ///         "type": "tag",
    ///         "begin": "<tool_call>\n",
    ///         "content": <per-tool or>,
    ///         "end": ["\n</tool_call>"]
    ///       },
    ///       <per-tool or>
    ///     ]
    ///   }
    /// }
    /// ```
    /// where `<per-tool or>` is an `or` over one `tag` per tool:
    /// ```json
    /// {
    ///   "type": "tag",
    ///   "begin": "{\"name\": \"set_flashlight\", \"arguments\": ",
    ///   "content": { "type": "json_schema", "json_schema": <tool params> },
    ///   "end": ["}"]
    /// }
    /// ```
    ///
    /// **Why per-tool tags instead of one `oneOf` json_schema.** The
    /// earlier shape embedded a single `{oneOf: [{name, arguments}, …]}`
    /// json_schema in each arm. The structural-tag path compiles that
    /// embedded schema with xgrammar's default (non-strict) property
    /// ordering, so greedy decoding could open `"arguments"` before
    /// `"name"` and dive into an unbounded free-text field before ever
    /// committing to a tool -- producing a nameless, unparseable buffer
    /// that ran the token budget dry (observed: Qwen filling `response`
    /// with `"1234567890…"`). Making the name a literal tag prefix forces
    /// the model to commit to a specific tool first, then fill only that
    /// tool's arguments. It also removes the JSON whitespace wiggle room
    /// around the `name`/`arguments` keys that open-source models tend to
    /// exploit into long whitespace runs.
    ///
    /// Accepting both wrapped and bare arms lets the model stay in its
    /// trained distribution -- Qwen-family models overwhelmingly prefer
    /// the wrapped form; the bare arm is a defensive fallback for models
    /// trained on raw JSON.
    ///
    /// **Why structural tag over hand-rolled GBNF.** Each tool's
    /// `arguments` is a JSON object whose shape depends on the tool's
    /// `parameters` schema. Emitting GBNF for it would require a
    /// Swift-side JSON-schema-to-GBNF compiler -- reinventing what
    /// xgrammar's `Grammar::FromJSONSchema` already does in C++.
    /// Structural tag composes the fixed dispatch prefix with the
    /// per-tool json_schema and lets xgrammar compile the embedded
    /// schema the same way the plain `jsonSchema:` path does.
    ///
    /// Requires a non-empty tool list.
    static func encodeToolCallingGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        // One tag per tool. `begin` fixes `{"name": "<tool>", "arguments": `
        // so the tool name is committed before the arguments schema opens;
        // `content` is the tool's parameter schema; `end` closes the object.
        let toolTags: [[String: Any]] = try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
            let nameData = try JSONSerialization.data(
                withJSONObject: tool.name, options: [.fragmentsAllowed])
            let nameJSON = String(data: nameData, encoding: .utf8) ?? "\"\(tool.name)\""
            return [
                "type": "tag",
                "begin": "{\"name\": \(nameJSON), \"arguments\": ",
                "content": [
                    "type": "json_schema",
                    "json_schema": paramsAny,
                ],
                "end": ["}"],
            ]
        }
        let perToolOr: [String: Any] = [
            "type": "or",
            "elements": toolTags,
        ]

        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "tag",
                        "begin": "<tool_call>\n",
                        "content": perToolOr,
                        "end": ["\n</tool_call>"],
                    ],
                    perToolOr,
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
        // `GenerationSchema` serializes named sub-schemas (e.g. a nested
        // `@Generable` type, or a named `DynamicGenerationSchema`) as
        // root-level `$defs` plus root-anchored `"$ref": "#/$defs/..."`
        // pointers. Embedding a tool's schema as a nested object under
        // `oneOf[i].properties.arguments` buries its `$defs` inside
        // `arguments` while the refs stay anchored to the document root —
        // and xgrammar resolves JSON Pointers from the document root, so
        // every ref dangles and grammar compilation hard-fails
        // ("Cannot find field $defs in {\"oneOf\": ...",
        // json_schema_converter.cc). Hoist each tool's `$defs` to the
        // envelope root instead, namespacing keys per tool
        // (`<tool>__<def>`) so same-named defs across tools cannot collide.
        var hoistedDefs: [String: Any] = [:]
        let oneOf: [[String: Any]] = try tools.map { tool in
            // Round-trip the tool's parameters through JSONSerialization so we
            // can embed it as a nested object in the envelope we assemble via
            // JSONSerialization.data(withJSONObject:). Cheap: schemas are small.
            let paramsData = try encoder.encode(tool.parameters)
            var paramsAny = rewriteDefsRefs(
                in: try JSONSerialization.jsonObject(with: paramsData),
                toolName: tool.name
            )
            if var paramsObj = paramsAny as? [String: Any] {
                if let defs = paramsObj.removeValue(forKey: "$defs") as? [String: Any] {
                    for (key, value) in defs {
                        hoistedDefs["\(tool.name)__\(key)"] = value
                    }
                }
                paramsAny = paramsObj
            }
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
        var envelope: [String: Any] = ["oneOf": oneOf]
        if !hoistedDefs.isEmpty {
            envelope["$defs"] = hoistedDefs
        }
        return envelope
    }

    /// Rewrites every `"$ref": "#/$defs/<name>"` in a parsed schema tree to
    /// the per-tool namespaced key (`#/$defs/<tool>__<name>`).
    ///
    /// The rewrite is structure-aware: only the string value directly under a
    /// `$ref` key is touched, so other strings that merely mention the
    /// pointer text — a `description`, `const`, `enum` entry, `default`, or
    /// `pattern` containing "#/$defs/" — survive verbatim. Runs before the
    /// `$defs` are hoisted out, and recurses through the whole tree because
    /// refs can appear anywhere, including inside other `$defs` bodies.
    private static func rewriteDefsRefs(in value: Any, toolName: String) -> Any {
        switch value {
        case let object as [String: Any]:
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, nested) in object {
                if key == "$ref", let ref = nested as? String, ref.hasPrefix("#/$defs/") {
                    result[key] = "#/$defs/\(toolName)__" + ref.dropFirst("#/$defs/".count)
                } else {
                    result[key] = rewriteDefsRefs(in: nested, toolName: toolName)
                }
            }
            return result
        case let array as [Any]:
            return array.map { rewriteDefsRefs(in: $0, toolName: toolName) }
        default:
            return value
        }
    }

    enum SchemaConversionError: Error {
        case encodingFailed
        case noTools
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
