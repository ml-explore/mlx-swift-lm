// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// A partial schema evaluator is safe only when it distinguishes a proven
/// result from one that depends on unsupported semantics. These tests pin all
/// three outcomes and the executable-call boundary that consumes them.
struct ToolSchemaValidatorTests {

    private func tool(
        _ name: String, parameters: [String: any Sendable]? = nil
    ) -> [String: any Sendable] {
        var function: [String: any Sendable] = ["name": name]
        if let parameters { function["parameters"] = parameters }
        return ["type": "function", "function": function]
    }

    private var weatherSchema: [String: any Sendable] {
        [
            "type": "object",
            "properties": [
                "city": ["type": "string", "minLength": 1] as [String: any Sendable],
                "limit": ["type": "integer", "minimum": 1, "maximum": 50]
                    as [String: any Sendable],
                "units": ["type": "string", "enum": ["celsius", "fahrenheit"]]
                    as [String: any Sendable],
            ],
            "required": ["city"],
        ]
    }

    private func descriptions(_ result: ToolSchemaValidator.Result) -> [String] {
        guard case .invalid(let violations) = result else { return [] }
        return violations.map(\.description)
    }

    // MARK: - Types and objects

    @Test("Arguments that satisfy every understood assertion are valid")
    func validArgumentsPass() {
        #expect(
            ToolSchemaValidator.validate(
                arguments: [
                    "city": .string("Paris"), "limit": .int(5),
                    "units": .string("celsius"),
                ],
                against: weatherSchema) == .valid)
    }

    @Test("A wrong type is a proven violation at the argument path")
    func wrongTypeIsReported() {
        let result = ToolSchemaValidator.validate(
            arguments: ["city": .string("Paris"), "limit": .string("five")],
            against: weatherSchema)
        #expect(descriptions(result) == ["arguments.limit must be an integer"])
    }

    @Test("An integral double satisfies integer while a fractional value does not")
    func integerAcceptsIntegralDoubles() {
        let schema: [String: any Sendable] = ["type": "integer"]
        #expect(ToolSchemaValidator.validate(.int(3), against: schema) == .valid)
        #expect(ToolSchemaValidator.validate(.double(3.0), against: schema) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.double(3.5), against: schema))
                == ["$ must be an integer"])
    }

    @Test("A type list accepts any declared type")
    func typeList() {
        let schema: [String: any Sendable] = ["type": ["string", "null"]]
        #expect(ToolSchemaValidator.validate(.string("x"), against: schema) == .valid)
        #expect(ToolSchemaValidator.validate(.null, against: schema) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(1), against: schema))
                == ["$ must be one of these types: string, null"])
    }

    @Test("A missing required argument is reported by name")
    func missingRequiredIsReported() {
        let result = ToolSchemaValidator.validate(
            arguments: ["limit": .int(3)], against: weatherSchema)
        #expect(descriptions(result) == ["arguments.city is required"])
    }

    @Test("Nested objects report a stable full path")
    func nestedPathIsReported() {
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "filters": [
                    "type": "object",
                    "properties": ["limit": ["type": "integer"]],
                    "required": ["limit"],
                ] as [String: any Sendable]
            ],
        ]
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    arguments: ["filters": .object([:])], against: schema))
                == ["arguments.filters.limit is required"])
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    arguments: ["filters": .object(["limit": .string("x")])],
                    against: schema))
                == ["arguments.filters.limit must be an integer"])
    }

    @Test("Diagnostic paths quote property names that are not identifiers")
    func nonIdentifierPath() {
        let schema: [String: any Sendable] = [
            "properties": ["postal-code": ["type": "integer"]]
        ]
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    arguments: ["postal-code": .string("x")], against: schema))
                == [#"arguments["postal-code"] must be an integer"#])
    }

    @Test("Diagnostic paths JSON-escape control and punctuation characters")
    func pathEscaping() {
        let key = "line\n\"\\"
        let schema: [String: any Sendable] = [
            "properties": [key: ["type": "integer"]]
        ]
        let description = descriptions(
            ToolSchemaValidator.validate(
                arguments: [key: .string("x")], against: schema)
        ).first
        #expect(description == #"arguments["line\n\"\\"] must be an integer"#)
        #expect(description?.contains("\n") == false)
    }

    @Test("additionalProperties false applies without a properties keyword")
    func additionalPropertiesWithoutProperties() {
        let schema: [String: any Sendable] = [
            "type": "object", "additionalProperties": false,
        ]
        #expect(
            descriptions(
                ToolSchemaValidator.validate(arguments: ["extra": .int(1)], against: schema))
                == ["arguments.extra is not a declared property"])
    }

    @Test("additionalProperties checks undeclared values against its schema")
    func additionalPropertiesSchema() {
        let schema: [String: any Sendable] = [
            "properties": ["city": ["type": "string"]],
            "additionalProperties": ["type": "integer"],
        ]
        #expect(
            ToolSchemaValidator.validate(arguments: ["count": .int(2)], against: schema)
                == .valid)
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    arguments: ["count": .string("two")], against: schema))
                == ["arguments.count must be an integer"])
    }

    @Test("Malformed properties make additional-property classification unknown")
    func malformedPropertiesAreUnknown() {
        let schema: [String: any Sendable] = [
            "properties": "not an object", "additionalProperties": false,
        ]
        #expect(
            ToolSchemaValidator.validate(arguments: ["x": .int(1)], against: schema)
                == .unknown)
    }

    @Test("Malformed nested schema values are unknown even when not traversed")
    func malformedSubschemasAreUnknown() {
        let cases: [(JSONValue, [String: any Sendable])] = [
            (.object([:]), ["properties": ["unused": "not a schema"]]),
            (.object([:]), ["additionalProperties": "not a schema"]),
            (.array([]), ["items": "not a schema"]),
        ]
        for (value, schema) in cases {
            #expect(ToolSchemaValidator.validate(value, against: schema) == .unknown)
        }
    }

    @Test("patternProperties prevents an unproven additional-property rejection")
    func patternPropertiesInteractionIsUnknown() {
        let schema: [String: any Sendable] = [
            "patternProperties": ["^x-": ["type": "integer"]],
            "additionalProperties": false,
        ]
        #expect(
            ToolSchemaValidator.validate(arguments: ["x-count": .int(1)], against: schema)
                == .unknown)
    }

    // MARK: - Arrays and strings

    @Test("Array items, count limits, and uniqueness are enforced")
    func arrayRules() {
        let schema: [String: any Sendable] = [
            "type": "array", "items": ["type": "string"],
            "minItems": 1, "maxItems": 3, "uniqueItems": true,
        ]
        #expect(ToolSchemaValidator.validate(.array([.string("a")]), against: schema) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.array([]), against: schema))
                == ["$ must have at least 1 item"])
        #expect(
            descriptions(
                ToolSchemaValidator.validate(.array([.string("a"), .int(1)]), against: schema))
                == ["$[1] must be a string"])
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    .array([.string("a"), .string("a")]), against: schema))
                == ["$ must not contain duplicate items"])
    }

    @Test("JSON numeric equality detects 1 and 1.0 as duplicate items")
    func uniqueItemsUsesJSONNumberEquality() {
        let schema: [String: any Sendable] = ["uniqueItems": true]
        #expect(
            descriptions(
                ToolSchemaValidator.validate(.array([.int(1), .double(1.0)]), against: schema))
                == ["$ must not contain duplicate items"])

        let first: JSONValue = .object([
            "a": .int(1), "b": .array([.bool(true), .null]),
        ])
        let reordered: JSONValue = .object([
            "b": .array([.bool(true), .null]), "a": .double(1.0),
        ])
        #expect(
            descriptions(
                ToolSchemaValidator.validate(
                    .array([first, reordered]), against: schema))
                == ["$ must not contain duplicate items"])
    }

    @Test("uniqueItems handles a large distinct array without quadratic scanning")
    func uniqueItemsScalesWithDistinctValues() {
        let values = (0 ..< 4_096).map(JSONValue.int)
        #expect(
            ToolSchemaValidator.validate(
                .array(values), against: ["uniqueItems": true]) == .valid)
    }

    @Test("Tuple-form items are unknown rather than approximated")
    func tupleItemsAreUnknown() {
        let schema: [String: any Sendable] = [
            "items": [["type": "string"], ["type": "integer"]]
        ]
        #expect(
            ToolSchemaValidator.validate(.array([.string("x"), .int(1)]), against: schema)
                == .unknown)
    }

    @Test("prefixItems prevents items from being applied to prefix elements")
    func prefixItemsInteractionIsUnknown() {
        let schema: [String: any Sendable] = [
            "prefixItems": [["type": "string"]], "items": ["type": "integer"],
        ]
        #expect(
            ToolSchemaValidator.validate(
                .array([.string("prefix"), .int(1)]), against: schema) == .unknown)
    }

    @Test("String bounds count Unicode code points rather than grapheme clusters")
    func stringLengthUsesUnicodeCodePoints() {
        let exactlyTwo: [String: any Sendable] = [
            "type": "string", "minLength": 2, "maxLength": 2,
        ]
        let decomposed = "e\u{301}"
        #expect(decomposed.count == 1)
        #expect(decomposed.unicodeScalars.count == 2)
        #expect(ToolSchemaValidator.validate(.string(decomposed), against: exactlyTwo) == .valid)
    }

    @Test("Malformed count limits are unknown and never trap")
    func malformedCountLimitsAreUnknown() {
        let cases: [(JSONValue, [String: any Sendable])] = [
            (.array([]), ["minItems": -1]),
            (.array([]), ["maxItems": Double.infinity]),
            (.string(""), ["minLength": "two"]),
        ]
        for (value, schema) in cases {
            #expect(ToolSchemaValidator.validate(value, against: schema) == .unknown)
        }
    }

    // MARK: - Numbers and constants

    @Test("Inclusive and exclusive bounds support modern and draft-4 forms")
    func numberBounds() {
        let inclusive: [String: any Sendable] = ["minimum": 1, "maximum": 10]
        #expect(ToolSchemaValidator.validate(.int(1), against: inclusive) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(0), against: inclusive))
                == ["$ must be at least 1"])

        let modern: [String: any Sendable] = [
            "exclusiveMinimum": 0, "exclusiveMaximum": 1,
        ]
        #expect(ToolSchemaValidator.validate(.double(0.5), against: modern) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(0), against: modern))
                == ["$ must be greater than 0"])

        let draft4: [String: any Sendable] = ["minimum": 0, "exclusiveMinimum": true]
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(0), against: draft4))
                == ["$ must be greater than 0"])
        #expect(ToolSchemaValidator.validate(.int(1), against: draft4) == .valid)
    }

    @Test("A modern exclusive bound does not replace its inclusive sibling")
    func simultaneousBoundsBothApply() {
        let schema: [String: any Sendable] = ["minimum": 10, "exclusiveMinimum": 0]
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(5), against: schema))
                == ["$ must be at least 10"])
    }

    @Test("Large integer comparisons do not lose precision through Double")
    func largeIntegerBoundsRemainExact() {
        let maximum = 9_007_199_254_740_992
        let schema: [String: any Sendable] = ["maximum": maximum]
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(maximum + 1), against: schema))
                == ["$ must be at most \(maximum)"])
    }

    @Test("Finite numbers outside Decimal's range are unknown")
    func numbersOutsideDecimalRangeAreUnknown() {
        let value = JSONValue.double(Double.greatestFiniteMagnitude)
        #expect(
            ToolSchemaValidator.validate(value, against: ["maximum": 1]) == .unknown)
        #expect(
            ToolSchemaValidator.validate(
                value, against: ["const": Double.greatestFiniteMagnitude]) == .unknown)
        #expect(
            ToolSchemaValidator.validate(
                .array([value, value]), against: ["uniqueItems": true]) == .unknown)
    }

    @Test("enum keeps Boolean and numeric identity distinct")
    func enumDoesNotConfuseBoolAndInt() {
        let schema: [String: any Sendable] = ["enum": [true]]
        #expect(ToolSchemaValidator.validate(.bool(true), against: schema) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(1), against: schema))
                == ["$ must be one of the allowed values"])
    }

    @Test("A malformed enum member prevents a proof even when another member matches")
    func malformedEnumIsUnknown() {
        let members: [any Sendable] = ["match", Date()]
        let schema: [String: any Sendable] = ["enum": members]
        #expect(ToolSchemaValidator.validate(.string("match"), against: schema) == .unknown)
    }

    @Test("enum and const compare JSON numbers by mathematical value")
    func numericEquality() {
        #expect(
            ToolSchemaValidator.validate(.double(1.0), against: ["const": 1]) == .valid)
        #expect(
            ToolSchemaValidator.validate(.int(1), against: ["enum": [1.0]]) == .valid)
    }

    // MARK: - Proof and uncertainty

    @Test("Unsupported assertions produce unknown")
    func unsupportedAssertionsAreUnknown() {
        let schemas: [[String: any Sendable]] = [
            ["type": "string", "pattern": "[0-9]+"],
            ["$ref": "#/$defs/Address"],
            ["not": ["type": "null"] as [String: any Sendable]],
        ]
        for schema in schemas {
            #expect(ToolSchemaValidator.validate(.string("anything"), against: schema) == .unknown)
        }
    }

    @Test("A reference makes sibling assertions unknown across schema dialects")
    func referenceSiblingIsUnknown() {
        let schema: [String: any Sendable] = [
            "$ref": "#/$defs/value", "type": "integer",
            "$defs": ["value": ["type": "string"]],
        ]
        #expect(ToolSchemaValidator.validate(.string("x"), against: schema) == .unknown)
    }

    @Test("Annotations and schema definitions do not create uncertainty")
    func annotationsAreNonAssertive() {
        let schema: [String: any Sendable] = [
            "type": "string", "description": "value", "default": "x",
            "$defs": ["Unused": ["type": "integer"]], "x-order": ["value"],
        ]
        #expect(ToolSchemaValidator.validate(.string("x"), against: schema) == .valid)
    }

    @Test("A known sibling violation remains invalid beside an unknown assertion")
    func provenViolationDominatesUnknown() {
        let schema: [String: any Sendable] = [
            "type": "integer", "pattern": "unsupported here",
        ]
        #expect(
            descriptions(ToolSchemaValidator.validate(.string("x"), against: schema))
                == ["$ must be an integer"])
    }

    @Test("oneOf branches with unsupported distinctions remain unknown")
    func oneOfDoesNotInventMatches() {
        let schema: [String: any Sendable] = [
            "oneOf": [
                ["type": "string", "pattern": "^a"],
                ["type": "string", "pattern": "^b"],
            ]
        ]
        #expect(ToolSchemaValidator.validate(.string("apple"), against: schema) == .unknown)
    }

    @Test("oneOf is invalid when two branches are proven matches")
    func oneOfProvenMultipleMatches() {
        let schema: [String: any Sendable] = [
            "oneOf": [
                ["type": "integer"], ["type": "number"], ["pattern": "unknown"],
            ]
        ]
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(1), against: schema))
                == ["$ must satisfy exactly one of the allowed schemas"])
    }

    @Test("anyOf and allOf propagate proof and uncertainty correctly")
    func combinatorProofRules() {
        let validAnyOf: [String: any Sendable] = [
            "anyOf": [["type": "string"], ["pattern": "unknown"]]
        ]
        #expect(ToolSchemaValidator.validate(.string("x"), against: validAnyOf) == .valid)

        let unknownAnyOf: [String: any Sendable] = [
            "anyOf": [["type": "integer"], ["pattern": "unknown"]]
        ]
        #expect(ToolSchemaValidator.validate(.string("x"), against: unknownAnyOf) == .unknown)

        let invalidAllOf: [String: any Sendable] = [
            "allOf": [["type": "integer"], ["pattern": "unknown"]]
        ]
        #expect(
            descriptions(ToolSchemaValidator.validate(.string("x"), against: invalidAllOf))
                == ["$ must be an integer"])
    }

    @Test("Boolean and malformed schemas preserve the proof boundary")
    func schemaShapes() {
        #expect(ToolSchemaValidator.validate(.int(1), against: true) == .valid)
        #expect(
            descriptions(ToolSchemaValidator.validate(.int(1), against: false))
                == ["$ is not permitted"])
        #expect(ToolSchemaValidator.validate(.int(1), against: "not a schema") == .unknown)
        #expect(
            ToolSchemaValidator.validate(.int(1), against: ["type": "unsupported"])
                == .unknown)
    }

    @Test("Excessive schema nesting becomes unknown rather than recursing without bound")
    func schemaDepthIsBounded() {
        var schema: [String: any Sendable] = ["type": "integer"]
        for _ in 0 ... 64 {
            schema = ["allOf": [schema]]
        }
        #expect(ToolSchemaValidator.validate(.int(1), against: schema) == .unknown)
    }

    // MARK: - Lookup, diagnostics, and executable boundary

    @Test("Both tool declaration shapes expose their parameters schema")
    func parametersSchemaLookup() {
        let openAI = tool("f", parameters: ["type": "object"])
        let flat: [String: any Sendable] = ["name": "g", "parameters": ["type": "object"]]
        #expect(ToolSchemaValidator.parametersSchema(ofToolNamed: "f", in: [openAI]) != nil)
        #expect(ToolSchemaValidator.parametersSchema(ofToolNamed: "g", in: [flat]) != nil)
        #expect(ToolSchemaValidator.parametersSchema(ofToolNamed: "h", in: [openAI]) == nil)
    }

    @Test("A malformed declared parameters schema is unknown, not absent")
    func malformedParametersSchemaIsUnknown() {
        let malformed: [String: any Sendable] = [
            "function": ["name": "f", "parameters": "not a schema"]
                as [String: any Sendable]
        ]
        #expect(
            ToolSchemaValidator.validate(
                arguments: [:], forToolNamed: "f", in: [malformed]) == .unknown)
    }

    @Test("describe is bounded and never includes argument values")
    func describeBounds() {
        let result = ToolSchemaValidator.validate(
            arguments: [
                "city": .int(1), "limit": .string("secret-value"),
                "units": .string("kelvin"),
            ],
            against: weatherSchema)
        guard case .invalid(let violations) = result else {
            Issue.record("Expected schema violations, got \(result)")
            return
        }
        let summary = ToolSchemaValidator.describe(violations, limit: 2)
        #expect(summary.hasSuffix("; and \(violations.count - 2) more"))
        #expect(!summary.contains("secret-value"))
        #expect(!summary.contains("kelvin"))
        #expect(ToolSchemaValidator.describe(violations, limit: -1) == "3 schema violations")
    }

    @Test("describe bounds a single adversarially long path")
    func describeBoundsLongPaths() {
        let key = String(repeating: "x", count: 2_048)
        let result = ToolSchemaValidator.validate(
            arguments: [key: .int(1)],
            against: ["additionalProperties": false])
        guard case .invalid(let violations) = result else {
            Issue.record("Expected a schema violation, got \(result)")
            return
        }
        let summary = ToolSchemaValidator.describe(violations)
        #expect(summary.unicodeScalars.count == 512)
        #expect(summary.hasSuffix("…"))
    }

    @Test("The processor rejects only a proven schema violation")
    func processorProofBoundary() {
        let invalid = ToolCallProcessor(
            format: .json, tools: [tool("weather", parameters: weatherSchema)])
        _ = invalid.processChunk(
            #"<tool_call>{"name":"weather","arguments":{"city":"Paris","limit":"five"}}</tool_call>"#
        )
        #expect(invalid.toolCalls.isEmpty)
        #expect(invalid.rejectedToolCalls.first?.reason == .invalidArguments)
        #expect(invalid.rejectedToolCallCount == 1)

        let uncertainSchema: [String: any Sendable] = [
            "type": "object",
            "properties": ["code": ["type": "string", "pattern": "^[0-9]+$"]],
        ]
        let uncertain = ToolCallProcessor(
            format: .json, tools: [tool("submit", parameters: uncertainSchema)])
        _ = uncertain.processChunk(
            #"<tool_call>{"name":"submit","arguments":{"code":"abc"}}</tool_call>"#)
        #expect(uncertain.toolCalls.map(\.function.name) == ["submit"])
        #expect(uncertain.rejectedToolCalls.isEmpty)
    }

    @Test("A missing schema or tool declaration preserves existing fail-open behavior")
    func absentSchemaPasses() {
        #expect(
            ToolSchemaValidator.validate(arguments: ["x": .int(1)], against: nil) == .valid)

        let withoutParameters = ToolCallProcessor(
            format: .json, tools: [tool("free_form")])
        _ = withoutParameters.processChunk(
            #"<tool_call>{"name":"free_form","arguments":{"anything":true}}</tool_call>"#)
        #expect(withoutParameters.toolCalls.count == 1)

        let withoutTools = ToolCallProcessor(format: .json)
        _ = withoutTools.processChunk(
            #"<tool_call>{"name":"any","arguments":{"x":"y"}}</tool_call>"#)
        #expect(withoutTools.toolCalls.count == 1)
    }
}
