// Copyright © 2026 Apple Inc.

import Foundation

/// Conservatively checks tool-call arguments against a declared JSON Schema.
///
/// Tool schemas may contain vocabularies this package does not implement. An
/// unsupported assertion must never turn a valid call into a rejection, so an
/// evaluation has three outcomes:
///
/// - ``Result/valid``: every applicable assertion is understood and satisfied;
/// - ``Result/invalid(_:)``: at least one understood assertion is definitively
///   violated;
/// - ``Result/unknown``: no understood assertion is violated, but unsupported
///   or malformed schema content prevents a proof of validity.
///
/// Callers may reject only ``Result/invalid(_:)``. `unknown` deliberately
/// fails open: schema declarations are trusted configuration, but a partial
/// validator is not an authority on semantics it does not implement.
package enum ToolSchemaValidator {

    /// The result of evaluating one value against a schema.
    package enum Result: Equatable, Sendable {
        case valid
        case invalid([Violation])
        case unknown
    }

    /// One understood schema assertion that a value did not satisfy.
    package struct Violation: Hashable, Sendable, CustomStringConvertible {
        /// The location of the value, for example `arguments.filters.limit`.
        package let path: String

        /// What the schema requires, for example "must be an integer".
        package let requirement: String

        package init(path: String, requirement: String) {
            self.path = path
            self.requirement = requirement
        }

        package var description: String { "\(path) \(requirement)" }
    }

    /// Checks arguments against the `parameters` schema of a tool.
    ///
    /// A missing schema declares no constraints and is therefore valid. An
    /// unsupported assertion returns `unknown`, which executable-call routing
    /// must accept rather than guess at its meaning.
    package static func validate(
        arguments: [String: JSONValue],
        against parameters: [String: any Sendable]?
    ) -> Result {
        guard let parameters else { return .valid }
        return check(.object(arguments), against: parameters, at: "arguments")
    }

    /// Finds and evaluates the schema for one declared tool call.
    package static func validate(
        arguments: [String: JSONValue],
        forToolNamed name: String,
        in tools: [[String: any Sendable]]?
    ) -> Result {
        guard let tools else { return .valid }
        guard let parameters = rawParametersSchema(ofToolNamed: name, in: tools) else {
            return .valid
        }
        return check(.object(arguments), against: parameters, at: "arguments")
    }

    /// Checks one value against a schema.
    ///
    /// Boolean schemas are supported. Any other schema must be a keyword
    /// object; a different shape is unknown rather than an argument failure.
    package static func validate(
        _ value: JSONValue, against schema: Any, at path: String = "$"
    ) -> Result {
        check(value, against: schema, at: path)
    }

    /// Finds the `parameters` schema of a declared tool, if it has one.
    ///
    /// Both declaration shapes are supported: the OpenAI envelope
    /// (`{"function": {"name": ..., "parameters": ...}}`) and the flat
    /// shape (`{"name": ..., "parameters": ...}`).
    package static func parametersSchema(
        ofToolNamed name: String, in tools: [[String: any Sendable]]
    ) -> [String: any Sendable]? {
        rawParametersSchema(ofToolNamed: name, in: tools) as? [String: any Sendable]
    }

    /// Finds the raw declaration so a malformed schema remains distinguishable
    /// from an absent one at the proof boundary.
    private static func rawParametersSchema(
        ofToolNamed name: String, in tools: [[String: any Sendable]]
    ) -> (any Sendable)? {
        for tool in tools {
            if let function = tool["function"] as? [String: any Sendable] {
                if function["name"] as? String == name {
                    return function["parameters"]
                }
            } else if tool["name"] as? String == name {
                return tool["parameters"]
            }
        }
        return nil
    }

    /// Makes a bounded, stable, non-sensitive diagnostic summary.
    package static func describe(_ violations: [Violation], limit: Int = 3) -> String {
        guard !violations.isEmpty else { return "" }
        let limit = max(0, limit)
        guard limit > 0 else { return "\(violations.count) schema violations" }

        let shown = violations.prefix(limit).map(\.description).joined(separator: "; ")
        let hidden = violations.count - min(violations.count, limit)
        let description = hidden > 0 ? "\(shown); and \(hidden) more" : shown
        guard description.unicodeScalars.count > maximumDescriptionScalarCount else {
            return description
        }
        return String(description.unicodeScalars.prefix(maximumDescriptionScalarCount - 1)) + "…"
    }

    // MARK: - Core evaluation

    private struct Assessment {
        var violations: [Violation] = []
        var containsUnknown = false

        mutating func merge(_ result: Result) {
            switch result {
            case .valid:
                break
            case .invalid(let incoming):
                violations.append(contentsOf: incoming)
            case .unknown:
                containsUnknown = true
            }
        }

        mutating func reject(at path: String, _ requirement: String) {
            violations.append(Violation(path: path, requirement: requirement))
        }

        var result: Result {
            if !violations.isEmpty { return .invalid(violations) }
            return containsUnknown ? .unknown : .valid
        }
    }

    private enum Keyword<Value> {
        case absent
        case value(Value)
        case malformed
    }

    private static let maximumSchemaDepth = 64
    private static let maximumDescriptionScalarCount = 512

    private static func check(
        _ value: JSONValue,
        against rawSchema: Any,
        at path: String,
        depth: Int = 0
    ) -> Result {
        guard depth <= maximumSchemaDepth else { return .unknown }
        if let permitted = strictBoolean(rawSchema) {
            return permitted
                ? .valid
                : .invalid([Violation(path: path, requirement: "is not permitted")])
        }
        guard let schema = rawSchema as? [String: any Sendable] else { return .unknown }

        // Before draft 2019-09, `$ref` replaced the schema object and its
        // siblings were ignored. Without resolving the reference and dialect,
        // no sibling assertion is a portable proof of rejection.
        if schema["$ref"] != nil || schema["$dynamicRef"] != nil
            || schema["$recursiveRef"] != nil
        {
            return .unknown
        }

        var assessment = Assessment()
        if schema.keys.contains(where: isUnsupportedAssertion) {
            assessment.containsUnknown = true
        }

        switch declaredTypes(schema["type"]) {
        case .absent:
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(let types):
            guard types.contains(where: { matchesType(value, $0) }) else {
                return .invalid([
                    Violation(path: path, requirement: typeRequirement(types))
                ])
            }
        }

        assessment.merge(checkConst(value, schema["const"], at: path))
        assessment.merge(checkEnum(value, schema["enum"], at: path))

        switch value {
        case .object(let object):
            assessment.merge(checkObject(object, schema, at: path, depth: depth))
        case .array(let elements):
            assessment.merge(checkArray(elements, schema, at: path, depth: depth))
        case .string(let string):
            assessment.merge(checkString(string, schema, at: path))
        case .int, .double:
            assessment.merge(checkNumber(value, schema, at: path))
        case .null, .bool:
            break
        }

        assessment.merge(checkCombinators(value, schema, at: path, depth: depth))
        return assessment.result
    }

    // MARK: - Object assertions

    private static func checkObject(
        _ object: [String: JSONValue],
        _ schema: [String: any Sendable],
        at path: String,
        depth: Int
    ) -> Result {
        var assessment = Assessment()

        switch stringArray(schema["required"]) {
        case .absent:
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(let required):
            for key in required.sorted() where object[key] == nil {
                assessment.reject(at: childPath(path, key), "is required")
            }
        }

        let properties: [String: any Sendable]?
        let propertiesAreMalformed: Bool
        if let rawProperties = schema["properties"] {
            properties = rawProperties as? [String: any Sendable]
            propertiesAreMalformed = properties == nil
            if propertiesAreMalformed { assessment.containsUnknown = true }
        } else {
            properties = nil
            propertiesAreMalformed = false
        }

        if let properties {
            if properties.values.contains(where: { !isSchema($0) }) {
                assessment.containsUnknown = true
            }
            for key in properties.keys.sorted() {
                guard let member = object[key], let memberSchema = properties[key] else { continue }
                assessment.merge(
                    check(
                        member, against: memberSchema, at: childPath(path, key),
                        depth: depth + 1))
            }
        }

        guard let additional = schema["additionalProperties"] else {
            return assessment.result
        }
        guard !propertiesAreMalformed else { return assessment.result }
        guard isSchema(additional) else {
            assessment.containsUnknown = true
            return assessment.result
        }
        // `patternProperties` participates in the definition of an additional
        // property. Until it is implemented, no key classification is proven.
        guard schema["patternProperties"] == nil else { return assessment.result }

        let declaredNames = Set(properties?.keys.map { $0 } ?? [])
        for key in object.keys.sorted() where !declaredNames.contains(key) {
            let memberPath = childPath(path, key)
            if let permitted = strictBoolean(additional) {
                if !permitted {
                    assessment.reject(at: memberPath, "is not a declared property")
                }
            } else {
                assessment.merge(
                    check(
                        object[key]!, against: additional, at: memberPath,
                        depth: depth + 1))
            }
        }
        return assessment.result
    }

    // MARK: - Array assertions

    private static func checkArray(
        _ elements: [JSONValue],
        _ schema: [String: any Sendable],
        at path: String,
        depth: Int
    ) -> Result {
        var assessment = Assessment()

        applyCountKeyword(
            schema["minItems"], actual: elements.count, at: path, unit: "items",
            bound: .minimum, assessment: &assessment)
        applyCountKeyword(
            schema["maxItems"], actual: elements.count, at: path, unit: "items",
            bound: .maximum, assessment: &assessment)

        switch booleanKeyword(schema["uniqueItems"]) {
        case .absent, .value(false):
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(true):
            var comparableElements: Set<ComparableJSON> = []
            comparableElements.reserveCapacity(elements.count)
            for element in elements {
                guard let comparable = comparable(element) else {
                    assessment.containsUnknown = true
                    comparableElements.removeAll(keepingCapacity: true)
                    break
                }
                if !comparableElements.insert(comparable).inserted {
                    assessment.reject(at: path, "must not contain duplicate items")
                    break
                }
            }
        }

        if let items = schema["items"] {
            if schema["prefixItems"] != nil {
                // In draft 2020-12, `items` applies only after `prefixItems`.
                // Applying it to every element could manufacture a violation.
                assessment.containsUnknown = true
            } else if items is [Any] {
                // The pre-2020 tuple form has positional semantics that the
                // single-schema implementation cannot safely approximate.
                assessment.containsUnknown = true
            } else if !isSchema(items) {
                assessment.containsUnknown = true
            } else {
                for (index, element) in elements.enumerated() {
                    assessment.merge(
                        check(
                            element, against: items, at: "\(path)[\(index)]",
                            depth: depth + 1))
                }
            }
        }
        return assessment.result
    }

    private enum CountBound {
        case minimum
        case maximum
    }

    private static func applyCountKeyword(
        _ rawLimit: Any?,
        actual: Int,
        at path: String,
        unit: String,
        bound: CountBound,
        assessment: inout Assessment
    ) {
        switch nonNegativeInteger(rawLimit) {
        case .absent:
            return
        case .malformed:
            assessment.containsUnknown = true
        case .value(let limit):
            let violated =
                switch bound {
                case .minimum: actual < limit
                case .maximum: actual > limit
                }
            guard violated else { return }
            let word =
                switch bound {
                case .minimum: "least"
                case .maximum: "most"
                }
            assessment.reject(
                at: path,
                countRequirement(unit, limit, word))
        }
    }

    // MARK: - String assertions

    private static func checkString(
        _ string: String, _ schema: [String: any Sendable], at path: String
    ) -> Result {
        var assessment = Assessment()
        // JSON Schema defines string length in Unicode code points. Swift's
        // `String.count` counts extended grapheme clusters instead.
        let length = string.unicodeScalars.count
        applyCountKeyword(
            schema["minLength"], actual: length, at: path, unit: "characters",
            bound: .minimum, assessment: &assessment)
        applyCountKeyword(
            schema["maxLength"], actual: length, at: path, unit: "characters",
            bound: .maximum, assessment: &assessment)
        return assessment.result
    }

    // MARK: - Numeric assertions

    private static func checkNumber(
        _ value: JSONValue, _ schema: [String: any Sendable], at path: String
    ) -> Result {
        guard let number = comparableNumber(value) else { return .unknown }
        var assessment = Assessment()

        let minimum = numericKeyword(schema["minimum"])
        let maximum = numericKeyword(schema["maximum"])
        let exclusiveMinimum = exclusiveBound(schema["exclusiveMinimum"])
        let exclusiveMaximum = exclusiveBound(schema["exclusiveMaximum"])

        if case .malformed = minimum { assessment.containsUnknown = true }
        if case .malformed = maximum { assessment.containsUnknown = true }
        if case .malformed = exclusiveMinimum { assessment.containsUnknown = true }
        if case .malformed = exclusiveMaximum { assessment.containsUnknown = true }

        let draft4LowerExclusive = exclusiveMinimum == .boolean(true)
        let draft4UpperExclusive = exclusiveMaximum == .boolean(true)

        if case .value(let lower) = minimum {
            let violated = number < lower || (draft4LowerExclusive && number == lower)
            if violated {
                assessment.reject(
                    at: path,
                    draft4LowerExclusive
                        ? "must be greater than \(format(lower))"
                        : "must be at least \(format(lower))")
            }
        } else if draft4LowerExclusive {
            assessment.containsUnknown = true
        }

        if case .number(let lower) = exclusiveMinimum, number <= lower {
            assessment.reject(at: path, "must be greater than \(format(lower))")
        }

        if case .value(let upper) = maximum {
            let violated = number > upper || (draft4UpperExclusive && number == upper)
            if violated {
                assessment.reject(
                    at: path,
                    draft4UpperExclusive
                        ? "must be less than \(format(upper))"
                        : "must be at most \(format(upper))")
            }
        } else if draft4UpperExclusive {
            assessment.containsUnknown = true
        }

        if case .number(let upper) = exclusiveMaximum, number >= upper {
            assessment.reject(at: path, "must be less than \(format(upper))")
        }

        return assessment.result
    }

    private enum ExclusiveBound: Equatable {
        case absent
        case boolean(Bool)
        case number(Decimal)
        case malformed
    }

    private static func exclusiveBound(_ value: Any?) -> ExclusiveBound {
        guard let value else { return .absent }
        if let boolean = strictBoolean(value) { return .boolean(boolean) }
        guard let number = comparableNumber(value) else { return .malformed }
        return .number(number)
    }

    // MARK: - Equality assertions

    private static func checkConst(_ value: JSONValue, _ rawConstant: Any?, at path: String)
        -> Result
    {
        guard let rawConstant else { return .valid }
        guard let value = comparable(value), let constant = comparable(rawConstant) else {
            return .unknown
        }
        return value == constant
            ? .valid
            : .invalid([
                Violation(path: path, requirement: "must equal the declared constant")
            ])
    }

    private static func checkEnum(_ value: JSONValue, _ rawAllowed: Any?, at path: String)
        -> Result
    {
        guard let rawAllowed else { return .valid }
        guard let allowed = rawAllowed as? [Any], !allowed.isEmpty,
            let comparableValue = comparable(value)
        else { return .unknown }

        var comparableAllowed: [ComparableJSON] = []
        comparableAllowed.reserveCapacity(allowed.count)
        for candidate in allowed {
            guard let comparableCandidate = comparable(candidate) else { return .unknown }
            // Duplicate enum members make the schema malformed.
            guard !comparableAllowed.contains(comparableCandidate) else { return .unknown }
            comparableAllowed.append(comparableCandidate)
        }
        if comparableAllowed.contains(comparableValue) { return .valid }
        return .invalid([
            Violation(path: path, requirement: "must be one of the allowed values")
        ])
    }

    private indirect enum ComparableJSON: Hashable {
        case null
        case bool(Bool)
        case number(Decimal)
        case string(String)
        case array([ComparableJSON])
        case object([String: ComparableJSON])
    }

    private static func comparable(_ value: JSONValue) -> ComparableJSON? {
        switch value {
        case .null: .null
        case .bool(let value): .bool(value)
        case .int, .double:
            comparableNumber(value).map(ComparableJSON.number)
        case .string(let value): .string(value)
        case .array(let values):
            sequence(values.map(comparable)).map(ComparableJSON.array)
        case .object(let values):
            sequence(values.mapValues(comparable)).map(ComparableJSON.object)
        }
    }

    private static func comparable(_ value: Any) -> ComparableJSON? {
        if value is NSNull { return .null }
        if let value = strictBoolean(value) { return .bool(value) }
        if let value = value as? String { return .string(value) }
        if let values = value as? [Any] {
            return sequence(values.map(comparable)).map(ComparableJSON.array)
        }
        if let values = value as? [String: Any] {
            return sequence(values.mapValues(comparable)).map(ComparableJSON.object)
        }
        return comparableNumber(value).map(ComparableJSON.number)
    }

    private static func sequence<T>(_ values: [T?]) -> [T]? {
        var result: [T] = []
        result.reserveCapacity(values.count)
        for value in values {
            guard let value else { return nil }
            result.append(value)
        }
        return result
    }

    private static func sequence<T>(_ values: [String: T?]) -> [String: T]? {
        var result: [String: T] = [:]
        result.reserveCapacity(values.count)
        for (key, value) in values {
            guard let value else { return nil }
            result[key] = value
        }
        return result
    }

    // MARK: - Combinators

    private static func checkCombinators(
        _ value: JSONValue,
        _ schema: [String: any Sendable],
        at path: String,
        depth: Int
    ) -> Result {
        var assessment = Assessment()

        switch schemaArray(schema["allOf"]) {
        case .absent:
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(let schemas):
            for subschema in schemas {
                assessment.merge(
                    check(value, against: subschema, at: path, depth: depth + 1))
            }
        }

        switch schemaArray(schema["anyOf"]) {
        case .absent:
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(let schemas):
            let results = schemas.map {
                check(value, against: $0, at: path, depth: depth + 1)
            }
            if results.contains(.valid) {
                break
            } else if results.contains(.unknown) {
                assessment.containsUnknown = true
            } else {
                assessment.reject(at: path, "must satisfy at least one of the allowed schemas")
            }
        }

        switch schemaArray(schema["oneOf"]) {
        case .absent:
            break
        case .malformed:
            assessment.containsUnknown = true
        case .value(let schemas):
            let results = schemas.map {
                check(value, against: $0, at: path, depth: depth + 1)
            }
            let validCount = results.count { $0 == .valid }
            let containsUnknown = results.contains(.unknown)

            if validCount > 1 || (validCount == 0 && !containsUnknown) {
                assessment.reject(at: path, "must satisfy exactly one of the allowed schemas")
            } else if containsUnknown {
                assessment.containsUnknown = true
            }
        }

        return assessment.result
    }

    // MARK: - Schema extraction

    private static let supportedTypes: Set<String> = [
        "null", "boolean", "string", "object", "array", "number", "integer",
    ]

    private static let supportedKeywords: Set<String> = [
        "type", "properties", "required", "additionalProperties", "items",
        "minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum",
        "const", "anyOf", "oneOf", "allOf",
    ]

    /// Keywords that carry annotations or reusable schema definitions but do
    /// not assert anything about the current instance by themselves.
    private static let annotationKeywords: Set<String> = [
        "$schema", "$id", "$anchor", "$dynamicAnchor", "$comment", "$defs",
        "definitions", "title", "description", "default", "examples", "deprecated",
        "readOnly", "writeOnly", "contentEncoding", "contentMediaType", "contentSchema",
    ]

    private static func isUnsupportedAssertion(_ key: String) -> Bool {
        !supportedKeywords.contains(key)
            && !annotationKeywords.contains(key)
            && !key.hasPrefix("x-")
    }

    private static func declaredTypes(_ value: Any?) -> Keyword<[String]> {
        guard let value else { return .absent }
        let types: [String]
        if let single = value as? String {
            types = [single]
        } else if let list = value as? [Any], !list.isEmpty,
            list.allSatisfy({ $0 is String })
        {
            types = list.map { $0 as! String }
        } else {
            return .malformed
        }
        guard types.allSatisfy(supportedTypes.contains), Set(types).count == types.count else {
            return .malformed
        }
        return .value(types)
    }

    private static func stringArray(_ value: Any?) -> Keyword<[String]> {
        guard let value else { return .absent }
        guard let list = value as? [Any], list.allSatisfy({ $0 is String }) else {
            return .malformed
        }
        let strings = list.map { $0 as! String }
        guard Set(strings).count == strings.count else { return .malformed }
        return .value(strings)
    }

    private static func schemaArray(_ value: Any?) -> Keyword<[Any]> {
        guard let value else { return .absent }
        guard let schemas = value as? [Any], !schemas.isEmpty else { return .malformed }
        return .value(schemas)
    }

    private static func booleanKeyword(_ value: Any?) -> Keyword<Bool> {
        guard let value else { return .absent }
        return strictBoolean(value).map(Keyword.value) ?? .malformed
    }

    private static func isSchema(_ value: Any) -> Bool {
        strictBoolean(value) != nil || value is [String: any Sendable]
    }

    private static func numericKeyword(_ value: Any?) -> Keyword<Decimal> {
        guard let value else { return .absent }
        return comparableNumber(value).map(Keyword.value) ?? .malformed
    }

    private static func nonNegativeInteger(_ value: Any?) -> Keyword<Int> {
        guard let value else { return .absent }
        guard strictBoolean(value) == nil else { return .malformed }
        if let value = value as? Int {
            return value >= 0 ? .value(value) : .malformed
        }
        guard let boxed = value as? NSNumber else { return .malformed }
        if let value = boxed as? Int {
            return value >= 0 ? .value(value) : .malformed
        }
        let doubleValue = boxed.doubleValue
        guard doubleValue.isFinite, doubleValue >= 0,
            doubleValue.rounded() == doubleValue,
            doubleValue < Double(Int.max)
        else { return .malformed }
        return .value(Int(doubleValue))
    }

    /// A Boolean only when the value really is a Boolean. A boxed `0` or `1`
    /// from `JSONSerialization` must not pass as one.
    private static func strictBoolean(_ value: Any) -> Bool? {
        if let boxed = value as? NSNumber {
            return CFGetTypeID(boxed) == CFBooleanGetTypeID() ? boxed.boolValue : nil
        }
        return value as? Bool
    }

    private static func comparableNumber(_ value: JSONValue) -> Decimal? {
        switch value {
        case .int(let value):
            return Decimal(value)
        case .double(let value):
            guard value.isFinite else { return nil }
            var decimal = Decimal(value)
            return NSDecimalIsNotANumber(&decimal) ? nil : decimal
        default:
            return nil
        }
    }

    private static func comparableNumber(_ value: Any) -> Decimal? {
        guard strictBoolean(value) == nil else { return nil }
        if let value = value as? Decimal { return value }
        guard let boxed = value as? NSNumber else { return nil }
        var decimal = boxed.decimalValue
        return NSDecimalIsNotANumber(&decimal) ? nil : decimal
    }

    // MARK: - Messages and paths

    private static func matchesType(_ value: JSONValue, _ type: String) -> Bool {
        switch (value, type) {
        case (.null, "null"), (.bool, "boolean"), (.string, "string"),
            (.object, "object"), (.array, "array"),
            (.int, "number"), (.double, "number"), (.int, "integer"):
            true
        case (.double(let value), "integer"):
            value.isFinite && value.rounded() == value
        default:
            false
        }
    }

    private static func typeRequirement(_ types: [String]) -> String {
        guard types.count > 1 else {
            let type = types[0]
            if type == "null" { return "must be null" }
            let vowels: Set<Character> = ["a", "e", "i", "o", "u"]
            let article = type.first.map(vowels.contains) == true ? "an" : "a"
            return "must be \(article) \(type)"
        }
        return "must be one of these types: \(types.joined(separator: ", "))"
    }

    private static func countRequirement(_ unit: String, _ limit: Int, _ bound: String)
        -> String
    {
        "must have at \(bound) \(limit) \(limit == 1 ? String(unit.dropLast()) : unit)"
    }

    private static func format(_ number: Decimal) -> String {
        NSDecimalNumber(decimal: number).stringValue
    }

    private static func childPath(_ path: String, _ key: String) -> String {
        let identifierStart = CharacterSet.letters.union(CharacterSet(charactersIn: "_"))
        let identifierBody = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        guard let first = key.unicodeScalars.first, identifierStart.contains(first),
            key.unicodeScalars.dropFirst().allSatisfy(identifierBody.contains)
        else {
            guard let data = try? JSONEncoder().encode(key),
                let quoted = String(data: data, encoding: .utf8)
            else { return "\(path)[<unrepresentable key>]" }
            return "\(path)[\(quoted)]"
        }
        return "\(path).\(key)"
    }
}
