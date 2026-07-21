// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLXGuidedGeneration
import FoundationModels
@testable import MLXFoundationModels

/// Schemas for fake developer-defined tools used across these tests.

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct AddArgs {
    @Guide(description: "First addend.")
    var a: Int
    @Guide(description: "Second addend.")
    var b: Int
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Passport {
    @Guide(description: "Issuing country.")
    var country: String
    @Guide(description: "Passport number.")
    var number: String
}

/// Contains a further nested `@Generable` type: `Traveler`'s own `$defs`
/// body carries a `"$ref": "#/$defs/Passport"`, exercising refs that live
/// inside other defs (not just under the schema root).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Traveler {
    @Guide(description: "Full name.")
    var name: String
    @Guide(description: "Age.")
    var age: Int
    @Guide(description: "Travel document.")
    var passport: Passport
}

/// No nested `@Generable` at all — but the description mentions the
/// `#/$defs/` pointer text, which a naive string-level ref rewrite would
/// mangle.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct PointerDocArgs {
    @Guide(description: "A JSON Pointer such as #/$defs/Foo to resolve.")
    var pointer: String
}

/// Arguments containing a nested `@Generable` type: `GenerationSchema`
/// serializes `Traveler` as a root-level `$defs` entry referenced via a
/// root-anchored `"$ref": "#/$defs/Traveler"` pointer.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct BookTripArgs {
    @Guide(description: "Origin city.")
    var origin: String
    @Guide(description: "Destination city.")
    var destination: String
    @Guide(description: "The traveler.")
    var traveler: Traveler
}

/// Unit tests for the tool-calling schema and grammar builders.
///
/// Covers both:
/// - `SchemaConverter.encodeToolCallingEnvelopeJSON(tools:)` - the inner
///   `{oneOf: [{name, arguments}, ...]}` JSON envelope, which must compile
///   cleanly with xgrammar's JSON-schema constructor and is also fed to
///   `CompletionReserve` as the structural-reserve seed.
/// - `SchemaConverter.encodeToolCallingGrammar(tools:)` - the xgrammar
///   structural-tag JSON envelope of the form
///   `{type: "structural_tag", format: {type: "or", elements: [tag(...,
///   per-tool-or), per-tool-or]}}`. Each per-tool tag fixes the name before
///   opening the arguments schema. The wrapped arm dispatches Qwen-style
///   `<tool_call>...</tool_call>` delimiters; the bare arm accepts the raw
///   tool-call object. Shape-only assertions here; real-tokenizer compilation
///   is exercised by the integration suite (the byte-tokenizer used in these
///   unit tests doesn't define Qwen's `<tool_call>` special tokens).
@Suite
struct ToolCallingSchemaTests {

    // MARK: - Envelope Structure

    @Test
    func emptyToolListThrows() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(throws: SchemaConverter.SchemaConversionError.noTools) {
            _ = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [])
        }
    }

    @Test
    func singleToolProducesOneOfWithSingleEntry() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [weather])
        let parsed = try parseAsDictionary(json)

        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        #expect(oneOf.count == 1)

        let entry = oneOf[0]
        #expect(entry["type"] as? String == "object")
        #expect(entry["additionalProperties"] as? Bool == false)

        let properties = try #require(entry["properties"] as? [String: Any])
        let nameSchema = try #require(properties["name"] as? [String: Any])
        #expect(nameSchema["const"] as? String == "get_weather")
        #expect(properties["arguments"] != nil, "arguments schema must be nested verbatim")
    }

    @Test
    func multipleToolsProduceOneEntryPerTool() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tools = [
            Transcript.ToolDefinition(
                name: "get_weather",
                description: "Get weather",
                parameters: WeatherArgs.generationSchema
            ),
            Transcript.ToolDefinition(
                name: "add",
                description: "Add two numbers",
                parameters: AddArgs.generationSchema
            ),
        ]

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: tools)
        let parsed = try parseAsDictionary(json)
        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        #expect(oneOf.count == 2)

        // Names preserved and in order supplied.
        let names: [String] = oneOf.compactMap { entry in
            (entry["properties"] as? [String: Any])
                .flatMap { $0["name"] as? [String: Any] }
                .flatMap { $0["const"] as? String }
        }
        #expect(names == ["get_weather", "add"])
    }

    // MARK: - $defs Hoisting

    @Test
    func nestedGenerableDefsAreHoistedToEnvelopeRoot() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let bookTrip = Transcript.ToolDefinition(
            name: "book_trip",
            description: "Books a trip",
            parameters: BookTripArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [bookTrip])
        let parsed = try parseAsDictionary(json)

        // The nested type's def must live at the envelope root, namespaced
        // per tool, because xgrammar resolves JSON Pointers from the
        // document root.
        let defs = try #require(parsed["$defs"] as? [String: Any])
        #expect(defs["book_trip__Traveler"] != nil)

        // The embedded arguments schema must no longer carry its own $defs.
        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        let arguments = try #require(
            (oneOf[0]["properties"] as? [String: Any])?["arguments"] as? [String: Any]
        )
        #expect(arguments["$defs"] == nil, "tool-local $defs must be hoisted, not duplicated")
    }

    @Test
    func everyRefInEnvelopeResolvesWithinTheDocument() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Two tools sharing a same-named nested type: hoisting must also
        // namespace, so the defs cannot collide or shadow each other.
        let tools = [
            Transcript.ToolDefinition(
                name: "book_trip",
                description: "Books a trip",
                parameters: BookTripArgs.generationSchema
            ),
            Transcript.ToolDefinition(
                name: "cancel_trip",
                description: "Cancels a trip",
                parameters: BookTripArgs.generationSchema
            ),
        ]

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: tools)
        let parsed = try parseAsDictionary(json)

        let defs = parsed["$defs"] as? [String: Any] ?? [:]
        #expect(defs["book_trip__Traveler"] != nil)
        #expect(defs["cancel_trip__Traveler"] != nil)
        #expect(defs["book_trip__Passport"] != nil)
        #expect(defs["cancel_trip__Passport"] != nil)

        let refs = collectRefs(in: parsed)
        #expect(!refs.isEmpty, "nested @Generable arguments must produce $refs")
        for ref in refs {
            #expect(ref.hasPrefix("#/$defs/"), "unexpected ref shape: \(ref)")
            let key = String(ref.dropFirst("#/$defs/".count))
            #expect(defs[key] != nil, "dangling $ref: \(ref)")
        }
    }

    @Test
    func nonRefStringMentioningDefsPointerIsNotRewritten() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // The ref rewrite must only touch strings under a `$ref` key: a
        // description (or const/enum/default/pattern) that merely mentions
        // the "#/$defs/" pointer text has to survive verbatim.
        let probe = Transcript.ToolDefinition(
            name: "probe_tool",
            description: "Resolves JSON Pointers",
            parameters: PointerDocArgs.generationSchema
        )
        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [probe])
        let strings = collectStringValues(in: try parseAsDictionary(json))

        // The pointer text made it into the schema...
        #expect(strings.contains { $0.contains("#/$defs/") })
        // ...and came through untouched by the namespace rewrite.
        #expect(strings.allSatisfy { !$0.contains("probe_tool__") })
    }

    @Test
    func nestedDefsEnvelopeCompilesWithXGrammar() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // End-to-end regression for the dangling-$refs failure: without
        // hoisting, xgrammar rejects this envelope outright
        // ("Cannot find field $defs in {\"oneOf\": ...").
        let bookTrip = Transcript.ToolDefinition(
            name: "book_trip",
            description: "Books a trip",
            parameters: BookTripArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(
            tools: [bookTrip]
        )

        let tokenizer = try makeByteTokenizer()
        _ = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: json, fastForward: false)
    }

    @Test
    func grammarBuilderHoistsNestedDefsInBothArms() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let bookTrip = Transcript.ToolDefinition(
            name: "book_trip",
            description: "Books a trip",
            parameters: BookTripArgs.generationSchema
        )

        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [bookTrip])
        let parsed = try parseAsDictionary(grammar)

        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        try #require(elements.count == 2)

        let wrappedSchema = try #require(
            (elements[0]["content"] as? [String: Any])?["json_schema"] as? [String: Any]
        )
        let bareSchema = try #require(elements[1]["json_schema"] as? [String: Any])
        for schema in [wrappedSchema, bareSchema] {
            let defs = try #require(schema["$defs"] as? [String: Any])
            #expect(defs["book_trip__Traveler"] != nil)
        }
    }

    // MARK: - Grammar Compilation

    @Test
    func envelopeCompilesWithXGrammar() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get weather",
            parameters: WeatherArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [weather])

        // Build a minimal byte-fallback tokenizer and attempt to compile the
        // envelope as a grammar.
        let tokenizer = try makeByteTokenizer()
        _ = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: json, fastForward: false)
    }

    // MARK: - Grammar Builder

    @Test
    func grammarBuilderRejectsEmptyToolList() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(throws: SchemaConverter.SchemaConversionError.noTools) {
            _ = try SchemaConverter.encodeToolCallingGrammar(tools: [])
        }
    }

    @Test
    func grammarExposesWrappedAndBareAlternatives() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let add = Transcript.ToolDefinition(
            name: "add",
            description: "Add two numbers",
            parameters: AddArgs.generationSchema
        )

        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [weather, add])
        let parsed = try parseAsDictionary(grammar)

        #expect(parsed["type"] as? String == "structural_tag")

        let format = try #require(parsed["format"] as? [String: Any])
        #expect(format["type"] as? String == "or")

        let elements = try #require(format["elements"] as? [[String: Any]])
        #expect(elements.count == 2)

        // Wrapped arm: tag(<tool_call>\n ... \n</tool_call>) around a per-tool dispatch.
        let wrapped = elements[0]
        #expect(wrapped["type"] as? String == "tag")
        #expect(wrapped["begin"] as? String == "<tool_call>\n")
        #expect(wrapped["end"] as? [String] == ["\n</tool_call>"])

        let wrappedContent = try #require(wrapped["content"] as? [String: Any])
        #expect(wrappedContent["type"] as? String == "or")
        let wrappedToolTags = try #require(wrappedContent["elements"] as? [[String: Any]])
        #expect(wrappedToolTags.count == 2)
        try assertToolTag(wrappedToolTags[0], for: weather)
        try assertToolTag(wrappedToolTags[1], for: add)

        // Bare arm: the same per-tool dispatch, without delimiters.
        let bare = elements[1]
        #expect(bare["type"] as? String == "or")
        let bareToolTags = try #require(bare["elements"] as? [[String: Any]])
        #expect(bareToolTags.count == 2)
        try assertToolTag(bareToolTags[0], for: weather)
        try assertToolTag(bareToolTags[1], for: add)
    }

    @Test
    func grammarEmbedsValidToolParametersJSON() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [weather])
        let parsed = try parseAsDictionary(grammar)

        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        try #require(elements.count == 2)

        // Each arm contains an equivalent per-tool tag whose JSON schema is
        // the tool's parameters schema, not the old `{name, arguments}` envelope.
        let wrappedDispatch = try #require(elements[0]["content"] as? [String: Any])
        let wrappedTags = try #require(wrappedDispatch["elements"] as? [[String: Any]])
        try assertToolTag(wrappedTags[0], for: weather)

        let bareTags = try #require(elements[1]["elements"] as? [[String: Any]])
        try assertToolTag(bareTags[0], for: weather)
    }

    // MARK: - Helpers

    private func parseAsDictionary(_ json: String) throws -> [String: Any] {
        let data = Data(json.utf8)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Envelope JSON did not parse as an object: \(json)")
            return [:]
        }
        return obj
    }

    /// Recursively collects every string value in a parsed JSON tree
    /// (dictionary values and array elements, at any depth).
    private func collectStringValues(in value: Any) -> [String] {
        switch value {
        case let string as String:
            return [string]
        case let object as [String: Any]:
            return object.values.flatMap { collectStringValues(in: $0) }
        case let array as [Any]:
            return array.flatMap { collectStringValues(in: $0) }
        default:
            return []
        }
    }

    /// Recursively collects every `"$ref"` string value in a parsed JSON tree.
    private func collectRefs(in value: Any) -> [String] {
        switch value {
        case let object as [String: Any]:
            return object.flatMap { key, nested -> [String] in
                if key == "$ref", let ref = nested as? String {
                    return [ref]
                }
                return collectRefs(in: nested)
            }
        case let array as [Any]:
            return array.flatMap { collectRefs(in: $0) }
        default:
            return []
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func assertToolTag(
        _ tag: [String: Any],
        for tool: Transcript.ToolDefinition
    ) throws {
        #expect(tag["type"] as? String == "tag")
        #expect(tag["begin"] as? String == "{\"name\": \"\(tool.name)\", \"arguments\": ")
        #expect(tag["end"] as? [String] == ["}"])

        let content = try #require(tag["content"] as? [String: Any])
        #expect(content["type"] as? String == "json_schema")
        let actualSchema = try #require(content["json_schema"] as? [String: Any])
        let expectedData = try JSONEncoder().encode(tool.parameters)
        let expectedSchema = try JSONSerialization.jsonObject(with: expectedData)
        #expect(try canonicalJSON(actualSchema) == canonicalJSON(expectedSchema))
    }

    private func canonicalJSON(_ object: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func makeByteTokenizer() throws -> GrammarTokenizer {
        let vocabSize = 256
        let vocab: [String] = (0 ..< vocabSize).map { byte in
            String(format: "<0x%02X>", byte)
        }
        return try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            eosTokenId: Int32(vocabSize - 1)
        )
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
