// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Ordered {
    var zulu: String
    var alpha: Int
    var mike: Bool
    var bravo: String
    var yankee: Int
    var charlie: String
    var xray: Bool
    var delta: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Inner {
    var tango: String
    var kilo: Int
    var papa: String
    var echo: Bool
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Outer {
    var whiskey: String
    var inner: Inner
    var items: [Inner]
    var foxtrot: Int
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Guided {
    @Guide(.minimum(3))
    var count: Int
    var label: String
}

/// Returns the names in the order their `"name":` key first appears in `json`
/// at or after `start`.
private func keyOrder(_ names: [String], in json: String, from start: String.Index? = nil)
    -> [String]
{
    let region = json[(start ?? json.startIndex)...]
    return
        names
        .compactMap { name in region.range(of: "\"\(name)\":").map { (name, $0.lowerBound) } }
        .sorted { $0.1 < $1.1 }
        .map(\.0)
}

@Suite
struct SchemaPropertyOrderTests {

    @Test
    func propertiesFollowDeclarationOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let declared = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie", "xray", "delta"]
        let json = try SchemaConverter.encodeToJSON(Ordered.generationSchema)
        let start = try #require(json.range(of: "\"properties\":")?.upperBound)
        #expect(keyOrder(declared, in: json, from: start) == declared)
    }

    @Test
    func nestedDefsFollowTheirOwnDeclarationOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let json = try SchemaConverter.encodeToJSON(Outer.generationSchema)
        let outer = ["whiskey", "inner", "items", "foxtrot"]
        let inner = ["tango", "kilo", "papa", "echo"]

        let root = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        let defs = try #require(root?["$defs"] as? [String: Any])
        #expect(defs.keys.contains { $0.contains("Inner") })

        #expect(keyOrder(inner, in: json) == inner)
        #expect(keyOrder(outer, in: json) == outer)
    }

    @Test
    func orderedOutputParsesToTheSameValue() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let schema = Outer.generationSchema
        let original = try JSONSerialization.jsonObject(with: JSONEncoder().encode(schema))
        let ordered = try JSONSerialization.jsonObject(
            with: Data(try SchemaConverter.encodeToJSON(schema).utf8))
        #expect((original as? NSDictionary) == (ordered as? NSDictionary))
    }

    @Test
    func booleansAndIntegersKeepTheirTypes() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let json = try SchemaConverter.encodeToJSON(Guided.generationSchema)
        #expect(json.contains("\"additionalProperties\":false"))
        #expect(json.contains("\"minimum\":3,") || json.contains("\"minimum\":3}"))
        #expect(!json.contains("3.0"))
    }

    @Test
    func toolEnvelopeAndStructuralTagUseDeclarationOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let declared = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie", "xray", "delta"]
        let tool = Transcript.ToolDefinition(
            name: "ordered", description: "d", parameters: Ordered.generationSchema)
        let envelope = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [tool])
        let tag = try SchemaConverter.encodeToolCallingGrammar(tools: [tool])
        #expect(keyOrder(declared, in: envelope) == declared)
        #expect(keyOrder(declared, in: tag) == declared)
    }
}

#endif
