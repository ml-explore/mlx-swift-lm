// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("JSONValue conversion")
struct JSONValueConversionTests {

    /// Values as `JSONSerialization` produces them, which is the path every
    /// parser that decodes an argument value as JSON goes through.
    private func decoded(_ json: String) -> JSONValue {
        let object = try! JSONSerialization.jsonObject(
            with: Data(json.utf8), options: [.fragmentsAllowed])
        return JSONValue.from(object)
    }

    @Test("Decoded 0 and 1 stay integers")
    func decodedZeroAndOneStayIntegers() {
        // `NSNumber` bridges to `Bool` whenever it holds 0 or 1, so a boxed
        // number used to be reported as `false` / `true`.
        #expect(decoded(#"{"limit": 1}"#) == .object(["limit": .int(1)]))
        #expect(decoded(#"{"offset": 0}"#) == .object(["offset": .int(0)]))
        #expect(decoded("[0, 1, 2]") == .array([.int(0), .int(1), .int(2)]))
    }

    @Test("Decoded booleans stay booleans")
    func decodedBooleansStayBooleans() {
        #expect(decoded(#"{"stream": true}"#) == .object(["stream": .bool(true)]))
        #expect(decoded(#"{"stream": false}"#) == .object(["stream": .bool(false)]))
    }

    @Test("Other decoded scalars keep their type")
    func decodedScalarsKeepTheirType() {
        #expect(decoded("2") == .int(2))
        #expect(decoded("2.5") == .double(2.5))
        #expect(decoded("-3") == .int(-3))
        #expect(decoded(#""text""#) == .string("text"))
        #expect(decoded("null") == .null)
    }

    @Test("Native Swift values keep their type")
    func nativeValuesKeepTheirType() {
        #expect(JSONValue.from(true) == .bool(true))
        #expect(JSONValue.from(false) == .bool(false))
        #expect(JSONValue.from(1) == .int(1))
        #expect(JSONValue.from(0) == .int(0))
        #expect(JSONValue.from(2.5) == .double(2.5))
        #expect(JSONValue.from("text") == .string("text"))
    }

    @Test("A tool call carries a 0/1 argument through unchanged")
    func toolCallPreservesZeroAndOne() throws {
        // XML values are typed from the schema, and object values are decoded as
        // JSON — the combination that previously rewrote 1 as true.
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "search",
                    "parameters": [
                        "properties": [
                            "page": ["type": "integer"],
                            "filters": ["type": "object"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <tool_call><function=search><parameter=page>1</parameter>\
            <parameter=filters>{"archived": false, "limit": 1}</parameter></function></tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: tools))

        #expect(call.function.arguments["page"] == .int(1))
        #expect(
            call.function.arguments["filters"]
                == .object(["archived": .bool(false), "limit": .int(1)]))
    }
}
