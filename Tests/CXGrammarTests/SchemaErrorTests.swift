// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for the CXGrammar shim's discriminated error
// surface. Each error category (InvalidJSONError,
// InvalidJSONSchemaError, InvalidStructuralTagError, ...) must map to a
// distinct XG_ERR_* status and populate xg_last_error_message so Swift
// can surface actionable diagnostics.

import MLXCXGrammar
import Foundation
import Testing

@Suite
struct SchemaErrorTests {

    /// A schema that is valid JSON but contains an unsupported
    /// type keyword must surface as `XG_ERR_INVALID_JSON_SCHEMA` with
    /// a non-empty error message. Exact message wording is
    /// intentionally not asserted (xgrammar's phrasing will
    /// differ). The assertion bar is: discriminated status + the
    /// thread-local error buffer surfaces something.
    @Test
    func testMalformedSchemaReturnsInvalidJSONSchemaStatus() throws {
        let invalidSchema = #"{"type":"flibbertigibbet"}"#

        var grammar: OpaquePointer?
        let status: XGStatus = invalidSchema.withCString { schemaPtr in
            xg_grammar_from_json_schema(schemaPtr, &grammar)
        }

        #expect(
            status == XG_ERR_INVALID_JSON_SCHEMA,
            "Expected XG_ERR_INVALID_JSON_SCHEMA (\(XG_ERR_INVALID_JSON_SCHEMA)); got \(status). Last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(grammar == nil, "out_grammar must remain untouched on failure")

        // xg_last_error_message must return a non-null, non-empty
        // buffer containing xgrammar's what() for the failure. We
        // assert only that something surfaced, not its wording.
        let messagePtr = xg_last_error_message()
        #expect(messagePtr != nil, "xg_last_error_message must be non-null after a failure")
        if let messagePtr {
            let message = String(cString: messagePtr)
            #expect(!message.isEmpty, "xg_last_error_message must be non-empty after a failure")
        }
    }
}
