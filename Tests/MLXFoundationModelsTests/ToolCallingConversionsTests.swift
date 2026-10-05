// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Seat {
    @Guide(description: "Seat row.")
    var row: Int
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ReserveSeatsArguments {
    @Guide(description: "Flight number.")
    var flight: String
    // Keep the count at 1. A boxed `1` also passes `as? Bool`, so a conversion
    // that casts with `as? Bool` turns `minItems` and `maxItems` into `true`.
    @Guide(description: "Seats to reserve.", .count(1))
    var seats: [Seat]
}

@Suite
struct ToolCallingConversionsTests {

    @Test
    func schemaBooleansAreSwiftBools() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let parameters = try parametersSchema(of: makeReserveSeatsToolSpec())

        let rootValue = try #require(parameters["additionalProperties"])
        #expect(isSwiftBool(rootValue))
        #expect(rootValue as? Bool == false)

        let definitions = try #require(parameters["$defs"] as? [String: Any])
        let seat = try #require(definitions["Seat"] as? [String: Any])
        let nestedValue = try #require(seat["additionalProperties"])
        #expect(isSwiftBool(nestedValue))
        #expect(nestedValue as? Bool == false)
    }

    @Test
    func schemaNumbersStayNumbers() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let parameters = try parametersSchema(of: makeReserveSeatsToolSpec())

        let properties = try #require(parameters["properties"] as? [String: Any])
        let seats = try #require(properties["seats"] as? [String: Any])
        for key in ["minItems", "maxItems"] {
            let bound = try #require(seats[key])
            #expect(!isSwiftBool(bound), "\(key) must stay a number")
            #expect(bound as? Int == 1)
        }
    }

    // MARK: - Helpers

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeReserveSeatsToolSpec() throws -> [String: any Sendable] {
        try ToolCallingConversions.makeToolSpec(
            from: Transcript.ToolDefinition(
                name: "reserve_seats",
                description: "Reserve seats on a flight",
                parameters: ReserveSeatsArguments.generationSchema
            )
        )
    }

    private func parametersSchema(of toolSpec: [String: any Sendable]) throws -> [String: Any] {
        let function = try #require(toolSpec["function"] as? [String: Any])
        return try #require(function["parameters"] as? [String: Any])
    }

    private func isSwiftBool(_ value: Any) -> Bool {
        // Keep the `type(of:)` check. A boxed boolean and a boxed `0` or `1` also pass `as? Bool`.
        type(of: value) == Bool.self
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
