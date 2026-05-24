// Copyright © 2026 Apple Inc.

import Foundation

public enum ServerSentEventEncoder {
    public static let done = "data: [DONE]\n\n"

    public static func encode<T: Encodable>(_ value: T, encoder: JSONEncoder = .openAIServer)
        throws -> String
    {
        let data = try encoder.encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw EncodingError.invalidValue(
                value,
                .init(codingPath: [], debugDescription: "Unable to encode SSE frame as UTF-8")
            )
        }
        return "data: \(json)\n\n"
    }
}

extension JSONEncoder {
    public static var openAIServer: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}
