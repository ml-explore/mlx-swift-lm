// Copyright © 2026 Apple Inc.

public enum HTTPMethod: String, Codable, Sendable, Hashable {
    case get = "GET"
    case post = "POST"
}

public struct MLXServerRoute: Codable, Sendable, Hashable {
    public let method: HTTPMethod
    public let path: String

    public init(_ method: HTTPMethod, _ path: String) {
        self.method = method
        self.path = path
    }

    public static let manifest: [MLXServerRoute] = [
        .init(.get, "/health"),
        .init(.get, "/v1/health"),
        .init(.get, "/metrics"),
        .init(.get, "/props"),
        .init(.get, "/v1/models"),
        .init(.get, "/models"),
        .init(.post, "/v1/chat/completions"),
        .init(.post, "/chat/completions"),
        .init(.post, "/v1/chat/completions/batch"),
        .init(.post, "/v1/completions"),
        .init(.post, "/completions"),
        .init(.post, "/completion"),
        .init(.post, "/v1/responses"),
        .init(.post, "/responses"),
        .init(.get, "/v1/responses/{response_id}"),
        .init(.post, "/v1/responses/{response_id}/cancel"),
        .init(.post, "/v1/embeddings"),
        .init(.post, "/embeddings"),
        .init(.post, "/embedding"),
        .init(.post, "/tokenize"),
        .init(.post, "/detokenize"),
        .init(.post, "/apply-template"),
    ]
}
