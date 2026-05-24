// Copyright © 2026 Apple Inc.

public actor InMemoryResponseStore {
    private var responses: [String: OpenAIResponse] = [:]

    public init() {}

    public func save(_ response: OpenAIResponse) {
        responses[response.id] = response
    }

    public func get(id: String) -> OpenAIResponse? {
        responses[id]
    }

    public func cancel(id: String) -> OpenAIResponse? {
        guard var response = responses[id] else {
            return nil
        }
        response.status = .cancelled
        responses[id] = response
        return response
    }
}
