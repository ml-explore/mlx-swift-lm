// Copyright © 2026 Apple Inc.

import Foundation

public struct ServerMetricsSnapshot: Codable, Sendable, Equatable {
    public var requestsTotal: Int
    public var chatCompletionsTotal: Int
    public var responsesTotal: Int
    public var errorsTotal: Int
    public var promptTokensTotal: Int
    public var completionTokensTotal: Int
    public var uptimeSeconds: Double

    private enum CodingKeys: String, CodingKey {
        case requestsTotal = "requests_total"
        case chatCompletionsTotal = "chat_completions_total"
        case responsesTotal = "responses_total"
        case errorsTotal = "errors_total"
        case promptTokensTotal = "prompt_tokens_total"
        case completionTokensTotal = "completion_tokens_total"
        case uptimeSeconds = "uptime_seconds"
    }
}

public actor ServerMetrics {
    private enum RequestKind {
        case chat
        case response
        case other
    }

    private let startedAt: Date
    private var requestsTotal = 0
    private var chatCompletionsTotal = 0
    private var responsesTotal = 0
    private var errorsTotal = 0
    private var promptTokensTotal = 0
    private var completionTokensTotal = 0

    public init(startedAt: Date = Date()) {
        self.startedAt = startedAt
    }

    func recordChatRequest() {
        record(.chat)
    }

    func recordResponseRequest() {
        record(.response)
    }

    func recordOtherRequest() {
        record(.other)
    }

    private func record(_ kind: RequestKind) {
        requestsTotal += 1
        switch kind {
        case .chat:
            chatCompletionsTotal += 1
        case .response:
            responsesTotal += 1
        case .other:
            break
        }
    }

    func recordError() {
        errorsTotal += 1
    }

    func recordUsage(_ usage: OpenAIUsage?) {
        guard let usage else {
            return
        }
        promptTokensTotal += usage.promptTokens
        completionTokensTotal += usage.completionTokens
    }

    public func snapshot(now: Date = Date()) -> ServerMetricsSnapshot {
        .init(
            requestsTotal: requestsTotal,
            chatCompletionsTotal: chatCompletionsTotal,
            responsesTotal: responsesTotal,
            errorsTotal: errorsTotal,
            promptTokensTotal: promptTokensTotal,
            completionTokensTotal: completionTokensTotal,
            uptimeSeconds: now.timeIntervalSince(startedAt)
        )
    }

    public func prometheusText(now: Date = Date()) -> String {
        let snapshot = snapshot(now: now)
        return """
            # TYPE mlx_server_requests_total counter
            mlx_server_requests_total \(snapshot.requestsTotal)
            # TYPE mlx_server_chat_completions_total counter
            mlx_server_chat_completions_total \(snapshot.chatCompletionsTotal)
            # TYPE mlx_server_responses_total counter
            mlx_server_responses_total \(snapshot.responsesTotal)
            # TYPE mlx_server_errors_total counter
            mlx_server_errors_total \(snapshot.errorsTotal)
            # TYPE mlx_server_prompt_tokens_total counter
            mlx_server_prompt_tokens_total \(snapshot.promptTokensTotal)
            # TYPE mlx_server_completion_tokens_total counter
            mlx_server_completion_tokens_total \(snapshot.completionTokensTotal)
            # TYPE mlx_server_uptime_seconds gauge
            mlx_server_uptime_seconds \(snapshot.uptimeSeconds)

            """
    }
}
