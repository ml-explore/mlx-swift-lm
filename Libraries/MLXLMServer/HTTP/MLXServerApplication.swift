// Copyright © 2026 Apple Inc.

import Foundation
import Hummingbird

public struct ServerHealthResponse: Codable, Sendable, Equatable {
    public var status: String
    public var server: String

    public init(status: String = "ok", server: String = MLXLMServer.defaultServerName) {
        self.status = status
        self.server = server
    }
}

public struct ServerPropsResponse: Codable, Sendable, Equatable {
    public var server: String
    public var routes: [MLXServerRoute]
    public var supportsChatCompletions: Bool
    public var supportsResponses: Bool
    public var supportsTokenUtilities: Bool

    private enum CodingKeys: String, CodingKey {
        case server
        case routes
        case supportsChatCompletions = "supports_chat_completions"
        case supportsResponses = "supports_responses"
        case supportsTokenUtilities = "supports_token_utilities"
    }

    public init(server: String = MLXLMServer.defaultServerName) {
        self.server = server
        self.routes = MLXServerRoute.manifest
        self.supportsChatCompletions = true
        self.supportsResponses = true
        self.supportsTokenUtilities = true
    }
}

public enum MLXServerApplication {
    public static func buildRouter(service: MLXOpenAIService) -> Router<BasicRequestContext> {
        let router = Router()

        router.get("/health") { _, _ async throws -> Response in
            try jsonResponse(ServerHealthResponse())
        }
        router.get("/v1/health") { _, _ async throws -> Response in
            try jsonResponse(ServerHealthResponse())
        }
        router.get("/props") { _, _ async throws -> Response in
            try jsonResponse(ServerPropsResponse())
        }
        router.get("/metrics") { _, _ async -> Response in
            textResponse(await service.prometheusMetrics(), contentType: "text/plain; charset=utf-8")
        }
        router.get("/models") { _, _ async throws -> Response in
            try jsonResponse(try await service.availableModels())
        }
        router.get("/v1/models") { _, _ async throws -> Response in
            try jsonResponse(try await service.availableModels())
        }

        registerChatRoutes(router, service: service)
        registerCompletionRoutes(router, service: service)
        registerResponseRoutes(router, service: service)
        registerTokenRoutes(router, service: service)
        registerEmbeddingRoutes(router, service: service)

        return router
    }

    public static func buildApplication(
        service: MLXOpenAIService,
        host: String,
        port: Int
    ) -> Application<RouterResponder<BasicRequestContext>> {
        let router = buildRouter(service: service)
        return Application(
            router: router,
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: MLXLMServer.defaultServerName
            )
        )
    }

    private static func registerChatRoutes(
        _ router: Router<BasicRequestContext>,
        service: MLXOpenAIService
    ) {
        let handler:
            @Sendable (Request, BasicRequestContext) async throws -> Response =
            { request, context in
                let chatRequest = try await request.decode(
                    as: OpenAIChatCompletionRequest.self,
                    context: context
                )
                if chatRequest.stream == true {
                    let frames = try await service.streamChatCompletionFrames(request: chatRequest)
                    return sseResponse(frames)
                }
                return try jsonResponse(try await service.createChatCompletion(request: chatRequest))
            }

        router.post("/v1/chat/completions", use: handler)
        router.post("/chat/completions", use: handler)
        router.post("/v1/chat/completions/batch") { request, context async throws -> Response in
            let requests = try await request.decode(
                as: [OpenAIChatCompletionRequest].self,
                context: context
            )
            var responses: [OpenAIChatCompletionResponse] = []
            for chatRequest in requests {
                responses.append(try await service.createChatCompletion(request: chatRequest))
            }
            return try jsonResponse(responses)
        }
    }

    private static func registerCompletionRoutes(
        _ router: Router<BasicRequestContext>,
        service: MLXOpenAIService
    ) {
        let handler:
            @Sendable (Request, BasicRequestContext) async throws -> Response =
            { request, context in
                let completionRequest = try await request.decode(
                    as: OpenAICompletionRequest.self,
                    context: context
                )
                let chatRequest = completionRequest.chatCompletionRequest
                if completionRequest.stream == true {
                    let frames = try await service.streamChatCompletionFrames(request: chatRequest)
                    return sseResponse(frames)
                }
                let chatResponse = try await service.createChatCompletion(request: chatRequest)
                return try jsonResponse(OpenAICompletionResponse(from: chatResponse))
            }

        router.post("/v1/completions", use: handler)
        router.post("/completions", use: handler)
        router.post("/completion", use: handler)
    }

    private static func registerResponseRoutes(
        _ router: Router<BasicRequestContext>,
        service: MLXOpenAIService
    ) {
        let create:
            @Sendable (Request, BasicRequestContext) async throws -> Response =
            { request, context in
                let responseRequest = try await request.decode(
                    as: OpenAIResponseRequest.self,
                    context: context
                )
                if responseRequest.stream == true {
                    let frames = try await service.streamChatCompletionFrames(
                        request: responseRequest.chatCompletionRequest
                    )
                    return sseResponse(frames)
                }
                return try jsonResponse(try await service.createResponse(request: responseRequest))
            }

        router.post("/v1/responses", use: create)
        router.post("/responses", use: create)
        router.get("/v1/responses/:response_id") { _, context async throws -> Response in
            let id = try context.parameters.require("response_id")
            return try jsonResponse(try await service.retrieveResponse(id: id))
        }
        router.post("/v1/responses/:response_id/cancel") { _, context async throws -> Response in
            let id = try context.parameters.require("response_id")
            return try jsonResponse(try await service.cancelResponse(id: id))
        }
    }

    private static func registerTokenRoutes(
        _ router: Router<BasicRequestContext>,
        service: MLXOpenAIService
    ) {
        router.post("/tokenize") { request, context async throws -> Response in
            let tokenRequest = try await request.decode(as: TokenizeRequest.self, context: context)
            return try jsonResponse(try await service.tokenize(tokenRequest))
        }
        router.post("/detokenize") { request, context async throws -> Response in
            let detokenizeRequest = try await request.decode(
                as: DetokenizeRequest.self,
                context: context
            )
            return try jsonResponse(try await service.detokenize(detokenizeRequest))
        }
        router.post("/apply-template") { request, context async throws -> Response in
            let templateRequest = try await request.decode(
                as: ApplyTemplateRequest.self,
                context: context
            )
            return try jsonResponse(try await service.applyTemplate(templateRequest))
        }
    }

    private static func registerEmbeddingRoutes(
        _ router: Router<BasicRequestContext>,
        service: MLXOpenAIService
    ) {
        let handler:
            @Sendable (Request, BasicRequestContext) async throws -> Response =
            { request, context in
                let embeddingRequest = try await request.decode(
                    as: OpenAIEmbeddingRequest.self,
                    context: context
                )
                do {
                    return try jsonResponse(
                        try await service.createEmbedding(request: embeddingRequest)
                    )
                } catch MLXOpenAIServiceError.embeddingsNotConfigured {
                    return try jsonResponse(
                        OpenAIErrorResponse(
                            message: MLXOpenAIServiceError.embeddingsNotConfigured.localizedDescription,
                            type: "unsupported_endpoint",
                            code: "embeddings_not_configured"
                        ),
                        status: .notImplemented
                    )
                }
            }
        router.post("/v1/embeddings", use: handler)
        router.post("/embeddings", use: handler)
        router.post("/embedding", use: handler)
    }
}

private func jsonResponse<T: Encodable>(
    _ value: T,
    status: HTTPResponse.Status = .ok
) throws -> Response {
    let data = try JSONEncoder.openAIServer.encode(value)
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}

private func textResponse(_ text: String, contentType: String) -> Response {
    Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init(byteBuffer: ByteBuffer(string: text))
    )
}

private func sseResponse(_ frames: AsyncThrowingStream<String, Error>) -> Response {
    let body = AsyncThrowingStream<ByteBuffer, Error> { continuation in
        let task = Task {
            do {
                for try await frame in frames {
                    continuation.yield(ByteBuffer(string: frame))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in
            task.cancel()
        }
    }
    return Response(
        status: .ok,
        headers: [
            .contentType: "text/event-stream; charset=utf-8",
            .cacheControl: "no-cache",
        ],
        body: .init(asyncSequence: body)
    )
}
