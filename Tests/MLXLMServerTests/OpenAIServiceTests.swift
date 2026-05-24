import Foundation
import Hummingbird
import HummingbirdTesting
import MLXLMCommon
@testable import MLXLMServer
import Testing

struct OpenAIServiceTests {
    @Test("chat completion collects generated content, reasoning, tool calls, usage, and metrics")
    func chatCompletionCollectsContentReasoningToolCallsUsageAndMetrics() async throws {
        let engine = ScriptedServerEngine(events: [
            .content("<think>check constraints</think>Hello "),
            .content("there"),
            .toolCall(
                ToolCall(
                    function: .init(
                        name: "get_weather",
                        arguments: ["location": .string("San Francisco")]
                    )
                )
            ),
            .info(.init(promptTokens: 9, completionTokens: 3)),
        ])
        let service = MLXOpenAIService(engine: engine)

        let response = try await service.createChatCompletion(
            request: .test(
                messages: [.init(role: .user, content: .text("hi"))],
                tools: [.weatherTool],
                reasoningParser: .deepseekR1
            )
        )

        #expect(response.object == "chat.completion")
        #expect(response.model == "local-model")
        #expect(response.choices.first?.message.role == .assistant)
        #expect(response.choices.first?.message.content == .text("Hello there"))
        #expect(response.choices.first?.message.reasoningContent == "check constraints")
        #expect(response.choices.first?.message.toolCalls?.first?.function.name == "get_weather")
        #expect(response.choices.first?.finishReason == "tool_calls")
        #expect(response.usage.promptTokens == 9)
        #expect(response.usage.completionTokens == 3)

        let snapshot = await service.metricsSnapshot()
        #expect(snapshot.requestsTotal == 1)
        #expect(snapshot.chatCompletionsTotal == 1)
        #expect(snapshot.promptTokensTotal == 9)
        #expect(snapshot.completionTokensTotal == 3)

        let lastRequest = await engine.lastChatRequest
        #expect(lastRequest?.tools?.first?.function.name == "get_weather")
    }

    @Test("streaming chat completion emits OpenAI SSE frames, usage, and done sentinel")
    func streamingChatCompletionEmitsSSEFramesUsageAndDone() async throws {
        let engine = ScriptedServerEngine(events: [
            .content("hel"),
            .content("lo"),
            .info(.init(promptTokens: 2, completionTokens: 2)),
        ])
        let service = MLXOpenAIService(engine: engine)

        let stream = try await service.streamChatCompletionFrames(
            request: .test(
                stream: true,
                streamOptions: .init(includeUsage: true, continuousUsageStats: nil)
            )
        )
        var frames: [String] = []
        for try await frame in stream {
            frames.append(frame)
        }

        #expect(frames.first?.contains("\"role\":\"assistant\"") == true)
        #expect(frames.contains { $0.contains("\"content\":\"hel\"") })
        #expect(frames.contains { $0.contains("\"content\":\"lo\"") })
        #expect(frames.contains { $0.contains("\"prompt_tokens\":2") })
        #expect(frames.last == ServerSentEventEncoder.done)
    }

    @Test("streaming tool calls finish with tool_calls even when generation info follows")
    func streamingToolCallsPreserveFinishReason() async throws {
        let engine = ScriptedServerEngine(events: [
            .toolCall(
                ToolCall(
                    function: .init(
                        name: "get_weather",
                        arguments: ["location": .string("San Francisco")]
                    )
                )
            ),
            .info(.init(promptTokens: 3, completionTokens: 1)),
        ])
        let service = MLXOpenAIService(engine: engine)

        let stream = try await service.streamChatCompletionFrames(request: .test(stream: true))
        var frames: [String] = []
        for try await frame in stream {
            frames.append(frame)
        }

        #expect(frames.contains { $0.contains("\"tool_calls\"") })
        #expect(frames.contains { $0.contains("\"finish_reason\":\"tool_calls\"") })
    }

    @Test("JSON object response format injects instructions and returns normalized JSON")
    func jsonObjectResponseFormatInjectsInstructionsAndNormalizesJSON() async throws {
        let engine = ScriptedServerEngine(events: [
            .content("```json\n{\"city\":\"San Francisco\"}\n```"),
            .info(.init(promptTokens: 8, completionTokens: 4)),
        ])
        let service = MLXOpenAIService(engine: engine)

        let response = try await service.createChatCompletion(
            request: .test(responseFormat: .jsonObject())
        )

        #expect(response.choices.first?.message.content == .text(#"{"city":"San Francisco"}"#))

        let lastRequest = await engine.lastChatRequest
        #expect(lastRequest?.messages.contains {
            $0.role == .system && $0.textContent.contains("single valid JSON object")
        } == true)
    }

    @Test("JSON schema response format validates required properties")
    func jsonSchemaResponseFormatValidatesRequiredProperties() async throws {
        let schema = OpenAIJSONSchemaResponseFormat(
            name: "city",
            strict: true,
            schema: .object([
                "type": .string("object"),
                "properties": .object([
                    "city": .object(["type": .string("string")])
                ]),
                "required": .array([.string("city")]),
                "additionalProperties": .bool(false),
            ])
        )
        let service = MLXOpenAIService(
            engine: ScriptedServerEngine(events: [
                .content(#"{"country":"US"}"#),
                .info(.init(promptTokens: 8, completionTokens: 4)),
            ])
        )

        await #expect(throws: MLXOpenAIServiceError.self) {
            try await service.createChatCompletion(
                request: .test(responseFormat: .jsonSchema(schema))
            )
        }
    }

    @Test("JSON schema response format accepts matching output")
    func jsonSchemaResponseFormatAcceptsMatchingOutput() async throws {
        let schema = OpenAIJSONSchemaResponseFormat(
            name: "city",
            strict: true,
            schema: .object([
                "type": .string("object"),
                "properties": .object([
                    "city": .object(["type": .string("string")])
                ]),
                "required": .array([.string("city")]),
                "additionalProperties": .bool(false),
            ])
        )
        let service = MLXOpenAIService(
            engine: ScriptedServerEngine(events: [
                .content(#"Here is the object: {"city":"San Francisco"}"#),
                .info(.init(promptTokens: 8, completionTokens: 4)),
            ])
        )

        let response = try await service.createChatCompletion(
            request: .test(responseFormat: .jsonSchema(schema))
        )

        #expect(response.choices.first?.message.content == .text(#"{"city":"San Francisco"}"#))
    }

    @Test("response format rejects malformed JSON output")
    func responseFormatRejectsMalformedJSONOutput() async throws {
        let service = MLXOpenAIService(
            engine: ScriptedServerEngine(events: [
                .content(#"{"city":"San Francisco""#),
                .info(.init(promptTokens: 8, completionTokens: 4)),
            ])
        )

        await #expect(throws: MLXOpenAIServiceError.self) {
            try await service.createChatCompletion(
                request: .test(responseFormat: .jsonObject())
            )
        }
    }

    @Test("response format does not reject tool call completions without content")
    func responseFormatDoesNotRejectToolCallCompletionsWithoutContent() async throws {
        let service = MLXOpenAIService(
            engine: ScriptedServerEngine(events: [
                .toolCall(
                    ToolCall(
                        function: .init(
                            name: "get_weather",
                            arguments: ["location": .string("San Francisco")]
                        )
                    )
                ),
                .info(.init(promptTokens: 8, completionTokens: 4)),
            ])
        )

        let response = try await service.createChatCompletion(
            request: .test(
                tools: [.weatherTool],
                responseFormat: .jsonObject()
            )
        )

        #expect(response.choices.first?.finishReason == "tool_calls")
        #expect(response.choices.first?.message.toolCalls?.first?.function.name == "get_weather")
    }

    @Test("responses API creates, stores, retrieves, and cancels responses")
    func responsesAPICreatesStoresRetrievesAndCancelsResponses() async throws {
        let engine = ScriptedServerEngine(events: [
            .content("response text"),
            .info(.init(promptTokens: 4, completionTokens: 2)),
        ])
        let service = MLXOpenAIService(engine: engine)

        let created = try await service.createResponse(
            request: .init(
                model: "local-model",
                input: .text("Say hi"),
                instructions: "Be short.",
                tools: nil,
                toolChoice: nil,
                reasoning: nil,
                stream: nil,
                store: true,
                temperature: nil,
                topP: nil,
                maxOutputTokens: nil,
                metadata: nil
            )
        )

        #expect(created.object == "response")
        #expect(created.status == .completed)
        #expect(created.outputText == "response text")
        #expect(created.output.first?.content?.first?.text == "response text")

        let retrieved = try await service.retrieveResponse(id: created.id)
        #expect(retrieved.id == created.id)
        #expect(retrieved.outputText == "response text")

        let cancelled = try await service.cancelResponse(id: created.id)
        #expect(cancelled.status == .cancelled)

        let snapshot = await service.metricsSnapshot()
        #expect(snapshot.responsesTotal == 1)
    }

    @Test("token utility requests delegate through the engine")
    func tokenUtilityRequestsDelegateThroughTheEngine() async throws {
        let engine = ScriptedServerEngine(events: [])
        let service = MLXOpenAIService(engine: engine)

        let tokenized = try await service.tokenize(.init(model: "local-model", prompt: "hello"))
        let detokenized = try await service.detokenize(.init(model: "local-model", tokens: [1, 2]))
        let templated = try await service.applyTemplate(
            .init(model: "local-model", messages: [.init(role: .user, content: .text("hello"))], tools: nil)
        )

        #expect(tokenized.tokens == [1, 2])
        #expect(detokenized.text == "hello")
        #expect(templated.tokens == [7, 8, 9])
    }

    @Test("embeddings endpoint delegates to optional embedding engine")
    func embeddingsEndpointDelegatesToOptionalEmbeddingEngine() async throws {
        let service = MLXOpenAIService(
            engine: ScriptedServerEngine(events: []),
            embeddingEngine: ScriptedEmbeddingEngine()
        )

        let response = try await service.createEmbedding(
            request: .init(model: "embed-model", input: .texts(["a", "b"]))
        )

        #expect(response.object == "list")
        #expect(response.model == "embed-model")
        #expect(response.data.count == 2)
        #expect(response.data[0].embedding == [0.1, 0.2])
        #expect(response.usage.promptTokens == 2)
    }

    @Test("embeddings endpoint reports not configured without embedding engine")
    func embeddingsEndpointReportsNotConfiguredWithoutEmbeddingEngine() async throws {
        let service = MLXOpenAIService(engine: ScriptedServerEngine(events: []))

        await #expect(throws: MLXOpenAIServiceError.embeddingsNotConfigured) {
            try await service.createEmbedding(
                request: .init(model: "embed-model", input: .text("a"))
            )
        }
    }

    @Test("Hummingbird application serves health, models, chat, and responses routes")
    func hummingbirdApplicationServesCoreRoutes() async throws {
        let engine = ScriptedServerEngine(events: [
            .content("route ok"),
            .info(.init(promptTokens: 1, completionTokens: 2)),
        ])
        let service = MLXOpenAIService(
            engine: engine,
            embeddingEngine: ScriptedEmbeddingEngine()
        )
        let app = MLXServerApplication.buildApplication(service: service, host: "127.0.0.1", port: 8080)

        try await app.test(.router) { client in
            try await client.execute(uri: "/health", method: .get) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"status\":\"ok\""))
            }

            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"id\":\"local-model\""))
            }

            let chatBody = ByteBuffer(
                string:
                    #"{"model":"local-model","messages":[{"role":"user","content":"hi"}]}"#
            )
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"route ok\""))
            }

            let responseBody = ByteBuffer(string: #"{"model":"local-model","input":"hi"}"#)
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                headers: [.contentType: "application/json"],
                body: responseBody
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"object\":\"response\""))
            }

            let embeddingBody = ByteBuffer(string: #"{"model":"embed-model","input":["a","b"]}"#)
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: embeddingBody
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("\"embedding\":[0.1,0.2]"))
            }
        }
    }

    @Test("Hummingbird chat route forwards response format to generation")
    func hummingbirdChatRouteForwardsResponseFormatToGeneration() async throws {
        let engine = ScriptedServerEngine(events: [
            .content(#"{"answer":"OK"}"#),
            .info(.init(promptTokens: 1, completionTokens: 2)),
        ])
        let service = MLXOpenAIService(engine: engine)
        let app = MLXServerApplication.buildApplication(service: service, host: "127.0.0.1", port: 8080)

        try await app.test(.router) { client in
            let chatBody = ByteBuffer(
                string:
                    #"{"model":"local-model","messages":[{"role":"user","content":"Return JSON"}],"response_format":{"type":"json_object"}}"#
            )
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: chatBody
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains(#""content":"{\"answer\":\"OK\"}""#))
            }
        }

        let lastRequest = await engine.lastChatRequest
        #expect(lastRequest?.responseFormat?.type == .jsonObject)
        #expect(lastRequest?.messages.contains {
            $0.role == .system && $0.textContent.contains("single valid JSON object")
        } == true)
    }
}

private actor ScriptedServerEngine: MLXServerEngine {
    var lastChatRequest: OpenAIChatCompletionRequest?
    private let events: [MLXServerGenerationEvent]

    init(events: [MLXServerGenerationEvent]) {
        self.events = events
    }

    func availableModels() async throws -> [MLXServerModel] {
        [.init(id: "local-model")]
    }

    func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        lastChatRequest = request
        let events = self.events
        return AsyncThrowingStream { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }

    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        .init(tokens: [1, 2])
    }

    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(text: "hello")
    }

    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        .init(tokens: [7, 8, 9])
    }
}

private struct ScriptedEmbeddingEngine: MLXEmbeddingServerEngine {
    func availableModels() async throws -> [MLXServerModel] {
        [.init(id: "embed-model")]
    }

    func createEmbedding(request: OpenAIEmbeddingRequest) async throws -> OpenAIEmbeddingResponse {
        .init(
            data: request.input.texts.enumerated().map { index, _ in
                .init(embedding: index == 0 ? [0.1, 0.2] : [0.3, 0.4], index: index)
            },
            model: request.model,
            usage: .init(promptTokens: request.input.texts.count, completionTokens: 0)
        )
    }
}

private extension ServerGenerationInfo {
    init(promptTokens: Int, completionTokens: Int) {
        self.init(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            promptTime: 0.01,
            generationTime: 0.01,
            stopReason: "stop"
        )
    }
}

private extension OpenAIChatCompletionRequest {
    static func test(
        model: String = "local-model",
        messages: [OpenAIChatMessage] = [.init(role: .user, content: .text("hello"))],
        tools: [OpenAITool]? = nil,
        reasoningParser: ReasoningParserFormat? = nil,
        responseFormat: OpenAIResponseFormat? = nil,
        stream: Bool? = nil,
        streamOptions: OpenAIStreamOptions? = nil
    ) -> OpenAIChatCompletionRequest {
        .init(
            model: model,
            messages: messages,
            tools: tools,
            toolChoice: nil,
            toolCallParser: nil,
            reasoningParser: reasoningParser,
            responseFormat: responseFormat,
            stream: stream,
            temperature: nil,
            topP: nil,
            topK: nil,
            minP: nil,
            maxTokens: nil,
            presencePenalty: nil,
            frequencyPenalty: nil,
            repetitionPenalty: nil,
            stop: nil,
            streamOptions: streamOptions
        )
    }
}

private extension OpenAIChatMessage {
    init(role: OpenAIRole, content: OpenAIMessageContent) {
        self.init(role: role, content: content, name: nil, toolCallID: nil, toolCalls: nil)
    }
}

private extension OpenAITool {
    static let weatherTool = OpenAITool(
        type: "function",
        function: .init(
            name: "get_weather",
            description: "Get weather",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object(["type": .string("string")])
                ]),
            ])
        )
    )
}
