import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

struct ServerSurfaceTests {
    @Test("server route manifest exposes production inference endpoints")
    func routeManifestExposesProductionInferenceEndpoints() {
        let routes = MLXServerRoute.manifest
        let pairs = Set(routes.map { "\($0.method.rawValue) \($0.path)" })

        #expect(pairs.contains("GET /health"))
        #expect(pairs.contains("GET /v1/health"))
        #expect(pairs.contains("GET /metrics"))
        #expect(pairs.contains("GET /v1/models"))
        #expect(pairs.contains("POST /v1/chat/completions"))
        #expect(pairs.contains("POST /chat/completions"))
        #expect(pairs.contains("POST /v1/completions"))
        #expect(pairs.contains("POST /completion"))
        #expect(pairs.contains("POST /v1/responses"))
        #expect(pairs.contains("POST /responses"))
        #expect(pairs.contains("GET /v1/responses/{response_id}"))
        #expect(pairs.contains("POST /v1/responses/{response_id}/cancel"))
        #expect(pairs.contains("POST /v1/embeddings"))
        #expect(pairs.contains("POST /tokenize"))
        #expect(pairs.contains("POST /detokenize"))
    }

    @Test("OpenAI chat request decodes tools, stream flag, tool parser, and reasoning parser")
    func openAIChatRequestDecodesAgentFields() throws {
        let json = """
            {
              "model": "mlx-community/Qwen3-4B-4bit",
              "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "weather in SF?"}
              ],
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                      "type": "object",
                      "properties": {"location": {"type": "string"}},
                      "required": ["location"]
                    }
                  }
                }
              ],
              "tool_choice": "auto",
              "tool_call_parser": "mistral",
              "reasoning_parser": "deepseek_r1",
              "response_format": {"type": "json_object"},
              "stream": true,
              "temperature": 0.2,
              "top_p": 0.95,
              "max_tokens": 128
            }
            """

        let request = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        #expect(request.model == "mlx-community/Qwen3-4B-4bit")
        #expect(request.messages.count == 2)
        #expect(request.messages[1].textContent == "weather in SF?")
        #expect(request.tools?.first?.function.name == "get_weather")
        #expect(request.toolChoice == .mode(.auto))
        #expect(request.toolCallParser == "mistral")
        #expect(request.reasoningParser == .deepseekR1)
        #expect(request.responseFormat?.type == .jsonObject)
        #expect(request.stream == true)
        #expect(request.generationParameters.maxTokens == 128)
        #expect(request.generationParameters.temperature == 0.2)
        #expect(request.generationParameters.topP == 0.95)
    }

    @Test("OpenAI chat request decodes JSON schema response format")
    func openAIChatRequestDecodesJSONSchemaResponseFormat() throws {
        let json = """
            {
              "model": "local-model",
              "messages": [{"role": "user", "content": "extract"}],
              "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "city",
                  "description": "City extraction result",
                  "strict": true,
                  "schema": {
                    "type": "object",
                    "properties": {
                      "city": {"type": "string"}
                    },
                    "required": ["city"],
                    "additionalProperties": false
                  }
                }
              }
            }
            """

        let request = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        #expect(request.responseFormat?.type == .jsonSchema)
        #expect(request.responseFormat?.jsonSchema?.name == "city")
        #expect(request.responseFormat?.jsonSchema?.strict == true)
        #expect(request.responseFormat?.jsonSchema?.schema == .object([
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string")])
            ]),
            "required": .array([.string("city")]),
            "additionalProperties": .bool(false),
        ]))
    }

    @Test("server tool parser resolver supports named and auto formats")
    func serverToolParserResolverSupportsNamedAndAutoFormats() throws {
        #expect(try ServerToolParser.resolve(requested: "mistral", modelType: nil) == .mistral)
        #expect(try ServerToolParser.resolve(requested: "llama3_json", modelType: nil) == .llama3)
        #expect(try ServerToolParser.resolve(requested: "gemma4", modelType: nil) == .gemma)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "gemma4") == .gemma)
        #expect(try ServerToolParser.resolve(requested: "auto", modelType: "qwen3_5") == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "lfm2") == .lfm2)
    }

    @Test("reasoning parser extracts think blocks without leaking them into final content")
    func reasoningParserExtractsThinkBlocks() {
        let parser = ReasoningParser(format: .deepseekR1)
        let parsed = parser.parse("<think>check constraints</think>The answer is 42.")

        #expect(parsed.reasoningContent == "check constraints")
        #expect(parsed.content == "The answer is 42.")
    }

    @Test("reasoning parser extracts harmony analysis and final channels")
    func reasoningParserExtractsHarmonyChannels() {
        let parser = ReasoningParser(format: .harmony)
        let parsed = parser.parse(
            "<|channel|>analysis<|message|>use tool first<|end|><|channel|>final<|message|>done<|end|>"
        )

        #expect(parsed.reasoningContent == "use tool first")
        #expect(parsed.content == "done")
    }

    @Test("SSE encoder formats OpenAI data frames and done sentinel")
    func sseEncoderFormatsDataFramesAndDoneSentinel() throws {
        let frame = try ServerSentEventEncoder.encode(
            OpenAIChatCompletionChunk(
                id: "chatcmpl-test",
                model: "local-model",
                choices: [
                    .init(
                        index: 0,
                        delta: .init(role: nil, content: "hello", reasoningContent: nil, toolCalls: nil),
                        finishReason: nil
                    )
                ],
                usage: nil
            )
        )

        #expect(frame.hasPrefix("data: {"))
        #expect(frame.contains("\"object\":\"chat.completion.chunk\""))
        #expect(frame.contains("\"content\":\"hello\""))
        #expect(ServerSentEventEncoder.done == "data: [DONE]\n\n")
    }
}
