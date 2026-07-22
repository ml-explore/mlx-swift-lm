import MLXLMCommon
import Testing

struct ToolCallIdTests {
    @Test("Assistant tool calls and tool results form a correlated transcript")
    func assistantToolCallsAndToolResultsFormCorrelatedTranscript() throws {
        let weatherCall = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: [
                    "location": .string("Paris")
                ]
            ),
            id: "call_123"
        )
        let timeCall = ToolCall(
            function: .init(
                name: "get_time",
                arguments: [
                    "timezone": .string("UTC")
                ]
            ),
            id: "call_456"
        )

        let messages: [Chat.Message] = [
            .assistant("", toolCalls: [weatherCall, timeCall]),
            .tool(#"{"temperature":18}"#, id: "call_123"),
            .tool(#"{"time":"12:00"}"#, id: "call_456"),
        ]

        let rawMessages = DefaultMessageGenerator().generate(messages: messages)
        let assistant = rawMessages[0]
        let calls = try #require(assistant["tool_calls"] as? [[String: any Sendable]])

        #expect(assistant["role"] as? String == "assistant")
        #expect(calls.count == 2)

        let weather = calls[0]
        #expect(weather["id"] as? String == "call_123")
        #expect(weather["type"] as? String == "function")

        let weatherFunction = try #require(weather["function"] as? [String: any Sendable])
        #expect(weatherFunction["name"] as? String == "get_weather")

        let weatherArguments = try #require(weatherFunction["arguments"] as? [String: any Sendable])
        #expect(weatherArguments["location"] as? String == "Paris")

        let time = calls[1]
        #expect(time["id"] as? String == "call_456")
        #expect(time["type"] as? String == "function")

        let timeFunction = try #require(time["function"] as? [String: any Sendable])
        #expect(timeFunction["name"] as? String == "get_time")
        let timeArguments = try #require(timeFunction["arguments"] as? [String: any Sendable])
        #expect(timeArguments["timezone"] as? String == "UTC")

        let weatherResult = rawMessages[1]
        #expect(weatherResult["role"] as? String == "tool")
        #expect(weatherResult["content"] as? String == #"{"temperature":18}"#)
        #expect(weatherResult["tool_call_id"] as? String == "call_123")

        let timeResult = rawMessages[2]
        #expect(timeResult["role"] as? String == "tool")
        #expect(timeResult["content"] as? String == #"{"time":"12:00"}"#)
        #expect(timeResult["tool_call_id"] as? String == "call_456")
    }

    @Test("Plain messages do not emit tool metadata")
    func plainMessageDoesNotEmitToolMetadata() throws {
        let dictionary = DefaultMessageGenerator().generate(message: .user("hi"))

        #expect(dictionary["role"] as? String == "user")
        #expect(dictionary["content"] as? String == "hi")
        #expect(dictionary["tool_call_id"] == nil)
        #expect(dictionary["tool_calls"] == nil)
    }

    @Test("ToolCall id defaults to nil for source compatibility")
    func toolCallDefaultIdIsNil() {
        let toolCall = ToolCall(function: .init(name: "noop", arguments: [:]))
        #expect(toolCall.id == nil)
    }

    @Test("Parsed tool calls preserve supplied ids")
    func parsedToolCallsPreserveSuppliedIDs() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"id":"call_model","name":"get_weather","arguments":{"location":"Paris"}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.id == "call_model")
        #expect(toolCall.function.name == "get_weather")
    }

    @Test("OpenAI style JSON tool calls preserve supplied ids")
    func openAIStyleJSONToolCallsPreserveSuppliedIDs() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"id":"call_openai","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.id == "call_openai")
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Nested OpenAI style JSON tool call ids are preserved")
    func nestedOpenAIStyleJSONToolCallIDsArePreserved() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"type":"function","function":{"id":"call_nested","name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.id == "call_nested")
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Parsed tool calls without ids receive generated ids")
    func parsedToolCallsWithoutIDsReceiveGeneratedIDs() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        let toolCall = try #require(processor.toolCalls.first)
        let id = try #require(toolCall.id)
        #expect(id.hasPrefix("call_"))

        let messages: [Chat.Message] = [
            .assistant("", toolCalls: [toolCall]),
            .tool(#"{"temperature":18}"#, id: id),
        ]
        let rawMessages = DefaultMessageGenerator().generate(messages: messages)
        let calls = try #require(rawMessages[0]["tool_calls"] as? [[String: any Sendable]])
        #expect(calls[0]["id"] as? String == id)
        #expect(rawMessages[1]["tool_call_id"] as? String == id)
    }

    @Test("Mistral format generates Mistral-compatible tool call ids")
    func mistralFormatGeneratesMistralCompatibleToolCallIDs() throws {
        let processor = ToolCallProcessor(format: .mistral)
        _ = processor.processChunk("[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}")
        processor.processEOS()

        let id = try #require(processor.toolCalls.first?.id)
        #expect(id.count == 9)
        #expect(
            id.utf8.allSatisfy { byte in
                (byte >= 48 && byte <= 57)
                    || (byte >= 65 && byte <= 90)
                    || (byte >= 97 && byte <= 122)
            })
    }

    @Test("Generated tool call ids are distinct within one assistant turn")
    func generatedToolCallIDsAreDistinctWithinOneAssistantTurn() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"name":"first_call","arguments":{}}</tool_call><tool_call>{"name":"second_call","arguments":{}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        #expect(processor.toolCalls.count == 2)
        let firstID = try #require(processor.toolCalls[0].id)
        let secondID = try #require(processor.toolCalls[1].id)
        #expect(firstID != secondID)
    }

    @Test("Duplicate supplied tool call ids are normalized")
    func duplicateSuppliedToolCallIDsAreNormalized() throws {
        let processor = ToolCallProcessor()
        let content =
            #"<tool_call>{"id":"call_duplicate","name":"first_call","arguments":{}}</tool_call><tool_call>{"id":"call_duplicate","name":"second_call","arguments":{}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)

        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].id == "call_duplicate")
        let secondID = try #require(processor.toolCalls[1].id)
        #expect(secondID != "call_duplicate")
        #expect(secondID.hasPrefix("call_"))
    }

    @Test("Tool result continuations preserve explicit ids")
    func toolResultContinuationsPreserveExplicitIDs() {
        let messages: [Chat.Message] = [
            .tool(#"{"temperature":18}"#, id: "call_123")
        ]

        let rawMessages = DefaultMessageGenerator().generate(messages: messages)
        #expect(rawMessages[0]["role"] as? String == "tool")
        #expect(rawMessages[0]["content"] as? String == #"{"temperature":18}"#)
        #expect(rawMessages[0]["tool_call_id"] as? String == "call_123")
    }

}
