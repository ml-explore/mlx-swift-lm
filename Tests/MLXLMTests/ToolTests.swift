import Foundation
import MLXLMCommon
import Testing

struct ToolTests {
    private func toolSchemas(_ names: String...) -> [[String: any Sendable]] {
        names.map { name in
            ["function": ["name": name] as [String: any Sendable]]
        }
    }

    @Test("Rejected tool-call preview is bounded and omitted from error descriptions")
    func rejectedToolCallDiagnosticPrivacy() throws {
        let rawText = "secret-" + String(repeating: "🧪", count: 20) + "-tail"
        let toolName = "secret-model-provided-name"
        let rejection = RejectedToolCall(
            reason: .malformedSyntax,
            format: .json,
            toolName: toolName,
            rawText: rawText,
            detail: "Could not parse payload.",
            previewByteLimit: 24)

        #expect(rejection.rawTextByteCount == rawText.utf8.count)
        #expect(rejection.isPreviewTruncated)
        #expect(rejection.rawTextPreview.hasPrefix("secret-"))
        #expect(rejection.rawTextPreview.hasSuffix("-tail"))

        let message = try #require(RejectedToolCallError(rejection).errorDescription)
        #expect(!message.contains("secret"))
        #expect(!message.contains(toolName))
        #expect(message.contains(RejectedToolCall.Reason.malformedSyntax.rawValue))
    }

    @Test("Lampo Qwen hybrid protocol is rejected at every chunk boundary")
    func lampoHybridProtocolRejectedAtEveryChunkBoundary() throws {
        let payload = """
            <tool_call>
            <|im_start|>mcp_filo_7bb95d18__read_file>
            <parameter=path>
            /Users/alessiopollero/dev/Lampo/Lampo/Lampo.docc/Lampo.md
            </parameter>
            </mcp:parameter>
            </mcp:tool>
            </mcp:thought>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("mcp_filo_7bb95d18__read_file"))
            var outputs = processor.processChunkOutputs(String(payload[..<split]))
            outputs += processor.processChunkOutputs(String(payload[split...]))
            outputs += processor.processEOSOutputs()

            let rejections = outputs.compactMap { output -> RejectedToolCall? in
                guard case .rejectedToolCall(let rejection) = output else { return nil }
                return rejection
            }
            #expect(rejections.count == 1)
            let rejection = try #require(rejections.first)
            #expect(rejection.reason == .incompleteOutput)
            #expect(rejection.format == .xmlFunction)
            #expect(!outputs.contains { if case .response = $0 { true } else { false } })
            #expect(processor.rejectedToolCallCount == 1)
        }
    }

    @Test("Rejected and accepted calls retain source order")
    func rejectedAndAcceptedCallsRetainSourceOrder() throws {
        let processor = ToolCallProcessor(format: .json, tools: toolSchemas("allowed"))
        let outputs = processor.processChunkOutputs(
            #"before<tool_call>{"name":"unknown","arguments":{}}</tool_call>between<tool_call>{"name":"allowed","arguments":{}}</tool_call>after"#
        )

        #expect(outputs.count == 5)
        #expect(outputs[0] == .response("before"))
        guard case .rejectedToolCall(let rejection) = outputs[1] else {
            Issue.record("Expected an undeclared-tool rejection")
            return
        }
        #expect(rejection.reason == .undeclaredTool)
        #expect(rejection.toolName == "unknown")
        #expect(outputs[2] == .response("between"))
        guard case .toolCall(let call) = outputs[3] else {
            Issue.record("Expected the declared call")
            return
        }
        #expect(call.function.name == "allowed")
        #expect(outputs[4] == .response("after"))
    }

    @Test("LFM2 EOS orders response before an unfinished second call")
    func lfm2EOSOrdersIncompleteSecondCall() {
        let processor = ToolCallProcessor(format: .lfm2, tools: toolSchemas("get_weather"))
        #expect(
            processor.processChunkOutputs(
                "<|tool_call_start|>[get_weather()]between <|tool_call_start|>[get_weather("
            ).isEmpty)

        let outputs = processor.processEOSOutputs()
        #expect(outputs.count == 3)
        guard outputs.count == 3 else { return }
        guard case .toolCall = outputs[0] else {
            Issue.record("Expected the completed first call")
            return
        }
        #expect(outputs[1] == .response("between "))
        guard case .rejectedToolCall(let rejection) = outputs[2] else {
            Issue.record("Expected an incomplete second call")
            return
        }
        #expect(rejection.reason == .incompleteOutput)
    }

    @Test("LFM2 EOS preserves events around a malformed inter-call marker")
    func lfm2EOSOrdersMalformedInterCallMarker() {
        let processor = ToolCallProcessor(format: .lfm2, tools: toolSchemas("get_weather"))
        #expect(
            processor.processChunkOutputs(
                "<|tool_call_start|>[get_weather()]before <|tool_call_startX> after <|tool_call_start|>[get_weather()]"
            ).isEmpty)

        let outputs = processor.processEOSOutputs()
        #expect(outputs.count == 5)
        guard outputs.count == 5 else { return }
        guard case .toolCall = outputs[0] else {
            Issue.record("Expected the completed first call")
            return
        }
        #expect(outputs[1] == .response("before "))
        guard case .rejectedToolCall(let rejection) = outputs[2] else {
            Issue.record("Expected a malformed marker rejection")
            return
        }
        #expect(rejection.reason == .malformedSyntax)
        #expect(outputs[3] == .response(" after "))
        guard case .toolCall = outputs[4] else {
            Issue.record("Expected the completed second call")
            return
        }
    }

    @Test("Tagged JSON rejection reasons distinguish name and argument failures")
    func taggedJSONRejectionReasonsAreSpecific() throws {
        let cases: [(String, RejectedToolCall.Reason)] = [
            (#"<tool_call>{"arguments":{}}</tool_call>"#, .missingToolName),
            (#"<tool_call>{"name":"allowed","arguments":[]}</tool_call>"#, .invalidArguments),
            (#"<tool_call>{"name":}</tool_call>"#, .malformedSyntax),
        ]

        for (payload, expectedReason) in cases {
            let processor = ToolCallProcessor(format: .json, tools: toolSchemas("allowed"))
            let outputs = processor.processChunkOutputs(payload)
            #expect(outputs.count == 1)
            guard case .rejectedToolCall(let rejection)? = outputs.first else {
                Issue.record("Expected a rejection for \(payload)")
                continue
            }
            #expect(rejection.reason == expectedReason)
        }
    }

    @Test("ChatConventionsProviding defaults to nil for both properties")
    func chatConventionsOptInDefaults() {
        struct Bare: ChatConventionsProviding {}
        #expect(Bare().toolCallFormat == nil)
        #expect(Bare().reasoningConfig == nil)
    }

    @Test("ToolCallProcessor drains calls once in parse order")
    func toolCallProcessorPublicDrain() {
        let processor = ToolCallProcessor(format: .json)
        _ = processor.processChunk(
            #"<tool_call>{"name":"first","arguments":{}}</tool_call><tool_call>{"name":"second","arguments":{}}</tool_call>"#
        )

        #expect(processor.drainToolCalls().map(\.function.name) == ["first", "second"])
        #expect(processor.drainToolCalls().isEmpty)
    }

    @Test("ToolCallProcessor ordered outputs retain split call-text-call order")
    func toolCallProcessorOrderedSplitOutput() {
        let processor = ToolCallProcessor(format: .json)
        #expect(
            processor.processChunkOutputs(
                #"<tool_call>{"name":"first","arguments":{"#
            ).isEmpty)

        let outputs = processor.processChunkOutputs(
            #"}}</tool_call>between<tool_call>{"name":"second","arguments":{}}</tool_call>"#)
        #expect(outputs.count == 3)
        guard case .toolCall(let first) = outputs[0] else {
            Issue.record("Expected first call")
            return
        }
        #expect(first.function.name == "first")
        #expect(outputs[1] == .response("between"))
        guard case .toolCall(let second) = outputs[2] else {
            Issue.record("Expected second call")
            return
        }
        #expect(second.function.name == "second")
    }

    @Test("Test Weather Tool Schema Generation")
    func testWeatherToolSchemaGeneration() throws {
        struct WeatherInput: Codable {
            let location: String
            let unit: String?
        }

        struct WeatherOutput: Codable {
            let temperature: Double
            let conditions: String
        }

        let tool = Tool<WeatherInput, WeatherOutput>(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .required(
                    "location", type: .string, description: "The city, e.g. Istanbul"
                ),
                .optional(
                    "unit",
                    type: .string,
                    description: "The unit of temperature",
                    extraProperties: [
                        "enum": ["celsius", "fahrenheit"]
                    ]
                ),
            ]
        ) { input in
            WeatherOutput(temperature: 14.0, conditions: "Sunny")
        }

        let actual = tool.schema as NSDictionary

        let expected: NSDictionary = [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city, e.g. Istanbul",
                        ],
                        "unit": [
                            "type": "string",
                            "description": "The unit of temperature",
                            "enum": ["celsius", "fahrenheit"],
                        ],
                    ],
                    "required": ["location"],
                ],
            ],
        ]

        #expect(actual == expected)
    }

    @Test("Test Tool Call Detection in Generated Text - Default JSON Format")
    func testToolCallDetection() throws {
        let processor = ToolCallProcessor()
        let chunks: [String] = [
            "<tool", "_", "call>", "{", "\"", "name", "\"", ":", " ", "\"", "get", "_", "current",
            "_", "weather", "\"", ",", " ", "\"", "arguments", "\"", ":", " ", "{", "\"",
            "location", "\"", ":", " ", "\"", "San", " Francisco", "\"", ",", " ", "\"", "unit",
            "\"", ":", " ", "\"", "celsius", "\"", "}", "}", "</tool", "_", "call>",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            #expect(result == nil)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)

        #expect(toolCall.function.name == "get_current_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    // MARK: - JSON Format Tests

    @Test("Test JSON Tool Call Parser - Default Tags")
    func testJSONParserDefaultTags() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test JSON Tool Call Parser - Custom Tags")
    func testJSONParserCustomTags() throws {
        let parser = JSONToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>{\"name\": \"search\", \"arguments\": {\"query\": \"swift programming\"}}<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test JSON Tool Call Parser - Stringified Arguments")
    func testJSONParserStringifiedArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test JSON Tool Call Parser - Stringified Empty Arguments")
    func testJSONParserStringifiedEmptyArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"current_time","arguments":"{}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test JSON Tool Call Parser - Stringified Array Arguments")
    func testJSONParserStringifiedArrayArguments() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            #"<tool_call>{"name":"search_many","arguments":"{\"queries\":[\"swift\",\"mlx\"],\"limit\":2}"}</tool_call>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search_many")
        #expect(toolCall.function.arguments["limit"] == .int(2))
        #expect(
            toolCall.function.arguments["queries"] == .array([.string("swift"), .string("mlx")]))
    }

    @Test("Test JSON Format via ToolCallProcessor - Bare JSON Fallback")
    func testJSONFormatProcessorBareJSONFallback() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunks: [String] = [
            "{\"name\": \"get_weather\", ",
            "\"arguments\": {\"location\": \"Rome\"}}",
        ]

        var emittedText = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) {
                emittedText += text
            }
        }

        if let text = processor.processEOS(returnBufferedText: true) {
            emittedText += text
        }

        #expect(emittedText.isEmpty)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Rome"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Bare JSON With Leading Text")
    func testJSONFormatProcessorBareJSONWithLeadingText() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "Let me check that.\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Milan\"}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "Let me check that.\n")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Milan"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Tagged JSON In Single Chunk")
    func testJSONFormatProcessorTaggedSingleChunk() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Tokyo\"}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Multiple Tagged Calls Preserve Order")
    func testJSONFormatProcessorMultipleTaggedCallsPreserveOrder() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "<tool_call>{\"name\":\"first_call\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"second_call\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].function.name == "first_call")
        #expect(processor.toolCalls[1].function.name == "second_call")
    }

    @Test("Test JSON Format via ToolCallProcessor - Tagged JSON With Leading Text")
    func testJSONFormatProcessorTaggedWithLeadingText() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk =
            "Let me check that.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Osaka\"}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "Let me check that.\n")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Osaka"))
    }

    @Test("Test JSON Format via ToolCallProcessor - Incomplete Bare Tool Rejects At EOS")
    func testJSONFormatProcessorIncompleteBareToolRejectsAtEOS() throws {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"name\": \"get_weather\", \"arguments\": "

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
        let rejections = processor.drainRejectedToolCalls()
        #expect(rejections.count == 1)
        let rejection = try #require(rejections.first)
        #expect(rejection.reason == .incompleteOutput)
    }

    @Test("Test JSON Format via ToolCallProcessor - Non Tool JSON Stays Text")
    func testJSONFormatProcessorNonToolJSONStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"status\": \"ok\", \"data\": {\"value\": 42}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == chunk)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Split Non Tool JSON Stays Text")
    func testJSONFormatProcessorSplitNonToolJSONStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunks = ["{\"status\": ", "\"ok\", \"data\": {\"value\": 42}}"]

        var emittedText = ""
        for chunk in chunks {
            if let output = processor.processChunk(chunk) {
                emittedText += output
            }
        }

        if let eosOutput = processor.processEOS(returnBufferedText: true) {
            emittedText += eosOutput
        }

        #expect(emittedText == "{\"status\": \"ok\", \"data\": {\"value\": 42}}")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Missing Arguments Stays Text")
    func testJSONFormatProcessorMissingArgumentsStaysText() {
        let processor = ToolCallProcessor(format: .json)
        let chunk = "{\"name\": \"not_a_tool_call_payload\"}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == chunk)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Brace Text Is Not Treated As JSON Tool Call")
    func testJSONFormatProcessorBraceTextNotToolCall() {
        let processor = ToolCallProcessor(format: .json)

        let first = processor.processChunk("Use {")
        let second = processor.processChunk("x} notation")
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(first == "Use ")
        #expect(second == "{x} notation")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Test JSON Format via ToolCallProcessor - Unknown Tool Name Is Rejected")
    func testJSONFormatProcessorUnknownToolNameIsRejected() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk = "{\"name\": \"not_declared\", \"arguments\": {}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.isEmpty)
        let rejections = processor.drainRejectedToolCalls()
        #expect(rejections.count == 1)
        let rejection = try #require(rejections.first)
        #expect(rejection.reason == .undeclaredTool)
        #expect(rejection.toolName == "not_declared")
    }

    @Test("Test JSON Format via ToolCallProcessor - Recovers Tagged Tool Call After Brace Text")
    func testJSONFormatProcessorRecoversTaggedToolCallAfterBraceText() throws {
        let processor = ToolCallProcessor(format: .json)
        var emittedText = ""

        if let output = processor.processChunk("note {x") {
            emittedText += output
        }
        if let output = processor.processChunk(
            "} <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}</tool_call>"
        ) {
            emittedText += output
        }
        if let eosOutput = processor.processEOS(returnBufferedText: true) {
            emittedText += eosOutput
        }

        #expect(emittedText == "note {x} ")
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test JSON Processor - Unknown Tagged Tool Rejected And Continues Parsing")
    func testJSONFormatProcessorUnknownTaggedToolRejectedAndContinuesParsing() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk =
            "<tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)
        let rejections = processor.drainRejectedToolCalls()
        #expect(rejections.count == 1)
        let rejection = try #require(rejections.first)
        #expect(rejection.reason == .undeclaredTool)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    @Test("Test JSON Processor - Unknown Tagged Tool Preserves Only Leading Text")
    func testJSONFormatProcessorUnknownTaggedToolPreservesOnlyLeadingText() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk =
            "Preface <tool_call>{\"name\":\"not_declared\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == "Preface ")
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)
        let rejections = processor.drainRejectedToolCalls()
        #expect(rejections.count == 1)
        let rejection = try #require(rejections.first)
        #expect(rejection.reason == .undeclaredTool)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    @Test(
        "Test JSON Format via ToolCallProcessor - Declared Tool Name Parses When Tools Are Provided"
    )
    func testJSONFormatProcessorDeclaredToolNameParsesWithTools() throws {
        struct EmptyInput: Codable {}
        struct EmptyOutput: Codable { let ok: Bool }

        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather",
            parameters: []
        ) { _ in
            EmptyOutput(ok: true)
        }

        let processor = ToolCallProcessor(format: .json, tools: [tool.schema])
        let chunk = "{\"name\": \"get_weather\", \"arguments\": {}}"

        let output = processor.processChunk(chunk)
        let eosOutput = processor.processEOS(returnBufferedText: true)

        #expect(output == nil)
        #expect(eosOutput == nil)
        #expect(processor.toolCalls.count == 1)

        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
    }

    // MARK: - Pythonic Format Tests (LFM2/LFM2.5)

    @Test("Test Pythonic Tool Call Parser - Basic")
    func testPythonicParserBasic() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[get_weather(location='Paris', unit='celsius')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Pythonic Tool Call Parser - Object Wrapper Argument (LFM2)")
    func testPythonicParserObjectWrapperArgument() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        // LFM2 emits the full parameter object under a `properties` wrapper key.
        // The object also contains a comma the old `[^,\)]+` value regex truncated on.
        let content =
            "<|tool_call_start|>[get_weather(properties={\"location\": \"Tokyo\", \"unit\": \"celsius\"})]<|tool_call_end|>"
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "properties": [
                            "location": ["type": "string"],
                            "unit": ["type": "string"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Pythonic Tool Call Parser - Object-Valued Argument Preserved")
    func testPythonicParserObjectValuedArgument() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        // A non-wrapper key is not unwrapped; the object value (with its inner
        // comma) is parsed intact rather than truncated.
        let content =
            "<|tool_call_start|>[configure(settings={\"width\": 10, \"height\": 20})]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "configure")
        #expect(
            toolCall.function.arguments["settings"]
                == .object(["width": .int(10), "height": .int(20)]))
    }

    @Test("Test Pythonic Tool Call Parser - Double Quotes")
    func testPythonicParserDoubleQuotes() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[search(query=\"swift programming\")]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test Pythonic Tool Call Parser - Without Brackets")
    func testPythonicParserWithoutBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>current_time(timezone='UTC')<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses in Argument Value")
    func testPythonicParserNestedParentheses() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[run_script(code=\"response = requests.get('https://api.example.com/data')\")] <|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(
            toolCall.function.arguments["code"]
                == .string("response = requests.get('https://api.example.com/data')"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses Without Brackets")
    func testPythonicParserNestedParenthesesNoBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>run_script(code=\"print('hello')\")<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(toolCall.function.arguments["code"] == .string("print('hello')"))
    }

    @Test("Test Pythonic Tool Call Parser - No Arguments")
    func testPythonicParserNoArguments() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[current_time()]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test Pythonic Tool Call Parser - Multiple Tools via parseEOS")
    func testPythonicParserMultipleToolsEOS() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")

        let content1 =
            "<|tool_call_start|>[get_weather(location='Paris'), current_time(timezone=\"UTC\")]<|tool_call_end|>"
        let toolCalls1 = parser.parseEOS(content1, tools: nil)

        #expect(toolCalls1.count == 2)
        #expect(toolCalls1[0].function.name == "get_weather")
        #expect(toolCalls1[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls1[1].function.name == "current_time")
        #expect(toolCalls1[1].function.arguments["timezone"] == .string("UTC"))

        // Multiple distinct tool call blocks
        let content2 =
            "<|tool_call_start|>[get_weather(location='London')]<|tool_call_end|> <text> <|tool_call_start|>[current_time(timezone='UTC')]<|tool_call_end|>"
        let toolCalls2 = parser.parseEOS(content2, tools: nil)

        #expect(toolCalls2.count == 2)
        #expect(toolCalls2[0].function.name == "get_weather")
        #expect(toolCalls2[0].function.arguments["location"] == .string("London"))
        #expect(toolCalls2[1].function.name == "current_time")
        #expect(toolCalls2[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Pythonic argument values may contain the closing delimiters")
    func testPythonicParserDelimitersInsideValue() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")

        // `)]` closes the call for a scanner that is not quote aware, which
        // truncated the value and left the stray quote behind.
        let quoted = "<|tool_call_start|>[notify(note='a)]b')]<|tool_call_end|>"
        let quotedCall = try #require(parser.parse(content: quoted, tools: nil))
        #expect(quotedCall.function.name == "notify")
        #expect(quotedCall.function.arguments["note"] == .string("a)]b"))

        let object = #"<|tool_call_start|>[notify(filters={"x": "a)]b"})]<|tool_call_end|>"#
        let objectCall = try #require(parser.parse(content: object, tools: nil))
        #expect(objectCall.function.arguments["filters"] == .object(["x": .string("a)]b")]))
    }

    @Test("Test Pythonic Tool Call Parser - Array Value With Inner Commas")
    func testPythonicParserArrayValue() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            #"<|tool_call_start|>[search_many(queries=["swift", "mlx"], limit=2)]<|tool_call_end|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search_many")
        #expect(
            toolCall.function.arguments["queries"] == .array([.string("swift"), .string("mlx")]))
        #expect(toolCall.function.arguments["limit"] == .int(2))
    }

    @Test("Pythonic collections accept Python literal syntax")
    func testPythonicParserPythonLiteralCollections() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content = #"""
            <|tool_call_start|>[configure(settings={'location': 'Tokyo', 'enabled': True, 'fallback': None, 'thresholds': [0, 1.5]})]<|tool_call_end|>
            """#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(
            toolCall.function.arguments["settings"]
                == .object([
                    "location": .string("Tokyo"),
                    "enabled": .bool(true),
                    "fallback": .null,
                    "thresholds": .array([.int(0), .double(1.5)]),
                ]))
    }

    @Test("Pythonic scalars are inferred without a schema")
    func testPythonicParserUnschematizedScalars() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content = #"""
            <|tool_call_start|>[configure(count=5, ratio=1.5, enabled=True, disabled=False, fallback=None, mode=fast)]<|tool_call_end|>
            """#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.arguments["count"] == .int(5))
        #expect(toolCall.function.arguments["ratio"] == .double(1.5))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
        #expect(toolCall.function.arguments["disabled"] == .bool(false))
        #expect(toolCall.function.arguments["fallback"] == .null)
        #expect(toolCall.function.arguments["mode"] == .string("fast"))
    }

    @Test("Pythonic scalar inference does not override a declared string schema")
    func testPythonicParserDeclaredStringsRemainStrings() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let stringProperties: [String: any Sendable] = [
            "count": ["type": "string"] as [String: any Sendable],
            "enabled": ["type": "string"] as [String: any Sendable],
            "fallback": ["type": "string"] as [String: any Sendable],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "configure",
                    "parameters": ["properties": stringProperties]
                        as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        let content = #"""
            <|tool_call_start|>[configure(count=5, enabled=True, fallback=None)]<|tool_call_end|>
            """#

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.arguments["count"] == .string("5"))
        #expect(toolCall.function.arguments["enabled"] == .string("True"))
        #expect(toolCall.function.arguments["fallback"] == .string("None"))
    }

    @Test("Test Pythonic Tool Call Parser - Object Value Is Not Unwrapped Alongside Another")
    func testPythonicParserWrapperGuardWithSecondArgument() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        // A wrapper name is only a wrapper when it is the *sole* argument.
        let content =
            #"<|tool_call_start|>[get_weather(properties={"location": "Paris"}, units='c')]<|tool_call_end|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.arguments.count == 2)
        #expect(
            toolCall.function.arguments["properties"] == .object(["location": .string("Paris")]))
        #expect(toolCall.function.arguments["units"] == .string("c"))
        #expect(toolCall.function.arguments["location"] == nil)
    }

    @Test("Test Pythonic Tool Call Parser - Wrapper Object via parseEOS")
    func testPythonicParserWrapperObjectViaEOS() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            #"<|tool_call_start|>[get_weather(properties={"location": "Paris"}), current_time(properties={"timezone": "UTC"})]<|tool_call_end|>"#

        let toolCalls = parser.parseEOS(content, tools: nil)

        #expect(toolCalls.count == 2)
        #expect(toolCalls.first?.function.name == "get_weather")
        #expect(toolCalls.first?.function.arguments["location"] == .string("Paris"))
        #expect(toolCalls.last?.function.name == "current_time")
        #expect(toolCalls.last?.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test Pythonic Tool Call Parser - Unbalanced Bracket Is Not A Call")
    func testPythonicParserUnbalancedBracket() {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")

        // A truncated list is incomplete output, not a call that is ready to run.
        #expect(
            parser.parse(
                content: "<|tool_call_start|>[notify(note='hi')<|tool_call_end|>", tools: nil)
                == nil)
    }

    @Test("Test Pythonic Tool Call Parser - Type Conversion")
    func testPythonicParserTypeConversion() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<|tool_call_start|>[set_temperature(value='25', enabled='true')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test LFM2 Format via ToolCallProcessor - Pythonic")
    func testLFM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .lfm2)
        let content =
            "<|tool_call_start|>[calculator(expression='2+2')]<|tool_call_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - XML Function Format Tests (Qwen3 Coder)

    @Test("Test XML Function Parser - Qwen3 Coder Format")
    func testXMLFunctionParser() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<function=get_weather><parameter=location>Tokyo</parameter><parameter=unit>celsius</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test XML Function Parser - With Type Conversion")
    func testXMLFunctionParserTypeConversion() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<function=set_temperature><parameter=value>25</parameter><parameter=enabled>true</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test XML Function Parser - Multiline Content (Qwen3.5 style)")
    func testXMLFunctionParserMultiline() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        // Qwen3.5 models generate newlines between the XML tags
        let content = """
            <tool_call>
            <function=get_current_datetime>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test XML Function Parser - Multiline Parameters")
    func testXMLFunctionParserMultilineParams() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <function=get_weather>
            <parameter=location>
            Tokyo
            </parameter>
            </function>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    // MARK: - Qwen3.5 Format Tests (XML Function with tool_call wrapper)

    @Test("Test Qwen3.5 XML Function Parser - With tool_call Tags")
    func testQwen35Parser() throws {
        let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <tool_call>
            <function=get_weather>
            <parameter=location>
            San Francisco
            </parameter>
            <parameter=unit>
            celsius
            </parameter>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Qwen3.5 Format via ToolCallProcessor")
    func testQwen35FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .qwen35)
        let chunks: [String] = [
            "<tool", "_call>", "\n<function=get_weather>\n",
            "<parameter=location>\nTokyo\n</parameter>",
            "\n</function>\n</tool_call>",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Qwen3.5 Format - No Arguments")
    func testQwen35FormatNoArgs() throws {
        let processor = ToolCallProcessor(format: .qwen35)
        let content = "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Qwen3.5 accepts framed Qwen/Hermes JSON fallback")
    func testQwen35JSONFallback() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.qwen35Tools)
        let content =
            #"<tool_call>{"name":"lampo_mcp_call_tool","arguments":{"action":"start","instructions":"Initialize interactive coding session","title":"filo server start","worker":"default"}}</tool_call>"#

        #expect(processor.processChunk(content) == nil)
        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(call.function.name == "lampo_mcp_call_tool")
        #expect(call.function.arguments["action"] == .string("start"))
        #expect(call.function.arguments["worker"] == .string("default"))
    }

    @Test("Qwen3.5 JSON fallback is streaming-boundary invariant")
    func testQwen35JSONFallbackEverySplitBoundary() throws {
        let content =
            #"<tool_call>{"name":"lampo_mcp_call_tool","arguments":{"message":"a } brace, a quote: \"ok\", and <function=fake>"}}</tool_call>"#
        let characters = Array(content)

        for split in 1 ..< characters.count {
            let processor = ToolCallProcessor(format: .qwen35, tools: Self.qwen35Tools)
            let first = String(characters[..<split])
            let second = String(characters[split...])

            #expect(processor.processChunk(first) == nil, "split at \(split)")
            #expect(processor.processChunk(second) == nil, "split at \(split)")
            #expect(processor.toolCalls.count == 1, "split at \(split)")
            #expect(
                processor.toolCalls.first?.function.name == "lampo_mcp_call_tool",
                "split at \(split)")
        }
    }

    @Test("Qwen3.5 commits a complete JSON body at EOS without a closing frame")
    func testQwen35JSONFallbackAtEOS() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.qwen35Tools)
        let content =
            #"<tool_call>{"name":"lampo_mcp_call_tool","arguments":{"action":"start"}}"#

        #expect(processor.processChunk(content) == nil)
        #expect(processor.toolCalls.isEmpty)
        processor.processEOS()

        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(call.function.name == "lampo_mcp_call_tool")
        #expect(call.function.arguments["action"] == .string("start"))
    }

    @Test("Qwen3.5 applies the declared-tool boundary to canonical XML too")
    func testQwen35XMLRejectsUndeclaredTool() {
        let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<tool_call><function=erase_everything></function></tool_call>"

        #expect(parser.parse(content: content, tools: Self.qwen35Tools) == nil)
    }

    @Test("Qwen3.5 fallback never authorizes undeclared JSON tools")
    func testQwen35JSONFallbackRejectsUndeclaredTool() {
        let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = #"<tool_call>{"name":"erase_everything","arguments":{}}</tool_call>"#

        #expect(parser.parse(content: content, tools: Self.qwen35Tools) == nil)
    }

    @Test("Qwen3.5 fallback rejects malformed and mixed-dialect payloads")
    func testQwen35JSONFallbackRejectsMalformedPayloads() {
        let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let malformed = [
            #"<tool_call>{"name":"lampo_mcp_call_tool","arguments":{}</tool_call>"#,
            #"<tool_call>prefix {"name":"lampo_mcp_call_tool","arguments":{}}</tool_call>"#,
            #"<tool_call><function=lampo_mcp_call_tool>{"arguments":{}}</function></tool_call>"#,
            "<tool_call><function=lampo_mcp_call_tool><parameter=action>start</parameter>garbage</function></tool_call>",
            #"<tool_call>{"name":"lampo_mcp_call_tool","arguments":{}}</tool_call>trailing"#,
        ]

        for content in malformed {
            #expect(parser.parse(content: content, tools: Self.qwen35Tools) == nil)
        }
    }

    private static let qwen35Tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "lampo_mcp_call_tool",
                "parameters": ["type": "object"],
            ] as [String: any Sendable],
        ]
    ]

    // MARK: - GLM4 Format Tests

    @Test("Test GLM4 Tool Call Parser")
    func testGLM4Parser() throws {
        let parser = GLM4ToolCallParser()
        let content =
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test GLM4 Format via ToolCallProcessor")
    func testGLM4FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .glm4)
        let content =
            "<tool_call>search<arg_key>query</arg_key><arg_value>machine learning</arg_value></tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("machine learning"))
    }

    // MARK: - Gemma Format Tests

    @Test("Test Gemma Function Parser")
    func testGemmaParser() throws {
        let parser = GemmaFunctionParser(
            startTag: "<start_function_call>", endTag: "<end_function_call>",
            escapeMarker: "<escape>")
        let content =
            "<start_function_call>call:get_weather{location:Paris,unit:celsius}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Gemma Function Parser - Escaped Strings")
    func testGemmaParserEscapedStrings() throws {
        let parser = GemmaFunctionParser(
            startTag: "<start_function_call>", endTag: "<end_function_call>",
            escapeMarker: "<escape>")
        // Note: Gemma uses <escape> for both start and end markers (not </escape>)
        let content =
            "<start_function_call>call:search{query:<escape>hello, world!<escape>}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("hello, world!"))
    }

    @Test("Test Gemma 4 Function Parser - Type Conversion")
    func testGemma4ParserTypeConversion() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "mail_read",
                    "parameters": [
                        "properties": [
                            "account": ["type": "string"],
                            "mailbox": ["type": "string"],
                            "id": ["type": "integer"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            #"<|tool_call>call:mail_read{account:<|"|>me@example.com<|"|>,mailbox:<|"|>INBOX<|"|>,id:<|"|>158348<|"|>}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "mail_read")
        #expect(toolCall.function.arguments["account"] == .string("me@example.com"))
        #expect(toolCall.function.arguments["mailbox"] == .string("INBOX"))
        #expect(toolCall.function.arguments["id"] == .int(158_348))
        #expect(toolCall.function.arguments["id"] != .string("158348"))
    }

    @Test("Gemma keeps a nested object value whole")
    func testGemmaNestedObjectValue() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content = #"<|tool_call>call:search{filters:{"city":"Paris","limit":1}}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        // The comma inside the object must not split the value, and the tail of
        // a split value must not become an argument of its own.
        #expect(toolCall.function.arguments.count == 1)
        #expect(
            toolCall.function.arguments["filters"]
                == .object(["city": .string("Paris"), "limit": .int(1)]))
    }

    @Test("Gemma keeps an array value whole")
    func testGemmaArrayValue() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content = #"<|tool_call>call:notify{ids:[1,2,3],urgent:true}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.arguments.count == 2)
        // The array survives its inner commas, and `1` stays an integer rather
        // than being rewritten as a boolean on the way into `JSONValue`.
        #expect(toolCall.function.arguments["ids"] == .array([.int(1), .int(2), .int(3)]))
        // Bare scalars are typed from the tool schema; without one they stay literal.
        #expect(toolCall.function.arguments["urgent"] == .string("true"))
    }

    @Test("Gemma reads a nested object whose keys are unquoted")
    func testGemmaBareKeyObjectValue() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content = #"<|tool_call>call:search{filters:{city: "Paris", limit: 1}}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        // The dialect writes nested objects without quoting their keys, which strict JSON refuses.
        #expect(
            toolCall.function.arguments["filters"]
                == .object(["city": .string("Paris"), "limit": .int(1)]))
    }

    @Test("Gemma reads a bare-key object into a parameter the schema declares an object")
    func testGemmaBareKeyObjectValueWithSchema() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "search",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "filters": ["type": "object"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        let content = #"<|tool_call>call:search{filters:{city: "Paris"}}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        // Without the brace-form read the schema-typed conversion hands back the raw text, and the
        // tool receives a string where it declared an object.
        #expect(toolCall.function.arguments["filters"] == .object(["city": .string("Paris")]))
    }

    @Test("Gemma refuses to quote bare object values")
    func testGemmaBareValueStaysLiteral() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content = #"<|tool_call>call:search{filters:{city: Paris}}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        // Quoting a bare value would invent meaning, so the literal text is kept instead.
        #expect(toolCall.function.arguments["filters"] == .string("{city: Paris}"))
    }

    @Test("Gemma keeps an escaped value that resembles an object as a string")
    func testGemmaEscapedObjectShapedValueStaysString() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content = #"<|tool_call>call:notify{note:<|"|>{city: "Paris"}<|"|>}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.arguments["note"] == .string(#"{city: "Paris"}"#))
    }

    @Test("Gemma escaped values may contain protocol punctuation")
    func testGemmaEscapedValuePunctuation() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)
        let content =
            #"<|tool_call>call:notify{note:<|"|>a, b} and {c<|"|>,seen:false}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.arguments.count == 2)
        #expect(toolCall.function.arguments["note"] == .string("a, b} and {c"))
        #expect(toolCall.function.arguments["seen"] == .string("false"))
    }

    @Test("Test Gemma 4 Format via ToolCallProcessor")
    func testGemma4FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gemma4)
        let content = #"<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>"#

        // Delivered one character at a time: the streaming path has to reassemble
        // the asymmetric Gemma 4 tags before the parser ever sees them.
        var visible = ""
        for character in content {
            if let text = processor.processChunk(String(character)) { visible += text }
        }
        processor.processEOS()

        #expect(visible.isEmpty)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["city"] == .string("Tokyo"))
    }

    @Test("Test Gemma Format via ToolCallProcessor")
    func testGemmaFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gemma)
        let content = "<start_function_call>call:calculator{expression:2+2}<end_function_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - Kimi K2 Format Tests

    @Test("Test Kimi K2 Tool Call Parser")
    func testKimiK2Parser() throws {
        let parser = KimiK2ToolCallParser()
        let content =
            "<|tool_calls_section_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"London\"}<|tool_calls_section_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("London"))
    }

    @Test("Test Kimi K2 Format via ToolCallProcessor")
    func testKimiK2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .kimiK2)
        let content =
            "<|tool_calls_section_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"swift\"}<|tool_calls_section_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift"))
    }

    // MARK: - MiniMax M2 Format Tests

    @Test("Test MiniMax M2 Tool Call Parser")
    func testMiniMaxM2Parser() throws {
        let parser = MiniMaxM2ToolCallParser()
        let content =
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Sydney</parameter></invoke></minimax:tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Sydney"))
    }

    @Test("Test MiniMax M2 Format via ToolCallProcessor")
    func testMiniMaxM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let content =
            "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">AI news</parameter></invoke></minimax:tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("AI news"))
    }

    // MARK: - Llama 3 Format Tests

    @Test("Test Llama 3 Tool Call Parser")
    func testLlama3Parser() throws {
        let parser = Llama3ToolCallParser()

        let content1 = """
            <|python_tag|>{"name": "knowledge_search", "parameters": {"query": "example"}}
            """

        let toolCall1 = try #require(parser.parse(content: content1, tools: nil))
        #expect(toolCall1.function.name == "knowledge_search")
        #expect(toolCall1.function.arguments["query"] == .string("example"))

        let content2 = """
            {"name": "get_weather", "arguments": {"location": "Tokyo"}}
            """

        let toolCall2 = try #require(parser.parse(content: content2, tools: nil))
        #expect(toolCall2.function.name == "get_weather")
        #expect(toolCall2.function.arguments["location"] == .string("Tokyo"))

        // Pythonic format
        let content3 = """
            <|python_tag|>get_weather(location="San Francisco, CA")
            """

        let toolCall3 = try #require(parser.parse(content: content3, tools: nil))
        #expect(toolCall3.function.name == "get_weather")
        #expect(toolCall3.function.arguments["location"] == .string("San Francisco, CA"))

        // Multiple arguments Pythonic
        let content4 = """
            <|python_tag|>calculate(expression="2 + 2", precision=4)
            """

        let toolCall4 = try #require(parser.parse(content: content4, tools: nil))
        #expect(toolCall4.function.name == "calculate")
        #expect(toolCall4.function.arguments["expression"] == .string("2 + 2"))
        #expect(toolCall4.function.arguments["precision"] == .int(4))

        // Multiple JSON list format via parseEOS
        let content5 = """
            <|python_tag|>[
              {"name": "get_weather", "parameters": {"location": "New York"}},
              {"name": "get_time", "parameters": {"location": "London"}}
            ]
            """
        let toolCalls5 = parser.parseEOS(content5, tools: nil)
        #expect(toolCalls5.count == 2)
        #expect(toolCalls5[0].function.name == "get_weather")
        #expect(toolCalls5[0].function.arguments["location"] == .string("New York"))
        #expect(toolCalls5[1].function.name == "get_time")
        #expect(toolCalls5[1].function.arguments["location"] == .string("London"))

        // Multiple pythonic format via parseEOS
        let content6 = """
            <|python_tag|>[get_weather(location="New York"), get_time(location="London")]
            """
        let toolCalls6 = parser.parseEOS(content6, tools: nil)
        #expect(toolCalls6.count == 2)
        #expect(toolCalls6[0].function.name == "get_weather")
        #expect(toolCalls6[0].function.arguments["location"] == .string("New York"))
        #expect(toolCalls6[1].function.name == "get_time")
        #expect(toolCalls6[1].function.arguments["location"] == .string("London"))
    }

    // MARK: - ToolCallFormat Serialization Tests

    @Test("Test ToolCallFormat Raw Values for Serialization")
    func testToolCallFormatRawValues() throws {
        // Test that raw values are suitable for JSON/CLI serialization
        #expect(ToolCallFormat.json.rawValue == "json")
        #expect(ToolCallFormat.lfm2.rawValue == "lfm2")
        #expect(ToolCallFormat.xmlFunction.rawValue == "xml_function")
        #expect(ToolCallFormat.qwen35.rawValue == "qwen3_5")
        #expect(ToolCallFormat.glm4.rawValue == "glm4")
        #expect(ToolCallFormat.gemma.rawValue == "gemma")
        #expect(ToolCallFormat.kimiK2.rawValue == "kimi_k2")
        #expect(ToolCallFormat.minimaxM2.rawValue == "minimax_m2")
        #expect(ToolCallFormat.atem.rawValue == "atem")
        #expect(ToolCallFormat.mistral.rawValue == "mistral")
        #expect(ToolCallFormat.gptOSS.rawValue == "gpt_oss")

        // Test round-trip via raw value
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat(rawValue: format.rawValue) == format)
        }
    }

    @Test("gptOSS createParser is a non-fatal compatibility fallback")
    func testGPTOSSCreateParserCompatibilityFallback() throws {
        // Alternate callers (e.g. MLXFoundationModels ToolCallProcessor paths)
        // must be able to construct a parser without trapping, even though
        // Harmony tool calling is not text-parser based.
        let parser = ToolCallFormat.gptOSS.createParser()
        #expect(parser.startTag == "<tool_call>")
        #expect(parser.endTag == "</tool_call>")
        // Processor construction must likewise be non-fatal.
        let processor = ToolCallProcessor(format: .gptOSS)
        #expect(processor.toolCalls.isEmpty)
    }

    // MARK: - Mistral Format Tests

    @Test("Test Mistral Tool Call Parser")
    func testMistralParser() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - With Call ID")
    func testMistralParserWithCallId() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather[CALL_ID]abc123xyz[ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.id == "abc123xyz")
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves [TOOL_CALLS] in Arguments")
    func testMistralParserPreservesStartTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal [TOOL_CALLS] marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal [TOOL_CALLS] marker"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves </s> in Arguments")
    func testMistralParserPreservesEndTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal </s> marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal </s> marker"))
    }

    @Test("Test Mistral Format via ToolCallProcessor")
    func testMistralFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL", "_CALLS]", "get_weather", " [ARGS]",
            "{\"location\":", " \"Tokyo\"}",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        // End tag never arrives in text, so tool call stays buffered until processEOS
        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Mistral Format Processor EOS")
    func testMistralFormatProcessorEOS() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Berlin\"}"

        _ = processor.processChunk(content)

        // Before processEOS, no tool calls extracted (end tag never arrives)
        #expect(processor.toolCalls.count == 0)

        // processEOS extracts the buffered tool call
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
    }

    @Test("Test Mistral Format Processor Multiple Tool Calls")
    func testMistralFormatProcessorMultipleToolCalls() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL_CALLS]get_weather[ARGS]",
            "{\"location\": \"Paris\"}",
            "[TOOL_CALLS]get_time",
            "[ARGS]{\"timezone\": \"UTC\"}",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            // All chunks should be buffered (nil) after the start tag
            if chunk == chunks.first {
                #expect(result == nil)
            }
        }

        // No tool calls before processEOS
        #expect(processor.toolCalls.count == 0)
        processor.processEOS()

        // Both tool calls should be extracted
        #expect(processor.toolCalls.count == 2)

        let first = try #require(processor.toolCalls.first)
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))

        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }
}
