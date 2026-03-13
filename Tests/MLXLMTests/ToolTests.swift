import Foundation
import MLXLMCommon
import Testing

struct ToolTests {
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
        let parser = XMLFunctionParser()
        let content =
            "<function=get_weather><parameter=location>Tokyo</parameter><parameter=unit>celsius</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test XML Function Parser - With Type Conversion")
    func testXMLFunctionParserTypeConversion() throws {
        let parser = XMLFunctionParser()
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
        let parser = XMLFunctionParser()
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
        let parser = XMLFunctionParser()
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
        let parser = GemmaFunctionParser()
        let content =
            "<start_function_call>call:get_weather{location:Paris,unit:celsius}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Gemma Function Parser - Escaped Strings")
    func testGemmaParserEscapedStrings() throws {
        let parser = GemmaFunctionParser()
        // Note: Gemma uses <escape> for both start and end markers (not </escape>)
        let content =
            "<start_function_call>call:search{query:<escape>hello, world!<escape>}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("hello, world!"))
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

    // MARK: - ToolCallFormat Serialization Tests

    @Test("Test ToolCallFormat Raw Values for Serialization")
    func testToolCallFormatRawValues() throws {
        // Test that raw values are suitable for JSON/CLI serialization
        #expect(ToolCallFormat.json.rawValue == "json")
        #expect(ToolCallFormat.lfm2.rawValue == "lfm2")
        #expect(ToolCallFormat.xmlFunction.rawValue == "xml_function")
        #expect(ToolCallFormat.glm4.rawValue == "glm4")
        #expect(ToolCallFormat.gemma.rawValue == "gemma")
        #expect(ToolCallFormat.kimiK2.rawValue == "kimi_k2")
        #expect(ToolCallFormat.minimaxM2.rawValue == "minimax_m2")
        #expect(ToolCallFormat.mistral.rawValue == "mistral")
        #expect(ToolCallFormat.gptOSS.rawValue == "gpt_oss")

        // Test round-trip via raw value
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat(rawValue: format.rawValue) == format)
        }
    }

    // MARK: - Format Inference Tests

    @Test("Test ToolCallFormat Inference from Model Type")
    func testToolCallFormatInference() throws {
        // LFM2 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm25") == .lfm2)

        // GLM4 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "glm4") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe_lite") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_5") == .glm4)
        #expect(ToolCallFormat.infer(from: "GLM4_5") == .glm4)

        // Gemma models
        #expect(ToolCallFormat.infer(from: "gemma") == .gemma)
        #expect(ToolCallFormat.infer(from: "GEMMA") == .gemma)

        // Mistral3 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "Mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)

        // GPT-OSS models
        #expect(ToolCallFormat.infer(from: "gpt_oss") == .gptOSS)
        #expect(ToolCallFormat.infer(from: "GPT_OSS") == .gptOSS)
        #expect(ToolCallFormat.infer(from: "gptoss") == .gptOSS)

        // Unknown models should return nil (use default JSON format)
        #expect(ToolCallFormat.infer(from: "llama") == nil)
        #expect(ToolCallFormat.infer(from: "qwen2") == nil)
        #expect(ToolCallFormat.infer(from: "mistral") == nil)
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

    // MARK: - GPT-OSS Harmony Format Tests

    @Test("Test GPT-OSS Harmony Tool Call Parser")
    func testGPTOSSHarmonyParser() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=get_weather<|message|>{\"location\": \"Tokyo\"}<|call|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - Multi Line")
    func testGPTOSSHarmonyParserMultiLine() throws {
        let parser = GPTOSSToolCallParser()
        let content = """
            <|channel|>commentary to=functions.get_weather
            <|message|>
            {"location": "San Francisco"}
            <|call|>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - Recipient in Role Header")
    func testGPTOSSHarmonyParserRoleHeaderRecipient() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|start|>assistant to=functions.get_weather<|channel|>commentary<|message|>{\"location\": \"Rome\"}<|call|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Rome"))
    }

    @Test("Test GPT-OSS Format via ToolCallProcessor")
    func testGPTOSSFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gptOSS)
        let chunks: [String] = [
            "<|chan", "nel|>com", "mentary to=search", "<|", "message|>",
            "{\"query\":", " \"swift\"}", "<|ca", "ll|>",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift"))
    }

    @Test("Test GPT-OSS Format Processor Multiple Tool Calls with EOS")
    func testGPTOSSFormatProcessorMultipleToolCalls() throws {
        let processor = ToolCallProcessor(format: .gptOSS)
        let content = """
            <|channel|>commentary to=get_weather<|message|>{"location": "Paris"}<|call|>
            <|channel|>commentary to=get_time<|message|>{"timezone": "UTC"}<|call|>
            """

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 2)

        let first = processor.toolCalls[0]
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))

        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test GPT-OSS Format Processor with Commentary Preamble")
    func testGPTOSSFormatProcessorWithCommentaryPreamble() throws {
        let processor = ToolCallProcessor(format: .gptOSS)
        let content = """
            <|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary<|message|>{"location":"Milan"}<|call|>
            """

        let passthrough = processor.processChunk(content)

        #expect(passthrough?.contains("Let me check.") == true)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Milan"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - With Constrain Tag")
    func testGPTOSSHarmonyParserWithConstrainTag() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Seattle\"}<|call|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Seattle"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - With Markdown Fences")
    func testGPTOSSHarmonyParserWithMarkdownFences() throws {
        let parser = GPTOSSToolCallParser()
        let content = """
            <|channel|>commentary to=get_weather
            <|message|>
            ```json
            {"location": "Boston"}
            ```
            <|call|>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Boston"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - With Nested Arguments Object")
    func testGPTOSSHarmonyParserWithNestedArguments() throws {
        let parser = GPTOSSToolCallParser()
        let content = """
            <|channel|>commentary to=get_weather
            <|message|>
            {"arguments": {"location": "Austin"}}
            <|call|>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Austin"))
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - Invalid Channel")
    func testGPTOSSHarmonyParserInvalidChannel() throws {
        let parser = GPTOSSToolCallParser()
        // Tools should only be parsed from the 'commentary' channel, not 'analysis' or 'final'
        let content =
            "<|channel|>analysis to=get_weather<|message|>{\"location\": \"Tokyo\"}<|call|>"

        let toolCall = parser.parse(content: content, tools: nil)
        #expect(toolCall == nil)
    }

    @Test("Test GPT-OSS Harmony Tool Call Parser - Invalid JSON")
    func testGPTOSSHarmonyParserInvalidJSON() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|channel|>commentary to=get_weather<|message|>{\"location\":<|call|>"

        let toolCall = parser.parse(content: content, tools: nil)
        #expect(toolCall == nil)
    }

    @Test("Test GPT-OSS Format Processor - Split Assistant+Channel Header Parsed at EOS")
    func testGPTOSSFormatProcessorSplitAssistantChannelHeaderAtEOS() throws {
        let processor = ToolCallProcessor(format: .gptOSS)
        let chunks: [String] = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "comment",
            "ary",
            " to",
            "=",
            "functions",
            ".web",
            "_search",
            " ",
            "<|constrain|>",
            "json",
            "<|message|>",
            "{\"query\":\"latest news Dubai\"}",
        ]

        for chunk in chunks {
            let passthrough = processor.processChunk(chunk)
            #expect(passthrough == nil)
        }

        #expect(processor.toolCalls.isEmpty)
        processor.processEOS()
        #expect(processor.toolCalls.count == 1)

        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "web_search")
        #expect(call.function.arguments["query"] == .string("latest news Dubai"))
    }

    @Test("Test GPT-OSS Harmony Parser EOS - Multiple Calls Separated by Channel Boundary")
    func testGPTOSSHarmonyParserEOSMultipleCallsChannelBoundary() throws {
        let parser = GPTOSSToolCallParser()
        // Two tool calls separated only by <|channel|>, no <|call|> end tokens
        let content =
            "<|channel|>commentary to=get_weather<|message|>{\"location\":\"Paris\"}<|channel|>commentary to=get_time<|message|>{\"timezone\":\"UTC\"}"

        let calls = parser.parseEOS(content, tools: nil)

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"] == .string("Paris"))
        #expect(calls[1].function.name == "get_time")
        #expect(calls[1].function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test GPT-OSS Harmony Parser EOS - Role Header without <|call|>")
    func testGPTOSSHarmonyParserEOSRoleHeaderWithoutCallToken() throws {
        let parser = GPTOSSToolCallParser()
        let content =
            "<|start|>assistant to=functions.get_weather<|channel|>commentary<|message|>{\"location\":\"Naples\"}"

        let calls = parser.parseEOS(content, tools: nil)

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"] == .string("Naples"))
    }
}
