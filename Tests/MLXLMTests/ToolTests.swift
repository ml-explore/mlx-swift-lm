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
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
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
        let processor = ToolCallProcessor(format: .xmlFunction)
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
        let processor = ToolCallProcessor(format: .xmlFunction)
        let content = "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
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
        #expect(toolCall4.function.arguments["precision"] == .string("4"))

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
        #expect(ToolCallFormat.glm4.rawValue == "glm4")
        #expect(ToolCallFormat.gemma.rawValue == "gemma")
        #expect(ToolCallFormat.kimiK2.rawValue == "kimi_k2")
        #expect(ToolCallFormat.minimaxM2.rawValue == "minimax_m2")
        #expect(ToolCallFormat.mistral.rawValue == "mistral")

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

        // Nemotron models (prefix matching)
        #expect(ToolCallFormat.infer(from: "nemotron_h") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "NEMOTRON_H") == .xmlFunction)

        // Qwen3.5 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "qwen3_5") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_5_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "QWEN3_5") == .xmlFunction)

        // Qwen3-Next models (prefix matching)
        #expect(ToolCallFormat.infer(from: "qwen3_next") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_next_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "QWEN3_NEXT") == .xmlFunction)

        // Mistral3 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "Mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)

        // Llama models - require secondary signals from configData
        #expect(ToolCallFormat.infer(from: "llama") == nil)  // Should be nil without configData

        let llama3RopeConfig = """
            {
                "model_type": "llama",
                "rope_scaling": {
                    "rope_type": "llama3"
                }
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: llama3RopeConfig) == .llama3)

        let llama3VocabConfig = """
            {
                "model_type": "llama",
                "vocab_size": 128256
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "LLAMA", configData: llama3VocabConfig) == .llama3)

        let llama2Config = """
            {
                "model_type": "llama",
                "vocab_size": 32000
            }
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: llama2Config) == nil)

        // Unknown models should return nil (use default JSON format)
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
}
