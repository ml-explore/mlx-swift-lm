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

    @Test("Test JSON Tool Call Parser - LFM2 Tags")
    func testJSONParserLFM2Tags() throws {
        let parser = JSONToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>{\"name\": \"search\", \"arguments\": {\"query\": \"swift programming\"}}<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test LFM2 Format via ToolCallProcessor")
    func testLFM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .lfm2)
        let content =
            "<|tool_call_start|>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}<|tool_call_end|>"

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

        // Test round-trip via raw value
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat(rawValue: format.rawValue) == format)
        }
    }

    // MARK: - Format Inference Tests

    @Test("Test ToolCallFormat Inference from Model Type")
    func testToolCallFormatInference() throws {
        // LFM2 models
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)

        // GLM4 models
        #expect(ToolCallFormat.infer(from: "glm4") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe_lite") == .glm4)

        // Gemma models
        #expect(ToolCallFormat.infer(from: "gemma") == .gemma)
        #expect(ToolCallFormat.infer(from: "GEMMA") == .gemma)

        // Unknown models should return nil (use default)
        #expect(ToolCallFormat.infer(from: "llama") == nil)
        #expect(ToolCallFormat.infer(from: "qwen2") == nil)
        #expect(ToolCallFormat.infer(from: "mistral") == nil)
    }
}
