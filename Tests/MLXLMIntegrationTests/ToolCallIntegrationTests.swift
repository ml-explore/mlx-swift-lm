// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

/// Integration tests for tool call format auto-detection and end-to-end parsing.
///
/// These tests verify that:
/// 1. Tool call formats are correctly auto-detected from model_type
/// 2. Tool calls are correctly parsed from actual model generation output
///
/// References:
/// - LFM2: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
/// - GLM4: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public class ToolCallIntegrationTests: XCTestCase {

    // MARK: - Tool Schema

    static let weatherToolSchema: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        ] as [String: any Sendable],
                        "unit": [
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    // MARK: - Model Loading

    private var lfm2Container: ModelContainer {
        get async throws {
            do {
                return try await IntegrationTestModels.shared.lfm2Container()
            } catch {
                throw XCTSkip("LFM2 model not available: \(error)")
            }
        }
    }

    private var glm4Container: ModelContainer {
        get async throws {
            do {
                return try await IntegrationTestModels.shared.glm4Container()
            } catch {
                throw XCTSkip("GLM4 model not available: \(error)")
            }
        }
    }

    private var mistral3Container: ModelContainer {
        get async throws {
            do {
                return try await IntegrationTestModels.shared.mistral3Container()
            } catch {
                throw XCTSkip("Mistral3 model not available: \(error)")
            }
        }
    }

    private var nemotronContainer: ModelContainer {
        get async throws {
            try XCTSkipIf(true, "Nemotron model is opt-in only because of its size")

            do {
                return try await IntegrationTestModels.shared.nemotronContainer()
            } catch {
                throw XCTSkip("Nemotron model not available: \(error)")
            }
        }
    }

    private var qwen35Container: ModelContainer {
        get async throws {
            do {
                return try await IntegrationTestModels.shared.qwen35Container()
            } catch {
                throw XCTSkip("Qwen3.5 model not available: \(error)")
            }
        }
    }

    // MARK: - LFM2 Tests

    func testLFM2ToolCallFormatAutoDetection() async throws {
        let config = try await lfm2Container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .lfm2,
            "LFM2 model should auto-detect .lfm2 tool call format"
        )
    }

    func testLFM2EndToEndToolCallGeneration() async throws {
        let container = try await lfm2Container

        // Create input with tool schema
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema
        )

        // Generate with tools
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("LFM2 Output: \(result)")
        print("LFM2 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            // Location should contain something related to Tokyo
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    // MARK: - GLM4 Tests

    func testGLM4ToolCallFormatAutoDetection() async throws {
        let config = try await glm4Container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .glm4,
            "GLM4 model should auto-detect .glm4 tool call format"
        )
    }

    func testGLM4EndToEndToolCallGeneration() async throws {
        let container = try await glm4Container

        // Create input with tool schema
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Paris?"),
            ],
            tools: Self.weatherToolSchema
        )

        // Generate with tools
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("GLM4 Output: \(result)")
        print("GLM4 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            // Location should contain something related to Paris
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("paris"),
                    "Expected location to contain 'Paris', got: \(location)"
                )
            }
        }
    }

    // MARK: - Mistral3 Tests

    func testMistral3ToolCallFormatAutoDetection() async throws {
        let config = try await mistral3Container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .mistral,
            "Mistral3 model should auto-detect .mistral tool call format"
        )
    }

    func testMistral3EndToEndToolCallGeneration() async throws {
        let container = try await mistral3Container

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("Mistral3 Output: \(result)")
        print("Mistral3 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    func testMistral3MultipleToolCallGeneration() async throws {
        let container = try await mistral3Container

        let multiToolSchema: [[String: any Sendable]] =
            Self.weatherToolSchema + [
                [
                    "type": "function",
                    "function": [
                        "name": "get_time",
                        "description": "Get the current time in a given timezone",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "timezone": [
                                    "type": "string",
                                    "description":
                                        "The timezone, e.g. America/New_York, Asia/Tokyo",
                                ] as [String: any Sendable]
                            ] as [String: any Sendable],
                            "required": ["timezone"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user(
                    "What's the weather in Tokyo and what time is it there?"
                ),
            ],
            tools: multiToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 150
        )

        print("Mistral3 Output: \(result)")
        print("Mistral3 Calls: \(toolCalls)")

        // Verify all returned tool calls have valid names from our schema
        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            XCTAssertTrue(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        // If the model made multiple calls, verify we got more than one
        if toolCalls.count > 1 {
            print("Successfully parsed \(toolCalls.count) tool calls from Mistral3")
        }
    }

    // MARK: - Nemotron Tests

    func testNemotronToolCallFormatAutoDetection() async throws {
        let config = try await nemotronContainer.configuration
        XCTAssertEqual(
            config.toolCallFormat, .xmlFunction,
            "Nemotron model should auto-detect .xmlFunction tool call format"
        )
    }

    func testNemotronEndToEndToolCallGeneration() async throws {
        let container = try await nemotronContainer

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema,
            additionalContext: ["enable_thinking": false]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 150
        )

        print("Nemotron Output: \(result)")
        print("Nemotron Tool Calls: \(toolCalls)")

        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    func testNemotronMultipleToolCallGeneration() async throws {
        let container = try await nemotronContainer

        let multiToolSchema: [[String: any Sendable]] =
            Self.weatherToolSchema + [
                [
                    "type": "function",
                    "function": [
                        "name": "get_time",
                        "description": "Get the current time in a given timezone",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "timezone": [
                                    "type": "string",
                                    "description":
                                        "The timezone, e.g. America/New_York, Asia/Tokyo",
                                ] as [String: any Sendable]
                            ] as [String: any Sendable],
                            "required": ["timezone"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user(
                    "What's the weather in Tokyo and what time is it there?"
                ),
            ],
            tools: multiToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 600
        )

        print("Nemotron Output: \(result)")
        print("Nemotron Calls: \(toolCalls)")

        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            XCTAssertTrue(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        if toolCalls.count > 1 {
            print("Successfully parsed \(toolCalls.count) tool calls from Nemotron")
        }
    }

    // MARK: - Qwen3.5 Tests

    func testQwen35ToolCallFormatAutoDetection() async throws {
        let config = try await qwen35Container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .xmlFunction,
            "Qwen3.5 model should auto-detect .xmlFunction tool call format"
        )
    }

    func testQwen35EndToEndToolCallGeneration() async throws {
        let container = try await qwen35Container

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 150
        )

        print("Qwen3.5 Output: \(result)")
        print("Qwen3.5 Tool Calls: \(toolCalls)")

        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    func testQwen35MultipleToolCallGeneration() async throws {
        let container = try await qwen35Container

        let multiToolSchema: [[String: any Sendable]] =
            Self.weatherToolSchema + [
                [
                    "type": "function",
                    "function": [
                        "name": "get_time",
                        "description": "Get the current time in a given timezone",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "timezone": [
                                    "type": "string",
                                    "description":
                                        "The timezone, e.g. America/New_York, Asia/Tokyo",
                                ] as [String: any Sendable]
                            ] as [String: any Sendable],
                            "required": ["timezone"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user(
                    "What's the weather in Tokyo and what time is it there?"
                ),
            ],
            tools: multiToolSchema,
            additionalContext: ["enable_thinking": true]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 300
        )

        print("Qwen3.5 Output: \(result)")
        print("Qwen3.5 Calls: \(toolCalls)")

        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            XCTAssertTrue(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        if toolCalls.count > 1 {
            print("Successfully parsed \(toolCalls.count) tool calls from Qwen3.5")
        }
    }

    // MARK: - Helper Methods

    /// Generate text and collect any tool calls
    private func generateWithTools(
        container: ModelContainer,
        input: UserInput,
        maxTokens: Int
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        let result = try await container.perform(nonSendable: input) {
            (context: ModelContext, input) in
            let lmInput = try await context.processor.prepare(input: input)
            let parameters = GenerateParameters(maxTokens: maxTokens)

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            var collectedText = ""
            var collectedToolCalls: [ToolCall] = []

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info:
                    break
                }
            }

            return (collectedText, collectedToolCalls)
        }

        return result
    }
}

// MARK: - JSONValue Extension for Testing

extension JSONValue {
    var asString: String? {
        if case .string(let s) = self {
            return s
        }
        return nil
    }
}
