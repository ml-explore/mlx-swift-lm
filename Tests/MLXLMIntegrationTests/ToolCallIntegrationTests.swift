// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
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

    // MARK: - Model IDs

    static let lfm2ModelId = "mlx-community/LFM2-2.6B-Exp-4bit"
    static let glm4ModelId = "mlx-community/GLM-4-9B-0414-4bit"

    // MARK: - Shared State

    nonisolated(unsafe) static var lfm2Container: ModelContainer?
    nonisolated(unsafe) static var glm4Container: ModelContainer?

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

    // MARK: - Setup

    override public class func setUp() {
        super.setUp()

        let lfm2Expectation = XCTestExpectation(description: "Load LFM2")
        let glm4Expectation = XCTestExpectation(description: "Load GLM4")

        Task {
            do {
                lfm2Container = try await LLMModelFactory.shared.loadContainer(
                    configuration: .init(id: lfm2ModelId)
                )
            } catch {
                print("Failed to load LFM2: \(error)")
            }
            lfm2Expectation.fulfill()
        }

        Task {
            do {
                glm4Container = try await LLMModelFactory.shared.loadContainer(
                    configuration: .init(id: glm4ModelId)
                )
            } catch {
                print("Failed to load GLM4: \(error)")
            }
            glm4Expectation.fulfill()
        }

        _ = XCTWaiter.wait(for: [lfm2Expectation, glm4Expectation], timeout: 600)
    }

    // MARK: - LFM2 Tests

    func testLFM2ToolCallFormatAutoDetection() async throws {
        guard let container = Self.lfm2Container else {
            throw XCTSkip("LFM2 model not available")
        }

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .lfm2,
            "LFM2 model should auto-detect .lfm2 tool call format"
        )
    }

    func testLFM2EndToEndToolCallGeneration() async throws {
        guard let container = Self.lfm2Container else {
            throw XCTSkip("LFM2 model not available")
        }

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
        guard let container = Self.glm4Container else {
            throw XCTSkip("GLM4 model not available")
        }

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .glm4,
            "GLM4 model should auto-detect .glm4 tool call format"
        )
    }

    func testGLM4EndToEndToolCallGeneration() async throws {
        guard let container = Self.glm4Container else {
            throw XCTSkip("GLM4 model not available")
        }

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

    // MARK: - Helper Methods

    /// Generate text and collect any tool calls
    private func generateWithTools(
        container: ModelContainer,
        input: UserInput,
        maxTokens: Int
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            let parameters = GenerateParameters(maxTokens: maxTokens)

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

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
