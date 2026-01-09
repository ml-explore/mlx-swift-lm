// Copyright © 2025 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import MLXVLM
import Tokenizers
import XCTest

/// Tests for the streamlined API using real models
public class ChatSessionIntegrationTests: XCTestCase {

    static let llmModelId = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    static let vlmModelId = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    nonisolated(unsafe) static var llmContainer: ModelContainer!
    nonisolated(unsafe) static var vlmContainer: ModelContainer!

    override public class func setUp() {
        super.setUp()
        // Load models once for all tests
        let llmExpectation = XCTestExpectation(description: "Load LLM")
        let vlmExpectation = XCTestExpectation(description: "Load VLM")

        Task {
            llmContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: llmModelId)
            )
            llmExpectation.fulfill()
        }

        Task {
            vlmContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: .init(id: vlmModelId)
            )
            vlmExpectation.fulfill()
        }

        _ = XCTWaiter.wait(for: [llmExpectation, vlmExpectation], timeout: 300)
    }

    func testOneShot() async throws {
        let session = ChatSession(Self.llmContainer)
        let result = try await session.respond(to: "What is 2+2? Reply with just the number.")
        print("One-shot result:", result)
        XCTAssertTrue(result.contains("4") || result.lowercased().contains("four"))
    }

    func testOneShotStream() async throws {
        let session = ChatSession(Self.llmContainer)
        var result = ""
        for try await token in session.streamResponse(
            to: "What is 2+2? Reply with just the number.")
        {
            print(token, terminator: "")
            result += token
        }
        print()  // newline
        XCTAssertTrue(result.contains("4") || result.lowercased().contains("four"))
    }

    func testMultiTurnConversation() async throws {
        let session = ChatSession(
            Self.llmContainer, instructions: "You are a helpful assistant. Keep responses brief.")

        let response1 = try await session.respond(to: "My name is Alice.")
        print("Response 1:", response1)

        let response2 = try await session.respond(to: "What is my name?")
        print("Response 2:", response2)

        // If multi-turn works, response2 should mention "Alice"
        XCTAssertTrue(
            response2.lowercased().contains("alice"),
            "Model should remember the name 'Alice' from previous turn")
    }

    func testVisionModel() async throws {
        let session = ChatSession(Self.vlmContainer)

        // Create a simple red image for testing
        let redImage = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

        let result = try await session.respond(
            to: "What color is this image? Reply with just the color name.",
            image: .ciImage(redImage))
        print("Vision result:", result)
        XCTAssertTrue(result.lowercased().contains("red"))
    }

    func testPromptRehydration() async throws {
        // Simulate a persisted history (e.g. loaded from SwiftData)
        let history: [Chat.Message] = [
            .system("You are a helpful assistant."),
            .user("My name is Bob."),
            .assistant("Hello Bob! How can I help you today?"),
        ]

        let session = ChatSession(Self.llmContainer, history: history)

        // Ask a question that requires the context
        let response = try await session.respond(to: "What is my name?")

        print("Rehydration result:", response)

        XCTAssertTrue(
            response.lowercased().contains("bob"),
            "Model should recognize the name 'Bob' from the injected history, proving successful prompt re-hydration."
        )
    }
}
