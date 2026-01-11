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
public class ChatSessionTests: XCTestCase {

    static let llmModelId = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    static let vlmModelId = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    static var llmContainer: ModelContainer!
    static var vlmContainer: ModelContainer!

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

    func testStopGeneration() async throws {
        let session = ChatSession(Self.llmContainer)

        var collectedOutput = ""
        let prompt =
            "Write a detailed 1000 word essay about the history of the Roman Empire. Please include dates and specific names."

        let generationTask = Task {
            do {
                for try await chunk in session.streamResponse(to: prompt) {
                    collectedOutput += chunk
                    print(chunk, terminator: "")
                }
            } catch {
                // Cancellation is expected
                if !(error is CancellationError) {
                    XCTFail("Unexpected error: \(error)")
                }
            }
            return collectedOutput
        }

        try await Task.sleep(for: .seconds(1))

        session.cancel()

        let result = await generationTask.value

        XCTAssertFalse(
            result.isEmpty,
            "The model should have generated some text before stopping."
        )

        XCTAssertTrue(
            result.count < 2000,
            "The generation should have stopped early. Got \(result.count) characters."
        )

        let recoveryResponse = try await session.respond(to: "Say 'ready' if you can hear me.")
        print("Recovery response:", recoveryResponse)
        XCTAssertFalse(
            recoveryResponse.isEmpty,
            "The session should function correctly after stopping."
        )
    }

    func testStopRespondNonStreaming() async throws {
        let session = ChatSession(Self.llmContainer)

        let generationTask = Task {
            try await session.respond(
                to: "Write a 500 word essay about quantum physics. Include mathematical formulas."
            )
        }

        // Allow some generation
        try await Task.sleep(for: .seconds(1))

        // Stop it
        session.cancel()

        // Should complete quickly
        let result = try await generationTask.value
        XCTAssertTrue(
            result.count < 2000,
            "Generation should have stopped early"
        )
    }

    /// Test rapid stop/start cycles don't cause issues.
    func testRapidStopStart() async throws {
        let session = ChatSession(Self.llmContainer)

        for i in 1 ... 3 {
            print("Cycle \(i)...")

            let task = Task {
                var output = ""
                for try await chunk in session.streamResponse(to: "Count from 1 to 100.") {
                    output += chunk
                }
                return output
            }

            try await Task.sleep(for: .milliseconds(500))
            session.cancel()

            _ = try await task.value
            try await Task.sleep(for: .milliseconds(100))
        }

        let finalResponse = try await session.respond(to: "Say 'ok'")
        XCTAssertFalse(finalResponse.isEmpty)
    }
}
