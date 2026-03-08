// Shared integration test logic for verifying end-to-end model loading and generation.
// Integration packages inject their own Downloader and TokenizerLoader, then call
// these functions which run the test and throw on failure.

import CoreImage
import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

// Both MLXLMCommon and MLXEmbedders define ModelContainer.
public typealias LMModelContainer = MLXLMCommon.ModelContainer
public typealias EmbeddingModelContainer = MLXEmbedders.ModelContainer

// MARK: - Error

public struct IntegrationTestFailure: LocalizedError {
    public let errorDescription: String?

    public init(_ message: String) {
        self.errorDescription = message
    }
}

private func check(_ condition: Bool, _ message: String) throws {
    guard condition else { throw IntegrationTestFailure(message) }
}

// MARK: - Model IDs

public enum IntegrationTestModelIDs {
    public static let llm = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    public static let vlm = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    public static let lfm2 = "mlx-community/LFM2-2.6B-Exp-4bit"
    public static let glm4 = "mlx-community/GLM-4-9B-0414-4bit"
}

// MARK: - Model Loading

/// Shared model cache that loads each model at most once per test run.
public actor IntegrationTestModels {
    private let downloader: any Downloader
    private let tokenizerLoader: any TokenizerLoader

    private var llmTask: Task<LMModelContainer, Error>?
    private var vlmTask: Task<LMModelContainer, Error>?
    private var lfm2Task: Task<LMModelContainer, Error>?
    private var glm4Task: Task<LMModelContainer, Error>?

    public init(downloader: any Downloader, tokenizerLoader: any TokenizerLoader) {
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
    }

    public func llmContainer() async throws -> LMModelContainer {
        if let task = llmTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.llm
        let task = Task {
            print("Loading LLM: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded LLM: \(id)")
            return container
        }
        llmTask = task
        return try await task.value
    }

    public func vlmContainer() async throws -> LMModelContainer {
        if let task = vlmTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.vlm
        let task = Task {
            print("Loading VLM: \(id)")
            let container = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded VLM: \(id)")
            return container
        }
        vlmTask = task
        return try await task.value
    }

    public func lfm2Container() async throws -> LMModelContainer {
        if let task = lfm2Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.lfm2
        let task = Task {
            print("Loading LFM2: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded LFM2: \(id)")
            return container
        }
        lfm2Task = task
        return try await task.value
    }

    public func glm4Container() async throws -> LMModelContainer {
        if let task = glm4Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.glm4
        let task = Task {
            print("Loading GLM4: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded GLM4: \(id)")
            return container
        }
        glm4Task = task
        return try await task.value
    }

    public func embeddingContainer() async throws -> EmbeddingModelContainer {
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = "nomic_text_v1_5"
        print("Loading embedding model: \(id)")
        let container = try await MLXEmbedders.loadModelContainer(
            from: downloader, using: tokenizerLoader, configuration: .nomic_text_v1_5,
            progressHandler: logProgress(id)
        )
        print("Loaded embedding model: \(id)")
        return container
    }
}

// MARK: - ChatSession Tests

private let generateParameters = GenerateParameters(maxTokens: 200, temperature: 0)

public enum ChatSessionTests {

    public static func oneShot(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "One-shot")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in response, got: \(result)"
        )
    }

    public static func oneShotStream(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "Stream")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in streamed response, got: \(result)"
        )
    }

    public static func multiTurnConversation(container: LMModelContainer) async throws {
        let session = ChatSession(
            container, instructions: "You are a helpful assistant. Keep responses brief.",
            generateParameters: generateParameters)

        _ = try await streamAndCollect(
            session.streamResponse(
                to: "My name is Alice."), label: "Turn 1")

        let response2 = try await streamAndCollect(
            session.streamResponse(
                to: "What is my name?"), label: "Turn 2")

        try check(
            response2.lowercased().contains("alice"),
            "Expected 'Alice' in response, got: \(response2)"
        )
    }

    public static func visionModel(container: LMModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let redImage = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 100, height: 100))

        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What color is this image? Reply with just the color name.",
                image: .ciImage(redImage)), label: "Vision")
        try check(
            result.lowercased().contains("red"),
            "Expected 'red' in response, got: \(result)"
        )
    }

    public static func streamDetailsWithTools(container: LMModelContainer) async throws {
        let tools: [ToolSpec] = [weatherToolSchema]
        let session = ChatSession(container, generateParameters: generateParameters, tools: tools)

        var responseText = ""
        var toolCalls: [ToolCall] = []

        var info: GenerateCompletionInfo?
        print("Tools: ", terminator: "")
        for try await generation in session.streamDetails(
            to: "What is the weather in San Francisco?", images: [], videos: [])
        {
            switch generation {
            case .chunk(let text):
                print(text, terminator: "")
                responseText += text
            case .toolCall(let toolCall):
                toolCalls.append(toolCall)
            case .info(let completionInfo):
                info = completionInfo
            }
        }
        print()
        if let info {
            print(
                "Generation info: \(info.generationTokenCount) tokens, stop reason: \(info.stopReason)"
            )
        }
        if !toolCalls.isEmpty {
            print("Tool calls: \(toolCalls)")
        }

        try check(
            !responseText.isEmpty || !toolCalls.isEmpty,
            "Expected either text or tool calls, got neither (generated \(info?.generationTokenCount ?? 0) tokens, stop reason: \(String(describing: info?.stopReason)))"
        )

        // If we got tool calls, feed back a tool result and verify the model responds
        if !toolCalls.isEmpty {
            let followUp = try await streamAndCollect(
                session.streamResponse(
                    to: "Foggy with a high in the low 60s, clearing later in the day",
                    role: .tool, images: [], videos: []),
                label: "Tool result")
            try check(
                !followUp.isEmpty,
                "Expected a response after providing tool result, got empty string"
            )
        }
    }

    public static func toolInvocation(container: LMModelContainer) async throws {
        struct EmptyInput: Codable {}

        struct TimeOutput: Codable {
            let time: String
        }

        let timeTool = Tool<EmptyInput, TimeOutput>(
            name: "get_time",
            description: "Get the current date and time including day of week.",
            parameters: []
        ) { _ in
            TimeOutput(time: "Wed Feb 18 17:50:43 PST 2026")
        }

        let session = ChatSession(
            container, generateParameters: generateParameters,
            tools: [timeTool.schema]
        ) { toolCall in
            if toolCall.function.name == timeTool.name {
                return try await toolCall.execute(with: timeTool).toolResult
            }
            return "Unknown tool: \(toolCall.function.name)"
        }

        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What day of week is it?"), label: "Tool invocation")

        try check(
            result.lowercased().contains("wed") || result.lowercased().contains("wednesday"),
            "Expected 'Wed' or 'Wednesday' in response, got: \(result)"
        )
    }

    public static func promptRehydration(container: LMModelContainer) async throws {
        let history: [Chat.Message] = [
            .system("You are a helpful assistant."),
            .user("My name is Bob."),
            .assistant("Hello Bob! How can I help you today?"),
        ]

        let session = ChatSession(
            container, history: history, generateParameters: generateParameters)
        let response = try await streamAndCollect(
            session.streamResponse(
                to: "What is my name?"), label: "Rehydration")

        try check(
            response.lowercased().contains("bob"),
            "Expected 'Bob' in response (prompt rehydration), got: \(response)"
        )
    }
}

// MARK: - Stream Helper

private func streamAndCollect(
    _ stream: AsyncThrowingStream<String, Error>,
    label: String
) async throws -> String {
    var result = ""
    print("\(label): ", terminator: "")
    for try await token in stream {
        print(token, terminator: "")
        result += token
    }
    print()
    return result
}

// MARK: - Embedder Tests

public enum EmbedderTests {

    public static func readmeExample(container: EmbeddingModelContainer) async throws {
        let searchInputs = [
            "search_query: Animals in Tropical Climates.",
            "search_document: Elephants",
            "search_document: Horses",
            "search_document: Polar Bears",
        ]

        let resultEmbeddings = await container.perform {
            (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
            let inputs = searchInputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let maxLength = inputs.reduce(into: 16) { acc, elem in
                acc = max(acc, elem.count)
            }
            let padded = stacked(
                inputs.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })
            let mask = (padded .!= tokenizer.eosTokenId ?? 0)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        let searchQueryEmbedding = resultEmbeddings[0]
        let documentEmbeddings = resultEmbeddings[1...]
        let similarities = documentEmbeddings.map { docEmbedding in
            zip(searchQueryEmbedding, docEmbedding).map(*).reduce(0, +)
        }
        let documentNames = searchInputs[1...].map {
            $0.replacingOccurrences(of: "search_document: ", with: "")
        }

        let expectedSimilarities: [Float] = [0.6854175, 0.6644787, 0.63326025]
        let tolerance: Float = 1e-4

        for (index, resultSimilarity) in similarities.enumerated() {
            try check(
                abs(resultSimilarity - expectedSimilarities[index]) < tolerance,
                "Similarity mismatch for \(documentNames[index]): expected \(expectedSimilarities[index]), got \(resultSimilarity)"
            )
        }
    }
}

// MARK: - Tool Call Tests

public enum ToolCallTests {

    public static func lfm2FormatAutoDetection(container: LMModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.lfm2,
            "Expected .lfm2 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func lfm2EndToEndGeneration(container: LMModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Tokyo?")

        print("LFM2 Output:", result)
        print("LFM2 Tool Calls:", toolCalls)

        if !toolCalls.isEmpty {
            let toolCall = toolCalls[0]
            try check(
                toolCall.function.name == "get_weather",
                "Expected tool name 'get_weather', got: \(toolCall.function.name)"
            )
            if case .string(let location) = toolCall.function.arguments["location"] {
                try check(
                    location.lowercased().contains("tokyo"),
                    "Expected location containing 'Tokyo', got: \(location)"
                )
            }
        }
    }

    public static func glm4FormatAutoDetection(container: LMModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.glm4,
            "Expected .glm4 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func glm4EndToEndGeneration(container: LMModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Paris?")

        print("GLM4 Output:", result)
        print("GLM4 Tool Calls:", toolCalls)

        if !toolCalls.isEmpty {
            let toolCall = toolCalls[0]
            try check(
                toolCall.function.name == "get_weather",
                "Expected tool name 'get_weather', got: \(toolCall.function.name)"
            )
            if case .string(let location) = toolCall.function.arguments["location"] {
                try check(
                    location.lowercased().contains("paris"),
                    "Expected location containing 'Paris', got: \(location)"
                )
            }
        }
    }

    private static func generateWithTools(
        container: LMModelContainer,
        userMessage: String
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        try await container.perform { context in
            let input = UserInput(
                chat: [
                    .system(
                        "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                    ),
                    .user(userMessage),
                ],
                tools: [weatherToolSchema]
            )
            let lmInput = try await context.processor.prepare(input: input)
            let stream = try generate(
                input: lmInput,
                parameters: GenerateParameters(maxTokens: 100),
                context: context
            )

            var text = ""
            var toolCalls: [ToolCall] = []
            for try await generation in stream {
                switch generation {
                case .chunk(let chunk):
                    text += chunk
                case .toolCall(let toolCall):
                    toolCalls.append(toolCall)
                case .info:
                    break
                }
            }
            return (text, toolCalls)
        }
    }
}

// MARK: - Progress Logging

private func logProgress(_ label: String) -> @Sendable (Progress) -> Void {
    let lock = NSLock()
    nonisolated(unsafe) var lastThreshold = -1
    return { progress in
        let pct = Int(progress.fractionCompleted * 100)
        let threshold = pct / 5
        lock.lock()
        let shouldPrint = threshold > lastThreshold
        if shouldPrint { lastThreshold = threshold }
        lock.unlock()
        if shouldPrint {
            print("  \(label): \(pct)%")
        }
    }
}

// MARK: - Shared Constants

private let weatherToolSchema: ToolSpec = [
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
