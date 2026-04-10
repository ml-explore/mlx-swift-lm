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
public typealias LLModelContainer = MLXLMCommon.ModelContainer
public typealias EmbeddingModelContainer = MLXEmbedders.EmbedderModelContainer

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
    public static let mistral3 = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
    public static let nemotron = "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
    public static let qwen35 = "mlx-community/Qwen3.5-2B-4bit"
}

// MARK: - Model Loading

/// Shared model cache that loads each model at most once per test run.
public actor IntegrationTestModels {
    private let downloader: any Downloader
    private let tokenizerLoader: any TokenizerLoader

    private var llmTask: Task<LLModelContainer, Error>?
    private var vlmTask: Task<LLModelContainer, Error>?
    private var lfm2Task: Task<LLModelContainer, Error>?
    private var glm4Task: Task<LLModelContainer, Error>?
    private var mistral3Task: Task<LLModelContainer, Error>?
    private var nemotronTask: Task<LLModelContainer, Error>?
    private var qwen35Task: Task<LLModelContainer, Error>?

    public init(downloader: any Downloader, tokenizerLoader: any TokenizerLoader) {
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
    }

    public func llmContainer() async throws -> LLModelContainer {
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

    public func vlmContainer() async throws -> LLModelContainer {
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

    public func lfm2Container() async throws -> LLModelContainer {
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

    public func glm4Container() async throws -> LLModelContainer {
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

    public func mistral3Container() async throws -> LLModelContainer {
        if let task = mistral3Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.mistral3
        let task = Task {
            print("Loading Mistral3: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded Mistral3: \(id)")
            return container
        }
        mistral3Task = task
        return try await task.value
    }

    public func nemotronContainer() async throws -> LLModelContainer {
        if let task = nemotronTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.nemotron
        let task = Task {
            print("Loading Nemotron: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded Nemotron: \(id)")
            return container
        }
        nemotronTask = task
        return try await task.value
    }

    public func qwen35Container() async throws -> LLModelContainer {
        if let task = qwen35Task {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = IntegrationTestModelIDs.qwen35
        let task = Task {
            print("Loading Qwen3.5: \(id)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: .init(id: id),
                progressHandler: logProgress(id)
            )
            print("Loaded Qwen3.5: \(id)")
            return container
        }
        qwen35Task = task
        return try await task.value
    }

    public func embeddingContainer() async throws -> EmbeddingModelContainer {
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = "nomic_text_v1_5"
        print("Loading embedding model: \(id)")
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: EmbedderRegistry.nomic_text_v1_5,
            progressHandler: logProgress(id)
        )
        print("Loaded embedding model: \(id)")
        return container
    }
}

// MARK: - ChatSession Tests

private let generateParameters = GenerateParameters(maxTokens: 200, temperature: 0)

public enum ChatSessionTests {

    public static func oneShot(container: LLModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "One-shot")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in response, got: \(result)"
        )
    }

    public static func oneShotStream(container: LLModelContainer) async throws {
        let session = ChatSession(container, generateParameters: generateParameters)
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What is 2+2? Reply with just the number."), label: "Stream")
        try check(
            result.contains("4") || result.lowercased().contains("four"),
            "Expected '4' or 'four' in streamed response, got: \(result)"
        )
    }

    public static func multiTurnConversation(container: LLModelContainer) async throws {
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

    public static func visionModel(container: LLModelContainer) async throws {
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

    public static func streamDetailsWithTools(container: LLModelContainer) async throws {
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

    public static func toolInvocation(container: LLModelContainer) async throws {
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

    public static func promptRehydration(container: LLModelContainer) async throws {
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

    public static func gemma3Embedder(
        downloader: any Downloader, tokenizerLoader: any TokenizerLoader
    ) async throws {
        let modelId = "mlx-community/gemma-3-1b-it-qat-4bit"
        print("Loading Gemma 3 embedding model: \(modelId)")
        let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: ModelConfiguration(id: modelId),
            progressHandler: logProgress(modelId)
        )
        print("Loaded Gemma 3 embedding model: \(modelId)")

        let inputs = [
            "The Coca-Cola Company is a soft drink company based in Atlanta, Georgia, USA.",
            "In the United States, PepsiCo Inc. is a leading soft drink company.",
        ]

        let resultEmbeddings = await modelContainer.perform { context in
            let tokenizer = context.tokenizer
            let encoded = inputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let maxLength = encoded.reduce(into: 1) { acc, elem in
                acc = max(acc, elem.count)
            }

            let padded = stacked(
                encoded.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })

            let mask = (padded .!= (tokenizer.eosTokenId ?? 0))
            let tokenTypes = MLXArray.zeros(like: padded)

            let modelOutput = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)

            let result = context.pooling(
                modelOutput,
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        try check(
            resultEmbeddings.count == inputs.count,
            "Should have one embedding per input, got \(resultEmbeddings.count)"
        )
        for embedding in resultEmbeddings {
            try check(
                embedding.count == 1152,
                "Gemma 3 1B embedding size should be 1152, got \(embedding.count)"
            )
            let l2Norm = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            try check(
                abs(l2Norm - 1.0) < 0.05,
                "Embeddings should be approximately L2-normalized, got L2 norm \(l2Norm)"
            )
        }

        let similarity = zip(resultEmbeddings[0], resultEmbeddings[1]).map(*).reduce(0, +)
        try check(
            similarity > 0.0,
            "Similarity between related sentences should be positive, got \(similarity)"
        )
    }

    public static func readmeExample(container: EmbeddingModelContainer) async throws {
        let searchInputs = [
            "search_query: Animals in Tropical Climates.",
            "search_document: Elephants",
            "search_document: Horses",
            "search_document: Polar Bears",
        ]

        let resultEmbeddings = await container.perform { context in
            let tokenizer = context.tokenizer

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
            let result = context.pooling(
                context.model(
                    padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
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

    public static func lfm2FormatAutoDetection(container: LLModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.lfm2,
            "Expected .lfm2 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func lfm2EndToEndGeneration(container: LLModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Tokyo?")

        print("LFM2 Output:", result)
        print("LFM2 Tool Calls:", toolCalls)

        try check(!toolCalls.isEmpty, "Expected at least one tool call, got none")
        let toolCall = toolCalls[0]
        try check(
            toolCall.function.name == "get_weather",
            "Expected tool name 'get_weather', got: \(toolCall.function.name)"
        )
        guard case .string(let location) = toolCall.function.arguments["location"] else {
            throw IntegrationTestFailure("Expected string 'location' argument")
        }
        try check(
            location.lowercased().contains("tokyo"),
            "Expected location containing 'Tokyo', got: \(location)"
        )
    }

    public static func glm4FormatAutoDetection(container: LLModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.glm4,
            "Expected .glm4 tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func glm4EndToEndGeneration(container: LLModelContainer) async throws {
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            userMessage: "What's the weather in Paris?")

        print("GLM4 Output:", result)
        print("GLM4 Tool Calls:", toolCalls)

        try check(!toolCalls.isEmpty, "Expected at least one tool call, got none")
        let toolCall = toolCalls[0]
        try check(
            toolCall.function.name == "get_weather",
            "Expected tool name 'get_weather', got: \(toolCall.function.name)"
        )
        guard case .string(let location) = toolCall.function.arguments["location"] else {
            throw IntegrationTestFailure("Expected string 'location' argument")
        }
        try check(
            location.lowercased().contains("paris"),
            "Expected location containing 'Paris', got: \(location)"
        )
    }

    // MARK: Mistral3

    public static func mistral3FormatAutoDetection(container: LLModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.mistral,
            "Expected .mistral tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func mistral3EndToEndGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: [weatherToolSchema]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 150)

        print("Mistral3 Output:", result)
        print("Mistral3 Tool Calls:", toolCalls)

        try check(!toolCalls.isEmpty, "Expected at least one tool call, got none")
        let toolCall = toolCalls[0]
        try check(
            toolCall.function.name == "get_weather",
            "Expected tool name 'get_weather', got: \(toolCall.function.name)"
        )
        guard case .string(let location) = toolCall.function.arguments["location"] else {
            throw IntegrationTestFailure("Expected string 'location' argument")
        }
        try check(
            location.lowercased().contains("tokyo"),
            "Expected location containing 'Tokyo', got: \(location)"
        )
    }

    public static func mistral3MultiToolGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user("What's the weather in Tokyo and what time is it there?"),
            ],
            tools: multiToolSchemas
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 150)

        print("Mistral3 Output:", result)
        print("Mistral3 Calls:", toolCalls)

        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            try check(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        try check(
            toolCalls.count > 1,
            "Expected multiple tool calls, got \(toolCalls.count)"
        )
    }

    // MARK: Nemotron

    public static func nemotronFormatAutoDetection(container: LLModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.xmlFunction,
            "Expected .xmlFunction tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func nemotronEndToEndGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: [weatherToolSchema],
            additionalContext: ["enable_thinking": false]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 150)

        print("Nemotron Output:", result)
        print("Nemotron Tool Calls:", toolCalls)

        try check(!toolCalls.isEmpty, "Expected at least one tool call, got none")
        let toolCall = toolCalls[0]
        try check(
            toolCall.function.name == "get_weather",
            "Expected tool name 'get_weather', got: \(toolCall.function.name)"
        )
        guard case .string(let location) = toolCall.function.arguments["location"] else {
            throw IntegrationTestFailure("Expected string 'location' argument")
        }
        try check(
            location.lowercased().contains("tokyo"),
            "Expected location containing 'Tokyo', got: \(location)"
        )
    }

    public static func nemotronMultiToolGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user("What's the weather in Tokyo and what time is it there?"),
            ],
            tools: multiToolSchemas,
            additionalContext: ["enable_thinking": false]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 600)

        print("Nemotron Output:", result)
        print("Nemotron Calls:", toolCalls)

        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            try check(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        try check(
            toolCalls.count > 1,
            "Expected multiple tool calls, got \(toolCalls.count)"
        )
    }

    // MARK: Qwen3.5

    public static func qwen35FormatAutoDetection(container: LLModelContainer) async throws {
        let config = await container.configuration
        try check(
            config.toolCallFormat == ToolCallFormat.xmlFunction,
            "Expected .xmlFunction tool call format, got: \(String(describing: config.toolCallFormat))"
        )
    }

    public static func qwen35EndToEndGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: [weatherToolSchema]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 150)

        print("Qwen3.5 Output:", result)
        print("Qwen3.5 Tool Calls:", toolCalls)

        try check(!toolCalls.isEmpty, "Expected at least one tool call, got none")
        let toolCall = toolCalls[0]
        try check(
            toolCall.function.name == "get_weather",
            "Expected tool name 'get_weather', got: \(toolCall.function.name)"
        )
        guard case .string(let location) = toolCall.function.arguments["location"] else {
            throw IntegrationTestFailure("Expected string 'location' argument")
        }
        try check(
            location.lowercased().contains("tokyo"),
            "Expected location containing 'Tokyo', got: \(location)"
        )
    }

    public static func qwen35MultiToolGeneration(container: LLModelContainer) async throws {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user("What's the weather in Tokyo and what time is it there?"),
            ],
            tools: multiToolSchemas,
            additionalContext: ["enable_thinking": true]
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container, input: input, maxTokens: 300)

        print("Qwen3.5 Output:", result)
        print("Qwen3.5 Calls:", toolCalls)

        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            try check(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        try check(
            toolCalls.count > 1,
            "Expected multiple tool calls, got \(toolCalls.count)"
        )
    }

    // MARK: Helpers

    private static func generateWithTools(
        container: LLModelContainer,
        input: UserInput,
        maxTokens: Int = 100
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        try await container.perform(nonSendable: input) { context, input in
            let lmInput = try await context.processor.prepare(input: input)
            let stream = try generate(
                input: lmInput,
                parameters: GenerateParameters(maxTokens: maxTokens),
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

    private static func generateWithTools(
        container: LLModelContainer,
        userMessage: String
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user(userMessage),
            ],
            tools: [weatherToolSchema]
        )
        return try await generateWithTools(
            container: container, input: input)
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

private let timeToolSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_time",
        "description": "Get the current time in a given timezone",
        "parameters": [
            "type": "object",
            "properties": [
                "timezone": [
                    "type": "string",
                    "description": "The timezone, e.g. America/New_York, Asia/Tokyo",
                ] as [String: any Sendable]
            ] as [String: any Sendable],
            "required": ["timezone"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

private let multiToolSchemas: [ToolSpec] = [weatherToolSchema, timeToolSchema]
