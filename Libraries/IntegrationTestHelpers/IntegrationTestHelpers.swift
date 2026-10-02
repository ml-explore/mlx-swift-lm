// Shared integration test logic for verifying end-to-end model loading and generation.
// Integration packages inject their own Downloader and TokenizerLoader, then call
// these functions which run the test and throw on failure.

import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXRerankers
import MLXVLM

#if canImport(CoreImage)
import CoreImage
#endif

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

// MARK: - Harmony tokenizer contract

public enum HarmonyProtocolTests {
    /// Exercises the production tokenizer adapter and GPT-OSS chat template
    /// without downloading model weights.
    public static func realTokenizerContract(tokenizer: any Tokenizer) throws {
        let expectedControlTokenIDs = [
            "<|return|>": 200_002,
            "<|constrain|>": 200_003,
            "<|channel|>": 200_005,
            "<|start|>": 200_006,
            "<|end|>": 200_007,
            "<|message|>": 200_008,
            "<|call|>": 200_012,
        ]

        for (token, expectedID) in expectedControlTokenIDs {
            try check(
                tokenizer.convertTokenToId(token) == expectedID,
                "GPT-OSS tokenizer resolved \(token) to "
                    + "\(String(describing: tokenizer.convertTokenToId(token))); expected \(expectedID)"
            )
            try check(
                tokenizer.encode(text: token, addSpecialTokens: false) == [expectedID],
                "GPT-OSS tokenizer did not encode \(token) atomically")
        }

        try check(
            HarmonyFrameParser(tokenizer: tokenizer) != nil,
            "HarmonyFrameParser rejected the production GPT-OSS tokenizer")
        try check(
            HarmonyFrameParser.stopTokenIDs(tokenizer: tokenizer) == [200_002, 200_012],
            "Harmony stop-token resolution did not match <|return|>/<|call|>")

        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": "Paris"]),
            id: "call_fixture")
        let generator: any MessageGenerator = GPTOSSMessageGenerator()
        let messages = generator.generate(messages: [
            .user("Weather in Paris?"),
            .assistant("", toolCalls: [call]),
            .tool(#"{"forecast":"sunny"}"#, id: "call_fixture"),
        ])
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "city": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let rendered = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        try check(rendered.contains(200_012), "Real GPT-OSS template omitted the tool-call token")
        let promptStart = rendered.lastIndex(of: 200_006)
        let promptSuffix = promptStart.map {
            tokenizer.decode(
                tokenIds: Array(rendered[$0...]), skipSpecialTokens: false)
        }
        try check(
            promptSuffix == "<|start|>assistant",
            "Real GPT-OSS template omitted the assistant generation prompt")
    }
}

// MARK: - Muse Glimmer tokenizer contract

public enum MuseGlimmerProtocolTests {
    /// Exercises Meta's production Onyx control vocabulary and ATEM chat
    /// template without loading the 30B checkpoint.
    public static func realTokenizerContract(tokenizer: any Tokenizer) throws {
        let expectedControlTokenIDs = [
            "<|begin_of_text|>": 200_000,
            "<|end_of_text|>": 200_001,
            "<|eom|>": 200_007,
            "<|eot|>": 200_008,
            "<|start|>": 200_022,
            "<|message|>": 200_023,
        ]
        for (token, expectedID) in expectedControlTokenIDs {
            try check(
                tokenizer.convertTokenToId(token) == expectedID,
                "Muse Glimmer tokenizer resolved \(token) to "
                    + "\(String(describing: tokenizer.convertTokenToId(token))); expected \(expectedID)"
            )
            try check(
                tokenizer.encode(text: token, addSpecialTokens: false) == [expectedID],
                "Muse Glimmer tokenizer did not encode \(token) atomically")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "weather.get",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "city": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
        guard
            var decoder = ToolCallFormat.atem.makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer, tools: tools, stopStrings: [])
        else {
            throw IntegrationTestFailure(
                "ATEM protocol adapter rejected the production Muse Glimmer tokenizer")
        }
        let payload =
            "<atem:function_calls><atem:invoke name=\"weather.get\">"
            + "<atem:parameter name=\"city\">Paris</atem:parameter>"
            + "</atem:invoke></atem:function_calls>"
        let frameTokens =
            tokenizer.encode(text: " to=weather.get", addSpecialTokens: false)
            + [200_023]
            + tokenizer.encode(text: payload, addSpecialTokens: false)
            + [200_008]
        var parsedCalls: [ToolCall] = []
        for token in frameTokens {
            _ = decoder.push(token) { event in
                if case .toolCall(let call) = event { parsedCalls.append(call) }
                return true
            }
        }
        try check(parsedCalls.count == 1, "Onyx decoder did not emit the production ATEM call")
        let call = parsedCalls[0]
        guard let callID = call.id else {
            throw IntegrationTestFailure("Onyx decoder emitted a tool call without an id")
        }
        let messages = DefaultMessageGenerator().generate(messages: [
            .user("Weather in Paris?"),
            .assistant("", toolCalls: [call]),
            .tool(
                #"{"forecast":"sunny"}"#, id: callID,
                name: call.function.name),
        ])
        let rendered = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let text = tokenizer.decode(tokenIds: rendered, skipSpecialTokens: false)
        try check(
            text.contains(#"<atem:invoke name="weather.get">"#),
            "Muse Glimmer template omitted the structured ATEM call")
        try check(
            text.contains(#"<tool_output name="weather.get">"#),
            "Muse Glimmer template did not correlate the tool result by call id")
        try check(
            text.hasSuffix("<|start|>assistant"),
            "Muse Glimmer template omitted the assistant continuation prompt")
        try check(
            decoder.additionalStopTokenIDs == Set([200_001, 200_008]),
            "Onyx decoder did not declare <|end_of_text|>/<|eot|> as stop tokens")
    }
}

// MARK: - Network Retry

/// Transient network failures worth retrying on a flaky CI network — chiefly
/// `-1005 networkConnectionLost`, which surfaced mid-download in CI.
private func isTransientNetworkError(_ error: Error) -> Bool {
    guard let urlError = error as? URLError else { return false }
    switch urlError.code {
    case .networkConnectionLost, .timedOut, .cannotConnectToHost,
        .notConnectedToInternet, .dnsLookupFailed, .cannotFindHost,
        .resourceUnavailable, .badServerResponse:
        return true
    default:
        return false
    }
}

/// Run `operation`, retrying a few times with linear backoff on transient
/// network errors. Non-network errors (and the final attempt) rethrow.
private func withNetworkRetry<T>(
    _ label: String, attempts: Int = 3, _ operation: () async throws -> T
) async throws -> T {
    var lastError: Error?
    for attempt in 1 ... attempts {
        do {
            return try await operation()
        } catch {
            lastError = error
            guard isTransientNetworkError(error), attempt < attempts else { throw error }
            print(
                "Transient network error loading \(label) "
                    + "(attempt \(attempt)/\(attempts)): \(error). Retrying…")
            try? await Task.sleep(nanoseconds: UInt64(attempt) * 2_000_000_000)
        }
    }
    throw lastError ?? IntegrationTestFailure("\(label): retry loop exited unexpectedly")
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

    private var llmTasksByName: [String: Task<LLModelContainer, Error>] = [:]
    private var vlmTasksByName: [String: Task<LLModelContainer, Error>] = [:]
    private var embeddingTask: Task<EmbeddingModelContainer, Error>?

    public init(downloader: any Downloader, tokenizerLoader: any TokenizerLoader) {
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
    }

    /// Load an arbitrary LLM container, cached by `configuration.name` so the same
    /// model is only loaded once per `IntegrationTestModels` instance.
    public func llmContainer(for configuration: ModelConfiguration) async throws
        -> LLModelContainer
    {
        let key = configuration.name
        if let task = llmTasksByName[key] {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let task = Task {
            print("Loading LLM: \(key)")
            let container = try await withNetworkRetry(key) {
                try await LLMModelFactory.shared.loadContainer(
                    from: downloader, using: tokenizerLoader,
                    configuration: configuration,
                    progress: .download(logProgress(key))
                )
            }
            print("Loaded LLM: \(key)")
            return container
        }
        llmTasksByName[key] = task
        return try await task.value
    }

    /// Drop the cached container for `configuration` so ARC can free its
    /// GPU-resident weights between tests. Pair with `Memory.clearCache()` at the
    /// call site to release the freed buffers back to the system — without this,
    /// loading many large models in one serialized run accumulates weights until
    /// the process is jetsammed (Metal compiler XPC failures / crashes).
    public func evictLLM(_ configuration: ModelConfiguration) {
        llmTasksByName[configuration.name] = nil
    }

    /// Load an arbitrary VLM container, cached by `configuration.name` so the same
    /// model is only loaded once per `IntegrationTestModels` instance.
    public func vlmContainer(for configuration: ModelConfiguration) async throws
        -> LLModelContainer
    {
        let key = configuration.name
        if let task = vlmTasksByName[key] {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let task = Task {
            print("Loading VLM: \(key)")
            let container = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: configuration,
                progress: .download(logProgress(key))
            )
            print("Loaded VLM: \(key)")
            return container
        }
        vlmTasksByName[key] = task
        return try await task.value
    }

    public func embeddingContainer() async throws -> EmbeddingModelContainer {
        if let task = embeddingTask {
            return try await task.value
        }
        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let id = "nomic_text_v1_5"
        let task = Task {
            print("Loading embedding model: \(id)")
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: EmbedderRegistry.nomic_text_v1_5,
                progress: .download(logProgress(id))
            )
            print("Loaded embedding model: \(id)")
            return container
        }
        embeddingTask = task
        return try await task.value
    }
}

// MARK: - Vision Test Images

/// A solid-color square image for vision smoke tests.
public enum VisionTestImages {
    public static func solidColor(_ color: CIColor, size: CGFloat = 100) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
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
        #if canImport(CoreImage)
        let session = ChatSession(container, generateParameters: generateParameters)
        let redImage = VisionTestImages.solidColor(.red)

        let result = try await streamAndCollect(
            session.streamResponse(
                to: "What color is this image? Reply with just the color name.",
                image: .ciImage(redImage)), label: "Vision")
        try check(
            result.lowercased().contains("red"),
            "Expected 'red' in response, got: \(result)"
        )
        #else
        fatalError(
            "Vision model test requires CoreImage, which is not available on this platform.")
        #endif
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
            case .rejectedToolCall(let rejection):
                throw RejectedToolCallError(rejection)
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
                session.streamResponse(to: [
                    .tool("Foggy with a high in the low 60s, clearing later in the day")
                ]),
                label: "Tool result")
            try check(
                !followUp.isEmpty,
                "Expected a response after providing tool result, got empty string"
            )
        }
    }

    /// Exercises the structured continuation path used by clients that execute
    /// tool calls outside `ChatSession` and feed the results back afterward.
    ///
    /// Conversation-aware templates must receive the original user query and
    /// assistant tool call again on the result turn. Rendering only the new
    /// `.tool` message reproduces the Qwen Jinja "No user query found" failure.
    public static func structuredToolContinuation(container: LLModelContainer) async throws {
        let session = ChatSession(
            container,
            instructions:
                "Use the weather tool whenever weather is requested. After receiving its result, answer the user briefly.",
            generateParameters: GenerateParameters(maxTokens: 150, temperature: 0),
            additionalContext: ["enable_thinking": false],
            tools: [weatherToolSchema]
        )

        var firstPassCalls: [ToolCall] = []
        for try await generation in session.streamDetails(
            to: "What's the weather in Tokyo? Use the weather tool.",
            images: [],
            videos: [])
        {
            if case .toolCall(let call) = generation {
                firstPassCalls.append(call)
            }
        }

        guard let call = firstPassCalls.first else {
            throw IntegrationTestFailure("Expected the first pass to call get_weather")
        }
        try check(
            call.function.name == "get_weather",
            "Expected get_weather, got \(call.function.name)")

        var followUpText = ""
        var followUpCalls: [ToolCall] = []
        var completion: GenerateCompletionInfo?
        for try await generation in session.streamDetails(to: [
            .tool(
                #"{"location":"Tokyo","temperature_celsius":24,"conditions":"clear"}"#,
                id: call.id)
        ]) {
            switch generation {
            case .chunk(let text):
                followUpText += text
            case .toolCall(let call):
                followUpCalls.append(call)
            case .rejectedToolCall(let rejection):
                throw RejectedToolCallError(rejection)
            case .info(let info):
                completion = info
            }
        }

        try check(
            completion != nil,
            "Expected structured tool continuation to complete generation")
        try check(
            !followUpText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "Expected a final text response after the tool result")
        try check(
            followUpCalls.isEmpty,
            "Expected the tool result to resolve the request, got another tool call")
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

    public static func planetsCoherence(container: LLModelContainer) async throws {
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 3000, temperature: 0))
        let result = try await streamAndCollect(
            session.streamResponse(
                to: "List all the planets in our solar system in order from the Sun."),
            label: "Response")

        let expected = [
            "Mercury", "Venus", "Earth", "Mars",
            "Jupiter", "Saturn", "Uranus", "Neptune",
        ]
        let missing = expected.filter { !result.contains($0) }
        try check(
            missing.isEmpty,
            "Expected all planets in response, missing: \(missing). Got: \(result)"
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
            progress: .download(logProgress(modelId))
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

// MARK: - Reranker Tests

/// End-to-end checks against reference checkpoint outputs.
///
/// These checks download large model weights and belong in the separate IntegrationTesting
/// project rather than the package test suite.
public enum RerankerIntegrationTests {
    /// Validate BGE v2 M3 logits against the values published in its model card.
    public static func bgeV2M3(
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        let modelID = "BAAI/bge-reranker-v2-m3"
        let reranker = try await RerankerModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: ModelConfiguration(
                id: modelID,
                revision: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
            progressHandler: logProgress(modelID))
        let response = try await reranker.scores(
            query: "what is panda?",
            documents: [
                "hi",
                "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            ])

        try check(
            response.scoreKind == .normalizedRelevance,
            "BGE must return normalized relevance scores")
        try check(response.results.count == 2, "BGE returned an unexpected score count")
        let logits = try response.results.map {
            try inverseSigmoid($0.score, modelID: modelID)
        }
        try check(
            abs(logits[0] - -8.1875) < 0.15,
            "BGE irrelevant-passage logit diverged: expected -8.1875, received \(logits[0])")
        try check(
            abs(logits[1] - 5.261_718_75) < 0.15,
            "BGE relevant-passage logit diverged: expected 5.26171875, received \(logits[1])")
    }

    /// Validate Qwen3 reranker logit margins against its published reference values.
    public static func qwen3(
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        let modelID = "Qwen/Qwen3-Reranker-0.6B"
        let reranker = try await RerankerModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: ModelConfiguration(
                id: modelID,
                revision: "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
            progressHandler: logProgress(modelID))
        let response = try await reranker.scores(
            query: "What is the capital of China?",
            documents: [
                "The capital of China is Beijing.",
                "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
            ])
        let sequentialResponse = try await reranker.scores(
            query: "What is the capital of China?",
            documents: [
                "The capital of China is Beijing.",
                "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
            ],
            options: .init(maxBatchSize: 1))

        try check(
            response.scoreKind == .normalizedRelevance,
            "Qwen3 must return normalized relevance scores")
        try check(response.results.count == 2, "Qwen3 returned an unexpected score count")
        let margins = try response.results.map {
            try inverseSigmoid($0.score, modelID: modelID)
        }
        let sequentialMargins = try sequentialResponse.results.map {
            try inverseSigmoid($0.score, modelID: modelID)
        }
        for index in margins.indices {
            try check(
                abs(margins[index] - sequentialMargins[index]) < 0.01,
                "Qwen3 batched and sequential margins diverged at index \(index): \(margins[index]) versus \(sequentialMargins[index])"
            )
        }
        try check(
            abs(margins[0] - 7.625) < 0.25,
            "Qwen3 relevant-passage logit margin diverged: expected 7.625, received \(margins[0])")
        try check(
            abs(margins[1] - -11.375) < 1,
            "Qwen3 irrelevant-passage logit margin diverged: expected -11.375, received \(margins[1])"
        )
    }

    /// Validate Jina reranker v3 scores against its published MLX implementation.
    public static func jinaV3(
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        try await jinaV3(
            modelID: "jinaai/jina-reranker-v3-mlx",
            revision: "1d19fe38ae4e6658221479747c1152d6136dd6ab",
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    /// Validate the same scores from the upstream checkpoint the MLX repo was converted from.
    ///
    /// The two repos package one model two ways: `jinaai/jina-reranker-v3` has no index and keeps
    /// the projector in its single `model.safetensors` as a numbered `nn.Sequential`, while
    /// `jinaai/jina-reranker-v3-mlx` carries an index that names neither the projector nor its
    /// file, and renames those layers. One model class has to read both, so both are covered.
    public static func jinaV3SourceCheckpoint(
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        try await jinaV3(
            modelID: "jinaai/jina-reranker-v3",
            revision: "d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    private static func jinaV3(
        modelID: String,
        revision: String,
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        let reranker = try await RerankerModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: ModelConfiguration(id: modelID, revision: revision),
            progressHandler: logProgress(modelID))
        let response = try await reranker.scores(
            query: "What is the capital of China?",
            documents: [
                "Gravity attracts bodies toward one another.",
                "Beijing is the capital city of China.",
            ])

        try check(
            response.scoreKind == .cosineSimilarity,
            "\(modelID) must return cosine-similarity scores")
        try check(response.results.count == 2, "\(modelID) returned an unexpected score count")
        // Generated by rerank.py from the pinned MLX checkpoint revision above. The upstream
        // checkpoint holds the same bfloat16 weights, so it has to reproduce them too.
        let expectedScores = [-0.156_369_954_347_610_47, 0.402_552_098_035_812_4]
        // Quantized kernels can vary slightly with backend evaluation order.
        let tolerance = 3e-3
        for index in expectedScores.indices {
            let result = response.results[index]
            try check(
                result.index == index,
                "\(modelID) scores did not preserve document order at index \(index)")
            try check(
                abs(result.score - expectedScores[index]) < tolerance,
                "\(modelID) score diverged at index \(index): expected \(expectedScores[index]), received \(result.score)"
            )
        }
        try check(
            response.results[1].score > response.results[0].score,
            "\(modelID) did not score the relevant passage above the irrelevant passage")
    }

    private static func inverseSigmoid(_ score: Double, modelID: String) throws -> Double {
        try check(
            score > 0 && score < 1,
            "\(modelID) returned a normalized score at a non-invertible boundary")
        return Foundation.log(score / (1 - score))
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
            config.toolCallFormat == ToolCallFormat.qwen35,
            "Expected .qwen35 tool call format, got: \(String(describing: config.toolCallFormat))"
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
                // temperature: 0 (greedy) so tool-call generation is deterministic.
                // The default sampling temperature makes these end-to-end checks
                // flaky — the model may emit no tool call or malformed arguments on
                // some runs (matches the temperature: 0 used by the coherence/MTP tests).
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
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
                case .rejectedToolCall(let rejection):
                    throw RejectedToolCallError(rejection)
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

// MARK: - Hugging Face cache locations

/// Returns the root directory for Hugging Face caches (`~/.cache/huggingface`).
///
/// `FileManager.homeDirectoryForCurrentUser` is unavailable on iOS, so this helper
/// falls back to `NSHomeDirectory()`. On macOS that resolves to the real user home
/// (matching the `huggingface_hub` Python client's default cache layout); on iOS it
/// resolves to the app sandbox home, where these integration tests do not normally run
/// but where the path is at least addressable for callers that pre-populate caches.
public func hfCacheDir() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        .appendingPathComponent(".cache/huggingface", isDirectory: true)
}

/// Returns the local snapshot directory for `modelId` inside the Hugging Face hub cache,
/// following the `models--{owner}--{name}/snapshots/{rev}` layout written by `huggingface_hub`.
/// When `revision` is `nil` (the default) picks the first entry under `snapshots/` — sufficient
/// for the usual single-revision case.
/// When `revision` is non-nil, returns that specific snapshot directory if it exists.
/// Returns `nil` when the model (or the requested revision) is not present in the cache.
public func hfSnapshotDir(modelId: String, revision: String? = nil) -> URL? {
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hfCacheDir()
        .appendingPathComponent("hub", isDirectory: true)
        .appendingPathComponent(folderName, isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
    if let revision {
        let pinned = snapshots.appendingPathComponent(revision, isDirectory: true)
        var isDir: ObjCBool = false
        guard
            FileManager.default.fileExists(atPath: pinned.path, isDirectory: &isDir),
            isDir.boolValue
        else { return nil }
        return pinned
    }
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

// MARK: - Dataset download

public enum IntegrationTestDatasetError: LocalizedError {
    case listingFailed(repo: String, statusCode: Int)
    case downloadFailed(file: String, statusCode: Int)
    case noFilesMatched(repo: String, patterns: [String])

    public var errorDescription: String? {
        switch self {
        case .listingFailed(let repo, let statusCode):
            return "Failed to list files for dataset '\(repo)' (HTTP \(statusCode))"
        case .downloadFailed(let file, let statusCode):
            return "Failed to download '\(file)' from dataset (HTTP \(statusCode))"
        case .noFilesMatched(let repo, let patterns):
            return "No files in dataset '\(repo)' matched patterns \(patterns)"
        }
    }
}

/// Download a public Hugging Face dataset snapshot to a per-revision local cache.
///
/// Lists files via `huggingface.co/api/datasets/{repo}/tree/{revision}`, then
/// downloads each file matching `patterns` (or all files if `patterns` is empty)
/// from `huggingface.co/datasets/{repo}/resolve/{revision}/{file}`. Files are
/// written to `~/.cache/huggingface/integration-test-datasets/{repo}/{revision}/`.
/// Already-cached files are reused without a second HTTP fetch.
///
/// Pattern syntax: simple shell-style glob with `*` matching any sequence
/// (including `/`). Examples: `"masks/*.safetensors"`, `"*.json"`, `"foo/bar"`.
///
/// Returns the snapshot directory URL; callers build per-file paths by
/// appending the file path (e.g. `snapshotDir.appendingPathComponent("masks/q1.safetensors")`).
///
/// Tests using this helper should catch thrown errors and skip via
/// `Issue.record(...)` rather than propagate — network unavailability or HF
/// outages should not surface as parity-test failures.
public func downloadDataset(
    repo: String,
    revision: String,
    matching patterns: [String] = []
) async throws -> URL {
    let cacheRoot = hfCacheDir()
        .appendingPathComponent("integration-test-datasets", isDirectory: true)
    let snapshotDir =
        cacheRoot
        .appendingPathComponent(repo, isDirectory: true)
        .appendingPathComponent(revision, isDirectory: true)
    try FileManager.default.createDirectory(
        at: snapshotDir, withIntermediateDirectories: true)

    let host = URL(string: "https://huggingface.co")!
    let treeBase = host.appendingPathComponent("api/datasets/\(repo)/tree/\(revision)")
    var treeComponents = URLComponents(url: treeBase, resolvingAgainstBaseURL: false)!
    treeComponents.queryItems = [URLQueryItem(name: "recursive", value: "true")]
    let treeURL = treeComponents.url!
    var request = URLRequest(url: treeURL)
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    let (treeData, treeResp) = try await URLSession.shared.data(for: request)
    let treeStatus = (treeResp as? HTTPURLResponse)?.statusCode ?? 0
    guard treeStatus == 200 else {
        throw IntegrationTestDatasetError.listingFailed(repo: repo, statusCode: treeStatus)
    }

    struct TreeEntry: Decodable {
        let type: String
        let path: String
    }
    let entries = try JSONDecoder().decode([TreeEntry].self, from: treeData)
    let matched =
        entries
        .filter { $0.type == "file" }
        .map(\.path)
        .filter { path in
            patterns.isEmpty
                || patterns.contains { datasetPathMatches(path, glob: $0) }
        }
    guard !matched.isEmpty else {
        throw IntegrationTestDatasetError.noFilesMatched(repo: repo, patterns: patterns)
    }

    for file in matched {
        let dest = snapshotDir.appendingPathComponent(file)
        if FileManager.default.fileExists(atPath: dest.path) { continue }
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fileURL = host.appendingPathComponent("datasets/\(repo)/resolve/\(revision)/\(file)")
        let (tmp, resp) = try await URLSession.shared.download(from: fileURL)
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            try? FileManager.default.removeItem(at: tmp)
            throw IntegrationTestDatasetError.downloadFailed(file: file, statusCode: status)
        }
        // Concurrent callers may have populated `dest` between the existence
        // check above and this move. Accept the lost race instead of erroring.
        do {
            try FileManager.default.moveItem(at: tmp, to: dest)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            if !FileManager.default.fileExists(atPath: dest.path) {
                throw error
            }
        }
    }

    return snapshotDir
}

private func datasetPathMatches(_ path: String, glob pattern: String) -> Bool {
    let parts = pattern.split(separator: "*", omittingEmptySubsequences: false).map(String.init)
    if parts.count == 1 { return path == pattern }
    var cursor = path.startIndex
    for (i, part) in parts.enumerated() {
        if part.isEmpty {
            if i == 0 || i == parts.count - 1 { continue }
            continue
        }
        if i == 0 {
            guard path[cursor...].hasPrefix(part) else { return false }
            cursor = path.index(cursor, offsetBy: part.count)
        } else if i == parts.count - 1 {
            return path[cursor...].hasSuffix(part)
        } else {
            guard let r = path.range(of: part, range: cursor ..< path.endIndex) else {
                return false
            }
            cursor = r.upperBound
        }
    }
    return true
}
