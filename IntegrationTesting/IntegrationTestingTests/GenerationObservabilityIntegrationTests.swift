// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct GenerationObservabilityIntegrationTests {
    @Test func composedHandlerPreservesModelGeneration() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.llama3_2_1B_4bit)
        let prompt = "Explain why the sky is blue in one sentence."
        let parameters = GenerateParameters(maxTokens: 32, temperature: 0)
        let input = try await container.prepare(input: UserInput(prompt: prompt))
        let (referenceStream, referenceTask) = try await container.perform(nonSendable: input) {
            context, input in
            try generateTokensTask(input: input, parameters: parameters, context: context)
        }
        var referenceTokens: [Int] = []
        for await event in referenceStream {
            if let token = event.token { referenceTokens.append(token) }
        }
        await referenceTask.value
        #expect(!referenceTokens.isEmpty)

        for topCount in [nil, 0, 5] as [Int?] {
            var parameters = parameters
            parameters.logProbabilities = topCount
            let input = try await container.prepare(input: UserInput(prompt: prompt))
            let (stream, task) = try await container.perform(nonSendable: input) {
                [parameters] context, input in
                let iterator = try TokenIterator(
                    input: input, model: context.model, parameters: parameters)
                return generateLoopTask(
                    promptTokenCount: input.text.tokens.size,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer, iterator: iterator,
                    handler: LogProbabilityTokenLoopHandler(RawTokenLoopHandler()))
            }

            var tokens: [Int] = []
            var probabilities: [GenerateTokenLogProbabilities] = []
            var completion: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .generation(.token(let token)): tokens.append(token)
                case .generation(.info(let info)): completion = info
                case .probability(let values): probabilities.append(values)
                }
            }
            await task.value

            #expect(tokens == referenceTokens)
            #expect(completion?.generationTokenCount == tokens.count)
            if let topCount {
                #expect(probabilities.map(\.chosen.token) == tokens)
                #expect(
                    probabilities.allSatisfy {
                        $0.chosen.logProbability.isFinite && $0.chosen.logProbability <= 0
                            && $0.topLogProbabilities.count == topCount
                    })
            } else {
                #expect(probabilities.isEmpty)
            }
            if let completion {
                print(
                    "Log probabilities \(String(describing: topCount)): \(completion.tokensPerSecond) tokens/s"
                )
            }
        }
    }

    @Test(arguments: [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Qwen3.5-4B-4bit",
    ])
    func cachedSessionObservationPreservesGeneration(_ modelID: String) async throws {
        let container = try await models.llmContainer(for: ModelConfiguration(id: modelID))
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)
        let plain = ChatSession(
            container, generateParameters: parameters,
            additionalContext: ["enable_thinking": false])
        let observed = ChatSession(
            container, generateParameters: parameters,
            additionalContext: ["enable_thinking": false])
        for prompt in ["What is 2 + 2? Answer in one sentence.", "Now add 1 to that answer."] {
            var expected = ""
            var expectedInfo: GenerateCompletionInfo?
            for try await event in plain.streamDetails(to: prompt) {
                if let text = event.chunk { expected += text }
                if let info = event.info { expectedInfo = info }
            }
            var probabilities: [GenerateTokenLogProbabilities] = []
            var actual = ""
            var actualInfo: GenerateCompletionInfo?
            for try await event in observed.observe(
                to: prompt, using: .logProbabilities(topCount: 4))
            {
                if let value = event.logProbabilities { probabilities.append(value) }
                if let generation = event.generation {
                    if let text = generation.chunk { actual += text }
                    if let info = generation.info { actualInfo = info }
                }
            }
            let expectedCompletion = try #require(expectedInfo)
            let actualCompletion = try #require(actualInfo)
            #expect(!expected.isEmpty)
            #expect(actual == expected)
            #expect(
                actualCompletion.generationTokenCount == expectedCompletion.generationTokenCount)
            #expect(
                actualCompletion.cachedPromptTokenCount == expectedCompletion.cachedPromptTokenCount
            )
            #expect(probabilities.count == actualCompletion.generationTokenCount)
            #expect(
                probabilities.allSatisfy {
                    $0.chosen.logProbability.isFinite && $0.chosen.logProbability <= 0
                        && $0.topLogProbabilities.count == 4
                })
            print(
                "ChatSession observation: \(modelID), cached tokens \(actualCompletion.cachedPromptTokenCount), generated tokens \(actualCompletion.generationTokenCount)"
            )
        }
    }

}
