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
}
