// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

struct SpeculativeDecodingIntegrationTests {

    @Test("Speculative decoding matches non-speculative at temperature=0")
    func speculativeMatchesNonSpeculative() async throws {
        let mainContainer = try await IntegrationTestModels.shared.llmContainer()
        let draftContainer = try await IntegrationTestModels.shared.llmDraftContainer()

        let parameters = GenerateParameters(maxTokens: 32, temperature: 0.0)
        let input = UserInput(
            prompt: "Hello! Who are you?",
            additionalContext: [
                "enable_thinking": false
            ]
        )

        let normalTokens: [Int] = try await mainContainer.perform { context in
            let input = try await context.processor.prepare(input: input)
            let stream = try MLXLMCommon.generateTokens(
                input: input,
                parameters: parameters,
                context: context
            )

            var tokens = [Int]()
            for await generation in stream {
                if let token = generation.token {
                    tokens.append(token)
                }
            }

            return tokens
        }

        let speculativeTokens: [Int] = try await mainContainer.perform { context in
            let input = try await context.processor.prepare(input: input)
            let draftModel = await draftContainer.perform(\.model)
            let stream = try MLXLMCommon.generateTokens(
                input: input,
                parameters: parameters,
                context: context,
                draftModel: draftModel,
                numDraftTokens: 2
            )

            var tokens = [Int]()
            for await generation in stream {
                if let token = generation.token {
                    tokens.append(token)
                }
            }

            return tokens
        }

        #expect(!normalTokens.isEmpty)
        #expect(!speculativeTokens.isEmpty)
        #expect(normalTokens == speculativeTokens)
    }
}
