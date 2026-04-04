// Integration test for Gemma 4 model loading and inference.
// Run: xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/Gemma4IntegrationTests

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

/// Bridges swift-transformers Tokenizer to MLXLMCommon.Tokenizer protocol
struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: Tokenizers.Tokenizer

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        return try upstream.applyChatTemplate(messages: messages)
    }
}

struct HFTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream: tokenizer)
    }
}

public class Gemma4IntegrationTests: XCTestCase {

    func testGemma4RealModel() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let modelDir =
            "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/dc1f72fa71acb997e1581a8ec8f69edd6e8f5707"

        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("Gemma 4 model not found in HF cache")
        }

        let modelURL = URL(fileURLWithPath: modelDir)

        print("Loading Gemma 4 E4B 8-bit with HF tokenizer...")
        let start = Date()

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL, using: HFTokenizerLoader())

        let loadTime = Date().timeIntervalSince(start)
        print("Loaded in \(String(format: "%.1f", loadTime))s")

        let prompt = "Write a detailed explanation of how neural network attention mechanisms work, including the mathematical formulation of scaled dot-product attention."
        print("Prompt: \(prompt)")

        let info: GenerateCompletionInfo = try await container.perform {
            (context: ModelContext) async throws -> GenerateCompletionInfo in

            let input = try await context.processor.prepare(input: .init(prompt: prompt))
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: context)

            var output = ""
            var completionInfo: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    output += text
                case .info(let i):
                    completionInfo = i
                default:
                    break
                }
            }
            print("Response: \(output)")
            return completionInfo ?? GenerateCompletionInfo(
                promptTokenCount: 0, generationTokenCount: 0,
                promptTime: 0, generationTime: 0)
        }

        print(info.summary())

        XCTAssertGreaterThan(info.generationTokenCount, 0, "Should generate at least 1 token")
        XCTAssertGreaterThan(info.tokensPerSecond, 1.0, "Should generate faster than 1 tok/s")
    }
}
