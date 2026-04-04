// Gemma 4 integration test — load real model and generate text
// Run via xcodebuild:
//   xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
//     -only-testing:MLXLMTests/Gemma4IntegrationTests/testGemma4RealModel

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// Bridges swift-transformers Tokenizer to MLXLMCommon.Tokenizer protocol
struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: Tokenizers.Tokenizer

    var bosToken: String? {
        upstream.bosToken
    }
    var eosToken: String? {
        upstream.eosToken
    }
    var unknownToken: String? {
        upstream.unknownToken
    }

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
        // Convert to swift-transformers Message format
        return try upstream.applyChatTemplate(messages: messages)
    }
}

/// Loads tokenizer from a local directory using swift-transformers
struct HFTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream: tokenizer)
    }
}

@main
struct Gemma4Test {
    static func main() async throws {
        print("=== Gemma 4 Swift Inference Test ===\n")

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let modelDir =
            if CommandLine.arguments.count > 1 {
                CommandLine.arguments[1]
            } else {
                "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/dc1f72fa71acb997e1581a8ec8f69edd6e8f5707"
            }

        let modelURL = URL(fileURLWithPath: modelDir)
        guard FileManager.default.fileExists(atPath: modelDir) else {
            print("ERROR: Model not found at \(modelDir)")
            return
        }

        print("Loading from: \(modelDir)")
        let start = Date()

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL, using: HFTokenizerLoader())

        let loadTime = Date().timeIntervalSince(start)
        print("Model loaded in \(String(format: "%.1f", loadTime))s\n")

        let prompt = "Write a detailed explanation of how neural network attention mechanisms work, including the mathematical formulation of scaled dot-product attention."
        print("Prompt: \(prompt)")
        print("Response: ", terminator: "")

        let genStart = Date()

        let info: GenerateCompletionInfo = try await container.perform {
            (context: ModelContext) async throws -> GenerateCompletionInfo in

            let input = try await context.processor.prepare(input: .init(prompt: prompt))
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: .init(maxTokens: 128, temperature: 0.0, topP: 1.0),
                context: context)

            var completionInfo: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    print(text, terminator: "")
                    fflush(stdout)
                case .info(let i):
                    completionInfo = i
                default:
                    break
                }
            }
            return completionInfo ?? GenerateCompletionInfo(
                promptTokenCount: 0, generationTokenCount: 0,
                promptTime: 0, generationTime: 0)
        }

        let genTime = Date().timeIntervalSince(genStart)
        print("\n\n--- Performance ---")
        print("Prompt:  \(info.promptTokenCount) tokens, \(String(format: "%.1f", info.promptTokensPerSecond)) tok/s")
        print("Gen:     \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tok/s")
        print("Total:   \(String(format: "%.2f", genTime))s")
        print("\nPython baseline: 79.8 tok/s gen, 9017 MB peak")
    }
}
