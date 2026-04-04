// Gemma 4 performance benchmark — measures raw model throughput
// Build: xcodebuild build -scheme Gemma4Test -destination 'platform=macOS' -configuration Release

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

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
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try upstream.applyChatTemplate(messages: messages)
    }
}

struct HFTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TransformersTokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory))
    }
}

@main
struct Gemma4Test {
    static func main() async throws {
        print("=== Gemma 4 Swift Performance Benchmark ===\n")

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let modelDir =
            if CommandLine.arguments.count > 1 { CommandLine.arguments[1] }
            else {
                "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/dc1f72fa71acb997e1581a8ec8f69edd6e8f5707"
            }

        guard FileManager.default.fileExists(atPath: modelDir) else {
            print("ERROR: Model not found at \(modelDir)")
            return
        }

        print("Loading model...")
        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelDir), using: HFTokenizerLoader())
        print("Model loaded.\n")

        // Warmup (compile Metal shaders)
        print("Warmup...")
        try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: "Hello"))
            let _: GenerateCompletionInfo = try generate(
                input: input, parameters: .init(maxTokens: 8, temperature: 0.0),
                context: ctx) { (_: Int) -> GenerateDisposition in .more }
        }
        MLX.Memory.clearCache()

        // === Benchmark: synchronous tight loop ===
        let prompt = "Write a detailed explanation of how neural network attention mechanisms work, including the mathematical formulation of scaled dot-product attention."
        print("Benchmark: 128 tokens (synchronous path)")
        print("Prompt: \(prompt)\n--- Output ---")

        let info: GenerateCompletionInfo = try await container.perform {
            (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: prompt))
            return try generate(
                input: input,
                parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: ctx
            ) { (token: Int) -> GenerateDisposition in
                let text = ctx.tokenizer.decode(tokenIds: [token])
                print(text, terminator: "")
                fflush(stdout)
                return .more
            }
        }

        print("\n\n--- Performance (synchronous) ---")
        print(info.summary())
        print("Python baseline: 75.5 tok/s gen, 619.8 tok/s prompt (128 tokens, warmed up)")

        // === Benchmark: async stream path ===
        print("\n--- Async Stream benchmark ---")
        MLX.Memory.clearCache()

        let info2: GenerateCompletionInfo = try await container.perform {
            (ctx: ModelContext) async throws -> GenerateCompletionInfo in
            let input = try await ctx.processor.prepare(input: .init(prompt: prompt))
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: ctx)
            var result: GenerateCompletionInfo?
            for await gen in stream {
                if case .info(let i) = gen { result = i }
            }
            return result ?? GenerateCompletionInfo(
                promptTokenCount: 0, generationTokenCount: 0,
                promptTime: 0, generationTime: 0)
        }
        print(info2.summary())
    }
}
