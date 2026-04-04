// Gemma 4 raw model throughput benchmark
// Isolates model forward pass from tokenizer, detokenizer, and async machinery

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
    ) throws -> [Int] { try upstream.applyChatTemplate(messages: messages) }
}

struct HFTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TransformersTokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory))
    }
}

/// Force synchronous evaluation of MLXArrays (calls into MLX C backend)
func forceEval(_ arrays: MLXArray...) {
    MLX.eval(arrays)
}

@main
struct Gemma4Test {
    static func main() async throws {
        print("=== Gemma 4 Raw Model Throughput Benchmark ===\n")

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let modelDir =
            if CommandLine.arguments.count > 1 { CommandLine.arguments[1] }
            else {
                "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/dc1f72fa71acb997e1581a8ec8f69edd6e8f5707"
            }

        guard FileManager.default.fileExists(atPath: modelDir) else {
            print("ERROR: Model not found"); return
        }

        print("Loading model...")
        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelDir), using: HFTokenizerLoader())
        print("Model loaded.\n")

        // ========== Test 1: Raw model forward pass (no tokenizer, no sampling) ==========
        print("=== Test 1: Raw model.callAsFunction() throughput ===")
        try await container.perform { (ctx: ModelContext) in
            let model = ctx.model

            guard let lm = model as? Gemma4TextModel else {
                print("ERROR: Model is not Gemma4TextModel"); return
            }
            let cache = lm.newCache()

            // Warmup: prefill + 4 token steps
            let promptTokens = MLXArray([2, 1596, 603, 476, 2195])[.newAxis, .ellipsis]
            let _ = model(promptTokens, cache: cache)
            MLX.eval(cache.flatMap { $0.state })

            var lastToken = MLXArray([235265])[.newAxis, .ellipsis]
            for _ in 0..<4 {
                let logits = model(lastToken, cache: cache)
                let nextToken = logits.argMax(axis: -1)
                forceEval(nextToken)
                lastToken = nextToken
            }
            MLX.Memory.clearCache()

            // Benchmark: 128 raw forward passes
            let numTokens = 128
            print("Running \(numTokens) raw forward passes...")
            let start = Date()

            for _ in 0..<numTokens {
                let logits = model(lastToken, cache: cache)
                let nextToken = logits.argMax(axis: -1)
                forceEval(nextToken)
                lastToken = nextToken
            }

            let elapsed = Date().timeIntervalSince(start)
            let tps = Double(numTokens) / elapsed
            print("Raw forward pass: \(String(format: "%.1f", tps)) tok/s (\(numTokens) tokens, \(String(format: "%.3f", elapsed))s)")
        }

        // ========== Test 2: Full pipeline without detokenization ==========
        print("\n=== Test 2: Full pipeline (no detokenization) ===")
        let prompt = "Write a detailed explanation of how neural network attention mechanisms work, including the mathematical formulation of scaled dot-product attention."

        // Warmup
        try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: "Hello"))
            let _: GenerateCompletionInfo = try generate(
                input: input, parameters: .init(maxTokens: 8, temperature: 0.0),
                context: ctx) { (_: Int) -> GenerateDisposition in .more }
        }
        MLX.Memory.clearCache()

        let info: GenerateCompletionInfo = try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: prompt))
            return try generate(
                input: input, parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: ctx) { (_: Int) -> GenerateDisposition in .more }
        }
        print("No detokenize: \(String(format: "%.1f", info.tokensPerSecond)) tok/s (\(info.generationTokenCount) tokens)")

        // ========== Test 3: Full pipeline WITH detokenization ==========
        print("\n=== Test 3: Full pipeline + detokenization ===")
        MLX.Memory.clearCache()

        let info3: GenerateCompletionInfo = try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: prompt))
            return try generate(
                input: input, parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: ctx) { (token: Int) -> GenerateDisposition in
                let _ = ctx.tokenizer.decode(tokenIds: [token])
                return .more
            }
        }
        print("With detokenize: \(String(format: "%.1f", info3.tokensPerSecond)) tok/s (\(info3.generationTokenCount) tokens)")

        print("\n=== Summary ===")
        print("Python baseline: 75.5 tok/s (full pipeline, 128 tokens, warmed up)")
    }
}
