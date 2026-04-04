// Gemma 4 diagnostic benchmark — Metal memory comparison + graph analysis

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

@main
struct Gemma4Test {
    static func main() async throws {
        print("=== Gemma 4 Diagnostic Benchmark ===\n")

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

        // ========== Metal Memory Comparison ==========
        print("=== Metal Memory Analysis (10 tokens) ===")
        try await container.perform { (ctx: ModelContext) in
            let model = ctx.model
            guard let lm = model as? Gemma4TextModel else {
                print("ERROR: Not Gemma4TextModel"); return
            }

            // Warmup
            let cache = lm.newCache()
            let warmupTokens = MLXArray([2, 1596, 603, 476, 2195])[.newAxis, .ellipsis]
            let _ = model(warmupTokens, cache: cache)
            MLX.eval(cache.flatMap { $0.state })

            var lastToken = MLXArray([235265])[.newAxis, .ellipsis]
            for _ in 0..<4 {
                let logits = model(lastToken, cache: cache)
                lastToken = logits.argMax(axis: -1)
                MLX.eval(lastToken)
            }
            MLX.Memory.clearCache()

            // Snapshot memory before generation
            let memBefore = MLX.Memory.snapshot()
            print("Before generation:")
            print("  Active:  \(memBefore.activeMemory / 1_000_000) MB")
            print("  Peak:    \(memBefore.peakMemory / 1_000_000) MB")
            print("  Cache:   \(memBefore.cacheMemory / 1_000_000) MB")

            // Print module types to verify quantization
            for (path, module) in lm.leafModules().flattened().prefix(20) {
                print("  \(path): \(type(of: module))")
            }

            // Measure exactly 1 token cache delta
            let logits1 = model(lastToken, cache: cache)
            let next1 = logits1.argMax(axis: -1)
            MLX.eval(next1)
            lastToken = next1

            let memFirst = MLX.Memory.snapshot()
            print("After 1st token:")
            print("  Active:  \(memFirst.activeMemory / 1_000_000) MB")
            print("  Cache:   \(memFirst.cacheMemory / 1_000_000) MB")
            print("  1-token cache delta: \(memFirst.cacheMemory / 1_000_000) MB (Python: 2.3 MB)")

            // Clear and measure 2nd token — does clearing actually work?
            print("Clearing cache...")
            MLX.Memory.clearCache()
            let afterClear = MLX.Memory.snapshot()
            print("After clear: cache = \(afterClear.cacheMemory / 1_000_000) MB")

            let logits2 = model(lastToken, cache: cache)
            // DON'T eval yet — check cache before eval
            let beforeEval = MLX.Memory.snapshot()
            print("After graph build, before eval: cache = \(beforeEval.cacheMemory / 1_000_000) MB")

            let next2 = logits2.argMax(axis: -1)
            MLX.eval(next2)
            lastToken = next2

            let afterEval = MLX.Memory.snapshot()
            print("After eval: cache = \(afterEval.cacheMemory / 1_000_000) MB")

            // Generate remaining 8 tokens
            for _ in 0..<8 {
                let logits = model(lastToken, cache: cache)
                lastToken = logits.argMax(axis: -1)
                MLX.eval(lastToken)
            }

            let memAfter = MLX.Memory.snapshot()
            print("After 10 tokens:")
            print("  Active:  \(memAfter.activeMemory / 1_000_000) MB")
            print("  Peak:    \(memAfter.peakMemory / 1_000_000) MB")
            print("  Cache:   \(memAfter.cacheMemory / 1_000_000) MB")
            print("  Delta:   \((Int(memAfter.activeMemory) - Int(memBefore.activeMemory)) / 1_000_000) MB")
        }

        // ========== Per-token timing breakdown ==========
        print("\n=== Per-Token Timing (10 tokens, manual loop) ===")
        try await container.perform { (ctx: ModelContext) in
            let model = ctx.model
            guard let lm = model as? Gemma4TextModel else { return }
            let cache = lm.newCache()

            // Prefill
            let prompt = ctx.tokenizer.encode(text: "Write about attention mechanisms.")
            let promptArray = MLXArray(prompt)[.newAxis, .ellipsis]
            let _ = model(promptArray, cache: cache)
            MLX.eval(cache.flatMap { $0.state })

            // Get first token
            var lastToken = model(MLXArray([prompt.last!])[.newAxis, .ellipsis], cache: cache).argMax(axis: -1)
            MLX.eval(lastToken)

            // Time 10 individual tokens
            var tokenTimes: [Double] = []
            for _ in 0..<10 {
                let start = CFAbsoluteTimeGetCurrent()
                let logits = model(lastToken, cache: cache)
                lastToken = logits.argMax(axis: -1)
                MLX.eval(lastToken)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                tokenTimes.append(elapsed)
            }

            let avg = tokenTimes.reduce(0, +) / Double(tokenTimes.count)
            let min = tokenTimes.min()!
            let max = tokenTimes.max()!
            print("Per-token times (ms):")
            for (i, t) in tokenTimes.enumerated() {
                print("  Token \(i): \(String(format: "%.2f", t * 1000)) ms")
            }
            print("  Avg: \(String(format: "%.2f", avg * 1000)) ms (\(String(format: "%.1f", 1.0/avg)) tok/s)")
            print("  Min: \(String(format: "%.2f", min * 1000)) ms  Max: \(String(format: "%.2f", max * 1000)) ms")
        }

        // ========== Full pipeline benchmark ==========
        print("\n=== Full Pipeline (128 tokens) ===")
        let prompt = "Write a detailed explanation of how neural network attention mechanisms work."

        // Warmup
        try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: "Hello"))
            let _: GenerateCompletionInfo = try generate(
                input: input, parameters: .init(maxTokens: 4, temperature: 0.0),
                context: ctx) { (_: Int) -> GenerateDisposition in .more }
        }
        MLX.Memory.clearCache()

        let info: GenerateCompletionInfo = try await container.perform { (ctx: ModelContext) in
            let input = try await ctx.processor.prepare(input: .init(prompt: prompt))
            return try generate(
                input: input, parameters: .init(maxTokens: 128, temperature: 0.6, topP: 0.95),
                context: ctx) { (_: Int) -> GenerateDisposition in .more }
        }
        print(info.summary())
        print("Python baseline: 75.5 tok/s gen, peak 9003 MB active")

        // ========== Buffer donation test ==========
        runDonationTest()

        // ========== Compiled generation test ==========
        try await runCompiledBenchmark(container: container)

        // ========== Gemma 3 1B comparison ==========
        print("\n=== Gemma 3 1B Cache Comparison ===")
        let g3Dir = "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-3-1b-it-qat-4bit/snapshots/15fed4eafb456c6fcb2a1165f19ac609670ed14b"
        if FileManager.default.fileExists(atPath: g3Dir) {
            let g3Container = try await LLMModelFactory.shared.loadContainer(
                from: URL(fileURLWithPath: g3Dir), using: HFTokenizerLoader())

            try await g3Container.perform { (ctx: ModelContext) in
                let model = ctx.model

                // Prefill + warmup
                let p = MLXArray([2, 1596, 603])[.newAxis, .ellipsis]
                let _ = model(p, cache: model.newCache(parameters: nil))
                let cache3 = model.newCache(parameters: nil)
                let _ = model(p, cache: cache3)
                MLX.eval(cache3.flatMap { $0.state })

                var last3 = MLXArray([235265])[.newAxis, .ellipsis]
                for _ in 0..<4 {
                    let out = model(last3, cache: cache3)
                    last3 = out.argMax(axis: -1)
                    MLX.eval(last3)
                }

                MLX.Memory.clearCache()

                let out = model(last3, cache: cache3)
                last3 = out.argMax(axis: -1)
                MLX.eval(last3)

                let g3Cache = MLX.Memory.snapshot().cacheMemory
                print("Gemma 3 1B Swift: cache = \(g3Cache / 1_000_000) MB per token (Python: 0.9 MB)")
            }
        }
    }
}
