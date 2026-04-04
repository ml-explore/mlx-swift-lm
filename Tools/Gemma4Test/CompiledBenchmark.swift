// Test if compile() eliminates cache churn by bypassing Swift ARC intermediates

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Wraps KV caches as Updatable state for compile()
struct CacheState: Updatable {
    let caches: [KVCache]

    func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }
}

/// Creates a compiled generation step function
func makeCompiledStep(
    model: any LanguageModel,
    caches: [KVCache]
) -> @Sendable (MLXArray) -> MLXArray {
    let cacheState = CacheState(caches: caches)

    return compile(
        inputs: [model, cacheState],
        outputs: [model, cacheState],
        shapeless: false
    ) { (input: MLXArray) -> MLXArray in
        let logits = model(input[.newAxis, .ellipsis], cache: caches)
        return logits.argMax(axis: -1)
    }
}

func runCompiledBenchmark(container: ModelContainer) async throws {
    print("\n=== Compiled Generation Benchmark ===")

    try await container.perform { (ctx: ModelContext) in
        let model = ctx.model
        guard let lm = model as? Gemma4TextModel else {
            print("ERROR: Not Gemma4TextModel"); return
        }
        let cache = lm.newCache()

        // Prefill
        let promptTokens = MLXArray([2, 1596, 603, 476, 2195])[.newAxis, .ellipsis]
        let _ = model(promptTokens, cache: cache)
        MLX.eval(cache.flatMap { $0.innerState() })

        // Get first token
        var lastToken = MLXArray([235265])
        let logits0 = model(lastToken[.newAxis, .ellipsis], cache: cache)
        lastToken = logits0.argMax(axis: -1).reshaped(-1)
        MLX.eval(lastToken)

        // Create compiled step function
        print("Compiling generation step...")
        let compiledStep = makeCompiledStep(model: model, caches: cache)

        // Warmup the compiled function (first call traces the graph)
        lastToken = compiledStep(lastToken)
        MLX.eval(lastToken)
        lastToken = lastToken.reshaped(-1)

        MLX.Memory.clearCache()

        // Measure 1-token cache with compiled function
        lastToken = compiledStep(lastToken)
        MLX.eval(lastToken)
        lastToken = lastToken.reshaped(-1)

        let mem = MLX.Memory.snapshot()
        print("Compiled 1-token cache: \(mem.cacheMemory / 1_000_000) MB (uncompiled: 237 MB, Python: 2.3 MB)")

        // Benchmark 10 tokens
        MLX.Memory.clearCache()
        var times: [Double] = []
        for _ in 0..<10 {
            let start = CFAbsoluteTimeGetCurrent()
            lastToken = compiledStep(lastToken)
            MLX.eval(lastToken)
            lastToken = lastToken.reshaped(-1)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        let avg = times.reduce(0, +) / Double(times.count)
        print("Compiled per-token: \(String(format: "%.2f", avg * 1000)) ms (\(String(format: "%.1f", 1.0/avg)) tok/s)")
        print("Cache after 10 compiled tokens: \(MLX.Memory.snapshot().cacheMemory / 1_000_000) MB")
        print("Uncompiled baseline: 16.4 ms (61 tok/s), Python: 14.6 ms (68 tok/s)")
    }
}
