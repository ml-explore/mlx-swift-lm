import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

private let benchmarksEnabled = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil

@Suite(.serialized)
struct ModelLoadingBenchmarks {

    /// Benchmark LLM model loading
    /// Tests: parallel tokenizer/weights, single config.json read
    @Test(.enabled(if: benchmarksEnabled))
    func loadLLM() async throws {
        let modelId = "mlx-community/Qwen3-0.6B-4bit"
        let hub = HubApi()
        let config = ModelConfiguration(id: modelId)

        // Warm up: ensure model is downloaded
        _ = try await LLMModelFactory.shared.load(hub: hub, configuration: config) { _ in }

        // Benchmark multiple runs
        let runs = 5
        var times: [Double] = []

        for i in 1 ... runs {
            let start = CFAbsoluteTimeGetCurrent()

            let _ = try await LLMModelFactory.shared.load(
                hub: hub,
                configuration: config
            ) { _ in }

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("LLM load run \(i): \(String(format: "%.0f", elapsed))ms")

            // Clear GPU cache to ensure independent measurements
            GPU.clearCache()
        }

        let avg = times.reduce(0, +) / Double(times.count)
        print("LLM load average: \(String(format: "%.0f", avg))ms")
    }

    /// Benchmark VLM model loading
    /// Tests: parallel tokenizer/weights, single config.json read, parallel processor config
    @Test(.enabled(if: benchmarksEnabled))
    func loadVLM() async throws {
        let modelId = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        let hub = HubApi()
        let config = ModelConfiguration(id: modelId)

        // Warm up: ensure model is downloaded
        _ = try await VLMModelFactory.shared.load(hub: hub, configuration: config) { _ in }

        // Benchmark multiple runs
        let runs = 5
        var times: [Double] = []

        for i in 1 ... runs {
            let start = CFAbsoluteTimeGetCurrent()

            let _ = try await VLMModelFactory.shared.load(
                hub: hub,
                configuration: config
            ) { _ in }

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("VLM load run \(i): \(String(format: "%.0f", elapsed))ms")

            // Clear GPU cache to ensure independent measurements
            GPU.clearCache()
        }

        let avg = times.reduce(0, +) / Double(times.count)
        print("VLM load average: \(String(format: "%.0f", avg))ms")
    }
}
