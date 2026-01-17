import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

private let benchmarksEnabled = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil

private struct BenchmarkStats {
    let mean: Double
    let median: Double
    let stdDev: Double
    let min: Double
    let max: Double

    init(times: [Double]) {
        precondition(!times.isEmpty, "BenchmarkStats requires at least one timing measurement")
        let sorted = times.sorted()
        self.min = sorted.first ?? 0
        self.max = sorted.last ?? 0
        let mean = times.reduce(0, +) / Double(times.count)
        self.mean = mean
        self.median = sorted[sorted.count / 2]

        let squaredDiffs = times.map { ($0 - mean) * ($0 - mean) }
        self.stdDev = sqrt(squaredDiffs.reduce(0, +) / Double(times.count))
    }

    func printSummary(label: String) {
        print("\(label) results:")
        print("  Mean:   \(String(format: "%.0f", mean))ms")
        print("  Median: \(String(format: "%.0f", median))ms")
        print("  StdDev: \(String(format: "%.1f", stdDev))ms")
        print("  Range:  \(String(format: "%.0f", min))-\(String(format: "%.0f", max))ms")
    }
}

@Suite(.serialized)
struct ModelLoadingBenchmarks {

    /// Benchmark LLM model loading
    /// Tests: parallel tokenizer/weights, single config.json read
    @Test(.enabled(if: benchmarksEnabled))
    func loadLLM() async throws {
        let modelId = "mlx-community/Qwen3-0.6B-4bit"
        let hub = HubApi()
        let config = ModelConfiguration(id: modelId)

        // Warm-up run: ensure model is downloaded and caches are primed
        _ = try await LLMModelFactory.shared.load(hub: hub, configuration: config) { _ in }
        Memory.clearCache()

        // Benchmark multiple runs
        let runs = 7
        var times: [Double] = []

        for i in 1 ... runs {
            let start = CFAbsoluteTimeGetCurrent()

            _ = try await LLMModelFactory.shared.load(
                hub: hub,
                configuration: config
            ) { _ in }

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("LLM load run \(i): \(String(format: "%.0f", elapsed))ms")

            // Clear GPU cache to ensure independent measurements
            Memory.clearCache()
        }

        BenchmarkStats(times: times).printSummary(label: "LLM load")
    }

    /// Benchmark VLM model loading
    /// Tests: parallel tokenizer/weights, single config.json read, parallel processor config
    @Test(.enabled(if: benchmarksEnabled))
    func loadVLM() async throws {
        let modelId = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        let hub = HubApi()
        let config = ModelConfiguration(id: modelId)

        // Warm-up run: ensure model is downloaded and caches are primed
        _ = try await VLMModelFactory.shared.load(hub: hub, configuration: config) { _ in }
        Memory.clearCache()

        // Benchmark multiple runs
        let runs = 7
        var times: [Double] = []

        for i in 1 ... runs {
            let start = CFAbsoluteTimeGetCurrent()

            _ = try await VLMModelFactory.shared.load(
                hub: hub,
                configuration: config
            ) { _ in }

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("VLM load run \(i): \(String(format: "%.0f", elapsed))ms")

            // Clear GPU cache to ensure independent measurements
            Memory.clearCache()
        }

        BenchmarkStats(times: times).printSummary(label: "VLM load")
    }
}
