// Shared benchmark logic for measuring model loading performance.
// Integration packages inject their own Downloader and TokenizerLoader.

import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

// MARK: - Stats

public struct BenchmarkStats: Sendable {
    public let mean: Double
    public let median: Double
    public let stdDev: Double
    public let min: Double
    public let max: Double

    public init(times: [Double]) {
        precondition(!times.isEmpty, "BenchmarkStats requires at least one timing measurement")
        let sorted = times.sorted()
        self.min = sorted.first!
        self.max = sorted.last!
        let mean = times.reduce(0, +) / Double(times.count)
        self.mean = mean
        self.median = sorted[sorted.count / 2]

        let squaredDiffs = times.map { ($0 - mean) * ($0 - mean) }
        self.stdDev = sqrt(squaredDiffs.reduce(0, +) / Double(times.count))
    }

    public func printSummary(label: String) {
        print("\(label) results:")
        print("  Mean:   \(String(format: "%.0f", mean))ms")
        print("  Median: \(String(format: "%.0f", median))ms")
        print("  StdDev: \(String(format: "%.1f", stdDev))ms")
        print("  Range:  \(String(format: "%.0f", min))-\(String(format: "%.0f", max))ms")
    }
}

// MARK: - Benchmark Runners

/// Benchmark LLM model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkLLMLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    runs: Int = 7
) async throws -> BenchmarkStats {
    let config = MLXLMCommon.ModelConfiguration(id: modelId)

    _ = try await LLMModelFactory.shared.load(
        from: downloader, using: tokenizerLoader, configuration: config
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await LLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader, configuration: config
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("LLM load run \(i): \(String(format: "%.0f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

/// Benchmark VLM model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkVLMLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    runs: Int = 7
) async throws -> BenchmarkStats {
    let config = MLXLMCommon.ModelConfiguration(id: modelId)

    _ = try await VLMModelFactory.shared.load(
        from: downloader, using: tokenizerLoader, configuration: config
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await VLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader, configuration: config
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("VLM load run \(i): \(String(format: "%.0f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

/// Benchmark embedding model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkEmbeddingLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXEmbedders.ModelConfiguration = .nomic_text_v1_5,
    runs: Int = 7
) async throws -> BenchmarkStats {
    _ = try await MLXEmbedders.loadModelContainer(
        from: downloader, using: tokenizerLoader, configuration: configuration
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await MLXEmbedders.loadModelContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Embedding load run \(i): \(String(format: "%.0f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}
