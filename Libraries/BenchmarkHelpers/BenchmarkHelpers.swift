// Shared benchmark logic for measuring model loading and download performance.
// Integration packages inject their own Downloader and TokenizerLoader.

import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

// MARK: - No-Op Tokenizer

/// A tokenizer loader that returns a stub tokenizer. Useful for benchmarking
/// model loading in downloader integration packages that don't provide a
/// real tokenizer.
public struct NoOpTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any Tokenizer {
        NoOpTokenizer()
    }
}

private struct NoOpTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
}

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
        print("  Mean:   \(String(format: "%.1f", mean))ms")
        print("  Median: \(String(format: "%.1f", median))ms")
        print("  StdDev: \(String(format: "%.1f", stdDev))ms")
        print("  Range:  \(String(format: "%.1f", min))-\(String(format: "%.1f", max))ms")
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
        print("LLM load run \(i): \(String(format: "%.1f", elapsed))ms")
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
        print("VLM load run \(i): \(String(format: "%.1f", elapsed))ms")
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
        print("Embedding load run \(i): \(String(format: "%.1f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

// MARK: - Download Benchmarks

/// Benchmark download cache hit performance. Ensures the model is cached with a warm-up
/// download, then measures repeated cache lookups.
public func benchmarkDownloadCacheHit(
    from downloader: any Downloader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    runs: Int = 7
) async throws -> BenchmarkStats {
    let patterns = ["*.safetensors", "*.json", "*.jinja"]

    // Warm-up: ensure the model is cached
    _ = try await downloader.download(
        id: modelId, revision: "main", matching: patterns,
        useLatest: false, progressHandler: { _ in })

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await downloader.download(
            id: modelId, revision: "main", matching: patterns,
            useLatest: false, progressHandler: { _ in })
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Download cache hit run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}
