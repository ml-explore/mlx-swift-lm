// Shared benchmark logic for measuring model loading, tokenizer performance,
// and download performance.
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

// MARK: - Benchmark Text

public enum BenchmarkDefaults {
    public static let textSource = BenchmarkTextSource.prideAndPrejudice
    public static let tokenizationTextCharacterCount = 20_000
    public static let decodingTextCharacterCount = 200_000
    public static let loadingRuns = 7
    public static let downloadRuns = 7
    public static let tokenizationRuns = 25
    public static let decodingRuns = 25
    public static let decodesPerRun = 10
}

public struct BenchmarkTextSource: Sendable {
    public let name: String
    public let url: URL
    public let contentStartMarker: String?

    public init(name: String, url: URL, contentStartMarker: String? = nil) {
        self.name = name
        self.url = url
        self.contentStartMarker = contentStartMarker
    }

    public static let prideAndPrejudice = BenchmarkTextSource(
        name: "pride-and-prejudice",
        url: URL(string: "https://www.gutenberg.org/ebooks/1342.txt.utf-8")!,
        contentStartMarker: "It is a truth universally acknowledged"
    )
}

public enum BenchmarkTextError: LocalizedError {
    case invalidResponse(URL)
    case decodeFailed(URL)
    case contentStartMarkerNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse(let url):
            return "Unexpected response while fetching benchmark text from \(url.absoluteString)."
        case .decodeFailed(let url):
            return "Failed to decode benchmark text from \(url.absoluteString) as UTF-8."
        case .contentStartMarkerNotFound(let marker):
            return "Benchmark text start marker not found: '\(marker)'."
        }
    }
}

private func benchmarkTextCacheURL(for source: BenchmarkTextSource) -> URL {
    FileManager.default.temporaryDirectory
        .appending(component: "BenchmarkHelpers", directoryHint: .isDirectory)
        .appending(component: "\(source.name).txt")
}

private func normalizeBenchmarkText(_ text: String) -> String {
    text.replacingOccurrences(of: "\r\n", with: "\n")
        .replacingOccurrences(of: "\r", with: "\n")
}

private func trimmedBenchmarkText(_ text: String, source: BenchmarkTextSource) throws -> String {
    guard let marker = source.contentStartMarker else {
        return text
    }
    guard let markerRange = text.range(of: marker) else {
        throw BenchmarkTextError.contentStartMarkerNotFound(marker)
    }
    return String(text[markerRange.lowerBound...])
}

private func fetchBenchmarkText(source: BenchmarkTextSource) async throws -> String {
    let cacheURL = benchmarkTextCacheURL(for: source)
    let fileManager = FileManager.default

    if fileManager.fileExists(atPath: cacheURL.path) {
        let cached = try String(contentsOf: cacheURL, encoding: .utf8)
        return try trimmedBenchmarkText(normalizeBenchmarkText(cached), source: source)
    }

    let (data, response) = try await URLSession.shared.data(from: source.url)
    guard let httpResponse = response as? HTTPURLResponse,
        (200 ..< 300).contains(httpResponse.statusCode)
    else {
        throw BenchmarkTextError.invalidResponse(source.url)
    }
    guard let downloaded = String(data: data, encoding: .utf8) else {
        throw BenchmarkTextError.decodeFailed(source.url)
    }

    try fileManager.createDirectory(
        at: cacheURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try downloaded.write(to: cacheURL, atomically: true, encoding: .utf8)

    return try trimmedBenchmarkText(normalizeBenchmarkText(downloaded), source: source)
}

/// Load benchmark text from a remote public-domain source and cache it locally in the temporary directory.
public func loadBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource,
    characterCount: Int = 20_000
) async throws -> String {
    precondition(characterCount > 0, "characterCount must be greater than zero")
    let text = try await fetchBenchmarkText(source: source)
    return String(text.prefix(characterCount))
}

public func loadTokenizationBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource
) async throws -> String {
    try await loadBenchmarkText(
        source: source,
        characterCount: BenchmarkDefaults.tokenizationTextCharacterCount
    )
}

public func loadDecodingBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource
) async throws -> String {
    try await loadBenchmarkText(
        source: source,
        characterCount: BenchmarkDefaults.decodingTextCharacterCount
    )
}

private func resolveTokenizerDirectory(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration,
    useLatest: Bool
) async throws -> URL {
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        return try await downloader.download(
            id: id,
            revision: revision,
            matching: tokenizerDownloadPatterns,
            useLatest: useLatest,
            progressHandler: { _ in }
        )
    case .directory(let directory):
        return directory
    case nil:
        switch configuration.id {
        case .id(let id, let revision):
            return try await downloader.download(
                id: id,
                revision: revision,
                matching: tokenizerDownloadPatterns,
                useLatest: useLatest,
                progressHandler: { _ in }
            )
        case .directory(let directory):
            return directory
        }
    }
}

private func loadTokenizerForBenchmark(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration,
    useLatest: Bool
) async throws -> any Tokenizer {
    let tokenizerDirectory = try await resolveTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )
    return try await tokenizerLoader.load(from: tokenizerDirectory)
}

// MARK: - Benchmark Runners

/// Benchmark tokenizer loading without downloading model weights or initializing a model.
public func benchmarkTokenizerLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    let tokenizerDirectory = try await resolveTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )

    _ = try await tokenizerLoader.load(from: tokenizerDirectory)

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await tokenizerLoader.load(from: tokenizerDirectory)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Tokenizer load run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}

/// Benchmark tokenization on a preloaded tokenizer without initializing a model.
public func benchmarkTokenization(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    text: String = "The quick brown fox jumps over the lazy dog.",
    addSpecialTokens: Bool = true,
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.tokenizationRuns
) async throws -> BenchmarkStats {
    let tokenizer = try await loadTokenizerForBenchmark(
        from: downloader,
        using: tokenizerLoader,
        configuration: configuration,
        useLatest: useLatest
    )

    _ = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print(
            "Tokenization run \(i): \(String(format: "%.1f", elapsed))ms (\(tokenIds.count) tokens)"
        )
    }

    return BenchmarkStats(times: times)
}

/// Benchmark decoding on a preloaded tokenizer without initializing a model.
public func benchmarkDecoding(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    text: String = "The quick brown fox jumps over the lazy dog.",
    addSpecialTokens: Bool = true,
    skipSpecialTokens: Bool = false,
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.decodingRuns,
    decodesPerRun: Int = BenchmarkDefaults.decodesPerRun
) async throws -> BenchmarkStats {
    precondition(decodesPerRun > 0, "decodesPerRun must be greater than zero")

    let tokenizer = try await loadTokenizerForBenchmark(
        from: downloader,
        using: tokenizerLoader,
        configuration: configuration,
        useLatest: useLatest
    )
    let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)

    _ = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)

    var times: [Double] = []
    for i in 1 ... runs {
        var decoded = ""
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< decodesPerRun {
            decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(decodesPerRun)
        times.append(elapsed)
        print(
            "Decoding run \(i): \(String(format: "%.1f", elapsed))ms avg over \(decodesPerRun)x "
                + "(\(decoded.count) chars)"
        )
    }

    return BenchmarkStats(times: times)
}

/// Benchmark LLM model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkLLMLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    runs: Int = BenchmarkDefaults.loadingRuns
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
    runs: Int = BenchmarkDefaults.loadingRuns
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
    configuration: ModelConfiguration = .init(
        id: "mlx-community/Qwen3-Embedding-0.6B-8bit"),
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    _ = try await EmbedderModelFactory.shared.loadContainer(
        from: downloader, using: tokenizerLoader, configuration: configuration
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await EmbedderModelFactory.shared.loadContainer(
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
    runs: Int = BenchmarkDefaults.downloadRuns
) async throws -> BenchmarkStats {
    let patterns = modelDownloadPatterns

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
