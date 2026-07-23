// Synthetic streaming-detokenization benchmarks. These isolate tokenizer
// decode work from model inference and require no network or model weights.

import Foundation
import MLXLMCommon

public enum DetokenizerBenchmarkStrategy: Sendable, Hashable {
    /// Preserve the 3.x behavior: retain every token since the last newline.
    case unbounded

    /// Advertise a bounded decoder suffix context to the streaming detokenizer.
    case bounded(contextTokens: Int)
}

public struct DetokenizerBenchmarkResult: Sendable {
    public let tokenCount: Int
    public let newlineEvery: Int?
    public let strategy: DetokenizerBenchmarkStrategy
    public let decodedTokenVisits: Int
    public let emittedCharacterCount: Int
    public let stats: BenchmarkStats

    public init(
        tokenCount: Int,
        newlineEvery: Int?,
        strategy: DetokenizerBenchmarkStrategy,
        decodedTokenVisits: Int,
        emittedCharacterCount: Int,
        stats: BenchmarkStats
    ) {
        self.tokenCount = tokenCount
        self.newlineEvery = newlineEvery
        self.strategy = strategy
        self.decodedTokenVisits = decodedTokenVisits
        self.emittedCharacterCount = emittedCharacterCount
        self.stats = stats
    }

    public var decodedTokenVisitsPerInputToken: Double {
        Double(decodedTokenVisits) / Double(tokenCount)
    }

    public func printSummary() {
        let cadence = newlineEvery.map(String.init) ?? "none"
        let strategyName: String
        switch strategy {
        case .unbounded:
            strategyName = "unbounded"
        case .bounded(let contextTokens):
            strategyName = "bounded(\(contextTokens))"
        }

        print(
            "detokenizer tokens=\(tokenCount) newlineEvery=\(cadence) "
                + "strategy=\(strategyName): median "
                + "\(String(format: "%.3f", stats.median))ms, "
                + "decodeVisits=\(decodedTokenVisits), "
                + "visits/token=\(String(format: "%.2f", decodedTokenVisitsPerInputToken))"
        )
    }
}

private final class DetokenizerDecodeWorkCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func record(_ count: Int) {
        lock.lock()
        value += count
        lock.unlock()
    }

    func read() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private struct SyntheticLinearTokenizer: Tokenizer {
    let counter: DetokenizerDecodeWorkCounter

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        counter.record(tokenIds.count)
        return tokenIds.map { $0 == 0 ? "\n" : "x" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { id == 0 ? "\n" : "x" }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

private struct SyntheticBoundedLinearTokenizer: BoundedStreamingDecodeTokenizer {
    let base: SyntheticLinearTokenizer
    let contextSize: Int

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }
    var streamingDecodeContextSize: Int? { contextSize }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

private func syntheticDetokenizerTokens(count: Int, newlineEvery: Int?) -> [Int] {
    (1 ... count).map { index in
        if let newlineEvery, index.isMultiple(of: newlineEvery) {
            return 0
        }
        return 1
    }
}

private func runDetokenizerTrial(
    tokens: [Int], strategy: DetokenizerBenchmarkStrategy
) -> (elapsedMilliseconds: Double, decodedTokenVisits: Int, emittedCharacterCount: Int) {
    let counter = DetokenizerDecodeWorkCounter()
    let baseTokenizer = SyntheticLinearTokenizer(counter: counter)
    let tokenizer: any Tokenizer
    switch strategy {
    case .unbounded:
        tokenizer = baseTokenizer
    case .bounded(let contextTokens):
        precondition(contextTokens > 0, "contextTokens must be greater than zero")
        tokenizer = SyntheticBoundedLinearTokenizer(
            base: baseTokenizer, contextSize: contextTokens)
    }

    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    var emittedCharacterCount = 0

    let start = CFAbsoluteTimeGetCurrent()
    for token in tokens {
        detokenizer.append(token: token)
        emittedCharacterCount += detokenizer.next()?.count ?? 0
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1_000

    return (elapsed, counter.read(), emittedCharacterCount)
}

/// Measure streaming detokenization with deterministic synthetic token streams.
///
/// The synthetic tokenizer performs linear work for every token ID passed to
/// `decode`, and `decodedTokenVisits` reports that work independently of noisy
/// wall-clock timings. Compare `newlineEvery: nil` with bounded newline
/// cadences to reproduce the historical quadratic worst case, then compare the
/// `.bounded` strategy to measure the bounded-context fast path.
public func benchmarkStreamingDetokenization(
    tokenCounts: [Int] = [256, 1_024, 4_096],
    newlineCadences: [Int?] = [16, 128, nil],
    strategies: [DetokenizerBenchmarkStrategy] = [
        .unbounded, .bounded(contextTokens: 16),
    ],
    runs: Int = 7,
    warmupRuns: Int = 1
) -> [DetokenizerBenchmarkResult] {
    precondition(!tokenCounts.isEmpty, "tokenCounts must not be empty")
    precondition(tokenCounts.allSatisfy { $0 > 0 }, "tokenCounts must be greater than zero")
    precondition(
        newlineCadences.allSatisfy { $0 == nil || $0! > 0 },
        "newline cadences must be greater than zero")
    precondition(!strategies.isEmpty, "strategies must not be empty")
    precondition(runs > 0, "runs must be greater than zero")
    precondition(warmupRuns >= 0, "warmupRuns must not be negative")

    var results: [DetokenizerBenchmarkResult] = []

    for strategy in strategies {
        for newlineEvery in newlineCadences {
            for tokenCount in tokenCounts {
                let tokens = syntheticDetokenizerTokens(
                    count: tokenCount, newlineEvery: newlineEvery)

                for _ in 0 ..< warmupRuns {
                    _ = runDetokenizerTrial(tokens: tokens, strategy: strategy)
                }

                var times: [Double] = []
                times.reserveCapacity(runs)
                var decodedTokenVisits = 0
                var emittedCharacterCount = 0

                for _ in 0 ..< runs {
                    let trial = runDetokenizerTrial(tokens: tokens, strategy: strategy)
                    times.append(trial.elapsedMilliseconds)
                    decodedTokenVisits = trial.decodedTokenVisits
                    emittedCharacterCount = trial.emittedCharacterCount
                }

                let result = DetokenizerBenchmarkResult(
                    tokenCount: tokenCount,
                    newlineEvery: newlineEvery,
                    strategy: strategy,
                    decodedTokenVisits: decodedTokenVisits,
                    emittedCharacterCount: emittedCharacterCount,
                    stats: BenchmarkStats(times: times)
                )
                result.printSummary()
                results.append(result)
            }
        }
    }

    return results
}
