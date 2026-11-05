// Copyright © 2026 Apple Inc.

// Greedy decode throughput of `TokenIterator` on a local model, as a median over rounds.
// Decode is timed after the prefill and the first token. Tokens are checked across rounds.
//
//   swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
//       [decode tokens: 128] [rounds: 5] [prompt tokens: 145]

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print("usage: decodebench <model directory> [decode tokens] [rounds] [prompt tokens]")
    exit(1)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let decodeTokens = arguments.count > 2 ? Int(arguments[2]) ?? 128 : 128
let rounds = arguments.count > 3 ? Int(arguments[3]) ?? 5 : 5
let promptLength = arguments.count > 4 ? Int(arguments[4]) ?? 145 : 145

let context = try await LLMModelFactory.shared.load(
    from: modelDirectory, using: NoOpTokenizerLoader())
let model = context.model
print("model: \(modelDirectory.lastPathComponent)")
print("prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds)")

/// A fixed pseudo-random prompt, so every round and every checkout decodes the same tokens.
@MainActor func prompt(seed: UInt64) -> [Int32] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0 ..< promptLength).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int32(1000 + (state >> 33) % 20000)
    }
}

func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

/// Prefills the prompt, then times `decodeTokens` greedy steps of `TokenIterator`.
@MainActor func runSingle(prompt: [Int32]) throws -> (tokensPerSecond: Double, tokens: [Int]) {
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)), model: model, processor: nil,
        sampler: ArgMaxSampler(), maxTokens: decodeTokens + 1)
    var tokens: [Int] = []
    // The first token comes from the prefill; the steps after it are the decode.
    if let first = iterator.next() { tokens.append(first) }
    let start = ContinuousClock.now
    while let token = iterator.next() { tokens.append(token) }
    return (Double(decodeTokens) / seconds(since: start), tokens)
}

let singlePrompt = prompt(seed: 1)
_ = try runSingle(prompt: singlePrompt)  // warm up: kernel compilation is not timed

var single: [Double] = []
var reference: [Int]? = nil
var mismatchedRounds: [Int] = []
for round in 0 ..< rounds {
    let result = try runSingle(prompt: singlePrompt)
    single.append(result.tokensPerSecond)
    if reference == nil { reference = result.tokens }
    if result.tokens != reference { mismatchedRounds.append(round + 1) }
    print(String(format: "round %d  single %.2f tok/s", round + 1, result.tokensPerSecond))
    Memory.clearCache()
}
print(String(format: "median  single %.2f tok/s", median(single)))
print(
    "tokens identical in every round: \(mismatchedRounds.isEmpty)"
        + (mismatchedRounds.isEmpty ? "" : ", not in rounds \(mismatchedRounds)"))
