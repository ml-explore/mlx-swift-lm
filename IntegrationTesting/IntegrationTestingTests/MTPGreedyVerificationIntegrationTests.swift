// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import Testing
import Tokenizers

@_spi(Testing) @testable import MLXLMCommon

private let greedyMTPModelDirectory = hfSnapshotDir(
    modelId: "mlx-community/Qwen3.5-4B-4bit",
    revision: "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c")
private let greedyMTPDrafterDirectory = hfSnapshotDir(
    modelId: "mlx-community/Qwen3.5-4B-MTP-4bit",
    revision: "ab6f59bc6627196c611ab8851638651078170485")

private struct SequentialMTPProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {}
}

@Suite(.serialized)
struct MTPGreedyVerificationIntegrationTests {
    @Test(.enabled(if: greedyMTPModelDirectory != nil && greedyMTPDrafterDirectory != nil))
    func tokenAndCacheParity() async throws {
        let targetDirectory = try #require(greedyMTPModelDirectory)
        let drafterDirectory = try #require(greedyMTPDrafterDirectory)
        let tokenizerLoader = #huggingFaceTokenizerLoader()
        let target = try await LLMModelFactory.shared.load(
            from: targetDirectory, using: tokenizerLoader)
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: drafterDirectory, using: tokenizerLoader)
        let input = try await target.processor.prepare(
            input: UserInput(chat: [
                .user("Why is the sky blue? Explain the physics, then give a simple analogy.")
            ]))
        let parameters = GenerateParameters(maxTokens: 128, temperature: 0)

        func run(batched: Bool, stopAfter: Int) throws -> (
            tokens: [Int], caches: [KVCache], position: Int, accepted: Int, proposed: Int,
            seconds: Double
        ) {
            let caches = try target.model.newCache(parameters: parameters)
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target.model, drafter: drafter.model,
                mainCache: caches, parameters: parameters, blockSize: 2)
            #expect(
                iterator._processorForTesting == nil
                    || iterator._processorForTesting is SuppressTokensProcessor)
            if !batched {
                let processors = iterator._processorForTesting.map { [$0] } ?? []
                iterator._setProcessorForTesting(
                    ChainedLogitProcessor(processors: processors + [SequentialMTPProcessor()]))
            }
            Stream.defaultStream(.gpu).synchronize()
            let start = ContinuousClock.now
            var tokens = [Int]()
            while tokens.count < stopAfter, let token = iterator.next() { tokens.append(token) }
            iterator.finalizeGeneration()
            let finalCaches = iterator.mainCache + (iterator.drafterState?.cache ?? [])
            eval(finalCaches.flatMap { $0.state })
            let elapsed = (ContinuousClock.now - start).components
            #expect(iterator.passthroughReason == nil)
            return (
                tokens, finalCaches, iterator.mainCacheStorage.processedTokenCount,
                iterator.acceptedCount, iterator.proposedCount,
                Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            )
        }

        func expectCacheParity(_ actual: [KVCache], _ expected: [KVCache]) {
            #expect(actual.count == expected.count)
            for (actualCache, expectedCache) in zip(actual, expected) {
                #expect(actualCache.offset == expectedCache.offset)
                #expect(actualCache.state.count == expectedCache.state.count)
                for (a, b) in zip(actualCache.state, expectedCache.state) {
                    #expect(a.shape == b.shape)
                    #expect(a.dtype == b.dtype)
                    #expect(a.asArray(Float.self) == b.asArray(Float.self))
                }
            }
        }

        var expectedTokens = [Int]()
        var expectedAccepted = 0
        var expectedProposed = 0
        for stopAfter in [5, 13, 128] {
            let sequential = try run(batched: false, stopAfter: stopAfter)
            let batched = try run(batched: true, stopAfter: stopAfter)
            #expect(batched.tokens == sequential.tokens)
            #expect(batched.tokens.count == stopAfter)
            #expect(batched.accepted == sequential.accepted)
            #expect(batched.proposed == sequential.proposed)
            #expect(batched.position == sequential.position)
            #expect(batched.proposed > 0)
            expectCacheParity(batched.caches, sequential.caches)
            if stopAfter == 128 {
                expectedTokens = sequential.tokens
                expectedAccepted = sequential.accepted
                expectedProposed = sequential.proposed
            }
            print(
                "[MTPGREEDYPARITY] tokens=\(stopAfter) accepted=\(batched.accepted)/\(batched.proposed) cache=exact"
            )
        }

        guard ProcessInfo.processInfo.environment["MTP_GREEDY_BENCHMARK"] == "1" else { return }
        var sequentialTimes = [Double]()
        var batchedTimes = [Double]()
        for repetition in 0 ..< 5 {
            for batched in repetition.isMultiple(of: 2) ? [false, true] : [true, false] {
                let result = try run(batched: batched, stopAfter: 128)
                #expect(result.tokens == expectedTokens)
                #expect(result.accepted == expectedAccepted)
                #expect(result.proposed == expectedProposed)
                if batched {
                    batchedTimes.append(result.seconds)
                } else {
                    sequentialTimes.append(result.seconds)
                }
                print(
                    "[MTPGREEDYRUN] batched=\(batched) seconds=\(result.seconds) tokens=\(result.tokens.count)"
                )
            }
        }
        let sequential = sequentialTimes.sorted()[2]
        let batched = batchedTimes.sorted()[2]
        print(
            String(
                format:
                    "[MTPGREEDYE2E] Qwen3.5-4B-4bit block=2 tokens=128 sequential=%.2f tok/s batched=%.2f tok/s speedup=%.3fx",
                128 / sequential, 128 / batched, sequential / batched))
    }
}
