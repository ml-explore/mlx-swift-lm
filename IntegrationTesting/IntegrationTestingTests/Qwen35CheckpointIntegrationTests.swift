// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

struct QwenMTPReference: Sendable {
    let id: String
    let revision: String
    let bits: Int
}

private let qwenMTPReferences = [
    QwenMTPReference(
        id: "mlx-community/Qwen3.5-4B-MTP-4bit",
        revision: "ab6f59bc6627196c611ab8851638651078170485", bits: 4),
    QwenMTPReference(
        id: "mlx-community/Qwen3.5-4B-MTP-5bit",
        revision: "aee6a907373f1b758cc8c406da663cf85eb01ed3", bits: 5),
    QwenMTPReference(
        id: "mlx-community/Qwen3.5-9B-MTP-4bit",
        revision: "222dfd2c23fc9518d7b817e4f8e0cb0571787489", bits: 4),
]

@Suite(.serialized)
struct Qwen35CheckpointIntegrationTests {
    private let downloader = #hubDownloader()
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    @Test(arguments: qwenMTPReferences)
    func publishedStandaloneHeadsLoad(reference: QwenMTPReference) async throws {
        let drafter = try await loadDrafter(reference)
        #expect(drafter.model.parameters().flattened().count == 31)
        let projections = drafter.model.modules().compactMap { $0 as? QuantizedLinear }
        #expect(projections.count == 8)
        #expect(projections.allSatisfy { $0.bits == reference.bits && $0.groupSize == 64 })
    }

    @Test
    func standaloneHeadProducesAcceptedDraftsAndMatchesGreedyTarget() async throws {
        let target = try await LLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader,
            configuration: .init(
                id: "mlx-community/Qwen3.5-4B-4bit",
                revision: "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"))
        let drafter = try await loadDrafter(qwenMTPReferences[0])
        let input = try await target.processor.prepare(
            input: UserInput(chat: [
                .user("Why is the sky blue? Explain briefly.")
            ]))
        let parameters = GenerateParameters(maxTokens: 64, temperature: 0)
        let baseline = try generateTokens(input: input, parameters: parameters, context: target)
        let baselineResult = await collect(baseline)
        let speculative = try generateTokens(
            input: input, parameters: parameters, context: target,
            mtpDrafter: drafter.model, blockSize: 2)
        let result = await collect(speculative)
        let info = try #require(result.info)
        #expect(!result.tokens.isEmpty)
        #expect(result.tokens == baselineResult.tokens)
        #expect((info.proposedDraftTokens ?? 0) > 0)
        #expect((info.acceptedDraftTokens ?? 0) > 0)
        #expect(info.passthroughReason == nil)
    }

    private func loadDrafter(_ reference: QwenMTPReference) async throws -> MTPDrafterContext {
        await Qwen35TextMTPRegistration.register()
        return try await MTPDrafterModelFactory.shared.load(
            from: downloader, using: tokenizerLoader,
            configuration: .init(id: reference.id, revision: reference.revision))
    }

    private func collect(_ stream: AsyncStream<TokenGeneration>) async -> (
        tokens: [Int], info: GenerateCompletionInfo?
    ) {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let completion): info = completion
            }
        }
        return (tokens, info)
    }
}
