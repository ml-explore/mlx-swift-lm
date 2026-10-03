// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ChatSessionQwenTextMTPTests {
    @Test(
        "MTP applies tool validation and authorization before automatic dispatch",
        arguments: ToolCallValidationPolicy.allCases, [true, false])
    func mtpValidatesToolsBeforeDispatch(
        validation: ToolCallValidationPolicy, declaresTool: Bool
    ) async throws {
        let probe = ToolRestartProbe()
        let tokenizer = ToolRestartMTPTokenizer(probe: probe)
        let processor = ToolRestartMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer)
        let restrictedTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string", "enum": ["Rome"]] as [String: any Sendable]
                    ],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(
                maxTokens: 24, temperature: 0,
                toolCallPolicy: .init(recovery: .disabled, validation: validation)),
            tools: declaresTool ? [restrictedTool] : nil,
            toolDispatch: { call in
                probe.recordDispatch(call)
                return "sunny"
            })

        do {
            _ = try await session.respond(to: "weather")
            #expect(declaresTool && validation == .permissive)
            #expect(probe.dispatchedToolCalls.count == 1)
            #expect(probe.dispatchedToolCalls.first?.function.arguments["city"] == .string("Paris"))
        } catch let error as RejectedToolCallError {
            #expect(!declaresTool || validation == .strict)
            #expect(error.rejection.reason == (declaresTool ? .invalidArguments : .undeclaredTool))
            #expect(probe.dispatchedToolCalls.isEmpty)
        }
    }

    @Test("single text turn reports MTP speculation")
    func singleTextTurnReportsMTPTelemetry() async throws {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: parameters
        )
        let greedySession = ChatSession(
            makeModelContext(),
            generateParameters: parameters
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let greedyResult = try await collect(greedySession.streamDetails(to: "a"))

        #expect(result.text == greedyResult.text)
        let telemetry = try #require(result.info.speculativeDecodingTelemetry)
        #expect(telemetry.roundCount > 0)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.acceptedDraftTokens == result.info.proposedDraftTokens)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
        #expect(greedyResult.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("one-token MTP generation has no proposals and no session fallback")
    func oneTokenMTPGenerationDoesNotReportFallback() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 1, temperature: 0)
        )

        let result = try await collect(session.streamDetails(to: "a"))

        #expect(result.text == "x")
        #expect(result.info.generationTokenCount == 1)
        #expect(result.info.proposedDraftTokens == 0)
        #expect(result.info.acceptedDraftTokens == 0)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("cold MTP staging refusal falls back before target prefill")
    func coldStagingRefusalUsesOrdinaryGeneration() async throws {
        let probe = MTPStagingProbe()
        probe.cache.refusesSpeculativeStaging = true
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: parameters
        )
        let baseline = ChatSession(makeModelContext(), generateParameters: parameters)

        let result = try await collect(session.streamDetails(to: "a"))
        let expected = try await collect(baseline.streamDetails(to: "a"))
        let preparation = try #require(probe.preparations.first)
        let status = try await session.cacheStatus()

        #expect(result.text == expected.text)
        #expect(result.info.promptTokenCount == expected.info.promptTokenCount)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.speculativeDecodingFallbackReason == .unsupportedCache)
        #expect(probe.cache.refusedWidths == [2])
        #expect(probe.cacheCreationCount == 1)
        #expect(probe.preparations.count == 1)
        #expect(preparation.tokens == [7])
        #expect(preparation.cachedTokens.isEmpty)
        #expect(!preparation.emitsMTPState)
        #expect(probe.draftCount == 0)
        #expect(status.processedTokenCount == 5)
        #expect(probe.cache.offset == status.processedTokenCount)
    }

    @Test("warm MTP staging refusal preserves target cache and carried state")
    func warmStagingRefusalPreservesTargetContinuation() async throws {
        let probe = MTPStagingProbe()
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let firstStatus = try await session.cacheStatus()
        let firstProcessedTokenCount = try #require(firstStatus.processedTokenCount)
        let prefix = probe.cache.cachedTokens
        let draftsBeforeRefusal = probe.draftCount
        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(probe.finalizedPositions == [firstProcessedTokenCount])
        #expect(prefix == [7, 4, 4, 4])

        probe.cache.refusesSpeculativeStaging = true
        let second = try await collect(session.streamDetails(to: "b"))
        let status = try await session.cacheStatus()
        #expect(probe.preparations.count == 2)
        let preparation = try #require(probe.preparations.last)

        #expect(second.text == first.text)
        #expect(probe.cache.refusedWidths == [2])
        #expect(probe.cacheCreationCount == 1)
        #expect(preparation.cacheID == ObjectIdentifier(probe.cache))
        #expect(preparation.cachedTokens == prefix)
        #expect(preparation.tokens == [4, 7])
        #expect(preparation.targetMarker == 42)
        #expect(!preparation.emitsMTPState)
        #expect(Array(probe.cache.cachedTokens.prefix(prefix.count)) == prefix)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount == firstProcessedTokenCount)
        #expect(
            status.processedTokenCount
                == second.info.cachedPromptTokenCount + second.info.promptTokenCount
                + second.info.generationTokenCount)
        #expect(probe.cache.offset == status.processedTokenCount)
        #expect(probe.draftCount == draftsBeforeRefusal)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.acceptedDraftTokens == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unsupportedCache)
    }

    @Test("MTP preparation errors after cache mutation propagate without retry")
    func preparationFailureAfterCacheMutationDoesNotFallBack() async throws {
        let probe = MTPStagingProbe()
        probe.failAfterPreparationWrite = true
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        do {
            _ = try await collect(session.streamDetails(to: "a"))
            Issue.record("Expected the target preparation error")
        } catch let error as KVCacheError {
            #expect(error.message == "Target preparation failed after a cache write")
        }

        #expect(probe.preparations.count == 1)
        #expect(probe.preparations.first?.emitsMTPState == true)
        #expect(probe.cache.cachedTokens == [7])
        #expect(probe.cacheCreationCount == 1)
        #expect(probe.cache.refusedWidths.isEmpty)
        #expect(probe.draftCount == 0)
    }

    @Test("default sampling bypasses Qwen MTP before cache policy")
    func defaultSamplingUsesOrdinaryWarmCacheGeneration() async throws {
        // Keep GenerateParameters' default temperature (0.6), which Qwen MTP
        // cannot speculate with until stochastic acceptance is implemented.
        let parameters = GenerateParameters(maxTokens: 4)
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: parameters
        )
        let regularSession = ChatSession(
            makeModelContext(),
            generateParameters: parameters
        )

        _ = try await collect(session.streamDetails(to: "a"))
        _ = try await collect(regularSession.streamDetails(to: "a"))
        let result = try await collect(session.streamDetails(to: "b"))
        let regularResult = try await collect(regularSession.streamDetails(to: "b"))

        #expect(result.text == regularResult.text)
        #expect(result.info.promptTokenCount == regularResult.info.promptTokenCount)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == .unsupportedSampling)
        #expect(regularResult.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("a target-only sampling turn drops the warm MTP continuation")
    func samplingChangeDropsMTPContinuationWithoutDroppingTargetCache() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        session.generateParameters.temperature = 0.6
        let targetOnly = try await collect(session.streamDetails(to: "b"))
        session.generateParameters.temperature = 0
        let afterDrop = try await collect(session.streamDetails(to: "c"))

        #expect(targetOnly.info.promptTokenCount == 2)
        #expect(targetOnly.info.cachedPromptTokenCount > 0)
        #expect(targetOnly.info.proposedDraftTokens == nil)
        #expect(targetOnly.info.speculativeDecodingFallbackReason == .unsupportedSampling)
        // A target-only pass commits its final token directly, so the next
        // append has only the new user token left to prefill.
        #expect(afterDrop.info.promptTokenCount == 1)
        #expect(afterDrop.info.cachedPromptTokenCount > 0)
        #expect(afterDrop.info.proposedDraftTokens == nil)
        #expect(afterDrop.info.speculativeDecodingFallbackReason == .unavailableContinuation)
    }

    @Test("a cache-plan change rebuilds target and MTP state together")
    func cachePlanChangeRebuildsMTPContinuation() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        session.generateParameters.maxKVSize = 64
        let rebuilt = try await collect(session.streamDetails(to: "b"))

        #expect(rebuilt.info.cachedPromptTokenCount == 0)
        #expect(rebuilt.info.promptTokenCount > 2)
        #expect((rebuilt.info.proposedDraftTokens ?? 0) > 0)
        #expect(rebuilt.info.passthroughReason == nil)
        #expect(rebuilt.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("second text turn resumes MTP from the warm prompt cache")
    func secondTextTurnResumesWarmMTP() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect(first.text == "xxxx")
        // The fixture's transcript render is append-only: the fourth generated
        // token and the new user token are the only uncached suffix.
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("stateless MTP reuses the warm main cache on the second text turn")
    func statelessMTPUsesWarmMainCache() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(requiresPromptPrefill: false),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect(second.info.promptTokenCount < 13)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
    }

    @Test("non-resumable stateful MTP preserves target cache and falls back target-only")
    func nonResumableStatefulMTPFallsBackWithoutRebuildingTarget() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(
                model: NonResumableMTPDrafter()),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.acceptedDraftTokens == nil)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(first.info.speculativeDecodingFallbackReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unavailableContinuation)
    }

    @Test("raw prompt cache keeps its prefix and falls back to target-only generation")
    func rawPromptCacheFallsBackWithoutDiscardingPrefix() async throws {
        let rawCache = KVCacheSimple()
        _ = rawCache.update(
            keys: MLXArray.zeros([1, 1, 3, 1]),
            values: MLXArray.zeros([1, 1, 3, 1])
        )
        let session = ChatSession(
            makeModelContext(),
            cache: [rawCache],
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let status = try await session.cacheStatus()

        #expect(result.info.promptTokenCount == 1)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.speculativeDecodingFallbackReason == .unavailableContinuation)
        // Existing raw prefix (3) + rendered fragment (1) + generated tokens (4).
        #expect(status.processedTokenCount == 8)
    }

    @Test(
        "prepared media reports the configured fallback and retains the main cache",
        arguments: [true, false], [Float(0), Float(0.6)])
    func preparedMediaFallsBackAndRetainsMainCache(
        speculationEnabled: Bool, temperature: Float
    ) async throws {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = PreparedMediaMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let configuration = try makeMTPConfiguration()
        let session = ChatSession(
            context,
            speculativeDecoding: speculationEnabled ? configuration : nil,
            generateParameters: .init(maxTokens: 4, temperature: temperature)
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let status = try await session.cacheStatus()

        #expect(result.info.promptTokenCount == 1)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        if speculationEnabled {
            #expect(
                result.info.speculativeDecodingFallbackReason
                    == (temperature == 0 ? .unsupportedMedia : .unsupportedSampling))
        } else {
            #expect(result.info.speculativeDecodingFallbackReason == nil)
        }
        #expect(status.processedTokenCount == 5)
    }

    @Test("prepared media discards an existing MTP continuation target-only")
    func preparedMediaDropsWarmMTPContinuation() async throws {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = SecondTurnMediaMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(first.info.speculativeDecodingFallbackReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unsupportedMedia)
    }

    @Test("consumer cancellation finalizes MTP before a clean next turn")
    func consumerCancellationLeavesCleanNextTurn() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 50, temperature: 0)
        )

        for try await event in session.streamDetails(to: "a") {
            if event.chunk != nil {
                break
            }
        }
        await session.synchronize()

        session.generateParameters = .init(maxTokens: 4, temperature: 0)
        let result = try await collect(session.streamDetails(to: "b"))
        let status = try await session.cacheStatus()

        #expect(result.text == "xxxx")
        // The cancelled assistant turn is not retained; the new user turn is
        // rendered from a clean transcript and cold-prefilled with MTP.
        #expect(result.info.promptTokenCount == 1)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.passthroughReason == nil)
        #expect(
            status.processedTokenCount
                == result.info.promptTokenCount + result.info.generationTokenCount - 1)
    }

    @Test("automatic tool restart reports MTP continuation eligibility", arguments: [true, false])
    func automaticToolRestartReportsMTPContinuation(resumable: Bool) async throws {
        let probe = ToolRestartProbe()
        let tokenizer = ToolRestartMTPTokenizer(probe: probe)
        let processor = ToolRestartMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let weatherTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let drafter: any MTPDrafterModel =
            resumable ? QwenStyleMTPDrafter() : NonResumableMTPDrafter()
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(model: drafter),
            generateParameters: .init(maxTokens: 24, temperature: 0),
            tools: [weatherTool],
            toolDispatch: { call in
                probe.recordDispatch(call)
                return #"{"forecast":"sunny"}"#
            }
        )

        let result = try await collectAll(session.streamDetails(to: "weather"))
        let status = try await session.cacheStatus()

        let dispatched = probe.dispatchedToolCalls
        #expect(dispatched.count == 1)
        let call = try #require(dispatched.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["city"] == .string("Paris"))

        let renderPasses = probe.renderPasses
        #expect(renderPasses.count == 2)
        let restart = try #require(renderPasses.count == 2 ? renderPasses[1] : nil)
        #expect(restart.map(\.role) == ["user", "assistant", "tool"])
        #expect(restart[0].content == "weather")
        #expect(restart[1].content.isEmpty)
        #expect(restart[1].toolCallNames == ["get_weather"])
        #expect(restart[2].content == #"{"forecast":"sunny"}"#)
        #expect(restart[2].toolResultName == "get_weather")

        #expect(result.infos.count == 2)
        let firstInfo = try #require(result.infos.first)
        let restartInfo = try #require(result.infos.last)
        #expect(firstInfo.promptTokenCount == 10)
        #expect(
            restartInfo.cachedPromptTokenCount
                == firstInfo.promptTokenCount + firstInfo.generationTokenCount - 1)
        for info in result.infos.prefix(resumable ? 2 : 1) {
            #expect(info.speculativeDecodingTelemetry != nil)
            #expect((info.proposedDraftTokens ?? 0) > 0)
            #expect(info.passthroughReason == nil)
            #expect(info.speculativeDecodingFallbackReason == nil)
        }
        if !resumable {
            #expect(restartInfo.speculativeDecodingTelemetry == nil)
            #expect(restartInfo.proposedDraftTokens == nil)
            #expect(restartInfo.acceptedDraftTokens == nil)
            #expect(restartInfo.passthroughReason == nil)
            #expect(restartInfo.speculativeDecodingFallbackReason == .unavailableContinuation)
        }
        // A speculative pass leaves its final verifier sample uncommitted.
        #expect(
            status.processedTokenCount
                == restartInfo.cachedPromptTokenCount + restartInfo.promptTokenCount
                + restartInfo.generationTokenCount - (resumable ? 1 : 0))
    }

    private func makeModelContext(probe: MTPStagingProbe? = nil) -> ModelContext {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        return ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(probe: probe),
            processor: processor,
            tokenizer: tokenizer
        )
    }

    private func makeMTPConfiguration(
        requiresPromptPrefill: Bool = true
    ) throws -> SpeculativeDecodingConfig {
        let model: any MTPDrafterModel =
            requiresPromptPrefill
            ? QwenStyleMTPDrafter()
            : StatelessMTPDrafter()
        return try makeMTPConfiguration(model: model)
    }

    private func makeMTPConfiguration(
        model: any MTPDrafterModel
    ) throws -> SpeculativeDecodingConfig {
        let container = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(id: "qwen-mtp-test"),
                model: model
            )
        )
        return try SpeculativeDecodingConfig(mtpDrafter: container, blockSize: 2)
    }

    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let completion = event.info {
                info = completion
            }
        }
        return (text, try #require(info))
    }

    private func collectAll(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, infos: [GenerateCompletionInfo]) {
        var text = ""
        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let info = event.info {
                infos.append(info)
            }
        }
        return (text, infos)
    }
}

private struct DeterministicMTPTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { $0 == "x" ? 4 : 7 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(repeating: "x", count: tokenIds.filter { $0 != 15 }.count)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            encode(
                text: message["content"] as? String ?? "",
                addSpecialTokens: false)
        }
    }
}

private struct DeterministicMTPInputProcessor: UserInputProcessor {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}

private struct PreparedMediaMTPInputProcessor: UserInputProcessor {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-media-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let base = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        let prepared = try await base.prepare(input: input)
        return LMInput(
            text: prepared.text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1]))
        )
    }
}

private final class SecondTurnMediaMTPInputProcessor: UserInputProcessor, @unchecked Sendable {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-media-switch-test")

    private let lock = NSLock()
    private var prepareCount = 0

    init(tokenizer: DeterministicMTPTokenizer) {
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let base = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        let prepared = try await base.prepare(input: input)

        let includesMedia = lock.withLock {
            prepareCount += 1
            return prepareCount > 1
        }

        guard includesMedia else { return prepared }
        return LMInput(
            text: prepared.text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1]))
        )
    }
}

private struct ToolRestartRenderedMessage: Sendable {
    let role: String
    let content: String
    let toolCallNames: [String]
    let toolResultName: String?
}

private final class ToolRestartProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var storedRenderPasses: [[ToolRestartRenderedMessage]] = []
    private var storedDispatchedToolCalls: [ToolCall] = []

    func recordRender(_ messages: [[String: any Sendable]]) {
        let snapshot = messages.map { message in
            let toolCalls = message["tool_calls"] as? [[String: any Sendable]] ?? []
            let names = toolCalls.compactMap { toolCall in
                let function = toolCall["function"] as? [String: any Sendable]
                return function?["name"] as? String
            }
            return ToolRestartRenderedMessage(
                role: message["role"] as? String ?? "",
                content: message["content"] as? String ?? "",
                toolCallNames: names,
                toolResultName: message["name"] as? String
            )
        }

        lock.lock()
        storedRenderPasses.append(snapshot)
        lock.unlock()
    }

    func recordDispatch(_ call: ToolCall) {
        lock.lock()
        storedDispatchedToolCalls.append(call)
        lock.unlock()
    }

    var renderPasses: [[ToolRestartRenderedMessage]] {
        lock.lock()
        defer { lock.unlock() }
        return storedRenderPasses
    }

    var dispatchedToolCalls: [ToolCall] {
        lock.lock()
        defer { lock.unlock() }
        return storedDispatchedToolCalls
    }
}

private final class ToolRestartMTPTokenizer: Tokenizer, @unchecked Sendable {
    private let probe: ToolRestartProbe
    private let lock = NSLock()
    private var pass = 0

    private let toolCallScript =
        #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
    private let finalScript = "Sunny in Paris."

    init(probe: ToolRestartProbe) {
        self.probe = probe
    }

    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        Array(repeating: 7, count: text.unicodeScalars.count)
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        lock.lock()
        let currentPass = pass
        lock.unlock()
        let script = currentPass == 1 ? toolCallScript : finalScript
        return String(script.prefix(min(tokenIds.count * 4, script.count)))
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        probe.recordRender(messages)
        lock.lock()
        pass += 1
        lock.unlock()

        var tokens: [Int] = []
        for message in messages {
            switch message["role"] as? String {
            case "user":
                tokens.append(1)
            case "assistant":
                // Match the generation marker below so a rendered assistant
                // tool call extends the exact live generation trajectory.
                tokens.append(9)
            case "tool":
                tokens.append(5)
            default:
                tokens.append(6)
            }
            let content = message["content"] as? String ?? ""
            tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
            if let toolCalls = message["tool_calls"] as? [[String: any Sendable]] {
                // The scripted first pass emits 24 token-4 values. Re-render
                // those exact tool-call tokens so the restart is append-only.
                tokens.append(contentsOf: Array(repeating: 4, count: toolCalls.count * 24))
            }
            tokens.append(8)
        }
        tokens.append(9)
        return tokens
    }
}

private struct ToolRestartMTPInputProcessor: UserInputProcessor {
    let tokenizer: ToolRestartMTPTokenizer
    let configuration = ModelConfiguration(
        id: "qwen-mtp-tool-restart-test",
        toolCallFormat: .json
    )

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}

/// Deterministic attention-only target that publishes the same MTP state shape
/// used by Qwen text models, while keeping this ChatSession test lightweight.
private final class MTPStateEmittingTarget: Module, LanguageModel, KVCacheDimensionProvider {
    private let probe: MTPStagingProbe?
    var kvHeads: [Int] { [1] }

    init(probe: MTPStagingProbe? = nil) {
        self.probe = probe
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        guard let probe else { return try [makeAttentionKVCache(parameters: parameters)] }
        probe.cacheCreationCount += 1
        return [probe.cache]
    }

    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        let cache: [KVCache] =
            probe == nil
            ? try newCache(parameters: parameters)
            : [SpeculativeStagingCache()]
        return KVCacheStatus(
            cache: cache, plan: try parameters?.kvCachePlan() ?? .disabled, phase: .planned)
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        if let probe, let cache = cache.first as? SpeculativeStagingCache {
            probe.preparations.append(
                .init(
                    cacheID: ObjectIdentifier(cache), cachedTokens: cache.cachedTokens,
                    tokens: input.text.tokens.asArray(Int.self),
                    emitsMTPState: state?[mtpEmitFlagKey] == true,
                    targetMarker: state?[stagingTargetMarkerKey]))
            if probe.failAfterPreparationWrite {
                let values = input.text.tokens.asType(.float32).reshaped([1, 1, -1, 1])
                _ = cache.update(keys: values, values: values)
                throw KVCacheError(message: "Target preparation failed after a cache write")
            }
        }
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        output(for: inputs, cache: cache, emitMTPState: false).logits
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        output(for: input.tokens, cache: cache, emitMTPState: state?[mtpEmitFlagKey] == true)
    }

    private func output(
        for tokens: MLXArray,
        cache: [KVCache]?,
        emitMTPState: Bool
    ) -> LMOutput {
        let positions = tokens.dim(-1)
        let newKV =
            probe == nil
            ? MLXArray.zeros([1, 1, positions, 1])
            : tokens.asType(.float32).reshaped([1, 1, positions, 1])
        let sharedKV = cache?.first?.update(keys: newKV, values: newKV) ?? (newKV, newKV)

        let vocabularySize = 16
        var values = [Float](repeating: -100, count: positions * vocabularySize)
        for position in 0 ..< positions {
            values[position * vocabularySize + 4] = 100
        }
        let logits = MLXArray(values, [1, positions, vocabularySize])

        var outputState = LMOutput.State()
        if probe != nil {
            outputState[stagingTargetMarkerKey] = 42
        }
        guard emitMTPState else {
            return LMOutput(logits: logits, state: probe == nil ? nil : outputState)
        }
        outputState[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        outputState[mtpSharedKVStatesKey] = ["full_attention": sharedKV]
        outputState[mtpSharedKVSourceIndicesKey] = ["full_attention": 0]
        outputState[mtpSharedKVOffsetsKey] = [
            "full_attention": cache?.first?.offset ?? positions
        ]
        return LMOutput(logits: logits, state: outputState)
    }
}

/// Qwen-like contract: private per-stream state, prompt prefill, greedy-only,
/// no dependency on target-shared K/V, and one drafted token per round.
private final class QwenStyleMTPDrafter: Module, ResumableMTPDrafterModel {
    private let probe: MTPStagingProbe?

    init(probe: MTPStagingProbe? = nil) {
        self.probe = probe
    }

    var maximumBlockSize: Int? { 2 }
    var requiresSharedTargetKV: Bool { false }
    let requiresPromptPrefill = true
    var requiresGreedySampling: Bool { true }

    func validateCompatibility(with _: any LanguageModel) throws {}

    func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [])
    }

    func finalizeDrafterState(
        target _: any LanguageModel,
        targetBoundaryHidden _: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens _: Int,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool {
        probe?.finalizedPositions.append(targetProcessedTokenCount)
        state.nextPosition = targetProcessedTokenCount
        state.seedToken = nil
        state.seedHidden = nil
        state.proposalAppended = 0
        return true
    }

    func resumeDrafterState(
        target _: any LanguageModel,
        suffixTokens _: MLXArray,
        suffixTargetHidden _: MLXArray,
        targetBoundaryHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> Bool {
        true
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(batchSize: lastToken.dim(0), blockSize: blockSize)
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(batchSize: lastToken.dim(0), blockSize: blockSize)
    }

    private func draftedTokens(batchSize: Int, blockSize: Int) -> MLXArray {
        if let probe { probe.draftCount += 1 }
        return MLXArray(
            Array(repeating: Int32(4), count: batchSize * (blockSize - 1)),
            [batchSize, blockSize - 1]
        )
    }
}

private let stagingTargetMarkerKey = LMOutput.Key<Int>("staging-test-target-marker")

private final class MTPStagingProbe {
    struct Preparation {
        let cacheID: ObjectIdentifier
        let cachedTokens: [Float]
        let tokens: [Int]
        let emitsMTPState: Bool
        let targetMarker: Int?
    }

    let cache = SpeculativeStagingCache()
    var cacheCreationCount = 0
    var preparations: [Preparation] = []
    var draftCount = 0
    var finalizedPositions: [Int] = []
    var failAfterPreparationWrite = false
}

private final class SpeculativeStagingCache: KVCache {
    private let storage: KVCacheSimple
    var refusesSpeculativeStaging = false
    var refusedWidths: [Int] = []

    init(storage: KVCacheSimple = KVCacheSimple()) {
        self.storage = storage
    }

    var offset: Int { storage.offset }
    var maxSize: Int? { storage.maxSize }
    var isTrimmable: Bool { storage.isTrimmable }
    var cachedTokens: [Float] { storage.state.first?.asArray(Float.self) ?? [] }
    var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }
    var metaState: [String] {
        get { storage.metaState }
        set { storage.metaState = newValue }
    }

    func isTrimmable(after positions: Int) -> Bool {
        if refusesSpeculativeStaging && positions > 1 {
            refusedWidths.append(positions)
            return false
        }
        return storage.isTrimmable(after: positions)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { storage.trim(n) }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    func innerState() -> [MLXArray] { storage.innerState() }

    func copy() -> any KVCache {
        let copy = SpeculativeStagingCache(storage: storage.copy() as! KVCacheSimple)
        copy.refusesSpeculativeStaging = refusesSpeculativeStaging
        return copy
    }
}

/// Gemma-style MTP fixture: all conditioning state comes from the target, so
/// no iterator-owned continuation is needed between turns.
private class StatelessMTPDrafter: Module, MTPDrafterModel {
    var maximumBlockSize: Int? { 2 }
    var requiresSharedTargetKV: Bool { true }
    var requiresPromptPrefill: Bool { false }

    func validateCompatibility(with _: any LanguageModel) throws {}

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        MLXArray(
            Array(repeating: Int32(4), count: lastToken.dim(0) * (blockSize - 1)),
            [lastToken.dim(0), blockSize - 1])
    }
}

/// Stateful but deliberately lacking `ResumableMTPDrafterModel`. It proves
/// that the capability is additive and absence degrades only speculation.
private final class NonResumableMTPDrafter: StatelessMTPDrafter,
    StatefulMTPDrafterModel
{
    override var requiresSharedTargetKV: Bool { false }
    override var requiresPromptPrefill: Bool { true }

    func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [])
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas,
            queryOffset: queryOffset, blockSize: blockSize, sampler: sampler)
    }
}
