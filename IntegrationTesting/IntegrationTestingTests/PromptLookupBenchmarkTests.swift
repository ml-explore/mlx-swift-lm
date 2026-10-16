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

private let promptLookupBenchmarkEnvironment = ProcessInfo.processInfo.environment

/// Opt-in, fixed-output-length GPU benchmark with interleaved baseline runs.
@Suite(
    .serialized,
    .enabled(if: promptLookupBenchmarkEnvironment["MLX_RUN_PROMPT_LOOKUP_BENCHMARK"] == "1")
)
struct PromptLookupBenchmarkTests {
    @Test
    func interleavedGeneration() async throws {
        let modelID =
            promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_MODEL"]
            ?? "mlx-community/gemma-3-1b-it-qat-4bit"
        let revision =
            promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_REVISION"]
            ?? Self.revisions[modelID] ?? "main"
        let runs = integerSetting("MLX_PROMPT_LOOKUP_RUNS", default: 3)
        let maxTokens = integerSetting("MLX_PROMPT_LOOKUP_TOKENS", default: 256)
        let draftLengths =
            (promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_DRAFT_LENGTHS"]
            ?? "2,4,8,16").split(separator: ",").compactMap { Int($0) }
        let sizes =
            (promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_CONTEXTS"]
            ?? "short,long").split(separator: ",").map(String.init)
        let workloads =
            (promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_WORKLOADS"]
            ?? "code-edit,document-copy,creative").split(separator: ",").map(String.init)
        let adaptiveModes =
            (promptLookupBenchmarkEnvironment["MLX_PROMPT_LOOKUP_ADAPTIVE_MODES"]
            ?? "true,false").split(separator: ",").map { $0 == "true" }
        #expect(!draftLengths.isEmpty && draftLengths.allSatisfy { $0 > 0 })
        #expect(sizes.allSatisfy { ["short", "long"].contains($0) })
        #expect(workloads.allSatisfy { ["code-edit", "document-copy", "creative"].contains($0) })

        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: .init(id: modelID, revision: revision))
        defer {
            Stream.gpu.synchronize()
            Memory.clearCache()
        }

        print(
            "PROMPT_LOOKUP_METADATA model=\(modelID) revision=\(revision) "
                + "os=\(ProcessInfo.processInfo.operatingSystemVersionString) "
                + "cores=\(ProcessInfo.processInfo.processorCount) "
                + "physicalMemory=\(ProcessInfo.processInfo.physicalMemory) "
                + "maxTokens=\(maxTokens) runs=\(runs)"
        )

        for size in sizes {
            for workload in workloads {
                let text = benchmarkPrompt(workload: workload, long: size == "long")
                let input = try await context.processor.prepare(input: UserInput(prompt: text))
                let promptIDs = input.text.tokens.asArray(Int.self)
                let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

                for draftLength in draftLengths {
                    for adaptive in adaptiveModes {
                        let configuration = PromptLookupConfiguration(
                            maxDraftTokens: draftLength, adaptive: adaptive)
                        _ = try trial(
                            model: context.model, promptIDs: promptIDs,
                            parameters: parameters, configuration: nil)
                        _ = try trial(
                            model: context.model, promptIDs: promptIDs,
                            parameters: parameters, configuration: configuration)

                        let oracle = try trial(
                            model: context.model, promptIDs: promptIDs,
                            parameters: parameters, configuration: nil)
                        for run in 0 ..< runs {
                            let baseline: Trial
                            let lookup: Trial
                            if run.isMultiple(of: 2) {
                                baseline = try trial(
                                    model: context.model, promptIDs: promptIDs,
                                    parameters: parameters, configuration: nil)
                                lookup = try trial(
                                    model: context.model, promptIDs: promptIDs,
                                    parameters: parameters, configuration: configuration)
                            } else {
                                lookup = try trial(
                                    model: context.model, promptIDs: promptIDs,
                                    parameters: parameters, configuration: configuration)
                                baseline = try trial(
                                    model: context.model, promptIDs: promptIDs,
                                    parameters: parameters, configuration: nil)
                            }
                            let firstMismatch = zip(baseline.tokens, lookup.tokens)
                                .enumerated().first { $0.element.0 != $0.element.1 }?.offset
                            let exactMatch = baseline.tokens == lookup.tokens
                            let baselineDeterministic = baseline.tokens == oracle.tokens
                            #expect(
                                baselineDeterministic, "Baseline was not deterministic")
                            #expect(baseline.tokens.count == maxTokens)
                            #expect(lookup.tokens.count == maxTokens)
                            for (variant, result) in [("baseline", baseline), ("lookup", lookup)] {
                                let row = Row(
                                    model: modelID, revision: revision, workload: workload,
                                    context: size, draftLength: draftLength, adaptive: adaptive,
                                    run: run + 1,
                                    variant: variant, promptTokens: promptIDs.count,
                                    generatedTokens: result.tokens.count,
                                    prefillMilliseconds: result.prefillMilliseconds,
                                    decodeMilliseconds: result.decodeMilliseconds,
                                    wallMilliseconds: result.wallMilliseconds,
                                    modelCalls: result.modelCalls,
                                    verifiedPositions: result.verifiedPositions,
                                    proposedTokens: result.telemetry?.draftTokenCount ?? 0,
                                    acceptedTokens: result.telemetry?.acceptedDraftTokenCount ?? 0,
                                    rounds: result.telemetry?.roundCount ?? 0,
                                    peakGPUBytes: result.peakGPUBytes,
                                    exactMatch: exactMatch, firstMismatch: firstMismatch,
                                    baselineDeterministic: baselineDeterministic,
                                    baselineTokenAtMismatch: firstMismatch.map {
                                        baseline.tokens[$0]
                                    },
                                    lookupTokenAtMismatch: firstMismatch.map { lookup.tokens[$0] })
                                let data = try JSONEncoder().encode(row)
                                print(
                                    "PROMPT_LOOKUP_RESULT \(String(decoding: data, as: UTF8.self))")
                            }
                        }
                    }
                }
            }
        }
    }

    private static let revisions = [
        "mlx-community/gemma-3-1b-it-qat-4bit": "15fed4eafb456c6fcb2a1165f19ac609670ed14b",
        "mlx-community/Qwen3-4B-Instruct-2507-4bit": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    ]
}

private func integerSetting(_ name: String, default fallback: Int) -> Int {
    guard let value = promptLookupBenchmarkEnvironment[name].flatMap(Int.init), value > 0 else {
        return fallback
    }
    return value
}

private struct Trial {
    let tokens: [Int]
    let prefillMilliseconds: Double
    let decodeMilliseconds: Double
    let wallMilliseconds: Double
    let modelCalls: Int
    let verifiedPositions: Int
    let telemetry: SpeculativeDecodingTelemetry?
    let peakGPUBytes: Int
}

private struct Row: Encodable {
    let model: String
    let revision: String
    let workload: String
    let context: String
    let draftLength: Int
    let adaptive: Bool
    let run: Int
    let variant: String
    let promptTokens: Int
    let generatedTokens: Int
    let prefillMilliseconds: Double
    let decodeMilliseconds: Double
    let wallMilliseconds: Double
    let modelCalls: Int
    let verifiedPositions: Int
    let proposedTokens: Int
    let acceptedTokens: Int
    let rounds: Int
    let peakGPUBytes: Int
    let exactMatch: Bool
    let firstMismatch: Int?
    let baselineDeterministic: Bool
    let baselineTokenAtMismatch: Int?
    let lookupTokenAtMismatch: Int?
}

private func trial(
    model: any LanguageModel,
    promptIDs: [Int],
    parameters: GenerateParameters,
    configuration: PromptLookupConfiguration?
) throws -> Trial {
    Stream.gpu.synchronize()
    Memory.clearCache()
    Memory.peakMemory = 0
    let counted = CountingLanguageModel(model)
    let input = LMInput(tokens: MLXArray(promptIDs))
    let start = DispatchTime.now().uptimeNanoseconds
    var iterator: any TokenIteratorProtocol
    if let configuration {
        iterator = try PromptLookupTokenIterator(
            input: input, model: counted, parameters: parameters, configuration: configuration)
    } else {
        iterator = try TokenIterator(input: input, model: counted, parameters: parameters)
    }
    // Settle lazy prefill work before starting the decode clock.
    Stream.gpu.synchronize()
    let prefilled = DispatchTime.now().uptimeNanoseconds
    counted.reset()
    var tokens: [Int] = []
    tokens.reserveCapacity(parameters.maxTokens ?? 256)
    while let token = iterator.next() {
        tokens.append(token)
    }
    Stream.gpu.synchronize()
    let end = DispatchTime.now().uptimeNanoseconds
    return Trial(
        tokens: tokens,
        prefillMilliseconds: Double(prefilled - start) / 1_000_000,
        decodeMilliseconds: Double(end - prefilled) / 1_000_000,
        wallMilliseconds: Double(end - start) / 1_000_000,
        modelCalls: counted.calls, verifiedPositions: counted.positions,
        telemetry: iterator.speculativeDecodingTelemetry, peakGPUBytes: Memory.peakMemory)
}

/// Counts public decode calls; model-internal prefill chunks are excluded.
private final class CountingLanguageModel: Module, LanguageModel {
    let wrapped: any LanguageModel
    var calls = 0
    var positions = 0

    init(_ wrapped: any LanguageModel) {
        self.wrapped = wrapped
        super.init()
    }

    func reset() {
        calls = 0
        positions = 0
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try wrapped.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        calls += 1
        positions += input.tokens.dim(-1)
        return wrapped(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try wrapped.newCache(parameters: parameters)
    }
}

private func benchmarkPrompt(workload: String, long: Bool) -> String {
    let count = long ? 160 : 24
    switch workload {
    case "code-edit":
        let code = (0 ..< count).map { index in
            """
            func readRecord\(index)(_ values: [String: Int]) -> Int {
                let fallback = 16
                return values["record_\(index)"] ?? fallback
            }
            """
        }.joined(separator: "\n\n")
        return """
            Return this entire Swift source verbatim, changing every fallback value from 16 to 32.
            Preserve all functions and formatting. Output only the code, with no explanation.

            \(code)
            """
    case "document-copy":
        let records = (0 ..< count).map { index in
            "Record \(index): The amber observatory opens at sunrise. Visitors enter through the north gate."
        }.joined(separator: "\n")
        return """
            Copy every record below exactly, in order. Include every word and record number.
            Do not summarize. Output only the records.

            \(records)
            """
    default:
        let background =
            long
            ? (0 ..< count).map { "Inventory \($0): bracket, copper wire, packaging, invoice." }
                .joined(separator: "\n")
            : "No background is needed."
        return """
            The inventory below is unrelated background and must not be quoted:
            \(background)

            Write an original 1000-word story about a botanist discovering a silent floating island.
            Use vivid varied language, detailed dialogue, and a surprising ending.
            """
    }
}
