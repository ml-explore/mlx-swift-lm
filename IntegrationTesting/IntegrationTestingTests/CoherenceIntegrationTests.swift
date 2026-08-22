// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// Load `configuration`, run the planets coherence check, then release the
/// model. Each coherence test uses a distinct model exactly once, so we evict
/// it from the shared cache and clear the GPU buffer pool afterwards: loading
/// ~20 multi-GB models in one serialized run otherwise accumulates resident
/// weights until the process is jetsammed (Metal compiler XPC crashes).
private func runCoherence(_ configuration: ModelConfiguration) async throws {
    let container = try await models.llmContainer(for: configuration)
    do {
        try await ChatSessionTests.planetsCoherence(container: container)
    } catch {
        await releaseModel(configuration)
        throw error
    }
    await releaseModel(configuration)
}

private func releaseModel(_ configuration: ModelConfiguration) async {
    await models.evictLLM(configuration)
    Stream.gpu.synchronize()
    Memory.clearCache()
}

/// Fast, reliable, architecturally-diverse subset that runs by default so the
/// nightly/manual job stays well under the job time limit. The full model
/// matrix lives in `CoherenceFullIntegrationTests` (opt-in).
@Suite(.serialized)
struct CoherenceIntegrationTests {

    @Test func llama3_2_1B() async throws {
        try await runCoherence(LLMRegistry.llama3_2_1B_4bit)
    }

    @Test func qwen3_1_7B() async throws {
        try await runCoherence(LLMRegistry.qwen3_1_7b_4bit)
    }

    @Test func gemma3_1B_qat() async throws {
        try await runCoherence(LLMRegistry.gemma3_1B_qat_4bit)
    }

    @Test func phi3_5() async throws {
        try await runCoherence(LLMRegistry.phi3_5_4bit)
    }

    @Test func granite3_3_2B() async throws {
        try await runCoherence(LLMRegistry.granite3_3_2b_4bit)
    }
}

/// The remaining coherence models — larger, slower, or otherwise not needed for
/// the default signal. Opt in with `MLX_RUN_FULL_COHERENCE=1`.
@Suite(
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_FULL_COHERENCE"] == "1"))
struct CoherenceFullIntegrationTests {

    @Test func bitnet_b1_58_2B() async throws {
        try await runCoherence(LLMRegistry.bitnet_b1_58_2b_4t_4bit)
    }

    @Test func exaone_4_0_1_2B() async throws {
        try await runCoherence(LLMRegistry.exaone_4_0_1_2b_4bit)
    }

    @Test func gemma3n_E2B() async throws {
        try await runCoherence(LLMRegistry.gemma3n_E2B_it_lm_4bit)
    }

    @Test func gemma4_e2b() async throws {
        try await runCoherence(LLMRegistry.gemma4_e2b_it_4bit)
    }

    @Test func glm4_9B() async throws {
        try await runCoherence(LLMRegistry.glm4_9b_4bit)
    }

    @Test func granite4_0_H_tiny() async throws {
        try await runCoherence(LLMRegistry.granite_4_0_h_tiny_4bit_dwq)
    }

    @Test func jamba_3B_4bit() async throws {
        try await runCoherence(LLMRegistry.jamba_3b_4bit)
    }

    @Test func lfm2_1_2B() async throws {
        try await runCoherence(LLMRegistry.lfm2_1_2b_4bit)
    }

    @Test func mistral_7B() async throws {
        try await runCoherence(LLMRegistry.mistral7B4bit)
    }

    @Test func olmo2_7B() async throws {
        try await runCoherence(LLMRegistry.olmo_2_1124_7B_Instruct_4bit)
    }

    // OLMoE ships a GPTNeoXTokenizer, which the tokenizer loader does not yet
    // support (fails with `.unsupportedTokenizer("GPTNeoXTokenizer")`), so this
    // model cannot currently pass the coherence check.
    @Test(.disabled("OLMoE uses GPTNeoXTokenizer, unsupported by the tokenizer loader"))
    func olmoe_1B_7B() async throws {
        try await runCoherence(LLMRegistry.olmoe_1b_7b_0125_instruct_4bit)
    }

    @Test func qwen3_5_2B() async throws {
        try await runCoherence(LLMRegistry.qwen3_5_2b_4bit)
    }

    @Test func smollm3_3B() async throws {
        try await runCoherence(LLMRegistry.smollm3_3b_4bit)
    }
}
