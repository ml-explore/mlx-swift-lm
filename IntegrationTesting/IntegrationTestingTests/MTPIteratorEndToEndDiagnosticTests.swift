// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - End-to-end MTP iterator diagnostic
//
// Exercises the full `MTPSpeculativeTokenIterator` cycle (prepare → multiple
// speculation rounds → info event) against the bit-exact-verified 31B
// target+drafter pair. Asserts on the iterator's per-stream draft proposal
// and acceptance counters as surfaced through Phase 0's
// `GenerateCompletionInfo` plumbing — closing a verification gap that the
// fixture-based Rung 1–4 tests didn't cover. Forward-component fixture
// tests verify correctness of the building blocks; they do not verify that
// the orchestrating state machine ever executes its full cycle on this
// branch.
//
// Discovered via @joelnishanth's integration benchmark on PR #308 reporting
// `proposedDraftTokens=0` across all MTP runs plus 3–6× slowdown vs.
// baseline. The harness here is the load-bearing measurement.

// MARK: - Helpers

private func hfSnapshotDir(modelId: String) -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hub.appendingPathComponent(folderName).appendingPathComponent("snapshots")
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

private struct LoadedPair {
    let context: ModelContext
    let drafter: any MTPDrafterModel
}

private func loadTargetAndDrafter(
    targetModelId: String,
    drafterModelId: String,
    drafterConfigType: any Codable.Type = Gemma4AssistantConfiguration.self
) async throws -> LoadedPair? {
    guard let targetDir = hfSnapshotDir(modelId: targetModelId) else { return nil }
    guard let drafterDir = hfSnapshotDir(modelId: drafterModelId) else { return nil }

    // Target — VLM factory's directory-load path gives us a full ModelContext
    // (model + tokenizer + processor + configuration) without needing a
    // Downloader. The 8-bit variant isn't in VLMRegistry but the factory
    // resolves model type from `config.json` at the directory root.
    let context = try await VLMModelFactory.shared.load(
        from: targetDir,
        using: #huggingFaceTokenizerLoader()
    )

    // Drafter — same pattern as `MTPRung4TokenParityTests.loadRung4Drafter`:
    // decode the Gemma4Assistant config, instantiate, then `loadWeights`.
    // `MTPSpeculativeTokenIterator.init` calls `drafter.bind(target:)`
    // internally, so explicit binding here would double-bind.
    let cfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self,
        from: Data(contentsOf: drafterDir.appendingPathComponent("config.json")))
    let drafter = Gemma4AssistantDraftModel(cfg)
    try loadWeights(modelDirectory: drafterDir, model: drafter)

    return LoadedPair(context: context, drafter: drafter)
}

// MARK: - 31B end-to-end iterator exercise

@Suite(.serialized)
struct MTPIteratorEndToEndDiagnosticTests {

    /// Load the bit-exact-verified 31B target+drafter pair and run the MTP
    /// `generate(...)` overload for 32 tokens at the production blockSize=4
    /// default. Captures the `.info` event and asserts on the Phase 0
    /// counters: speculation must run (`proposedDraftTokens > 0`), at least
    /// one draft must be accepted (`acceptedDraftTokens > 0`), and the
    /// iterator must not engage sticky-passthrough (`passthroughReason
    /// == nil`).
    ///
    /// No acceptance-rate floor is asserted here; the first successful run
    /// logs the observed rate and a floor is added in a follow-up commit
    /// after a real number is on the table.
    @Test
    func testMTP31BPairProducesAcceptedDrafts() async throws {
        guard
            let loaded = try await loadTargetAndDrafter(
                targetModelId: "mlx-community/gemma-4-31b-it-8bit",
                drafterModelId: "mlx-community/gemma-4-31B-it-assistant-bf16"
            )
        else {
            Issue.record(
                "required checkpoint not in HF cache (31B 8-bit target or 31B drafter); skipping"
            )
            return
        }

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )

        var info: GenerateCompletionInfo?
        var text = ""
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                text += chunk
            case .toolCall:
                break
            case .info(let completionInfo):
                info = completionInfo
            }
        }

        guard let info else {
            Issue.record("stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? -1
        let accepted = info.acceptedDraftTokens ?? -1
        let passthrough = info.passthroughReason
        let rate =
            proposed > 0 ? "\(accepted)/\(proposed)" : "n/a (proposed=0)"

        print(
            "[MTPIteratorEndToEndDiagnostic 31B] proposed=\(proposed), accepted=\(accepted), rate=\(rate), passthrough=\(passthrough ?? "nil"), generated=\(info.generationTokenCount) tokens in \(info.generateTime.formatted())s"
        )
        print("[MTPIteratorEndToEndDiagnostic 31B] text: \(text)")

        #expect(
            info.proposedDraftTokens != nil, "proposedDraftTokens nil — Phase 0 plumbing broken")
        #expect(
            info.acceptedDraftTokens != nil, "acceptedDraftTokens nil — Phase 0 plumbing broken")
        #expect(
            info.passthroughReason == nil,
            "iterator engaged sticky-passthrough: \(info.passthroughReason ?? "")")
        #expect(
            (info.proposedDraftTokens ?? 0) > 0,
            "no tokens proposed across all rounds — speculation did not run")
        #expect(
            (info.acceptedDraftTokens ?? 0) > 0,
            "drafts proposed but none accepted — target rejected every draft")
    }

    /// Speculative decoding's load-bearing correctness property: at greedy
    /// decoding (temperature=0), MTP MUST produce token-identical output to
    /// autoregressive generation against the same target. Internal acceptance
    /// counters being healthy and forward-component fixtures passing are
    /// NECESSARY but not SUFFICIENT — only direct byte-comparison against
    /// baseline catches divergence from prepare-time bonus mishandling,
    /// position-id drift, sampling-determinism breaks, etc.
    ///
    /// Regression test for the prepare-time bonus yield bug fixed in
    /// commit ee86bff (Phase 4b): the iterator's prepare(input:) sampled one
    /// or two bonus tokens but never appended them to pendingTokens, so the
    /// output stream silently started 1 or 2 positions ahead of baseline.
    @Test
    func testMTP31BMatchesBaselineByteIdentical() async throws {
        guard
            let loaded = try await loadTargetAndDrafter(
                targetModelId: "mlx-community/gemma-4-31b-it-8bit",
                drafterModelId: "mlx-community/gemma-4-31B-it-assistant-bf16"
            )
        else {
            Issue.record(
                "required checkpoint not in HF cache (31B 8-bit target or 31B drafter); skipping"
            )
            return
        }

        let prompt = "Why is the sky blue? Explain in one paragraph."
        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        // MTP run.
        let mtpStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )
        var mtpInfo: GenerateCompletionInfo?
        var mtpText = ""
        for await event in mtpStream {
            switch event {
            case .chunk(let chunk): mtpText += chunk
            case .toolCall: break
            case .info(let i): mtpInfo = i
            }
        }

        // Baseline (non-speculative) run against the same target.
        let baselineStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: loaded.context
        )
        var baselineText = ""
        for await event in baselineStream {
            if case .chunk(let chunk) = event { baselineText += chunk }
        }

        guard let mtpInfo else {
            Issue.record("MTP stream completed without emitting an .info event")
            return
        }

        print(
            "[MTPIteratorEndToEndDiagnostic byte-identity] mtpText: \(mtpText)"
        )
        print(
            "[MTPIteratorEndToEndDiagnostic byte-identity] baselineText: \(baselineText)"
        )

        // Defense in depth — if speculation regresses to passthrough or
        // produces no proposals, the byte-identity assertion below might still
        // pass (passthrough output matches baseline by definition).
        #expect(
            mtpInfo.proposedDraftTokens != nil && (mtpInfo.proposedDraftTokens ?? 0) > 0,
            "MTP run produced no proposals — speculation regressed")
        #expect(
            mtpInfo.passthroughReason == nil,
            "MTP iterator engaged sticky-passthrough: \(mtpInfo.passthroughReason ?? "")")

        #expect(
            mtpText == baselineText,
            "MTP and baseline outputs diverged at temp=0 — speculative decoding violated bit-exact equivalence to autoregressive"
        )
    }

    /// Same shape as the 31B test but against E4B. Gated behind
    /// `TEST_E4B_PAIR` env var because the E4B drafter requires the centroid
    /// embedder which isn't on this branch. Documents the slot for future
    /// activation once centroid lands; not load-bearing for this PLAN's
    /// Phase 2 decision matrix.
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["TEST_E4B_PAIR"] != nil)
    )
    func testMTPE4BPairProducesAcceptedDrafts() async throws {
        guard
            let loaded = try await loadTargetAndDrafter(
                targetModelId: "mlx-community/gemma-4-e4b-it-4bit",
                drafterModelId: "mlx-community/gemma-4-E4B-it-assistant-bf16"
            )
        else {
            Issue.record(
                "required checkpoint not in HF cache (E4B target or E4B drafter); skipping"
            )
            return
        }

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )

        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completionInfo) = event { info = completionInfo }
        }

        guard let info else {
            Issue.record("stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? -1
        let accepted = info.acceptedDraftTokens ?? -1
        print(
            "[MTPIteratorEndToEndDiagnostic E4B] proposed=\(proposed), accepted=\(accepted), passthrough=\(info.passthroughReason ?? "nil")"
        )

        #expect(info.proposedDraftTokens != nil)
        #expect(info.acceptedDraftTokens != nil)
        #expect(info.passthroughReason == nil)
        #expect((info.proposedDraftTokens ?? 0) > 0)
        #expect((info.acceptedDraftTokens ?? 0) > 0)
    }
}
