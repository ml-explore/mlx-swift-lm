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

    /// E4B failure-mode characterization. Both E-series drafters have
    /// `use_ordered_embeddings=true` and route through
    /// `Gemma4AssistantMaskedEmbedder.callAsFunction`, which is a `fatalError`
    /// stub on this branch (see `Gemma4Assistant.swift:94-99`, message:
    /// "Gemma4AssistantMaskedEmbedder forward not implemented yet — requires
    /// a use_ordered_embeddings=true checkpoint to verify…"). The centroid
    /// embedder is downstream work — Joel's PR #1 against this fork.
    ///
    /// Expected shape on this branch: the run reaches the first drafter
    /// forward call inside `speculateRound` and traps at the
    /// `Gemma4AssistantMaskedEmbedder` stub. If it traps elsewhere (e.g. in
    /// weight sanitization or `bind`), that's news worth surfacing — it
    /// means there's E-series-specific surface beyond just the centroid
    /// embedder. If it does NOT trap and instead produces garbage tokens
    /// with near-zero acceptance, the stub isn't gating the path it should.
    ///
    /// Env-gated (`TEST_E4B_PAIR`) because triggering the trap intentionally
    /// crashes the test process, and because the cached-checkpoint
    /// requirement makes routine CI invocation unhelpful. Will be repurposed
    /// into an assertion-based test once the centroid embedder lands.
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["TEST_E4B_PAIR"] != nil)
    )
    func testMTPE4BPairFailureMode() async throws {
        try await runEseriesFailureModeCharacterization(
            label: "E4B",
            targetModelId: "mlx-community/gemma-4-e4b-it-4bit",
            drafterModelId: "mlx-community/gemma-4-E4B-it-assistant-bf16"
        )
    }

    /// E2B counterpart of `testMTPE4BPairFailureMode`. Same expected shape:
    /// the iterator reaches the drafter forward call and traps in the
    /// `Gemma4AssistantMaskedEmbedder` stub.
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["TEST_E2B_PAIR"] != nil)
    )
    func testMTPE2BPairFailureMode() async throws {
        try await runEseriesFailureModeCharacterization(
            label: "E2B",
            targetModelId: "mlx-community/gemma-4-e2b-it-4bit",
            drafterModelId: "mlx-community/gemma-4-E2B-it-assistant-bf16"
        )
    }

    /// Shared body for the E-series failure-mode tests. Does NOT assert on
    /// the iterator's counters — the point is to characterize where (and how)
    /// the run fails, not to pass anything. Any output that survives to the
    /// info-event print is itself information about the failure mode.
    private func runEseriesFailureModeCharacterization(
        label: String,
        targetModelId: String,
        drafterModelId: String
    ) async throws {
        guard
            let loaded = try await loadTargetAndDrafter(
                targetModelId: targetModelId,
                drafterModelId: drafterModelId
            )
        else {
            Issue.record(
                "required checkpoint not in HF cache (\(label) target or \(label) drafter); skipping characterization"
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

        // Drain the stream. On this branch, the drafter's first forward call
        // is expected to fatalError, taking down the test process before
        // `info` is yielded. If we DO get an info event, log everything
        // observable; if generation succeeds at all, the stub isn't being
        // reached, which is itself worth surfacing.
        var info: GenerateCompletionInfo?
        var text = ""
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .toolCall: break
            case .info(let completionInfo): info = completionInfo
            }
        }

        if let info {
            let proposed = info.proposedDraftTokens ?? -1
            let accepted = info.acceptedDraftTokens ?? -1
            print(
                "[MTPIteratorEndToEndDiagnostic \(label) failure-mode] proposed=\(proposed), accepted=\(accepted), passthrough=\(info.passthroughReason ?? "nil"), generated=\(info.generationTokenCount) tokens"
            )
            print(
                "[MTPIteratorEndToEndDiagnostic \(label) failure-mode] text: \(text)"
            )
        } else {
            print(
                "[MTPIteratorEndToEndDiagnostic \(label) failure-mode] stream completed without an .info event"
            )
        }
    }
}
