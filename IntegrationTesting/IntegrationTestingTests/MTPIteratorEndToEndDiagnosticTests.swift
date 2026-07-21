// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
@_spi(Testing) import MLXLMCommon
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

private struct LoadedPair {
    let context: ModelContext
    let drafter: any MTPDrafterModel
}

/// Recording `LogitProcessor` for the emit-only invariant test
/// `testMTPLogitProcessorReceivesOnlyEmittedTokens`. Struct so that
/// `var verifyProcessorCopy = processor` in `speculateRound` forks the
/// `recordedTokens` array via Swift copy-on-write — mutations to the
/// verify-loop copy do not propagate to the canonical processor.
private struct EmissionLog: LogitProcessor {
    var recordedTokens: [Int] = []
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {
        recordedTokens.append(token.item(Int.self))
    }
}

private let target31BModelId = "mlx-community/gemma-4-31b-it-8bit"
private let drafter31BModelId = "mlx-community/gemma-4-31B-it-assistant-bf16"

/// Pinned checkpoint revisions matching the weights that were live when the
/// Rung 4 `drafter_block` fixtures were generated — see the note in
/// `MTPRung4TokenParityTests.swift`. Kept in sync here so this diagnostic
/// suite exercises the same bit-exact-verified checkpoint pair.
private let target31BRevision = "fe92291011fc698452920c0b558b52f790dff711"
private let drafter31BRevision = "28e92270316e89288579ec59c17939541d9ca433"

/// Shared downloader for the 31B target+drafter pair. Fetches to the local
/// HF cache on first use (~35GB); subsequent tests and runs reuse the cache.
private let downloader: any Downloader = #hubDownloader()

private func loadTargetAndDrafter(
    targetModelId: String,
    drafterModelId: String,
    targetRevision: String? = nil,
    drafterRevision: String? = nil,
    drafterConfigType: any Codable.Type = Gemma4AssistantConfiguration.self
) async throws -> LoadedPair {
    // Target — VLM factory's downloader-based load path gives us a full
    // ModelContext (model + tokenizer + processor + configuration),
    // auto-fetching the snapshot from the HF Hub if it isn't cached yet.
    let context = try await VLMModelFactory.shared.load(
        from: downloader,
        using: #huggingFaceTokenizerLoader(),
        configuration: .init(id: targetModelId, revision: targetRevision ?? "main")
    )

    // Drafter — same pattern as `MTPRung4TokenParityTests.loadRung4Drafter`:
    // fetch the snapshot, decode the Gemma4Assistant config, instantiate,
    // then `loadWeights`. No explicit binding needed: the iterator passes
    // the target through `draftBlock(target:...)` per round.
    let drafterDir = try await downloader.download(
        id: drafterModelId,
        revision: drafterRevision,
        matching: ["*.safetensors", "*.json"],
        useLatest: false,
        progressHandler: { _ in }
    )
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
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

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

    /// Asserts a healthy acceptance-rate floor at blockSize=4 against the
    /// 31B sky-blue prompt. Locks in the iterator's per-round hidden-slice
    /// fix at the canonical production block size: the drafter must see the
    /// verify-position hidden at the slot that produced the newly-accepted
    /// bonus's prediction (mlx-lm's `verify.hidden[:, accepted : accepted + 1, :]`
    /// semantic) — NOT the unconditional last verify position.
    ///
    /// Pre-fix measurement (clean 61777c8 on this hardware, sky-blue, bs=4,
    /// maxTokens=64): 32.3% acceptance, 9.66 tok/s.
    /// Post-fix measurement: 60.6% acceptance, 13.64 tok/s — +28.3pp,
    /// +3.98 tok/s. The 45% floor below catches a regression toward the
    /// pre-fix baseline while leaving ~16pp headroom for natural run-to-run
    /// variance.
    ///
    /// Note: at maxTokens=32 (the basic smoke test `testMTP31BPairProduces‐
    /// AcceptedDrafts`), the same configuration yields higher acceptance
    /// (post-fix 72.4%) because shorter streams stay within the regime
    /// where MLX SDPA shape-determinism keeps verify-position predictions
    /// aligned with the autoregressive baseline. The maxTokens=64
    /// measurement here is the more stable rate over a larger sample
    /// (n=66 vs n=29) and is the better regression gate.
    @Test
    func testMTP31BBlockSize4AcceptanceLifted() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )

        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let i) = event { info = i }
        }

        guard let info else {
            Issue.record("MTP stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        let rate = proposed > 0 ? Double(accepted) / Double(proposed) : 0.0
        let tokPerSec =
            info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0.0

        print(
            "[MTPIteratorEndToEndDiagnostic blockSize=4 acceptance] proposed=\(proposed), accepted=\(accepted), rate=\(String(format: "%.1f%%", rate * 100)), generated=\(info.generationTokenCount) tokens in \(info.generateTime.formatted())s, tok/s=\(String(format: "%.2f", tokPerSec))"
        )

        #expect(
            info.passthroughReason == nil,
            "iterator engaged sticky-passthrough: \(info.passthroughReason ?? "")")
        #expect(
            proposed > 0,
            "no tokens proposed across all rounds — speculation did not run")
        #expect(
            rate >= 0.45,
            "blockSize=4 acceptance rate \(String(format: "%.1f%%", rate * 100)) below 45% floor — bonus-slot hidden-slice fix likely regressed (pre-fix 32.3% on this hardware at maxTokens=64 under the unconditional last-position slice)"
        )
    }

    /// Asserts a healthy acceptance-rate floor at blockSize=6 against the
    /// 31B sky-blue prompt. Locks in the iterator's per-round hidden-slice
    /// fix at the block size where the bug bites hardest: at higher K,
    /// `accepted < numDraft` happens more often (per the compounding-
    /// acceptance math), so the wrong-slot bug hurts speculation most here.
    /// The drafter must see the verify-position hidden at the slot that
    /// produced the newly-accepted bonus's prediction (mlx-lm's
    /// `verify.hidden[:, accepted : accepted + 1, :]` semantic) — NOT the
    /// unconditional last verify position.
    ///
    /// Pre-fix measurement (clean 61777c8 on this hardware, sky-blue, bs=6,
    /// maxTokens=64): 17.9% acceptance, 7.10 tok/s.
    /// Post-fix measurement: 29.8% acceptance, 8.60 tok/s — +11.9pp,
    /// +1.50 tok/s. The 25% floor below catches a regression toward the
    /// pre-fix baseline while leaving ~8pp headroom for variance.
    @Test
    func testMTP31BBlockSize6AcceptanceLifted() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 6
        )

        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let i) = event { info = i }
        }

        guard let info else {
            Issue.record("MTP stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        let rate = proposed > 0 ? Double(accepted) / Double(proposed) : 0.0
        let tokPerSec =
            info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0.0

        print(
            "[MTPIteratorEndToEndDiagnostic blockSize=6 acceptance] proposed=\(proposed), accepted=\(accepted), rate=\(String(format: "%.1f%%", rate * 100)), generated=\(info.generationTokenCount) tokens in \(info.generateTime.formatted())s, tok/s=\(String(format: "%.2f", tokPerSec))"
        )

        #expect(
            info.passthroughReason == nil,
            "iterator engaged sticky-passthrough: \(info.passthroughReason ?? "")")
        #expect(
            proposed > 0,
            "no tokens proposed across all rounds — speculation did not run")
        #expect(
            rate >= 0.25,
            "blockSize=6 acceptance rate \(String(format: "%.1f%%", rate * 100)) below 25% floor — bonus-slot hidden-slice fix likely regressed (pre-fix 17.9% on this hardware under the unconditional last-position slice)"
        )
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
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

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

    /// Cross-prompt byte-identity check (rainbows). The Phase 4b bonus-yield
    /// fix runs once per stream inside `prepare(input:)` and is content-
    /// independent by construction — but the sky-blue regression test alone
    /// doesn't prove that empirically. This test extends the bit-exact-
    /// equivalence check to a second prompt whose acceptance profile differs
    /// from sky-blue (46% acceptance per v1.1 §5.2 sweep).
    @Test
    func testMTP31BMatchesBaselineRainbowsPrompt() async throws {
        try await runMTPVsBaselineByteIdentityCheck(
            prompt: "Write a paragraph about how rainbows form.",
            maxTokens: 32
        )
    }

    /// Cross-prompt byte-identity check (Swift). Third prompt in the v1.1
    /// §5.2 sweep, with the lowest observed acceptance rate (25%). Lower
    /// acceptance means more correction-token positions per stream, which
    /// exercises the speculation/verify/accept boundary differently from the
    /// rainbows prompt.
    @Test
    func testMTP31BMatchesBaselineSwiftPrompt() async throws {
        try await runMTPVsBaselineByteIdentityCheck(
            prompt: "What's the best way to learn Swift?",
            maxTokens: 32
        )
    }

    /// Asserts that MTP generation at maxTokens=128 produces a valid
    /// continuation with healthy speculation activity, AND shares a
    /// byte-identical prefix of at least 64 tokens with baseline
    /// non-speculative generation at temp=0.
    ///
    /// This is a weaker assertion than the maxTokens=32 byte-identity
    /// test (`testMTP31BMatchesBaselineByteIdentical`) and intentionally
    /// so. MLX's fused SDPA kernel produces shape-dependent floating-
    /// point results: a multi-token verify (positions `[bonus, d_1,
    /// ..., d_K]`) processes through a different reduction path than
    /// the K+1 single-position autoregressive forwards baseline uses.
    /// At temp=0 with argmax sampling, this drift eventually flips a
    /// tied or near-tied logit choice and the streams diverge.
    /// Empirically on the 31B target+drafter pair, byte-identity
    /// holds through 64 tokens and breaks somewhere between 64 and
    /// 96.
    ///
    /// The MTP algorithm's correctness guarantee is "speculation-
    /// correctness under acceptance" — at each emitted position N,
    /// the iterator returns what the target would have returned at
    /// position N given the iterator's accumulated history through
    /// N-1. That guarantee holds throughout this test. Byte-identity
    /// to an independent baseline is a STRONGER property that the
    /// underlying numerical substrate doesn't support at arbitrary
    /// stream lengths. The 64-token prefix-identity assertion
    /// captures the empirical boundary on this hardware + model
    /// configuration.
    @Test
    func testMTP31BLongerStreamProducesValidContinuation() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let prompt = "Why is the sky blue? Explain in one paragraph."
        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        // MTP run.
        let mtpStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 128, temperature: 0),
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
            parameters: GenerateParameters(maxTokens: 128, temperature: 0),
            context: loaded.context
        )
        var baselineInfo: GenerateCompletionInfo?
        var baselineText = ""
        for await event in baselineStream {
            switch event {
            case .chunk(let chunk): baselineText += chunk
            case .toolCall: break
            case .info(let i): baselineInfo = i
            }
        }

        guard let mtpInfo, let baselineInfo else {
            Issue.record("MTP or baseline stream completed without emitting an .info event")
            return
        }

        // 1. MTP run completes — either EOS or budget. Both are valid
        //    completion signals; the prompt happens to natural-stop with EOS
        //    well before the 128 cap, so stopReason is typically `.stop`.
        //    Generated count must be at least the prefix-identity floor (64).
        #expect(
            mtpInfo.stopReason == .length || mtpInfo.stopReason == .stop,
            "MTP stopReason unexpected: \(mtpInfo.stopReason)")
        #expect(
            mtpInfo.generationTokenCount >= 64,
            "MTP generated only \(mtpInfo.generationTokenCount) tokens, less than the 64-token prefix floor"
        )

        // 2. Baseline run completes: same.
        #expect(
            baselineInfo.stopReason == .length || baselineInfo.stopReason == .stop,
            "Baseline stopReason unexpected: \(baselineInfo.stopReason)")
        #expect(
            baselineInfo.generationTokenCount >= 64,
            "Baseline generated only \(baselineInfo.generationTokenCount) tokens, less than the 64-token prefix floor"
        )

        // 3. MTP speculation actually ran end-to-end (no sticky-passthrough).
        #expect(
            (mtpInfo.proposedDraftTokens ?? 0) > 0,
            "MTP run produced no proposals — speculation regressed")
        #expect(
            (mtpInfo.acceptedDraftTokens ?? 0) > 0,
            "MTP run accepted no drafts — drafter quality regression or 31B pair mismatch")
        #expect(
            mtpInfo.passthroughReason == nil,
            "MTP iterator engaged sticky-passthrough: \(mtpInfo.passthroughReason ?? "")")

        // 4. Output decoded to non-empty text.
        #expect(!mtpText.isEmpty, "MTP generated text is empty")

        // 5. First 64 tokens byte-identical to baseline. Re-encode both decoded
        //    outputs without special tokens — Gemma's BPE is deterministic, so
        //    re-encode round-trips the model's emitted token IDs for the
        //    decoded text. If MTP and baseline produced the same token
        //    sequence for any prefix, their re-encodings of that prefix match.
        let mtpTokens = loaded.context.tokenizer.encode(text: mtpText, addSpecialTokens: false)
        let baselineTokens = loaded.context.tokenizer.encode(
            text: baselineText, addSpecialTokens: false)
        let prefixCount = 64
        #expect(
            mtpTokens.count >= prefixCount,
            "MTP re-encoded only \(mtpTokens.count) tokens, expected at least \(prefixCount)")
        #expect(
            baselineTokens.count >= prefixCount,
            "Baseline re-encoded only \(baselineTokens.count) tokens, expected at least \(prefixCount)"
        )
        let mtpPrefix = Array(mtpTokens.prefix(prefixCount))
        let baselinePrefix = Array(baselineTokens.prefix(prefixCount))
        #expect(
            mtpPrefix == baselinePrefix,
            "MTP and baseline diverged within the first \(prefixCount) tokens — SDPA shape-determinism boundary moved closer than the empirical 64-token floor"
        )

        print(
            "[MTPIteratorEndToEndDiagnostic longer-stream] mtpText: \(mtpText)"
        )
        print(
            "[MTPIteratorEndToEndDiagnostic longer-stream] baselineText: \(baselineText)"
        )
    }

    /// Diagnostic harness for Phase B's MTP-vs-baseline divergence at
    /// maxTokens=128. Compares baseline-against-baseline (two non-speculative
    /// runs against the same target, no MTP drafter) to determine whether the
    /// baseline path is itself bit-deterministic at this stream length, which
    /// is the load-bearing assumption that makes byte-identity a meaningful
    /// MTP correctness invariant.
    ///
    /// If both baseline runs produce identical text, baseline determinism is
    /// upheld and the Phase B divergence is genuinely MTP-vs-baseline drift.
    /// If they differ, the byte-identity framework's premise is broken at
    /// this stream length and the Phase B "failure" is reframed.
    @Test
    func testBaselineSelfDeterminism128() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let prompt = "Why is the sky blue? Explain in one paragraph."
        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        // First baseline run.
        let streamA = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 128, temperature: 0),
            context: loaded.context
        )
        var textA = ""
        for await event in streamA {
            if case .chunk(let chunk) = event { textA += chunk }
        }

        // Second baseline run against the same context.
        let streamB = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 128, temperature: 0),
            context: loaded.context
        )
        var textB = ""
        for await event in streamB {
            if case .chunk(let chunk) = event { textB += chunk }
        }

        print(
            "[BaselineSelfDeterminism128] textA: \(textA)"
        )
        print(
            "[BaselineSelfDeterminism128] textB: \(textB)"
        )

        #expect(
            textA == textB,
            "Baseline self-determinism violated at maxTokens=128 — the byte-identity invariant's premise is broken"
        )
    }

    /// Quantization-onset characterization. Exercises the iterator's R8/R13
    /// mitigation: when `maybeQuantizeKVCache` converts the full-attention
    /// `KVCacheSimple` layers to `QuantizedKVCache` mid-stream, Gemma 4's
    /// emit hook returns `sharedKV: nil`, the iterator's guard at
    /// `MTPSpeculativeTokenIterator.swift:207` fires, and
    /// `switchToPassthrough(reason: "main model did not emit drafter state")`
    /// engages for the remainder of the stream. This test verifies that
    /// designed mechanism actually engages end-to-end.
    ///
    /// `quantizedKVStart=40` is chosen so that a small number of speculation
    /// rounds (sky-blue chat-template prompt ≈ 30 tokens; +3-4 cache positions
    /// per blockSize=4 round) complete on the regular cache before
    /// quantization triggers around round 3.
    @Test
    func testMTP31BQuantizationOnsetEngagesPassthrough() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(
                maxTokens: 32,
                kvBits: 4,
                kvGroupSize: 64,
                quantizedKVStart: 40,
                temperature: 0
            ),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )

        var info: GenerateCompletionInfo?
        var text = ""
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .toolCall: break
            case .info(let completionInfo): info = completionInfo
            }
        }

        guard let info else {
            Issue.record("stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? -1
        let accepted = info.acceptedDraftTokens ?? -1
        print(
            "[quantization-onset 31B] proposed=\(proposed), accepted=\(accepted), passthrough=\(info.passthroughReason ?? "nil"), generated=\(info.generationTokenCount), text=\(text)"
        )

        // C2: at least one pre-quantization speculation round ran.
        #expect(
            (info.proposedDraftTokens ?? 0) > 0,
            "no tokens proposed — quantization may have triggered before the first speculation round; consider raising quantizedKVStart"
        )
        // C4: at least one pre-quantization draft was accepted.
        #expect(
            (info.acceptedDraftTokens ?? 0) > 0,
            "drafts proposed but none accepted before quantization — surprising given §5.1 acceptance numbers"
        )
        // C3: passthrough engaged due to quantization-induced sharedKV nil.
        #expect(
            info.passthroughReason != nil,
            "quantization did not engage passthrough — quantizedKVStart may be too high (no quantization triggered) or the emit hook still emits sharedKV after partial quantization"
        )
        #expect(
            (info.passthroughReason ?? "").contains("did not emit drafter state"),
            "unexpected passthrough reason: \(info.passthroughReason ?? "nil")"
        )
        // C5: stream filled the budget via the passthrough path.
        #expect(
            info.generationTokenCount == 32,
            "stream did not reach the 32-token budget: generated=\(info.generationTokenCount)"
        )
        // C6: passthrough produces real continuations.
        #expect(!text.isEmpty, "generated text is empty after passthrough engaged")
    }

    /// End-to-end regression for the LogitProcessor emit-only invariant.
    /// Locks in the behavior that `MTPSpeculativeTokenIterator.processor`
    /// (the canonical processor) receives `didSample(token:)` calls ONLY
    /// for emitted tokens (accepted drafts + correction per round, plus
    /// passthrough tokens if engaged). The verify loop's sampling, which
    /// runs on a struct value-copy (`verifyProcessorCopy` inside
    /// `speculateRound`), must NOT pollute the canonical processor.
    ///
    /// Uses the test-only `_setProcessorForTesting` / `_processorForTesting`
    /// accessors (guarded by `@_spi(Testing)`) to install a recording
    /// `EmissionLog` AFTER `init`. `init` calls `prepare()` internally,
    /// which also calls `didSample` — but it hits the parameters-derived
    /// processor (nil here, since no penalties are configured) rather than
    /// the EmissionLog. The probe records speculation-round emissions only.
    ///
    /// Assertion: `EmissionLog.recordedTokens` is a contiguous suffix of
    /// the emitted token stream. Equivalently, the recorded count is no
    /// more than the emitted count, and the contents match position-by-
    /// position from the recorded-sequence start.
    @Test
    func testMTPLogitProcessorReceivesOnlyEmittedTokens() async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        // Construct the iterator directly so we can install the EmissionLog
        // probe via the @_spi(Testing) setter AFTER init's prepare() has
        // run. The invariant we're testing is internal to the iterator's
        // verify/accept flow, not generate(...)'s task wrapping.
        var iter = try MTPSpeculativeTokenIterator(
            input: lmInput,
            mainModel: loaded.context.model,
            drafter: loaded.drafter,
            mainCache: nil,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            blockSize: 4
        )

        iter._setProcessorForTesting(EmissionLog())

        var emitted: [Int] = []
        while let token = iter.next() {
            emitted.append(token)
        }

        guard let log = iter._processorForTesting as? EmissionLog else {
            Issue.record("EmissionLog probe lost between install and drain")
            return
        }

        print(
            "[LogitProcessor emit-only] emitted=\(emitted.count) tokens, recorded=\(log.recordedTokens.count), passthroughReason=\(iter.passthroughReason ?? "nil")"
        )

        // Under the emit-only invariant, log.recordedTokens.count is close
        // to emitted.count — equal modulo: (a) 1–2 prepare-time bonuses
        // the probe missed (probe was installed AFTER init's prepare()
        // ran, so prepare's didSample hit the parameters-derived nil
        // processor instead), and (b) up to `blockSize - 1` trailing
        // entries from a final speculation round that began before but
        // overshot `maxTokens` (didSample fires inside speculateRound's
        // accept loop BEFORE pendingTokens.append; the maxTokens cap is
        // checked at next() boundaries, so the last round's tail can
        // exceed the emit budget).
        //
        // Under a regression where verify-loop didSample leaks from the
        // local copy into the canonical processor, log size inflates
        // by ~(numDraft + 1) / (accepted + 1) — roughly 2× at the
        // ~50% acceptance rate of this configuration. The tight upper
        // bound below catches that case.
        let blockSize = 4
        let upperBound = emitted.count + blockSize
        let lowerBound = emitted.count - 2
        #expect(
            log.recordedTokens.count <= upperBound,
            "log size \(log.recordedTokens.count) > emitted+overhang \(upperBound) — verify-loop didSample is leaking from `verifyProcessorCopy` into `self.processor`. The struct value-semantics scoping has regressed."
        )
        #expect(
            log.recordedTokens.count >= lowerBound,
            "log size \(log.recordedTokens.count) < emitted-bonuses \(lowerBound) — fewer recordings than expected. self.processor.didSample is not firing for some emitted tokens."
        )
        #expect(
            iter.passthroughReason == nil,
            "passthrough engaged unexpectedly: \(iter.passthroughReason ?? "")")
        #expect(
            log.recordedTokens.count >= 1,
            "no speculation-round didSamples — speculation did not run on the canonical processor")

        // Content sanity: log should overlap with emitted's speculation
        // suffix. Detect the prepare-bonus offset by matching log[0]
        // against the first 3 emitted positions (Gemma 4 yields 1 or 2
        // prepare bonuses), then verify the overlapping prefix matches.
        if let offset = (0 ..< Swift.min(3, emitted.count))
            .first(where: { emitted[$0] == log.recordedTokens[0] })
        {
            let commonCount = Swift.min(
                log.recordedTokens.count, emitted.count - offset)
            let logPrefix = Array(log.recordedTokens.prefix(commonCount))
            let emittedSlice = Array(emitted[offset ..< offset + commonCount])
            #expect(
                logPrefix == emittedSlice,
                "recorded didSample sequence diverged from emitted-stream at offset \(offset). Recorded prefix: \(logPrefix). Emitted slice: \(emittedSlice). Verify-loop didSample is leaking from the local copy into the canonical processor."
            )
        } else {
            Issue.record(
                "log[0]=\(log.recordedTokens[0]) not found in emitted[0..<3]=\(Array(emitted.prefix(3))) — content overlap broken"
            )
        }
    }

    /// E4B end-to-end MTP. Both E-series drafters have
    /// `use_ordered_embeddings=true` and route through
    /// `Gemma4AssistantMaskedEmbedder.callAsFunction` (the centroid-routed
    /// sparse head). With that head implemented, the pair must speculate and
    /// produce accepted drafts — not trap, not fall straight to passthrough.
    ///
    /// Env-gated (`TEST_E4B_PAIR`) because it requires the E4B target+drafter
    /// in the HF cache; skips (records an Issue) when they're absent.
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["TEST_E4B_PAIR"] != nil)
    )
    func testMTPE4BPairProducesAcceptedDrafts() async throws {
        try await runEseriesMTPEndToEnd(
            label: "E4B",
            targetModelId: "mlx-community/gemma-4-e4b-it-4bit",
            drafterModelId: "mlx-community/gemma-4-E4B-it-assistant-bf16"
        )
    }

    /// E2B counterpart of `testMTPE4BPairProducesAcceptedDrafts`.
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["TEST_E2B_PAIR"] != nil)
    )
    func testMTPE2BPairProducesAcceptedDrafts() async throws {
        try await runEseriesMTPEndToEnd(
            label: "E2B",
            targetModelId: "mlx-community/gemma-4-e2b-it-4bit",
            drafterModelId: "mlx-community/gemma-4-E2B-it-assistant-bf16"
        )
    }

    /// Shared body for byte-identity tests. Loads the 31B target+drafter,
    /// runs MTP and a baseline (non-speculative) `generate(...)` against the
    /// same context at temp=0, drains both streams, and asserts the MTP
    /// stream's output is byte-identical to baseline. Defense-in-depth
    /// assertions guard against the byte-identity check passing trivially
    /// (e.g., if MTP silently regressed to passthrough, its output would
    /// match baseline by definition but speculation wouldn't be doing
    /// anything).
    ///
    /// Reused unchanged across the rainbows, Swift, and longer-stream tests.
    /// Not used by `testMTP31BMatchesBaselineByteIdentical` to keep that
    /// commit's surface stable.
    private func runMTPVsBaselineByteIdentityCheck(
        prompt: String,
        maxTokens: Int,
        blockSize: Int = 4
    ) async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: target31BModelId,
            drafterModelId: drafter31BModelId,
            targetRevision: target31BRevision,
            drafterRevision: drafter31BRevision
        )

        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        // MTP run.
        let mtpStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: blockSize
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
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
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
            "[MTPIteratorEndToEndDiagnostic byte-identity prompt=\"\(prompt)\" maxTokens=\(maxTokens)] mtpText: \(mtpText)"
        )
        print(
            "[MTPIteratorEndToEndDiagnostic byte-identity prompt=\"\(prompt)\" maxTokens=\(maxTokens)] baselineText: \(baselineText)"
        )

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

    /// Shared body for the E-series failure-mode tests. Does NOT assert on
    /// the iterator's counters — the point is to characterize where (and how)
    /// the run fails, not to pass anything. Any output that survives to the
    /// info-event print is itself information about the failure mode.
    /// Runs an E-series target+drafter pair through the high-level MTP
    /// `generate(...)` API and asserts the centroid-routed drafter actually
    /// speculates: the stream completes with an `.info` event, produces
    /// non-empty text, proposes drafts, and gets at least one accepted
    /// (i.e. it didn't trap and didn't degrade straight to passthrough).
    private func runEseriesMTPEndToEnd(
        label: String,
        targetModelId: String,
        drafterModelId: String
    ) async throws {
        let loaded = try await loadTargetAndDrafter(
            targetModelId: targetModelId,
            drafterModelId: drafterModelId
        )

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
            case .chunk(let chunk): text += chunk
            case .toolCall: break
            case .info(let completionInfo): info = completionInfo
            }
        }

        guard let info else {
            Issue.record("[\(label)] stream completed without an .info event")
            return
        }
        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        print(
            "[MTPIteratorEndToEndDiagnostic \(label)] proposed=\(proposed), accepted=\(accepted), passthrough=\(info.passthroughReason ?? "nil"), generated=\(info.generationTokenCount) tokens, text: \(text)"
        )
        #expect(!text.isEmpty, "[\(label)] MTP produced no output")
        #expect(
            info.passthroughReason == nil,
            "[\(label)] degraded to passthrough: \(info.passthroughReason ?? "")")
        #expect(proposed > 0, "[\(label)] drafter proposed no tokens — centroid head not engaged")
        #expect(
            accepted > 0, "[\(label)] zero drafts accepted — drafter output diverges from target")
    }
}
