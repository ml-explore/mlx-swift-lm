// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
import os

@testable import MLXLLM
@testable import MLXLMCommon

struct Raon1MiniTests {

    // MARK: - Configuration

    @Test("The release configuration decodes with its layer layout and default effort")
    func configurationDecodes() throws {
        let config = try Self.configuration()
        #expect((0 ..< 3).allSatisfy(config.isRDALayer))
        #expect(!config.isRDALayer(3))
        #expect(!config.isMoELayer(0))
        #expect((1 ..< 4).allSatisfy(config.isMoELayer))
        #expect(config.rda.gateLowerBound == -5)
        #expect(config.effortSteering?.alpha(for: nil) == 0.05)
        try config.validateModelConfiguration()

        let softplus = try Self.configuration(gateLowerBound: "null")
        #expect(softplus.rda.gateLowerBound == nil)
    }

    @Test func rejectsZeroMoELayerFrequency() throws {
        var config = try Self.configuration()
        config.moeLayerFreq = 0
        #expect(throws: Raon1MiniError.self) { try config.validateModelConfiguration() }
    }

    @Test("Each reasoning effort resolves to its alpha; thinking off falls back to 0.1")
    func effortAlphas() throws {
        let steering = try #require(try Self.configuration().effortSteering)
        #expect(steering.alpha(for: nil) == 0.05)
        #expect(steering.alpha(for: .high) == 0.05)
        #expect(steering.alpha(for: .max) == 0)
        #expect(steering.alpha(for: .off) == 0.1)
        #expect(steering.alpha(for: .low) == 0.05)
    }

    @Test("The request's reasoning effort steers every forward of a token iterator")
    func effortSteersTheIterator() throws {
        let model = try Self.preparedModel()
        let prompt = LMInput(tokens: MLXArray((0 ..< 6).map { Int32($0 * 3 + 2) }))
        func step(_ effort: ReasoningEffort?) throws -> (MLXArray, LMOutput.State?) {
            var parameters = GenerateParameters(maxTokens: 3, temperature: 0)
            parameters.reasoningEffort = effort
            var iterator = try TokenIterator(input: prompt, model: model, parameters: parameters)
            _ = iterator.next()
            let next = model.nextTokenLogits(
                LMInput.Text(tokens: MLXArray([Int32(7)])[.newAxis]), cache: iterator.cache,
                state: iterator.state)
            return (next.logits, iterator.state)
        }

        let (high, defaultState) = try step(nil)
        let (explicitHigh, _) = try step(.high)
        let (off, offState) = try step(.off)
        let (unsteered, _) = try step(.max)

        #expect(defaultState?[reasoningEffortKey] == nil)
        #expect(offState?[reasoningEffortKey] == .off)
        #expect(arrayEqual(high, explicitHigh).item(Bool.self))
        #expect(!arrayEqual(high, off).item(Bool.self))
        #expect(!arrayEqual(high, unsteered).item(Bool.self))
    }

    @Test("A configuration whose layer lists do not partition the layers is rejected")
    func configurationRejectsOverlappingLayers() throws {
        let config = try Self.configuration(mlaLayers: "[3, 4]")
        #expect(throws: Raon1MiniError.self) { try config.validateModelConfiguration() }
    }

    @Test("raon1_mini resolves to the model with XML tool calls and a template-shaped budget close")
    func registryAndConventions() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.json().utf8), modelType: "raon1_mini")
        #expect(model is Raon1MiniModel)
        #expect(model.supportedReasoningEfforts == [.off, .high, .max])
        #expect(model.toolCallFormat == .xmlFunction)
        #expect(model.reasoningConfig?.startDelimiter == "<think>")
        #expect(
            model.reasoningConfig?.budgetTransition
                == ReasoningBudgetTransition(beforeEndDelimiter: "\n", afterEndDelimiter: "\n\n"))
    }

    @Test("Native discovery preserves checkpoint names and excludes inactive steering")
    func nativeEffortDiscovery() throws {
        var config = try Self.configuration()
        config.effortSteering?.alpha = ["deep": 0.25]
        #expect(Raon1MiniModel(config).supportedReasoningEfforts == [.off, .init("deep")])

        config.effortSteering?.layers = []
        #expect(Raon1MiniModel(config).supportedReasoningEfforts.isEmpty)

        config.effortSteering = nil
        #expect(Raon1MiniModel(config).supportedReasoningEfforts.isEmpty)
    }

    // MARK: - Checkpoint

    @Test(
        "sanitize merges the convolutions, splits kv_b_proj, fills a missing steering vector and drops extra layers"
    )
    func sanitizeConvertsTheReleaseLayout() throws {
        let config = try Self.configuration()
        let model = Raon1MiniModel(config)
        let channels = config.rdaHeads * config.rdaHeadDim
        let kvB = MLXRandom.normal([2 * (128 + 128), 64])
        let gateUp = MLXRandom.normal([8, 128, 256])
        let weights = try model.sanitize(weights: [
            "model.layers.0.self_attn.q_conv1d.weight": MLXArray.zeros([channels, 4]),
            "model.layers.0.self_attn.k_conv1d.weight": MLXArray.ones([channels, 4]),
            "model.layers.0.self_attn.v_conv1d.weight": MLXArray.zeros([channels, 4]),
            "model.layers.3.self_attn.kv_b_proj.weight": kvB,
            "model.layers.1.moe.switch_mlp.gate_up.weight": gateUp,
            "model.layers.1.moe.gate.weight": MLXArray.zeros([8, 256], dtype: .bfloat16),
            "model.layers.4.eh_proj.weight": MLXArray.zeros([4]),
        ])

        let convolution = try #require(weights["model.layers.0.self_attn.qkv_conv1d.weight"])
        #expect(convolution.shape == [3 * channels, 4, 1])
        #expect(convolution[channels ..< (2 * channels)].min().item(Float.self) == 1)

        let embedQ = try #require(weights["model.layers.3.self_attn.embed_q.weight"])
        let unembed = try #require(weights["model.layers.3.self_attn.unembed_out.weight"])
        #expect(embedQ.shape == [2, 64, 128])
        #expect(unembed.shape == [2, 128, 64])
        let heads = kvB.reshaped(2, 256, 64)
        #expect(arrayEqual(embedQ, heads[0..., ..<128].swappedAxes(-1, -2)).item(Bool.self))
        #expect(arrayEqual(unembed, heads[0..., 128...]).item(Bool.self))

        #expect(weights["model.layers.1.moe.switch_mlp.up_proj.weight"]?.shape == [8, 64, 256])
        #expect(
            arrayEqual(
                try #require(weights["model.layers.1.moe.switch_mlp.gate_proj.weight"]),
                gateUp[0..., 64...]
            ).item(Bool.self))
        #expect(weights["model.layers.1.moe.gate.weight"]?.dtype == .float32)

        let steering = try #require(weights["model.layers.2.moe.reasoning_effort_steering"])
        #expect(steering.dtype == .float32 && steering.shape == [256])
        #expect(weights["model.layers.1.moe.reasoning_effort_steering"] == nil)
        #expect(weights["model.layers.4.eh_proj.weight"] == nil)
    }

    // MARK: - Attention

    /// The cache keeps only the latent and the shared key channels. Expanding per-head keys for a
    /// prompt and attending in the latent space for decode must both equal attention over the
    /// full 192-wide keys built from `kv_b_proj`, as the Transformers reference computes it.
    @Test(
        "MLA prompt and decode attention equal attention over the uncompressed keys",
        arguments: [false, true])
    func mlaMatchesUncompressedAttention(quantizedCache: Bool) throws {
        let config = try Self.configuration()
        let model = Raon1MiniModel(config)
        Self.randomize(model, dtype: .float32)
        let kvB = withRandomState(MLXRandom.RandomState(seed: 3)) {
            MLXRandom.normal([2 * (128 + 128), 64]) * 0.1
        }
        let split = try model.sanitize(weights: [
            "model.layers.3.self_attn.kv_b_proj.weight": kvB
        ])
        let attention = try #require(model.model.layers[3].attention as? Raon1MLA)
        attention.update(
            parameters: ModuleParameters.unflattened([
                "embed_q.weight": try #require(split["model.layers.3.self_attn.embed_q.weight"]),
                "unembed_out.weight": try #require(
                    split["model.layers.3.self_attn.unembed_out.weight"]),
            ]))

        let expanded = Raon1MLA.minimumBlockRows + 8
        let length = expanded + 5
        let x = withRandomState(MLXRandom.RandomState(seed: 4)) {
            MLXRandom.normal([1, length, 256])
        }
        // The CPU computes float32 products exactly enough to compare the two forms.
        let (expected, actual) = Device.withDefaultDevice(.cpu) {
            // An expanded prompt chunk, a latent chunk over cached keys, then two decode steps.
            let cache: KVCache = quantizedCache ? QuantizedKVCache(groupSize: 32) : KVCacheSimple()
            let chunks = [0 ..< expanded, expanded ..< (length - 2)].map {
                attention(x[0..., $0], mask: .causal, cache: cache)
            }
            let steps = ((length - 2) ..< length).map {
                attention(x[0..., $0 ..< ($0 + 1)], mask: .none, cache: cache)
            }
            return (
                Self.uncompressedAttention(attention, x, kvB: kvB),
                concatenated(chunks + steps, axis: 1)
            )
        }

        #expect(Self.relativeDrift(actual, expected) < (quantizedCache ? 1e-2 : 1e-5))
    }

    @Test("Packed experts and the optional shortlist preserve the model's parameter count")
    func parameterCount() throws {
        let dense: Module = Raon1MiniModel(try Self.configuration())
        let packed = Raon1MiniModel(try Self.configuration(quantized: true))
        let denseExpertWeights = 8 * 256 * 64
        #expect(
            Raon1ExpertProjection(
                experts: 8, inputDims: 256, outputDims: 64,
                quantization: packed.configuration.quantization
            ).numParameters() == denseExpertWeights)
        #expect(packed.numParameters() == dense.numParameters())

        try packed.update(
            modules: ModuleChildren.unflattened([
                "lm_head_stage1": QuantizedLinear(
                    256, 64, bias: false, groupSize: 64, bits: 2)
            ]), verify: .none)
        #expect(packed.numParameters() == dense.numParameters())
    }

    // MARK: - Decode paths

    @Test(
        "Rejecting a verify suffix preserves the convolution and recurrent state",
        arguments: [3, 16]
    )
    func speculativeRewindMatchesRetainedPrefix(rejected: Int) throws {
        let model = try Self.preparedModel()
        let tokens = MLXArray((0 ..< (5 + rejected)).map { Int32(($0 * 7 + 3) % 64) })[.newAxis]
        let rewound = try model.newCache(parameters: nil)
        let reference = try model.newCache(parameters: nil)

        for cache in [rewound, reference] {
            eval(model(tokens[0..., ..<4], cache: cache))
        }
        eval(model(tokens[0..., 4 ..< (5 + rejected)], cache: rewound))
        eval(model(tokens[0..., 4 ..< 5], cache: reference))
        #expect(rewindSpeculativePromptCache(rewound, numTokens: rejected) == rejected)

        let next = tokens[0..., 5 ..< 6]
        let actual = model(next, cache: rewound)
        let expected = model(next, cache: reference)
        #expect(Self.relativeDrift(actual, expected) < 1e-2)
    }

    @Test(
        "DFlash2 rejects a suffix and drops undelivered tokens before the next turn",
        arguments: [2, 512]
    )
    func dflashRewindsAfterEarlyStop(prefillStepSize: Int) throws {
        let model = try Self.preparedModel()
        let draftConfiguration = Raon1DFlash2Configuration(
            architectures: ["DFlash2DraftModel"], hiddenSize: 256, hiddenLayers: 1,
            attentionHeads: 2, kvHeads: 1, headDimension: 128, intermediateSize: 128,
            rmsNormEps: 1e-5, vocabularySize: 64, targetLayerCount: 4,
            rope: .init(theta: 10_000),
            dflash: .init(
                blockSize: 8, maskTokenID: 0, convolutionTaps: 2, convolutionGroupSize: 64,
                selectorRank: 8, selectorTopK: 1, window: 32,
                targetLayerIDs: [3], targetStreamsPerLayer: model.configuration.mhcStreams))
        let drafter = Raon1DFlash2DraftModel(draftConfiguration)
        let proposal = Linear(
            weight: MLXArray.zeros([64, 256]),
            bias: (MLXArray(0 ..< 64) .== 7).asType(.float32))
        try drafter.update(
            modules: ModuleChildren.unflattened(["head": proposal]), verify: .none)

        let prompt: [Int32] = [1, 2, 3, 4]
        var parameters = GenerateParameters(maxTokens: 32, temperature: 0)
        parameters.prefill.stepSize = prefillStepSize
        let progress = OSAllocatedUnfairLock(initialState: [Int]())
        parameters.prefill.progress = { processed, _ in
            progress.withLock { $0.append(processed) }
        }
        let cache = try model.newCache(parameters: nil)
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), mainModel: model, drafter: drafter,
            mainCache: cache, parameters: parameters, blockSize: 8,
            components: GenerationComponents().appendingLogitProcessor { DraftAcceptance() })
        #expect(progress.withLock { $0 } == (prefillStepSize == 2 ? [2, 3, 4] : [4]))
        #expect(iterator.next() == 7)
        #expect(iterator.next() == 7)
        #expect(iterator.proposedCount == 7)
        #expect(iterator.acceptedCount == 2)
        iterator.finalizeGeneration()
        #expect(cache.allSatisfy { $0.offset == prompt.count + 2 })

        let suffix: [Int32] = [10, 11]
        let actual = model(MLXArray(suffix)[.newAxis], cache: cache)
        let expected = model(
            MLXArray(prompt + [7, 7] + suffix)[.newAxis], cache: try model.newCache(parameters: nil)
        )
        #expect(Self.relativeDrift(actual, expected[0..., (-2)...]) < 1e-2)
    }

    private struct DraftAcceptance: LogitProcessor {
        private var sampled = 0

        mutating func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            let forced = (MLXArray(0 ..< logits.dim(-1)) .== (sampled < 3 ? 7 : 9))
            return broadcast(forced.asType(logits.dtype), to: logits.shape)
        }

        mutating func didSample(token: MLXArray) { sampled += 1 }
    }

    @Test("Token-by-token decode reproduces the logits of one prompt pass")
    func decodeMatchesPrompt() throws {
        let model = try Self.preparedModel()
        let tokens = MLXArray((0 ..< 12).map { Int32(($0 * 7 + 3) % 64) })[.newAxis]

        let full = model(tokens, cache: try model.newCache(parameters: nil))
        let cache = try model.newCache(parameters: nil)
        let prompt = model(tokens[0..., ..<8], cache: cache)
        let steps = (8 ..< 12).map { model(tokens[0..., $0 ..< ($0 + 1)], cache: cache) }
        let stepwise = concatenated([prompt[0..., 7...]] + steps, axis: 1)

        #expect(Self.relativeDrift(stepwise, full[0..., 7...]) < 1e-2)
        #expect(
            argMax(stepwise, axis: -1).asArray(Int32.self)
                == argMax(full[0..., 7...], axis: -1).asArray(Int32.self))
    }

    @Test("The mHC kernels match the compiled operations")
    func mhcKernelsMatchOperations() throws {
        let model = try Self.preparedModel()
        let site = model.model.layers[1].mlpMHC
        let norm = model.model.layers[1].postAttentionLayerNorm
        let (rows, branch, post, residual) = withRandomState(MLXRandom.RandomState(seed: 5)) {
            (
                MLXRandom.normal([3, 4, 256]).asType(.bfloat16),
                MLXRandom.normal([3, 256]).asType(.bfloat16),
                MLXRandom.uniform(low: 0, high: 2, [3, 4]),
                MLXRandom.uniform(low: 0, high: 0.5, [3, 4, 4])
            )
        }
        let pending = Raon1PendingBranch(output: branch, post: post, residual: residual)
        let kernels = site.enter(rows, pending: pending, norm: norm)
        let operations = site.enterThroughOperations(rows, pending: pending, norm: norm)

        #expect(Raon1MHCKernels.supports(rows))
        #expect(Self.relativeDrift(kernels.rows, operations.rows) < 1e-3)
        #expect(Self.relativeDrift(kernels.x, operations.x) < 1e-3)
        #expect(Self.relativeDrift(kernels.post, operations.post) < 1e-4)
        #expect(Self.relativeDrift(kernels.residual, operations.residual) < 1e-4)
    }

    @Test("The prompt recurrence kernel matches the gated delta operations")
    func promptRecurrenceMatchesOperations() throws {
        let (q, k, v, g, beta, state) = withRandomState(MLXRandom.RandomState(seed: 6)) {
            (
                MLXRandom.normal([1, 10, 2, 128]).asType(.bfloat16) * 0.1,
                MLXRandom.normal([1, 10, 2, 128]).asType(.bfloat16) * 0.1,
                MLXRandom.normal([1, 10, 2, 128]).asType(.bfloat16),
                MLXRandom.uniform(low: 0.8, high: 1.0, [1, 10, 2, 128]),
                MLXRandom.uniform(low: 0, high: 1, [1, 10, 2]).asType(.bfloat16),
                MLXRandom.normal([1, 2, 128, 128]) * 0.1
            )
        }
        #expect(Raon1RDAKernels.supportsPrompt(q: q, v: v, g: g, state: state))
        let (y, next) = Raon1RDAKernels.prompt(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let (yOps, nextOps) = gatedDeltaRecurrence(
            q: q, k: k, v: v, g: g, beta: beta, state: state, useKernel: false)

        #expect(Self.relativeDrift(y, yOps) < 1e-2)
        #expect(Self.relativeDrift(next, nextOps) < 1e-4)
    }

    @Test("A fused projection returns the bits of its parts and keeps a single copy")
    func fusedProjectionMatchesParts() throws {
        let parts = (0 ..< 3).map { index in
            withRandomState(MLXRandom.RandomState(seed: UInt64(10 + index))) {
                Self.scaledLinear(input: 64, output: 16 * (index + 1))
            }
        }
        let x = withRandomState(MLXRandom.RandomState(seed: 9)) {
            MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
        }
        let expected = parts.map { $0(x) }

        let slot = Raon1FusedSlot()
        slot.prepare(parts)
        let fused = slot.project(x, parts, training: false)

        #expect(fused.count == 3)
        for (lhs, rhs) in zip(fused, expected) {
            #expect(arrayEqual(lhs, rhs).item(Bool.self))
        }
        for (part, rhs) in zip(parts, expected) {
            #expect(arrayEqual(part(x), rhs).item(Bool.self))
        }
    }

    @Test func mixedProjectionsPreserveAllOutputs() {
        let parts: [Linear] = [
            Self.scaledLinear(input: 64, output: 16),
            Linear(64, 32, bias: false),
            Self.scaledLinear(input: 64, output: 48),
        ]
        let slot = Raon1FusedSlot()
        slot.prepare(parts)
        let x = MLXArray.ones([1, 1, 64])
        #expect(slot.project(x, parts, training: false).map { $0.dim(-1) } == [16, 32, 48])
    }

    @Test(arguments: ["weight", "global_scale"])
    func directChildUpdateBypassesFusedProjection(key: String) {
        let first = Self.scaledLinear(input: 64, output: 16)
        let parts = [first, Self.scaledLinear(input: 64, output: 32)]
        let slot = Raon1FusedSlot()
        slot.prepare(parts)
        let x = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        let before = slot.project(x, parts, training: false)[0]
        eval(before)
        let replacement =
            key == "weight"
            ? MLXArray.zeros(first.weight.shape, dtype: first.weight.dtype) : MLXArray(Float(2))
        first.update(parameters: ModuleParameters.unflattened([(key, replacement)]))
        let after = slot.project(x, parts, training: false)[0]
        #expect(!arrayEqual(before, after).item(Bool.self))
        #expect(arrayEqual(after, first(x)).item(Bool.self))
    }

    @Test("A parameter update drops the prepared state, so the next forward reads the update")
    func updateInvalidatesPreparedState() throws {
        let model = try Self.preparedModel()
        let tokens = MLXArray([Int32(5)])[.newAxis]
        let before = model(tokens, cache: try model.newCache(parameters: nil))

        let site = model.model.layers[1].mlpMHC
        site.update(
            parameters: ModuleParameters.unflattened([
                "b_post": MLXArray.zeros(like: site.bPost) + 4
            ]))
        let after = model(tokens, cache: try model.newCache(parameters: nil))
        try model.prepare()
        let prepared = model(tokens, cache: try model.newCache(parameters: nil))

        #expect(Self.relativeDrift(before, after) > 1e-2)
        #expect(Self.relativeDrift(after, prepared) < 1e-2)
    }

    @Test("The MoE decode kernels match the routed and shared expert operations")
    func moeKernelsMatchOperations() throws {
        let config = try Self.configuration(hidden: 512, experts: 32, quantized: true)
        let moe = withRandomState(MLXRandom.RandomState(seed: 20)) {
            Raon1MoE(
                config, steered: false, arithmetic: Raon1Arithmetic(config),
                linear: { _, input, output in Self.scaledLinear(input: input, output: output) })
        }
        withRandomState(MLXRandom.RandomState(seed: 21)) {
            var parameters = [String: MLXArray]()
            for name in ["up_proj", "gate_proj", "down_proj"] {
                let shape = name == "down_proj" ? [32, 512, 64] : [32, 64, 512]
                let (weight, scales, _) = MLX.quantized(
                    MLXRandom.normal(shape), groupSize: 16, bits: 4, mode: .nvfp4)
                parameters["switch_mlp.\(name).weight"] = weight
                parameters["switch_mlp.\(name).scales"] = scales
                parameters["switch_mlp.\(name).global_scale"] =
                    MLXRandom.uniform(low: 1000, high: 3000, [32])
            }
            parameters["gate.weight"] = MLXRandom.normal([32, 512]) * 0.05
            parameters["gate.e_score_correction_bias"] = MLXRandom.normal([32]) * 0.01
            moe.update(parameters: ModuleParameters.unflattened(parameters))
        }
        let x = withRandomState(MLXRandom.RandomState(seed: 22)) {
            MLXRandom.normal([2, 1, 512]).asType(.bfloat16)
        }

        moe.train(false)
        let kernels = moe(x, effort: 0)
        moe.train(true)
        let operations = moe(x, effort: 0)

        #expect(Self.relativeDrift(kernels, operations) < 1e-2)
    }

    @Test("The shortlist head keeps the full head's arg max and its exact top logits")
    func shortlistHeadKeepsTheTopLogits() throws {
        let model = Raon1MiniModel(try Self.configuration(vocabulary: 65_536))
        let (weight, h) = withRandomState(MLXRandom.RandomState(seed: 30)) {
            (
                MLXRandom.normal([65_536, 256]) * 0.1,
                MLXRandom.normal([1, 3, 256]).asType(.bfloat16)
            )
        }
        let head = QuantizedLinear(weight: weight, bias: nil, groupSize: 32, bits: 8, mode: .mxfp8)
        let shortlist = QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 8)
        model.update(
            modules: ModuleChildren.unflattened(["lm_head": head, "lm_head_stage1": shortlist]))
        model.train(false)

        // One row per product, as the shortlist recomputes each row.
        let full = concatenated((0 ..< 3).map { head(h[0..., $0 ..< ($0 + 1)]) }, axis: 1)
            .asType(.float32)[0]
        let logits = try #require(model.shortlistLogits(h)).asType(.float32)[0]
        let top = argSort(-full, axis: -1)[0..., ..<Raon1MiniModel.shortlistTailRank]

        #expect(
            argMax(logits, axis: -1).asArray(Int32.self)
                == argMax(full, axis: -1).asArray(Int32.self))
        #expect(
            arrayEqual(takeAlong(logits, top, axis: -1), takeAlong(full, top, axis: -1)).item(
                Bool.self))
        let cap = takeAlong(full, top[0..., (-1)...], axis: -1)
        #expect((logits .<= takeAlong(full, top[0..., ..<1], axis: -1)).all().item(Bool.self))
        #expect(logicalOr(logits .<= cap, logits .== full).all().item(Bool.self))
    }

    @Test("Generation uses the exact head by default and shortlists only on opt-in")
    func shortlistServesOnlyNextTokenLogits() throws {
        let model = try Self.preparedModel(vocabulary: 65_536)
        let weight = try #require(model.lmHead).weight
        let head = QuantizedLinear(weight: weight, bias: nil, groupSize: 32, bits: 8, mode: .mxfp8)
        let shortlist = QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 2)
        model.update(
            modules: ModuleChildren.unflattened(["lm_head": head, "lm_head_stage1": shortlist]))
        let tokens = MLXArray((0 ..< 6).map { Int32($0 * 5 + 1) })[.newAxis]

        let hidden = model.hiddenStates(tokens, cache: try model.newCache(parameters: nil))
        let full = model(tokens, cache: try model.newCache(parameters: nil))
        var parameters = GenerateParameters()
        parameters.logitsAccuracy = .allowApproximate
        let iterator = try TokenIterator(
            input: LMInput(tokens: tokens.reshaped(-1)), model: model, parameters: parameters)
        let last = hidden[0..., (-1)...]
        let exact = model.nextTokenLogits(
            LMInput.Text(tokens: tokens), cache: try model.newCache(parameters: nil), state: nil
        ).logits
        let approximate = model.nextTokenLogits(
            LMInput.Text(tokens: tokens), cache: try model.newCache(parameters: nil),
            state: iterator.state
        ).logits

        #expect(arrayEqual(full, model.projectLogits(hidden)).item(Bool.self))
        #expect(arrayEqual(exact, model.projectLogits(last)).item(Bool.self))
        let expected = try #require(model.shortlistLogits(last))
        #expect(arrayEqual(approximate, expected).item(Bool.self))
        #expect(!arrayEqual(approximate, exact).item(Bool.self))
    }

    @Test("The candidate search returns the largest values of each row")
    func largestFindsTheTopValues() {
        // Distinct values, so the largest 256 are one set.
        let scores = withRandomState(MLXRandom.RandomState(seed: 31)) {
            argSort(MLXRandom.uniform(low: 0, high: 1, [2, 65_536]), axis: -1).asType(.float32)
        }
        let found = sorted(Raon1MiniModel.largest(256, of: scores).asType(.int32), axis: -1)
        let expected = sorted(argSort(-scores, axis: -1)[0..., ..<256].asType(.int32), axis: -1)
        #expect(arrayEqual(found, expected).item(Bool.self))
    }

    @Test("A checkpoint attaches the shortlist head only when its tensors match lm_head")
    func checkpointAttachesTheShortlistHead() throws {
        let config = try Self.configuration(vocabulary: 65_536)
        let tensors: [String: MLXArray] = [
            "weight": MLXArray.zeros([65_536, 16], dtype: .uint32),
            "scales": MLXArray.zeros([65_536, 4], dtype: .bfloat16),
            "biases": MLXArray.zeros([65_536, 4], dtype: .bfloat16),
        ]

        let model = Raon1MiniModel(config)
        let checkpoint = try model.prepareCheckpoint(ModelCheckpoint(weights: tensors))
        #expect(model.shortlistHead?.bits == 2)
        #expect(checkpoint.weights["lm_head_stage1.weight"]?.shape == [65_536, 16])
        #expect(checkpoint.weights["weight"] == nil)

        let mismatched = Raon1MiniModel(config)
        var wrong = tensors
        wrong["scales"] = MLXArray.zeros([65_536, 2], dtype: .bfloat16)
        let dropped = try mismatched.prepareCheckpoint(ModelCheckpoint(weights: wrong))
        #expect(mismatched.shortlistHead == nil)
        #expect(
            dropped.weights.keys.allSatisfy { !$0.hasPrefix("lm_head_stage1") && $0 != "weight" })
    }

    // MARK: - Helpers

    private static func configuration(
        gateLowerBound: String = "-5.0", mlaLayers: String = "[4]", hidden: Int = 256,
        experts: Int = 8, vocabulary: Int = 64, quantized: Bool = false
    ) throws -> Raon1MiniConfiguration {
        try JSONDecoder().decode(
            Raon1MiniConfiguration.self,
            from: Data(
                json(
                    gateLowerBound: gateLowerBound, mlaLayers: mlaLayers, hidden: hidden,
                    experts: experts, vocabulary: vocabulary, quantized: quantized
                ).utf8))
    }

    private static func json(
        gateLowerBound: String = "-5.0", mlaLayers: String = "[4]", hidden: Int = 256,
        experts: Int = 8, vocabulary: Int = 64, quantized: Bool = false
    ) -> String {
        let quantization =
            quantized ? #""quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"},"# : ""
        return """
            {
              "model_type": "raon1_mini",\(quantization)
              "hidden_size": \(hidden),
              "num_hidden_layers": 4,
              "num_attention_heads": 2,
              "head_dim": 128,
              "vocab_size": \(vocabulary),
              "intermediate_size": 128,
              "num_experts": \(experts),
              "num_experts_per_token": 2,
              "moe_intermediate_size": 64,
              "num_shared_experts": 1,
              "first_k_dense_replace": 1,
              "q_lora_rank": 32,
              "kv_lora_rank": 64,
              "qk_nope_head_dim": 128,
              "qk_rope_head_dim": 64,
              "v_head_dim": 128,
              "rda_config": {
                "num_heads": 2, "head_dim": 128, "conv_kernel_size": 4,
                "full_rank_gate": true, "gate_lower_bound": \(gateLowerBound),
                "rda_layers": [1, 2, 3], "mla_layers": \(mlaLayers)
              },
              "reasoning_effort_steering": {
                "alpha": {"none": 0, "high": 0.05, "max": 0}, "default": "high", "layers": [2]
              },
              "tie_word_embeddings": false
            }
            """
    }

    /// A tiny bfloat16 model with random parameters, prepared for inference as the factory does.
    private static func preparedModel(vocabulary: Int = 64) throws -> Raon1MiniModel {
        let model = Raon1MiniModel(try configuration(vocabulary: vocabulary))
        model.train(false)
        randomize(model, dtype: .bfloat16)
        try model.prepare()
        eval(model)
        return model
    }

    /// Replaces every parameter with a seeded random value. The parameters KRAFTON keeps in
    /// float32 stay float32.
    private static func randomize(_ model: Raon1MiniModel, dtype: DType) {
        let float32 = [
            "A_log", "dt_bias", "e_score_correction_bias", "eta", ".gate.weight",
            "reasoning_effort_steering",
        ]
        withRandomState(MLXRandom.RandomState(seed: 1)) {
            let parameters = model.parameters().flattened().map { key, value in
                let scale: Float =
                    key.contains("embed_tokens") ? 1 : key.contains("conv1d") ? 0.3 : 0.05
                var random = MLXRandom.normal(value.shape) * scale
                if key.contains("norm") { random = random + 1 }
                let keepsFloat32 = float32.contains { key.hasSuffix($0) }
                return (key, random.asType(keepsFloat32 ? .float32 : dtype))
            }
            model.update(parameters: ModuleParameters.unflattened(parameters))
        }
    }

    private static func scaledLinear(input: Int, output: Int) -> Raon1ScaledLinear {
        let linear = Raon1ScaledLinear(input, output)
        let (weight, scales, _) = MLX.quantized(
            MLXRandom.normal([output, input]), groupSize: 16, bits: 4, mode: .nvfp4)
        linear.update(
            parameters: ModuleParameters.unflattened([
                "weight": weight, "scales": scales,
                "global_scale": MLXRandom.uniform(low: 0.5, high: 2, []),
            ]))
        return linear
    }

    /// The Transformers reference: keys `[k_nope, k_shared]` per head, causal softmax attention.
    private static func uncompressedAttention(_ attention: Raon1MLA, _ x: MLXArray, kvB: MLXArray)
        -> MLXArray
    {
        let (B, L, heads) = (x.dim(0), x.dim(1), attention.heads)
        let q = attention.qBProj(attention.qANorm(attention.qAProj(x)))
            .reshaped(B, L, heads, 192).transposed(0, 2, 1, 3)
        let kv = attention.kvAProj(x)
        let latent = attention.kvANorm(kv[.ellipsis, ..<64])
        let shared = kv[.ellipsis, 64...]
        let expanded = matmul(latent, kvB.T).reshaped(B, L, heads, 256).transposed(0, 2, 1, 3)
        let keys = concatenated(
            [
                expanded[.ellipsis, ..<128],
                broadcast(shared.expandedDimensions(axis: 1), to: [B, heads, L, 64]),
            ], axis: -1)
        let values = expanded[.ellipsis, 128...]
        var scores = matmul(q, keys.swappedAxes(-1, -2)) * attention.scale
        scores = MLX.where(createCausalMask(n: L, offset: 0), scores, -Float.infinity)
        var out = matmul(softmax(scores, axis: -1), values).transposed(0, 2, 1, 3).reshaped(
            B, L, -1)
        out = out * sigmoid(attention.gProj!(x))
        return attention.oProj(out)
    }

    /// The largest difference as a fraction of the largest magnitude.
    private static func relativeDrift(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let (left, right) = (lhs.asType(.float32), rhs.asType(.float32))
        let drift = abs(left - right).max().item(Float.self)
        let scale = maximum(abs(left).max(), abs(right).max()).item(Float.self)
        return scale > 0 ? drift / scale : drift
    }
}
