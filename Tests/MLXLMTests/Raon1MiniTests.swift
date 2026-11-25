// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

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
        #expect(config.effortSteering?.defaultAlpha == 0.05)
        try config.validateModelConfiguration()

        let softplus = try Self.configuration(gateLowerBound: "null")
        #expect(softplus.rda.gateLowerBound == nil)
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
        #expect(model.toolCallFormat == .xmlFunction)
        #expect(model.reasoningConfig?.startDelimiter == "<think>")
        #expect(
            model.reasoningConfig?.budgetTransition
                == ReasoningBudgetTransition(beforeEndDelimiter: "\n", afterEndDelimiter: "\n\n"))
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
    @Test("MLA prompt and decode attention equal attention over the uncompressed keys")
    func mlaMatchesUncompressedAttention() throws {
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

        let x = withRandomState(MLXRandom.RandomState(seed: 4)) { MLXRandom.normal([1, 8, 256]) }
        // The CPU computes float32 products exactly enough to compare the two forms.
        let (expected, actual) = Device.withDefaultDevice(.cpu) {
            // Two prompt chunks, so the second attends to cached keys, then two decode steps.
            let cache = KVCacheSimple()
            let chunks = [0 ..< 4, 4 ..< 6].map {
                attention(x[0..., $0], mask: .causal, cache: cache)
            }
            let steps = (6 ..< 8).map {
                attention(x[0..., $0 ..< ($0 + 1)], mask: .none, cache: cache)
            }
            return (
                Self.uncompressedAttention(attention, x, kvB: kvB),
                concatenated(chunks + steps, axis: 1)
            )
        }

        #expect(Self.relativeDrift(actual, expected) < 1e-5)
    }

    // MARK: - Decode paths

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

    // MARK: - Helpers

    private static func configuration(
        gateLowerBound: String = "-5.0", mlaLayers: String = "[4]"
    ) throws -> Raon1MiniConfiguration {
        try JSONDecoder().decode(
            Raon1MiniConfiguration.self,
            from: Data(json(gateLowerBound: gateLowerBound, mlaLayers: mlaLayers).utf8))
    }

    private static func json(gateLowerBound: String = "-5.0", mlaLayers: String = "[4]") -> String {
        """
        {
          "model_type": "raon1_mini",
          "hidden_size": 256,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "head_dim": 128,
          "vocab_size": 64,
          "intermediate_size": 128,
          "num_experts": 8,
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
    private static func preparedModel() throws -> Raon1MiniModel {
        let model = Raon1MiniModel(try configuration())
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
