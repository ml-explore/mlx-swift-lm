//  NemotronLabsDiffusion.swift
//  mlx-swift-lm
//
//  Created by Sachin Desai on 5/21/26.
//

// The model is a Ministral3-style transformer with a separate `diffusion_head`
// LM projection. Inference supports three modes that all share the same weights
// and only differ in attention pattern:
//   - AR: causal mask, single-token autoregressive decoding
//   - Diffusion: bidirectional attention, block-wise iterative denoising
//   - Linear self-speculation: diffusion drafts a block, then AR verifies
//
// AR mode uses the standard `LLMModel` pipeline (causal mask is the default and
// the model exposes a normal `callAsFunction(_:cache:)`). The diffusion and
// linear-spec generators are exposed as model-level methods that take token
// arrays directly.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Llama-4 Attention Scaling
//
// Identical to Mistral3Text.swift's helper but kept private here to keep the
// file self-contained — the existing helper there is `private` so it cannot be
// reused.

private func nlsLlama4AttentionScale(
    start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = arange(start, stop)
    let scaling =
        1 + beta
        * MLX.log(
            1 + MLX.floor(positions.asType(.float32) / Float(maxPositionEmbeddings))
        )
    return scaling[0..., .newAxis]
}

// MARK: - Attention

class NemotronLabsDiffusionAttention: Module {
    let args: NemotronLabsDiffusionConfiguration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: NemotronLabsDiffusionConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: args.attentionBias)

        let ropeTheta = args.ropeParameters?["rope_theta"]?.asFloat() ?? args.ropeTheta
        self.rope = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: false,
            scalingConfig: args.ropeParameters,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        queries = queries * attnScale

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }
}

// MARK: - MLP

class NemotronLabsDiffusionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: NemotronLabsDiffusionConfiguration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize
        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: args.mlpBias)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: args.mlpBias)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder Layer

class NemotronLabsDiffusionDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NemotronLabsDiffusionAttention
    @ModuleInfo(key: "mlp") var mlp: NemotronLabsDiffusionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: NemotronLabsDiffusionConfiguration) {
        self._attention.wrappedValue = NemotronLabsDiffusionAttention(args)
        self._mlp.wrappedValue = NemotronLabsDiffusionMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Encoder (Inner Model)

/// The Ministral3 backbone. Weights live under `encoder.*` in the safetensors,
/// matching the upstream Python class name `Ministral3Model` exposed as
/// `self.encoder` on `NemotronLabsDiffusionModel`.
public class NemotronLabsDiffusionEncoder: Module {
    let args: NemotronLabsDiffusionConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [NemotronLabsDiffusionDecoderLayer]
    let norm: RMSNorm

    init(_ args: NemotronLabsDiffusionConfiguration) {
        self.args = args

        precondition(args.vocabularySize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in
            NemotronLabsDiffusionDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    /// Forward pass.
    /// - Parameters:
    ///   - inputs: token ids, shape `[B, L]`
    ///   - cache: optional per-layer KV cache
    ///   - bidirectional: if true, attention is fully bidirectional (no mask).
    ///     Used in diffusion mode. AR / verify paths leave this `false`.
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        bidirectional: Bool = false
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let offset = cache?[0].offset ?? 0

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if bidirectional {
            // Diffusion mode: full attention.
            mask = .none
        } else {
            mask = createAttentionMask(h: h, cache: cache?[0])
        }

        // Llama-4 style query scaling, applied per query position.
        let attnScale: MLXArray
        if let ropeParams = args.ropeParameters,
            let beta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
            let origMax = ropeParams["original_max_position_embeddings"]?.asInt()
        {
            attnScale = nlsLlama4AttentionScale(
                start: offset,
                stop: offset + inputs.dim(1),
                beta: beta,
                maxPositionEmbeddings: origMax
            ).asType(h.dtype)
        } else {
            attnScale = MLXArray.ones([inputs.dim(1), 1]).asType(h.dtype)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-level Model

/// `nvidia/Nemotron-Labs-Diffusion-3B` and family.
///
/// AR generation works through the standard `LLMModel` pipeline. Diffusion and
/// linear-spec modes are exposed via dedicated methods.
public class NemotronLabsDiffusionModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let args: NemotronLabsDiffusionConfiguration
    public let encoder: NemotronLabsDiffusionEncoder

    @ModuleInfo(key: "diffusion_head") var diffusionHead: Linear

    /// Cached list of LoRA layers in the model, populated lazily on the
    /// first toggle. Walking `namedModules()` on every draft/verify boundary
    /// in `linearSpecGenerate` was costing ~0.5 ms per call.
    private var cachedLoraLayers: [LoRALayer]?

    public init(_ args: NemotronLabsDiffusionConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.encoder = NemotronLabsDiffusionEncoder(args)
        self._diffusionHead.wrappedValue = Linear(
            args.hiddenSize, args.vocabularySize, bias: false)
        super.init()
    }

    /// AR forward pass used by the `LLMModel` pipeline (causal attention).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = encoder(inputs, cache: cache, bidirectional: false)
        return diffusionHead(h)
    }

    /// Bidirectional forward pass, returning logits. Used for diffusion drafting.
    public func bidirectionalLogits(_ inputs: MLXArray) -> MLXArray {
        let h = encoder(inputs, cache: nil, bidirectional: true)
        return diffusionHead(h)
    }

    /// Bidirectional forward over only the block, attending to a causally
    /// prefilled KV cache (the prompt + previously committed blocks). The
    /// cache is updated with the block's K/V — callers must `trim` the cache
    /// by `inputs.dim(1)` after the forward to discard speculative entries.
    public func bidirectionalLogits(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
        let h = encoder(inputs, cache: cache, bidirectional: true)
        return diffusionHead(h)
    }

    /// Reset the cached LoRA-layer list. Call after attaching, fusing, or
    /// removing an adapter so the next toggle re-discovers layers.
    public func invalidateLoRACache() {
        cachedLoraLayers = nil
    }

    /// Fast LoRA toggle. Walks `namedModules()` once on first call (or after
    /// `invalidateLoRACache()`) and caches the resulting `[LoRALayer]` list.
    fileprivate func setLoRAEnabledFast(_ enabled: Bool) {
        if cachedLoraLayers == nil {
            var found: [LoRALayer] = []
            for (_, module) in self.namedModules() {
                if let layer = module as? LoRALayer {
                    found.append(layer)
                }
            }
            cachedLoraLayers = found
        }
        // Empty cache means no LoRA layers attached — nothing to do.
        guard let layers = cachedLoraLayers, !layers.isEmpty else { return }
        for layer in layers {
            // `LoRALayer` inherits from `Module` (a class), so the setter
            // mutates the class instance through the protocol existential.
            layer.loraEnabled = enabled
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Strip:
        // - rotary inv_freq buffers (recomputed from rope_theta at runtime)
        // - PEFT-format LoRA adapter weights (`base_model.*`) when the
        //   adapter directory lives inside the model directory and the
        //   standard loadWeights enumerator picks them up. The LoRA adapter
        //   is loaded separately via LoRAContainer.fromPEFT(...).
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq")
                && !$0.key.hasPrefix("base_model.")
        }
    }
}

// MARK: - LoRA

extension NemotronLabsDiffusionModel: LoRAModel {
    public var loraLayers: [Module] {
        encoder.layers
    }
}

// MARK: - Configuration

public struct NemotronLabsDiffusionConfiguration: Codable, Sendable {
    public var modelType: String = "nemotron_labs_diffusion"

    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDimensions: Int?

    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var maxPositionEmbeddings: Int?

    public var ropeTheta: Float = 1_000_000
    public var ropeParameters: [String: StringOrNumber]?

    public var attentionBias: Bool = false
    public var mlpBias: Bool = false

    // Diffusion-specific
    public var maskTokenId: Int = 100
    public var blockSize: Int = 32
    public var dlmParadigm: String = "bidirectional"

    // Optional config knobs not used at inference but kept so we can decode
    // the upstream config.json verbatim.
    public var slidingWindow: Int?
    public var attentionDropout: Float = 0.0

    public var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case maskTokenId = "mask_token_id"
        case blockSize = "block_size"
        case dlmParadigm = "dlm_paradigm"
        case slidingWindow = "sliding_window"
        case attentionDropout = "attention_dropout"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_labs_diffusion"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        // Mirror Python: rope_theta lives inside rope_parameters but may also be
        // present at the top level.
        if let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = theta
        } else if let theta = ropeParameters?["rope_theta"]?.asFloat() {
            ropeTheta = theta
        }
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        maskTokenId = try c.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 100
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 32
        dlmParadigm =
            try c.decodeIfPresent(String.self, forKey: .dlmParadigm) ?? "bidirectional"
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
    }
}

// MARK: - Generation Helpers

extension NemotronLabsDiffusionModel {

    /// Number of tokens to commit per denoising step within a single block,
    /// distributed evenly with leftovers spread to the first few steps. Mirrors
    /// `_get_num_transfer_tokens` in the Python reference.
    fileprivate static func numTransferTokensPerStep(
        maskCount: Int, steps: Int
    ) -> [Int] {
        precondition(steps > 0)
        let base = maskCount / steps
        let remainder = maskCount % steps
        var schedule = Array(repeating: base, count: steps)
        for i in 0 ..< remainder {
            schedule[i] += 1
        }
        return schedule
    }

    /// Greedy token IDs — argmax over the last axis, no probabilities.
    /// Cheapest option, used by the verify pass and the threshold==0 path
    /// where confidences aren't needed.
    fileprivate static func greedyTokens(logits: MLXArray) -> MLXArray {
        MLX.argMax(logits, axis: -1)
    }

    /// Greedy token IDs plus per-position confidence (probability of the
    /// argmax token).
    ///
    /// Equivalent to `(argmax(logits), softmax(logits).max(-1))` but avoids
    /// materializing the full softmax tensor — for vocab=131072 that's a
    /// ~131072× reduction in float ops per step. Computed via the
    /// log-sum-exp identity: `p_max = exp(max_logit - logSumExp(logits))`.
    fileprivate static func greedyTokensWithConfidence(
        logits: MLXArray
    ) -> (tokens: MLXArray, confidence: MLXArray) {
        let tokens = MLX.argMax(logits, axis: -1)
        let maxLogits = MLX.max(logits, axis: -1)
        let logZ = MLX.logSumExp(logits, axis: -1)
        let confidence = MLX.exp(maxLogits - logZ)
        return (tokens, confidence)
    }
}

// MARK: - AR Generation (model-level helper)

extension NemotronLabsDiffusionModel {

    /// Pure autoregressive generation (causal attention).
    ///
    /// Mirrors `ar_generate` in `modeling_nemotron_labs_diffusion.py`. Returns
    /// the prompt concatenated with newly generated tokens, and the number of
    /// forward passes performed.
    ///
    /// > Note: This method is provided for **symmetry** with
    /// > ``diffusionGenerate(promptIds:maxNewTokens:blockLength:threshold:eosTokenId:onStep:)``
    /// > and ``linearSpecGenerate(promptIds:maxNewTokens:blockLength:threshold:eosTokenId:onAccept:)``
    /// > so callers can pick a mode by method name. For real applications
    /// > prefer driving AR generation through the standard pipeline
    /// > (`ChatSession` / `ModelContainer.generate(...)`), which calls
    /// > ``callAsFunction(_:cache:)`` token-by-token via `TokenIterator` and
    /// > adds streaming, temperature / top-p / repetition-penalty sampling,
    /// > chunked prompt prefill, sliding-window and quantized KV caches, and
    /// > wired-memory ticket coordination — none of which this method does.
    public func arGenerate(
        promptIds: [Int],
        maxNewTokens: Int = 128,
        eosTokenId: Int? = nil
    ) -> (tokens: [Int], nfe: Int) {
        let cache = newCache(parameters: nil)

        let prompt = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        var logits = self(prompt, cache: cache)
        eval(cache, logits)
        var nfe = 1

        var generated: [Int] = []
        for _ in 0 ..< maxNewTokens {
            let lastLogits = logits[0..., -1, 0...]
            let next = Self.greedyTokens(logits: lastLogits)
            let nextInt = Int(next.item(Int32.self))
            generated.append(nextInt)
            if let eos = eosTokenId, nextInt == eos { break }

            let nextArr = MLXArray([Int32(nextInt)])[.newAxis, 0...]
            logits = self(nextArr, cache: cache)
            eval(cache, logits)
            nfe += 1
        }

        return (promptIds + generated, nfe)
    }
}

// MARK: - Diffusion Generation

extension NemotronLabsDiffusionModel {

    /// Block-wise diffusion (parallel) decoding.
    ///
    /// Mirrors `generate(...)` in `modeling_nemotron_labs_diffusion.py` with
    /// `causal_context = False`. The prompt is causally prefilled into a
    /// shared KV cache once. Each block of `blockLength` tokens is initialized
    /// to `mask_token_id` and refined in up to `blockLength` denoising steps:
    /// at each step the block is forwarded bidirectionally, attending against
    /// the cached prefix without re-running it. After each step the cache is
    /// trimmed by `blockLength` so the next step writes over the same slots.
    /// Once a block stabilizes, we run one causal forward over its finalized
    /// tokens to commit their K/V into the cache for subsequent blocks.
    ///
    /// > Design note — model-local rather than protocol-driven:
    /// >
    /// > The block-wise denoising loop here (causal prefill → bidirectional
    /// > block forward → cache-trim → confidence-driven commit → causal
    /// > commit) is the standard mask-diffusion LM recipe and would
    /// > generalize to other mask-diffusion models.
    /// > A `DiffusionLanguageModel` protocol could in
    /// > principle hoist this into a default extension, parameterized by a
    /// > model-supplied `bidirectionalLogits(_:cache:)` and a `maskTokenId`.
    /// >
    /// > That generalization is intentionally not done now. Nemotron is
    /// > the only mask-diffusion port in this repo, and several knobs
    /// > here are model-specific: the bidirectional-vs-causal mode switch on
    /// > the encoder is a Nemotron design, the threshold/top-N commit policy
    /// > and the "fall back to single most-confident token" behavior are
    /// > implementation choices, and `causal_context=False` is one of two
    /// > paradigms the upstream Python exposes. Premature abstraction over
    /// > a single example tends to encode whichever model arrived first.
    /// >
    /// > Re-visit when a second mask-diffusion model is added. With two
    /// > implementations to compare, the right protocol shape (which knobs
    /// > are common, which need policy-strategy hooks, what the bidirectional
    /// > forward should look like generically) becomes obvious from the
    /// > differences, and the abstraction pays for itself.
    public func diffusionGenerate(
        promptIds: [Int],
        maxNewTokens: Int,
        blockLength: Int = 32,
        threshold: Float? = 0.9,
        eosTokenId: Int? = nil,
        onStep: ((_ tokens: [Int], _ nfe: Int) -> Void)? = nil
    ) -> (tokens: [Int], nfe: Int) {
        precondition(
            maxNewTokens % blockLength == 0,
            "maxNewTokens must be a multiple of blockLength")

        let maskId = Int32(args.maskTokenId)
        let numBlocks = maxNewTokens / blockLength

        var xAccum: [Int32] = promptIds.map { Int32($0) }
        var nfe = 0

        // Causal prefill of the prompt once. After this, cache.offset == promptIds.count.
        let cache = newCache(parameters: nil)
        let promptArr = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        _ = self(promptArr, cache: cache)
        eval(cache)
        nfe += 1

        for _ in 0 ..< numBlocks {
            xAccum.append(contentsOf: Array(repeating: maskId, count: blockLength))

            let blockStart = xAccum.count - blockLength
            let stepsPerBlock = blockLength

            let initialMaskCount = (0 ..< blockLength).filter {
                xAccum[blockStart + $0] == maskId
            }.count
            let schedule = Self.numTransferTokensPerStep(
                maskCount: initialMaskCount, steps: stepsPerBlock)

            for step in 0 ..< stepsPerBlock {
                let stillMasked = (0 ..< blockLength).filter {
                    xAccum[blockStart + $0] == maskId
                }
                if stillMasked.isEmpty { break }

                // Forward only the block tokens, bidirectional, against the
                // already-cached prefix. The cache writes the block's K/V at
                // the end — we trim those out below before the next step.
                let blockSlice = xAccum[blockStart ..< (blockStart + blockLength)]
                let blockTensor = MLXArray(Array(blockSlice))[.newAxis, 0...]
                let blockLogits = self.bidirectionalLogits(blockTensor, cache: cache)
                eval(blockLogits)
                nfe += 1

                // For the threshold==0 path we only need a relative ranking
                // among masked positions, so raw max-logits suffice (they're
                // monotonically related to probabilities). For threshold>0 we
                // need actual probability values, computed cheaply via
                // logSumExp instead of materializing the full softmax.
                let useConfidence = (threshold ?? 0) > 0
                let sampledTokens: MLXArray
                let scores: MLXArray
                if useConfidence {
                    let (t, c) = Self.greedyTokensWithConfidence(logits: blockLogits)
                    sampledTokens = t
                    scores = c
                } else {
                    sampledTokens = Self.greedyTokens(logits: blockLogits)
                    scores = MLX.max(blockLogits, axis: -1)
                }
                eval(sampledTokens, scores)

                let scoreArr = scores[0].asArray(Float.self)
                let tokenArr = sampledTokens[0].asArray(Int32.self)
                var candidates: [(pos: Int, score: Float, token: Int32)] = []
                for p in stillMasked {
                    candidates.append((pos: p, score: scoreArr[p], token: tokenArr[p]))
                }

                var commit: [(pos: Int, token: Int32)] = []
                if let thr = threshold, thr > 0 {
                    let thresholded = candidates.filter { $0.score >= thr }
                    if thresholded.isEmpty {
                        if let best = candidates.max(by: { $0.score < $1.score }) {
                            commit.append((pos: best.pos, token: best.token))
                        }
                    } else {
                        for c in thresholded {
                            commit.append((pos: c.pos, token: c.token))
                        }
                    }
                } else {
                    let n = schedule[step]
                    let sorted = candidates.sorted { $0.score > $1.score }
                    for c in sorted.prefix(n) {
                        commit.append((pos: c.pos, token: c.token))
                    }
                }

                for c in commit {
                    xAccum[blockStart + c.pos] = c.token
                }

                // Trim the speculative block K/V from the cache before the
                // next denoising step (or before the post-block causal commit
                // below). This is the cache-reuse trick: we pay one prefix
                // prefill, then each denoising step is O(blockLength) tokens.
                for c in cache { _ = c.trim(blockLength) }

                onStep?(xAccum.map { Int($0) }, nfe)

                if let eos = eosTokenId {
                    let eosI32 = Int32(eos)
                    if let firstEos = (0 ..< blockLength).firstIndex(where: {
                        xAccum[blockStart + $0] == eosI32
                    }) {
                        let beforeEos = (0 ..< firstEos).contains {
                            xAccum[blockStart + $0] == maskId
                        }
                        if !beforeEos { break }
                    }
                }
            }

            // Block has stabilized. Commit its finalized K/V into the cache
            // with a causal forward so subsequent blocks see the right
            // attention pattern from this block.
            let finalizedSlice = xAccum[blockStart ..< (blockStart + blockLength)]
            let finalizedTensor = MLXArray(Array(finalizedSlice))[.newAxis, 0...]
            _ = self(finalizedTensor, cache: cache)
            eval(cache)
            nfe += 1

            if let eos = eosTokenId {
                let eosI32 = Int32(eos)
                let genStart = promptIds.count
                if let firstEos = xAccum[genStart...].firstIndex(of: eosI32) {
                    let cutoff = firstEos + 1
                    return (xAccum[..<cutoff].map { Int($0) }, nfe)
                }
            }
        }

        return (xAccum.map { Int($0) }, nfe)
    }
}

// MARK: - Linear Self-Speculation

extension NemotronLabsDiffusionModel {

    /// Linear speculative decoding: bidirectional drafter + AR verifier with a
    /// shared KV cache. Mirrors `linear_spec_generate(...)`.
    ///
    /// Requires `batch_size == 1`. The drafter runs with bidirectional
    /// attention (no KV-cache update) over a `blockLength`-token block; the
    /// verifier runs the same block causally and **does** update the cache.
    /// Accepted tokens are the longest matching prefix (plus one bonus AR
    /// token), and the cache is trimmed to length to discard rejected tokens.
    public func linearSpecGenerate(
        promptIds: [Int],
        maxNewTokens: Int = 128,
        blockLength: Int = 32,
        threshold: Float = 0.0,
        eosTokenId: Int? = nil,
        onAccept: ((_ tokens: [Int], _ nfe: Int) -> Void)? = nil
    ) -> (tokens: [Int], nfe: Int) {
        let maskId = Int32(args.maskTokenId)

        let cache = newCache(parameters: nil)

        // Causal prefill: LoRA off so the cached prefix matches AR mode.
        setLoRAEnabledFast(false)
        let prompt = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        var logits = self(prompt, cache: cache)
        eval(cache, logits)
        var nfe = 1

        let lastLogits = logits[0..., -1, 0...]
        let firstNext = Self.greedyTokens(logits: lastLogits)
        var nextToken = Int32(firstNext.item(Int32.self))

        var generated: [Int32] = [nextToken]
        onAccept?(promptIds + generated.map { Int($0) }, nfe)
        if let eos = eosTokenId, Int(nextToken) == eos {
            return (promptIds + [Int(nextToken)], nfe)
        }

        var totalGen = 1

        while totalGen < maxNewTokens {
            let cacheLenBefore = cache[0].offset

            // Build a draft block of size blockLength: position 0 holds the
            // last accepted next-token, the rest are masked.
            var block = [Int32](repeating: maskId, count: blockLength)
            block[0] = nextToken

            // Draft phase — LoRA on, bidirectional, against the causal
            // cache. The encoder sees only the block; the prefix is read from
            // the cache we built during prefill + previous accepted blocks.
            // Each draft forward writes K/V for the speculative block; we
            // trim it back before the next iteration so only verified tokens
            // persist.
            setLoRAEnabledFast(true)
            while true {
                let isMasked = block.contains(maskId)
                if !isMasked { break }

                let blockTensor = MLXArray(block)[.newAxis, 0...]
                let draftLogits = self.bidirectionalLogits(
                    blockTensor, cache: cache)
                eval(draftLogits)
                nfe += 1

                // The threshold==0 path commits everything in one shot and
                // breaks out — no confidence values needed. Only the
                // threshold>0 path pays for the (cheap) per-position prob.
                if threshold > 0 {
                    let (sampledTokens, confidences) =
                        Self.greedyTokensWithConfidence(logits: draftLogits)
                    eval(sampledTokens, confidences)
                    let tokenArr = sampledTokens[0].asArray(Int32.self)
                    let confArr = confidences[0].asArray(Float.self)

                    // Roll the cache back before the next iteration.
                    for c in cache { _ = c.trim(blockLength) }

                    var anyCommitted = false
                    var bestIdx = -1
                    var bestConf: Float = -.infinity
                    for i in 0 ..< blockLength where block[i] == maskId {
                        if confArr[i] >= threshold {
                            block[i] = tokenArr[i]
                            anyCommitted = true
                        }
                        if confArr[i] > bestConf {
                            bestConf = confArr[i]
                            bestIdx = i
                        }
                    }
                    if !anyCommitted, bestIdx >= 0 {
                        block[bestIdx] = tokenArr[bestIdx]
                    }
                } else {
                    let sampledTokens = Self.greedyTokens(logits: draftLogits)
                    eval(sampledTokens)
                    let tokenArr = sampledTokens[0].asArray(Int32.self)

                    for c in cache { _ = c.trim(blockLength) }

                    for i in 0 ..< blockLength where block[i] == maskId {
                        block[i] = tokenArr[i]
                    }
                    break
                }
            }

            // Verify phase — LoRA off, causal, cache update on.
            setLoRAEnabledFast(false)
            let blockTensor = MLXArray(block)[.newAxis, 0...]
            logits = self(blockTensor, cache: cache)
            eval(cache, logits)
            nfe += 1

            let verifyTokens = Self.greedyTokens(logits: logits)
            eval(verifyTokens)
            let arTokens = verifyTokens[0].asArray(Int32.self)

            // Acceptance: longest prefix where verify_tokens[i] == draft_block[i+1],
            // plus one bonus AR token.
            var accepted = 0
            for i in 0 ..< (blockLength - 1) {
                if arTokens[i] == block[i + 1] {
                    accepted += 1
                } else {
                    break
                }
            }
            accepted += 1  // bonus

            let acceptedToks = Array(arTokens.prefix(accepted))
            generated.append(contentsOf: acceptedToks)
            totalGen += accepted

            // Trim cache so that rejected speculative kv entries are discarded.
            let desired = cacheLenBefore + accepted
            for c in cache {
                let trimAmount = max(0, c.offset - desired)
                if trimAmount > 0 {
                    _ = c.trim(trimAmount)
                }
            }

            nextToken = arTokens[accepted - 1]

            if let eos = eosTokenId {
                let eosI32 = Int32(eos)
                if let pos = acceptedToks.firstIndex(of: eosI32) {
                    let truncated = Array(acceptedToks.prefix(pos + 1))
                    generated.removeLast(acceptedToks.count)
                    generated.append(contentsOf: truncated)
                    totalGen = totalGen - accepted + truncated.count
                    onAccept?(promptIds + generated.map { Int($0) }, nfe)
                    break
                }
            }

            onAccept?(promptIds + generated.map { Int($0) }, nfe)

            if totalGen >= maxNewTokens { break }
        }

        let outTokens = promptIds + generated.prefix(maxNewTokens).map { Int($0) }
        return (outTokens, nfe)
    }
}
