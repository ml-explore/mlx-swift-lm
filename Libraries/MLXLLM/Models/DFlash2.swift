// Copyright © 2026 Apple Inc.

// DFlash2 block drafter (https://inco.ai/blog/dflash2/), a port of the reference
// `dflash/model_mlx.py` (https://github.com/z-lab/dflash).
//
// The drafter embeds an `[anchor, MASK, ...]` block, runs it through a few
// bidirectional layers that attend to a window of the target's hidden states,
// keeps the top-K candidates per position, and walks one path through them with
// a pairwise selector. Embedding and LM head are the target's.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The `dflash_config` object of a DFlash2 checkpoint.
public struct DFlash2SpecConfiguration: Codable, Sendable {
    public var blockSize: Int = 16
    public var convGroupSize: Int = 16
    public var convKernelSize: Int = 2
    public var maskTokenId: Int = 0
    public var selectorRank: Int = 256
    public var selectorTopK: Int = 16
    public var targetLayerIds: [Int] = []
    public var inputEmbeddingScale: Float = 1.0
    public var outputMultiplier: Float = 1.0
    public var finalLogitSoftcapping: Float?
    /// The mlx-community layout's sliding window: `use_swa` with `swa_window_size`.
    public var useSlidingWindow: Bool?
    public var slidingWindowSize: Int?

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case maskTokenId = "mask_token_id"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useSlidingWindow = "use_swa"
        case slidingWindowSize = "swa_window_size"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        convGroupSize = try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 16
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 2
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 0
        selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 256
        selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 16
        targetLayerIds =
            try container.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        inputEmbeddingScale =
            try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1.0
        outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
        slidingWindowSize = try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
    }
}

/// Configuration of a DFlash2 draft checkpoint (`model_type: qwen3`,
/// `architectures: [DFlash2DraftModel]`).
///
/// Reads both published layouts: `layer_types` with `sliding_window` (z-lab,
/// inco.ai), and `dflash_config.use_swa` with a top-level `block_size`
/// (mlx-community). Every layer must be non-causal sliding-window attention.
public struct DFlash2Configuration: Decodable, Sendable {
    public var hiddenSize: Int = 5120
    public var hiddenLayers: Int = 5
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 17408
    public var vocabularySize: Int = 248320
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 262144
    public var slidingWindow: Int = 2048
    public var ropeTheta: Float = 10_000_000
    public var ropeScaling: [String: StringOrNumber]? = nil
    /// Depth of the target the drafter was distilled for; mlx-community omits it.
    public var numTargetLayers: Int?
    public var dflash: DFlash2SpecConfiguration = .init()

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case dflash = "dflash_config"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 5
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / attentionHeads)
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 17408
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248320
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        numTargetLayers = try container.decodeIfPresent(Int.self, forKey: .numTargetLayers)
        dflash =
            try container.decodeIfPresent(DFlash2SpecConfiguration.self, forKey: .dflash)
            ?? .init()
        let nested = try? container.nestedContainer(
            keyedBy: DFlash2SpecConfiguration.CodingKeys.self, forKey: .dflash)
        if try nested?.decodeIfPresent(Int.self, forKey: .blockSize) == nil,
            let blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
        {
            dflash.blockSize = blockSize
        }

        var ropeParameters =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters?["type"] == nil, let ropeType = ropeParameters?["rope_type"] {
            ropeParameters?["type"] = ropeType
        }
        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? 10_000_000
        ropeScaling = ropeParameters

        let window: Int?
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            let allSliding =
                layerTypes.count == hiddenLayers
                && layerTypes.allSatisfy { $0 == "sliding_attention" }
            window =
                allSliding ? try container.decodeIfPresent(Int.self, forKey: .slidingWindow) : nil
        } else {
            window = dflash.useSlidingWindow == true ? dflash.slidingWindowSize : nil
        }
        guard let window, window > 1 else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "DFlash2 needs sliding-window attention in every layer")
        }
        slidingWindow = window
        // The reference treats a missing `is_causal` as causal for sliding layers.
        guard try container.decodeIfPresent(Bool.self, forKey: .isCausal) == false else {
            throw DecodingError.dataCorruptedError(
                forKey: .isCausal, in: container,
                debugDescription: "DFlash2 needs a non-causal (is_causal: false) drafter")
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids is empty")
        }
    }

    /// Context rows a block can see: the window minus the block's own anchor.
    public var contextWindow: Int { slidingWindow - 1 }
}

// MARK: - Two-tap dynamic depthwise convolution

/// `Conv(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` on `groupSize`-channel groups:
/// a learned base kernel plus a per-group correction projected from the input.
/// `prepare` applies slot 0 and returns the dynamic kernel; `finish` applies slot 1.
final class DFlash2DynamicConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        precondition(hiddenSize % groupSize == 0, "conv groups must divide hidden size")
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// hidden `[B, L, H]`, dynamic `[B, L, K, groups]`, base `[K, H]`. The first
    /// position reads zeros before it.
    private static func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = h / groupSize
        let blocks = hidden.reshaped(b, l, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for tap in 0 ..< base.dim(0) {
            let values: MLXArray
            if tap == 0 {
                values = blocks
            } else {
                values = concatenated(
                    [MLXArray.zeros(like: blocks[0..., ..<tap]), blocks[0..., ..<(l - tap)]],
                    axis: 1)
            }
            let baseKernel = base[tap].asType(hidden.dtype).reshaped(1, 1, groups, groupSize)
            output = output + baseKernel * values
            output = output + dynamic[0..., 0..., tap, 0..., .newAxis] * values
        }
        return output.reshaped(hidden.shape)
    }

    func prepare(_ hidden: MLXArray) -> (convolved: MLXArray, kernel: MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, kernel: MLXArray) -> MLXArray {
        Self.convolve(hidden, dynamic: kernel, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - Attention

/// Block queries attend to the cached context K/V plus the block's own K/V.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration) {
        numHeads = config.attentionHeads
        numKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// Rotated block queries and keys plus values, heads first.
    func projectBlock(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        let keys = kNorm(kProj(x).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (
            applyRotaryPosition(rope, to: queries, offset: .batch(position)),
            applyRotaryPosition(rope, to: keys, offset: .batch(position)),
            values
        )
    }

    /// Rotated context keys plus values for the cache, heads first.
    func projectContext(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let (b, s) = (x.dim(0), x.dim(1))
        let keys = kNorm(kProj(x).reshaped(b, s, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, s, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (applyRotaryPosition(rope, to: keys, offset: .batch(position)), values)
    }

    func project(_ attention: MLXArray) -> MLXArray {
        let (b, l) = (attention.dim(0), attention.dim(2))
        return oProj(attention.transposed(0, 2, 1, 3).reshaped(b, l, numHeads * headDim))
    }
}

// MARK: - Decoder layer

/// Pre-norm layer with a dynamic conv pair around each sublayer, split at the
/// attention so the SDPA over the growing context runs between two traces.
final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2DynamicConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2DynamicConv

    init(_ config: DFlash2Configuration) {
        _selfAttn.wrappedValue = DFlash2Attention(config)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    /// Input norm, conv prepare and the block projections.
    func preAttention(
        x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (kernel: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        let (queries, keys, values) = selfAttn.projectBlock(h, rope: rope, position: position)
        return (kernel, queries, keys, values)
    }

    /// Output projection, conv finish, both residual adds and the MLP.
    func postAttention(x: MLXArray, attention: MLXArray, kernel: MLXArray) -> MLXArray {
        let h = x + attentionConv.finish(selfAttn.project(attention), kernel: kernel)
        let (m, mlpKernel) = mlpConv.prepare(postAttentionLayerNorm(h))
        return h + mlpConv.finish(mlp(m), kernel: mlpKernel)
    }
}

final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Candidate selector

/// Scores adjacent pairs of top-K candidates with a low-rank bilinear match
/// gated by the successor's hidden state,
/// `S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>`, then walks the best path.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    /// The greedy walk is about 60 small ops; one trace per block width.
    private let compiledGreedy = KeyedCompiledTraces<Int, DFlash2CandidateSelector>()

    init(_ config: DFlash2Configuration) {
        topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: normed block hidden states past the anchor, `[B, L, H]`.
    ///   - logits: matching draft logits, `[B, L, V]`.
    ///   - anchor: the verified token before the block, `[B]`.
    ///   - temperature: 0 walks greedily; above 0 samples each step from
    ///     `softmax(scores / temperature)` and returns those distributions.
    func select(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, temperature: Float
    ) -> DFlash2Proposal {
        guard temperature <= 0 else {
            return walk(hidden: hidden, logits: logits, anchor: anchor, temperature: temperature)
        }
        let trace = compiledGreedy.trace(logits.dim(1)) {
            CompiledTrace<DFlash2CandidateSelector> { selector, args in
                let proposal = selector.walk(
                    hidden: args[0], logits: args[1], anchor: args[2], temperature: 0)
                return [proposal.tokens, proposal.candidates]
            }
        }
        let outputs = trace(self, [hidden, logits, anchor])
        return DFlash2Proposal(tokens: outputs[0], candidates: outputs[1])
    }

    private func walk(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, temperature: Float
    ) -> DFlash2Proposal {
        let length = logits.dim(1)
        let kth = logits.dim(-1) - topK
        let candidates = argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        let successors = successorCodebook(candidates)
        let predecessors = predecessorCodebook(candidates)
        // Boundary 0 pairs the anchor with position 0; boundary t pairs the
        // candidates of t - 1 with those of t, gated by position t's hidden.
        let anchorEdges =
            ((predecessorCodebook(anchor) * projected[0..., 0])[0..., .newAxis, 0...]
            * successors[0..., 0, 0..., 0...]).sum(axis: -1)
        let gated =
            predecessors[0..., 0 ..< (length - 1), 0..., 0...]
            * projected[0..., 1..., .newAxis, 0...]
        let edges = matmul(gated, successors[0..., 1..., 0..., 0...].transposed(0, 1, 3, 2))

        var path: [MLXArray] = []
        var distributions: [MLXArray] = []
        var previous: MLXArray? = nil
        for position in 0 ..< length {
            var scores = unary[0..., position]
            if let previous {
                scores =
                    scores
                    + takeAlong(
                        edges[0..., position - 1, 0..., 0...],
                        previous[0..., .newAxis, .newAxis], axis: 1
                    )[0..., 0, 0...]
            } else {
                scores = scores + anchorEdges
            }
            let chosen: MLXArray
            if temperature > 0 {
                let distribution = softmax(scores.asType(.float32) / temperature, axis: -1)
                distributions.append(distribution)
                chosen = categorical(log(distribution))
            } else {
                chosen = argMax(scores, axis: -1)
            }
            previous = chosen
            path.append(
                takeAlong(candidates[0..., position], chosen[.newAxis, 0...], axis: -1)[0..., 0])
        }
        return DFlash2Proposal(
            tokens: stacked(path, axis: 1), candidates: candidates,
            probabilities: distributions.isEmpty ? nil : stacked(distributions, axis: 1))
    }
}

// MARK: - Draft model

/// The DFlash2 drafter.
///
/// A proposal runs as compiled segments split at each layer's attention. The
/// block position and every accept-dependent value enter as arrays, so one
/// graph serves any accept outcome.
public final class DFlash2DraftModel: Module, DFlash2DrafterModel {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    let rope: RoPELayer

    /// Test hook: run the segment bodies eagerly instead of through traces.
    var compiledTracesEnabled = true

    public init(_ config: DFlash2Configuration) {
        self.config = config
        let concatDim = config.dflash.targetLayerIds.count * config.hiddenSize
        _fc.wrappedValue = Linear(concatDim, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            DFlash2DecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: DFlash2DrafterModel

    public var blockSize: Int { config.dflash.blockSize }
    public var maskTokenId: Int { config.dflash.maskTokenId }
    public var targetLayerIds: [Int] { config.dflash.targetLayerIds }
    public var contextWindow: Int { config.contextWindow }

    public var targetGeometry: DFlash2Geometry {
        DFlash2Geometry(
            layerCount: config.numTargetLayers, hiddenSize: config.hiddenSize,
            vocabularySize: config.vocabularySize)
    }

    public func makeState() -> DFlash2DrafterState {
        DFlash2DrafterState(
            contextCaches: layers.map { _ in DFlash2ContextCache(window: contextWindow) })
    }

    public func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        let block = request.block
        precondition(block.dim(0) == 1, "DFlash2 drafts one stream at a time")
        precondition(
            state.contextCaches.count == layers.count, "one context cache per drafter layer")
        let rows = request.targetHidden.dim(1)
        precondition(rows <= contextWindow, "targetHidden exceeds the drafter's context window")
        let width = block.dim(1)
        let contextPosition = MLXArray([Int32(request.contextPosition)])
        let blockPosition = contextPosition + request.validRows.asType(.int32).reshaped([1])

        // Context K/V for the new rows, appended as placeholders.
        let contextKV = projectContext(request.targetHidden, position: contextPosition)
        let positions = (0 ..< rows).map { Int32(request.contextPosition + $0) }
        var layerKV: [(MLXArray, MLXArray)] = []
        for (i, cache) in state.contextCaches.enumerated() {
            layerKV.append(
                cache.append(
                    keys: contextKV[2 * i], values: contextKV[2 * i + 1], positions: positions))
        }
        // The caches move in lockstep, so one mask serves every layer.
        let mask = Self.visibilityMask(
            cache: state.contextCaches[0], appended: rows, validRows: request.validRows,
            blockPosition: blockPosition, width: width, window: config.slidingWindow)

        // The target's embedding and head run outside the traces.
        var x = target.dflash2Embedding(block) * config.dflash.inputEmbeddingScale
        var attention: MLXArray? = nil
        var kernel: MLXArray? = nil
        for index in 0 ... layers.count {
            let args = attention.map { [x, $0, kernel!, blockPosition] } ?? [x, blockPosition]
            let outputs =
                compiledTracesEnabled
                ? segmentTrace(index: index, width: width)(self, args)
                : segmentBody(at: index, args)
            x = outputs[0]
            guard index < layers.count else { break }
            kernel = outputs[1]
            let (contextKeys, contextValues) = layerKV[index]
            attention = MLXFast.scaledDotProductAttention(
                queries: outputs[2],
                keys: concatenated([contextKeys, outputs[3]], axis: 2),
                values: concatenated([contextValues, outputs[4]], axis: 2),
                scale: layers[index].selfAttn.scale, mask: mask)
        }

        let hidden = x[0..., 1..., 0...]
        var logits = target.dflash2Head?(hidden) ?? target.dflash2Embedding.asLinear(hidden)
        logits = logits * config.dflash.outputMultiplier
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        if let suppressor = request.suppressedTokens {
            logits = suppressor.process(logits: logits[0]).expandedDimensions(axis: 0)
        }
        return candidateSelector.select(
            hidden: hidden, logits: logits, anchor: block[0..., 0],
            temperature: request.temperature)
    }

    /// Bool mask `[width, cache.count + width]`. Committed context rows within the
    /// window are visible, placeholders never, and the block sees itself in full.
    static func visibilityMask(
        cache: DFlash2ContextCache, appended: Int, validRows: MLXArray,
        blockPosition: MLXArray, width: Int, window: Int
    ) -> MLXArray {
        let older = cache.count - appended
        let keyPositions = MLXArray(cache.rowPositions)
        let olderValid = MLXArray(Array(cache.rowValid[0 ..< older]))
        let appendedValid = MLXArray(Int32(0) ..< Int32(appended)) .< validRows.asType(.int32)
        let valid = concatenated([olderValid, appendedValid])
        let queryPositions = blockPosition.asType(.int32) + MLXArray(Int32(0) ..< Int32(width))
        let distance =
            queryPositions.expandedDimensions(axis: 1) - keyPositions.expandedDimensions(axis: 0)
        let contextVisible = valid.expandedDimensions(axis: 0) & (distance .< Int32(window))
        return concatenated(
            [contextVisible, MLXArray.ones([width, width], dtype: .bool)], axis: 1)
    }

    // MARK: Compiled segments

    private struct SegmentKey: Hashable {
        var index: Int
        var width: Int
    }

    private let segmentTraces = KeyedCompiledTraces<SegmentKey, DFlash2DraftModel>()
    private let contextTraces = KeyedCompiledTraces<Int, DFlash2DraftModel>()

    /// The prompt window projects eagerly; a trace per prompt length would only churn.
    private static let maxTracedContextRows = 16

    /// Live traces, so tests can check that the compiled route ran.
    var compiledTraceCount: Int { segmentTraces.count + contextTraces.count }

    /// Segment `i` closes layer `i - 1` and opens layer `i`; the last segment
    /// applies the final norm. In: `[x, position]` for segment 0, else
    /// `[x, attention, kernel, position]`. Out: `[x, kernel, queries, keys, values]`,
    /// or `[normed]` for the last.
    private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
        var x = args[0]
        if index > 0 {
            x = layers[index - 1].postAttention(x: x, attention: args[1], kernel: args[2])
        }
        guard index < layers.count else { return [norm(x)] }
        let position = args[index == 0 ? 1 : 3]
        let (kernel, queries, keys, values) = layers[index].preAttention(
            x: x, rope: rope, position: position)
        return [x, kernel, queries, keys, values]
    }

    private func segmentTrace(index: Int, width: Int) -> CompiledTrace<DFlash2DraftModel> {
        segmentTraces.trace(SegmentKey(index: index, width: width)) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.segmentBody(at: index, args) }
        }
    }

    /// `[targetHidden, position]` to `[keys, values]` per layer.
    private func contextBody(_ args: [MLXArray]) -> [MLXArray] {
        let context = hiddenNorm(fc(args[0]))
        return layers.flatMap { layer in
            let (keys, values) = layer.selfAttn.projectContext(
                context, rope: rope, position: args[1])
            return [keys, values]
        }
    }

    private func projectContext(_ targetHidden: MLXArray, position: MLXArray) -> [MLXArray] {
        let rows = targetHidden.dim(1)
        guard compiledTracesEnabled, rows <= Self.maxTracedContextRows else {
            return contextBody([targetHidden, position])
        }
        let trace = contextTraces.trace(rows) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.contextBody(args) }
        }
        return trace(self, [targetHidden, position])
    }

    /// The checkpoint stores the selector codebooks as bare tensors; `Embedding`
    /// expects a `.weight` suffix.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: bare) {
                weights["\(bare).weight"] = value
            }
        }
        return weights
    }
}
