// Copyright © 2026 Apple Inc.
//
// Port of https://github.com/ml-explore/mlx-lm/blob/c4c9c762cc797fd979fc98645f5db5112753bbe9/mlx_lm/models/qwen4_exp.py
//
// Qwen3.8-Flash-Next: qwen3_next with QSA sparse attention, hyper-connections
// and an n-gram embedding (PLE).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen4ExpConfiguration: Codable, Sendable {
    var modelType: String
    var text: Qwen4ExpTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        self.text = try container.decode(Qwen4ExpTextConfiguration.self, forKey: .text)
    }
}

struct Qwen4ExpTextConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeParameters: [String: StringOrNumber]
    var eosTokenIds: StringOrNumber
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var linearNumKeyHeads: Int
    var linearNumValueHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var hcCount: Int
    var hcLowrank: Int
    var indexerHeads: Int
    var indexerKVHeads: Int
    var indexerHeadDim: Int
    var indexerBudget: Int
    var indexerCompressRatio: Int
    var ngramSize: Int
    var headsPerNgram: Int
    var ngramVocabSizeBase: Int
    var pleEmbedDim: Int
    var pleLayerIds: [Int]
    // config.json has no `seed`, which the n-gram hash uses.
    var pleConvKernelSize: Int
    var seed: Int
    var layerTypes: [String]
    var fullAttentionInterval: Int
    var outputGateType: String
    var hiddenAct: String
    var tieWordEmbeddings: Bool

    // Derived from `rope_parameters` and `eos_token_id`.
    var ropeTheta: Float = 0
    var partialRotaryFactor: Float = 1
    var eosTokenId: Int = 0

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case eosTokenIds = "eos_token_id"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case pleEmbedDim = "ple_embed_dim"
        case pleLayerIds = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case seed
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case outputGateType = "output_gate_type"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeParameters = try c.decode([String: StringOrNumber].self, forKey: .ropeParameters)
        eosTokenIds = try c.decode(StringOrNumber.self, forKey: .eosTokenIds)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        hcCount = try c.decode(Int.self, forKey: .hcCount)
        hcLowrank = try c.decode(Int.self, forKey: .hcLowrank)
        indexerHeads = try c.decode(Int.self, forKey: .indexerHeads)
        indexerKVHeads = try c.decode(Int.self, forKey: .indexerKVHeads)
        indexerHeadDim = try c.decode(Int.self, forKey: .indexerHeadDim)
        indexerBudget = try c.decode(Int.self, forKey: .indexerBudget)
        indexerCompressRatio = try c.decode(Int.self, forKey: .indexerCompressRatio)
        ngramSize = try c.decode(Int.self, forKey: .ngramSize)
        headsPerNgram = try c.decode(Int.self, forKey: .headsPerNgram)
        ngramVocabSizeBase = try c.decode(Int.self, forKey: .ngramVocabSizeBase)
        pleEmbedDim = try c.decode(Int.self, forKey: .pleEmbedDim)
        pleLayerIds = try c.decode([Int].self, forKey: .pleLayerIds)
        pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
        fullAttentionInterval =
            try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? hiddenAct
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        if layerTypes.isEmpty {
            let interval = fullAttentionInterval
            layerTypes = (0 ..< hiddenLayers).map {
                ($0 + 1) % interval == 0 ? "full_attention" : "linear_attention"
            }
        }

        guard let theta = ropeParameters["rope_theta"]?.asFloat() else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeParameters, in: c,
                debugDescription: "rope_parameters needs a numeric rope_theta")
        }
        ropeTheta = theta
        partialRotaryFactor = ropeParameters["partial_rotary_factor"]?.asFloat() ?? 1

        switch eosTokenIds {
        case .int(let id):
            eosTokenId = id
        case .ints(let ids) where !ids.isEmpty:
            eosTokenId = ids[0]
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .eosTokenIds, in: c,
                debugDescription: "eos_token_id must be an integer or a nonempty integer list")
        }
    }
}

extension Qwen4ExpConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let args = text
        func invalid(_ message: String) -> ModelFactoryError {
            .invalidConfiguration("qwen4_exp: \(message)")
        }
        guard args.layerTypes.count == args.hiddenLayers else {
            throw invalid(
                "layer_types has \(args.layerTypes.count) entries for \(args.hiddenLayers) layers")
        }
        let kinds = Set(args.layerTypes)
        guard kinds == ["linear_attention", "full_attention"] else {
            throw invalid(
                "layer_types must mix linear_attention and full_attention layers, got \(kinds.sorted())"
            )
        }
        // The PLE state lives in the linear-attention layer cache.
        for id in args.pleLayerIds {
            guard (1 ... args.hiddenLayers).contains(id),
                args.layerTypes[id - 1] == "linear_attention"
            else {
                throw invalid("ple_layer_ids entry \(id) is not a linear_attention layer (1-based)")
            }
        }
        guard args.indexerKVHeads == 1 else {
            throw invalid("indexer_kv_heads must be 1, got \(args.indexerKVHeads)")
        }
        guard args.indexerCompressRatio > 0, args.indexerBudget >= args.indexerCompressRatio else {
            throw invalid("indexer_budget must hold at least one indexer_compress_ratio block")
        }
        guard args.ngramSize >= 2, args.headsPerNgram > 0 else {
            throw invalid("ngram_size must be at least 2 and heads_per_ngram positive")
        }
        let ngramHeads = (args.ngramSize - 1) * args.headsPerNgram
        guard args.pleEmbedDim % ngramHeads == 0 else {
            throw invalid("ple_embed_dim must divide into \(ngramHeads) n-gram heads")
        }
    }
}

// MARK: - Norms

/// RMSNorm, optionally per `groupSize` slice; `weight` holds the folded `1 + w`.
private final class Qwen4ExpRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        self.groupSize = groupSize
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let groupSize else {
            return MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }
        let shape = x.shape
        let grouped = x.reshaped(Array(shape.dropLast()) + [-1, groupSize])
        return MLXFast.rmsNorm(grouped, weight: .mlxNone, eps: eps).reshaped(shape) * weight
    }
}

/// Gated RMSNorm; the checkpoint stores `weight` as is.
private final class Qwen4ExpRMSNormGated: Module {
    let weight: MLXArray
    let eps: Float
    let sigmoidGate: Bool

    init(dimensions: Int, activation: String, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        self.sigmoidGate = activation == "sigmoid"
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let out = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let gate = gate.asType(.float32)
        return (out * (sigmoidGate ? sigmoid(gate) : silu(gate))).asType(x.dtype)
    }
}

// MARK: - QSA sparse attention

/// The raw indexer keys of one attention layer, and the pooled keys of their complete blocks.
///
/// Cache policy skips it: KV quantization and capacity would break the block indices,
/// and the Python model never quantizes these layers' caches. The pooled blocks are not
/// saved: they are rebuilt from the raw keys.
private final class Qwen4ExpIndexerKeyCache: BaseKVCache, AuxiliaryKVCache {
    override class var serializedClassName: String? { "MLXLLM.Qwen4ExpIndexerKeyCache" }

    private static let registration: Void = {
        try! registerKVCacheType(Qwen4ExpIndexerKeyCache.self)
    }()

    static func register() {
        _ = registration
    }

    override class func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        guard metaState == [""], state.count <= 1 else {
            throw CorruptState()
        }
        let cache = Qwen4ExpIndexerKeyCache()
        if !state.isEmpty {
            cache.state = state
        }
        return cache
    }

    private struct CorruptState: LocalizedError {
        var errorDescription: String? {
            "Corrupt prompt cache: Qwen4ExpIndexerKeyCache holds at most one array."
        }
    }

    private var keys: MLXArray?
    private var pooled: MLXArray?
    private var pooledCount = 0
    private var pooledRatio = 1
    private let step = 256

    override init() {
        Self.register()
        super.init()
    }

    /// `buffer` with room for `new` after its first `used` rows.
    private func reserve(_ buffer: MLXArray?, used: Int, for new: MLXArray) -> MLXArray {
        if let buffer, used + new.dim(1) <= buffer.dim(1) {
            return buffer
        }
        let steps = (step + new.dim(1) - 1) / step
        let grown = MLXArray.zeros([new.dim(0), steps * step, new.dim(2)], dtype: new.dtype)
        guard var current = buffer else {
            return grown
        }
        if used % step != 0 {
            current = current[0..., ..<used, 0...]
        }
        return concatenated([current, grown], axis: 1)
    }

    /// Append `newKeys` `[B, S, D]` and return every key so far.
    func append(_ newKeys: MLXArray) -> MLXArray {
        let previous = offset
        keys = reserve(keys, used: previous, for: newKeys)
        offset += newKeys.dim(1)
        keys![0..., previous ..< offset, 0...] = newKeys
        return keys![0..., ..<offset, 0...]
    }

    /// The pooled keys `[B, offset / ratio, D]` of every complete block.
    ///
    /// `pool(keys, first)` pools the raw keys of the blocks from `first` on. It runs only
    /// on blocks completed since the last call.
    func pooledBlocks(ratio: Int, pool: (MLXArray, Int) -> MLXArray) -> MLXArray {
        if ratio != pooledRatio {
            pooledRatio = ratio
            pooledCount = 0
        }
        let blockCount = offset / ratio
        if blockCount > pooledCount, let keys {
            let first = pooledCount
            let new = pool(keys[0..., (first * ratio) ..< (blockCount * ratio), 0...], first)
            pooled = reserve(pooled, used: first, for: new)
            pooled![0..., first ..< blockCount, 0...] = new
            pooledCount = blockCount
        }
        guard let pooled else {
            preconditionFailure("Qwen4ExpIndexerKeyCache has no complete block")
        }
        return pooled[0..., ..<blockCount, 0...]
    }

    override func innerState() -> [MLXArray] {
        [keys, pooled].compactMap { $0 }
    }

    override var state: [MLXArray] {
        get {
            guard let keys else { return [] }
            return offset == keys.dim(1) ? [keys] : [keys[0..., ..<offset, 0...]]
        }
        set {
            precondition(newValue.count <= 1, "Qwen4ExpIndexerKeyCache state holds one array")
            keys = newValue.first
            offset = keys?.dim(1) ?? 0
            pooled = nil
            pooledCount = 0
        }
    }

    override var isTrimmable: Bool { true }

    // A block that loses a token is pooled again once it is complete.
    @discardableResult
    override func trim(_ n: Int) -> Int {
        let trimmed = Swift.min(Swift.max(n, 0), offset)
        offset -= trimmed
        pooledCount = Swift.min(pooledCount, offset / pooledRatio)
        return trimmed
    }

    override func copy() -> any KVCache {
        let new = Qwen4ExpIndexerKeyCache()
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        if let pooled, pooledCount > 0 {
            new.pooled = pooled[0..., ..<pooledCount, 0...]
            new.pooledCount = pooledCount
            new.pooledRatio = pooledRatio
        }
        return new
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("Qwen4ExpIndexerKeyCache stores indexer keys; use append(_:)")
    }
}

/// Select, per query, a budget of compressed key blocks.
private final class Qwen4ExpQSAIndexer: Module {
    let heads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopK: Int

    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: Qwen4ExpRMSNorm
    let rope: RoPE
    // Block n starts at position n * compress_ratio.
    let blockRope: RoPE

    init(_ args: Qwen4ExpTextConfiguration) {
        heads = args.indexerHeads
        headDim = args.indexerHeadDim
        tokenBudget = args.indexerBudget
        compressRatio = args.indexerCompressRatio
        blockTopK = tokenBudget / compressRatio
        _indexQKProj.wrappedValue = Linear(
            args.hiddenSize, (heads + args.indexerKVHeads) * headDim, bias: false)
        _qLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        let rotaryDim = Int(Float(args.headDim) * args.partialRotaryFactor)
        rope = RoPE(dimensions: rotaryDim, base: args.ropeTheta)
        blockRope = RoPE(dimensions: rotaryDim, base: args.ropeTheta, scale: Float(compressRatio))
        super.init()
    }

    /// Pool raw keys `[B, n * ratio, D]` into `[B, n, D]` blocks, the first being block `first`.
    private func pool(_ keys: MLXArray, first: Int) -> MLXArray {
        let blocks = keys.reshaped(keys.dim(0), -1, compressRatio, headDim)
        let pooled = blocks.asType(.float32).mean(axis: 2).asType(keys.dtype)
        return blockRope(kLayerNorm(pooled), offset: first)
    }

    /// A boolean mask `[B, 1, S, keys]`, or `nil` when the causal mask is enough.
    func callAsFunction(_ x: MLXArray, cache: Qwen4ExpIndexerKeyCache?) -> MLXArray? {
        let B = x.dim(0)
        let S = x.dim(1)
        let r = compressRatio
        let qk = indexQKProj(x)
        let split = heads * headDim
        let q = qk[.ellipsis, ..<split].reshaped(B, S, heads, headDim)
        // One key head (`indexer_kv_heads` is 1).
        var rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)

        let offset = cache?.offset ?? 0
        if let cache {
            rawK = cache.append(rawK)
        }
        let kvLength = rawK.dim(1)

        // Every visible token fits in the budget: the causal mask is enough.
        if kvLength <= tokenBudget {
            return nil
        }

        // Left padding is not supported, so each row's blocks start at position 0.
        let blockCount = kvLength / r
        let pooled =
            cache?.pooledBlocks(ratio: r, pool: pool)
            ?? pool(rawK[0..., ..<(blockCount * r), 0...], first: 0)

        let queries = rope(qLayerNorm(q).transposed(0, 2, 1, 3), offset: offset)

        // Per block: the sum over heads of relu(q.k).
        var scores = einsum(
            "bhsd,bnd->bsnh", queries.asType(.float32), pooled.asType(.float32))
        scores = maximum(scores, 0).sum(axis: -1) / Float(headDim).squareRoot()

        // Only blocks fully in the query's past.
        let queryPositions = MLXArray(Int32(offset)) + MLXArray(Int32(0) ..< Int32(S))
        let completeBlocks = floorDivide(maximum(queryPositions + 1, 0), r)
        let visible = broadcast(
            MLXArray(Int32(0) ..< Int32(blockCount)) .< completeBlocks[0..., .newAxis],
            to: [B, S, blockCount])
        scores = MLX.where(visible, scores, MLXArray(-Float.infinity))

        let k = Swift.min(blockTopK, blockCount)
        let top = stopGradient(argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        let picked = takeAlong(visible, top, axis: -1)

        // The tokens of the picked blocks; the others go to a dropped column.
        var tokens =
            top[.ellipsis, .newAxis].asType(.int32) * Int32(r) + MLXArray(Int32(0) ..< Int32(r))
        tokens = MLX.where(picked[.ellipsis, .newAxis], tokens, MLXArray(Int32(kvLength)))
        tokens = tokens.reshaped(B, S, -1)
        var keep = MLXArray.zeros([B, S, kvLength + 1], dtype: .bool)
        keep = putAlong(
            keep, tokens, values: broadcast(MLXArray(true), to: tokens.shape), axis: -1)
        keep = keep[.ellipsis, ..<kvLength]

        // The reference also keeps each query's own partial block (its `tail`).
        let ownStart = completeBlocks * Int32(r)
        let columns = MLXArray(Int32(0) ..< Int32(kvLength))
        let queryColumns = MLXArray(Int32(kvLength - S) ..< Int32(kvLength))[0..., .newAxis]
        let own = (columns .>= ownStart[0..., .newAxis]) .&& (columns .<= queryColumns)
        return expandedDimensions(keep .|| own, axis: 1)
    }
}

private final class Qwen4ExpAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    // q_proj also carries the output gate.
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer
    let rope: RoPE

    init(_ args: Qwen4ExpTextConfiguration) {
        heads = args.attentionHeads
        kvHeads = args.kvHeads
        headDim = args.headDim
        scale = pow(Float(headDim), -0.5)
        let d = args.hiddenSize
        _qProj.wrappedValue = Linear(d, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, d, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)
        rope = RoPE(
            dimensions: Int(Float(args.headDim) * args.partialRotaryFactor), base: args.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        // `cache` holds the attention keys and values, then the indexer keys.
        let kvCache = cache?[0]
        let indexerCache = cache.map { list -> Qwen4ExpIndexerKeyCache in
            guard let keys = list[1] as? Qwen4ExpIndexerKeyCache else {
                preconditionFailure(
                    "qwen4_exp attention cache must come from newCache(parameters:)")
            }
            return keys
        }

        let qSplit = qProj(x).reshaped(B, S, heads, -1).split(parts: 2, axis: -1)
        let gate = qSplit[1].reshaped(B, S, -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, S, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, S, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = kvCache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // The sparse mask is causal and has no padding: it replaces `mask`.
        let sparse = indexer(x, cache: indexerCache)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: kvCache, scale: scale,
            mask: sparse.map { .array($0) } ?? mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, S, -1) * sigmoid(gate))
    }
}

// MARK: - Gated DeltaNet

/// The last `n` steps of `x`; a right-padded row ends at its length.
private func keepLast(_ x: MLXArray, n: Int, lengths: MLXArray?) -> MLXArray {
    guard let lengths else {
        // A copy: a slice would keep all of `x` alive in the cache.
        return contiguous(x[0..., (x.dim(1) - n)...])
    }
    var positions =
        clip(lengths, min: 0, max: x.dim(1) - n)[0..., .newAxis] + MLXArray(Int32(0) ..< Int32(n))
    if x.ndim == 3 {
        positions = positions[.ellipsis, .newAxis]
    }
    return takeAlong(x, positions, axis: 1)
}

private final class Qwen4ExpGatedDeltaNet: Module {
    let numValueHeads: Int
    let numKeyHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen4ExpTextConfiguration) {
        numValueHeads = args.linearNumValueHeads
        numKeyHeads = args.linearNumKeyHeads
        keyHeadDim = args.linearKeyHeadDim
        valueHeadDim = args.linearValueHeadDim
        keyDim = keyHeadDim * numKeyHeads
        valueDim = valueHeadDim * numValueHeads
        convKernelSize = args.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        let d = args.hiddenSize

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            padding: 0, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(d, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(d, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(d, numValueHeads, bias: false)
        _inProjA.wrappedValue = Linear(d, numValueHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numValueHeads])
        _aLog.wrappedValue = MLXArray.zeros([numValueHeads])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: valueHeadDim, activation: args.outputGateType, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, d, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        var mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numValueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

        let convState =
            cache?[0] ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype)
        if let mask {
            mixedQKV = MLX.where(
                mask[.ellipsis, .newAxis], mixedQKV, MLXArray(0, dtype: mixedQKV.dtype))
        }
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        if let cache {
            cache[0] = keepLast(convInput, n: convKernelSize - 1, lengths: cache.currentLengths)
        }
        let convOut = silu(conv1d(convInput))

        let qkv = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = qkv[0].reshaped(B, S, numKeyHeads, keyHeadDim)
        var k = qkv[1].reshaped(B, S, numKeyHeads, keyHeadDim)
        let v = qkv[2].reshaped(B, S, numValueHeads, valueHeadDim)

        // normalize_qk: rmsNorm adds eps to mean(x^2), the reference l2norm to sum(x^2).
        let invScale = pow(Float(keyHeadDim), -0.5)
        let qkEps = 1e-6 * invScale * invScale
        q =
            MLXArray(invScale * invScale).asType(q.dtype)
            * MLXFast.rmsNorm(q, weight: .mlxNone, eps: qkEps)
        k = MLXArray(invScale).asType(k.dtype) * MLXFast.rmsNorm(k, weight: .mlxNone, eps: qkEps)

        let (out, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: cache?[1], mask: mask, useKernel: !training)
        if let cache {
            cache[1] = state
            cache.advance(S)
        }

        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }
}

// MARK: - MoE

private final class Qwen4ExpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class Qwen4ExpSparseMoeBlock: Module, UnaryLayer {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen4ExpTextConfiguration) {
        topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExpert.wrappedValue = Qwen4ExpMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x.asType(.float32))
        let indices = stopGradient(
            argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        // The softmax of the top-k logits: the reference's renormalized top-k.
        let weights = softmax(takeAlong(logits, indices, axis: -1), axis: -1)
        let routed = (switchMLP(x, indices) * weights[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(x.dtype)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - Hyper-connections

private final class Qwen4ExpGatedResidual: Module {
    let hc: Int
    let d: Int
    let lowrank: Int
    let useCombine: Bool

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    // The last hc rows are the checkpoint's block inject: one matmul for both.
    @ModuleInfo(key: "input_mix_weight_down") var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixWeightUp: Linear

    init(_ args: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        hc = args.hcCount
        d = args.hiddenSize
        lowrank = args.hcLowrank
        self.useCombine = useCombine
        let hcDim = hc * d
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _inputMixWeightDown.wrappedValue = Linear(
            hcDim, lowrank + (useCombine ? hc : 0), bias: false)
        _inputMixWeightUp.wrappedValue = Linear(lowrank, hcDim, bias: false)
        super.init()
    }

    /// The mixed streams `[..., d]` and, with `useCombine`, the per-stream inject `[..., hc]`.
    func callAsFunction(_ hyper: MLXArray) -> (mixed: MLXArray, inject: MLXArray?) {
        let normed = hcNorm(hyper)
        let down = inputMixWeightDown(normed)
        var w = silu(down[.ellipsis, ..<lowrank] / Float(hc))
        w = sigmoid(inputMixWeightUp(w))
        let prefix = Array(w.shape.dropLast())
        w = w.reshaped(prefix + [hc, d])
        let mixed = (w * normed.reshaped(prefix + [hc, d])).mean(axis: -2)
        guard useCombine else { return (mixed, nil) }
        return (mixed, 2 * sigmoid(down[.ellipsis, lowrank...] / Float(hc)))
    }
}

/// Add `x` to every hyper-connection stream, scaled per stream.
private func injectResidual(_ hyper: MLXArray, _ x: MLXArray, _ inject: MLXArray) -> MLXArray {
    hyper
        + (x[.ellipsis, .newAxis, 0...] * inject[.ellipsis, .newAxis])
        .reshaped(Array(x.shape.dropLast()) + [-1])
}

// MARK: - N-gram embedding (PLE)

/// splitmix64 and the n-gram hash constants, with Python's arbitrary-precision math as
/// wrapping `UInt64` arithmetic.
private enum Qwen4ExpNGramHash {
    static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let m1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let m2: UInt64 = 0x94D0_49BB_1331_11EB
    static let layerSeedPrime = 10007

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ gamma
        v = (v ^ (v >> 30)) &* m1
        v = (v ^ (v >> 27)) &* m2
        return v ^ (v >> 31)
    }

    /// Odd hash multipliers, small enough that `token * multiplier` fits in int64.
    static func multipliers(vocabularySize: Int, ngramSize: Int, seed: Int) -> [Int64] {
        let half = UInt64(Swift.max(1, Int.max / Swift.max(vocabularySize, 1) / 2))
        let seed = UInt64(bitPattern: Int64(seed))
        return (0 ..< ngramSize).map { i in
            Int64(2 * (splitmix64(seed &+ gamma &* UInt64(i + 1)) % half) + 1)
        }
    }

    static func isPrime(_ v: Int) -> Bool {
        if v < 2 { return false }
        if v % 2 == 0 { return v == 2 }
        var d = 3
        while d * d <= v {
            if v % d == 0 { return false }
            d += 2
        }
        return true
    }

    /// The primes from `start` on, skipping the first `skip` of them.
    static func primes(from start: Int, skipping skip: Int, count: Int) -> [Int] {
        var result: [Int] = []
        var seen = 0
        var candidate = start
        while result.count < count {
            if isPrime(candidate) {
                if seen >= skip { result.append(candidate) }
                seen += 1
            }
            candidate += 1
        }
        return result
    }
}

/// N-gram hash tables, one per head.
private final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNgram: Int
    let ngramHeads: Int
    let eosTokenId: Int
    let headVocabSizes: [Int]

    // One table per head keeps each checkpoint file small (~6 GB in bf16).
    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: [Embedding]

    // Not parameters: rebuilt from the config, checked in `prepareCheckpoint`.
    let _mults: MLXArray
    let _sizes: MLXArray
    let _offsets: MLXArray

    init(_ args: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        ngramSize = args.ngramSize
        headsPerNgram = args.headsPerNgram
        ngramHeads = (ngramSize - 1) * headsPerNgram
        eosTokenId = args.eosTokenId
        let headDim = args.pleEmbedDim / ngramHeads

        // Head g of all layers takes the (g+1)-th prime from ngram_vocab_size_base on.
        let sizes = Qwen4ExpNGramHash.primes(
            from: args.ngramVocabSizeBase, skipping: pleLayerIndex * ngramHeads,
            count: ngramHeads)
        headVocabSizes = sizes
        _ngramEmbedding.wrappedValue = sizes.map {
            Embedding(embeddingCount: $0, dimensions: headDim)
        }

        let multipliers = Qwen4ExpNGramHash.multipliers(
            vocabularySize: args.vocabularySize, ngramSize: ngramSize,
            seed: args.seed + Qwen4ExpNGramHash.layerSeedPrime * pleLayerIndex)
        _mults = MLXArray(multipliers)
        _sizes = MLXArray(sizes.map(Int64.init))
        _offsets = MLXArray(sizes.indices.map { Int64(sizes[..<$0].reduce(0, +)) })
        super.init()
    }

    /// x[t], x[t-1], ... for the new tokens; `history` starts with the n-1 tokens
    /// before them. Older tokens past an EOS read as EOS.
    private func ngramTokens(_ history: MLXArray) -> [MLXArray] {
        let n = ngramSize
        let new = history.dim(1) - (n - 1)
        // HF's `_shift_right_ignore_eos`, at the new positions only.
        var tokens = [history[0..., (n - 1)...], history[0..., (n - 2) ..< (n - 2 + new)]]
        for s in 2 ..< n {
            let older = history[0..., (n - 1 - s) ..< (n - 1 - s + new)]
            tokens.append(
                MLX.where(tokens.last! .== eosTokenId, MLXArray(Int64(eosTokenId)), older))
        }
        return tokens
    }

    func callAsFunction(_ ids: MLXArray, cache: ArraysCache?) -> MLXArray {
        let n = ngramSize
        // Slot 3 holds `token ^ eos`: XOR with eos again gives the token, and the
        // zeros that batching puts in a new row read as EOS (no tokens yet).
        let previous =
            cache?[3].map { $0 ^ eosTokenId }
            ?? MLXArray.full(
                [ids.dim(0), n - 1], values: MLXArray(eosTokenId), dtype: ids.dtype)
        let history = concatenated([previous, ids], axis: 1)
        if let cache {
            cache[3] = keepLast(history, n: n - 1, lengths: cache.currentLengths) ^ eosTokenId
        }

        let tokens = ngramTokens(history.asType(.int64))

        // Step s gives the hash of the (s+1)-gram.
        var mixed = tokens[0] * _mults[0]
        var rows: [MLXArray] = []
        for s in 1 ..< n {
            mixed = mixed ^ (tokens[s] * _mults[s])
            let lo = (s - 1) * headsPerNgram
            rows.append(mixed[.ellipsis, .newAxis] % _sizes[lo ..< (lo + headsPerNgram)])
        }
        let hashed = concatenated(rows, axis: -1)

        return concatenated(
            ngramEmbedding.enumerated().map { h, table in table(hashed[.ellipsis, h]) },
            axis: -1)
    }

    /// The constant a checkpoint stores under `name`, rebuilt from the config.
    func rebuiltConstant(_ name: String) -> MLXArray? {
        switch name {
        case "layer_multipliers": _mults
        case "ngram_heads_vocab_sizes": _sizes
        case "ngram_heads_offsets": _offsets
        default: nil
        }
    }
}

private final class Qwen4ExpPLELayer: Module {
    let d: Int
    let hc: Int
    let shortConvStateLength: Int

    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ args: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        d = args.hiddenSize
        hc = args.hcCount
        let hcDim = d * hc
        let k = args.pleConvKernelSize
        let dilation = args.ngramSize
        shortConvStateLength = (k - 1) * dilation
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(args, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, hcDim, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, d, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: hcDim, outputChannels: hcDim, kernelSize: k, dilation: dilation,
            groups: hcDim, bias: false)
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let n = shortConvStateLength
        let state = cache?[2] ?? MLXArray.zeros([x.dim(0), n, x.dim(-1)], dtype: x.dtype)
        let full = concatenated([state, x], axis: 1)
        if let cache {
            cache[2] = keepLast(full, n: n, lengths: cache.currentLengths)
        }
        return silu(conv1d(full))
    }

    func callAsFunction(
        _ hidden: MLXArray, ids: MLXArray, cache: ArraysCache?, mask: MLXArray?
    ) -> MLXArray {
        var ids = ids
        if let mask {
            // Padding reads as EOS, like the start of a prompt.
            ids = MLX.where(mask, ids, MLXArray(pleEmbedding.eosTokenId, dtype: ids.dtype))
        }
        let embedding = pleEmbedding(ids, cache: cache)

        let prefix = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(embedding)).reshaped(prefix + [hc, d])
        let value = valueProj(embedding)
        let query = normQuery(hidden).reshaped(prefix + [hc, d])

        var gate = (key * query).sum(axis: -1, keepDims: true) / Float(d).squareRoot()
        gate = sqrt(maximum(abs(gate), Float(1e-6))) * sign(gate)
        var gated = (sigmoid(gate) * value[.ellipsis, .newAxis, 0...]).reshaped(prefix + [-1])
        if let mask {
            gated = MLX.where(mask[.ellipsis, .newAxis], gated, MLXArray(0, dtype: gated.dtype))
        }

        return gated + shortConv(normConv(gated), cache: cache)
    }
}

// MARK: - Model

private final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration, layerIndex: Int) {
        isLinear = args.layerTypes[layerIndex] == "linear_attention"
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(args)
        // `ple_layer_ids` counts layers from 1.
        if let pleIndex = args.pleLayerIds.firstIndex(of: layerIndex + 1) {
            _ple.wrappedValue = Qwen4ExpPLELayer(args, pleLayerIndex: pleIndex)
        }
        _attnHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        super.init()
    }

    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        convMask: MLXArray?,
        cache: KVCache?,
        ids: MLXArray
    ) -> MLXArray {
        var h = h
        if isLinear {
            let cache = cache.map { cache -> ArraysCache in
                guard let arrays = cache as? ArraysCache else {
                    preconditionFailure(
                        "qwen4_exp linear cache must come from newCache(parameters:)")
                }
                return arrays
            }
            if let ple {
                h = h + ple(h, ids: ids, cache: cache, mask: convMask)
            }
            let (x, inject) = attnHyperConnection(h)
            h = injectResidual(h, linearAttn!(x, mask: convMask, cache: cache), inject!)
        } else {
            let cache = cache.map { cache -> CacheList in
                guard let list = cache as? CacheList else {
                    preconditionFailure(
                        "qwen4_exp attention cache must come from newCache(parameters:)")
                }
                return list
            }
            let (x, inject) = attnHyperConnection(h)
            h = injectResidual(h, selfAttn!(x, mask: mask, cache: cache), inject!)
        }

        let (x, inject) = mlpHyperConnection(h)
        return injectResidual(h, mlp(x), inject!)
    }
}

private final class Qwen4ExpModelInner: Module {
    let hc: Int
    let firstAttention: Int
    let firstLinear: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Qwen4ExpDecoderLayer]
    // No final norm: the mixer normalizes the streams before it merges them.
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration) {
        hc = args.hcCount
        firstAttention = args.layerTypes.firstIndex(of: "full_attention")!
        firstLinear = args.layerTypes.firstIndex(of: "linear_attention")!
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        layers = (0 ..< args.hiddenLayers).map { Qwen4ExpDecoderLayer(args, layerIndex: $0) }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: false)
        super.init()
    }

    func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(ids)

        // Each kind of layer shares the mask of its first layer.
        let attentionCache = (cache?[firstAttention] as? CacheList)?[0]
        let mask = createAttentionMask(h: h, cache: attentionCache)
        // The deltanet and PLE must skip the padding.
        let convMask = (cache?[firstLinear] as? ArraysCache)?.makeMask(N: h.dim(1))

        h = tiled(h, repetitions: [1, 1, hc])
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, convMask: convMask, cache: cache?[i], ids: ids)
        }
        return hyperConnectionMixer(h).mixed
    }
}

public final class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: Qwen4ExpTextConfiguration
    @ModuleInfo(key: "model") private var model: Qwen4ExpModelInner
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ args: Qwen4ExpConfiguration) {
        let text = args.text
        configuration = text
        vocabularySize = text.vocabularySize
        kvHeads = Array(repeating: text.kvHeads, count: text.hiddenLayers)
        _model.wrappedValue = Qwen4ExpModelInner(text)
        if !text.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
        }
        // Prompt cache files name the indexer key cache.
        Qwen4ExpIndexerKeyCache.register()
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Attention layers keep a `CacheList` of their keys and values, then the indexer keys.
    /// Linear layers keep the deltanet conv and recurrent state; a PLE layer adds its conv
    /// state and n-gram context.
    ///
    /// - Throws: `KVCacheConfigurationError.incompatibleCapacity` for a bounded KV cache:
    ///   the QSA indexer needs every past key.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        if let capacity = try parameters?.effectiveKVCacheCapacity() {
            throw KVCacheConfigurationError.incompatibleCapacity(
                expected: capacity.maxTokens, count: model.layers.count { !$0.isLinear })
        }
        return model.layers.map { layer in
            if layer.isLinear {
                return ArraysCache(size: layer.ple == nil ? 2 : 4)
            }
            return CacheList(KVCacheSimple(), Qwen4ExpIndexerKeyCache())
        }
    }

    // Not in this text-only port: the MTP head and the vision tower.
    public var excludedCheckpointNamespaces: [String] {
        ["mtp", "model.mtp", "model.visual", "visual", "vision_tower"]
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // HF and mlx-vlm (flat layout, `shards.{i}` n-gram names) store these norms as
        // `w` for a `1 + w` scale; other checkpoints already hold `1 + w`.
        let fold = checkpoint.weights.keys.contains { key in
            key.hasPrefix("model.language_model.")
                || (key.hasPrefix("language_model.") && key.contains(".ngram_embedding.shards."))
        }

        var checkpoint = try checkpoint.mapNames(Self.checkpointName)
        var weights = filterLMHeadWeights(
            from: checkpoint.weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for key in weights.keys.sorted() {
            let value = weights[key]!

            // The n-gram hash constants: check them against the config, then drop.
            if Self.ngramConstants.contains(where: { key.hasSuffix($0) }) {
                try checkNGramConstant(key, value: value)
                weights[key] = nil
                continue
            }

            // The experts stack as (E, out, in); gate_up_proj holds gate, then up.
            if key.hasSuffix("mlp.experts.gate_up_proj") {
                let base = key.dropLast("experts.gate_up_proj".count)
                let mid = value.dim(-2) / 2
                weights[key] = nil
                weights["\(base)switch_mlp.gate_proj.weight"] = value[.ellipsis, ..<mid, 0...]
                weights["\(base)switch_mlp.up_proj.weight"] = value[.ellipsis, mid..., 0...]
                continue
            }
            if key.hasSuffix("mlp.experts.down_proj") {
                let base = key.dropLast("experts.down_proj".count)
                weights[key] = nil
                weights["\(base)switch_mlp.down_proj.weight"] = value
                continue
            }

            // torch (C, 1, K) to mlx (C, K, 1); a converted weight has shape[1] > 1.
            if key.hasSuffix("conv1d.weight"), value.ndim == 3, value.dim(1) == 1 {
                weights[key] = value.transposed(0, 2, 1)
            }
            if fold, Self.foldOne.contains(where: { key.hasSuffix($0) }) {
                weights[key] = value + MLXArray(1, dtype: value.dtype)
            }
        }

        // Merge the checkpoint's block inject into the mix down.
        for key in weights.keys.sorted() where key.hasSuffix("block_inject_weight.weight") {
            let prefix = key.dropLast("block_inject_weight.weight".count)
            let down = "\(prefix)input_mix_weight_down"
            let inject = "\(prefix)block_inject_weight"
            try Self.moveQuantization(&checkpoint, from: [down, inject], to: [down])
            Self.mergeLinears(&weights, [down, inject], target: down)
        }

        // The checkpoint has equal row shards; the model keeps one table per head.
        for key in weights.keys.sorted() where key.hasSuffix("ngram_embedding.shards.0.weight") {
            let prefix = String(key.dropLast("shards.0.weight".count))
            let count = weights.keys.count {
                $0.hasPrefix("\(prefix)shards.") && $0.hasSuffix(".weight")
            }
            let sizes = try ngramEmbedding(forKey: key).headVocabSizes
            try Self.moveQuantization(
                &checkpoint, from: (0 ..< count).map { "\(prefix)shards.\($0)" },
                to: sizes.indices.map { "\(prefix)\($0)" })
            // A quantized table splits its scales and biases by the same rows.
            for name in ["weight", "scales", "biases"]
            where weights["\(prefix)shards.0.\(name)"] != nil {
                let shards = try (0 ..< count).map { i in
                    guard let shard = weights.removeValue(forKey: "\(prefix)shards.\(i).\(name)")
                    else {
                        throw ModelFactoryError.invalidConfiguration(
                            "qwen4_exp: missing \(prefix)shards.\(i).\(name)")
                    }
                    return shard
                }
                for (h, table) in try Self.splitRows(shards, sizes: sizes).enumerated() {
                    weights["\(prefix)\(h).\(name)"] = table
                }
            }
        }

        checkpoint.weights = weights
        return checkpoint
    }

    // Norms stored as `w` for a `1 + w` scale; `linear_attn.norm` is not one.
    private static let foldOne = [
        "q_layernorm.weight", "k_layernorm.weight", "q_norm.weight", "k_norm.weight",
        "hc_norm.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight",
    ]

    private static let ngramConstants = [
        "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_vocab_sizes",
        "ple_embedding.ngram_heads_offsets",
    ]

    /// The model name of a checkpoint tensor or layer, or `nil` to drop it.
    private static func checkpointName(_ name: String) -> String? {
        if name.hasPrefix("mtp.") || name.hasPrefix("model.mtp.") {
            return nil
        }
        if name.hasPrefix("model.visual.") || name.hasPrefix("visual.")
            || name.hasPrefix("vision_tower.")
        {
            return nil
        }
        var name = name
        // Prefixes: HF `model.language_model.`, mlx-vlm `language_model.`.
        if name.hasPrefix("model.language_model.") {
            name = "model." + name.dropFirst("model.language_model.".count)
        } else if name.hasPrefix("language_model.") {
            name.removeFirst("language_model.".count)
        }
        // Rename the n-gram shards to `shards.{i}`, as mlx-vlm saves them.
        return name.replacing(/\.ngram_embedding\.shard_(\d+)\./) {
            ".ngram_embedding.shards.\($0.1)."
        }
    }

    private func ngramEmbedding(forKey key: String) throws -> Qwen4ExpNGramEmbedding {
        let index = key.components(separatedBy: ".layers.").dropFirst().first?
            .split(separator: ".").first.flatMap { Int($0) }
        guard let index, model.layers.indices.contains(index),
            let ple = model.layers[index].ple
        else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: \(key) is not in a ple_layer_ids layer")
        }
        return ple.pleEmbedding
    }

    private func checkNGramConstant(_ key: String, value: MLXArray) throws {
        let table = try ngramEmbedding(forKey: key)
        let rebuilt = table.rebuiltConstant(String(key.split(separator: ".").last!))!
        if !arrayEqual(rebuilt, value.asType(rebuilt.dtype)).item(Bool.self) {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: \(key): the n-gram hash constants rebuilt from the config do not "
                    + "match the checkpoint (\(rebuilt.asArray(Int64.self)) vs "
                    + "\(value.asType(.int64).asArray(Int64.self))); check `seed` and the hash "
                    + "parameters")
        }
    }

    /// Give the layers `destinations` the one quantization the layers `sources` share,
    /// in place of the sources' own settings.
    private static func moveQuantization(
        _ checkpoint: inout ModelCheckpoint, from sources: [String], to destinations: [String]
    ) throws {
        guard var settings = checkpoint.perLayerQuantization, let first = sources.first else {
            return
        }
        let quantization = settings.quantization(layer: first)
        guard sources.allSatisfy({ settings.quantization(layer: $0) == quantization }) else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: \(sources.joined(separator: ", ")) are stored as one layer "
                    + "but quantized differently")
        }
        for source in sources {
            settings.perLayerQuantization[source] = nil
        }
        for destination in destinations {
            settings.perLayerQuantization[destination] = quantization.map { .quantize($0) } ?? .skip
        }
        checkpoint.perLayerQuantization = settings
    }

    /// Concatenate the layers `parts` into `target` along the output axis, with their
    /// scales and biases when quantized.
    private static func mergeLinears(
        _ weights: inout [String: MLXArray], _ parts: [String], target: String
    ) {
        for name in ["weight", "scales", "biases"] {
            let keys = parts.map { "\($0).\(name)" }
            if keys.allSatisfy({ weights[$0] != nil }) {
                weights["\(target).\(name)"] = concatenated(
                    keys.map { weights.removeValue(forKey: $0)! }, axis: -2)
            }
        }
    }

    /// Split a table stored as equal row shards into tables of `sizes` rows.
    private static func splitRows(_ shards: [MLXArray], sizes: [Int]) throws -> [MLXArray] {
        let rows = shards[0].dim(0)
        guard sizes.reduce(0, +) <= rows * shards.count else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: n-gram shards hold \(rows * shards.count) rows, need \(sizes.reduce(0, +))"
            )
        }
        var tables: [MLXArray] = []
        var start = 0
        for size in sizes {
            let end = start + size
            let first = start / rows
            let last = (end - 1) / rows
            let pieces = (first ... last).map { i in
                shards[i][Swift.max(start - i * rows, 0) ..< Swift.min(end - i * rows, rows)]
            }
            tables.append(concatenated(pieces))
            start = end
        }
        return tables
    }
}

// MARK: - LoRA

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

// The checkpoint's template prompts Qwen3-Coder XML calls inside `<think>` turns.
// No hard-budget transition has been measured for this model.
extension Qwen4ExpModel {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
