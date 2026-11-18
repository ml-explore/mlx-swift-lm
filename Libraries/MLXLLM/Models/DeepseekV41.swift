// Copyright © 2026 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v41.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for DeepSeek-V4.1 (`deepseek_v41`), decoded from `config.json`.
///
/// A config that nests the text model under `text_config` is read from there, with
/// `model_type` taken from the top level.
public struct DeepseekV41Configuration: Codable, Sendable {
    /// The fp8 metadata of DeepSeek's own checkpoint, read only to reject formats the loader
    /// cannot keep packed.
    public struct QuantizationConfig: Codable, Sendable {
        var quantMethod: String?
        var scaleFmt: String?
        var weightBlockSize: [Int]?

        enum CodingKeys: String, CodingKey {
            case quantMethod = "quant_method"
            case scaleFmt = "scale_fmt"
            case weightBlockSize = "weight_block_size"
        }
    }

    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var headDim: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var oGroups: Int
    var oLoraRank: Int
    var moeIntermediateSize: Int
    var nRoutedExperts: Int
    var numExpertsPerTok: Int
    var scoringFunc: String
    var gateTemp: Float
    var normTopkProb: Bool
    var routedScalingFactor: Float
    var swigluLimit: Float
    var rmsNormEps: Float

    var slidingWindow: Int
    var compressRatios: [Int]
    var kvSourceLayerIds: [Int]
    var indexSourceLayerIds: [Int]
    var compressRopeTheta: Float
    var candidateSourceLayerId: Int
    var candidateTopkBlocks: Int
    var candidateBlockSize: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    var indexTopk: Int

    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float

    var engramLayerIds: [Int]
    var quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case moeIntermediateSize = "moe_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case gateTemp = "gate_temp"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case kvSourceLayerIds = "kv_source_layer_ids"
        case indexSourceLayerIds = "index_source_layer_ids"
        case compressRopeTheta = "compress_rope_theta"
        case candidateSourceLayerId = "candidate_source_layer_id"
        case candidateTopkBlocks = "candidate_topk_blocks"
        case candidateBlockSize = "candidate_block_size"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case engramLayerIds = "engram_layer_ids"
        case quantizationConfig = "quantization_config"
    }

    private enum WrapperKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let wrapper = try decoder.container(keyedBy: WrapperKeys.self)
        let c =
            wrapper.contains(.textConfig)
            ? try wrapper.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try wrapper.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v41"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1280
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        oGroups = try c.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        oLoraRank = try c.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1024
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2304
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 384
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        gateTemp = try c.decodeIfPresent(Float.self, forKey: .gateTemp) ?? 1.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-20

        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        compressRatios = try c.decodeIfPresent([Int].self, forKey: .compressRatios) ?? []
        kvSourceLayerIds = try c.decodeIfPresent([Int].self, forKey: .kvSourceLayerIds) ?? []
        indexSourceLayerIds =
            try c.decodeIfPresent([Int].self, forKey: .indexSourceLayerIds) ?? []
        compressRopeTheta =
            try c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160000.0
        candidateSourceLayerId =
            try c.decodeIfPresent(Int.self, forKey: .candidateSourceLayerId) ?? -1
        candidateTopkBlocks =
            try c.decodeIfPresent(Int.self, forKey: .candidateTopkBlocks) ?? 2048
        candidateBlockSize = try c.decodeIfPresent(Int.self, forKey: .candidateBlockSize) ?? 8
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 512

        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6

        engramLayerIds = try c.decodeIfPresent([Int].self, forKey: .engramLayerIds) ?? []
        quantizationConfig =
            try wrapper.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig)
            ?? c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig)
    }
}

extension DeepseekV41Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "DeepseekV41Configuration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "deepseek_v41: \(message)"])
        }
        if !engramLayerIds.isEmpty {
            throw fail(
                "Layers \(engramLayerIds) use engram memory, which this implementation does "
                    + "not have: the n-gram hash tables, the hashed embedding lookup and the "
                    + "engram attention are all missing.")
        }
        // A source layer reads its own compress ratio, so it must have one.
        for layer in Set(kvSourceLayerIds + indexSourceLayerIds)
        where layer < numHiddenLayers && layer >= compressRatios.count {
            throw fail("source layer \(layer) has no entry in compress_ratios")
        }
        guard hcMult > 0, numAttentionHeads % oGroups == 0 else {
            throw fail("hc_mult must be positive and o_groups must divide num_attention_heads")
        }
        if let quantizationConfig, quantizationConfig.quantMethod == "fp8" {
            // Only power-of-two scales over 32x32 blocks are mxfp8 and stay packed.
            guard quantizationConfig.scaleFmt == "ue8m0",
                quantizationConfig.weightBlockSize == [32, 32]
            else {
                throw fail("fp8 checkpoints need ue8m0 scales over 32x32 blocks")
            }
        }
    }
}

// MARK: - RoPE and fake quantization

/// `n ** -0.5` in double precision, rounded once to float32 as Python does.
private func deepseekV41InverseSqrt(_ n: Int) -> Float {
    Float(1 / Double(n).squareRoot())
}

/// Apply interleaved RoPE to the trailing `ropeDim` features.
func deepseekV41ApplyRope(
    _ x: MLXArray, _ rope: YarnRoPE, ropeDim: Int, offset: Int = 0, scale: Float = 1.0,
    inverse: Bool = false
) -> MLXArray {
    if ropeDim == 0 {
        return x
    }
    let width = x.dim(-1)
    let head = x[.ellipsis, ..<(width - ropeDim)]
    var tail = x[.ellipsis, (width - ropeDim)...].asType(.float32)
    if x.ndim == 4 {
        tail = tail.transposed(0, 2, 1, 3)
        tail = rope(tail, offset: offset, scale: scale, inverse: inverse)
        tail = tail.transposed(0, 2, 1, 3)
    } else {
        tail = rope(tail, offset: offset, scale: scale, inverse: inverse)
    }
    return concatenated([head, tail.asType(x.dtype)], axis: -1)
}

/// Round-trip FP8 e4m3 with power-of-two block scales.
func deepseekV41FakeQuantFP8(_ x: MLXArray, blockSize: Int = 32) -> MLXArray {
    if x.dim(-1) % blockSize != 0 {
        return x
    }
    let shape = x.shape
    let blocks = x.asType(.float32).reshaped(Array(shape.dropLast()) + [-1, blockSize])
    let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), Float(1e-4))
    let scale = pow(Float(2), ceil(log2(amax / Float(448))))
    let quantized = clip(blocks / scale, min: Float(-448), max: Float(448))
    return (fromFP8(toFP8(quantized), dtype: .float32) * scale).reshaped(shape).asType(x.dtype)
}

func deepseekV41RoundFP4(_ x: MLXArray) -> MLXArray {
    let lut = MLXArray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] as [Float])
    let magnitude = abs(x)
    var index = MLXArray.zeros(magnitude.shape, dtype: .int32)
    for threshold: Float in [0.25, 1.25, 2.5, 5.0] {
        index = index + (magnitude .> threshold).asType(.int32)
    }
    for threshold: Float in [0.75, 1.75, 3.5] {
        index = index + (magnitude .>= threshold).asType(.int32)
    }
    return sign(x) * take(lut, index)
}

/// Round-trip FP4 e2m1 with power-of-two (or, with `e4m3Scale`, fp8) block scales.
func deepseekV41FakeQuantFP4(_ x: MLXArray, blockSize: Int, e4m3Scale: Bool = false)
    -> MLXArray
{
    if x.dim(-1) % blockSize != 0 {
        return x
    }
    let shape = x.shape
    let blocks = x.asType(.float32).reshaped(Array(shape.dropLast()) + [-1, blockSize])
    let floor: Float = e4m3Scale ? 6.0 * pow(2, -9) : 6.0 * pow(2, -126)
    let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), floor)
    let scale =
        e4m3Scale
        ? fromFP8(toFP8(amax / Float(6)), dtype: .float32)
        : pow(Float(2), ceil(log2(amax / Float(6))))
    let q = deepseekV41RoundFP4(clip(blocks / scale, min: Float(-6), max: Float(6))) * scale
    return q.reshaped(shape).asType(x.dtype)
}

// MARK: - Hyper-connections

// The loop of `deepseekV41Sinkhorn` in one launch, one thread per matrix, in the same order.
private let deepseekV41SinkhornKernel = MLXFast.metalKernel(
    name: "deepseek_v41_sinkhorn",
    inputNames: ["comb", "eps"],
    outputNames: ["out"],
    source: """
            uint n = thread_position_in_grid.x;
            if (n >= comb_shape[0]) {
                return;
            }
            float m[HC * HC];
            for (int k = 0; k < HC * HC; k++) {
                m[k] = comb[n * HC * HC + k];
            }
            float e = eps[0];
            for (int t = 0; t < 2 * ITERS - 1; t++) {
                bool rows = t % 2 == 1;
                for (int a = 0; a < HC; a++) {
                    float s = 0.0f;
                    for (int b = 0; b < HC; b++) {
                        s += rows ? m[a * HC + b] : m[b * HC + a];
                    }
                    float d = s + e;
                    for (int b = 0; b < HC; b++) {
                        int k = rows ? a * HC + b : b * HC + a;
                        m[k] = m[k] / d;
                    }
                }
            }
            for (int k = 0; k < HC * HC; k++) {
                out[n * HC * HC + k] = m[k];
            }
        """)

/// Normalize the columns, then the rows and columns `iterations - 1` more times. The kernel
/// runs on the GPU outside training (it has no gradient); the ops path is the same math.
func deepseekV41Sinkhorn(
    _ comb: MLXArray, iterations: Int, eps: MLXArray, useKernel: Bool = true
) -> MLXArray {
    guard useKernel, Device.defaultDevice().deviceType == .gpu, iterations >= 1, comb.size > 0
    else {
        var comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        for _ in 0 ..< max(iterations - 1, 0) {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        }
        return comb
    }
    let hc = comb.dim(-1)
    let flat = comb.reshaped(-1, hc, hc)
    let count = flat.dim(0)
    let out = deepseekV41SinkhornKernel(
        [flat, eps],
        template: [("HC", hc), ("ITERS", iterations)],
        grid: (count, 1, 1),
        threadGroup: (min(count, 256), 1, 1),
        outputShapes: [flat.shape],
        outputDTypes: [.float32])[0]
    return out.reshaped(comb.shape)
}

private final class DeepseekV41HyperConnection: Module {
    let hc: Int
    let iterations: Int
    let hcEps: Float
    let normEps: Float
    // The leading underscores keep these out of the parameters.
    let _eps: MLXArray
    // The scale of each mix column: pre, post, then comb.
    let _scaleIds: MLXArray

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ args: DeepseekV41Configuration) {
        self.hc = args.hcMult
        self.iterations = args.hcSinkhornIters
        self.hcEps = args.hcEps
        self.normEps = args.rmsNormEps
        self._eps = MLXArray([args.hcEps])
        let ids =
            [Int32](repeating: 0, count: hc) + [Int32](repeating: 1, count: hc)
            + [Int32](repeating: 2, count: hc * hc)
        self._scaleIds = MLXArray(ids)
        let mix = (2 + hc) * hc
        _fn.wrappedValue = MLXArray.zeros([mix, hc * args.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([3], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ streams: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // The reference normalizes in float32.
        var flat = streams.reshaped(streams.dim(0), streams.dim(1), -1).asType(.float32)
        flat = MLXFast.rmsNorm(flat, weight: MLXArray.mlxNone, eps: normEps)
        let mix = matmul(flat, fn.T) * take(scale, _scaleIds) + base
        let parts = split(mix, indices: [hc, 2 * hc], axis: -1)
        let pre = sigmoid(parts[0]) + hcEps
        let post = Float(2) * sigmoid(parts[1])
        var comb = parts[2].reshaped(Array(parts[2].shape.dropLast()) + [hc, hc])
        comb = softmax(comb, axis: -1) + hcEps
        let combined = deepseekV41Sinkhorn(
            comb, iterations: iterations, eps: _eps, useKernel: !training)
        return (pre, post, combined)
    }
}

private func deepseekV41Collapse(_ streams: MLXArray, _ mix: MLXArray) -> MLXArray {
    let out = (expandedDimensions(mix, axis: -1) * streams.asType(.float32)).sum(axis: 2)
    return out.asType(streams.dtype)
}

/// `post * x + comb^T @ residual`, as the reference's sum over the source stream.
private func deepseekV41Expand(
    _ x: MLXArray, _ residual: MLXArray, _ post: MLXArray, _ comb: MLXArray
) -> MLXArray {
    let mixed = matmul(comb.swappedAxes(-1, -2), residual.asType(.float32))
    let out = expandedDimensions(post, axis: -1) * expandedDimensions(x, axis: -2) + mixed
    return out.asType(residual.dtype)
}

private func deepseekV41Append(_ cached: MLXArray?, _ new: MLXArray) -> MLXArray {
    guard let cached else { return new }
    return concatenated([cached, new], axis: 1)
}

// MARK: - Cache

/// Cache for one decoder layer: the sliding window, the closed compressed groups and their
/// index keys, and the rows of the group that is still open.
///
/// Evicted window rows and closed groups cannot be restored, so the cache cannot be trimmed.
/// It saves to a prompt cache file under its own class name, registered when the model is made.
final class DeepseekV41LayerCache: BaseKVCache {
    override class var serializedClassName: String? { "MLXLLM.DeepseekV41LayerCache" }

    static let registration: Void = {
        try! registerKVCacheType(DeepseekV41LayerCache.self)
    }()

    let windowSize: Int
    var window: MLXArray?
    var compressKV: MLXArray?
    var indexK: MLXArray?
    var pendingKV: MLXArray?
    var pendingGate: MLXArray?

    init(windowSize: Int) {
        self.windowSize = windowSize
        super.init()
    }

    /// Return the window with `kv` appended, and keep its last rows.
    func updateWindow(_ kv: MLXArray) -> MLXArray {
        let window = deepseekV41Append(self.window, kv)
        let keep = min(windowSize, window.dim(1))
        self.window = window[0..., (window.dim(1) - keep)...]
        offset += kv.dim(1)
        return window
    }

    /// The window, compressed groups, index keys, open rows and open gates, in saved order.
    private var slots: [MLXArray?] { [window, compressKV, indexK, pendingKV, pendingGate] }

    override func innerState() -> [MLXArray] {
        slots.compactMap { $0 }
    }

    override var state: [MLXArray] {
        get { innerState() }
        set {
            precondition(newValue.isEmpty, "Use restore(state:metaState:) instead")
        }
    }

    /// The window size, the offset, and a `0`/`1` flag per slot for the arrays `state` holds.
    override var metaState: [String] {
        get { [String(windowSize), String(offset), slots.map { $0 == nil ? "0" : "1" }.joined()] }
        set {
            precondition(newValue == metaState, "Use restore(state:metaState:) instead")
        }
    }

    override class func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        guard metaState.count == 3, let windowSize = Int(metaState[0]), windowSize > 0,
            let offset = Int(metaState[1]), offset >= 0, metaState[2].count == 5,
            metaState[2].allSatisfy({ $0 == "0" || $0 == "1" }),
            metaState[2].filter({ $0 == "1" }).count == state.count,
            state.allSatisfy({ $0.ndim == 3 })
        else {
            throw DeepseekV41Weights.LoadError(
                message: "Corrupt prompt cache: invalid DeepseekV41LayerCache state or metadata.")
        }
        var arrays = state.makeIterator()
        let slots = metaState[2].map { $0 == "1" ? arrays.next() : nil }
        guard (slots[0]?.dim(1) ?? 0) <= windowSize else {
            throw DeepseekV41Weights.LoadError(
                message: "Corrupt prompt cache: the window holds more rows than its size.")
        }
        let cache = DeepseekV41LayerCache(windowSize: windowSize)
        (cache.window, cache.compressKV, cache.indexK, cache.pendingKV, cache.pendingGate) =
            (slots[0], slots[1], slots[2], slots[3], slots[4])
        cache.offset = offset
        return cache
    }

    override func copy() -> any KVCache {
        // The arrays are replaced, never written through, so sharing them is a deep copy.
        let new = DeepseekV41LayerCache(windowSize: windowSize)
        new.window = window
        new.compressKV = compressKV
        new.indexK = indexK
        new.pendingKV = pendingKV
        new.pendingGate = pendingGate
        new.offset = offset
        return new
    }
}

/// The state that later layers of one forward read from earlier ones.
private struct DeepseekV41SharedState {
    var compressKV: MLXArray?
    var indexK: MLXArray?
    var topkIdx: MLXArray?
    var candidates: MLXArray?
}

// MARK: - Compressor and indexer

private final class DeepseekV41Compressor: Module {
    let ratio: Int

    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    init(_ args: DeepseekV41Configuration, layerId: Int) {
        self.ratio = args.compressRatios[layerId]
        _kvProj.wrappedValue = Linear(args.hiddenSize, args.headDim, bias: false)
        if ratio > 1 {
            _gateProj.wrappedValue = Linear(args.hiddenSize, args.headDim, bias: false)
        }
        _kvNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        super.init()
    }

    /// Pool each full group of `ratio` tokens into one latent. Return the latents, or `nil`
    /// if no group is full, and the index of the first one.
    func callAsFunction(_ x: MLXArray, startPos: Int, cache: DeepseekV41LayerCache)
        -> (MLXArray?, Int)
    {
        if ratio == 1 {
            return (kvNorm(kvProj(x)), startPos)
        }
        guard let gateProj else { fatalError("compress ratio \(ratio) needs gate_proj") }

        let h = x.asType(.float32)
        let pending = cache.pendingKV?.dim(1) ?? 0
        var kv = deepseekV41Append(cache.pendingKV, kvProj(h))
        var gate = deepseekV41Append(cache.pendingGate, gateProj(h))
        let full = kv.dim(1) - kv.dim(1) % ratio
        cache.pendingKV = kv[0..., full...]
        cache.pendingGate = gate[0..., full...]
        // A group takes the position of its first token, and the RoPE scale multiplies the
        // offset by the ratio, so return the group index.
        let first = (startPos - pending) / ratio
        if full == 0 {
            return (nil, first)
        }
        let (batch, dim) = (kv.dim(0), kv.dim(2))
        kv = kv[0..., ..<full].reshaped(batch, -1, ratio, dim)
        gate = gate[0..., ..<full].reshaped(batch, -1, ratio, dim)
        let latent = (kv * softmax(gate, axis: 2)).sum(axis: 2)
        return (kvNorm(latent.asType(x.dtype)), first)
    }
}

/// The window rows each query attends to, -1 for none. The window holds the previous rows,
/// then the new ones.
private func deepseekV41WindowIndices(previous: Int, length: Int, window: Int) -> MLXArray {
    let width = min(window, previous + length)
    let end = MLXArray(Int32(previous) ..< Int32(previous + length))
    let start = maximum(end - Int32(window - 1), Int32(0))
    let indices =
        expandedDimensions(start, axis: 1)
        + expandedDimensions(MLXArray(Int32(0) ..< Int32(width)), axis: 0)
    return MLX.where(indices .> expandedDimensions(end, axis: 1), Int32(-1), indices)
}

private func deepseekV41GatherRows(_ values: MLXArray, _ indices: MLXArray) -> MLXArray {
    let (batch, length, dim) = (values.dim(0), values.dim(1), values.dim(2))
    let safe = maximum(indices, Int32(0))
    let base = (MLXArray(Int32(0) ..< Int32(batch)) * Int32(length)).reshaped(batch, 1, 1)
    let flat = values.reshaped(batch * length, dim)
    return take(flat, (safe + base).reshaped(-1), axis: 0).reshaped(indices.shape + [dim])
}

/// Attend each query to the rows its indices pick from each KV array.
///
/// `q` is `[B, S, H, D]`, each KV array `[B, K, D]` and its indices `[B, S, k]`. An index of
/// -1 picks nothing.
private func deepseekV41SparseAttention(
    _ q: MLXArray, kvs: [MLXArray], indices: [MLXArray], sinks: MLXArray, scale: Float
) -> MLXArray {
    let (batch, length, heads, dim) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let rows = zip(kvs, indices).map { deepseekV41GatherRows($0, $1) }
    // Each query has its own keys, so each query is one batch entry.
    let keys = concatenated(rows, axis: 2).reshaped(batch * length, 1, -1, dim)
        .asType(.float32)
    let mask = concatenated(indices, axis: -1) .>= Int32(0)
    // SDPA rounds the scores to the input dtype at this head size, so use fp32.
    let out = MLXFast.scaledDotProductAttention(
        queries: q.reshaped(batch * length, heads, 1, dim).asType(.float32),
        keys: keys,
        values: keys,
        scale: scale,
        mask: .array(mask.reshaped(batch * length, 1, 1, -1)),
        sinks: sinks.asType(.float32))
    return out.reshaped(batch, length, heads, dim).asType(q.dtype)
}

private final class DeepseekV41Indexer: Module {
    let ratio: Int
    let isCandidateSource: Bool
    let candidateTopkBlocks: Int
    let candidateBlockSize: Int
    let numHeads: Int
    let headDim: Int
    let indexTopk: Int
    let ropeDim: Int

    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(_ args: DeepseekV41Configuration, layerId: Int) {
        self.ratio = args.compressRatios[layerId]
        self.isCandidateSource = layerId == args.candidateSourceLayerId
        self.candidateTopkBlocks = args.candidateTopkBlocks
        self.candidateBlockSize = args.candidateBlockSize
        self.numHeads = args.indexNHeads
        self.headDim = args.indexHeadDim
        self.indexTopk = args.indexTopk
        self.ropeDim = args.qkRopeHeadDim
        _qBProj.wrappedValue = Linear(args.qLoraRank, numHeads * headDim, bias: false)
        _weightsProj.wrappedValue = Linear(args.hiddenSize, numHeads, bias: false)
        // Only a layer that compresses its own KV makes index keys.
        if args.kvSourceLayerIds.contains(layerId) {
            _kProj.wrappedValue = Linear(args.headDim, args.indexHeadDim, bias: false)
            _kNorm.wrappedValue = RMSNorm(dimensions: args.indexHeadDim, eps: args.rmsNormEps)
        }
        super.init()
    }

    /// Return the top-k compressed entries of each query, -1 for none, and the candidate
    /// mask: the one this layer selects, or the given one.
    func callAsFunction(
        _ x: MLXArray, qResidual: MLXArray, indexK: MLXArray?, candidates: MLXArray?,
        rope: YarnRoPE, startPos: Int
    ) -> (MLXArray, MLXArray?) {
        let (batch, length) = (x.dim(0), x.dim(1))
        guard let indexK, indexK.dim(1) > 0 else {
            return (MLXArray.zeros([batch, length, 0], dtype: .int32), candidates)
        }

        var q = qBProj(qResidual).reshaped(batch, length, numHeads, headDim)
        q = deepseekV41ApplyRope(q, rope, ropeDim: ropeDim, offset: startPos)
        q = deepseekV41FakeQuantFP4(q, blockSize: 32)
        let keys = indexK.asType(.float32)
        let weights = weightsProj(x).asType(.float32) * deepseekV41InverseSqrt(numHeads)
        var scores = matmul(
            q.asType(.float32), expandedDimensions(keys, axis: 1).swappedAxes(-1, -2))
        scores = maximum(scores, Float(0)) * deepseekV41InverseSqrt(headDim)
        scores = (scores * expandedDimensions(weights, axis: -1)).sum(axis: 2)
        // A query sees the groups that end at or before it.
        let count = indexK.dim(1)
        let lens = expandedDimensions(
            MLXArray(Int32(startPos + 1) ..< Int32(startPos + 1 + length))
                .floorDivide(Int32(ratio)), axis: 1)
        let visible = expandedDimensions(MLXArray(Int32(0) ..< Int32(count)), axis: 0) .< lens
        scores = MLX.where(expandedDimensions(visible, axis: 0), scores, -Float.infinity)

        var candidates = candidates
        if isCandidateSource && candidateTopkBlocks > 0 {
            let block = candidateBlockSize
            let pad = (block - count % block) % block
            let paddedScores = padded(
                scores, widths: [0, 0, .init((0, pad))], value: MLXArray(-Float.infinity))
            var blockScores = paddedScores.reshaped(batch, length, -1, block).max(axis: -1)
            // Always keep the newest block: it is only partly full, so an older full block
            // can have a higher score.
            let last = MLX.where(
                lens .> Int32(0), (lens - Int32(1)).floorDivide(Int32(block)), Int32(-1))
            let numBlocks = blockScores.dim(-1)
            blockScores = MLX.where(
                MLXArray(Int32(0) ..< Int32(numBlocks)) .== last, Float.infinity, blockScores)
            let kBlocks = min(candidateTopkBlocks, numBlocks)
            let chosen = argPartition(-blockScores, kth: kBlocks - 1, axis: -1)[
                .ellipsis, ..<kBlocks]
            // With fewer reachable blocks than kBlocks, the extra picks are -inf.
            let reachable = takeAlong(blockScores, chosen, axis: -1) .> -Float.infinity
            var mask = MLXArray.zeros(blockScores.shape, dtype: .bool)
            mask = putAlong(mask, chosen, values: reachable, axis: -1)
            candidates = repeated(mask, count: block, axis: -1)[.ellipsis, ..<count]
        } else if let candidates {
            scores = MLX.where(candidates, scores, -Float.infinity)
        }

        let k = min(indexTopk, count)
        let chosen = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k].asType(.int32)
        return (
            MLX.where(chosen .< expandedDimensions(lens, axis: 0), chosen, Int32(-1)), candidates
        )
    }
}

// MARK: - Attention

private final class DeepseekV41Attention: Module {
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let oGroups: Int
    let ratio: Int
    let isKVSource: Bool
    let isIndexSource: Bool
    let _groupIds: MLXArray
    let rope: YarnRoPE

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_norm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    // wo_a is block diagonal: one projection per output group.
    @ModuleInfo(key: "o_a_proj") var oAProj: SwitchLinear
    @ModuleInfo(key: "o_b_proj") var oBProj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV41Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV41Indexer?

    init(_ args: DeepseekV41Configuration, layerId: Int) {
        self.numHeads = args.numAttentionHeads
        self.headDim = args.headDim
        self.ropeDim = args.qkRopeHeadDim
        self.oGroups = args.oGroups
        self.ratio = layerId < args.compressRatios.count ? args.compressRatios[layerId] : 0
        self.isKVSource = args.kvSourceLayerIds.contains(layerId)
        self.isIndexSource = args.indexSourceLayerIds.contains(layerId)
        self._groupIds = MLXArray(Int32(0) ..< Int32(args.oGroups)).reshaped(1, -1)

        _qAProj.wrappedValue = Linear(args.hiddenSize, args.qLoraRank, bias: false)
        _qANorm.wrappedValue = RMSNorm(dimensions: args.qLoraRank, eps: args.rmsNormEps)
        _qBProj.wrappedValue = Linear(
            args.qLoraRank, args.numAttentionHeads * args.headDim, bias: false)
        _kvProj.wrappedValue = Linear(args.hiddenSize, args.headDim, bias: false)
        _kvNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _oAProj.wrappedValue = SwitchLinear(
            inputDims: args.numAttentionHeads * args.headDim / args.oGroups,
            outputDims: args.oLoraRank, numExperts: args.oGroups, bias: false)
        _oBProj.wrappedValue = Linear(
            args.oGroups * args.oLoraRank, args.hiddenSize, bias: false)
        _sinks.wrappedValue = MLXArray.zeros([args.numAttentionHeads], dtype: .float32)
        if isKVSource {
            _compressor.wrappedValue = DeepseekV41Compressor(args, layerId: layerId)
        }
        if isIndexSource {
            _indexer.wrappedValue = DeepseekV41Indexer(args, layerId: layerId)
        }
        if ratio != 0 {
            let scaling = args.ropeScaling ?? [:]
            let original =
                scaling["original_max_position_embeddings"]?.asFloat()
                ?? Float(args.maxPositionEmbeddings)
            // V4.1 uses attention_factor=1 for its YaRN RoPE.
            self.rope = YarnRoPE(
                dimensions: args.qkRopeHeadDim,
                traditional: true,
                base: args.compressRopeTheta,
                scalingFactor: scaling["factor"]?.asFloat() ?? 1.0,
                originalMaxPositionEmbeddings: Int(original),
                betaFast: Float(Int(scaling["beta_fast"]?.asFloat() ?? 32)),
                betaSlow: Float(Int(scaling["beta_slow"]?.asFloat() ?? 1)),
                mscale: 0,
                mscaleAllDim: 0)
        } else {
            // A layer with only the window uses the base theta and no YaRN.
            self.rope = YarnRoPE(
                dimensions: args.qkRopeHeadDim, traditional: true, base: args.ropeTheta)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, startPos: Int, windowIdx: MLXArray, cache: DeepseekV41LayerCache,
        shared: inout DeepseekV41SharedState
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let qResidual = qANorm(qAProj(x))
        var q = qBProj(qResidual).reshaped(batch, length, numHeads, headDim)
        q = deepseekV41ApplyRope(q, rope, ropeDim: ropeDim, offset: startPos)

        var kv = kvNorm(kvProj(x))
        kv = deepseekV41ApplyRope(kv, rope, ropeDim: ropeDim, offset: startPos)
        kv = deepseekV41FakeQuantFP8(kv, blockSize: 32)
        var kvs = [cache.updateWindow(kv)]
        var indices = [windowIdx]
        if ratio != 0 {
            if isKVSource {
                compress(x, startPos: startPos, cache: cache, shared: &shared)
            }
            if isIndexSource, let indexer {
                let (topk, candidates) = indexer(
                    x, qResidual: qResidual, indexK: shared.indexK,
                    candidates: shared.candidates, rope: rope, startPos: startPos)
                shared.topkIdx = topk
                if indexer.isCandidateSource {
                    shared.candidates = candidates
                }
            }
            if let compressed = shared.compressKV, let topk = shared.topkIdx, topk.dim(-1) > 0 {
                kvs.append(compressed)
                indices.append(topk)
            }
        }

        var output = deepseekV41SparseAttention(
            q, kvs: kvs, indices: indices, sinks: sinks, scale: deepseekV41InverseSqrt(headDim))
        output = deepseekV41ApplyRope(
            output, rope, ropeDim: ropeDim, offset: startPos, inverse: true)
        output = output.reshaped(batch * length, oGroups, 1, -1)
        output = oAProj(output, _groupIds)
        return oBProj(output.reshaped(batch, length, -1))
    }

    /// Add the new compressed KV and index keys to the cache, and share them.
    private func compress(
        _ x: MLXArray, startPos: Int, cache: DeepseekV41LayerCache,
        shared: inout DeepseekV41SharedState
    ) {
        guard let compressor else { return }
        let (latent, first) = compressor(x, startPos: startPos, cache: cache)
        if let latent {
            var kv = deepseekV41ApplyRope(
                latent, rope, ropeDim: ropeDim, offset: first, scale: Float(ratio))
            kv = deepseekV41FakeQuantFP4(kv, blockSize: 16, e4m3Scale: true)
            cache.compressKV = deepseekV41Append(cache.compressKV, kv)
            if isIndexSource, let indexer, let kProj = indexer.kProj, let kNorm = indexer.kNorm {
                var k = kNorm(kProj(latent))
                k = deepseekV41ApplyRope(
                    k, rope, ropeDim: ropeDim, offset: first, scale: Float(ratio))
                k = deepseekV41FakeQuantFP4(k, blockSize: 32)
                cache.indexK = deepseekV41Append(cache.indexK, k)
            }
        }
        shared.compressKV = cache.compressKV
        if isIndexSource {
            shared.indexK = cache.indexK
        }
    }
}

// MARK: - Feed-forward

/// DeepSeek-V4.1's clamped SwiGLU activation.
private func deepseekV41SwiGLU(up: MLXArray, gate: MLXArray, limit: Float) -> MLXArray {
    let dtype = up.dtype
    var gate = gate.asType(.float32)
    var up = up.asType(.float32)
    if limit > 0 {
        gate = minimum(gate, limit)
        up = clip(up, min: -limit, max: limit)
    }
    return (silu(gate) * up).asType(dtype)
}

private final class DeepseekV41SharedMLP: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: DeepseekV41Configuration) {
        self.limit = args.swigluLimit
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.moeIntermediateSize, bias: false)
        _upProj.wrappedValue = Linear(args.hiddenSize, args.moeIntermediateSize, bias: false)
        _downProj.wrappedValue = Linear(args.moeIntermediateSize, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = deepseekV41SwiGLU(up: upProj(x), gate: gateProj(x), limit: limit)
        return downProj(activated.asType(x.dtype))
    }
}

private final class DeepseekV41Router: Module {
    let topK: Int
    let scoreFunc: String
    let gateTemp: Float
    let normTopkProb: Bool
    let routeScale: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: DeepseekV41Configuration) {
        self.topK = args.numExpertsPerTok
        self.scoreFunc = args.scoringFunc
        self.gateTemp = args.gateTemp
        self.normTopkProb = args.normTopkProb
        self.routeScale = args.routedScalingFactor
        _weight.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts, args.hiddenSize], dtype: .float32)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        logits = logits / gateTemp
        let scores: MLXArray
        switch scoreFunc {
        case "sigmoid": scores = sigmoid(logits)
        case "softmax": scores = softmax(logits, axis: -1)
        default: scores = sqrt(softplus(logits))
        }
        let biased = scores + eScoreCorrectionBias
        let chosen = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, chosen, axis: -1)
        if normTopkProb && topK > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + Float(1e-20))
        }
        return (chosen, weights * routeScale)
    }
}

private final class DeepseekV41Experts: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: DeepseekV41Configuration) {
        let limit = args.swigluLimit
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts,
            activationProduct: { gate, up in deepseekV41SwiGLU(up: up, gate: gate, limit: limit) }
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let routed = switchMLP(x, indices)
        return (routed.asType(.float32) * expandedDimensions(weights.asType(.float32), axis: -1))
            .sum(axis: 1)
    }
}

private final class DeepseekV41MoE: Module, UnaryLayer {
    let dim: Int

    @ModuleInfo(key: "gate") var gate: DeepseekV41Router
    @ModuleInfo(key: "experts") var experts: DeepseekV41Experts
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV41SharedMLP

    init(_ args: DeepseekV41Configuration) {
        self.dim = args.hiddenSize
        _gate.wrappedValue = DeepseekV41Router(args)
        _experts.wrappedValue = DeepseekV41Experts(args)
        _sharedExperts.wrappedValue = DeepseekV41SharedMLP(args)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let flat = x.reshaped(-1, dim)
        let (indices, weights) = gate(flat)
        let routed = experts(flat, indices: indices, weights: weights)
        let out = routed + sharedExperts(flat).asType(.float32)
        return out.reshaped(x.shape).asType(x.dtype)
    }
}

// MARK: - Decoder

private final class DeepseekV41DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV41Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV41MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV41HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV41HyperConnection

    init(_ args: DeepseekV41Configuration, layerId: Int) {
        _selfAttn.wrappedValue = DeepseekV41Attention(args, layerId: layerId)
        _mlp.wrappedValue = DeepseekV41MoE(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _attnHC.wrappedValue = DeepseekV41HyperConnection(args)
        _ffnHC.wrappedValue = DeepseekV41HyperConnection(args)
        super.init()
    }

    /// `streams` is `[B, L, hc, hidden]`. Each branch collapses the streams with the `pre`
    /// mix of the hyper-connection before it, as the reference does.
    func callAsFunction(
        _ streams: MLXArray, preMix: MLXArray, startPos: Int, windowIdx: MLXArray,
        cache: DeepseekV41LayerCache, shared: inout DeepseekV41SharedState
    ) -> (MLXArray, MLXArray) {
        let (attnPre, attnPost, attnComb) = attnHC(streams)
        var h = inputLayerNorm(deepseekV41Collapse(streams, preMix))
        h = selfAttn(h, startPos: startPos, windowIdx: windowIdx, cache: cache, shared: &shared)
        var streams = deepseekV41Expand(h, streams, attnPost, attnComb)

        let (ffnPre, ffnPost, ffnComb) = ffnHC(streams)
        h = mlp(postAttentionLayerNorm(deepseekV41Collapse(streams, attnPre)))
        streams = deepseekV41Expand(h, streams, ffnPost, ffnComb)
        return (streams, ffnPre)
    }
}

private final class DeepseekV41TextModel: Module {
    let args: DeepseekV41Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [DeepseekV41DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: DeepseekV41Configuration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.numHiddenLayers).map {
            DeepseekV41DecoderLayer(args, layerId: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func makeCache() -> [DeepseekV41LayerCache] {
        layers.map { _ in DeepseekV41LayerCache(windowSize: args.slidingWindow) }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [DeepseekV41LayerCache]) -> MLXArray {
        precondition(cache.count == layers.count, "deepseek_v41 needs one cache per layer")
        let h = embedTokens(inputs)
        let (batch, length) = (h.dim(0), h.dim(1))
        let startPos = cache[0].offset
        // Every layer's window holds the same rows, so all use the same indices.
        let previous = cache[0].window?.dim(1) ?? 0
        var windowIdx = deepseekV41WindowIndices(
            previous: previous, length: length, window: args.slidingWindow)
        windowIdx = broadcast(windowIdx, to: [batch] + windowIdx.shape)

        var streams = broadcast(
            expandedDimensions(h, axis: 2), to: [batch, length, args.hcMult, args.hiddenSize])
        var preMix = MLXArray.ones([batch, length, 1], dtype: .float32)
        if args.hcMult > 1 {
            preMix = concatenated(
                [preMix, MLXArray.zeros([batch, length, args.hcMult - 1], dtype: .float32)],
                axis: -1)
        }
        var shared = DeepseekV41SharedState()
        for (layer, layerCache) in zip(layers, cache) {
            (streams, preMix) = layer(
                streams, preMix: preMix, startPos: startPos, windowIdx: windowIdx,
                cache: layerCache, shared: &shared)
        }
        return norm(deepseekV41Collapse(streams, preMix))
    }
}

// MARK: - Model

/// DeepSeek-V4.1 text model (`deepseek_v41`).
///
/// Each layer keeps a `KVCache` of its own kind: the sliding window plus compressed history.
/// It cannot be trimmed, quantized or capped with `maxKVSize`. Prompt cache files can be
/// loaded once a model has been created. Engram layers, multi-token prediction and the vision
/// tower are not supported.
public final class DeepseekV41Model: Module, LLMModel {
    let args: DeepseekV41Configuration

    @ModuleInfo(key: "model") private var model: DeepseekV41TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: DeepseekV41Configuration) {
        _ = DeepseekV41LayerCache.registration
        self.args = args
        _model.wrappedValue = DeepseekV41TextModel(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        super.init()
    }

    /// Without a cache the call runs on fresh caches.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let layerCaches =
            cache.map { caches in
                caches.map { cache -> DeepseekV41LayerCache in
                    guard let cache = cache as? DeepseekV41LayerCache else {
                        fatalError("deepseek_v41 needs the caches from newCache(parameters:)")
                    }
                    return cache
                }
            } ?? model.makeCache()
        return lmHead(model(inputs, cache: layerCaches))
    }

    /// The window and the compressed history are the architecture's own, so `maxKVSize` and
    /// KV quantization do not apply. Generation rejects `maxKVSize` and typed quantization
    /// requests; the legacy `kvBits` request leaves these caches unquantized.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        _ = try parameters?.kvCachePlan()
        return model.makeCache()
    }

    /// Rename the checkpoint keys and keep the fp8/fp4 weights packed. The loader gives each
    /// packed layer the group size, bits and mode its weights were stored with.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try DeepseekV41Weights.sanitize(weights, args: args)
    }

    public var loraLayers: [Module] { model.layers }
}

// MARK: - Weight sanitization

enum DeepseekV41Weights {
    struct LoadError: LocalizedError {
        let message: String
        var errorDescription: String? { "deepseek_v41: \(message)" }
    }

    static let dropPrefixes = ["vision.", "aligner.", "image_", "mtp."]
    static let expertProjections = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

    /// Map one DeepSeek checkpoint name to its MLX name.
    static func rename(_ key: String) -> String {
        if key == "head.weight" {
            return "lm_head.weight"
        }
        if key == "embed.weight" {
            return "model.embed_tokens.weight"
        }
        var key = "model.\(key)"
        let replacements = [
            (".attn.", ".self_attn."),
            (".ffn.", ".mlp."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".hc_attn_", ".attn_hc."),
            (".hc_ffn_", ".ffn_hc."),
            (".attn_sink", ".sinks"),
            (".wq_a.", ".q_a_proj."),
            (".wq_b.", ".q_b_proj."),
            (".wkv.", ".kv_proj."),
            (".wgate.", ".gate_proj."),
            (".wo_a.", ".o_a_proj."),
            (".wo_b.", ".o_b_proj."),
            (".q_norm.", ".q_a_norm."),
            (".compressor.norm.", ".compressor.kv_norm."),
            (".indexer.wk.", ".indexer.k_proj."),
            (".gate.bias", ".gate.e_score_correction_bias"),
            (".shared_experts.w1.", ".shared_experts.gate_proj."),
            (".shared_experts.w2.", ".shared_experts.down_proj."),
            (".shared_experts.w3.", ".shared_experts.up_proj."),
        ]
        for (old, new) in replacements {
            key = key.replacingOccurrences(of: old, with: new)
        }
        return key
    }

    /// Put an fp8 or fp4 weight and its ue8m0 scale in MLX's quantized layout.
    static func repack(weight: MLXArray, scale: MLXArray) -> (MLXArray, MLXArray) {
        var scale = scale
        if scale.dtype != .uint8 {
            scale = (log2(scale.asType(.float32)) + 127).asType(.uint8)
        }
        let rows = weight.dim(-2)
        if scale.dim(-2) != rows {
            // fp8 keeps one scale per 32x32 tile, MLX one per output row.
            scale = repeated(scale, count: 32, axis: -2)[.ellipsis, ..<rows, 0...]
        }
        return (weight.view(dtype: .uint32), scale)
    }

    /// `layers.N.ffn.experts.E.wK.(weight|scale)` as `(N, E, wK, weight|scale)`.
    static func expertKey(_ key: String) -> (Int, Int, String, String)? {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 7, parts[0] == "layers", parts[2] == "ffn", parts[3] == "experts",
            parts[1].allSatisfy({ $0.isASCII && $0.isNumber }),
            parts[4].allSatisfy({ $0.isASCII && $0.isNumber }),
            let layer = Int(parts[1]), let expert = Int(parts[4]),
            expertProjections[parts[5]] != nil, ["weight", "scale"].contains(parts[6])
        else { return nil }
        return (layer, expert, parts[5], parts[6])
    }

    static func sanitize(_ weights: [String: MLXArray], args: DeepseekV41Configuration) throws
        -> [String: MLXArray]
    {
        // A converted checkpoint already has MLX names; the published one never starts a key
        // with "model.".
        if weights.keys.contains(where: { $0.hasPrefix("model.") }) {
            return weights
        }
        let text = weights.filter { key, _ in
            !dropPrefixes.contains(where: { key.hasPrefix($0) }) && !key.contains(".engram.")
        }
        var clean = [String: MLXArray]()
        struct ExpertGroup: Hashable {
            let layer: Int
            let projection: String
        }
        var experts = [ExpertGroup: [Int: [String: MLXArray]]]()
        for (key, value) in text {
            if let (layer, expert, projection, kind) = expertKey(key) {
                experts[ExpertGroup(layer: layer, projection: projection), default: [:]][
                    expert, default: [:]][kind] = value
                continue
            }
            // bias_vl routes image tokens, which the text model never sees.
            if key.hasSuffix(".gate.bias_vl") {
                continue
            }
            if key.hasSuffix(".scale"), text["\(key.dropLast(".scale".count)).weight"] != nil {
                continue
            }
            var value = value
            var scale: MLXArray?
            if key.hasSuffix(".weight") {
                scale = text["\(key.dropLast(".weight".count)).scale"]
            }
            let name = rename(key)
            if let s = scale {
                (value, scale) = repack(weight: value, scale: s)
            }
            if name.hasSuffix(".o_a_proj.weight") {
                // wo_a is block diagonal, so split it back into its groups.
                value = value.reshaped(args.oGroups, -1, value.dim(-1))
                scale = scale?.reshaped(args.oGroups, -1, scale!.dim(-1))
            }
            clean[name] = value
            if let scale {
                clean["\(name.dropLast("weight".count))scales"] = scale
            }
        }

        for (group, rows) in experts {
            guard rows.count == args.nRoutedExperts else {
                throw LoadError(
                    message:
                        "Layer \(group.layer) has \(rows.count) \(group.projection) expert "
                        + "weights, but the config declares \(args.nRoutedExperts) experts.")
            }
            let parts = rows.keys.sorted().map { rows[$0]! }
            let name =
                "model.layers.\(group.layer).mlp.experts.switch_mlp."
                + expertProjections[group.projection]!
            guard let weightParts = parts.map({ $0["weight"] }) as? [MLXArray] else {
                throw LoadError(message: "\(name) is missing an expert weight")
            }
            var value = stacked(weightParts)
            if parts[0]["scale"] != nil {
                guard let scaleParts = parts.map({ $0["scale"] }) as? [MLXArray] else {
                    throw LoadError(message: "\(name) is missing an expert scale")
                }
                let scales: MLXArray
                (value, scales) = repack(weight: value, scale: stacked(scaleParts))
                clean["\(name).scales"] = scales
            }
            clean["\(name).weight"] = value
        }
        return clean
    }
}
