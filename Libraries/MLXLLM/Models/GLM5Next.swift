//
//  GLM5Next.swift
//  mlx-swift-lm
//
//  GLM-5-Next (GLM-5.3-Flash) text model: mHC hyper-connections, Kimi Delta
//  Attention (KDA) linear layers, DeepSeek sparse attention (MLA latent KV
//  plus a pooled-key lightning indexer) and an MoE feed-forward.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm5_next/language.py
//  The multi-token-prediction layer is not loaded.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Helpers

private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let quantized as QuantizedMultiLinear: quantized(x)
    case let dense as MultiLinear: dense(x)
    default: fatalError("Expected MultiLinear or QuantizedMultiLinear, got \(type(of: module))")
    }
}

/// `post * x + comb^T @ residual`, the residual expansion of one mHC branch.
private let glm5NextHCExpand: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { x, residual, post, comb in
        let expanded =
            expandedDimensions(post, axis: -1) * expandedDimensions(x, axis: 2).asType(.float32)
        let mixed = matmul(comb.swappedAxes(-1, -2), residual.asType(.float32))
        return (expanded + mixed).asType(x.dtype)
    }

private func makeLimitedSwiGLU(limit: Float?) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    guard let limit else { return compiledSiluProduct }
    return compile(shapeless: true) { gate, up in
        silu(minimum(gate, limit)) * clip(up, min: -limit, max: limit)
    }
}

// MARK: - Hyper-connections (mHC)

final class GLM5NextHyperConnection: Module {
    let hcMult: Int
    let normEps: Float

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    private let sinkhorn: @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)

    init(_ config: GLM5NextConfiguration) {
        let hc = config.hcMult
        let eps = config.hcEps
        let iterations = config.hcSinkhornIters
        self.hcMult = hc
        self.normEps = config.rmsNormEps

        let mix = (2 + hc) * hc
        _fn.wrappedValue = MLXArray.zeros([mix, hc * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([3], dtype: .float32)

        self.sinkhorn = compile { mixes, scale, base in
            let mixes = mixes.asType(.float32)
            let scale = scale.asType(.float32)
            let base = base.asType(.float32)
            let pre = sigmoid(mixes[.ellipsis, ..<hc] * scale[0] + base[..<hc]) + eps
            let post =
                2 * sigmoid(mixes[.ellipsis, hc ..< (2 * hc)] * scale[1] + base[hc ..< (2 * hc)])
            let leading = Array(mixes.shape.dropLast())
            var comb =
                mixes[.ellipsis, (2 * hc)...].reshaped(leading + [hc, hc]) * scale[2]
                + base[(2 * hc)...].reshaped(hc, hc)
            comb = softmax(comb, axis: -1, precise: true) + eps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            for _ in 0 ..< max(iterations - 1, 0) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            }
            return (pre, post, comb)
        }
        super.init()
    }

    /// Collapse `[B, L, hc, D]` to `[B, L, D]`; returns `(collapsed, post, comb)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = x.asType(.float32)
        let z = MLXFast.rmsNorm(y.reshaped(B, L, H * D), weight: MLXArray.mlxNone, eps: normEps)
        let mixes = matmul(z, fn.T)
        let (pre, post, comb) = sinkhorn(mixes, scale, base)
        let collapsed = (expandedDimensions(pre, axis: -1) * y).sum(axis: 2).asType(x.dtype)
        return (collapsed, post, comb)
    }
}

// MARK: - Feed-forward

final class GLM5NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(_ config: GLM5NextConfiguration, intermediateSize: Int? = nil) {
        let hidden = intermediateSize ?? config.intermediateSize
        _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
        self.activation = makeLimitedSwiGLU(limit: config.swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x), upProj(x)))
    }
}

final class GLM5NextMoEGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    private let select: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray)

    init(_ config: GLM5NextConfiguration) {
        let experts = config.nRoutedExperts
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize], dtype: .float32)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([experts], dtype: .float32)

        let topK = config.numExpertsPerTok
        let nGroup = config.nGroup
        let topkGroup = config.topkGroup
        let scaling = config.routedScalingFactor
        let normalize = config.normTopkProb
        self.select = compile { logits, bias in
            let scores = sigmoid(logits.asType(.float32))
            var choice = scores + bias
            if nGroup > 1 {
                let shape = choice.shape
                let grouped = choice.reshaped(Array(shape.dropLast()) + [nGroup, -1])
                let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
                let dropCount = nGroup - topkGroup
                if dropCount > 0 {
                    let drop = argPartition(groupScores, kth: dropCount - 1, axis: -1)[
                        .ellipsis, ..<dropCount]
                    var keep = MLXArray.ones(groupScores.shape, dtype: .bool)
                    keep = putAlong(keep, drop, values: MLXArray(false), axis: -1)
                    let expanded = repeated(
                        expandedDimensions(keep, axis: -1), count: grouped.dim(-1), axis: -1
                    ).reshaped(shape)
                    choice = MLX.where(expanded, choice, MLXArray(-Float.infinity))
                }
            }
            let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, indices, axis: -1)
            if topK > 1, normalize {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            return (indices, weights * scaling)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        return select(logits, eScoreCorrectionBias)
    }
}

final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLM5NextMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP?

    init(_ config: GLM5NextConfiguration) {
        _gate.wrappedValue = GLM5NextMoEGate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activationProduct: makeLimitedSwiGLU(limit: config.swigluLimit))
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = GLM5NextMLP(
                config, intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = weightedExpertSum(switchMLP(x, indices), scores).asType(x.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Kimi Delta Attention (linear layers)

final class GLM5NextGatedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = (xf * xf).mean(axis: -1, keepDims: true)
        let normed = weight.asType(.float32) * (xf * rsqrt(variance + eps))
        return (normed * sigmoid(gate.asType(.float32))).asType(x.dtype)
    }
}

/// `exp(lower * sigmoid(exp(A_log) * (a + dt_bias)))`, the safe gate.
private let glm5NextSafeDecay: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { aLog, a, dtBias, lowerBound in
        exp(lowerBound * sigmoid(exp(aLog.asType(.float32)) * (a + dtBias)))
    }

/// `exp(-exp(A_log) * softplus(a + dt_bias))`.
private let glm5NextSoftplusDecay: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { aLog, a, dtBias in
        exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    }

final class GLM5NextForgetGate: Module {
    let numHeads: Int
    let headDim: Int
    let lowerBound: Float?

    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.linearNumHeads
        self.headDim = config.linearHeadDim
        self.lowerBound = config.linearLowerBound
        let projection = numHeads * headDim
        _fAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projection, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([numHeads], dtype: .float32)
        _dtBias.wrappedValue = MLXArray.zeros([projection], dtype: .float32)
        super.init()
    }

    /// Multiplicative per-channel decay `[B, L, H, Dk]` in float32.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let a = fBProj(fAProj(x)).reshaped(B, L, numHeads, headDim)
        let aLog = aLog.reshaped(numHeads, 1)
        let dtBias = dtBias.reshaped(numHeads, headDim)
        if let lowerBound {
            return glm5NextSafeDecay(aLog, a, dtBias, MLXArray(lowerBound))
        }
        return glm5NextSoftplusDecay(aLog, a, dtBias)
    }
}

final class GLM5NextLinearAttention: Module {
    let numHeads: Int
    let headDim: Int
    let projectionDim: Int
    let convKernelSize: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "forget_gate") var forgetGate: GLM5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: GLM5NextGatedRMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.linearNumHeads
        self.headDim = config.linearHeadDim
        self.projectionDim = numHeads * headDim
        self.convKernelSize = config.linearConvKernelDim
        let hidden = config.hiddenSize
        let convDim = 3 * projectionDim

        _qProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _forgetGate.wrappedValue = GLM5NextForgetGate(config)
        _bProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _oNorm.wrappedValue = GLM5NextGatedRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let mixed = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)

        let convState =
            cache?[0]
            ?? MLXArray.zeros([B, convKernelSize - 1, 3 * projectionDim], dtype: x.dtype)
        let convInput = concatenated([convState, mixed], axis: 1)
        let convOut = silu(conv1d(convInput))

        let parts = split(convOut, indices: [projectionDim, 2 * projectionDim], axis: -1)
        let shape = [B, L, numHeads, headDim]
        // x * rsqrt(sum(x^2) + 1e-6) in float32; q is also scaled by Dk^-1/2.
        func l2norm(_ x: MLXArray) -> MLXArray {
            x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
        }
        let q =
            (l2norm(parts[0].reshaped(shape).asType(.float32)) * Float(pow(Double(headDim), -0.5)))
            .asType(x.dtype)
        let k = l2norm(parts[1].reshaped(shape).asType(.float32)).asType(x.dtype)
        let v = parts[2].reshaped(shape)

        let decay = forgetGate(x)
        let beta = sigmoid(bProj(x))
        let (out, state) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: decay, beta: beta, state: cache?[1], useKernel: !training)

        if let cache {
            cache[0] = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])
            cache[1] = state
            cache.advance(L)
        }

        let gate = gBProj(gAProj(x)).reshaped(shape)
        return oProj(oNorm(out, gate: gate).reshaped(B, L, -1))
    }
}

// MARK: - DeepSeek sparse attention (MLA + lightning indexer)

/// A sparse-attention layer's cache: the MLA latent `[1, 1, T, kv_lora_rank]`,
/// the indexer's pooled keys `[1, P, D]`, and the keys and gate scores of the
/// pool window that is not complete yet.
final class GLM5NextAttentionCache: BaseKVCache {
    let kpool: Int
    var step = 256

    private(set) var latent: MLXArray?
    private(set) var pooledKeys: MLXArray?
    private var pendingKeys: MLXArray?
    private var pendingGates: MLXArray?

    init(kpool: Int) {
        precondition(kpool >= 1)
        self.kpool = kpool
        super.init()
    }

    override func innerState() -> [MLXArray] {
        [latent, pooledKeys, pendingKeys, pendingGates].compactMap { $0 }
    }

    /// Append `[1, 1, L, R]` and return the cached latent `[1, 1, offset, R]`.
    func appendLatent(_ values: MLXArray) -> MLXArray {
        let previous = offset
        let length = values.dim(2)
        if latent == nil || previous + length > latent!.dim(2) {
            let steps = (step + length - 1) / step
            let grown = MLXArray.zeros(
                [values.dim(0), values.dim(1), steps * step, values.dim(3)], dtype: values.dtype)
            if var current = latent {
                if previous % step != 0 {
                    current = current[.ellipsis, ..<previous, 0...]
                }
                latent = concatenated([current, grown], axis: 2)
            } else {
                latent = grown
            }
        }
        offset += length
        latent![.ellipsis, previous ..< offset, 0...] = values
        return latent![.ellipsis, ..<offset, 0...]
    }

    /// Add indexer keys and gate scores `[1, L, D]`; return the pool windows
    /// they complete, if any. The rest waits for later tokens.
    func takeCompleteWindows(keys: MLXArray, gates: MLXArray) -> (MLXArray, MLXArray)? {
        let allKeys = pendingKeys.map { concatenated([$0, keys], axis: 1) } ?? keys
        let allGates = pendingGates.map { concatenated([$0, gates], axis: 1) } ?? gates
        let total = allKeys.dim(1)
        let usable = (total / kpool) * kpool
        if usable < total {
            pendingKeys = contiguous(allKeys[0..., usable...])
            pendingGates = contiguous(allGates[0..., usable...])
        } else {
            pendingKeys = nil
            pendingGates = nil
        }
        guard usable > 0 else { return nil }
        return (allKeys[0..., ..<usable], allGates[0..., ..<usable])
    }

    func appendPooled(_ pooled: MLXArray) {
        pooledKeys = pooledKeys.map { concatenated([$0, pooled], axis: 1) } ?? pooled
    }

    /// `[latent, pooledKeys, pendingKeys, pendingGates]`; absent parts are empty arrays.
    override var state: [MLXArray] {
        get {
            let empty = MLXArray.zeros([0])
            return [
                offset > 0 ? latent.map { $0[.ellipsis, ..<offset, 0...] } ?? empty : empty,
                pooledKeys ?? empty, pendingKeys ?? empty, pendingGates ?? empty,
            ]
        }
        set {
            precondition(newValue.count == 4, "GLM5NextAttentionCache state has 4 arrays")
            func present(_ array: MLXArray) -> MLXArray? { array.size == 0 ? nil : array }
            latent = present(newValue[0])
            offset = latent?.dim(2) ?? 0
            pooledKeys = present(newValue[1])
            pendingKeys = present(newValue[2])
            pendingGates = present(newValue[3])
        }
    }

    override var metaState: [String] {
        get { ["\(kpool)"] }
        set {
            precondition(newValue == ["\(kpool)"], "GLM5NextAttentionCache pool width mismatch")
        }
    }

    override func copy() -> any KVCache {
        let new = GLM5NextAttentionCache(kpool: kpool)
        new.step = step
        new.state = state.map { $0[.ellipsis] }
        return new
    }
}

final class GLM5NextIndexer: Module {
    let numHeads: Int
    let headDim: Int
    let indexTopk: Int
    let kpool: Int
    let alwaysSelectTail: Bool
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var compressApe: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.indexTopk = config.indexTopk
        self.kpool = config.indexKpool
        self.alwaysSelectTail = config.indexKpoolAlwaysSelectTail
        self.softmaxScale = pow(Float(headDim), -0.5)
        _wqB.wrappedValue = Linear(config.qLoraRank, numHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, numHeads, bias: false)
        _compressApe.wrappedValue = MLXArray.zeros([kpool, headDim], dtype: .float32)
        _compressGate.wrappedValue = MLXArray.zeros(
            [headDim, config.hiddenSize], dtype: .float32)
        super.init()
    }

    /// Queries scored per chunk: at most 512, and at most 2^26 float32 score
    /// elements, so long contexts stay bounded.
    private func scoreChunk(poolCount: Int) -> Int {
        max(1, min(512, (1 << 26) / max(numHeads * poolCount, 1)))
    }

    /// Fold this chunk's keys into the pooled-key cache; returns all pooled keys.
    func updatePools(_ x: MLXArray, cache: GLM5NextAttentionCache) -> MLXArray? {
        let keys = kNorm(wk(x))
        let gates = matmul(x.asType(.float32), compressGate.asType(.float32).T).asType(keys.dtype)
        if let (readyKeys, readyGates) = cache.takeCompleteWindows(keys: keys, gates: gates) {
            cache.appendPooled(compress(keys: readyKeys, gates: readyGates))
        }
        return cache.pooledKeys
    }

    private func compress(keys: MLXArray, gates: MLXArray) -> MLXArray {
        let (B, S, D) = (keys.dim(0), keys.dim(1), keys.dim(2))
        let groupedKeys = keys.reshaped(B, S / kpool, kpool, D)
        let logits = gates.reshaped(B, S / kpool, kpool, D).asType(.float32) + compressApe
        let probs = softmax(logits, axis: 2, precise: true).asType(keys.dtype)
        return (probs * groupedKeys).sum(axis: 2)
    }

    /// Token indices `[B, L, index_topk + kpool - 1]` (`-1` = unused) attended by
    /// the last `L` queries of a `kvLength`-token sequence.
    func select(_ x: MLXArray, qResid: MLXArray, pooledKeys: MLXArray?, kvLength: Int)
        -> MLXArray
    {
        let (B, L) = (x.dim(0), x.dim(1))
        let start = kvLength - L
        let poolCount = pooledKeys?.dim(1) ?? 0
        let selectK = min(indexTopk / kpool, poolCount)
        let queryPositions = MLXArray(Int32(start) ..< Int32(kvLength))

        var topk: MLXArray
        if let pooledKeys, poolCount > 0 {
            let poolEnd =
                MLXArray(Int32(0) ..< Int32(poolCount)) * Int32(kpool) + Int32(kpool - 1)
            let candidates = broadcast(
                expandedDimensions(poolEnd, axes: [0, 1])
                    .<= expandedDimensions(queryPositions, axes: [0, 2]),
                to: [B, L, poolCount])

            var selected: MLXArray
            if selectK == poolCount {
                selected = broadcast(
                    MLXArray(Int32(0) ..< Int32(poolCount)).reshaped(1, 1, poolCount),
                    to: [B, L, poolCount])
            } else {
                let q = wqB(qResid).reshaped(B, L, numHeads, headDim).asType(.float32)
                let weights =
                    weightsProj(x).asType(.float32) * (pow(Float(numHeads), -0.5) * softmaxScale)
                let poolT = expandedDimensions(pooledKeys.asType(.float32), axis: 1)
                    .swappedAxes(-1, -2)
                var chunks = [MLXArray]()
                let chunk = scoreChunk(poolCount: poolCount)
                for c0 in stride(from: 0, to: L, by: chunk) {
                    let c1 = min(c0 + chunk, L)
                    var scores = maximum(matmul(q[0..., c0 ..< c1], poolT), 0)
                    scores = (scores * expandedDimensions(weights[0..., c0 ..< c1], axis: -1))
                        .sum(axis: 2)
                    scores = MLX.where(
                        candidates[0..., c0 ..< c1], scores,
                        MLXArray(-Float.greatestFiniteMagnitude))
                    chunks.append(
                        argPartition(-scores, kth: selectK - 1, axis: -1)[.ellipsis, ..<selectK]
                            .asType(.int32))
                }
                selected = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 1)
            }
            var valid = takeAlong(candidates, selected, axis: -1)

            // argPartition fixes the set, not its order: keep pools in order
            // with invalid slots last.
            let order = argSort(MLX.where(valid, selected, Int32(poolCount)), axis: -1)
            selected = takeAlong(selected, order, axis: -1)
            valid = takeAlong(valid, order, axis: -1)

            let offsets = MLXArray(Int32(0) ..< Int32(kpool))
            topk = (expandedDimensions(selected, axis: -1) * Int32(kpool) + offsets)
                .reshaped(B, L, selectK * kpool)
            let tokenValid = repeated(expandedDimensions(valid, axis: -1), count: kpool, axis: -1)
                .reshaped(B, L, selectK * kpool)
            topk = MLX.where(tokenValid, topk, Int32(-1))
        } else {
            topk = MLXArray.zeros([B, L, 0], dtype: .int32)
        }

        if topk.dim(-1) < indexTopk {
            let padding = MLXArray.zeros([B, L, indexTopk - topk.dim(-1)], dtype: .int32) - 1
            topk = concatenated([topk, padding], axis: -1)
        }
        topk = topk[.ellipsis, ..<indexTopk]

        if alwaysSelectTail && kpool > 1 {
            // Tokens after the last complete pool each query can see.
            let tailWidth = kpool - 1
            var tail = [Int32]()
            tail.reserveCapacity(L * tailWidth)
            for position in start ..< kvLength {
                let visible = position + 1
                let count = visible % kpool
                for j in 0 ..< tailWidth {
                    let index = visible - count + j
                    tail.append(j < count && index < kvLength ? Int32(index) : -1)
                }
            }
            let tailArray = broadcast(MLXArray(tail, [1, L, tailWidth]), to: [B, L, tailWidth])
            topk = concatenated([topk, tailArray], axis: -1)
        }
        return topk.asType(.int32)
    }
}

final class GLM5NextAttention: Module {
    let numHeads: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let indexTopk: Int
    let kpool: Int
    let scale: Float

    /// Queries per gathered sparse-attention block.
    static let sparseChunk = 64

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: GLM5NextIndexer?

    init(_ config: GLM5NextConfiguration, sharedIndexer: Bool) {
        self.numHeads = config.attentionHeads
        self.qHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.indexTopk = config.indexTopk
        self.kpool = config.indexKpool
        self.scale = pow(Float(qHeadDim), -0.5)
        let hidden = config.hiddenSize
        let kvLoraRank = config.kvLoraRank

        _qAProj.wrappedValue = Linear(hidden, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProj.wrappedValue = Linear(hidden, kvLoraRank, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, hidden, bias: config.attentionBias)
        if !sharedIndexer {
            _indexer.wrappedValue = GLM5NextIndexer(config)
        }
        super.init()
    }

    /// Returns the projected output and the token selection (`nil` while the
    /// context fits in `index_topk`), which a following shared-indexer layer reuses.
    func callAsFunction(
        _ x: MLXArray, cache: GLM5NextAttentionCache?, previousTopk: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let (B, L) = (x.dim(0), x.dim(1))
        let cache = cache ?? GLM5NextAttentionCache(kpool: kpool)

        let qResid = qALayerNorm(qAProj(x))
        let q = qBProj(qResid).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qLatent = callMultiLinear(embedQ, q)
        let latent = cache.appendLatent(expandedDimensions(kvALayerNorm(kvAProj(x)), axis: 1))
        let kvLength = cache.offset

        // While the whole context fits in `index_topk` the indexer would
        // select every causal token, so attention is dense.
        let dense = kvLength <= indexTopk
        var topk: MLXArray?
        if let indexer {
            let pooled = indexer.updatePools(x, cache: cache)
            if !dense {
                topk = indexer.select(x, qResid: qResid, pooledKeys: pooled, kvLength: kvLength)
            }
        } else if !dense {
            guard let previousTopk else {
                fatalError("GLM5Next shared-indexer layer has no previous top-k selection")
            }
            topk = previousTopk
        }

        let attended: MLXArray
        if let topk {
            attended = sparseAttention(qLatent, latent: latent, topk: topk)
        } else {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                L == 1 ? .none : .array(createCausalMask(n: L, offset: kvLength - L))
            attended = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: latent, values: latent, scale: scale, mask: mask)
        }
        let output = callMultiLinear(unembedOut, attended)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * vHeadDim)
        return (oProj(output), topk)
    }

    /// Absorbed-MLA attention over each query's selected latent rows.
    /// `qLatent: [B, H, L, R]`, `latent: [B, 1, T, R]`, `topk: [B, L, K]`.
    private func sparseAttention(_ qLatent: MLXArray, latent: MLXArray, topk: MLXArray)
        -> MLXArray
    {
        let (B, H, L, R) = (qLatent.dim(0), qLatent.dim(1), qLatent.dim(2), qLatent.dim(3))
        let T = latent.dim(2)
        let K = topk.dim(-1)
        let flatLatent = latent.reshaped(B * T, R)
        let batchOffsets = (MLXArray(Int32(0) ..< Int32(B)) * Int32(T)).reshaped(B, 1, 1)

        var outputs = [MLXArray]()
        for c0 in stride(from: 0, to: L, by: Self.sparseChunk) {
            let c1 = min(c0 + Self.sparseChunk, L)
            let rows = c1 - c0
            let indices = topk[0..., c0 ..< c1]
            let valid = logicalAnd(indices .>= 0, indices .< Int32(T))
            let safe = clip(indices, min: 0, max: max(T - 1, 0)) + batchOffsets
            let selected = take(flatLatent, safe.flattened(), axis: 0)
                .reshaped(B * rows, 1, K, R)
            // Prefill puts the heads on the query axis of one shared latent
            // head: one matmul per query instead of one per head.
            let queries = qLatent[0..., 0..., c0 ..< c1]
                .transposed(0, 2, 1, 3)
                .reshaped(L > 1 ? [B * rows, 1, H, R] : [B * rows, H, 1, R])
            let out = MLXFast.scaledDotProductAttention(
                queries: queries, keys: selected, values: selected, scale: scale,
                mask: .array(valid.reshaped(B * rows, 1, 1, K)))
            outputs.append(out.reshaped(B, rows, H, R).transposed(0, 2, 1, 3))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }
}

// MARK: - Decoder

final class GLM5NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: GLM5NextHyperConnection

    init(_ config: GLM5NextConfiguration, layerIdx: Int) {
        self.isLinear = config.layerTypes[layerIdx] == "linear_attention"
        if isLinear {
            _selfAttn.wrappedValue = GLM5NextLinearAttention(config)
        } else {
            _selfAttn.wrappedValue = GLM5NextAttention(
                config, sharedIndexer: config.indexerTypes[layerIdx] == "shared")
        }
        if config.mlpLayerTypes[layerIdx] == "sparse" {
            _mlp.wrappedValue = GLM5NextMoE(config)
        } else {
            _mlp.wrappedValue = GLM5NextMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnHC.wrappedValue = GLM5NextHyperConnection(config)
        _ffnHC.wrappedValue = GLM5NextHyperConnection(config)
        super.init()
    }

    /// `x: [B, L, hc_mult, D]`.
    func callAsFunction(_ x: MLXArray, cache: KVCache?, previousTopk: MLXArray?) -> (
        MLXArray, MLXArray?
    ) {
        let (collapsed, post, comb) = attnHC(x)
        let normed = inputLayerNorm(collapsed)
        let attention: MLXArray
        var topk = previousTopk
        if isLinear {
            attention = (selfAttn as! GLM5NextLinearAttention)(
                normed, cache: cache.map { $0 as! MambaCache })
        } else {
            (attention, topk) = (selfAttn as! GLM5NextAttention)(
                normed, cache: cache.map { $0 as! GLM5NextAttentionCache },
                previousTopk: previousTopk)
        }
        let h = glm5NextHCExpand(attention, x, post, comb)

        let (ffnCollapsed, ffnPost, ffnComb) = ffnHC(h)
        let ffnOut = (mlp as! UnaryLayer)(postAttentionLayerNorm(ffnCollapsed))
        return (glm5NextHCExpand(ffnOut, h, ffnPost, ffnComb), topk)
    }
}

// MARK: - Model

public final class GLM5NextModelInner: Module {
    let hcMult: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [GLM5NextDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM5NextConfiguration) {
        self.hcMult = config.hcMult
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            GLM5NextDecoderLayer(config, layerIdx: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = repeated(expandedDimensions(embedTokens(inputs), axis: 2), count: hcMult, axis: 2)
        var topk: MLXArray?
        for (index, layer) in layers.enumerated() {
            (h, topk) = layer(h, cache: cache?[index], previousTopk: topk)
        }
        return norm(h.mean(axis: 2))
    }
}

/// `glm5_next_text` checkpoints (`model.*`, `lm_head.*`).
public final class GLM5NextTextModel: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: GLM5NextConfiguration

    @ModuleInfo(key: "model") public var model: GLM5NextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _model.wrappedValue = GLM5NextModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Sparse-attention layers keep the whole context, so `maxKVSize` and KV
    /// quantization do not apply to them.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { layer -> KVCache in
            if layer.isLinear {
                return MambaCache()
            }
            return GLM5NextAttentionCache(kpool: configuration.indexKpool)
        }
    }

    public var excludedCheckpointNamespaces: [String] {
        GLM5NextWeights.excludedNamespaces(
            configuration, mtpPrefixes: ["mtp"], layerPrefixes: ["model.layers"])
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        filterLMHeadWeights(
            from: GLM5NextWeights.sanitize(weights, configuration: configuration),
            tiedWordEmbeddings: configuration.tieWordEmbeddings)
    }
}

/// `glm5_next` checkpoints: the text model under `language_model`, with
/// vision tensors alongside (not loaded).
public final class GLM5NextModel: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: GLM5NextConfiguration

    @ModuleInfo(key: "language_model") public var languageModel: GLM5NextTextModel

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _languageModel.wrappedValue = GLM5NextTextModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public var excludedCheckpointNamespaces: [String] {
        GLM5NextWeights.excludedNamespaces(
            configuration, mtpPrefixes: ["mtp", "language_model.mtp"],
            layerPrefixes: ["language_model.model.layers", "model.language_model.layers"])
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let text = languageModel.sanitize(weights: weights)
        return Dictionary(uniqueKeysWithValues: text.map { ("language_model." + $0.key, $0.value) })
    }
}

// MARK: - Weight sanitization

enum GLM5NextWeights {
    static let visionPrefixes = ["vision_model", "vision_tower", "visual", "model.visual"]

    /// Vision tensors and the multi-token-prediction layers, which are not read.
    static func excludedNamespaces(
        _ config: GLM5NextConfiguration, mtpPrefixes: [String], layerPrefixes: [String]
    ) -> [String] {
        let nextn = config.hiddenLayers ..< config.hiddenLayers + config.numNextnPredictLayers
        return visionPrefixes + mtpPrefixes
            + layerPrefixes.flatMap { prefix in nextn.map { "\(prefix).\($0)" } }
    }

    /// Map any supported checkpoint layout to text-model keys (`model.*`,
    /// `lm_head.*`) and drop vision tensors.
    static func textKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if visionPrefixes.contains(where: { key.hasPrefix($0 + ".") }) {
                continue
            }
            if key.hasPrefix("model.language_model.") {
                result["model." + key.dropFirst("model.language_model.".count)] = value
            } else if key.hasPrefix("language_model.") {
                result[String(key.dropFirst("language_model.".count))] = value
            } else {
                result[key] = value
            }
        }
        return result
    }

    static func sanitize(
        _ weights: [String: MLXArray], configuration config: GLM5NextConfiguration
    ) -> [String: MLXArray] {
        var weights = dropMTP(textKeys(weights), config: config)
        weights = renameParts(weights)
        stackExperts(&weights)
        splitKVB(&weights, config: config)
        dequantizeRouterGates(&weights, hiddenSize: config.hiddenSize)

        for (key, value) in weights where keepsFloat32(key) && value.dtype != .float32 {
            weights[key] = value.asType(.float32)
        }
        return weights
    }

    /// Drop the multi-token-prediction layers: `model.layers.<L + i>.*` and `mtp.*`.
    private static func dropMTP(_ weights: [String: MLXArray], config: GLM5NextConfiguration)
        -> [String: MLXArray]
    {
        weights.filter { key, _ in
            if key.hasPrefix("mtp.") { return false }
            guard key.hasPrefix("model.layers.") else { return true }
            let rest = key.dropFirst("model.layers.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else {
                return true
            }
            return index < config.hiddenLayers
        }
    }

    /// Hugging Face names to module names: `hc_attn_*`, top-level forget-gate
    /// parameters, and per-projection short convolutions.
    private static func renameParts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let forgetGateParts = [
            "A_log", "dt_bias",
            "f_a_proj.weight", "f_a_proj.scales", "f_a_proj.biases",
            "f_b_proj.weight", "f_b_proj.scales", "f_b_proj.biases",
        ]
        var result = [String: MLXArray]()
        var convParts = [String: [Character: MLXArray]]()
        for (key, value) in weights {
            var key =
                key
                .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
                .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")
            if let part = ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"]
                .first(where: { key.hasSuffix(".self_attn." + $0) })
            {
                let prefix = String(key.dropLast(part.count))
                convParts[prefix, default: [:]][part.first!] = value
                continue
            }
            if let part = forgetGateParts.first(where: { key.hasSuffix(".self_attn." + $0) }) {
                key = String(key.dropLast(part.count)) + "forget_gate." + part
            }
            result[key] = value
        }
        for (prefix, parts) in convParts {
            if let q = parts["q"], let k = parts["k"], let v = parts["v"] {
                result[prefix + "conv1d.weight"] = concatenated([q, k, v], axis: 0)
            } else {
                for (name, value) in parts {
                    result[prefix + "\(name)_conv1d.weight"] = value
                }
            }
        }
        for (key, value) in result
        where key.hasSuffix("conv1d.weight") && value.ndim == 3 && value.dim(-1) != 1 {
            result[key] = value.movedAxis(source: 2, destination: 1)
        }
        return result
    }

    /// `<prefix>.mlp.experts.<e>.<proj>.<part>` to `<prefix>.mlp.switch_mlp.<proj>.<part>`.
    private static func stackExperts(_ weights: inout [String: MLXArray]) {
        var groups = [String: [Int: String]]()
        for key in weights.keys {
            guard let range = key.range(of: ".mlp.experts.") else { continue }
            let rest = key[range.upperBound...]
            guard let dot = rest.firstIndex(of: "."), let expert = Int(rest[..<dot]) else {
                continue
            }
            let target =
                key[..<range.lowerBound] + ".mlp.switch_mlp." + rest[rest.index(after: dot)...]
            groups[String(target), default: [:]][expert] = key
        }
        for (target, members) in groups {
            let ordered = members.keys.sorted()
            guard ordered == Array(0 ..< ordered.count) else { continue }
            weights[target] = stacked(ordered.map { weights.removeValue(forKey: members[$0]!)! })
        }
    }

    /// `kv_b_proj` (`[H * (nope + v), kv_lora_rank]`) to `embed_q` / `unembed_out`.
    private static func splitKVB(_ weights: inout [String: MLXArray], config: GLM5NextConfiguration)
    {
        let heads = config.attentionHeads
        let nope = config.qkNopeHeadDim
        let rank = config.kvLoraRank
        for key in Array(weights.keys) where key.hasSuffix(".self_attn.kv_b_proj.weight") {
            let prefix = String(key.dropLast("kv_b_proj.weight".count))
            var value = weights.removeValue(forKey: key)!
            var quantization: (bits: Int, groupSize: Int)?
            if let scales = weights.removeValue(forKey: prefix + "kv_b_proj.scales") {
                let biases = weights.removeValue(forKey: prefix + "kv_b_proj.biases")
                let bits = value.dim(-1) * 32 / rank
                let groupSize = rank / scales.dim(-1)
                value = dequantized(
                    value, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
                quantization = (bits, groupSize)
            }
            value = value.reshaped(heads, nope + config.vHeadDim, rank)
            let wk = contiguous(value[0..., ..<nope, 0...].swappedAxes(-1, -2))
            let wv = contiguous(value[0..., nope..., 0...])
            for (name, matrix) in [("embed_q", wk), ("unembed_out", wv)] {
                if let quantization {
                    let (q, s, b) = MLX.quantized(
                        matrix, groupSize: quantization.groupSize, bits: quantization.bits)
                    weights[prefix + name + ".weight"] = q
                    weights[prefix + name + ".scales"] = s
                    weights[prefix + name + ".biases"] = b
                } else {
                    weights[prefix + name + ".weight"] = matrix
                }
            }
        }
    }

    /// The router has no quantized module; restore an affine-quantized gate to float32.
    private static func dequantizeRouterGates(_ weights: inout [String: MLXArray], hiddenSize: Int)
    {
        for key in Array(weights.keys) where key.hasSuffix(".mlp.gate.scales") {
            let prefix = String(key.dropLast("scales".count))
            guard let packed = weights[prefix + "weight"] else { continue }
            let scales = weights.removeValue(forKey: key)!
            let biases = weights.removeValue(forKey: prefix + "biases")
            weights[prefix + "weight"] = dequantized(
                packed, scales: scales, biases: biases,
                groupSize: hiddenSize / scales.dim(-1),
                bits: packed.dim(-1) * 32 / hiddenSize
            ).asType(.float32)
        }
    }

    private static func keepsFloat32(_ key: String) -> Bool {
        key.contains(".attn_hc.") || key.contains(".ffn_hc.")
            || key.hasSuffix("A_log") || key.hasSuffix("dt_bias")
            || key.hasSuffix("mlp.gate.weight") || key.hasSuffix("e_score_correction_bias")
    }
}

// MARK: - Configuration

public struct GLM5NextConfiguration: Codable, Sendable {
    struct LinearAttentionConfiguration: Codable, Sendable {
        var numHeads: Int?
        var headDim: Int?
        var shortConvKernelSize: Int?
        var gateLowerBound: Float?
        var safeGate: Bool?
        var kdaLayers: [Int]?

        enum CodingKeys: String, CodingKey {
            case numHeads = "num_heads"
            case headDim = "head_dim"
            case shortConvKernelSize = "short_conv_kernel_size"
            case gateLowerBound = "gate_lower_bound"
            case safeGate = "safe_gate"
            case kdaLayers = "kda_layers"
        }
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var numNextnPredictLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int
    var nRoutedExperts: Int
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var normTopkProb: Bool
    var rmsNormEps: Float
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var indexTopk: Int
    var indexHeadDim: Int
    var indexNHeads: Int
    var indexKpool: Int
    var indexKpoolAlwaysSelectTail: Bool
    var layerTypes: [String]
    var mlpLayerTypes: [String]
    var indexerTypes: [String]
    var swigluLimit: Float?
    var linearHeadDim: Int
    var linearNumHeads: Int
    var linearConvKernelDim: Int
    var linearLowerBound: Float?
    var hcMult: Int
    var hcEps: Float
    var hcSinkhornIters: Int
    var firstKDenseReplace: Int
    var scoringFunc: String
    var topkMethod: String
    var mlaUseNope: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case indexTopk = "index_topk"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexKpool = "index_kpool"
        case indexKpoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case indexerTypes = "indexer_types"
        case swigluLimit = "swiglu_limit"
        case linearHeadDim = "linear_head_dim"
        case linearNumHeads = "linear_num_heads"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearLowerBound = "linear_lower_bound"
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case firstKDenseReplace = "first_k_dense_replace"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case mlaUseNope = "mla_use_nope"
    }

    private enum ExtraKeys: String, CodingKey {
        case textConfig = "text_config"
        case linearAttnConfig = "linear_attn_config"
        case indexTopkPattern = "index_topk_pattern"
        case indexTopkFreq = "index_topk_freq"
        case indexSkipTopkOffset = "index_skip_topk_offset"
    }

    public init(from decoder: Decoder) throws {
        // glm5_next nests the text model under text_config; glm5_next_text is flat.
        let outer = try decoder.container(keyedBy: ExtraKeys.self)
        let nested = outer.contains(.textConfig)
        let c =
            nested
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)
        let extra =
            nested ? try outer.nestedContainer(keyedBy: ExtraKeys.self, forKey: .textConfig) : outer

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm5_next_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 154880
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 45
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 288
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1536
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 0
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 256
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 256
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexKpool = try c.decodeIfPresent(Int.self, forKey: .indexKpool) ?? 16
        indexKpoolAlwaysSelectTail =
            try c.decodeIfPresent(Bool.self, forKey: .indexKpoolAlwaysSelectTail) ?? true
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        linearHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearHeadDim) ?? 128
        linearNumHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumHeads) ?? 64
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        linearLowerBound =
            c.contains(.linearLowerBound)
            ? try c.decodeIfPresent(Float.self, forKey: .linearLowerBound) : -5.0
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 3
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? true

        let linear = try extra.decodeIfPresent(
            LinearAttentionConfiguration.self, forKey: .linearAttnConfig)
        if let linear {
            linearHeadDim = linear.headDim ?? linearHeadDim
            linearNumHeads = linear.numHeads ?? linearNumHeads
            linearConvKernelDim = linear.shortConvKernelSize ?? linearConvKernelDim
            linearLowerBound = linear.gateLowerBound ?? linearLowerBound
            if linear.safeGate ?? true, linearLowerBound == nil {
                linearLowerBound = -5.0
            }
        }

        let layers = hiddenLayers
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else if let kda = linear?.kdaLayers {
            let kdaSet = Set(kda)
            layerTypes = (0 ..< layers).map {
                kdaSet.contains($0) ? "linear_attention" : "deepseek_sparse_attention"
            }
        } else {
            layerTypes = (0 ..< layers).map {
                $0 % 4 != 3 ? "linear_attention" : "deepseek_sparse_attention"
            }
        }
        layerTypes = layerTypes.map { $0 == "full_attention" ? "deepseek_sparse_attention" : $0 }

        if let types = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = types
        } else {
            let dense = min(firstKDenseReplace, layers)
            mlpLayerTypes =
                Array(repeating: "dense", count: dense)
                + Array(repeating: "sparse", count: layers - dense)
        }

        if let types = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = types
        } else if let pattern = try? extra.decodeIfPresent(String.self, forKey: .indexTopkPattern) {
            indexerTypes = pattern.map { $0 == "S" ? "shared" : "full" }
        } else if let pattern = try? extra.decodeIfPresent(
            [String].self, forKey: .indexTopkPattern)
        {
            indexerTypes = pattern
        } else {
            let frequency = max(try extra.decodeIfPresent(Int.self, forKey: .indexTopkFreq) ?? 1, 1)
            let offset = try extra.decodeIfPresent(Int.self, forKey: .indexSkipTopkOffset) ?? 2
            indexerTypes = (0 ..< layers).map {
                max($0 - offset + 1, 0) % frequency == 0 ? "full" : "shared"
            }
        }
    }
}

extension GLM5NextConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "GLM5NextConfiguration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard attentionHeads == kvHeads else {
            throw fail("num_attention_heads must equal num_key_value_heads for GLM-5-Next")
        }
        guard layerTypes.count == hiddenLayers, mlpLayerTypes.count == hiddenLayers,
            indexerTypes.count == hiddenLayers
        else {
            throw fail("layer_types, mlp_layer_types and indexer_types need one entry per layer")
        }
        guard qkRopeHeadDim == 0, mlaUseNope else {
            throw fail("GLM-5-Next requires NoPE sparse attention (qk_rope_head_dim == 0)")
        }
        guard indexKpool >= 1, indexTopk >= indexKpool, indexTopk % indexKpool == 0 else {
            throw fail("index_topk must be a positive multiple of index_kpool")
        }
        guard scoringFunc == "sigmoid", topkMethod == "noaux_tc" else {
            throw fail("GLM-5-Next routing supports sigmoid noaux_tc only")
        }
        guard layerTypes.allSatisfy(["linear_attention", "deepseek_sparse_attention"].contains)
        else {
            throw fail("unsupported layer type in \(layerTypes)")
        }
        guard mlpLayerTypes.allSatisfy(["dense", "sparse"].contains) else {
            throw fail("unsupported mlp layer type in \(mlpLayerTypes)")
        }
        guard indexerTypes.allSatisfy(["full", "shared"].contains) else {
            throw fail("unsupported indexer type in \(indexerTypes)")
        }
        // A shared indexer reuses the selection of an earlier sparse layer.
        if let first = layerTypes.firstIndex(where: { $0 != "linear_attention" }),
            indexerTypes[first] == "shared"
        {
            throw fail("the first sparse-attention layer cannot use a shared indexer")
        }
        if mlpLayerTypes.contains("sparse") {
            guard nRoutedExperts >= 1, numExpertsPerTok >= 1, numExpertsPerTok <= nRoutedExperts
            else {
                throw fail("sparse MLP layers need 1 <= num_experts_per_tok <= n_routed_experts")
            }
        }
    }
}

// MARK: - LoRA

extension GLM5NextTextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

extension GLM5NextModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}

// MARK: - Chat conventions

extension GLM5NextTextModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}

extension GLM5NextModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}
