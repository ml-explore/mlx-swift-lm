// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Inkling2LoopConfiguration: Codable, Sendable {
    public let coreStart: Int
    public let coreEnd: Int
    public let passes: Int
    public let gateRank: Int

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
    }

    public var core: Range<Int> { coreStart ..< coreEnd }
}

public struct Inkling2TextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    public let unpaddedVocabularySize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let slidingAttentionHeads: Int
    public let slidingKVHeads: Int
    public let slidingHeadDim: Int
    public let slidingWindow: Int
    public let localLayerIds: [Int]
    public let relativeDim: Int
    public let relativeExtent: Int
    public let logScalingFloor: Int
    public let logScalingAlpha: Float
    public let rmsNormEps: Float
    public let qkClipThreshold: Float
    public let relativeClipThreshold: Float
    public let convolutionKernelSize: Int
    public let layersPerBlock: Int
    public let experts: Int
    public let expertsPerToken: Int
    public let intermediateSize: Int
    public let gateSoftcap: Float
    public let upSoftcap: Float
    public let routeScale: Float
    public let finalLogitSoftcap: Float
    public let loop: Inkling2LoopConfiguration?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingAttentionHeads = "swa_num_attention_heads"
        case slidingKVHeads = "swa_num_key_value_heads"
        case slidingHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case relativeDim = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkClipThreshold = "qk_l2_clip_threshold"
        case relativeClipThreshold = "rel_l2_clip_threshold"
        case convolutionKernelSize = "sconv_kernel_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case experts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case gateSoftcap = "mlp_gate_softcap"
        case upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case finalLogitSoftcap = "final_logit_softcapping"
        case loop = "loop_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        unpaddedVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize) ?? vocabularySize
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        slidingAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .slidingAttentionHeads) ?? attentionHeads
        slidingKVHeads = try c.decodeIfPresent(Int.self, forKey: .slidingKVHeads) ?? kvHeads
        slidingHeadDim = try c.decodeIfPresent(Int.self, forKey: .slidingHeadDim) ?? headDim
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        localLayerIds = try c.decodeIfPresent([Int].self, forKey: .localLayerIds) ?? []
        relativeDim = try c.decode(Int.self, forKey: .relativeDim)
        relativeExtent = try c.decode(Int.self, forKey: .relativeExtent)
        logScalingFloor = try c.decode(Int.self, forKey: .logScalingFloor)
        logScalingAlpha = try c.decode(Float.self, forKey: .logScalingAlpha)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        qkClipThreshold = try c.decode(Float.self, forKey: .qkClipThreshold)
        relativeClipThreshold = try c.decode(Float.self, forKey: .relativeClipThreshold)
        convolutionKernelSize = try c.decodeIfPresent(Int.self, forKey: .convolutionKernelSize) ?? 4
        layersPerBlock = try c.decode(Int.self, forKey: .layersPerBlock)
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        gateSoftcap = try c.decodeIfPresent(Float.self, forKey: .gateSoftcap) ?? 0
        upSoftcap = try c.decodeIfPresent(Float.self, forKey: .upSoftcap) ?? 0
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 1
        finalLogitSoftcap = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 0
        loop = try c.decodeIfPresent(Inkling2LoopConfiguration.self, forKey: .loop)
    }

    /// The layer of each layer visit: the prelude, every pass of the loop core, then the coda.
    public var layerVisits: [Int] {
        guard let loop else { return Array(0 ..< hiddenLayers) }
        return Array(0 ..< loop.coreStart)
            + (0 ..< loop.passes).flatMap { _ in loop.core }
            + Array(loop.coreEnd ..< hiddenLayers)
    }

    func isSliding(_ layer: Int) -> Bool { localLayerIds.contains(layer) }
}

public struct Inkling2VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let patchSize: Int
    public let mergeSize: Int
    public let positionGridSize: Int
    public let projectionHiddenSize: Int
    public let projectorBias: Bool
    public let mlpRatio: Float
    public let layerNormEps: Float
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case patchSize = "native_patch_size"
        case mergeSize = "adapter_merge_size"
        case positionGridSize = "posemb_grid_size"
        case projectionHiddenSize = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case mlpRatio = "mlp_ratio"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 23
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 20
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        positionGridSize = try c.decodeIfPresent(Int.self, forKey: .positionGridSize) ?? 32
        projectionHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectionHiddenSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
    }

    var headDim: Int { hiddenSize / attentionHeads }
    var intermediateSize: Int { Int(Float(hiddenSize) * mlpRatio) }
    /// The pixels on each side of one image token.
    var blockSize: Int { patchSize * mergeSize }
}

public struct Inkling2Configuration: Codable, Sendable {
    public let modelType: String
    public let textConfiguration: Inkling2TextConfiguration
    public let visionConfiguration: Inkling2VisionConfiguration?
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenId = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        textConfiguration = try c.decode(Inkling2TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decodeIfPresent(
            Inkling2VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200054
    }
}

// MARK: - Fused element-wise helpers

private func unweightedRMSNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
}

/// Clips the L2 norm of each head vector of `x * scale` to `threshold`, in float32.
private let l2Clip: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { args in
    let (x, scale, threshold) = (args[0], args[1], args[2])
    let states = x.asType(.float32) * scale
    let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
    let clipped = states * minimum(threshold / (norm + 1e-6), 1)
    return [clipped.asType(x.dtype)]
}

/// `l2Clip` followed by the log-length scaling of the full-attention layers. Each step
/// rounds to the input dtype, as the reference does.
private let l2ClipScaled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) {
    args in
    let (x, scale, threshold, tau) = (args[0], args[1], args[2], args[3])
    let states = x.asType(.float32) * scale
    let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
    let clipped = (states * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
    return [(clipped.asType(.float32) * tau).asType(x.dtype)]
}

/// `output * sigmoid(gate)` in float32, rounded to the gate's dtype.
private let sigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    output, gate in
    (output.asType(.float32) * sigmoid(gate.asType(.float32))).asType(gate.dtype)
}

/// The unweighted RMS norm of the values, in float32.
private let valueNorm: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { v in
    unweightedRMSNorm(v.asType(.float32)).asType(v.dtype)
}

/// `softcap(gate) * sigmoid(gate) * softcap(up)` in float32. The gate and up columns of
/// the fused projection alternate.
private func softcappedSwiGLU(gateSoftcap: Float, upSoftcap: Float)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: true) { gate, up in
        let dtype = gate.dtype
        let gate = gate.asType(.float32)
        let up = up.asType(.float32)
        let gateCapped = tanh(gate / gateSoftcap) * gateSoftcap
        let upCapped = tanh(up / upSoftcap) * upSoftcap
        return (gateCapped * sigmoid(gate) * upCapped).asType(dtype)
    }
}

/// The routed experts' outputs `[..., K, D]` weighted by `[..., K, 1]` and summed in float32.
private let weightedExperts: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs.asType(.float32) * weights).sum(axis: -2).asType(outputs.dtype)
}

/// `residual + x` in float32, rounded to `x`'s dtype.
private let residualAdd: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    residual, x in
    (residual + x.asType(.float32)).asType(x.dtype)
}

/// Hides an inference-only projection from `Module` reflection.
private final class QuantizedLinearBox: @unchecked Sendable {
    var value: QuantizedLinear?
}

/// Hides a constant `MLXArray` from `Module` parameter reflection.
private final class ConstantBox: @unchecked Sendable {
    let value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

// MARK: - Small modules

private final class Inkling2ScaledL2Clip: Module {
    @ParameterInfo var scale: MLXArray
    private let threshold: ConstantBox

    init(heads: Int, headDim: Int, threshold: Float) {
        self._scale.wrappedValue = MLXArray.ones([heads, headDim])
        self.threshold = ConstantBox(MLXArray(threshold))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tau: MLXArray? = nil) -> MLXArray {
        if let tau {
            return l2ClipScaled([x, scale, threshold.value, tau])[0]
        }
        return l2Clip([x, scale, threshold.value])[0]
    }
}

private final class Inkling2RelativeLogits: Module {
    @ParameterInfo var proj: MLXArray

    init(relativeDim: Int, extent: Int) {
        self._proj.wrappedValue = MLXArray.zeros([relativeDim, extent])
        super.init()
    }

    func callAsFunction(_ relative: MLXArray) -> MLXArray {
        matmul(relative, proj.asType(relative.dtype))
    }
}

/// The weight of a depthwise convolution, `[channels, kernel, 1]`.
private final class Inkling2DepthwiseWeight: Module {
    @ParameterInfo var weight: MLXArray

    init(channels: Int, kernelSize: Int) {
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
        super.init()
    }
}

/// The inputs of a causal convolution: the previous state, then `x`, with masked rows zeroed.
private func convolutionWindow(
    _ x: MLXArray, state: MLXArray?, mask: MLXArray?, kernelSize: Int
) -> MLXArray {
    var x = x
    if let mask {
        x = which(mask[.ellipsis, .newAxis], x, MLXArray(0, dtype: x.dtype))
    }
    let state =
        state ?? MLXArray.zeros([x.dim(0), kernelSize - 1, x.dim(-1)], dtype: x.dtype)
    return concatenated([state, x], axis: 1)
}

/// The last `kernelSize - 1` rows of each row's window, at its length when the batch is ragged.
private func convolutionState(_ window: MLXArray, lengths: MLXArray?, count: Int) -> MLXArray {
    if let lengths {
        let ends = clip(lengths, min: 0, max: window.dim(1) - count)
        let positions = (ends[0..., .newAxis] + MLXArray(0 ..< count))[.ellipsis, .newAxis]
        return takeAlong(window, positions.asType(.int32), axis: 1)
    }
    let state = window[0..., (window.dim(1) - count)..., 0...]
    // A slice of a prefill window would keep the whole window alive.
    return window.dim(1) > 2 * count ? contiguous(state) : state
}

/// One decode step of the static convolution, from the `[C, K, 1]` weight: the bfloat16
/// products accumulate in float32.
private func staticConvolutionStep(kernelSize: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let divisor = Float(kernelSize + 1).squareRoot()
    return compile { args in
        let (window, weight, x) = (args[0], args[1], args[2])
        let taps = weight.squeezed(axis: -1).transposed().asType(.float32)
        let out = (window.asType(.float32) * taps).sum(axis: 1, keepDims: true).asType(x.dtype)
        return [((out.asType(.float32) + x.asType(.float32)) / divisor).asType(x.dtype)]
    }
}

/// A depthwise causal convolution with a residual, `(conv(x) + x) / sqrt(K + 1)`.
private final class Inkling2ShortConvolution: Module {
    @ModuleInfo(key: "conv1d") var conv1d: Inkling2DepthwiseWeight

    let kernelSize: Int
    private let step: @Sendable ([MLXArray]) -> [MLXArray]

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self._conv1d.wrappedValue = Inkling2DepthwiseWeight(
            channels: channels, kernelSize: kernelSize)
        self.step = staticConvolutionStep(kernelSize: kernelSize)
        super.init()
    }

    /// Returns the output and the window, whose last rows are the next state.
    func callAsFunction(_ x: MLXArray, state: MLXArray?, mask: MLXArray?) -> (MLXArray, MLXArray) {
        let length = x.dim(1)
        if length > 1, usesKernel(mask: mask) {
            return inkling2RowConvolution(
                x, state: state, weight: conv1d.weight, outputType: x.dtype)
        }
        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: kernelSize)
        if length == 1 {
            return (step([window, conv1d.weight, x])[0], window)
        }
        // [K, C]: tap k multiplies the k-th oldest input.
        let taps = conv1d.weight.squeezed(axis: -1).transposed().asType(.float32)
        var acc = window[0..., 0 ..< length, 0...].asType(.float32) * taps[0]
        for k in 1 ..< kernelSize {
            acc = acc + window[0..., k ..< (k + length), 0...].asType(.float32) * taps[k]
        }
        let out = acc.asType(x.dtype).asType(.float32)
        let divisor = Float(kernelSize + 1).squareRoot()
        return (((out + x.asType(.float32)) / divisor).asType(x.dtype), window)
    }

    /// `base + self(x)` in float32, `self(x)` without a base, and the window.
    func residual(_ x: MLXArray, base: MLXArray?, state: MLXArray?, mask: MLXArray?)
        -> (MLXArray, MLXArray)
    {
        if x.dim(1) > 1, usesKernel(mask: mask) {
            return inkling2RowConvolution(
                x, state: state, weight: conv1d.weight, base: base, outputType: .float32)
        }
        let (out, window) = self(x, state: state, mask: mask)
        let residual = out.asType(.float32)
        return (base.map { $0.asType(.float32) + residual } ?? residual, window)
    }

    /// Whether a prefill chunk runs through ``inkling2RowConvolution``, which needs every row of
    /// the batch to be valid and returns the next state as the window.
    func usesKernel(mask: MLXArray?) -> Bool {
        kernelSize == 4 && mask == nil && Device.defaultDevice().deviceType == .gpu
    }
}

/// A convolution whose taps per head are a static part plus a projection of the layer input,
/// `(conv(x) + x) / sqrt(2K + 1)`.
private final class Inkling2DynamicShortConvolution: Module {
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProj: Linear

    let kernelSize: Int
    let heads: Int

    init(channels: Int, inputDims: Int, heads: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.heads = heads
        self._staticWeight.wrappedValue = MLXArray.zeros([kernelSize, channels])
        self._weightProj.wrappedValue = Linear(inputDims, heads * kernelSize, bias: false)
        super.init()
    }

    /// - Parameter dynamicWeights: `weightProj` of the layer input, `[B, L, heads * K]`.
    func callAsFunction(
        _ x: MLXArray, dynamicWeights: MLXArray, state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let k = kernelSize
        if l > 1, k == 4, mask == nil, Device.defaultDevice().deviceType == .gpu {
            return inkling2RowConvolution(
                x, state: state, weight: staticWeight, outputType: x.dtype,
                dynamicTaps: dynamicWeights)
        }
        let dynamicWeights = dynamicWeights.asType(.float32).reshaped(b, l, heads, 1, k)
        let staticWeights = staticWeight.asType(.float32).transposed().reshaped(heads, -1, k)
        let weights = staticWeights + dynamicWeights

        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: k)
        // Lag 0 is the current input.
        let windows = stacked(
            (0 ..< k).map { lag in window[0..., (k - 1 - lag) ..< (k - 1 - lag + l), 0...] },
            axis: -1
        ).reshaped(b, l, heads, -1, k)
        let out = (windows.asType(.float32) * weights).sum(axis: -1).reshaped(b, l, c)
        let divisor = Float(2 * k + 1).squareRoot()
        return (((out + x.asType(.float32)) / divisor).asType(x.dtype), window)
    }
}

/// Softmax attention of learned queries over the depth sources: the completed blocks and
/// the partial block, all normalized.
private final class Inkling2BlockAttentionResidual: Module {
    @ParameterInfo var query: MLXArray

    init(dimensions: Int, queries: Int = 1) {
        self._query.wrappedValue =
            queries == 1
            ? MLXArray.zeros([dimensions], dtype: .float32)
            : MLXArray.zeros([queries, dimensions], dtype: .float32)
        super.init()
    }

    /// float32 `[..., Q, D]` from blocks `[K, ..., D]` and the partial block `[..., D]`.
    func callAsFunction(_ blocks: MLXArray, _ partial: MLXArray) -> MLXArray {
        let d = partial.dim(-1)
        return inkling2DepthReadout(
            blocks: blocks, partial: partial, queries: query.reshaped(-1, d))
    }
}

/// The query, key, value and MLP message queries of each loop core layer, `[layers, 4, D]`.
private final class Inkling2MessageInput: Module {
    @ParameterInfo var query: MLXArray

    init(dimensions: Int, layers: Int) {
        self._query.wrappedValue = MLXArray.zeros([layers, 4, dimensions], dtype: .float32)
        super.init()
    }
}

/// The depth attention on the GPU, otherwise the reference computation.
private func inkling2DepthReadout(
    blocks: MLXArray, partial: MLXArray, queries: MLXArray, queriesPerGroup: Int? = nil
) -> MLXArray {
    let scale = 1 / Float(partial.dim(-1)).squareRoot()
    if Device.defaultDevice().deviceType == .gpu {
        return inkling2DepthAttention(
            blocks: blocks, partial: partial, queries: queries, scale: scale,
            queriesPerGroup: queriesPerGroup)
    }
    let sources = movedAxis(
        concatenated([blocks, partial.expandedDimensions(axis: 0)]), source: 0, destination: -2)
    let weights = softmax(matmul(sources, queries.transposed()) * scale, axis: -2, precise: true)
    return matmul(weights.swappedAxes(-1, -2), sources)
}

/// Merges a loop message into a sublayer input:
/// `(1 - tanh(forget(h))) * input + tanh(gain(h)) * message`, with `h` the rank-`r`
/// projection of the RMS-normalized input and message.
private final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetProj: Linear
    @ModuleInfo(key: "message_gate_proj") var gainProj: Linear

    init(dimensions: Int, rank: Int) {
        self._downProj.wrappedValue = Linear(2 * dimensions, rank, bias: false)
        self._forgetProj.wrappedValue = Linear(rank, dimensions, bias: false)
        self._gainProj.wrappedValue = Linear(rank, dimensions, bias: false)
        super.init()
    }

    /// float32 `input` and `message` to float32.
    func callAsFunction(_ input: MLXArray, _ message: MLXArray) -> MLXArray {
        inkling2MessageMerge(
            [input, message, downProj.weight, forgetProj.weight, gainProj.weight])[0]
    }
}

private let inkling2MessageMerge: @Sendable ([MLXArray]) -> [MLXArray] = compile {
    args in
    let (input, message, down, forget, gain) = (args[0], args[1], args[2], args[3], args[4])
    let states = concatenated(
        [unweightedRMSNorm(input), unweightedRMSNorm(message)], axis: -1)
    let hidden = matmul(states.asType(down.dtype), down.transposed()).asType(.float32)
    let forgetGate = tanh(matmul(hidden, forget.transposed().asType(.float32)))
    let messageGate = tanh(matmul(hidden, gain.transposed().asType(.float32)))
    return [(1 - forgetGate) * input + messageGate * message]
}

// MARK: - Attention

private final class Inkling2Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "r_proj") var relativeProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") var relativeLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") var queryClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") var relativeClip: Inkling2ScaledL2Clip

    /// Exactly one of the static and the dynamic convolution pairs is set.
    @ModuleInfo(key: "k_sconv") var keyConvolution: Module
    @ModuleInfo(key: "v_sconv") var valueConvolution: Module

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let isSliding: Bool
    let logScalingFloor: Float
    let logScalingAlpha: Float
    let qkThreshold: Float
    let relativeThreshold: Float
    private let keyClipScale: ConstantBox
    private let keyClipThreshold: ConstantBox

    /// Inference-only projections that run several projections of one input as one matmul:
    /// q, r, gate and sink, then k and v in sliding layers. The registered projections become
    /// views of the fused storage.
    private let fusedQuery = FusedQuantizedLinearProjectionCache()
    /// The q, r, gate and sink rows of a sliding layer's fused projection, for loop passes
    /// whose keys and values have their own inputs.
    private let queryRows = QuantizedLinearBox()
    /// k_proj and v_proj of a full layer, each with the convolution taps of its input.
    private let fusedKey = FusedQuantizedLinearProjectionCache()
    private let fusedValue = FusedQuantizedLinearProjectionCache()
    /// Set while ``prepareFusedProjections()`` installs the views, which must not invalidate the
    /// projections prepared before them.
    private var installingViews = false

    private static let projectionKeys: Set<String> = [
        "q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj", "k_sconv", "v_sconv",
    ]

    init(_ config: Inkling2TextConfiguration, sliding: Bool) {
        self.isSliding = sliding
        if sliding {
            heads = config.slidingAttentionHeads
            kvHeads = config.slidingKVHeads
            headDim = config.slidingHeadDim
            window = config.slidingWindow
        } else {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            window = 0
        }
        scale = 1 / Float(headDim).squareRoot()
        logScalingFloor = Float(config.logScalingFloor)
        logScalingAlpha = config.logScalingAlpha
        qkThreshold = config.qkClipThreshold
        relativeThreshold = config.relativeClipThreshold
        let extent = sliding ? config.slidingWindow : config.relativeExtent

        let dims = config.hiddenSize
        let queryDim = heads * headDim
        let kvDim = kvHeads * headDim
        _queryProj.wrappedValue = Linear(dims, queryDim, bias: false)
        _keyProj.wrappedValue = Linear(dims, kvDim, bias: false)
        _valueProj.wrappedValue = Linear(dims, kvDim, bias: false)
        _relativeProj.wrappedValue = Linear(dims, heads * config.relativeDim, bias: false)
        _outputProj.wrappedValue = Linear(queryDim, dims, bias: false)
        _gateProj.wrappedValue = Linear(dims, queryDim, bias: false)
        _sinkProj.wrappedValue = Linear(dims, heads, bias: false)
        _relativeLogits.wrappedValue = Inkling2RelativeLogits(
            relativeDim: config.relativeDim, extent: extent)
        _queryClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, headDim: headDim, threshold: config.qkClipThreshold)
        _relativeClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, headDim: config.relativeDim, threshold: config.relativeClipThreshold)
        keyClipScale = ConstantBox(MLXArray(Float(1)))
        keyClipThreshold = ConstantBox(MLXArray(config.qkClipThreshold))

        let k = config.convolutionKernelSize
        if sliding {
            _keyConvolution.wrappedValue = Inkling2ShortConvolution(channels: kvDim, kernelSize: k)
            _valueConvolution.wrappedValue = Inkling2ShortConvolution(
                channels: kvDim, kernelSize: k)
        } else {
            _keyConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvDim, inputDims: dims, heads: kvHeads, kernelSize: k)
            _valueConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvDim, inputDims: dims, heads: kvHeads, kernelSize: k)
        }
        super.init()
    }

    private var dynamicKeyConvolution: Inkling2DynamicShortConvolution? {
        keyConvolution as? Inkling2DynamicShortConvolution
    }

    private var dynamicValueConvolution: Inkling2DynamicShortConvolution? {
        valueConvolution as? Inkling2DynamicShortConvolution
    }

    // MARK: Fused projections

    func prepareFusedProjections() throws {
        installingViews = true
        defer { installingViews = false }
        if isSliding {
            let prepared = try fusedQuery.prepare(
                enabled: true,
                linears: [queryProj, relativeProj, gateProj, sinkProj, keyProj, valueProj]
            ) { views in
                try update(
                    modules: ModuleChildren(values: [
                        "q_proj": .value(views[0]), "r_proj": .value(views[1]),
                        "gate_proj": .value(views[2]), "sink_proj": .value(views[3]),
                        "k_proj": .value(views[4]), "v_proj": .value(views[5]),
                    ]), verify: [])
            }
            if prepared, let fused = fusedQuery.fused {
                let rows = ..<(fused.weight.dim(0) - 2 * kvHeads * headDim)
                queryRows.value = QuantizedLinear(
                    weight: fused.weight[rows], bias: nil, scales: fused.scales[rows],
                    biases: fused.biases?[rows], groupSize: fused.groupSize, bits: fused.bits,
                    mode: fused.mode)
            }
            return
        }
        try fusedQuery.prepare(
            enabled: true, linears: [queryProj, relativeProj, gateProj, sinkProj]
        ) { views in
            try update(
                modules: ModuleChildren(values: [
                    "q_proj": .value(views[0]), "r_proj": .value(views[1]),
                    "gate_proj": .value(views[2]), "sink_proj": .value(views[3]),
                ]), verify: [])
        }
        if let convolution = dynamicKeyConvolution {
            try fusedKey.prepare(enabled: true, linears: [keyProj, convolution.weightProj]) {
                views in
                try update(
                    modules: ModuleChildren(values: ["k_proj": .value(views[0])]), verify: [])
                try convolution.update(
                    modules: ModuleChildren(values: ["weight_proj": .value(views[1])]), verify: [])
            }
        }
        if let convolution = dynamicValueConvolution {
            try fusedValue.prepare(enabled: true, linears: [valueProj, convolution.weightProj]) {
                views in
                try update(
                    modules: ModuleChildren(values: ["v_proj": .value(views[0])]), verify: [])
                try convolution.update(
                    modules: ModuleChildren(values: ["weight_proj": .value(views[1])]), verify: [])
            }
        }
    }

    private func invalidateFusedProjections() {
        guard !installingViews else { return }
        fusedQuery.invalidate()
        fusedKey.invalidate()
        fusedValue.invalidate()
        queryRows.value = nil
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let replacesProjection = parameters.flattened().contains { key, _ in
            Self.projectionKeys.contains(String(key.prefix { $0 != "." }))
        }
        defer { if replacesProjection { invalidateFusedProjections() } }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { if Self.projectionKeys.contains(key) { invalidateFusedProjections() } }
        try super.updateModule(key: key, value)
    }

    private struct Projections {
        var queries: MLXArray
        var relative: MLXArray
        var gate: MLXArray
        var sinks: MLXArray
        var keys: MLXArray
        var values: MLXArray
        var keyTaps: MLXArray?
        var valueTaps: MLXArray?
    }

    /// The projections of the attention inputs, fused into as few matmuls as the inputs allow.
    private func project(queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray)
        -> Projections
    {
        let queryDim = heads * headDim
        let relativeDim = relativeProj.weight.dim(0)
        let kvDim = kvHeads * headDim
        func split(_ x: MLXArray, _ sizes: [Int]) -> [MLXArray] {
            var start = 0
            return sizes.map { size in
                defer { start += size }
                return x[.ellipsis, start ..< (start + size)]
            }
        }
        let sharedInput = queryInput === keyInput && keyInput === valueInput

        var result: Projections
        if isSliding, sharedInput, let fused = fusedQuery.fused {
            let parts = split(
                fused(queryInput), [queryDim, relativeDim, queryDim, heads, kvDim, kvDim])
            return Projections(
                queries: parts[0], relative: parts[1], gate: parts[2], sinks: parts[3],
                keys: parts[4], values: parts[5])
        } else if let fused = isSliding ? queryRows.value : fusedQuery.fused {
            let parts = split(fused(queryInput), [queryDim, relativeDim, queryDim, heads])
            result = Projections(
                queries: parts[0], relative: parts[1], gate: parts[2], sinks: parts[3],
                keys: keyProj(keyInput), values: valueProj(valueInput))
        } else {
            result = Projections(
                queries: queryProj(queryInput), relative: relativeProj(queryInput),
                gate: gateProj(queryInput), sinks: sinkProj(queryInput),
                keys: keyProj(keyInput), values: valueProj(valueInput))
        }

        if let convolution = dynamicKeyConvolution {
            if let fused = fusedKey.fused {
                let parts = split(fused(keyInput), [kvDim, convolution.weightProj.weight.dim(0)])
                (result.keys, result.keyTaps) = (parts[0], parts[1])
            } else {
                result.keyTaps = convolution.weightProj(keyInput)
            }
        }
        if let convolution = dynamicValueConvolution {
            if let fused = fusedValue.fused {
                let parts = split(fused(valueInput), [kvDim, convolution.weightProj.weight.dim(0)])
                (result.values, result.valueTaps) = (parts[0], parts[1])
            } else {
                result.valueTaps = convolution.weightProj(valueInput)
            }
        }
        return result
    }

    private func convolve(
        _ convolution: Module, _ x: MLXArray, taps: MLXArray?, state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        if let convolution = convolution as? Inkling2ShortConvolution {
            return convolution(x, state: state, mask: mask)
        }
        let convolution = convolution as! Inkling2DynamicShortConvolution
        return convolution(x, dynamicWeights: taps!, state: state, mask: mask)
    }

    /// - Parameters:
    ///   - queryInput: the normalized input of the queries, relative states, gate and sinks.
    ///   - convolutions: the key and value convolution states, updated in place.
    func callAsFunction(
        queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray,
        cache: KVCache?, convolutions: [MLXArray?], convolutionMask: MLXArray?,
        masks: Inkling2DecodeMasks
    ) -> (MLXArray, keyWindow: MLXArray, valueWindow: MLXArray) {
        let (b, l) = (queryInput.dim(0), queryInput.dim(1))
        let offset = cache?.offset ?? 0

        let projections = project(
            queryInput: queryInput, keyInput: keyInput, valueInput: valueInput)
        let (keys, keyWindow) = convolve(
            keyConvolution, projections.keys, taps: projections.keyTaps, state: convolutions[0],
            mask: convolutionMask)
        let (values, valueWindow) = convolve(
            valueConvolution, projections.values, taps: projections.valueTaps,
            state: convolutions[1], mask: convolutionMask)

        var tau: MLXArray?
        if !isSliding {
            let positions = MLXArray(Int32(offset) ..< Int32(offset + l)).asType(.float32)
            let t = 1 + logScalingAlpha * log(maximum((positions + 1) / logScalingFloor, 1))
            tau = t.reshaped(1, l, 1, 1)
        }

        var queries: MLXArray
        var k: MLXArray
        var v: MLXArray
        let relative: MLXArray
        if Device.defaultDevice().deviceType == .gpu, headDim % 32 == 0,
            relativeLogits.proj.dim(0) <= 32
        {
            let out = inkling2PrefillPrologue(
                q: projections.queries, r: projections.relative, k: keys, v: values,
                qScale: queryClip.scale, rScale: relativeClip.scale, tau: tau?.reshaped(-1),
                heads: heads, kvHeads: kvHeads, headDim: headDim,
                relativeDim: relativeLogits.proj.dim(0), qkThreshold: qkThreshold,
                relativeThreshold: relativeThreshold)
            (queries, relative, k, v) = (out[0], out[1], out[2], out[3])
        } else {
            queries = queryClip(projections.queries.reshaped(b, l, heads, headDim), tau: tau)
            k =
                l2Clip([
                    keys.reshaped(b, l, kvHeads, headDim), keyClipScale.value,
                    keyClipThreshold.value,
                ])[0]
            v = valueNorm(values.reshaped(b, l, kvHeads, headDim))
            relative = relativeClip(projections.relative.reshaped(b, l, heads, -1), tau: tau)
        }
        let positionBias = relativeLogits(relative)
        let gate = projections.gate
        let sinks = projections.sinks

        queries = queries.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        let output: MLXArray
        if let quantized = cache as? QuantizedKVCacheProtocol {
            let (quantizedKeys, quantizedValues) = quantized.updateQuantized(keys: k, values: v)
            let s = quantizedKeys.0.dim(-2)
            func grouped(_ x: MLXArray?) -> MLXArray? { x?.expandedDimensions(axis: 2) }
            output = referenceAttention(
                queries: queries, positionBias: positionBias, sinks: sinks,
                distance: masks.distance(keys: s, queries: l, ringIndex: nil),
                scores: { q in
                    quantizedMM(
                        q, grouped(quantizedKeys.0)!, scales: grouped(quantizedKeys.1)!,
                        biases: grouped(quantizedKeys.2), transpose: true,
                        groupSize: quantized.groupSize, bits: quantized.bits,
                        mode: quantized.mode
                    ).asType(.float32)
                },
                values: { weights in
                    quantizedMM(
                        weights, grouped(quantizedValues.0)!, scales: grouped(quantizedValues.1)!,
                        biases: grouped(quantizedValues.2), transpose: false,
                        groupSize: quantized.groupSize, bits: quantized.bits,
                        mode: quantized.mode)
                })
        } else {
            precondition(
                !(cache is KVCacheAttentionProtocol),
                "Inkling 2 attention needs full-precision or affine-quantized KV caches")
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
            output = attend(
                queries: queries, keys: k, values: v, positionBias: positionBias, sinks: sinks,
                cache: cache, end: offset + l, masks: masks)
        }
        let gated = sigmoidGate(output.reshaped(b, l, -1), gate)
        return (outputProj(gated), keyWindow, valueWindow)
    }

    /// Whether ``decode(queryInput:keyInput:valueInput:cache:states:)`` handles this cache.
    static func supportsDecode(_ cache: KVCache) -> Bool {
        !(cache is QuantizedKVCacheProtocol) && !(cache is KVCacheAttentionProtocol)
    }

    /// One token per row through the decode kernels.
    ///
    /// - Parameter states: the key and value convolution states, `[B, 3, kvHeads * headDim]`.
    /// - Returns: the output `[B, 1, D]` and the next key and value convolution states.
    func decode(
        queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray, cache: KVCache,
        states: (MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray, MLXArray) {
        let rows = queryInput.dim(0)
        let offset = cache.offset
        let p = project(queryInput: queryInput, keyInput: keyInput, valueInput: valueInput)

        var taps: (k: MLXArray, v: MLXArray)?
        let weights: (k: MLXArray, v: MLXArray)
        var tau: MLXArray?
        if let keyConvolution = dynamicKeyConvolution,
            let valueConvolution = dynamicValueConvolution
        {
            taps = (p.keyTaps!, p.valueTaps!)
            weights = (keyConvolution.staticWeight, valueConvolution.staticWeight)
            tau = MLXArray([
                1 + logScalingAlpha * log(max(Float(offset + 1) / logScalingFloor, 1))
            ])
        } else {
            let keyConvolution = keyConvolution as! Inkling2ShortConvolution
            let valueConvolution = valueConvolution as! Inkling2ShortConvolution
            weights = (keyConvolution.conv1d.weight, valueConvolution.conv1d.weight)
        }
        let out = inkling2AttentionPrologue(
            q: p.queries, r: p.relative, k: p.keys, v: p.values, taps: taps, weights: weights,
            states: states, qScale: queryClip.scale, rScale: relativeClip.scale, tau: tau,
            heads: heads, kvHeads: kvHeads, headDim: headDim,
            relativeDim: relativeLogits.proj.dim(0), qkThreshold: qkThreshold,
            relativeThreshold: relativeThreshold)
        let (keys, values) = cache.update(keys: out[2], values: out[3])
        let attention = inkling2DecodeAttention(
            q: out[0], k: keys, v: values, relative: out[1], table: relativeLogits.proj,
            sinks: p.sinks, gate: p.gate, scale: scale,
            ringIndex: (cache as? RotatingKVCache)?.ringWriteIndex, window: window)
        return (outputProj(attention.reshaped(rows, 1, -1)), out[4], out[5])
    }

    /// Attention with a relative position bias and a sink, `[B, L, H, D]`.
    private func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, cache: KVCache?, end: Int, masks: Inkling2DecodeMasks
    ) -> MLXArray {
        let (b, l) = (queries.dim(0), queries.dim(2))
        let s = keys.dim(2)
        let onGPU = Device.defaultDevice().deviceType == .gpu
        let bf16 = queries.dtype == .bfloat16 && keys.dtype == .bfloat16
        if l > 1 && onGPU && bf16 {
            return inkling2RelativeAttention(
                q: queries, k: keys, v: values, rel: positionBias, sinks: sinks, scale: scale,
                window: window, keyStart: MLXArray.full([b], values: MLXArray(Int32(s - end))))
        }

        let ringIndex = (cache as? RotatingKVCache)?.ringWriteIndex
        let distance = masks.distance(keys: s, queries: l, ringIndex: ringIndex)
        return referenceAttention(
            queries: queries, positionBias: positionBias, sinks: sinks, distance: distance,
            scores: { q in
                matmul(
                    q.asType(.float32),
                    keys.asType(.float32).expandedDimensions(axis: 2).swappedAxes(-1, -2))
            },
            values: { weights in matmul(weights, values.expandedDimensions(axis: 2)) })
    }

    /// The materialized attention, for prefill off the GPU, batched decode and quantized caches.
    ///
    /// - Parameters:
    ///   - scores: float32 logits `[B, Hkv, G, L, S]` of the grouped queries `[B, Hkv, G, L, D]`.
    ///   - values: the attention weights `[B, Hkv, G, L, S]` applied to the values.
    private func referenceAttention(
        queries: MLXArray, positionBias: MLXArray, sinks: MLXArray, distance: MLXArray,
        scores: (MLXArray) -> MLXArray, values: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let (b, l, d) = (queries.dim(0), queries.dim(2), queries.dim(3))
        let s = distance.dim(-1)
        let extent = positionBias.dim(-1)
        var visible = distance .>= 0
        if window > 0 {
            visible = logicalAnd(visible, distance .< window)
        }
        // [B, H, L, S] gather of the bias at each distance.
        let index = broadcast(
            clip(distance, min: 0, max: extent - 1), to: [b, heads, l, s])
        var bias = takeAlong(positionBias.transposed(0, 2, 1, 3), index, axis: -1)
        bias = which(distance .< extent, bias.asType(.float32), MLXArray(Float(0)))
        let mask = which(visible, bias, MLXArray(-Float.infinity)).reshaped(
            b, kvHeads, -1, l, s)

        let logits = scores(queries.reshaped(b, kvHeads, -1, l, d)) * scale + mask
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(b, kvHeads, -1, l, 1)
        let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[
            .ellipsis, ..<s]
        let output = values(weights.asType(queries.dtype))
        return output.reshaped(b, heads, l, d).transposed(0, 2, 1, 3)
    }
}

/// The key distances of one forward pass, shared by the layers whose caches agree.
private final class Inkling2DecodeMasks {
    private var distances = [String: MLXArray]()

    /// int32 `[L, S]` (`[S]` for one query): each query's distance to each key.
    func distance(keys: Int, queries: Int, ringIndex: Int?) -> MLXArray {
        let key = "\(keys):\(queries):\(ringIndex ?? -1)"
        if let cached = distances[key] { return cached }
        let k = MLXArray.arange(keys, dtype: .int32)
        let result: MLXArray
        if let ringIndex {
            // A decode step wrote the newest key at ringIndex - 1.
            result = (MLXArray(Int32(ringIndex - 1 + keys)) - k) % Int32(keys)
        } else if queries == 1 {
            result = MLXArray(Int32(keys - 1)) - k
        } else {
            let q = MLXArray.arange(queries, dtype: .int32) + Int32(keys - queries)
            result = q[0..., .newAxis] - k[.newAxis, 0...]
        }
        distances[key] = result
        return result
    }
}

// MARK: - Mixture of experts

/// One projection of all experts, quantized in the checkpoint's own format.
private final class Inkling2ExpertProjection: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray
    @ParameterInfo var biases: MLXArray?
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?

    let format: Inkling2ExpertFormat

    init(experts: Int, inputDims: Int, outputDims: Int, format: Inkling2ExpertFormat) {
        self.format = format
        let packed = inputDims * format.bits / 32
        let groups = inputDims / format.groupSize
        self._weight.wrappedValue = MLXArray.zeros([experts, outputDims, packed], dtype: .uint32)
        let scaleType: DType = format.mode == .affine ? .bfloat16 : .uint8
        self._scales.wrappedValue = MLXArray.zeros(
            [experts, outputDims, groups], dtype: scaleType)
        if format.mode == .affine {
            self._biases.wrappedValue = MLXArray.zeros(
                [experts, outputDims, groups], dtype: .bfloat16)
        }
        if format.hasGlobalScale {
            self._globalScale.wrappedValue = MLXArray.ones([experts], dtype: .float32)
        }
        super.init()
        freeze()
    }

    var tensors: Inkling2ExpertTensors {
        .init(
            weight: weight, scales: scales, biases: biases, globalScale: globalScale,
            format: format)
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices, transpose: true,
            groupSize: format.groupSize, bits: format.bits, mode: format.mode,
            globalScale: globalScale, sortedIndices: sorted)
    }
}

/// The quantization of an expert projection, read from the checkpoint's tensor shapes.
struct Inkling2ExpertFormat: Equatable, Sendable {
    let mode: QuantizationMode
    let bits: Int
    let groupSize: Int
    let hasGlobalScale: Bool

    static let nvfp4 = Inkling2ExpertFormat(
        mode: .nvfp4, bits: 4, groupSize: 16, hasGlobalScale: true)

    init(mode: QuantizationMode, bits: Int, groupSize: Int, hasGlobalScale: Bool) {
        self.mode = mode
        self.bits = bits
        self.groupSize = groupSize
        self.hasGlobalScale = hasGlobalScale
    }

    /// The format of the tensors at `path`, or `nil` when they are not quantized.
    init?(path: String, weights: [String: MLXArray], inputDims: Int) {
        guard let weight = weights["\(path).weight"], let scales = weights["\(path).scales"],
            weight.dtype == .uint32
        else { return nil }
        let bits = weight.dim(-1) * 32 / inputDims
        let groupSize = inputDims / scales.dim(-1)
        if weights["\(path).biases"] != nil {
            self.init(mode: .affine, bits: bits, groupSize: groupSize, hasGlobalScale: false)
            return
        }
        let mode: QuantizationMode
        switch (bits, groupSize) {
        case (4, 16): mode = .nvfp4
        case (4, 32): mode = .mxfp4
        case (8, 32): mode = .mxfp8
        default: return nil
        }
        self.init(
            mode: mode, bits: bits, groupSize: groupSize,
            hasGlobalScale: weights["\(path).global_scale"] != nil)
    }
}

private final class Inkling2Experts: Module {
    /// The rows alternate between gate and up, as in the checkpoint.
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Inkling2ExpertProjection
    @ModuleInfo(key: "down_proj") var downProj: Inkling2ExpertProjection

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray
    private let softcaps: ConstantBox

    init(
        _ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat,
        down: Inkling2ExpertFormat
    ) {
        let (dims, hidden) = (config.hiddenSize, config.intermediateSize)
        _gateUpProj.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: dims, outputDims: 2 * hidden, format: gateUp)
        _downProj.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: hidden, outputDims: dims, format: down)
        activation = softcappedSwiGLU(gateSoftcap: config.gateSoftcap, upSoftcap: config.upSoftcap)
        softcaps = ConstantBox(MLXArray([config.gateSoftcap, config.upSoftcap]))
        super.init()
    }

    /// `[..., K, D]` outputs of the `[..., K]` experts of each row of `x`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let (out, order) = outputs(x, indices)
        guard let order else { return out.reshaped(indices.shape + [out.dim(-1)]) }
        return scatterUnsort(x: out, invOrder: order, shape: indices.shape)
    }

    /// The expert outputs `[N * K, D]` of the rows of `x` and, when they are sorted by expert,
    /// the row of each output: output `i` of the flattened `indices` is at row `order[i]`.
    func outputs(_ x: MLXArray, _ indices: MLXArray) -> (MLXArray, order: MLXArray?) {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let sort = indices.size >= 64
        var idx = indices
        var inverseOrder: MLXArray?
        if sort {
            let sorted = gatherSort(x: x, indices: indices)
            (x, idx, inverseOrder) = (sorted.0, sorted.1, sorted.2)
        }
        let gateUp = gateUpProj(x, idx, sorted: sort)
        let activated = activation(
            gateUp[.ellipsis, .stride(from: 0, by: 2)], gateUp[.ellipsis, .stride(from: 1, by: 2)])
        let out = downProj(activated, idx, sorted: sort)
        return (out.reshaped(-1, out.dim(-1)), inverseOrder)
    }

    /// The layer output `residual + conv(sum(weights * experts(x)))` of one token per row and
    /// the next convolution state. Few rows run through the fused expert kernels.
    func decode(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, residual: MLXArray, state: MLXArray,
        convolutionWeight: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (gateUp, down) = (gateUpProj.tensors, downProj.tensors)
        let (dims, hidden) = (down.weight.dim(1), gateUp.weight.dim(1) / 2)
        if indices.size < 64, indices.dim(-1) <= 32, hidden % 4 == 0, dims % 2 == 0,
            gateUp.supportsFusedDecode(inputDims: dims),
            down.supportsFusedDecode(inputDims: hidden)
        {
            let activations = inkling2ExpertGateUp(
                x, experts: gateUp, indices: indices, softcaps: softcaps.value)
            return inkling2ExpertDown(
                activations, experts: down, indices: indices, weights: weights,
                residual: residual, state: state, convolutionWeight: convolutionWeight)
        }
        return inkling2ExpertEpilogue(
            outputs: self(x, indices), weights: weights, residual: residual, state: state,
            convolutionWeight: convolutionWeight)
    }
}

private final class Inkling2Router: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray

    let topK: Int
    let routeScale: Float

    init(_ config: Inkling2TextConfiguration) {
        topK = config.expertsPerToken
        routeScale = config.routeScale
        _weight.wrappedValue = MLXArray.zeros([config.experts, config.hiddenSize], dtype: .float32)
        _bias.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = matmul(x.asType(.float32), weight.transposed())
        if Device.defaultDevice().deviceType == .gpu {
            return inkling2ExpertRouting(
                logits: logits, bias: bias, topK: topK, scoreEps: 1e-20, routeScale: routeScale)
        }
        let scores = sigmoid(logits)
        let biased = scores + bias
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices.asType(.uint32), weights * routeScale)
    }
}

private final class Inkling2SparseMoE: Module {
    @ModuleInfo(key: "gate") var router: Inkling2Router
    @ModuleInfo var experts: Inkling2Experts

    init(_ config: Inkling2TextConfiguration) {
        _router.wrappedValue = Inkling2Router(config)
        _experts.wrappedValue = Inkling2Experts(config, gateUp: .nvfp4, down: .nvfp4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = router(x)
        return weightedExperts(experts(x, indices), weights[.ellipsis, .newAxis])
    }

    /// The layer output `residual + convolution(self(x))` of a prefill chunk and the window of
    /// the convolution. Without a mask, the expert outputs go straight into the convolution.
    func prefill(
        _ x: MLXArray, residual: MLXArray, convolution: Inkling2ShortConvolution,
        state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        guard convolution.usesKernel(mask: mask) else {
            let (output, window) = convolution(self(x), state: state, mask: mask)
            return (residualAdd(residual, output), window)
        }
        let (indices, weights) = router(x)
        let (outputs, order) = experts.outputs(x, indices)
        return inkling2RowConvolution(
            outputs, state: state, weight: convolution.conv1d.weight, base: residual,
            outputType: x.dtype, experts: (weights, order))
    }

    /// ``Inkling2Experts/decode(_:indices:weights:residual:state:convolutionWeight:)``, routed
    /// from the float32 copy of the input.
    func decode(
        _ x: MLXArray, routerInput: MLXArray, residual: MLXArray, state: MLXArray,
        convolutionWeight: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (indices, weights) = router(routerInput)
        return experts.decode(
            x, indices: indices, weights: weights, residual: residual, state: state,
            convolutionWeight: convolutionWeight)
    }
}

// MARK: - Decoder layer

private final class Inkling2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention
    @ModuleInfo var mlp: Inkling2SparseMoE
    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockAttentionResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConvolution:
        Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConvolution: Inkling2ShortConvolution

    // Sliding layers normalize one attention input, full layers three.
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var queryInputNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var keyInputNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var valueInputNorm: RMSNorm?

    // Loop core layers merge a message into each sublayer input in later passes.
    @ModuleInfo(key: "q_message_gate") var queryMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var keyMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var valueMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpMessageGate: Inkling2MessageGate?

    let isSliding: Bool
    let isBlockBoundary: Bool
    let convolutionStateLength: Int

    init(_ config: Inkling2TextConfiguration, layer: Int) {
        let dims = config.hiddenSize
        let eps = config.rmsNormEps
        isSliding = config.isSliding(layer)
        isBlockBoundary = layer % config.layersPerBlock == 0
        convolutionStateLength = config.convolutionKernelSize - 1

        _attention.wrappedValue = Inkling2Attention(config, sliding: isSliding)
        _mlp.wrappedValue = Inkling2SparseMoE(config)
        if layer > 0 {
            _attentionInput.wrappedValue = Inkling2BlockAttentionResidual(
                dimensions: dims, queries: isSliding ? 1 : 3)
        }
        _mlpInput.wrappedValue = Inkling2BlockAttentionResidual(dimensions: dims)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        _postAttentionConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: dims, kernelSize: config.convolutionKernelSize)
        _postMLPConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: dims, kernelSize: config.convolutionKernelSize)
        if isSliding {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        } else {
            _queryInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
            _keyInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
            _valueInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        }
        if let loop = config.loop, loop.core.contains(layer) {
            _queryMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _keyMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _valueMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _mlpMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
        }
        super.init()
    }

    /// The normalized query, key and value inputs of the attention.
    private func attentionInputs(
        _ hidden: MLXArray, readout: MLXArray?, messages: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = hidden.dtype
        var sources: (MLXArray, MLXArray, MLXArray)
        if let readout {
            sources =
                isSliding
                ? (
                    readout[.ellipsis, 0, 0...], readout[.ellipsis, 0, 0...],
                    readout[.ellipsis, 0, 0...]
                )
                : (
                    readout[.ellipsis, 0, 0...], readout[.ellipsis, 1, 0...],
                    readout[.ellipsis, 2, 0...]
                )
        } else {
            sources = (hidden, hidden, hidden)
        }
        if let messages, let queryMessageGate, let keyMessageGate, let valueMessageGate {
            sources = (
                queryMessageGate(sources.0, messages[.ellipsis, 0, 0...]),
                keyMessageGate(sources.1, messages[.ellipsis, 1, 0...]),
                valueMessageGate(sources.2, messages[.ellipsis, 2, 0...])
            )
        } else if isSliding, let inputNorm {
            let normed = inputNorm(sources.0).asType(dtype)
            return (normed, normed, normed)
        }
        if let inputNorm {
            return (
                inputNorm(sources.0).asType(dtype), inputNorm(sources.1).asType(dtype),
                inputNorm(sources.2).asType(dtype)
            )
        }
        return (
            queryInputNorm!(sources.0).asType(dtype), keyInputNorm!(sources.1).asType(dtype),
            valueInputNorm!(sources.2).asType(dtype)
        )
    }

    /// - Parameters:
    ///   - hidden: the partial sum of the current block.
    ///   - blocks: float32 `[K, B, L, D]`, the normalized completed blocks.
    ///   - messages: float32 `[B, L, 4 * coreLayers, D]` and the core index of this layer, whose
    ///     query, key, value and MLP messages start at `4 * index`.
    func callAsFunction(
        _ hidden: MLXArray, blocks: MLXArray?, cache: CacheList?, convolutionMask: MLXArray?,
        messages bank: (MLXArray, Int)?, masks: Inkling2DecodeMasks
    ) -> (MLXArray, MLXArray) {
        if let cache, let convolutionCache = cache[1] as? ArraysCache, hidden.dim(1) == 1,
            hidden.dtype == .bfloat16, convolutionMask == nil,
            Inkling2Attention.supportsDecode(cache[0]),
            Device.defaultDevice().deviceType == .gpu
        {
            return decode(
                hidden, blocks: blocks, cache: cache, convolutionCache: convolutionCache,
                bank: bank)
        }
        let messages = bank.map { bank, index in
            bank[.ellipsis, (index * 4) ..< (index * 4 + 4), 0...]
        }
        let dtype = hidden.dtype
        let convolutionCache = cache?[1] as? ArraysCache
        let states: [MLXArray?] = (0 ..< 4).map { convolutionCache?[$0] }
        let normedResidual = unweightedRMSNorm(hidden.asType(.float32))

        var readout: MLXArray?
        if let attentionInput, let blocks {
            readout = attentionInput(blocks, normedResidual)
        }
        let (b, l, dims) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let eps = postAttentionNorm.eps
        var gates: (controllers: MessageControllers, messages: MLXArray, first: Int)?
        if let bank, Device.defaultDevice().deviceType == .gpu,
            let controllers = messageControllers(), controllers.down.dtype == dtype
        {
            gates = (controllers, bank.0.reshaped(b * l, -1, dims), bank.1 * 4)
        }
        let (q, k, v): (MLXArray, MLXArray, MLXArray)
        if let gates, let readout {
            let (controllers, messages, first) = gates
            let norms =
                inputNorm.map { $0.weight.reshaped(1, dims) }
                ?? stacked([queryInputNorm!.weight, keyInputNorm!.weight, valueInputNorm!.weight])
            let out = inkling2MessageGates(
                inputs: readout.reshaped(b * l, -1, dims), messages: messages, first: first,
                down: controllers.down[..<3], forget: controllers.forget[..<3],
                gain: controllers.gain[..<3], normWeight: norms, eps: eps)
            (q, k, v) = (
                out[0].reshaped(b, l, dims), out[1].reshaped(b, l, dims),
                out[2].reshaped(b, l, dims)
            )
        } else {
            (q, k, v) = attentionInputs(hidden, readout: readout, messages: messages)
        }
        let (attentionOutput, keyWindow, valueWindow) = attention(
            queryInput: q, keyInput: k, valueInput: v, cache: cache?[0], convolutions: states,
            convolutionMask: convolutionMask, masks: masks)
        let (residual, attentionWindow) = postAttentionConvolution.residual(
            attentionOutput, base: isBlockBoundary ? nil : hidden, state: states[2],
            mask: convolutionMask)

        var blocks = blocks
        if isBlockBoundary {
            let block = normedResidual.expandedDimensions(axis: 0)
            blocks = blocks.map { concatenated([$0, block]) } ?? block
        }

        var mlpInputs = mlpInput(blocks!, unweightedRMSNorm(residual))[.ellipsis, 0, 0...]
        let expertInput: MLXArray
        if let gates {
            let (controllers, messages, first) = gates
            expertInput = inkling2MessageGates(
                inputs: mlpInputs.reshaped(b * l, 1, dims), messages: messages, first: first + 3,
                down: controllers.down[3...], forget: controllers.forget[3...],
                gain: controllers.gain[3...],
                normWeight: postAttentionNorm.weight.reshaped(1, dims),
                eps: eps
            ).reshaped(b, l, dims)
        } else {
            if let messages, let mlpMessageGate {
                mlpInputs = mlpMessageGate(mlpInputs, messages[.ellipsis, 3, 0...])
            }
            expertInput = postAttentionNorm(mlpInputs).asType(dtype)
        }
        let (output, mlpWindow) = mlp.prefill(
            expertInput, residual: residual,
            convolution: postMLPConvolution, state: states[3], mask: convolutionMask)

        if let convolutionCache {
            let lengths = convolutionCache.currentLengths
            for (slot, window) in [keyWindow, valueWindow, attentionWindow, mlpWindow].enumerated()
            {
                convolutionCache[slot] = convolutionState(
                    window, lengths: lengths, count: convolutionStateLength)
            }
            convolutionCache.advance(hidden.dim(1))
        }
        return (output, blocks!)
    }

    /// One token per row through the decode kernels.
    private func decode(
        _ hidden: MLXArray, blocks: MLXArray?, cache: CacheList, convolutionCache: ArraysCache,
        bank: (MLXArray, Int)?
    ) -> (MLXArray, MLXArray) {
        let (rows, dims) = (hidden.dim(0), hidden.dim(2))
        let kvDims = attention.kvHeads * attention.headDim
        func state(_ slot: Int, _ channels: Int) -> MLXArray {
            convolutionCache[slot] ?? MLXArray.zeros([rows, 3, channels], dtype: hidden.dtype)
        }
        let x = hidden.reshaped(rows, dims)
        let eps = postAttentionNorm.eps
        let messages = bank.map { ($0.0.reshaped(rows, -1, dims), $0.1 * 4) }
        let controllers = messages == nil ? nil : messageControllers()
        let attentionNorms =
            inputNorm.map { $0.weight.reshaped(1, dims) }
            ?? stacked([queryInputNorm!.weight, keyInputNorm!.weight, valueInputNorm!.weight])

        // The normalized attention inputs.
        let normedResidual: MLXArray
        var inputs: [MLXArray]
        if let attentionInput, let blocks {
            let mix = inkling2DepthMix(
                blocks: blocks.reshaped(-1, rows, dims), base: x, convolution: nil,
                queries: attentionInput.query.reshaped(-1, dims),
                normWeight: controllers == nil ? attentionNorms : nil, eps: eps)
            normedResidual = mix.normed
            if let messages, let controllers {
                let (out, _) = inkling2MessageControllers(
                    inputs: mix.readout.transposed(1, 0, 2), messages: messages.0,
                    first: messages.1, down: controllers.down[..<3],
                    forget: controllers.forget[..<3], gain: controllers.gain[..<3],
                    normWeight: attentionNorms, eps: eps, f32Copy: false)
                inputs = (0 ..< 3).map { out[$0 ..< ($0 + 1)].reshaped(rows, 1, dims) }
            } else {
                let readout = mix.readout.reshaped(rows, 1, -1, dims)
                inputs = (0 ..< readout.dim(2)).map { readout[0..., 0..., $0, 0...] }
            }
        } else {
            normedResidual = unweightedRMSNorm(x.asType(.float32))
            let (q, k, v) = attentionInputs(
                hidden, readout: nil,
                messages: bank.map { bank, index in
                    bank[.ellipsis, (index * 4) ..< (index * 4 + 4), 0...]
                })
            inputs = [q, k, v]
        }
        if inputs.count == 1 {
            inputs = [inputs[0], inputs[0], inputs[0]]
        }

        let (attentionOutput, keyState, valueState) = attention.decode(
            queryInput: inputs[0], keyInput: inputs[1], valueInput: inputs[2], cache: cache[0],
            states: (state(0, kvDims), state(1, kvDims)))

        // The MoE input from the blocks and the residual after the attention.
        var blocks = blocks?.reshaped(-1, rows, dims)
        if isBlockBoundary {
            let block = normedResidual.expandedDimensions(axis: 0)
            blocks = blocks.map { concatenated([$0, block]) } ?? block
        }
        let mlpNorm = postAttentionNorm.weight.reshaped(1, dims)
        let mix = inkling2DepthMix(
            blocks: blocks!, base: isBlockBoundary ? nil : x,
            convolution: (
                attentionOutput.reshaped(rows, dims), state(2, dims),
                postAttentionConvolution.conv1d.weight
            ),
            queries: mlpInput.query.reshaped(1, dims),
            normWeight: controllers == nil ? mlpNorm : nil, f32Copy: controllers == nil,
            eps: eps)
        var expertInput = mix.readout.reshaped(rows, dims)
        var routerInput = mix.readoutF32?.reshaped(rows, dims)
        if let messages, let controllers {
            let (out, f32) = inkling2MessageControllers(
                inputs: mix.readout.reshaped(1, rows, dims), messages: messages.0,
                first: messages.1 + 3, down: controllers.down[3...],
                forget: controllers.forget[3...], gain: controllers.gain[3...],
                normWeight: mlpNorm, eps: eps, f32Copy: true)
            expertInput = out.reshaped(rows, dims)
            routerInput = f32!.reshaped(rows, dims)
        }
        let (output, mlpState) = mlp.decode(
            expertInput, routerInput: routerInput!, residual: mix.residual, state: state(3, dims),
            convolutionWeight: postMLPConvolution.conv1d.weight)

        convolutionCache[0] = keyState
        convolutionCache[1] = valueState
        convolutionCache[2] = mix.state
        convolutionCache[3] = mlpState
        convolutionCache.advance(1)
        return (output.reshaped(rows, 1, dims), blocks!.reshaped(-1, rows, 1, dims))
    }

    // MARK: Stacked message controllers

    struct MessageControllers {
        /// bfloat16 `[4, R, 2D]`: the query, key, value and MLP controllers.
        var down: MLXArray
        /// bfloat16 `[4, R, D]`.
        var forget: MLXArray
        /// bfloat16 `[4, R, D]`.
        var gain: MLXArray
    }

    private let stackedControllers = MessageControllersBox()

    private var gates: [Inkling2MessageGate] {
        [queryMessageGate, keyMessageGate, valueMessageGate, mlpMessageGate].compactMap { $0 }
    }

    /// The controllers stacked for the decode kernels. The gates' weights become views of the
    /// stacked storage, in bfloat16, which holds the float32 checkpoint values exactly.
    func prepareMessageControllers() throws {
        guard gates.count == 4, stackedControllers.value == nil else { return }
        let stacked = MessageControllers(
            down: MLX.stacked(gates.map { $0.downProj.weight.asType(.bfloat16) }),
            forget: MLX.stacked(gates.map { $0.forgetProj.weight.T.asType(.bfloat16) }),
            gain: MLX.stacked(gates.map { $0.gainProj.weight.T.asType(.bfloat16) }))
        eval(stacked.down, stacked.forget, stacked.gain)
        stackedControllers.installing = true
        defer { stackedControllers.installing = false }
        for (index, gate) in gates.enumerated() {
            try gate.update(
                modules: ModuleChildren(values: [
                    "down_proj": .value(Linear(weight: stacked.down[index])),
                    "forget_gate_proj": .value(Linear(weight: stacked.forget[index].T)),
                    "message_gate_proj": .value(Linear(weight: stacked.gain[index].T)),
                ]), verify: [])
        }
        stackedControllers.value = stacked
    }

    private func messageControllers() -> MessageControllers? {
        if let value = stackedControllers.value { return value }
        guard gates.count == 4 else { return nil }
        return MessageControllers(
            down: MLX.stacked(gates.map { $0.downProj.weight.asType(.bfloat16) }),
            forget: MLX.stacked(gates.map { $0.forgetProj.weight.T.asType(.bfloat16) }),
            gain: MLX.stacked(gates.map { $0.gainProj.weight.T.asType(.bfloat16) }))
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let replacesGates = parameters.flattened().contains { key, _ in
            key.contains("_message_gate.")
        }
        defer { if replacesGates { stackedControllers.invalidate() } }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { if key.hasSuffix("_message_gate") { stackedControllers.invalidate() } }
        try super.updateModule(key: key, value)
    }
}

/// Hides the stacked controllers from `Module` reflection.
private final class MessageControllersBox: @unchecked Sendable {
    var value: Inkling2DecoderLayer.MessageControllers?
    var installing = false

    func invalidate() {
        if !installing { value = nil }
    }
}

// MARK: - Text model

private final class Inkling2TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockAttentionResidual
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    let config: Inkling2TextConfiguration
    let visits: [Int]

    init(_ config: Inkling2TextConfiguration) {
        self.config = config
        self.visits = config.layerVisits
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Inkling2DecoderLayer(config, layer: $0)
        }
        _finalInput.wrappedValue = Inkling2BlockAttentionResidual(dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if let loop = config.loop {
            // The query, key, value and MLP message queries of each core layer.
            _messageInput.wrappedValue = Inkling2MessageInput(
                dimensions: config.hiddenSize, layers: loop.core.count)
        }
        super.init()
    }

    func embed(_ inputs: MLXArray) -> MLXArray {
        embedNorm(embedTokens(inputs))
    }

    /// The final hidden states before the norm's output, for the rows of `outputRows`.
    func callAsFunction(
        _ embeddings: MLXArray, cache: [KVCache]?, outputRows: Range<Int>? = nil
    ) -> MLXArray {
        var hidden = embeddings
        let caches = cache?.map { $0 as? CacheList }
        let masks = Inkling2DecodeMasks()
        var convolutionMask: MLXArray?
        if let first = caches?.first??[1] as? ArraysCache {
            convolutionMask = first.makeMask(N: hidden.dim(1))
        }

        var visit = 0
        var blocks: MLXArray?
        /// The messages of a loop pass read the previous pass's blocks after the prelude and its
        /// partial block, with the `[layers, 4, D]` queries of the core layers.
        typealias MessageSources = (blocks: MLXArray, partial: MLXArray, queries: MLXArray)
        func run(_ layerRange: Range<Int>, messages: MessageSources?) {
            // One token reads every core layer's messages at once; a chunk reads them per layer.
            let bank = messages.flatMap { sources in
                hidden.dim(1) == 1
                    ? inkling2DepthReadout(
                        blocks: sources.blocks, partial: sources.partial,
                        queries: sources.queries.reshaped(-1, config.hiddenSize),
                        queriesPerGroup: 4) : nil
            }
            for (index, layer) in layerRange.enumerated() {
                var layerMessages = bank.map { ($0, index) }
                if let messages, bank == nil {
                    let readout = inkling2DepthReadout(
                        blocks: messages.blocks, partial: messages.partial,
                        queries: messages.queries[index], queriesPerGroup: 4)
                    layerMessages = (readout, 0)
                }
                (hidden, blocks) = layers[layer](
                    hidden, blocks: blocks, cache: caches?[visit], convolutionMask: convolutionMask,
                    messages: layerMessages, masks: masks)
                visit += 1
            }
        }

        if let loop = config.loop, let messageInput {
            run(0 ..< loop.coreStart, messages: nil)
            let preludeHidden = hidden
            let preludeBlocks = blocks!
            let preludeCount = preludeBlocks.dim(0)
            for pass in 0 ..< loop.passes {
                var messages: MessageSources?
                if pass > 0 {
                    messages = (
                        blocks![(preludeCount + 1)...],
                        unweightedRMSNorm(hidden.asType(.float32)), messageInput.query
                    )
                    hidden = preludeHidden
                    blocks = preludeBlocks
                }
                run(loop.core, messages: messages)
            }
            run(loop.coreEnd ..< config.hiddenLayers, messages: nil)
        } else {
            run(0 ..< config.hiddenLayers, messages: nil)
        }

        var finalBlocks = blocks!
        if let outputRows {
            hidden = hidden[0..., outputRows, 0...]
            finalBlocks = finalBlocks[0..., 0..., outputRows, 0...]
        }
        let normed = unweightedRMSNorm(hidden.asType(.float32))
        let output = finalInput(finalBlocks, normed)[.ellipsis, 0, 0...]
        return norm(output).asType(hidden.dtype)
    }
}

// MARK: - Model

/// The vision stack, with the parameter path it has in ``Inkling2``.
private final class Inkling2Vision: Module {
    @ModuleInfo(key: "vision_tower") var tower: Inkling2VisionModel

    init(_ config: Inkling2Configuration) {
        _tower.wrappedValue = Inkling2VisionModel(
            config.visionConfiguration!, textHiddenSize: config.textConfiguration.hiddenSize)
        super.init()
    }
}

/// Inkling 2 Mini and Inkling 2 Small: a multimodal mixture-of-experts model whose text stack
/// can run its middle layers several times per token.
public final class Inkling2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Inkling2VisionModel?
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    public let config: Inkling2Configuration

    /// Long chunks give each of the 128 experts enough rows to fill its matmul tiles.
    static let prefillStepSize = 2048

    /// Loads the vision tower on the first image when the factory loads the model without it.
    private let onDemandVision: OnDemandComponent<Inkling2Vision>?

    public var vocabularySize: Int { config.textConfiguration.unpaddedVocabularySize }

    /// Thinking, answers and tool calls arrive in their own message frames.
    public var toolCallFormat: ToolCallFormat? { .inkling }

    public var kvHeads: [Int] {
        let text = config.textConfiguration
        return text.layerVisits.map { text.isSliding($0) ? text.slidingKVHeads : text.kvHeads }
    }

    public init(_ config: Inkling2Configuration) {
        self.config = config
        let text = config.textConfiguration
        _languageModel.wrappedValue = Inkling2TextModel(text)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
        if let vision = config.visionConfiguration {
            _visionTower.wrappedValue = Inkling2VisionModel(
                vision, textHiddenSize: text.hiddenSize)
            onDemandVision = OnDemandComponent(.vision) { Inkling2Vision(config) }
        } else {
            onDemandVision = nil
        }
        super.init()
    }

    /// Each layer visit has its own attention cache and the states of its four convolutions.
    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        if let capacity = try parameters?.effectiveKVCacheCapacity(),
            capacity.preservedPrefixTokens > 0
        {
            throw VLMError.processing(
                "Inkling 2 does not support preserved prefixes in rotating KV caches")
        }
        let text = config.textConfiguration
        return try text.layerVisits.map { layer in
            let attention: KVCache =
                text.isSliding(layer)
                ? try makeSlidingWindowKVCache(parameters: parameters, window: text.slidingWindow)
                : try makeAttentionKVCache(parameters: parameters)
            return CacheList(attention, ArraysCache(size: 4))
        }
    }

    private func vision() throws -> Inkling2VisionModel {
        if let visionTower { return visionTower }
        guard let onDemandVision else { throw VLMError.imageProcessingFailure("no vision tower") }
        return try onDemandVision.load().tower
    }

    /// Token embeddings with the image features spliced in, `[1, L, D]`.
    private func inputEmbeddings(_ tokens: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let embeddings = languageModel.embed(tokens)
        guard let image, let grids = image.frames else { return embeddings }

        let tower = try vision()
        let features = try tower.encode(pixels: image.pixels, grids: grids)
            .asType(embeddings.dtype)
        var positions = [Int32]()
        for (index, id) in tokens.reshaped(-1).asArray(Int.self).enumerated()
        where id == config.imageTokenId {
            positions.append(Int32(index))
        }
        guard positions.count == features.dim(0) else {
            throw VLMError.processing(
                "Image features and image tokens do not match: tokens=\(positions.count), "
                    + "features=\(features.dim(0))")
        }
        embeddings[0..., MLXArray(positions), 0...] = features.expandedDimensions(axis: 0)
        return embeddings
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        let text = config.textConfiguration
        var logits = lmHead(hidden).asType(.float32)
        if text.finalLogitSoftcap > 0 {
            logits = tanh(logits / text.finalLogitSoftcap) * text.finalLogitSoftcap
        }
        if text.unpaddedVocabularySize < text.vocabularySize {
            logits = logits[.ellipsis, ..<text.unpaddedVocabularySize]
        }
        return logits
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let embeddings = try inputEmbeddings(tokens, image: input.image)

        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(
            total: total, defaultStepSize: Self.prefillStepSize
        ) { range in
            _ = languageModel(embeddings[0..., range, 0...], cache: cache, outputRows: 0 ..< 1)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let remaining = embeddings[0..., processed..., 0...]
        let last = remaining.dim(1)
        let hidden = languageModel(remaining, cache: cache, outputRows: (last - 1) ..< last)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits(hidden)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(languageModel(languageModel.embed(inputs), cache: cache))
    }

    public func prepare() throws {
        for layer in languageModel.layers {
            try layer.attention.prepareFusedProjections()
            try layer.prepareMessageControllers()
        }
    }

    /// Strips the `model.` scope and sets each expert layer to its checkpoint's quantization.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var weights = [String: MLXArray](minimumCapacity: checkpoint.weights.count)
        for (key, value) in checkpoint.weights {
            var key = key
            if key.hasPrefix("model.") { key = String(key.dropFirst("model.".count)) }
            weights[key] = value
        }

        let text = config.textConfiguration
        for layer in 0 ..< text.hiddenLayers {
            let path = "language_model.layers.\(layer).mlp.experts"
            guard
                let gateUp = Inkling2ExpertFormat(
                    path: "\(path).gate_up_proj", weights: weights, inputDims: text.hiddenSize),
                let down = Inkling2ExpertFormat(
                    path: "\(path).down_proj", weights: weights,
                    inputDims: text.intermediateSize)
            else { continue }
            let mlp = languageModel.layers[layer].mlp
            if mlp.experts.gateUpProj.format != gateUp || mlp.experts.downProj.format != down {
                let experts = Inkling2Experts(text, gateUp: gateUp, down: down)
                try mlp.update(
                    modules: ModuleChildren(values: ["experts": .value(experts)]), verify: .none)
            }
        }
        checkpoint.weights = weights
        return checkpoint
    }
}

extension Inkling2: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}

extension Inkling2: ModelComponentsProviding {
    public var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [.vision: .modules("vision_tower")]
    }

    public var onDemandComponents: [AnyOnDemandComponent] {
        onDemandVision.map { [$0] } ?? []
    }
}
