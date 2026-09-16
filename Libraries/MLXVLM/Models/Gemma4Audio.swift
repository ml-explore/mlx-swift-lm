// Copyright © 2026 Apple Inc.

//
// Gemma 4 audio encoder (`gemma4_audio`), a Universal Speech Model (USM)
// Conformer. Ported from the Python reference
// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio.py
// and cross-checked against VincentGourbin/gemma-4-swift-mlx.
//
// Pipeline (mel -> text-hidden features):
//   mel [B, T, 128]
//     -> SubSampleConvProjection  (2× Conv2d 3×3 stride-2 -> flatten(F·C) -> Linear)  [B, T/4, hidden]
//     -> 12 × ConformerBlock  (macaron FFN -> chunked local attn -> depthwise light-conv -> FFN)
//     -> output_proj (Linear hidden -> output_proj_dims)                              [B, T/4, 1536]
//
// The `embed_audio` projection (`Gemma4MultimodalEmbedder`) and the scatter of
// audio soft tokens into the text stream live in `Gemma4.swift`; this file is the
// `audio_tower` only. All `@ModuleInfo`/`@ParameterInfo` keys match the checkpoint
// `audio_tower.*` weight names.
//

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Gemma4AudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let residualWeight: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let outputProjectionDimensions: Int?
    public let useClippedLinears: Bool

    public var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjectionDimensions = "output_proj_dims"
        case useClippedLinears = "use_clipped_linears"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_audio"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        attentionChunkSize = try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        attentionContextLeft = try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        attentionContextRight = try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        attentionLogitCap = try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50.0
        attentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1e9
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        gradientClipping = try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: .outputProjectionDimensions)
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
    }
}

// MARK: - Norms

/// RMSNorm with the weight applied directly (no Gemma `1 + weight` offset).
/// Matches the reference `AudioRMSNorm`.
private final class Gemma4AudioRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - SubSample Conv Projection (SSCP)

/// One subsample stage: `Conv2d(3×3, stride 2) -> LayerNorm(channels) -> ReLU`
/// with symmetric `(1,1)` padding on the time and frequency axes.
private final class Gemma4AudioSSCPConvBlock: Module {
    let timeStride = 2

    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(config: Gemma4AudioConfiguration, idx: Int) {
        let inChannels = idx == 0 ? 1 : config.subsamplingConvChannels[idx - 1]
        let outChannels = config.subsamplingConvChannels[idx]
        // MLX Conv2d: input [B, H, W, C], weight [C_out, kH, kW, C_in].
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair((3, 3)),
            stride: IntOrPair((2, 2)),
            padding: IntOrPair(0),
            bias: false
        )
        // LayerNorm over the channel (last) dim; weight only, no bias.
        self._norm.wrappedValue = LayerNorm(
            dimensions: outChannels, eps: config.rmsNormEps, affine: true, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, F, C]` (channel-last).
    ///   - mask: `[B, T]` boolean, `true == padding`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        // Zero invalid (padding) positions.
        let maskExpanded = expandedDimensions(expandedDimensions(mask, axis: -1), axis: -1)
        var h = MLX.where(maskExpanded, MLXArray(0, dtype: x.dtype), x)

        // Manual symmetric pad on T and F (the conv itself uses padding 0).
        h = padded(h, widths: [0, 1, 1, 0])
        h = conv(h)  // [B, T_out, F_out, C_out]
        let tOut = h.dim(1)

        // Downsample the (unpadded) mask by the time stride.
        let tIn = mask.dim(1)
        let evenIndices = MLXArray(stride(from: 0, to: tIn, by: timeStride).map { Int32($0) })
        var outputMask = take(mask, evenIndices, axis: 1)
        if outputMask.dim(1) > tOut {
            outputMask = outputMask[0..., ..<tOut]
        }

        h = norm(h)
        h = relu(h)
        return (h, outputMask)
    }
}

/// Two `SSCPConvBlock`s → flatten `(freq · channels)` → `Linear` to `hidden_size`.
private final class Gemma4AudioSubSampleConvProjection: Module {
    static let inputFeatSize = 128

    @ModuleInfo(key: "layer0") var layer0: Gemma4AudioSSCPConvBlock
    @ModuleInfo(key: "layer1") var layer1: Gemma4AudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(config: Gemma4AudioConfiguration) {
        self._layer0.wrappedValue = Gemma4AudioSSCPConvBlock(config: config, idx: 0)
        self._layer1.wrappedValue = Gemma4AudioSSCPConvBlock(config: config, idx: 1)

        // Frequency after two `(f + 2 - 3) / 2 + 1` stride-2 stages (128 -> 64 -> 32).
        var freq = Self.inputFeatSize
        for _ in 0 ..< 2 { freq = (freq + 2 - 3) / 2 + 1 }
        let projInputDim = freq * (config.subsamplingConvChannels.last ?? 32)
        self._inputProjLinear.wrappedValue = Linear(projInputDim, config.hiddenSize, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - audioMel: `[B, T, F_in]`.
    ///   - mask: `[B, T]` boolean, `true == padding`.
    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = expandedDimensions(audioMel, axis: -1)  // [B, T, F, 1]
        var m = mask
        (x, m) = layer0(x, mask: m)
        (x, m) = layer1(x, mask: m)

        // Flatten (F, C) -> [B, T, F*C], then project.
        let (b, t) = (x.dim(0), x.dim(1))
        x = x.reshaped(b, t, x.dim(2) * x.dim(3))
        x = inputProjLinear(x)
        return (x, m)
    }
}

// MARK: - Conformer sub-layers

/// Macaron feed-forward: `RMSNorm -> Linear(4×) -> SiLU -> Linear -> RMSNorm`,
/// added back with a `residual_weight` (0.5) half-step.
private final class Gemma4AudioFeedForward: Module, UnaryLayer {
    let gradientClipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: Gemma4ClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: Gemma4ClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma4AudioRMSNorm

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self.residualWeight = config.residualWeight
        self._preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffwLayer1.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 4,
            useClipping: config.useClippedLinears)
        self._ffwLayer2.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize * 4, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._postLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gc = MLXArray(gradientClipping)
        let residual = x
        var h = clip(x, min: -gc, max: gc)
        h = preLayerNorm(h)
        h = ffwLayer1(h)
        h = silu(h)
        h = ffwLayer2(h)
        h = clip(h, min: -gc, max: gc)
        h = postLayerNorm(h)
        return residual + h * MLXArray(residualWeight)
    }
}

/// Chunked local self-attention with Transformer-XL relative-position bias and
/// logit softcapping. Query positions are grouped into non-overlapping chunks of
/// `chunk_size`; each chunk attends over a `context_size` window (`max_past` left,
/// `max_future` right). Attention math runs in float32.
private final class Gemma4AudioAttention: Module {
    let numHeads: Int
    let headDim: Int
    let chunkSize: Int
    let maxPastHorizon: Int
    let maxFutureHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let qScale: Float
    let kScale: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "post") var post: Gemma4ClippableLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray

    init(config: Gemma4AudioConfiguration) {
        self.numHeads = config.attentionHeads
        self.headDim = config.hiddenSize / config.attentionHeads
        self.chunkSize = config.attentionChunkSize
        self.maxFutureHorizon = config.attentionContextRight
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)
        self.contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        self.invalidLogitsValue = config.attentionInvalidLogitsValue
        self.softcap = config.attentionLogitCap
        self.qScale = Float(pow(Double(headDim), -0.5) / log(2.0))
        self.kScale = Float(log(1.0 + exp(1.0)) / log(2.0))

        let dim = numHeads * headDim
        self._qProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: dim, useClipping: config.useClippedLinears)
        self._kProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: dim, useClipping: config.useClippedLinears)
        self._vProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: dim, useClipping: config.useClippedLinears)
        self._post.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._relativeKProj.wrappedValue = Linear(config.hiddenSize, dim, bias: false)
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])
        super.init()
    }

    private func padDim1(_ x: MLXArray, _ left: Int, _ right: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair(0), count: x.ndim)
        widths[1] = IntOrPair((left, right))
        return padded(x, widths: widths)
    }

    /// `[B, T, ...] -> [B, num_blocks, chunk_size, ...]` (non-overlapping query chunks).
    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let t = x.dim(1)
        let numBlocks = (t + chunkSize - 1) / chunkSize
        let padLen = numBlocks * chunkSize - t
        let xp = padLen > 0 ? padDim1(x, 0, padLen) : x
        var shape = [xp.dim(0), numBlocks, chunkSize]
        for d in 2 ..< xp.ndim { shape.append(xp.dim(d)) }
        return xp.reshaped(shape)
    }

    /// `[B, T, ...] -> [B, num_blocks, context_size, ...]` (overlapping key/value windows).
    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let xp = padDim1(x, padLeft, padRight)
        let tPadded = xp.dim(1)
        let numBlocks = (tPadded - contextSize) / chunkSize + 1
        var idx = [Int32]()
        idx.reserveCapacity(numBlocks * contextSize)
        for b in 0 ..< numBlocks {
            let start = b * chunkSize
            for o in 0 ..< contextSize { idx.append(Int32(start + o)) }
        }
        let gathered = take(xp, MLXArray(idx), axis: 1)
        var shape = [xp.dim(0), numBlocks, contextSize]
        for d in 2 ..< xp.ndim { shape.append(xp.dim(d)) }
        return gathered.reshaped(shape)
    }

    /// Transformer-XL relative-position bias (arXiv:1901.02860 App. B).
    /// - queries: `[B, U, W, N, H]`, keys: `[B, U, C, N, H]` → logits `[B, N, U, W, C]`.
    private func relativePositionLogits(queries: MLXArray, keys: MLXArray) -> MLXArray {
        let (b, u, w) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let c = keys.dim(2)

        // Sinusoidal timing signal over positions [max_past ... -max_future].
        let positions = stride(from: maxPastHorizon, through: -maxFutureHorizon, by: -1)
            .map { Float($0) }
        let maxSpan = positions.count
        let numTimescales = (numHeads * headDim) / 2
        let logInc = Float(log(10000.0) / Double(max(numTimescales - 1, 1)))
        let invTimescales = exp(
            MLXArray(Array(0 ..< numTimescales)).asType(.float32) * (-logInc))
        let pos = expandedDimensions(MLXArray(positions), axis: -1)  // [maxSpan, 1]
        let scaledTime = pos * expandedDimensions(invTimescales, axis: 0)  // [maxSpan, numTimescales]
        var sinEmb = concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)  // [maxSpan, 2·nt]
        sinEmb = relativeKProj(sinEmb.asType(queries.dtype))  // [maxSpan, N·H]
        sinEmb = sinEmb.reshaped(maxSpan, numHeads, headDim)

        // Content term: [B, N, U, W, H] @ [B, N, U, H, C] -> [B, N, U, W, C].
        let queriesP = queries.transposed(0, 3, 1, 2, 4)
        let keysP = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(queriesP, keysP)

        // Relative term: [B, N, U*W, H] @ [1, N, H, maxSpan] -> [B, N, U*W, maxSpan].
        let sinEmbT = sinEmb.transposed(1, 2, 0)  // [N, H, maxSpan]
        let qReshaped = queriesP.reshaped(b, numHeads, u * w, headDim)
        var termBD = matmul(qReshaped, expandedDimensions(sinEmbT, axis: 0))
        termBD = termBD.reshaped(b, numHeads, u, w, maxSpan)

        // Relative shift: pad -> reshape -> slice -> reshape, aligning span -> context.
        let padAmount = (c + 1) - maxSpan
        termBD = padded(termBD, widths: [0, 0, 0, 0, IntOrPair((0, padAmount))])
        termBD = termBD.reshaped(b, numHeads, u, w * (c + 1))
        termBD = termBD[0..., 0..., 0..., ..<(w * c)]
        termBD = termBD.reshaped(b, numHeads, u, w, c)

        return termAC + termBD
    }

    /// - Parameters:
    ///   - hiddenStates: `[B, T, D]`.
    ///   - mask: `[B, T]` boolean, `true == padding`.
    ///   - causalValidMask: `[chunk_size, context_size]` boolean, `true == attend`.
    func callAsFunction(
        _ hiddenStates: MLXArray, mask: MLXArray, causalValidMask: MLXArray
    ) -> MLXArray {
        let (b, t) = (hiddenStates.dim(0), hiddenStates.dim(1))

        var q = qProj(hiddenStates).asType(.float32).reshaped(b, t, numHeads, headDim)
        var k = kProj(hiddenStates).asType(.float32).reshaped(b, t, numHeads, headDim)
        let v = vProj(hiddenStates).asType(.float32).reshaped(b, t, numHeads, headDim)

        let perDim = softplus(perDimScale).asType(.float32)  // [H]
        q = q * (perDim * MLXArray(qScale))
        k = k * MLXArray(kScale)

        let queryBlocks = convertToBlock(q)  // [B, U, W, N, H]
        let keyBlocks = extractBlockContext(k)  // [B, U, C, N, H]
        let valueBlocks = extractBlockContext(v)  // [B, U, C, N, H]
        let u = queryBlocks.dim(1)

        let validMask = logicalNot(mask)  // [B, T], true == valid
        let extractedValid = extractBlockContext(validMask)  // [B, U, C]
        let condition =
            extractedValid.reshaped(b, 1, u, 1, contextSize)
            & causalValidMask.reshaped(1, 1, 1, chunkSize, contextSize)

        var logits = relativePositionLogits(queries: queryBlocks, keys: keyBlocks)  // [B,N,U,W,C]
        logits = tanh(logits / MLXArray(softcap)) * MLXArray(softcap)
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))

        let probs = softmax(logits, axis: -1)  // [B, N, U, W, C]

        // context = einsum("bnuwc,bucnh->buwnh", probs, valueBlocks) via batched matmul.
        let probsP = probs.transposed(0, 2, 1, 3, 4)  // [B, U, N, W, C]
        let valuesP = valueBlocks.transposed(0, 1, 3, 2, 4)  // [B, U, N, C, H]
        var context = matmul(probsP, valuesP)  // [B, U, N, W, H]
        context = context.transposed(0, 1, 3, 2, 4)  // [B, U, W, N, H]
        context = context.reshaped(b, u * chunkSize, numHeads, headDim)
        context = context[0..., ..<t]
        context = context.reshaped(b, t, numHeads * headDim)
        return post(context)
    }
}

/// Light convolution module: `norm -> Linear(2×) -> GLU -> depthwise causal Conv1d
/// -> norm -> SiLU -> Linear`, added back as a plain residual.
private final class Gemma4AudioLightConv1d: Module, UnaryLayer {
    let gradientClipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Gemma4ClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Gemma4ClippableLinear

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self.causalPadding = config.convKernelSize - 1
        self._preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearStart.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 2,
            useClipping: config.useClippedLinears)
        // Depthwise: groups == channels. MLX Conv1d weight is [C_out, K, C_in/groups=1].
        self._depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            groups: config.hiddenSize,
            bias: false
        )
        self._convNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearEnd.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preLayerNorm(x)
        h = linearStart(h)  // [B, T, 2H]

        // GLU gate over the last dim.
        let half = h.dim(-1) / 2
        h = h[0..., 0..., ..<half] * sigmoid(h[0..., 0..., half...])

        // Causal left-pad on time, then depthwise conv (padding 0 inside the conv).
        h = padded(h, widths: [0, IntOrPair((causalPadding, 0)), 0])
        h = depthwiseConv1d(h)

        let gc = MLXArray(gradientClipping)
        h = clip(h, min: -gc, max: gc)
        h = convNorm(h)
        h = silu(h)
        h = linearEnd(h)
        return h + residual
    }
}

/// Macaron Conformer block:
/// `FFN → norm → attention (+residual) → zero-invalid → light-conv → FFN → clamp → norm_out`.
private final class Gemma4AudioConformerBlock: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "feed_forward1") var feedForward1: Gemma4AudioFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: Gemma4AudioLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: Gemma4AudioFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: Gemma4AudioRMSNorm

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self._feedForward1.wrappedValue = Gemma4AudioFeedForward(config: config)
        self._selfAttn.wrappedValue = Gemma4AudioAttention(config: config)
        self._lconv1d.wrappedValue = Gemma4AudioLightConv1d(config: config)
        self._feedForward2.wrappedValue = Gemma4AudioFeedForward(config: config)
        self._normPreAttn.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._normPostAttn.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._normOut.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray
    ) -> MLXArray {
        let gc = MLXArray(gradientClipping)
        var h = feedForward1(x)

        let residual = h
        h = clip(h, min: -gc, max: gc)
        h = normPreAttn(h)
        h = selfAttn(h, mask: mask, causalValidMask: causalValidMask)
        h = clip(h, min: -gc, max: gc)
        h = residual + normPostAttn(h)

        // Zero invalid positions before the (causal) light-conv.
        let validity = expandedDimensions(logicalNot(mask), axis: -1).asType(h.dtype)
        h = h * validity

        h = lconv1d(h)
        h = feedForward2(h)
        h = clip(h, min: -gc, max: gc)
        return normOut(h)
    }
}

// MARK: - Audio tower

/// Gemma 4 `audio_tower`: the USM-Conformer encoder. Consumes log-mel features
/// `[B, T, 128]` (+ padding mask, `true == padding`) and returns encoder outputs
/// `[B, T/4, output_proj_dims]` with padded frames zeroed, plus the subsampled mask.
final class Gemma4AudioModel: Module {
    let config: Gemma4AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") private var subsampleConvProjection:
        Gemma4AudioSubSampleConvProjection
    @ModuleInfo(key: "layers") private var layers: [Gemma4AudioConformerBlock]
    @ModuleInfo(key: "output_proj") private var outputProj: Linear?

    init(config: Gemma4AudioConfiguration) {
        self.config = config
        self._subsampleConvProjection.wrappedValue = Gemma4AudioSubSampleConvProjection(
            config: config)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4AudioConformerBlock(config: config)
        }
        if let outputDims = config.outputProjectionDimensions {
            self._outputProj.wrappedValue = Linear(config.hiddenSize, outputDims, bias: true)
        } else {
            self._outputProj.wrappedValue = nil
        }
        super.init()
    }

    /// Local causal+validity mask `[chunk_size, context_size]` for chunked attention.
    private func buildCausalValidMask() -> MLXArray {
        let w = config.attentionChunkSize
        let maxFuture = config.attentionContextRight
        let maxPast = max(0, config.attentionContextLeft - 1)
        let upperDiagonal = maxPast + maxFuture
        let c = w + maxPast + maxFuture

        let lowerCausal = tril(MLXArray.ones([c, w])).transposed(1, 0)  // [W, C]
        let upperCausal = tril(MLXArray.ones([w, c]), k: upperDiagonal)  // [W, C]
        return (lowerCausal * upperCausal).asType(.bool)
    }

    /// - Parameters:
    ///   - audioMel: `[B, T, 128]` log-mel features.
    ///   - audioMelMask: `[B, T]` boolean, `true == padding`.
    /// - Returns: encoder outputs `[B, T/4, output_proj_dims]` and the subsampled mask.
    func callAsFunction(
        _ audioMel: MLXArray, audioMelMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        var (encodings, currentMask) = subsampleConvProjection(audioMel, mask: audioMelMask)

        let causalValidMask = buildCausalValidMask()
        for block in layers {
            encodings = block(encodings, mask: currentMask, causalValidMask: causalValidMask)
        }

        if let outputProj {
            encodings = outputProj(encodings)
        }

        if currentMask.dim(1) != encodings.dim(1) {
            currentMask = currentMask[0..., ..<encodings.dim(1)]
        }
        encodings = MLX.where(
            expandedDimensions(currentMask, axis: -1), MLXArray(0, dtype: encodings.dtype),
            encodings)
        return (encodings, currentMask)
    }
}
