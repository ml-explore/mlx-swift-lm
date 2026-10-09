// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

private func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

private func gemma4DefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default"),
    ]
}

private func gemma4RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.shape.count - 1] / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func gemma4ApplyMultiDimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float
) -> MLXArray {
    let headDim = inputs.shape[inputs.ndim - 1]
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents =
            (2.0 / Float(headDim)) * MLXArray(0 ..< half).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let sinusoid = positions.asType(.float32).expandedDimensions(axis: -1) / timescale
        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        return inputs * cosValue + gemma4RotateHalf(inputs) * sinValue
    }

    let numDimensions = positions.shape[positions.ndim - 1]
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)

    for d in 0 ..< numDimensions {
        let start = d * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start ..< end]

        let freqExponents =
            (2.0 / Float(channelsPerDimension)) * MLXArray(0 ..< halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, d ..< d + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)

        parts.append(part * cosValue + gemma4RotateHalf(part) * sinValue)
    }

    return concatenated(parts, axis: -1)
}

private func gemma4EnsureFusedSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let fusedDims = [64, 80, 128]
    let d = queries.dim(queries.ndim - 1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d

    if target == d {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    let paddedQueries = MLX.padded(
        queries, widths: [0, 0, 0, .init((0, target - d))])
    let paddedKeys = MLX.padded(
        keys, widths: [0, 0, 0, .init((0, target - d))])
    let paddedValues = MLX.padded(
        values, widths: [0, 0, 0, .init((0, target - d))])

    return MLXFast.scaledDotProductAttention(
        queries: paddedQueries, keys: paddedKeys, values: paddedValues, scale: scale, mask: mask
    )[.ellipsis, ..<d]
}

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_vision"
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 768
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads) ?? 12
        keyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.keyValueHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        defaultOutputLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.defaultOutputLength) ?? 280
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.positionEmbeddingSize) ?? 10_240
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.standardize) ?? false
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: CodingKeys.ropeParameters)
            ?? gemma4DefaultVisionRopeParameters()
    }
}

public struct Gemma4AudioConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let rmsNormEps: Float
    public let outputProjectionDimensions: Int
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let residualWeight: Float
    public let gradientClipping: Float
    public let useClippedLinears: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case outputProjectionDimensions = "output_proj_dims"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case residualWeight = "residual_weight"
        case gradientClipping = "gradient_clipping"
        case useClippedLinears = "use_clipped_linears"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: .outputProjectionDimensions) ?? 1536
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        attentionChunkSize = try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        attentionContextLeft =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        attentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        attentionLogitCap =
            try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50
        residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        gradientClipping =
            try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
        guard hiddenSize > 0, hiddenSize % 2 == 0, hiddenLayers > 0, attentionHeads > 0,
            hiddenSize % attentionHeads == 0, outputProjectionDimensions > 0,
            subsamplingConvChannels.count == 2, subsamplingConvChannels.allSatisfy({ $0 > 0 }),
            subsamplingConvChannels[0] % 4 == 0, convKernelSize > 0,
            attentionChunkSize > 0, attentionContextLeft > 0, attentionContextRight >= 0,
            rmsNormEps.isFinite, rmsNormEps > 0, attentionLogitCap.isFinite, attentionLogitCap > 0,
            gradientClipping.isFinite, gradientClipping > 0, residualWeight.isFinite
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize, in: c,
                debugDescription: "Unsupported Gemma 4 audio encoder layout.")
        }
    }
}

package final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    package init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

package final class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    package init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Vision

final class Gemma4ClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput =
            if let inputMin, let inputMax {
                clip(x, min: inputMin, max: inputMax)
            } else {
                x
            }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax {
            return clip(projected, min: outputMin, max: outputMax)
        }
        return projected
    }
}

final class Gemma4VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * weight.asType(.float32)).asType(x.dtype)
    }
}

final class Gemma4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
    }
}

final class Gemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: Gemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: Gemma4VisionRMSNormNoScale

    init(config: Gemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeParameters["rope_theta"]?.asFloat() ?? 100.0

        self._qProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._kProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._vProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._oProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: numHeads * headDim,
            outFeatures: hiddenSize,
            useClipping: config.useClippedLinears
        )
        self._qNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)
        values = vNorm(values)

        queries = gemma4ApplyMultiDimensionalRoPE(
            queries, positions: positions, baseFrequency: ropeBaseFrequency)
        keys = gemma4ApplyMultiDimensionalRoPE(
            keys, positions: positions, baseFrequency: ropeBaseFrequency)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                .none
            }
        let output = gemma4EnsureFusedSDPA(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

final class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4ClippableLinear

    init(config: Gemma4VisionConfiguration) {
        self._gateProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._upProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._downProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config: config)
        self._mlp.wrappedValue = Gemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attentionOutput)
        let ff = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(ff)
    }
}

final class Gemma4VisionPatchEmbedder: Module {
    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.positionEmbeddingSize = config.positionEmbeddingSize
        self._inputProjection.wrappedValue = Linear(
            3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, positionEmbeddingSize, hiddenSize,
        ])
        super.init()
    }

    private func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let (batch, channels, height, width) = (
            pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3)
        )
        let patchesH = height / patchSize
        let patchesW = width / patchSize

        var patches = pixelValues.reshaped(
            batch, channels, patchesH, patchSize, patchesW, patchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(batch, patchesH * patchesW, channels * patchSize * patchSize)
        patches = 2 * (patches - 0.5)
        return inputProjection(patches.asType(inputProjection.weight.dtype))
    }

    func callAsFunction(
        _ pixelValues: MLXArray, patchPositions: MLXArray
    ) -> MLXArray {
        let hiddenStates = patchify(pixelValues)
        let batch = patchPositions.dim(0)
        let seqLen = patchPositions.dim(1)

        let xIndices = patchPositions[0..., 0..., 0].flattened().asType(.int32)
        let yIndices = patchPositions[0..., 0..., 1].flattened().asType(.int32)
        let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        return hiddenStates + xEmbeddings + yEmbeddings
    }
}

final class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        patchesW: Int,
        outputLength: Int
    ) -> MLXArray {
        let scale = MLXArray(rootHiddenSize, dtype: hiddenStates.dtype)
        let numPatches = hiddenStates.dim(1)
        let length = max(outputLength, 1)
        if numPatches <= length {
            return hiddenStates * scale
        }

        // All batch rows share one position grid, so a single [l, L] weight
        // matrix pools every row. The processor's resize keeps both sides
        // divisible by kernel * patchSize, so the pooled grid covers the
        // image exactly.
        let positions = patchPositions[0]
        let kernel = max(Int(sqrt(Double(numPatches / length))), 1)
        let divisor = kernel * kernel

        let kernelIndices = floor(positions.asType(.float32) / Float(kernel)).asType(.int32)
        let flatKernel =
            kernelIndices[0..., 0] + MLXArray(Int32(max(patchesW / kernel, 1)))
            * kernelIndices[0..., 1]
        let weights =
            gemma4OneHot(flatKernel, numClasses: length).asType(.float32)
            / Float(divisor)
        let output = einsum("lL,bld->bLd", weights, hiddenStates)
            .asType(hiddenStates.dtype)
        return output * scale
    }
}

final class Gemma4VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [Gemma4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4VisionTransformerBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray
    {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

/// The Gemma 4 vision tower: patch embedding, axial-RoPE encoder, and spatial pooling
/// to `numPatches / poolingKernelSize^2` soft tokens per image.
///
/// Public so client modules can run the tower without a generation head — for example
/// multimodal embedding models such as EmbeddingGemma 2, which embed images into the
/// text space. Configure it with ``Gemma4VisionConfiguration``.
public final class Gemma4VisionModel: Module {
    let config: Gemma4VisionConfiguration
    let patchSize: Int
    let poolingKernelSize: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: Gemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    public init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = Gemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = Gemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    private func patchPositions(batch: Int, patchesH: Int, patchesW: Int) -> MLXArray {
        // .xy indexing makes x vary fastest, matching the row-major patch
        // order the embedder and pooler expect.
        let grids = meshGrid([
            MLXArray.arange(patchesW, dtype: .int32),
            MLXArray.arange(patchesH, dtype: .int32),
        ])
        let positions = stacked([grids[0].flattened(), grids[1].flattened()], axis: 1)
            .reshaped(1, patchesH * patchesW, 2)
        return batch == 1
            ? positions
            : broadcast(positions, to: [batch, patchesH * patchesW, 2])
    }

    /// Encodes a batch of same-sized images. Every patch is real (callers
    /// slice padded canvases down to each image's true size first), so
    /// attention is dense and the pooled output length falls out of the
    /// patch grid: numPatches / poolingKernelSize².
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let pixels =
            if pixelValues.ndim == 3 {
                expandedDimensions(pixelValues, axis: 0)
            } else {
                pixelValues
            }
        let batch = pixels.dim(0)
        let patchesH = pixels.dim(2) / patchSize
        let patchesW = pixels.dim(3) / patchSize
        let numPatches = patchesH * patchesW
        let outputLength = max(numPatches / (poolingKernelSize * poolingKernelSize), 1)

        let patchPositions = patchPositions(batch: batch, patchesH: patchesH, patchesW: patchesW)
        var hiddenStates = patchEmbedder(pixels, patchPositions: patchPositions)
        hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: nil)
        hiddenStates = pooler(
            hiddenStates, patchPositions: patchPositions, patchesW: patchesW,
            outputLength: outputLength)

        if let standardizationBias, let standardizationScale {
            hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
        }
        return hiddenStates
    }
}

/// Projects one modality encoder's soft tokens into the text model's embedding space
/// (`embed_vision` / `embed_audio` in Gemma 4 checkpoints, `embed_vision` in
/// EmbeddingGemma 2).
public final class Gemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm:
        Gemma4RMSNormNoScale

    public init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Audio Encoder

/// The Gemma 4 audio encoder, from the Universal Speech Model family: a
/// conformer-style stack of feed-forward blocks, chunked local attention with
/// a relative position bias, and causal light convolutions.
///
/// Inputs are unpadded log-mel spectrograms, `[T, featureSize]` or
/// `[B, T, featureSize]`; the output is `[B, T', outputProjectionDimensions]`,
/// where two stride-2 subsampling convolutions reduce `T` to about a quarter.
public final class Gemma4AudioModel: Module {

    public let config: Gemma4AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") private var subsampleConvProjection:
        Gemma4AudioSubsampleConvProjection
    @ModuleInfo(key: "layers") private var layers: [Gemma4AudioEncoderLayer]
    @ModuleInfo(key: "output_proj") private var outputProjection: Linear

    public init(config: Gemma4AudioConfiguration) {
        self.config = config
        self._subsampleConvProjection.wrappedValue = Gemma4AudioSubsampleConvProjection(config)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4AudioEncoderLayer(config)
        }
        self._outputProjection.wrappedValue = Linear(
            config.hiddenSize, config.outputProjectionDimensions, bias: true)
        super.init()
    }

    /// Encodes log-mel frames into soft tokens.
    public func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        let frames =
            inputFeatures.ndim == 2
            ? inputFeatures.expandedDimensions(axis: 0)
            : inputFeatures
        var hidden = subsampleConvProjection(frames)
        // The reference keeps the sinusoids in the activation dtype.
        let positions = Self.relativePositionalEmbeddings(config).asType(hidden.dtype)
        let mask = Gemma4AudioAttention.visibleKeys(length: hidden.dim(1), config: config)
        for layer in layers {
            hidden = layer(hidden, positions: positions, mask: mask)
        }
        return outputProjection(hidden)
    }

    /// `[sin | cos]` rows for the relative distances `contextSize / 2` down
    /// to `0`, as the reference `Gemma4AudioRelPositionalEncoding` builds them.
    private static func relativePositionalEmbeddings(_ config: Gemma4AudioConfiguration)
        -> MLXArray
    {
        let halfSize = config.hiddenSize / 2
        let contextSize =
            config.attentionChunkSize + config.attentionContextLeft - 1
            + config.attentionContextRight
        let logIncrement = log(10_000.0) / Double(max(halfSize - 1, 1))
        let inverseTimescales = MLXArray(
            (0 ..< halfSize).map { Float(exp(Double($0) * -logIncrement)) })
        let distances = MLXArray(Array((0 ... contextSize / 2).reversed()))
        let angles = distances.expandedDimensions(axis: -1) * inverseTimescales
        return concatenated([sin(angles), cos(angles)], axis: -1)
    }
}

/// Two stride-2 convolutions with channel norms that subsample the mel
/// frames, followed by a linear projection into the encoder's hidden size.
final class Gemma4AudioSubsampleConvProjection: Module {

    @ModuleInfo(key: "layer0") private var layer0: Gemma4AudioSubsampleConvLayer
    @ModuleInfo(key: "layer1") private var layer1: Gemma4AudioSubsampleConvLayer
    @ModuleInfo(key: "input_proj_linear") private var inputProjection: Linear

    init(_ config: Gemma4AudioConfiguration) {
        let channels = config.subsamplingConvChannels
        self._layer0.wrappedValue = Gemma4AudioSubsampleConvLayer(
            inputChannels: 1, outputChannels: channels[0], config: config)
        self._layer1.wrappedValue = Gemma4AudioSubsampleConvLayer(
            inputChannels: channels[0], outputChannels: channels[1], config: config)
        // The reference flattens [mel bins / 4, channels] after its
        // channel-first permute; channels-last convolutions flatten to the
        // same layout.
        self._inputProjection.wrappedValue = Linear(
            (channels[0] / 4) * channels[1], config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ frames: MLXArray) -> MLXArray {
        // One input channel, so the first convolution mixes time and
        // frequency.
        let hidden = layer1(layer0(frames.expandedDimensions(axis: -1)))
        return inputProjection(hidden.reshaped(hidden.dim(0), hidden.dim(1), -1))
    }
}

/// One stride-2 `3x3` convolution with a channel norm and ReLU.
final class Gemma4AudioSubsampleConvLayer: Module {

    @ModuleInfo(key: "conv") private var convolution: Conv2d
    @ModuleInfo(key: "norm") private var norm: Gemma4AudioChannelNorm

    init(inputChannels: Int, outputChannels: Int, config: Gemma4AudioConfiguration) {
        self._convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: false)
        self._norm.wrappedValue = Gemma4AudioChannelNorm(
            dimensions: outputChannels, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm(convolution(x.asType(convolution.weight.dtype))))
    }
}

/// LayerNorm over the last axis with a scale and no bias.
final class Gemma4AudioChannelNorm: Module {

    @ModuleInfo var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: nil, eps: eps)
    }
}

/// Chunked local attention with a relative position bias and a logit soft cap.
///
/// Each chunk of queries reads a window of `contextSize` keys around it; the
/// bias adds the sinusoidal distance terms after the reference's blocked
/// relative shift, and ``visibleKeys(length:config:)`` limits each query to
/// its sliding window inside the sequence.
final class Gemma4AudioAttention: Module {

    @ModuleInfo(key: "q_proj") private var queryProjection: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") private var keyProjection: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") private var valueProjection: Gemma4ClippableLinear
    @ModuleInfo(key: "post") private var outputProjection: Gemma4ClippableLinear
    @ModuleInfo(key: "relative_k_proj") private var relativeKeyProjection: Linear
    @ModuleInfo(key: "per_dim_scale") private var perDimensionScale: MLXArray

    private let heads: Int
    private let headDim: Int
    private let chunkSize: Int
    private let contextSize: Int
    private let pastHorizon: Int
    private let futureHorizon: Int
    private let queryScale: Float
    private let keyScale: Float
    private let softCap: Float

    init(_ config: Gemma4AudioConfiguration) {
        self.heads = config.attentionHeads
        self.headDim = config.hiddenSize / config.attentionHeads
        self.chunkSize = config.attentionChunkSize
        self.pastHorizon = config.attentionContextLeft - 1
        self.futureHorizon = config.attentionContextRight
        self.contextSize = config.attentionChunkSize + pastHorizon + futureHorizon
        // Query and key norms replace the 1/sqrt(headDim) scale.
        self.queryScale = pow(Float(headDim), -0.5) / log(2)
        self.keyScale = log(1 + exp(Float(1))) / log(2)
        self.softCap = config.attentionLogitCap
        self._queryProjection.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._keyProjection.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._valueProjection.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._outputProjection.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._relativeKeyProjection.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false)
        self._perDimensionScale.wrappedValue = MLXArray.zeros([headDim])
        super.init()
    }

    /// Keys each query can see, `[blocks, chunk, context]`: the reference's
    /// sliding window, without the padding outside the sequence.
    static func visibleKeys(length: Int, config: Gemma4AudioConfiguration) -> MLXArray {
        let chunk = config.attentionChunkSize
        let past = config.attentionContextLeft - 1
        let future = config.attentionContextRight
        let blocks = (length + chunk - 1) / chunk
        let row = MLXArray(0 ..< chunk).reshaped(1, chunk, 1)
        let column = MLXArray(0 ..< chunk + past + future).reshaped(1, 1, -1)
        let key = MLXArray(0 ..< blocks).reshaped(blocks, 1, 1) * chunk - past + column
        let distance = row + past - column
        let window =
            ((distance .>= 0) .&& (distance .< past))
            .|| ((distance .< 0) .&& (distance .> -future))
        return window .&& (key .>= 0) .&& (key .< length)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let blockCount = (length + chunkSize - 1) / chunkSize

        // Queries learn a per-dimension scale; values stay unscaled.
        let scale = MLXArray(queryScale) * softplus(perDimensionScale.asType(.float32))
        let queries = queryBlocks(
            queryProjection(x).asType(.float32).reshaped(batch, length, heads, headDim) * scale,
            padTo: blockCount * chunkSize)
        let keys = contextBlocks(
            keyProjection(x).asType(.float32).reshaped(batch, length, heads, headDim)
                * MLXArray(keyScale),
            blockCount: blockCount)
        let values = contextBlocks(
            valueProjection(x).asType(.float32).reshaped(batch, length, heads, headDim),
            blockCount: blockCount)
        let relativeKeys = relativeKeyProjection(positions)
            .reshaped(-1, heads, headDim).transposed(1, 2, 0)

        var scores = matmul(queries, keys.transposed(0, 1, 2, 4, 3))
        scores =
            scores
            + relativeShift(
                matmul(queries.reshaped(batch, heads, -1, headDim), relativeKeys)
                    .reshaped(batch, heads, blockCount, chunkSize, -1))
        scores = tanh(scores / MLXArray(softCap)) * MLXArray(softCap)
        scores = MLX.where(mask, scores, MLXArray(Self.hiddenLogit))

        let weighted = softmax(scores, axis: -1)
        let attended = matmul(weighted, values)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(batch, blockCount * chunkSize, -1)[0..., 0 ..< length, 0...]
        return outputProjection(attended.asType(x.dtype))
    }

    /// Non-overlapping query blocks: `[B, H, blocks, chunk, D]`.
    private func queryBlocks(_ x: MLXArray, padTo: Int) -> MLXArray {
        MLX.padded(x, widths: [0, .init((0, padTo - x.dim(1))), 0, 0])
            .reshaped(x.dim(0), -1, chunkSize, heads, headDim)
            .transposed(0, 3, 1, 2, 4)
    }

    /// Overlapping key and value context windows, strided by the chunk size:
    /// `[B, H, blocks, context, D]`.
    private func contextBlocks(_ x: MLXArray, blockCount: Int) -> MLXArray {
        let padded = MLX.padded(
            x, widths: [0, .init((pastHorizon, futureHorizon + chunkSize - 1)), 0, 0])
        let starts = MLXArray(Array(stride(from: 0, to: blockCount * chunkSize, by: chunkSize)))
            .expandedDimensions(axis: -1)
        let offsets = MLXArray(0 ..< contextSize)
        let indices = (starts + offsets).flattened().asType(.int32)
        return padded.take(indices, axis: 1)
            .reshaped(x.dim(0), blockCount, contextSize, heads, headDim)
            .transposed(0, 3, 1, 2, 4)
    }

    /// Relative position shift for blocked attention (reference `_rel_shift`).
    private func relativeShift(_ x: MLXArray) -> MLXArray {
        let (batch, headCount, blockCount, rows) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flattened = MLX.padded(
            x, widths: [0, 0, 0, 0, .init((0, contextSize + 1 - x.dim(4)))]
        ).reshaped(batch, headCount, blockCount, rows * (contextSize + 1))
        return flattened[0..., 0..., 0..., 0 ..< rows * contextSize]
            .reshaped(batch, headCount, blockCount, rows, contextSize)
    }

    /// `softplus` with PyTorch's overflow threshold.
    private func softplus(_ x: MLXArray) -> MLXArray {
        MLX.where(x .> 20, x, log(1 + exp(x)))
    }

    /// The reference's `attention_invalid_logits_value`.
    private static let hiddenLogit: Float = -1e9
}

/// A gated feed-forward block whose output joins the residual at half weight.
final class Gemma4AudioFeedForward: Module {

    @ModuleInfo(key: "ffw_layer_1") private var expand: Gemma4ClippableLinear
    @ModuleInfo(key: "ffw_layer_2") private var contract: Gemma4ClippableLinear
    @ModuleInfo(key: "pre_layer_norm") private var preNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "post_layer_norm") private var postNorm: Gemma4VisionRMSNorm

    private let clippingLimit: Float
    private let residualWeight: Float

    init(_ config: Gemma4AudioConfiguration) {
        self._expand.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 4,
            useClipping: config.useClippedLinears)
        self._contract.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize * 4, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._preNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.clippingLimit = config.gradientClipping
        self.residualWeight = config.residualWeight
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = clip(x, min: -clippingLimit, max: clippingLimit)
        hidden = preNorm(hidden)
        hidden = contract(silu(expand(hidden)))
        hidden = clip(hidden, min: -clippingLimit, max: clippingLimit)
        return postNorm(hidden) * MLXArray(residualWeight) + x
    }
}

/// A causal depthwise convolution between two gated linears.
final class Gemma4AudioLightConv1d: Module {

    @ModuleInfo(key: "linear_start") private var linearStart: Gemma4ClippableLinear
    @ModuleInfo(key: "linear_end") private var linearEnd: Gemma4ClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") private var depthwiseConvolution: Conv1d
    @ModuleInfo(key: "pre_layer_norm") private var preNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "conv_norm") private var convNorm: Gemma4VisionRMSNorm

    private let clippingLimit: Float
    private let leftPad: Int

    init(_ config: Gemma4AudioConfiguration) {
        self._linearStart.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 2,
            useClipping: config.useClippedLinears)
        self._linearEnd.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize, outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears)
        self._depthwiseConvolution.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            groups: config.hiddenSize,
            bias: false)
        self._preNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._convNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.clippingLimit = config.gradientClipping
        self.leftPad = config.convKernelSize - 1
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = linearStart(preNorm(x))
        let half = hidden.dim(2) / 2
        let gated = hidden[0..., 0..., 0 ..< half] * sigmoid(hidden[0..., 0..., half...])
        hidden = depthwiseConvolution(MLX.padded(gated, widths: [0, .init((leftPad, 0)), 0]))
        hidden = clip(hidden, min: -clippingLimit, max: clippingLimit)
        return linearEnd(silu(convNorm(hidden))) + x
    }
}

/// One conformer block: two feed-forward sub-blocks around attention and a
/// light convolution, closing on the output norm.
final class Gemma4AudioEncoderLayer: Module {

    @ModuleInfo(key: "feed_forward1") private var feedForward1: Gemma4AudioFeedForward
    @ModuleInfo(key: "feed_forward2") private var feedForward2: Gemma4AudioFeedForward
    @ModuleInfo(key: "self_attn") private var selfAttention: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") private var lightConvolution: Gemma4AudioLightConv1d
    @ModuleInfo(key: "norm_pre_attn") private var preAttentionNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "norm_post_attn") private var postAttentionNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "norm_out") private var outputNorm: Gemma4VisionRMSNorm

    private let clippingLimit: Float

    init(_ config: Gemma4AudioConfiguration) {
        self._feedForward1.wrappedValue = Gemma4AudioFeedForward(config)
        self._feedForward2.wrappedValue = Gemma4AudioFeedForward(config)
        self._selfAttention.wrappedValue = Gemma4AudioAttention(config)
        self._lightConvolution.wrappedValue = Gemma4AudioLightConv1d(config)
        self._preAttentionNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._outputNorm.wrappedValue = Gemma4VisionRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.clippingLimit = config.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = feedForward1(x)
        let residual = hidden
        hidden = clip(hidden, min: -clippingLimit, max: clippingLimit)
        hidden = selfAttention(preAttentionNorm(hidden), positions: positions, mask: mask)
        hidden = clip(hidden, min: -clippingLimit, max: clippingLimit)
        hidden = postAttentionNorm(hidden) + residual
        hidden = lightConvolution(hidden)
        hidden = feedForward2(hidden)
        return outputNorm(clip(hidden, min: -clippingLimit, max: clippingLimit))
    }
}
