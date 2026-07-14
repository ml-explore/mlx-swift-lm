import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

private enum Gemma4Error: LocalizedError {
    case imageTokenCountMismatch(expectedVisionTokens: Int, actualPromptTokens: Int)
    case multimodalTokenCountMismatch(kind: String, featureTokens: Int, promptTokens: Int)

    var errorDescription: String? {
        switch self {
        case .imageTokenCountMismatch(let expectedVisionTokens, let actualPromptTokens):
            return
                "Gemma4 image token count mismatch: vision encoder produced \(expectedVisionTokens) soft tokens, but the prompt contains \(actualPromptTokens) image tokens."
        case .multimodalTokenCountMismatch(let kind, let featureTokens, let promptTokens):
            return
                "Gemma4 \(kind) token count mismatch: encoder produced \(featureTokens) soft tokens, but the prompt contains \(promptTokens) \(kind) tokens."
        }
    }
}

private func gemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4DefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default"),
    ]
}

private func gemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.dim(0) == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.dim(0)) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

private func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

/// Average-pool kernel for Gemma 4's vision pooler.
///
/// The padded patch tensor has length
/// `paddedPatchCount = outputLength × pool²` where `pool` is the
/// model's `pooling_kernel_size`. Recovering `pool` from these
/// two values yields `floor(sqrt(paddedPatchCount / outputLength))`.
///
/// Matches HuggingFace's reference image processor (see
/// `image_processing_gemma4.py`: `max_patches = max_soft_tokens *
/// pooling_kernel_size**2`).
internal func gemma4VisionPoolingKernel(
    paddedPatchCount: Int, outputLength: Int
) -> Int {
    let safeLength = max(outputLength, 1)
    let ratio = max(1, paddedPatchCount / safeLength)
    return Int(sqrt(Double(ratio)))
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

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        let maskLength = maskArray.dim(-1)
        guard maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}

private func gemma4TokenTypeIds(
    inputIds: MLXArray,
    imageTokenId: Int?,
    videoTokenId: Int?,
    audioTokenId: Int?
) -> MLXArray {
    var tokenTypeIds = MLXArray.zeros(like: inputIds).asType(.int32)
    if let imageTokenId {
        tokenTypeIds = MLX.where(inputIds .== imageTokenId, MLXArray(1), tokenTypeIds)
    }
    if let videoTokenId {
        tokenTypeIds = MLX.where(inputIds .== videoTokenId, MLXArray(2), tokenTypeIds)
    }
    if let audioTokenId {
        tokenTypeIds = MLX.where(inputIds .== audioTokenId, MLXArray(3), tokenTypeIds)
    }
    return tokenTypeIds
}

private func gemma4TextOnlyPromptTokens(_ input: LMInput) -> MLXArray {
    let tokens = input.text.tokens
    if tokens.ndim == 2, tokens.dim(0) == 1 {
        return tokens[0]
    }
    if tokens.ndim == 1 {
        return tokens
    }
    return tokens.flattened()
}

private func gemma4PrepareTextOnly(
    _ input: LMInput,
    cache: [any KVCache],
    windowSize: Int?,
    languageModel: Gemma4TextLanguageModel
) -> PrepareResult {
    let prefillStepSize = max(windowSize ?? 512, 1)
    let y = gemma4TextOnlyPromptTokens(input).expandedDimensions(axis: 0)
    let convertedCache = cache.map { $0 }
    let totalPositions = y.dim(1)

    var processed = 0
    while totalPositions - processed > 1 {
        let chunkLength = min(prefillStepSize, totalPositions - processed - 1)
        _ = languageModel(
            y[0..., processed ..< (processed + chunkLength)],
            cache: convertedCache
        )
        asyncEval(cache)
        processed += chunkLength
    }

    eval(cache)
    return .logits(languageModel(y[0..., processed...], cache: convertedCache))
}

private func gemma4BlockSequenceIdsForMask(_ tokenTypeIds: MLXArray) -> MLXArray {
    let isVision = (tokenTypeIds .== 1) | (tokenTypeIds .== 2)
    let sequenceLength = isVision.dim(1)
    guard sequenceLength > 0 else {
        return MLXArray.zeros(like: tokenTypeIds).asType(.int32) - 1
    }

    let previous =
        if sequenceLength == 1 {
            MLXArray.zeros(like: isVision)
        } else {
            concatenated(
                [
                    MLXArray.zeros(like: isVision[0..., ..<1]),
                    isVision[0..., ..<(sequenceLength - 1)],
                ],
                axis: 1
            )
        }
    let starts = isVision & logicalNot(previous)
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray.zeros(like: groupIds) - 1)
}

private func gemma4ApplyBlockwiseBidirectionalOverlay(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tokenTypeIds: MLXArray,
    sequenceLength: Int,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let baseMask: MLXArray
    switch mask {
    case .array(let array):
        baseMask = array
    case .arrays(let arrays):
        guard let first = arrays.first else {
            return mask
        }
        baseMask = first
    case .causal:
        baseMask = createCausalMask(n: sequenceLength, offset: 0, windowSize: windowSize)
    case .none:
        baseMask = MLXArray.ones([sequenceLength, sequenceLength], dtype: .bool)
    }

    guard tokenTypeIds.dim(1) == baseMask.dim(-1) else {
        return mask
    }

    let blockSequenceIds = gemma4BlockSequenceIdsForMask(tokenTypeIds)
    let queryBlocks = expandedDimensions(blockSequenceIds, axis: -1)
    let keyBlocks = expandedDimensions(blockSequenceIds, axis: -2)
    let sameBlock = (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
    return .array(baseMask | expandedDimensions(sameBlock, axis: 1))
}

private func gemma4CompactPrefixRows(features: MLXArray, validMask: MLXArray) -> MLXArray {
    let maskRows = validMask.asArray(Bool.self)
    let batch = validMask.dim(0)
    let length = validMask.dim(1)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for batchIdx in 0 ..< batch {
        let start = batchIdx * length
        let count = maskRows[start ..< start + length].reduce(0) { $0 + ($1 ? 1 : 0) }
        if count > 0 {
            rows.append(features[batchIdx, ..<count, 0...])
        }
    }

    guard !rows.isEmpty else {
        return features.reshaped(-1, features.dim(-1))[..<0, 0...]
    }
    return concatenated(rows, axis: 0)
}
// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let useBidirectionalAttention: String?
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers)
            ?? (isUnified ? 48 : 35)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads)
            ?? (isUnified ? 16 : 8)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? (isUnified ? 8 : 1)
        globalKVHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
            ?? (isUnified ? 1 : nil)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers)
            ?? (isUnified ? 0 : 20)
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow)
            ?? (isUnified ? 1024 : 512)
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern)
            ?? (isUnified ? 6 : 5)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? isUnified
        useBidirectionalAttention =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.useBidirectionalAttention)
            ?? (isUnified ? "vision" : nil)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? (isUnified
                ? [
                    "full_attention": [
                        "partial_rotary_factor": .float(0.25),
                        "rope_theta": .float(1_000_000.0),
                        "rope_type": .string("proportional"),
                    ],
                    "sliding_attention": [
                        "rope_theta": .float(10_000.0),
                        "rope_type": .string("default"),
                    ],
                ] : gemma4DefaultTextRopeParameters())
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? gemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
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

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
        visionConfiguration = try c.decode(
            Gemma4VisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage)
            ?? visionConfiguration.defaultOutputLength
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}

// MARK: - Text

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4RMSNormZeroShift: Module, UnaryLayer {
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

final class Gemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4TextRouter: Module {
    let topKExperts: Int
    let config: Gemma4TextConfiguration
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: config.rmsNormEps)

        let scores = proj(normed)

        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

final class Gemma4ScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
/// Use `kvSharedOnly: true` in the constructor to skip building local K/V
/// projections (the drafter consumes the target's K/V via `sharedKV` instead).
final class Gemma4TextAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale?
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        if !kvSharedOnly {
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
            if !useKEqV {
                self._vProj.wrappedValue = Linear(
                    config.hiddenSize, numKVHeads * headDim, bias: false)
            }
            self._kNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: headDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            // Non-`kvSharedOnly` path: K/V projections must be present. If they
            // are nil here the layer was built with `kvSharedOnly: true` and the
            // caller forgot to pass `sharedKV` — a configuration bug.
            guard let kProj, let kNorm, let vNorm else {
                fatalError(
                    "Gemma4 attention called without sharedKV on a kvSharedOnly layer")
            }
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma4 attention expected a KV state")
        }
        let localMask = gemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = Gemma4TextAttention(
            config: config, layerIdx: layerIdx, kvSharedOnly: kvSharedOnly)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config: config)
            self._experts.wrappedValue = Gemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (for target-side
/// `embed_tokens` / `embed_scale` / `layer_types` access during drafter bind).
final class Gemma4TextBackbone: Module {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        // KV-shared tail layers (`layer_idx >= hidden_layers - num_kv_shared_layers`)
        // reuse an earlier same-type layer's K/V and own no local K/V projection or
        // K norm. Build them with `kvSharedOnly: true` so the module tree omits those
        // tensors — matching what `sanitize` drops from the checkpoint, and mirroring
        // both `Gemma4AssistantDraftInner` and upstream mlx-lm (`Attention.has_kv`).
        // Without this, shared layers declare a `v_proj`/`k_proj`/`k_norm` that no
        // weight fills, so loading a Gemma 4 checkpoint fails under the strict loader
        // verify with e.g. `keyNotFound(… layers.24.self_attn.v_proj.weight …)`.
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { idx in
            Gemma4TextDecoderLayer(
                config: config, layerIdx: idx,
                kvSharedOnly: firstKVSharedLayer > 0 && idx >= firstKVSharedLayer)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds) * perLayerProjectionScale
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        emitDrafterState: Bool = false
    ) -> (hidden: MLXArray, sharedKV: [String: (MLXArray, MLXArray)]?) {
        // Tolerate callers that hand us a 1D `(L,)` token array instead
        // of the canonical 2D `(B, L)` produced by `Gemma4Processor.prepare`.
        // The downstream `perLayerInputs` indexing path (`finalPerLayerInputs[
        // 0..., 0..., idx, 0...]`) requires 4D shapes; with 1D inputs the
        // model otherwise crashes inside `MLXArray.subscript.getter`
        // → `mlx_array_dim` → `_mlx_error`. This expansion is zero-copy
        // and behaves identically when the caller already passed 2D.
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let inputsEmbeds = inputsEmbeds.map {
            $0.ndim == 2 ? $0.expandedDimensions(axis: 0) : $0
        }

        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let hasExplicitCache = cache != nil
        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            let tokenTypeIds = tokenTypeIds.map {
                $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0
            }
            let hasAudioTokens =
                if let tokenTypeIds {
                    (tokenTypeIds .== 3).asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let hasVisualTokens =
                if let tokenTypeIds {
                    ((tokenTypeIds .== 1) | (tokenTypeIds .== 2))
                        .asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let useBidirectionalVision =
                config.useBidirectionalAttention == "vision"
                && tokenTypeIds != nil
                && hasVisualTokens
                && !hasAudioTokens
                && h0.dim(1) > 1

            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil,
                returnArray: useBidirectionalVision)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow,
                returnArray: useBidirectionalVision
            )
            if useBidirectionalVision, let tokenTypeIds {
                fullMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    fullMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: nil
                )
                slidingMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    slidingMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: config.slidingWindow
                )
            }
        }

        var h = h0
        var intermediates = [(kv: Gemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            let (output, kvState, attentionOffset) = layer(
                h,
                mask: layerMask,
                cache: layerCache,
                perLayerInput: layerInput,
                sharedKV: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].kv : nil,
                offset: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].offset : nil
            )
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        let finalHidden = norm(h)

        guard emitDrafterState else {
            return (finalHidden, nil)
        }

        // Walk intermediates from the last layer backward; for each unique
        // `layer_type`, take the first `.regular` K/V encountered. Quantized
        // cases are skipped — the iterator treats absent `sharedKV` as a
        // signal to fall back to single-token generation (R8/R13 limitation,
        // documented).
        var sharedKV: [String: (MLXArray, MLXArray)] = [:]
        var seenTypes = Set<String>()
        let targetTypes: Set<String> = ["full_attention", "sliding_attention"]
        for idx in stride(from: layers.count - 1, through: 0, by: -1) {
            let layerType = layers[idx].layerType
            guard targetTypes.contains(layerType), !seenTypes.contains(layerType) else {
                continue
            }
            if case .regular(let keys, let values) = intermediates[idx].kv {
                sharedKV[layerType] = (keys, values)
                seenTypes.insert(layerType)
            }
            if seenTypes == targetTypes { break }
        }
        // Treat partial coverage (e.g. only one layer_type populated, or
        // quantized cache for the other) as no-emit — iterator falls back.
        return (finalHidden, seenTypes == targetTypes ? sharedKV : nil)
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (the MTP drafter
/// reaches `embed_tokens` / `embed_scale` / `config.layer_types` through this).
final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: Gemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map {
            layerType in
            if layerType == "full_attention" {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: slidingWindow, keep: 0)
            }
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        emitDrafterState: Bool = false
    ) -> LMOutput {
        let (hidden, sharedKV) = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            emitDrafterState: emitDrafterState
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.embedTokens.asLinear(hidden)
        }
        let softcappedLogits: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softcappedLogits = tanh(logits / scale) * scale
        } else {
            softcappedLogits = logits
        }

        guard emitDrafterState, let sharedKV else {
            return LMOutput(logits: softcappedLogits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = hidden
        state[mtpSharedKVStatesKey] = sharedKV
        return LMOutput(logits: softcappedLogits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight.
            // Scope: text backbone only — the vision/audio towers share the
            // `layers.N.self_attn.{k,v}_proj` naming, so without these guards the drop
            // would amputate tower layers >= firstKVSharedLayer.
            if firstKVSharedLayer > 0,
                !key.contains("vision_tower"),
                !key.contains("audio_tower"),
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKVSharedLayer
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}

// MARK: - Vision

private final class Gemma4ClippableLinear: Module, UnaryLayer {
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

private final class Gemma4VisionRMSNorm: Module, UnaryLayer {
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

private final class Gemma4VisionRMSNormNoScale: Module, UnaryLayer {
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

private final class Gemma4VisionAttention: Module {
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

private final class Gemma4VisionMLP: Module, UnaryLayer {
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

private final class Gemma4VisionTransformerBlock: Module {
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

private final class Gemma4VisionPatchEmbedder: Module {
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

private final class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.defaultOutputLength = config.defaultOutputLength
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        validCount: Int,
        outputLength: Int? = nil
    ) -> MLXArray {
        let paddingPositions = patchPositions[0..., 0..., 0] .< 0
        let pooledHiddenStates = MLX.where(
            expandedDimensions(paddingPositions, axis: -1),
            MLXArray(0.0, dtype: hiddenStates.dtype),
            hiddenStates
        )
        let length = outputLength ?? defaultOutputLength
        if pooledHiddenStates.dim(1) <= length {
            return pooledHiddenStates * MLXArray(rootHiddenSize, dtype: pooledHiddenStates.dtype)
        }

        let actualPositions = patchPositions[0, ..<validCount]
        let maxX = Int(actualPositions[0..., 0].max().item(Int32.self)) + 1
        let kernel = gemma4VisionPoolingKernel(
            paddedPatchCount: pooledHiddenStates.dim(1), outputLength: length)
        let divisor = max(kernel * kernel, 1)
        let pooledLength = max(length, 1)

        var kernelIndices = actualPositions.asType(.int32)
        kernelIndices = floor(kernelIndices.asType(.float32) / Float(kernel)).asType(.int32)
        let flatKernel =
            kernelIndices[0..., 0] + MLXArray(Int32(max(maxX / max(kernel, 1), 1)))
            * kernelIndices[0..., 1]
        let weights =
            gemma4OneHot(flatKernel, numClasses: pooledLength).asType(.float32)
            / Float(divisor)
        let output = einsum(
            "lL,bld->bLd", weights, pooledHiddenStates[0..., ..<validCount, 0...]
        )
        .asType(pooledHiddenStates.dtype)
        return output * MLXArray(rootHiddenSize, dtype: pooledHiddenStates.dtype)
    }
}

private final class Gemma4VisionTransformerModel: Module {
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

private final class Gemma4VisionModel: Module {
    let config: Gemma4VisionConfiguration
    let patchSize: Int
    let defaultOutputLength: Int
    let poolingKernelSize: Int
    let maxPatches: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: Gemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.defaultOutputLength = config.defaultOutputLength
        self.poolingKernelSize = config.poolingKernelSize
        self.maxPatches =
            config.defaultOutputLength * config.poolingKernelSize * config.poolingKernelSize
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = Gemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = Gemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    private func patchPositions(batch: Int, height: Int, width: Int) -> (MLXArray, Int) {
        let patchesH = height / patchSize
        let patchesW = width / patchSize
        let realCount = patchesH * patchesW
        let paddedCount = max(maxPatches - realCount, 0)

        var values = [Int32]()
        values.reserveCapacity(batch * (realCount + paddedCount) * 2)

        for _ in 0 ..< batch {
            for y in 0 ..< patchesH {
                for x in 0 ..< patchesW {
                    values.append(Int32(x))
                    values.append(Int32(y))
                }
            }
            for _ in 0 ..< paddedCount {
                values.append(-1)
                values.append(-1)
            }
        }

        let count = realCount + paddedCount
        return (MLXArray(values, [batch, count, 2]), realCount)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let pixels =
            if pixelValues.ndim == 3 {
                expandedDimensions(pixelValues, axis: 0)
            } else {
                pixelValues
            }
        let batch = pixels.dim(0)
        let height = pixels.dim(2)
        let width = pixels.dim(3)
        let (patchPositions, realCount) = patchPositions(batch: batch, height: height, width: width)

        let realPositions = patchPositions[0..., ..<realCount, 0...]
        var hiddenStates = patchEmbedder(pixels, patchPositions: realPositions)

        let paddingCount = maxPatches - realCount
        if paddingCount > 0 {
            let pad = MLXArray.zeros(
                [batch, paddingCount, hiddenStates.dim(2)], dtype: hiddenStates.dtype)
            hiddenStates = concatenated([hiddenStates, pad], axis: 1)
        }

        let validMask = patchPositions[0..., 0..., 0] .>= 0
        var attentionMask =
            expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
        attentionMask = MLX.where(
            attentionMask,
            MLXArray(0.0, dtype: hiddenStates.dtype),
            MLXArray(-Float.infinity, dtype: hiddenStates.dtype)
        )
        attentionMask = expandedDimensions(attentionMask, axis: 1)

        hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: attentionMask)
        hiddenStates = pooler(hiddenStates, patchPositions: patchPositions, validCount: realCount)

        if let standardizationBias, let standardizationScale {
            hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
        }
        return hiddenStates
    }
}

private final class Gemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm:
        Gemma4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Model

public final class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    /// Module-internal — also reached by `Gemma4Assistant.swift` (drafter `bind()`
    /// walks here to cache the target's input embeddings, embed scale, and
    /// per-layer type metadata).
    @ModuleInfo(key: "language_model") var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    public let config: Gemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            let imageMask = inputIds .== config.imageTokenId
            let audioMask =
                if let audioTokenId = config.audioTokenId {
                    inputIds .== audioTokenId
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let textMask = logicalNot(logicalOr(imageMask, audioMask))
            let perLayerTokens = MLX.where(textMask, inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        guard let pixelValues else {
            return (inputsEmbeds, perLayerInputs)
        }

        var imageFeatures = visionTower(pixelValues)
        imageFeatures = embedVision(imageFeatures)
        imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)

        let imageMask = inputIds .== config.imageTokenId
        let expectedImageTokens = imageMask.asType(.int32).sum().item(Int.self)

        if expectedImageTokens != imageFeatures.dim(1) {
            throw Gemma4Error.imageTokenCountMismatch(
                expectedVisionTokens: imageFeatures.dim(1), actualPromptTokens: expectedImageTokens)
        }

        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
        inputsEmbeds = gemma4MaskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: imageFeatures
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, windowSize: Int?
    ) throws
        -> PrepareResult
    {
        let convertedCache = cache.map { $0 }
        if let imagePixels = input.image?.pixels {
            let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
                inputIds: input.text.tokens, pixelValues: imagePixels)
            let result = languageModel(
                nil,
                cache: convertedCache,
                inputsEmbeds: inputsEmbeds,
                perLayerInputs: perLayerInputs,
                tokenTypeIds: gemma4TokenTypeIds(
                    inputIds: input.text.tokens,
                    imageTokenId: config.imageTokenId,
                    videoTokenId: nil,
                    audioTokenId: config.audioTokenId)
            )
            return .logits(result)
        } else {
            return gemma4PrepareTextOnly(
                input, cache: convertedCache, windowSize: windowSize, languageModel: languageModel)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `Gemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = languageModel.sanitize(weights: weights)

        // This port currently supports text + vision only.
        sanitized = sanitized.filter { key, _ in
            !key.contains("audio_tower") && !key.contains("embed_audio")
        }

        if !config.visionConfiguration.useClippedLinears {
            sanitized = sanitized.filter { key, _ in
                !key.contains("input_min")
                    && !key.contains("input_max")
                    && !key.contains("output_min")
                    && !key.contains("output_max")
            }
        }

        return sanitized
    }
}

// MARK: - Gemma 4 Unified

public struct Gemma4UnifiedAudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioSamplesPerToken: Int
    public let audioEmbedDim: Int
    public let hiddenSize: Int
    public let outputProjectionDimensions: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioSamplesPerToken = "audio_samples_per_token"
        case audioEmbedDim = "audio_embed_dim"
        case hiddenSize = "hidden_size"
        case outputProjectionDimensions = "output_proj_dims"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_audio"
        audioSamplesPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSamplesPerToken) ?? 640
        audioEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioEmbedDim) ?? 640
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 640
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 640
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
    }
}

public struct Gemma4UnifiedVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let mmEmbedDim: Int
    public let mmPositionEmbeddingSize: Int
    public let numSoftTokens: Int
    public let rmsNormEps: Float
    public let outputProjectionDimensions: Int

    public var hiddenSize: Int { outputProjectionDimensions }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case mmEmbedDim = "mm_embed_dim"
        case mmPositionEmbeddingSize = "mm_posemb_size"
        case numSoftTokens = "num_soft_tokens"
        case rmsNormEps = "rms_norm_eps"
        case outputProjectionDimensions = "output_proj_dims"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_vision"
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? patchSize * poolingKernelSize
        mmEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmEmbedDim) ?? 3840
        mmPositionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmPositionEmbeddingSize) ?? 1120
        numSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens) ?? 280
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 3840
    }
}

public struct Gemma4UnifiedConfiguration: Decodable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4UnifiedVisionConfiguration?
    public let audioConfiguration: Gemma4UnifiedAudioConfiguration?
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let boaTokenId: Int
    public let eoaTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let audioSoftTokensPerImage: Int
    public let audioMsPerToken: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case audioMsPerToken = "audio_ms_per_token"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration =
            try c.decodeIfPresent(
                Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
            ?? Gemma4TextConfiguration(from: decoder)
        visionConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedVisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        audioConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedAudioConfiguration.self, forKey: CodingKeys.audioConfiguration)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_unified"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boaTokenId) ?? 256_000
        eoaTokenId =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenId)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenIndex)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage) ?? 280
        visionSoftTokensPerVideoFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerVideoFrame) ?? 70
        audioSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSoftTokensPerImage) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}

private final class Gemma4UnifiedVisionEmbedder: Module {
    let patchDim: Int

    @ModuleInfo(key: "patch_ln1") var patchLayerNorm1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLayerNorm2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var positionNorm: LayerNorm

    init(config: Gemma4UnifiedVisionConfiguration) {
        self.patchDim = config.modelPatchSize * config.modelPatchSize * 3
        self._patchLayerNorm1.wrappedValue = LayerNorm(dimensions: patchDim)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim)
        self._patchLayerNorm2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        self._positionEmbedding.wrappedValue = MLXArray.zeros([
            config.mmPositionEmbeddingSize, 2, config.mmEmbedDim,
        ])
        self._positionNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray, imagePositionIds: MLXArray? = nil) -> MLXArray {
        var pixels = pixelValues
        if pixels.ndim == 4 && pixels.dim(-1) == patchDim {
            pixels = pixels.reshaped(pixels.dim(0), -1, patchDim)
        }

        var hiddenStates = patchLayerNorm1(pixels)
        hiddenStates = patchDense(hiddenStates)
        hiddenStates = patchLayerNorm2(hiddenStates)

        if let imagePositionIds {
            let clamped = maximum(imagePositionIds, MLXArray(0)).asType(.int32)
            let batch = clamped.dim(0)
            let sequenceLength = clamped.dim(1)
            let xIndices = clamped[0..., 0..., 0].flattened()
            let yIndices = clamped[0..., 0..., 1].flattened()
            let xEmbeddings = take(positionEmbedding[0..., 0], xIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let yEmbeddings = take(positionEmbedding[0..., 1], yIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let valid = (imagePositionIds .!= -1).asType(hiddenStates.dtype)
            hiddenStates =
                hiddenStates
                + xEmbeddings * expandedDimensions(valid[0..., 0..., 0], axis: -1)
                + yEmbeddings * expandedDimensions(valid[0..., 0..., 1], axis: -1)
        }

        return positionNorm(hiddenStates)
    }
}

public final class Gemma4Unified: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "vision_embedder") private var visionEmbedder: Gemma4UnifiedVisionEmbedder?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    public let config: Gemma4UnifiedConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: Gemma4UnifiedConfiguration) {
        self.config = config
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        if let visionConfiguration = config.visionConfiguration {
            self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(
                config: visionConfiguration)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: visionConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: visionConfiguration.rmsNormEps
            )
        }
        if let audioConfiguration = config.audioConfiguration {
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audioConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: audioConfiguration.rmsNormEps
            )
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func getImageFeatures(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray?
    ) throws -> MLXArray {
        guard let visionEmbedder, let embedVision else {
            throw VLMError.processing("Vision inputs were provided, but vision_config is missing.")
        }
        var projected = embedVision(visionEmbedder(pixelValues, imagePositionIds: imagePositionIds))
        if let imagePositionIds {
            let validMask =
                (imagePositionIds[0..., 0..., 0] .!= -1)
                | (imagePositionIds[0..., 0..., 1] .!= -1)
            projected = gemma4CompactPrefixRows(features: projected, validMask: validMask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func getAudioFeatures(features: MLXArray, mask: MLXArray?) throws -> MLXArray {
        guard let embedAudio else {
            throw VLMError.processing("Audio inputs were provided, but audio_config is missing.")
        }
        var projected = embedAudio(features)
        if let mask {
            projected = gemma4CompactPrefixRows(features: projected, validMask: mask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func scatterFeatures(
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        tokenId: Int?,
        features: MLXArray?,
        kind: String
    ) throws -> MLXArray {
        guard let tokenId, let features else {
            return inputEmbeds
        }
        let tokenMask = inputIds .== tokenId
        let expectedTokens = tokenMask.asType(.int32).sum().item(Int.self)
        guard expectedTokens == features.dim(0) else {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: kind, featureTokens: features.dim(0), promptTokens: expectedTokens)
        }
        var expandedMask = expandedDimensions(tokenMask, axis: -1)
        expandedMask = broadcast(expandedMask, to: inputEmbeds.shape)
        return gemma4MaskedScatter(
            inputTensor: inputEmbeds,
            mask: expandedMask,
            source: features.asType(inputEmbeds.dtype)
        )
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        imagePositionIds: MLXArray? = nil,
        videoPixelValues: MLXArray? = nil,
        videoPositionIds: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            var multimodalMask = inputIds .== config.imageTokenId
            if let audioTokenId = config.audioTokenId {
                multimodalMask = multimodalMask | (inputIds .== audioTokenId)
            }
            if let videoTokenId = config.videoTokenId {
                multimodalMask = multimodalMask | (inputIds .== videoTokenId)
            }
            let perLayerTokens = MLX.where(
                logicalNot(multimodalMask), inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        let imageFeatures =
            if let pixelValues {
                try getImageFeatures(pixelValues: pixelValues, imagePositionIds: imagePositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.imageTokenId,
            features: imageFeatures,
            kind: "image"
        )

        let videoFeatures =
            if let videoPixelValues {
                try getImageFeatures(
                    pixelValues: videoPixelValues, imagePositionIds: videoPositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.videoTokenId,
            features: videoFeatures,
            kind: "video"
        )

        let audioFeatures =
            if let audioFeatures {
                try getAudioFeatures(features: audioFeatures, mask: audioMask)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.audioTokenId,
            features: audioFeatures,
            kind: "audio"
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, windowSize: Int?
    ) throws
        -> PrepareResult
    {
        if input.image == nil, input.video == nil, input.audio == nil {
            return gemma4PrepareTextOnly(
                input, cache: cache, windowSize: windowSize, languageModel: languageModel)
        }

        let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            imagePositionIds: input.image?.positionIds,
            videoPixelValues: input.video?.pixels,
            videoPositionIds: input.video?.positionIds,
            audioFeatures: input.audio?.features,
            audioMask: input.audio?.mask
        )
        let tokenTypeIds = gemma4TokenTypeIds(
            inputIds: input.text.tokens,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            audioTokenId: config.audioTokenId
        )
        let result = languageModel(
            nil,
            cache: cache.map { $0 },
            inputsEmbeds: inputsEmbeds,
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds
        )
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "lm_head.weight" {
                continue
            }
            if embedAudio == nil && key.contains("embed_audio") {
                continue
            }
            if visionEmbedder == nil
                && (key.contains("vision_embedder") || key.contains("embed_vision"))
            {
                continue
            }
            filtered[key] = value
        }
        return languageModel.sanitize(weights: filtered)
    }
}

// MARK: - Processor

public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary: MLXLMCommon.Message
        if message.role == .system {
            dictionary = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
        } else {
            dictionary = [
                "role": message.role.rawValue,
                "content": message.images.map { _ in
                    ["type": "image"]
                }
                    + message.videos.map { _ in
                        ["type": "video"]
                    }
                    + [
                        ["type": "text", "text": message.content]
                    ],
            ]
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        var userProcessing = processing ?? UserInput.Processing()
        let targetSize = config.fixedSize
        userProcessing.resize = targetSize

        let processedImages = images.map { image in
            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
            let finalImage =
                if config.doNormalize {
                    MediaProcessing.normalize(
                        resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
                } else {
                    resizedImage
                }
            return MediaProcessing.asMLXArray(finalImage)
        }

        let pixelValues = concatenated(processedImages)

        return (pixelValues, THW(images.count, Int(targetSize.height), Int(targetSize.width)))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )

            var expandedTokens: [Int] = []
            for token in promptTokens {
                if token == config.imageTokenId {
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: config.imageTokenId, count: config.imageSeqLength))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?

    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case size
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decode(String.self, forKey: CodingKeys.processorClass)
        doNormalize = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize) ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean) ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd) ?? [0.5, 0.5, 0.5]
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength) ?? 280
        size = try c.decodeIfPresent(
            Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.size)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        if let size {
            return CGSize(width: size.width, height: size.height)
        }
        // 800x800 keeps the patch count under Gemma4's 280 * 3^2 vision budget.
        return CGSize(width: 800, height: 800)
    }
}

public struct Gemma4UnifiedProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let doResize: Bool
    public let doRescale: Bool
    public let rescaleFactor: CGFloat
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    public let audioMsPerToken: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let maxSoftTokens: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    private struct ImageProcessorConfiguration: Decodable, Sendable {
        let doResize: Bool?
        let doRescale: Bool?
        let rescaleFactor: CGFloat?
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let modelPatchSize: Int?
        let maxSoftTokens: Int?
        let numSoftTokens: Int?
        let size: Gemma3ProcessorConfiguration.ImageSize?

        enum CodingKeys: String, CodingKey {
            case doResize = "do_resize"
            case doRescale = "do_rescale"
            case rescaleFactor = "rescale_factor"
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case modelPatchSize = "model_patch_size"
            case maxSoftTokens = "max_soft_tokens"
            case numSoftTokens = "num_soft_tokens"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case rescaleFactor = "rescale_factor"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case audioMsPerToken = "audio_ms_per_token"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case numSoftTokens = "num_soft_tokens"
        case size
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)

        processorClass =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.processorClass)
            ?? "Gemma4UnifiedProcessor"
        doResize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doResize)
            ?? imageProcessor?.doResize
            ?? true
        doRescale =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doRescale)
            ?? imageProcessor?.doRescale
            ?? true
        rescaleFactor =
            try c.decodeIfPresent(CGFloat.self, forKey: CodingKeys.rescaleFactor)
            ?? imageProcessor?.rescaleFactor
            ?? (1.0 / 255.0)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? imageProcessor?.doNormalize
            ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? imageProcessor?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? imageProcessor?.imageStd
            ?? [0.5, 0.5, 0.5]
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? imageProcessor?.patchSize
            ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? imageProcessor?.poolingKernelSize
            ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? imageProcessor?.modelPatchSize
            ?? patchSize * poolingKernelSize
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens)
            ?? imageProcessor?.maxSoftTokens
            ?? imageProcessor?.numSoftTokens
            ?? 280
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength) ?? maxSoftTokens
        audioSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSeqLength) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        size =
            try c.decodeIfPresent(
                Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.size)
            ?? imageProcessor?.size
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        let patchesPerSide = max(1, Int(floor(sqrt(Double(maxSoftTokens)))))
        let side = patchesPerSide * patchSize * poolingKernelSize
        return CGSize(width: side, height: side)
    }

    public func aspectRatioPreservingSize(for imageSize: CGSize) throws -> CGSize {
        let width = max(1, Int(ceil(imageSize.width)))
        let height = max(1, Int(ceil(imageSize.height)))
        let sideMultiple = max(1, patchSize * poolingKernelSize)
        let maxTokens = max(1, maxSoftTokens)

        let targetPixels = Double(maxTokens * sideMultiple * sideMultiple)
        let resizeFactor = sqrt(targetPixels / Double(width * height))

        var targetWidth =
            Int(floor(Double(width) * resizeFactor / Double(sideMultiple))) * sideMultiple
        var targetHeight =
            Int(floor(Double(height) * resizeFactor / Double(sideMultiple))) * sideMultiple

        if targetWidth == 0 && targetHeight == 0 {
            throw VLMError.processing("Image is too small to resize for Gemma4 unified vision.")
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(width) / Double(height))) * sideMultiple))
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(height) / Double(width))) * sideMultiple))
        }

        return CGSize(width: targetWidth, height: targetHeight)
    }
}

public struct Gemma4UnifiedProcessor: UserInputProcessor {
    private let config: Gemma4UnifiedProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4UnifiedProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func patchify(_ pixelValues: MLXArray) -> (MLXArray, MLXArray, Int, THW) {
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)
        let patchHeight = height / config.modelPatchSize
        let patchWidth = width / config.modelPatchSize
        let realCount = min(patchHeight * patchWidth, config.maxSoftTokens)
        let patchDim = config.modelPatchSize * config.modelPatchSize * channels

        var patches = pixelValues.reshaped(
            1, channels, patchHeight, config.modelPatchSize, patchWidth, config.modelPatchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(patchHeight * patchWidth, patchDim)
        if realCount < patches.dim(0) {
            patches = patches[..<realCount, 0...]
        }
        if realCount < config.maxSoftTokens {
            patches = padded(patches, widths: [.init((0, config.maxSoftTokens - realCount)), 0])
        }

        var positionValues: [Int32] = []
        positionValues.reserveCapacity(config.maxSoftTokens * 2)
        var emitted = 0
        for y in 0 ..< patchHeight {
            for x in 0 ..< patchWidth where emitted < realCount {
                positionValues.append(Int32(x))
                positionValues.append(Int32(y))
                emitted += 1
            }
        }
        while emitted < config.maxSoftTokens {
            positionValues.append(-1)
            positionValues.append(-1)
            emitted += 1
        }
        let positions = MLXArray(positionValues, [config.maxSoftTokens, 2])
        return (patches, positions, realCount, THW(1, height, width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        pixels: MLXArray, positionIds: MLXArray, tokenCounts: [Int], frames: [THW]
    ) {
        var patchRows: [MLXArray] = []
        var positionRows: [MLXArray] = []
        var tokenCounts: [Int] = []
        var frames: [THW] = []

        for image in images {
            let processedImage = MediaProcessing.apply(image, processing: processing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage =
                if config.doResize {
                    try MediaProcessing.resampleBicubic(
                        srgbImage,
                        to: config.aspectRatioPreservingSize(for: srgbImage.extent.size))
                } else {
                    srgbImage
                }

            var pixelValues = MediaProcessing.asMLXArray(resizedImage)
            let rescaleMultiplier = Float(config.doRescale ? config.rescaleFactor * 255 : 255)
            if rescaleMultiplier != 1 {
                pixelValues = pixelValues * MLXArray(rescaleMultiplier, dtype: pixelValues.dtype)
            }
            if config.doNormalize {
                let mean = MLXArray(
                    config.imageMean.map { Float($0) }, [1, config.imageMean.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                let std = MLXArray(
                    config.imageStd.map { Float($0) }, [1, config.imageStd.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                pixelValues = (pixelValues - mean) / std
            }
            let (patches, positions, tokenCount, frame) = patchify(pixelValues)
            patchRows.append(patches)
            positionRows.append(positions)
            tokenCounts.append(tokenCount)
            frames.append(frame)
        }

        return (
            pixels: stacked(patchRows, axis: 0),
            positionIds: stacked(positionRows, axis: 0),
            tokenCounts: tokenCounts,
            frames: frames
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageData = try preprocess(
                images: input.images.map { try $0.asCIImage() },
                processing: input.processing
            )
            processedImage = LMInput.ProcessedImage(
                pixels: imageData.pixels,
                positionIds: imageData.positionIds,
                frames: imageData.frames
            )

            var imageIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + imageData.tokenCounts.reduce(0, +))
            for token in promptTokens {
                if token == config.imageTokenId {
                    let count =
                        imageIndex < imageData.tokenCounts.count
                        ? imageData.tokenCounts[imageIndex]
                        : config.imageSeqLength
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.imageTokenId, count: count))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}
