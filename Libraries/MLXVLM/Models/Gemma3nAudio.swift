//
//  Gemma3nAudio.swift
//  mlx-swift-examples
//
//  Gemma 3n Conformer Audio Encoder for MLX Swift.
//  Converts mel-spectrogram features into language model embeddings.
//
//  Architecture: Conformer (Gulati et al., 2020) with modifications:
//  - SubSample Conv Projection: 2× Conv2d for initial feature extraction + downsampling
//  - Conformer blocks: FFW → Attention → LightConv1d → FFW → RMSNorm
//  - Chunked local self-attention with relative position embeddings
//  - Cumulative group normalization for streaming compatibility
//  - Temporal reduction (4×) before output
//
//  Reference: mlx_vlm/models/gemma3n/audio.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Relative Position Embedding

/// Sinusoidal relative position embedding for conformer attention.
///
/// Instead of absolute position encodings (which require knowing the sequence length),
/// this computes position biases based on the *relative distance* between query and key
/// positions. The bias is computed as:
///
///   1. Generate sinusoidal timing signals for relative positions [L, L-1, ..., -R]
///   2. Project through a learned linear layer (pos_proj)
///   3. Compute query-position interaction via matmul
///   4. Apply relative shift to align with the attention logits
///
/// This produces a bias tensor that is added to the content-based attention logits,
/// allowing the model to learn position-dependent attention patterns.
class Gemma3nAudioRelativePositionEmbedding: Module {

    let config: Gemma3nAudioConfiguration
    let numHeads: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int

    @ModuleInfo(key: "pos_proj") var posProj: Linear

    /// Precomputed inverse timescales for sinusoidal encoding.
    /// Shape: [1, 1, channels/2]
    let invTimescales: MLXArray

    init(_ config: Gemma3nAudioConfiguration) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.headDim = config.headDim
        self.maxBackward = config.maxBackward
        self.maxForward = config.maxForward

        _posProj.wrappedValue = Linear(
            config.hiddenSize, config.confNumAttentionHeads * config.headDim, bias: false)

        // Compute inverse timescales for sinusoidal position encoding.
        // These create a spectrum of frequencies from 1.0 to 1/10000, giving the model
        // access to both fine-grained (high frequency) and coarse (low frequency) position info.
        let numTimescales = config.hiddenSize / 2
        let logTimescaleIncrement =
            log(Float(1.0e4) / Float(1.0)) / Float(max(numTimescales - 1, 1))
        let invTs =
            MLXArray(1.0)
            * MLX.exp(
                MLXArray(0 ..< numTimescales).asType(.float32) * (-logTimescaleIncrement)
            )
        self.invTimescales = invTs.reshaped(1, 1, numTimescales)

        super.init()
    }

    /// Generate 1D sinusoidal timing signal from position indices.
    /// Input: position [B, P] → Output: [B, P, channels] (sin/cos interleaved)
    private func getTimingSignal1D(_ position: MLXArray) -> MLXArray {
        // position: [B, P] → [B, P, 1]
        let pos = expandedDimensions(position.asType(.float32), axis: -1)
        // Broadcast multiply: [B, P, 1] * [1, 1, C/2] → [B, P, C/2]
        let scaledTime = pos * invTimescales
        // Concatenate sin and cos: [B, P, C/2] → [B, P, C]
        return concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: -1)
    }

    /// Apply relative shift to align position biases with attention logits.
    ///
    /// The raw query-position interaction produces a tensor indexed by relative position,
    /// but attention logits are indexed by absolute key position within the context window.
    /// This shift converts from relative to absolute indexing via:
    ///   1. Pad the last dimension
    ///   2. Reshape to flatten query×(context+1)
    ///   3. Slice to query×context
    ///   4. Reshape back to [B, N, U, W, C]
    private func relativeShift(
        _ termBD: MLXArray,
        batchSize: Int, numHeads: Int, numQueryBlocks: Int,
        queryBlockSize: Int, keyContextSize: Int, maxSpanPlusOne: Int
    ) -> MLXArray {
        let padAmount = keyContextSize + 1 - maxSpanPlusOne

        // Pad last dimension: [B, N, U, W, F] → [B, N, U, W, F+pad]
        var padWidths = Array(repeating: (0, 0), count: termBD.ndim)
        padWidths[padWidths.count - 1] = (0, padAmount)
        let padded = padded(termBD, widths: padWidths.map { .init($0) })

        // Reshape: [B, N, U, W*(C+1)]
        let reshaped = padded.reshaped(
            batchSize, numHeads, numQueryBlocks,
            queryBlockSize * (keyContextSize + 1)
        )

        // Slice: keep only W*C elements
        let sliced = reshaped[0..., 0..., 0..., ..<(queryBlockSize * keyContextSize)]

        // Reshape back: [B, N, U, W, C]
        return sliced.reshaped(
            batchSize, numHeads, numQueryBlocks, queryBlockSize, keyContextSize
        )
    }

    /// Compute relative position bias for attention.
    ///
    /// - Parameters:
    ///   - queries: [B, U, W, N, H] (batch, blocks, querySize, heads, headDim)
    ///   - keys: [B, U, C, N, H] (batch, blocks, contextSize, heads, headDim)
    /// - Returns: attention logits with position bias [B, N, U, W, C]
    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        let (batchSize, numQueryBlocks, queryBlockSize, _, _) = (
            queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3), queries.dim(4)
        )
        let keyContextSize = keys.dim(2)

        // Generate relative position indices: [L, L-1, ..., -R]
        let posIndices = MLXArray(stride(from: maxBackward, through: -maxForward, by: -1))
            .reshaped(1, -1)  // [1, F_span]

        let maxSpanPlusOne = posIndices.dim(1)

        // Sinusoidal timing signal: [1, F_span, channels]
        let sinEmbTimingSignal = getTimingSignal1D(posIndices)

        // Project: [1, F_span, channels] → [1, F_span, N*H] → [F_span, N, H]
        let projected = posProj(sinEmbTimingSignal)
        let sinEmb = projected.reshaped(1, maxSpanPlusOne, numHeads, headDim).squeezed(axis: 0)

        // term_ac: Content-based attention (query @ key^T)
        // queries: [B, U, W, N, H] → [B, N, U, W, H]
        let queriesP = queries.transposed(0, 3, 1, 2, 4)
        // keys: [B, U, C, N, H] → [B, N, U, H, C]
        let keysPT = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(queriesP, keysPT)  // [B, N, U, W, C]

        // term_bd: Position-based attention (query @ sinEmb^T)
        // sinEmb: [F, N, H] → [N, H, F]
        let sTransposed = sinEmb.transposed(1, 2, 0)

        // Reshape queries for matmul: [B, N, U*W, H]
        let qReshaped = queriesP.reshaped(
            batchSize, numHeads, numQueryBlocks * queryBlockSize, headDim)

        // [B, N, U*W, H] @ [N, H, F] → [B, N, U*W, F]
        let termBDUnshifted = matmul(qReshaped, sTransposed)

        // Reshape: [B, N, U, W, F]
        let termBDReshaped = termBDUnshifted.reshaped(
            batchSize, numHeads, numQueryBlocks, queryBlockSize, maxSpanPlusOne
        )

        // Apply relative shift: convert from relative to absolute position indexing
        let termBDShifted = relativeShift(
            termBDReshaped,
            batchSize: batchSize, numHeads: numHeads, numQueryBlocks: numQueryBlocks,
            queryBlockSize: queryBlockSize, keyContextSize: keyContextSize,
            maxSpanPlusOne: maxSpanPlusOne
        )

        return termAC + termBDShifted
    }
}

// MARK: - Conformer Attention

/// Chunked local self-attention with relative position embeddings and logit softcapping.
///
/// This attention mechanism operates on fixed-size chunks of the audio sequence, with each
/// chunk attending to its local context window (left + self + right). This is more efficient
/// than global attention for long audio sequences and enables streaming inference.
///
/// The attention computation:
///   1. Project input to Q, K, V
///   2. Scale queries by (head_dim^-0.5) * softplus(per_dim_scale)
///   3. Chunk queries into blocks, extract overlapping context for keys/values
///   4. Compute attention logits with relative position bias
///   5. Apply logit softcap: tanh(logits/cap) * cap
///   6. Mask invalid positions (padding + causal)
///   7. Softmax → weighted sum of values
class Gemma3nAudioAttention: Module {

    let config: Gemma3nAudioConfiguration
    let numHeads: Int
    let headDim: Int
    let chunkSize: Int
    let contextSize: Int
    let maxPastHorizon: Int
    let maxFutureHorizon: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "relative_position_embedding") var relPosEmb:
        Gemma3nAudioRelativePositionEmbedding

    /// Per-dimension learned scaling for queries. Initialized to zero (softplus(0) = ln(2)).
    let perDimScale: MLXArray

    /// Precomputed query scale factor incorporating head_dim and softplus normalization.
    let qScale: Float

    /// Precomputed local causal validity mask [W, C] — True where attention is allowed.
    let localCausalValidMask: MLXArray

    /// Softcap value for attention logits.
    let softcap: MLXArray

    /// Value for masked attention positions.
    let invalidLogitsValue: Float

    init(_ config: Gemma3nAudioConfiguration) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.headDim = config.headDim
        self.chunkSize = config.confAttentionChunkSize
        self.maxPastHorizon = config.maxBackward
        self.maxFutureHorizon = config.maxForward
        self.contextSize = config.contextSize
        self.invalidLogitsValue = config.confAttentionInvalidLogitsValue

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _relPosEmb.wrappedValue = Gemma3nAudioRelativePositionEmbedding(config)

        // Per-dim scale: softplus(0) = ln(2) ≈ 0.693, so initial effective scale ≈ head_dim^-0.5 * 0.693
        self.perDimScale = MLXArray.zeros([headDim])

        // Query scale: head_dim^-0.5 * (1/ln(2)) — the 1/ln(2) compensates for softplus
        let rSoftplus0 = 1.0 / log(2.0)
        self.qScale = pow(Float(headDim), -0.5) * Float(rSoftplus0)

        // Build local causal validity mask [W, C]
        // This mask encodes which (query_position, key_position) pairs are valid
        // based on the local attention window structure.
        let lower = MLX.tril(MLXArray.ones([contextSize, chunkSize]).asType(.bool)).transposed()
        let upper = MLX.tril(
            MLXArray.ones([chunkSize, contextSize]).asType(.bool),
            k: maxPastHorizon + maxFutureHorizon
        )
        self.localCausalValidMask = lower * upper

        self.softcap = MLXArray(config.confAttentionLogitCap).asType(.float32)

        super.init()
    }

    /// Pad dimension at index 1 (time dimension) of a tensor.
    private func padDim1(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var padWidths = Array(repeating: (0, 0), count: x.ndim)
        padWidths[1] = (left, right)
        return padded(x, widths: padWidths.map { .init($0) })
    }

    /// Convert a sequence into non-overlapping blocks of chunkSize.
    /// Input [B, T, ...] → Output [B, numBlocks, chunkSize, ...]
    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let (b, t) = (shape[0], shape[1])
        let numBlocks = (t + chunkSize - 1) / chunkSize

        // Pad time to multiple of chunkSize if needed
        var padded = x
        let paddingLen = numBlocks * chunkSize - t
        if paddingLen > 0 {
            padded = padDim1(x, left: 0, right: paddingLen)
        }

        // Reshape: [B, T_padded, ...] → [B, numBlocks, chunkSize, ...]
        var newShape = [b, numBlocks, chunkSize]
        newShape.append(contentsOf: shape[2...])
        return padded.reshaped(newShape)
    }

    /// Extract overlapping context windows for keys/values.
    /// Input [B, T, ...] → Output [B, numBlocks, contextSize, ...]
    ///
    /// Each block gets its own chunk plus maxPastHorizon frames before and
    /// maxFutureHorizon + chunkSize - 1 frames after (padded as needed).
    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let paddedX = padDim1(x, left: padLeft, right: padRight)

        // Sliding window extraction matching Python unfold_mlx behavior:
        // Python stacks at axis=dimension+1 (dimension=1, so axis=2)
        // This produces [B, T_remaining, numWindows, contextSize, ...] for 4D
        // Then transpose (0,2,1,3,4) → [B, numWindows, contextSize, ...]
        let timeDim = paddedX.dim(1)
        let numWindows = (timeDim - contextSize) / chunkSize + 1

        var windows: [MLXArray] = []
        for i in 0 ..< numWindows {
            let start = i * chunkSize
            let end = start + contextSize
            windows.append(paddedX[0..., start ..< end])
        }

        // Stack at axis=2 to match Python unfold_mlx(x, 1, size, step)
        // which stacks at axis=dimension+1=2
        var result = stacked(windows, axis: 2)

        // For 4D+ input: transpose (0,2,1,3,4) to get [B, numWindows, contextSize, N, H]
        if x.ndim > 2 && result.ndim > 3 {
            result = result.transposed(0, 2, 1, 3, 4)
        }

        // For 2D mask input: stack at axis=2 gives [B, T_remaining, numWindows]
        // Transpose to [B, numWindows, contextSize]
        if x.ndim == 2 && result.ndim == 3 {
            result = result.transposed(0, 2, 1)
        }

        return result
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        // Project to Q, K, V: [B, T, D] → [B, T, N, H]
        var queryStates = qProj(x).reshaped(x.dim(0), x.dim(1), numHeads, headDim)
        let keyStates = kProj(x).reshaped(x.dim(0), x.dim(1), numHeads, headDim)
        let valueStates = vProj(x).reshaped(x.dim(0), x.dim(1), numHeads, headDim)

        // Apply per-dimension scaling: softplus(perDimScale) * queryScale
        // softplus(x) = log(1 + exp(x)) = logaddexp(x, 0)
        let perDimScaleSP = MLX.logAddExp(perDimScale, MLXArray(0.0))
        let scaleShape = [1, 1, 1, headDim]
        queryStates = queryStates * qScale * perDimScaleSP.reshaped(scaleShape)

        let (batchSize, qTime) = (queryStates.dim(0), queryStates.dim(1))

        // Chunk queries into blocks, extract context windows for K, V
        let queryBlocks = convertToBlock(queryStates)
        let keyBlocks = extractBlockContext(keyStates)
        let valueBlocks = extractBlockContext(valueStates)
        let numQueryBlocks = queryBlocks.dim(1)

        // Build validity mask from input padding mask
        // extractBlockContext now matches Python: returns [B, numWindows, contextSize] for 2D mask
        let originalValidMask = mask .== false  // True for valid positions
        let extractedValidMask = extractBlockContext(originalValidMask)

        var reshapedValidMask = extractedValidMask
        if reshapedValidMask.ndim == 4 {
            reshapedValidMask = reshapedValidMask.reshaped(
                batchSize, numQueryBlocks, contextSize)
        }

        // Expand for broadcasting with logits [B, N, U, W, C]
        // [B, U, C] → [B, 1, U, 1, C]
        var conditionFromInput = expandedDimensions(reshapedValidMask, axis: 1)
        conditionFromInput = expandedDimensions(conditionFromInput, axis: -2)

        // Local causal mask: [1, 1, 1, W, C]
        let conditionFromCausality =
            localCausalValidMask
            .reshaped(1, 1, 1, chunkSize, contextSize)

        // Combined mask: both valid AND causally accessible
        let finalCondition = conditionFromInput & conditionFromCausality

        // Compute attention logits with relative position bias
        var logits = relPosEmb(queryBlocks, keyBlocks)

        // Apply attention logit softcap: tanh(logits/cap) * cap
        logits = logits / softcap
        logits = MLX.tanh(logits)
        logits = logits * softcap

        // Apply mask: invalid positions get -1e9
        logits = MLX.where(finalCondition, logits, MLXArray(invalidLogitsValue))

        // Softmax over key dimension
        let probabilities = softmax(logits.asType(.float32), axis: -1)
            .asType(valueBlocks.dtype)

        // Weighted sum of values: einsum("BNuwc,BucNH->BuwNH")
        let (bDim, nDim, uDim, wDim, cDim) = (
            probabilities.dim(0), probabilities.dim(1), probabilities.dim(2),
            probabilities.dim(3), probabilities.dim(4)
        )
        let hDim = valueBlocks.dim(-1)

        // Reshape for batch matmul
        let probBUN = probabilities.transposed(0, 2, 1, 3, 4)
            .reshaped(bDim * uDim * nDim, wDim, cDim)
        let vBUN = valueBlocks.transposed(0, 1, 3, 2, 4)
            .reshaped(bDim * uDim * nDim, cDim, hDim)
        let resultBMM = matmul(probBUN, vBUN)

        // Reshape back: [B, U, N, W, H] → [B, U, W, N, H] → [B, T, N, H]
        var contextVectors = resultBMM.reshaped(bDim, uDim, nDim, wDim, hDim)
            .transposed(0, 1, 3, 2, 4)
        contextVectors = contextVectors.reshaped(
            batchSize, numQueryBlocks * chunkSize, numHeads, headDim)

        // Trim to original time length (remove block padding)
        return contextVectors[0..., ..<qTime]
    }
}

// MARK: - Cumulative Group Normalization

/// Group normalization with cumulative statistics over the time dimension.
///
/// Unlike standard batch/layer norm which requires the full sequence to compute statistics,
/// cumulative group norm computes a running mean and variance over time steps seen so far.
/// This is critical for streaming audio processing where future frames aren't available.
///
/// For each time step t, the statistics are:
///   mean(t) = cumsum(values[0..t]) / cumsum(counts[0..t])
///   var(t)  = cumsum((values[0..t] - mean(t))²) / cumsum(counts[0..t])
///
/// Masked (padded) positions don't contribute to statistics and produce zero output.
class Gemma3nCumulativeGroupNorm: Module {

    let numChannels: Int
    let featureDims: [Int]
    let eps: Float
    let reductionAxes: [Int]

    /// Scale parameter [C], applied per-channel.
    let weight: MLXArray?

    init(
        numChannels: Int,
        featureDims: [Int],
        eps: Float = 1e-3,
        useScale: Bool = true,
        useBias: Bool = false
    ) {
        self.numChannels = numChannels
        self.featureDims = featureDims
        self.eps = eps
        // Reduction over all dims except batch (0) and time (1)
        self.reductionAxes = Array(2 ..< (2 + featureDims.count + 1))

        self.weight = useScale ? MLXArray.ones([numChannels]) : nil

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let inputDtype = x.dtype
        let xCalc = x.asType(.float32)

        // Build broadcastable mask
        let maskCalc: MLXArray
        if let mask = mask {
            let suffixShape = Array(repeating: 1, count: x.ndim - 2)
            maskCalc = mask.reshaped(Array(mask.shape) + suffixShape).asType(.float32)
        } else {
            maskCalc = MLXArray.ones(like: xCalc)
        }

        // Masked input for sum calculation
        let xMasked = xCalc * maskCalc

        // Cumulative statistics over time dimension
        let sumValuesAtT = xMasked.sum(axes: reductionAxes, keepDims: true)
        let cumSumValues = cumsum(sumValuesAtT, axis: 1)

        let elementsAtT = maskCalc.sum(axes: reductionAxes, keepDims: true)
        let cumCountElements = cumsum(elementsAtT, axis: 1)
        let safeCumCount = clip(cumCountElements, min: 1)

        // Cumulative mean
        let cumMean = cumSumValues / safeCumCount

        // Cumulative variance
        let sqDiff = (xCalc - cumMean) ** 2
        let sumSqDiffAtT = (sqDiff * maskCalc).sum(axes: reductionAxes, keepDims: true)
        let cumSumSqDiff = cumsum(sumSqDiffAtT, axis: 1)
        let cumVariance = cumSumSqDiff / safeCumCount

        // Normalize
        var normalized = (xCalc - cumMean) * rsqrt(cumVariance + eps)

        // Apply scale
        if let weight = weight {
            var scaleShape = Array(repeating: 1, count: x.ndim - 1)
            scaleShape.append(numChannels)
            normalized = normalized * weight.asType(.float32).reshaped(scaleShape)
        }

        // Zero out masked positions
        return (normalized * maskCalc).asType(inputDtype)
    }
}

// MARK: - SubSample Conv Projection

/// Convolutional subsampling block: Conv2d → CumulativeGroupNorm → ReLU.
///
/// Treats the mel-spectrogram as a 2D image where:
/// - Height = Time (number of frames)
/// - Width = Frequency (number of mel bins)
/// - Channels = 1 (single spectrogram) or output of previous block
///
/// Each block applies stride 2×2 convolution, reducing both time and frequency by 2×.
/// Two blocks in sequence give 4× reduction in both dimensions.
class Gemma3nAudioSSCPConvBlock: Module {

    let manualPadding: (Int, Int, Int, Int)  // (padFLeft, padFRight, padTTop, padTBottom)

    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: Gemma3nCumulativeGroupNorm

    init(
        idx: Int,
        inputFreqDim: Int,
        config: Gemma3nAudioConfiguration,
        manualPadding: (Int, Int, Int, Int)
    ) {
        self.manualPadding = manualPadding

        let inChannels = idx == 0 ? 1 : config.sscpConvChannelSize[idx - 1]
        let outChannels = config.sscpConvChannelSize[idx]
        let (kernelH, kernelW) = (
            config.sscpConvKernelSize[idx][0], config.sscpConvKernelSize[idx][1]
        )
        let (strideH, strideW) = (
            config.sscpConvStrideSize[idx][0], config.sscpConvStrideSize[idx][1]
        )

        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair((kernelH, kernelW)),
            stride: IntOrPair((strideH, strideW)),
            padding: IntOrPair(0),
            bias: false
        )

        // Calculate output frequency dimension after conv
        let fInPadded = inputFreqDim + manualPadding.0 + manualPadding.1
        let fOutConv = (fInPadded - kernelW) / strideW + 1

        _norm.wrappedValue = Gemma3nCumulativeGroupNorm(
            numChannels: outChannels,
            featureDims: [fOutConv],
            eps: config.sscpConvEps,
            useScale: true,
            useBias: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C_in, T_in, F_in]
        // Apply manual padding: (padFLeft, padFRight, padTTop, padTBottom)
        var padWidths = Array(repeating: (0, 0), count: x.ndim)
        padWidths[x.ndim - 1] = (manualPadding.0, manualPadding.1)  // Frequency
        padWidths[x.ndim - 2] = (manualPadding.2, manualPadding.3)  // Time
        let xPadded = padded(x, widths: padWidths.map { .init($0) })

        // Conv2d expects [B, H, W, C] in MLX (channels-last)
        let convOut = conv(xPadded.transposed(0, 2, 3, 1))

        // Norm expects [B, T, F, C]
        let normed = norm(convOut)

        // Back to [B, C, T, F] for next block
        return relu(normed.transposed(0, 3, 1, 2))
    }
}

/// Two-stage convolutional subsampling + linear projection.
///
/// Takes raw mel-spectrogram [B, T, F] and produces hidden representations [B, T_sub, D]
/// where T_sub = T / (stride_h1 * stride_h2) and D = hiddenSize.
///
/// Pipeline:
///   [B, T, F=80] → [B, 1, T, F] (add channel dim)
///   → Conv2d(1→128, k=3×3, s=2×2) → GroupNorm → ReLU
///   → Conv2d(128→32, k=3×3, s=2×2) → GroupNorm → ReLU
///   → [B, T/4, F/4*32] (flatten freq×channels)
///   → Linear(F/4*32, hiddenSize) → [B, T/4, D]
class Gemma3nAudioSubSampleConvProjection: Module {

    @ModuleInfo(key: "conv_0") var conv0: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "conv_1") var conv1: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: Gemma3nAudioConfiguration) {
        // Calculate padding and output dimensions for each conv block
        var currentFreqDim = config.inputFeatSize
        var blockPaddings: [(Int, Int, Int, Int)] = []
        var fOutDims: [Int] = []

        for i in 0 ..< 2 {
            let (kernelH, kernelW) = (
                config.sscpConvKernelSize[i][0], config.sscpConvKernelSize[i][1]
            )
            let (_, strideW) = (config.sscpConvStrideSize[i][0], config.sscpConvStrideSize[i][1])

            // Reverse-causal padding for time, SAME-like padding for frequency
            let padTTop = 0
            let padTBottom = kernelH - 1
            let padFLeft = 1
            let padFRight = 1

            blockPaddings.append((padFLeft, padFRight, padTTop, padTBottom))

            let fInPadded = currentFreqDim + padFLeft + padFRight
            let fOut = (fInPadded - kernelW) / strideW + 1
            fOutDims.append(fOut)
            currentFreqDim = fOut
        }

        _conv0.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 0, inputFreqDim: config.inputFeatSize,
            config: config, manualPadding: blockPaddings[0]
        )
        _conv1.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 1, inputFreqDim: fOutDims[0],
            config: config, manualPadding: blockPaddings[1]
        )

        // Projection from flattened conv output to hidden size
        let finalCOut = config.sscpConvChannelSize[1]
        let finalFOut = fOutDims[1]
        _inputProjLinear.wrappedValue = Linear(
            finalCOut * finalFOut, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, F_in=80]
        // Add channel dimension: [B, T, F] → [B, 1, T, F]
        var audio = expandedDimensions(x, axis: 1)
        audio = conv0(audio)
        audio = conv1(audio)

        // audio: [B, C_out, T_out, F_out]
        let (b, cOut, tOut, fOut) = (audio.dim(0), audio.dim(1), audio.dim(2), audio.dim(3))

        // Flatten frequency and channels: [B, T_out, F_out * C_out]
        let transposed = audio.transposed(0, 2, 3, 1)
        let flattened = transposed.reshaped(b, tOut, fOut * cOut)

        // Project to hidden size: [B, T_out, D]
        return inputProjLinear(flattened)
    }
}

// MARK: - Conformer Block Components

/// Conformer attention wrapper: pre-norm → attention → post-projection → residual.
class Gemma3nAudioConformerAttention: Module {

    let gradientClipping: MLXArray

    @ModuleInfo(key: "pre_attn_norm") var preAttnNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: Gemma3nAudioAttention
    @ModuleInfo(key: "post") var post: Linear
    @ModuleInfo(key: "post_norm") var postNorm: RMSNorm

    init(_ config: Gemma3nAudioConfiguration) {
        self.gradientClipping = MLXArray(config.gradientClipping)

        _preAttnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        _attn.wrappedValue = Gemma3nAudioAttention(config)
        _post.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let residual = x
        var h = clip(x, min: -gradientClipping, max: gradientClipping)
        h = preAttnNorm(h)
        h = attn(h, mask: mask)

        // Reshape from [B, T, N, H] → [B, T, D]
        let (b, t, _, _) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        h = h.reshaped(b, t, -1)

        h = post(h)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        return residual + postNorm(h)
    }
}

/// Conformer feed-forward: pre-norm → linear → SiLU → linear → post-norm + residual scaling.
class Gemma3nAudioConformerFeedForward: Module {

    let gradientClipping: MLXArray
    let postLayerScale: MLXArray

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: Linear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: Linear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: RMSNorm

    init(_ config: Gemma3nAudioConfiguration) {
        self.gradientClipping = MLXArray(config.gradientClipping)
        self.postLayerScale = MLXArray(config.confResidualWeight)

        _preLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        _ffwLayer1.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 4, bias: false)
        _ffwLayer2.wrappedValue = Linear(config.hiddenSize * 4, config.hiddenSize, bias: false)
        _postLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = clip(x, min: -gradientClipping, max: gradientClipping)
        h = preLayerNorm(h)
        h = ffwLayer1(h)
        h = silu(h)
        h = ffwLayer2(h)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        h = postLayerNorm(h)
        return residual + (h * postLayerScale)
    }
}

/// Depthwise separable 1D convolution block for the conformer.
///
/// This captures local patterns in the audio that attention might miss (e.g., phoneme
/// boundaries, attack transients). Uses causal padding so only past context is visible.
///
/// Pipeline: pre-norm → linear(D→2D) → GLU → depthwise_conv1d → norm → SiLU → linear(D→D)
class Gemma3nAudioConformerLightConv1d: Module {

    let causalPadding: Int
    let gradientClipping: MLXArray

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Linear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: RMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Linear

    init(_ config: Gemma3nAudioConfiguration) {
        self.causalPadding = config.confConvKernelSize - 1
        self.gradientClipping = MLXArray(config.gradientClipping)

        _preLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _linearStart.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize * 2, bias: false)
        _depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.confConvKernelSize,
            stride: 1,
            padding: 0,  // Manual causal padding
            groups: config.hiddenSize  // Depthwise: each channel convolved independently
        )
        _convNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _linearEnd.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preLayerNorm(x)
        h = linearStart(h)
        h = glu(h, axis: -1)

        // Apply causal padding (left only) for conv1d
        // Conv1d in MLX expects [B, T, C]
        var padWidths = Array(repeating: (0, 0), count: h.ndim)
        padWidths[1] = (causalPadding, 0)  // Pad time dimension on left only
        h = padded(h, widths: padWidths.map { .init($0) })

        h = depthwiseConv1d(h)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        h = convNorm(h)
        h = silu(h)
        h = linearEnd(h)
        return h + residual
    }
}

/// Full conformer block: FFW → Attention → LightConv1d → FFW → Norm.
///
/// The Macaron-Net structure (two half-step FFW layers surrounding the attention and
/// convolution) has been shown to outperform single-FFW conformers. The final RMSNorm
/// stabilizes the output before the next block.
class Gemma3nAudioConformerBlock: Module {

    let gradientClipping: MLXArray

    @ModuleInfo(key: "ffw_layer_start") var ffwLayerStart: Gemma3nAudioConformerFeedForward
    @ModuleInfo(key: "attention") var attention: Gemma3nAudioConformerAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: Gemma3nAudioConformerLightConv1d
    @ModuleInfo(key: "ffw_layer_end") var ffwLayerEnd: Gemma3nAudioConformerFeedForward
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Gemma3nAudioConfiguration) {
        self.gradientClipping = MLXArray(config.gradientClipping)

        _ffwLayerStart.wrappedValue = Gemma3nAudioConformerFeedForward(config)
        _attention.wrappedValue = Gemma3nAudioConformerAttention(config)
        _lconv1d.wrappedValue = Gemma3nAudioConformerLightConv1d(config)
        _ffwLayerEnd.wrappedValue = Gemma3nAudioConformerFeedForward(config)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = ffwLayerStart(x)
        h = attention(h, mask: mask)

        // Mask invalid positions before conv (conv shouldn't see padded frames)
        let validMask = mask .== false  // True for valid
        h = h * expandedDimensions(validMask, axis: -1).asType(h.dtype)

        h = lconv1d(h)
        h = ffwLayerEnd(h)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        return norm(h)
    }
}

// MARK: - Audio Model (Top-Level Encoder)

/// Complete Gemma 3n audio encoder.
///
/// Takes mel-spectrogram features and produces embeddings ready for the language model.
/// The output sequence is temporally reduced by a total factor of:
///   stride_h1 × stride_h2 × conf_reduction_factor = 2 × 2 × 4 = 16×
///
/// For 16kHz audio with 10ms frame shift (100 frames/sec):
///   Input: 100 frames/sec
///   After SSCP: 25 frames/sec (4× reduction from two stride-2 convolutions)
///   After conformer reduction: ~6.25 frames/sec (additional 4× reduction)
///
/// This means each output embedding token represents approximately 160ms of audio.
public class Gemma3nAudioModel: Module {

    let config: Gemma3nAudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection:
        Gemma3nAudioSubSampleConvProjection
    @ModuleInfo(key: "conformer") var conformer: [Gemma3nAudioConformerBlock]

    public init(_ config: Gemma3nAudioConfiguration) {
        self.config = config

        _subsampleConvProjection.wrappedValue = Gemma3nAudioSubSampleConvProjection(config)

        var blocks: [Gemma3nAudioConformerBlock] = []
        for _ in 0 ..< config.confNumHiddenLayers {
            blocks.append(Gemma3nAudioConformerBlock(config))
        }
        _conformer.wrappedValue = blocks

        super.init()
    }

    /// Encode mel-spectrogram features into language model embeddings.
    ///
    /// - Parameters:
    ///   - audioMel: Mel-spectrogram features [B, T, F] where F=inputFeatSize (80)
    ///   - audioMelMask: Boolean mask [B, T] where True = padded/invalid
    /// - Returns: (encodings [B, T_out, D], mask [B, T_out])
    public func callAsFunction(
        _ audioMel: MLXArray, mask audioMelMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Subsample conv: [B, T, 80] → [B, T_sub, D]
        var audioEncodings = subsampleConvProjection(audioMel)
        let tSub = audioEncodings.dim(1)

        // Subsample the mask to match reduced time dimension
        var timeStrideProduct = 1
        for i in 0 ..< config.sscpConvStrideSize.count {
            timeStrideProduct *= config.sscpConvStrideSize[i][0]
        }

        var indices = MLXArray(0 ..< tSub) * timeStrideProduct
        indices = clip(indices, max: audioMelMask.dim(1) - 1)

        // Expand indices for batch dimension
        if audioMelMask.ndim > 1 {
            indices = indices.reshaped(1, -1)
            indices = broadcast(indices, to: [audioMelMask.dim(0), tSub])
        }

        var currentMask = takeAlong(audioMelMask, indices, axis: 1)

        // Ensure mask length matches feature length
        if currentMask.dim(1) != tSub {
            if currentMask.dim(1) > tSub {
                currentMask = currentMask[0..., ..<tSub]
            } else {
                let paddingNeeded = tSub - currentMask.dim(1)
                var padWidths = Array(repeating: (0, 0), count: currentMask.ndim)
                padWidths[padWidths.count - 1] = (0, paddingNeeded)
                currentMask = padded(currentMask, widths: padWidths.map { .init($0) })
            }
        }

        // Process through conformer blocks
        for block in conformer {
            audioEncodings = block(audioEncodings, mask: currentMask)
        }

        // Temporal reduction: keep every Nth frame
        if config.confReductionFactor > 1 {
            let stride = config.confReductionFactor
            let reducedLen = (audioEncodings.dim(1) + stride - 1) / stride
            let reducedIndices = MLXArray(0 ..< reducedLen) * stride
            let clippedIndices = clip(reducedIndices, max: audioEncodings.dim(1) - 1)

            // Gather along time dimension
            audioEncodings = takeAlong(
                audioEncodings,
                expandedDimensions(clippedIndices, axes: [0, -1]),
                axis: 1
            )
            currentMask = takeAlong(
                currentMask,
                clippedIndices.reshaped(1, -1),
                axis: 1
            )
        }

        // Final masking: zero out padded positions
        audioEncodings = MLX.where(
            expandedDimensions(currentMask, axis: -1),
            MLXArray(0.0),
            audioEncodings
        )

        return (audioEncodings, currentMask)
    }

    /// Sanitize weights loaded from Python format.
    /// Conv2d weights need transposition: PyTorch [O, I, H, W] → MLX [O, H, W, I]
    /// Conv1d weights need transposition: PyTorch [O, I, K] → MLX [O, K, I]
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("conv.weight") && value.ndim == 4 {
                // Conv2d: check if already in MLX format
                // MLX expects [O, H, W, I], PyTorch has [O, I, H, W]
                if value.dim(3) > value.dim(1) {
                    sanitized[key] = value  // Already MLX format
                } else {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                }
            } else if key.contains("conv1d.weight") && value.ndim == 3 {
                // Conv1d: check if already in MLX format
                // MLX expects [O, K, I], PyTorch has [O, I, K]
                if value.dim(2) > value.dim(1) {
                    sanitized[key] = value  // Already MLX format
                } else {
                    sanitized[key] = value.transposed(0, 2, 1)
                }
            } else {
                sanitized[key] = value
            }
        }

        return sanitized
    }
}
