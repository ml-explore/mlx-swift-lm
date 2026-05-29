//
//  Gemma3nAudioAttention.swift
//  mlx-swift-examples
//
//  Relative position embeddings and chunked local self-attention
//  for the Gemma 3n Conformer audio encoder.
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

// MARK: - Chunked Local Self-Attention

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
