//
//  Gemma3nAudio.swift
//  mlx-swift-examples
//
//  Gemma 3n Conformer audio encoder for MLX Swift.
//  Converts mel-spectrogram features into language model embeddings.
//
//  Architecture: Conformer (Gulati et al., 2020) with modifications:
//  - SubSample Conv Projection: 2× Conv2d for initial feature extraction + downsampling
//  - Conformer blocks: FFW → Attention → LightConv1d → FFW → RMSNorm
//  - Chunked local self-attention with relative position embeddings
//  - Cumulative group normalization for streaming compatibility
//  - Temporal reduction (4×) before output
//
//  Split across files:
//  - Gemma3nAudioAttention.swift — relative position embeddings, chunked self-attention
//  - Gemma3nAudioNorm.swift — cumulative group normalization
//  - Gemma3nAudioConv.swift — convolutional subsampling and projection
//  - Gemma3nAudio.swift (this file) — conformer block components, top-level encoder
//
//  Reference: mlx_vlm/models/gemma3n/audio.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
