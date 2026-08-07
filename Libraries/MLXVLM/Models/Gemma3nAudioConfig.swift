//
//  Gemma3nAudioConfig.swift
//  mlx-swift-examples
//
//  Audio encoder configuration for Google Gemma 3n multimodal models.
//  Parses audio_config from the HuggingFace config.json.
//
//  The Gemma 3n audio encoder is a Conformer-based architecture that converts
//  mel-spectrogram features into embeddings for the language model. The pipeline:
//
//    Raw audio (16kHz PCM)
//      → 80-bin mel spectrogram (input_feat_size=80)
//      → SubSampleConvProjection (2× Conv2d + CumulativeGroupNorm, stride 2×2 each)
//      → 12 Conformer blocks (self-attention + depthwise conv + 2× FFW)
//      → Temporal reduction (conf_reduction_factor=4, keeps every 4th frame)
//      → MultimodalEmbedder (norm + linear projection to text hidden_size)
//      → Merged with text token embeddings at audio_token positions
//      → Language model generates response
//
//  Architecture reference: mlx_vlm/models/gemma3n/config.py
//

import Foundation

// MARK: - Audio Encoder Configuration

/// Configuration for the Gemma 3n Conformer audio encoder.
///
/// The Conformer architecture combines self-attention (for global context) with
/// depthwise separable convolutions (for local patterns) in each block, making it
/// highly effective for speech and audio processing. Key design choices:
///
/// - **Chunked local attention** (`conf_attention_chunk_size=12`): processes audio
///   in 12-frame chunks with left context of 13 and right context of 0 (causal).
///   This enables streaming-capable inference.
///
/// - **Relative position embeddings**: sinusoidal embeddings projected through a
///   learned linear layer, shifted to create relative position biases. This is
///   more efficient than absolute position embeddings for variable-length audio.
///
/// - **Cumulative group normalization**: instead of standard batch/layer norm,
///   uses cumulative statistics over the time dimension. This enables proper
///   normalization during streaming where future frames aren't available.
///
/// - **Temporal reduction** (`conf_reduction_factor=4`): after all conformer blocks,
///   keeps every 4th frame to reduce sequence length before feeding to the LM.
///   For 16kHz audio with 10ms frame shift, this means ~40ms per output token.
public struct Gemma3nAudioConfiguration: Codable, Sendable {

    // MARK: Input

    /// Number of mel-frequency bins in the input spectrogram. Default: 80.
    /// Standard for speech models — covers 0-8kHz with mel spacing.
    public let inputFeatSize: Int

    // MARK: Conformer Hidden Dimensions

    /// Hidden dimension throughout the conformer blocks and projection layers.
    /// All attention, FFW, and conv layers operate at this dimension.
    public let hiddenSize: Int

    // MARK: Conformer Self-Attention

    /// Number of attention heads in each conformer attention layer.
    /// head_dim = hiddenSize / confNumAttentionHeads (e.g., 1536/8 = 192).
    public let confNumAttentionHeads: Int

    /// Number of conformer blocks stacked sequentially.
    public let confNumHiddenLayers: Int

    /// Size of each attention chunk (number of query frames per block).
    /// The conformer processes audio in non-overlapping chunks of this size.
    public let confAttentionChunkSize: Int

    /// Left context for attention (number of past frames visible).
    /// Effective left context = confAttentionContextLeft - 1 frames.
    /// Set to 13 → 12 frames of past context per chunk.
    public let confAttentionContextLeft: Int

    /// Right context for attention (number of future frames visible).
    /// Set to 0 for causal (streaming-capable) processing.
    public let confAttentionContextRight: Int

    /// Value used to mask invalid attention logits (effectively -infinity).
    public let confAttentionInvalidLogitsValue: Float

    /// Soft cap applied to attention logits: tanh(logits/cap) * cap.
    /// Prevents attention weights from becoming too extreme.
    public let confAttentionLogitCap: Float

    // MARK: Conformer Convolution

    /// Kernel size for the depthwise separable 1D convolution in each conformer block.
    /// Applied causally (padding only on the left side).
    public let confConvKernelSize: Int

    // MARK: Conformer Output

    /// Temporal reduction factor applied after all conformer blocks.
    /// Keeps every Nth frame, reducing sequence length by this factor.
    /// For 16kHz audio with 10ms shift and factor=4, output is ~40ms/frame.
    public let confReductionFactor: Int

    /// Residual connection weight for FFW sublayers.
    /// output = residual + (ffwOutput * confResidualWeight)
    public let confResidualWeight: Float

    // MARK: SubSample Conv Projection (SSCP)

    /// Output channel sizes for each SSCP conv block. Array of 2 values.
    /// Default: [128, 32] — first conv expands to 128 channels, second reduces to 32.
    public let sscpConvChannelSize: [Int]

    /// Kernel sizes for each SSCP conv block. Array of 2 (height, width) pairs.
    /// Default: [[3,3], [3,3]] — 3×3 kernels operating on (time, frequency).
    public let sscpConvKernelSize: [[Int]]

    /// Stride sizes for each SSCP conv block. Array of 2 (height, width) pairs.
    /// Default: [[2,2], [2,2]] — 2× downsampling in both time and frequency.
    /// Combined effect: 4× temporal reduction, 4× frequency reduction.
    public let sscpConvStrideSize: [[Int]]

    /// Epsilon for cumulative group norm in SSCP conv blocks.
    public let sscpConvEps: Float

    // MARK: Normalization

    /// Epsilon for RMS normalization layers.
    public let rmsNormEps: Float

    /// Gradient clipping value applied inside conformer blocks.
    /// Very large default (1e10) means effectively no clipping during inference.
    public let gradientClipping: Float

    // MARK: Vocabulary / Tokenization

    /// Size of the audio token vocabulary (for hard audio token embeddings).
    public let vocabSize: Int

    /// Offset into the unified vocabulary where audio tokens begin.
    /// audio_token_id = vocabOffset + audio_token_index
    /// Default: 262144 (text) + 128 (vision) = 262272
    public let vocabOffset: Int

    // MARK: CodingKeys

    enum CodingKeys: String, CodingKey {
        case inputFeatSize = "input_feat_size"
        case hiddenSize = "hidden_size"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confAttentionChunkSize = "conf_attention_chunk_size"
        case confAttentionContextLeft = "conf_attention_context_left"
        case confAttentionContextRight = "conf_attention_context_right"
        case confAttentionInvalidLogitsValue = "conf_attention_invalid_logits_value"
        case confAttentionLogitCap = "conf_attention_logit_cap"
        case confConvKernelSize = "conf_conv_kernel_size"
        case confReductionFactor = "conf_reduction_factor"
        case confResidualWeight = "conf_residual_weight"
        case sscpConvChannelSize = "sscp_conv_channel_size"
        case sscpConvKernelSize = "sscp_conv_kernel_size"
        case sscpConvStrideSize = "sscp_conv_stride_size"
        case sscpConvEps = "sscp_conv_eps"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
    }

    // MARK: Defaults

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        inputFeatSize = try container.decodeIfPresent(Int.self, forKey: .inputFeatSize) ?? 128
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        confNumAttentionHeads =
            try container.decodeIfPresent(
                Int.self, forKey: .confNumAttentionHeads) ?? 8
        confNumHiddenLayers =
            try container.decodeIfPresent(
                Int.self, forKey: .confNumHiddenLayers) ?? 12
        confAttentionChunkSize =
            try container.decodeIfPresent(
                Int.self, forKey: .confAttentionChunkSize) ?? 12
        confAttentionContextLeft =
            try container.decodeIfPresent(
                Int.self, forKey: .confAttentionContextLeft) ?? 13
        confAttentionContextRight =
            try container.decodeIfPresent(
                Int.self, forKey: .confAttentionContextRight) ?? 0
        confAttentionInvalidLogitsValue =
            try container.decodeIfPresent(
                Float.self, forKey: .confAttentionInvalidLogitsValue) ?? -1e9
        confAttentionLogitCap =
            try container.decodeIfPresent(
                Float.self, forKey: .confAttentionLogitCap) ?? 50.0
        confConvKernelSize =
            try container.decodeIfPresent(
                Int.self, forKey: .confConvKernelSize) ?? 5
        confReductionFactor =
            try container.decodeIfPresent(
                Int.self, forKey: .confReductionFactor) ?? 4
        confResidualWeight =
            try container.decodeIfPresent(
                Float.self, forKey: .confResidualWeight) ?? 0.5
        sscpConvChannelSize =
            try container.decodeIfPresent(
                [Int].self, forKey: .sscpConvChannelSize) ?? [128, 32]
        sscpConvKernelSize =
            try container.decodeIfPresent(
                [[Int]].self, forKey: .sscpConvKernelSize) ?? [[3, 3], [3, 3]]
        sscpConvStrideSize =
            try container.decodeIfPresent(
                [[Int]].self, forKey: .sscpConvStrideSize) ?? [[2, 2], [2, 2]]
        sscpConvEps =
            try container.decodeIfPresent(
                Float.self, forKey: .sscpConvEps) ?? 1e-3
        rmsNormEps =
            try container.decodeIfPresent(
                Float.self, forKey: .rmsNormEps) ?? 1e-6
        gradientClipping =
            try container.decodeIfPresent(
                Float.self, forKey: .gradientClipping) ?? 1e10
        vocabSize =
            try container.decodeIfPresent(
                Int.self, forKey: .vocabSize) ?? 128
        vocabOffset =
            try container.decodeIfPresent(
                Int.self, forKey: .vocabOffset) ?? (262_144 + 128)
    }

    // MARK: Computed Properties

    /// Head dimension for conformer attention: hiddenSize / numHeads.
    public var headDim: Int { hiddenSize / confNumAttentionHeads }

    /// Maximum backward span for relative position embeddings.
    /// If confAttentionContextLeft > 0, this is contextLeft - 1.
    public var maxBackward: Int {
        confAttentionContextLeft > 0 ? confAttentionContextLeft - 1 : 0
    }

    /// Maximum forward span for relative position embeddings.
    /// Equal to confAttentionContextRight (0 for causal).
    public var maxForward: Int { confAttentionContextRight }

    /// Total context size for each attention chunk: chunk + left + right context.
    public var contextSize: Int {
        confAttentionChunkSize + maxBackward + maxForward
    }

    /// Total relative position span: maxBackward + maxForward + 1.
    public var maxSpanPlusOne: Int { maxBackward + maxForward + 1 }
}

// NOTE: Top-level multimodal configuration is defined in Gemma3nVLM.swift
// as Gemma3nConfiguration (includes text config, audio config, and model params).
