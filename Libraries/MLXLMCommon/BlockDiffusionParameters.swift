// Copyright © 2026 Apple Inc.

/// Request-level controls for block-diffusion generation.
///
/// Model-level denoising constants remain on ``BlockDiffusionLanguageModel``;
/// these values are the caller's per-generation overrides.
public struct BlockDiffusionParameters: Sendable, Equatable {

    /// Strategy used to reveal tokens while denoising a canvas.
    public enum Sampler: Sendable, Equatable {
        /// Reveals the lowest-entropy tokens within the model's entropy budget.
        case entropyBound

        /// Reveals tokens whose sampled probability reaches `threshold`.
        case confidenceThreshold(threshold: Float = 0.9)
    }

    /// Policy used to choose the size of each denoising canvas.
    public enum Canvas: Sendable, Equatable {
        /// Adapts canvas length to the remaining token budget and model defaults.
        ///
        /// Bounds are clamped to the model's supported canvas length. A `nil`
        /// bound uses the corresponding model default.
        case adaptive(minimumLength: Int? = nil, maximumLength: Int? = nil)

        /// Always denoises a full model-size canvas.
        case full
    }

    /// Strategy used to reveal tokens while denoising a canvas.
    public var sampler: Sampler

    /// Denoiser sampling temperature, independently of autoregressive sampling.
    /// A value of `0` selects the most likely token.
    public var temperature: Float

    /// Policy used to choose the size of each denoising canvas.
    public var canvas: Canvas

    public init(
        sampler: Sampler = .entropyBound,
        temperature: Float = 0,
        canvas: Canvas = .adaptive()
    ) {
        self.sampler = sampler
        self.temperature = temperature
        self.canvas = canvas
    }
}
