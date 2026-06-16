//
//  Gemma3Text+AllHiddenStates.swift
//  mlx-swift-lm
//
//  Encoder-style multi-layer hidden-state extraction for Gemma 3 text models.
//  Supports using Gemma 3 as a frozen text ENCODER — a pattern in multimodal
//  generation where a downstream model conditions on every layer's hidden state
//  rather than on generated tokens (e.g. Lightricks LTX-2 conditions its video
//  DiT on all 49 Gemma-3-12B hidden states).
//

import Foundation
import MLX
import MLXLMCommon

extension Gemma3Model {
    /// Returns the embedding output plus each transformer layer's output —
    /// `numHiddenLayers + 1` states, each shaped `(B, T, hiddenSize)`.
    ///
    /// A SINGLE uniform `mask` is applied to every layer. This is the
    /// text-encoder use: the caller supplies a combined causal+padding mask and
    /// the per-layer sliding-window/global mask selection used by
    /// `callAsFunction(_:mask:cache:)` is intentionally bypassed.
    public func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> [MLXArray] {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        var states: [MLXArray] = [h]
        for layer in layers {
            h = layer(h, mask: mask, cache: nil)
            // Per-layer materialization keeps each Metal command buffer small so
            // long-sequence encodes on large checkpoints stay under the macOS GPU
            // watchdog — without it, all layers fuse into one dispatch.
            eval(h)
            states.append(h)
        }
        return states
    }
}

extension Gemma3TextModel {
    /// See ``Gemma3Model/allHiddenStates(_:mask:)``. Convenience forwarding from
    /// the top-level text model to its inner `model`.
    public func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> [MLXArray] {
        model.allHiddenStates(inputs, mask: mask)
    }
}
