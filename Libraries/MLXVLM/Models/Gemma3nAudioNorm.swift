//
//  Gemma3nAudioNorm.swift
//  mlx-swift-examples
//
//  Cumulative group normalization for streaming-compatible audio processing
//  in the Gemma 3n Conformer encoder.
//
//  Reference: mlx_vlm/models/gemma3n/audio.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
