//
//  Gemma3nAudioConv.swift
//  mlx-swift-examples
//
//  Convolutional subsampling and projection layers for the Gemma 3n
//  Conformer audio encoder. Handles initial mel-spectrogram feature
//  extraction and downsampling.
//
//  Reference: mlx_vlm/models/gemma3n/audio.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - SubSample Conv Block

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

// MARK: - SubSample Conv Projection

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
