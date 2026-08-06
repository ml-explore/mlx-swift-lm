import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_vl

// MARK: - Common Utilities for Qwen 2 VL and Qwen 2.5 VL

private func debug(_ message: @autoclosure () -> String) {
    // print(message())
}

public struct QwenVL {
    /// Rotates half the hidden dims of the input
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let index = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< index]
        let x2 = x[.ellipsis, index...]
        return concatenated([-x2, x1], axis: -1)
    }

    static func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray,
        imageTokenId: Int, videoTokenId: Int
    ) -> MLXArray {
        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenId || v == videoTokenId {
                imageIndices.append(i)
            }
        }

        // Make sure shapes match before assignment
        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures[.newAxis, 0..., 0...]
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float) {
            self.dimensions = dimensions
            self.theta = theta
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            self.inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            let freqs = outer(seq, inverseFreq)
            return freqs
        }
    }

    public class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let outputDimensions: Int

        // For Qwen 2 VL
        convenience init(
            patchSize: Int, temporalPatchSize: Int, inChannels: Int, embedDimensions: Int
        ) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: embedDimensions)
        }

        // For Qwen 2.5 VL
        convenience init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, hiddenSize: Int) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: hiddenSize)
        }

        // Common initializer
        init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, outputDimensions: Int) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.outputDimensions = outputDimensions

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: outputDimensions,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: false
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            hiddenStates = proj(hiddenStates)
            hiddenStates = hiddenStates.reshaped(-1, outputDimensions)
            return hiddenStates
        }
    }

    // image_processing_qwen2_vl.smart_resize
    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws
        -> (Int, Int)
    {
        debug("Original dimensions: \(width) × \(height)")
        debug("Factor: \(factor), minPixels: \(minPixels), maxPixels: \(maxPixels)")

        if height < factor {
            throw VLMError.imageProcessingFailure(
                "Height: \(height) must be larger than factor: \(factor)")
        }
        if width < factor {
            throw VLMError.imageProcessingFailure(
                "Width: \(width) must be larger than factor: \(factor)")
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }

        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)
        debug("After rounding to factor multiples: \(wBar) × \(hBar)")

        // Scale based on total pixel count
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
            debug("After scaling down for maxPixels: \(wBar) × \(hBar)")
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
            debug("After scaling up for minPixels: \(wBar) × \(hBar)")
        }

        // Ensure dimensions are divisible by the factor
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor
        debug("Final dimensions: \(wBar) × \(hBar)")
        debug("Total pixels: \(wBar * hBar)")

        // Final sanity check
        if hBar <= 0 || wBar <= 0 {
            throw VLMError.imageProcessingFailure(
                "Invalid target dimensions: \(wBar) × \(hBar)")
        }

        return (hBar, wBar)
    }

    static func replacePaddingTokens(
        in promptTokens: [Int], frames: [THW], paddingToken: String, mergeSize: Int,
        tokenizer: any Tokenizer
    ) throws -> [Int] {
        // Replace single padding token with correct number for each image or video frame
        let placeholderTokens = tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>")
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens does not match number of frames")
        }
        let mergeLength = mergeSize * mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>"
            )
        }
        // Build the final array
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            // Add tokens before the placeholder
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            // Add replacement sequence
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        // Add any remaining tokens after the last replacement
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    static func patchify(images: [MLXArray], mergeSize: Int, patchSize: Int, temporalPatchSize: Int)
        throws -> (
            MLXArray, THW
        )
    {
        guard let firstImage = images.first else {
            throw VLMError.imageProcessingFailure("No images in video sequence")
        }
        let resizedHeight = firstImage.dim(-2)
        let resizedWidth = firstImage.dim(-1)
        var patches = concatenated(images)

        // Pad to match temporal patch size if needed
        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let lastPatchRepeated = tiled(
                lastPatch, repetitions: [temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, lastPatchRepeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT,
            temporalPatchSize,
            channel,
            gridH / mergeSize,
            mergeSize,
            patchSize,
            gridW / mergeSize,
            mergeSize,
            patchSize
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let flattenedPatches = patches.reshaped(
            gridT * gridH * gridW,
            channel * temporalPatchSize * patchSize * patchSize
        )

        return (flattenedPatches, .init(gridT, gridH, gridW))
    }

    // MARK: - Prepared input splitting

    /// Split a Qwen VL prepared input at `prefixTokenCount`, keeping only the media
    /// whose placeholder tokens fall in the suffix.
    ///
    /// Shared by ``Qwen25VL`` and ``Qwen2VL``, whose prepared-input layout is the
    /// same: `pixels` is the images' patch rows concatenated along axis 0 in prompt
    /// order, and `frames` carries one `THW` per image, so image *i* owns
    /// `frames[i].product` rows and `frames[i].product / (mergeSize * mergeSize)`
    /// placeholder tokens (see ``patchify(images:mergeSize:patchSize:temporalPatchSize:)``
    /// and ``replacePaddingTokens(in:frames:paddingToken:mergeSize:tokenizer:)``).
    ///
    /// Returns `nil` — meaning "fall back to a full prefill" — unless every
    /// assumption above is checked against the input actually handed in.
    static func splitPreparedInput(
        _ input: LMInput,
        droppingFirst prefixTokenCount: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        mergeSize: Int
    ) -> LMInput? {
        // Audio is not part of the Qwen VL prepared-input contract.
        guard input.audio == nil else { return nil }

        // Images only. Video is refused for two reasons, both checked against the
        // encoders rather than assumed:
        //
        //  * ``Qwen25VL``'s vision full-attention mask mis-accumulates `cuSeqlens`
        //    for frames with `t > 1` — `cuSeqlens.last!` is read inside a `map`
        //    that has not appended yet, so a 3-frame block yields `[16, 16, 16]`
        //    rather than `[16, 32, 48]`. A video's rows are therefore not isolated
        //    from the rest of the buffer, and dropping earlier rows changes the
        //    features of the rows that are kept.
        //  * Both models pass `videoGridTHW: nil` to `getRopeIndex`, which makes
        //    video positions degenerate.
        //
        // Neither is this change's to fix, but either one makes a video split
        // unsound, so it is not offered.
        guard input.video == nil else { return nil }

        // Only the processor's own `[1, seq]` layout. A rank-1 suffix would route
        // back to the cold path in `prepare(_:cache:state:windowSize:)`, which
        // computes positions from zero — silently wrong against a warm cache.
        let tokens = input.text.tokens
        guard tokens.ndim == 2, tokens.dim(0) == 1 else { return nil }

        let ids = tokens.asArray(Int.self)
        guard prefixTokenCount > 0, prefixTokenCount < ids.count else { return nil }

        // The processor pairs media with an all-ones mask. Anything else is a real
        // padding mask this routine will not reinterpret.
        if let mask = input.text.mask {
            guard mask.size == ids.count else { return nil }
            guard mask.asType(.int32).asArray(Int32.self).allSatisfy({ $0 == 1 }) else {
                return nil
            }
        }

        var splitImage: LMInput.ProcessedImage?
        if let image = input.image {
            guard
                let split = splitVisionPayload(
                    pixels: image.pixels, positionIds: image.positionIds, frames: image.frames,
                    ids: ids, prefixTokenCount: prefixTokenCount, padTokenId: imageTokenId,
                    mergeSize: mergeSize)
            else { return nil }
            splitImage = LMInput.ProcessedImage(pixels: split.pixels, frames: split.frames)
        }

        // A video placeholder inside an image-only payload would mean the prompt
        // carries media this routine is not accounting for.
        guard !ids.contains(videoTokenId) else { return nil }

        let suffixIds = Array(ids[prefixTokenCount...])
        let suffixTokens = MLXArray(suffixIds).expandedDimensions(axis: 0)
        let suffixMask = input.text.mask.map { mask in
            ones([1, suffixIds.count]).asType(mask.dtype)
        }

        return LMInput(
            text: .init(tokens: suffixTokens, mask: suffixMask),
            image: splitImage)
    }

    /// Drop the leading media items that the cached prefix already covers.
    ///
    /// Returns `nil` when the payload cannot be attributed item-by-item, or when the
    /// cut falls *inside* a media block — in that case the suffix would carry a
    /// partial set of placeholders and neither the feature merge nor the position
    /// walk would line up.
    private static func splitVisionPayload(
        pixels: MLXArray,
        positionIds: MLXArray?,
        frames: [THW]?,
        ids: [Int],
        prefixTokenCount: Int,
        padTokenId: Int,
        mergeSize: Int
    ) -> (pixels: MLXArray, frames: [THW])? {
        // Precomputed position ids describe the full prompt; slicing them is not
        // attempted here.
        guard positionIds == nil else { return nil }
        guard let frames, !frames.isEmpty else { return nil }

        let mergeLength = mergeSize * mergeSize
        guard mergeLength > 0 else { return nil }

        var prefixPadCount = 0
        var totalPadCount = 0
        for (index, id) in ids.enumerated() where id == padTokenId {
            totalPadCount += 1
            if index < prefixTokenCount { prefixPadCount += 1 }
        }

        var padCounts: [Int] = []
        var rowCounts: [Int] = []
        padCounts.reserveCapacity(frames.count)
        rowCounts.reserveCapacity(frames.count)
        for frame in frames {
            guard frame.product > 0, frame.product % mergeLength == 0 else { return nil }
            padCounts.append(frame.product / mergeLength)
            rowCounts.append(frame.product)
        }

        // The payload must describe exactly the placeholders in the prompt, and the
        // rows must account for the whole pixel buffer. If either is off, the layout
        // is not the one documented above and nothing here may be assumed.
        guard padCounts.reduce(0, +) == totalPadCount else { return nil }
        guard pixels.ndim == 2, pixels.dim(0) == rowCounts.reduce(0, +) else { return nil }

        var consumedPads = 0
        var consumedRows = 0
        var droppedItems = 0
        while droppedItems < frames.count, consumedPads < prefixPadCount {
            consumedPads += padCounts[droppedItems]
            consumedRows += rowCounts[droppedItems]
            droppedItems += 1
        }
        // Overshoot means the boundary landed inside a media block.
        guard consumedPads == prefixPadCount else { return nil }

        let remainingFrames = Array(frames[droppedItems...])
        guard !remainingFrames.isEmpty else { return nil }

        let remainingPixels = pixels[consumedRows ..< pixels.dim(0), 0...]
        return (remainingPixels, remainingFrames)
    }

}
