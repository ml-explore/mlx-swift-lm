// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Shared runtime support for the dense and mixture-of-experts LFM2 families.
///
/// Both architectures interleave attention with a compact causal convolution.
/// Keeping their cache construction and convolution-state commits identical
/// prevents the two implementations from drifting on long-context, ragged-batch,
/// and prompt-cache behavior.
enum LFM2RuntimeSupport {
    /// Maximum verifier tail retained by every convolution layer. DSpark
    /// advertises this same bound before selecting the in-place hybrid-cache
    /// path, so cache construction and speculative rollback cannot drift.
    static let speculativeRollbackCapacity = 64

    static func makeHybridCache(
        hiddenLayers: Int,
        fullAttentionIndices: Set<Int>,
        convolutionStateLength: Int,
        parameters: GenerateParameters?
    ) throws -> [KVCache] {
        let attentionCache = try makeAttentionCache(parameters: parameters)
        return (0 ..< hiddenLayers).map { layerIndex in
            if fullAttentionIndices.contains(layerIndex) {
                attentionCache.copy()
            } else {
                RewindableConvolutionCache(
                    stateLength: convolutionStateLength,
                    rollbackCapacity: speculativeRollbackCapacity)
            }
        }
    }

    /// Prepends the previous convolution state and atomically commits the next
    /// state. For ragged batches, each row is committed at its logical sequence
    /// endpoint rather than at the padded physical endpoint.
    static func convolutionTimeline(
        input: MLXArray,
        stateLength: Int,
        hiddenSize: Int,
        cache: MambaCache?
    ) -> MLXArray {
        let initialState =
            cache?[0]
            ?? MLXArray.zeros(
                [input.dim(0), stateLength, hiddenSize], dtype: input.dtype)
        let timeline = concatenated([initialState, input], axis: 1)

        guard let cache else { return timeline }

        let sequenceLength = input.dim(1)
        let nextState: MLXArray
        if let lengths = cache.currentLengths {
            let ends = clip(lengths, min: 0, max: sequenceLength)
            let positions =
                (ends[0..., .newAxis] + MLXArray(0 ..< stateLength))[.ellipsis, .newAxis]
            nextState = contiguous(MLX.takeAlong(timeline, positions, axis: 1))
        } else {
            nextState = contiguous(timeline[0..., (timeline.dim(1) - stateLength)..., 0...])
        }

        if let cache = cache as? RewindableConvolutionCache {
            cache.record(
                input: input,
                initialState: initialState,
                currentState: nextState)
        } else {
            cache[0] = nextState
            cache.advance(sequenceLength)
        }
        return timeline
    }

    private static func makeAttentionCache(parameters: GenerateParameters?) throws -> any KVCache {
        if let capacity = try parameters?.effectiveKVCacheCapacity() {
            return capacity.makeRotatingCache()
        }

        if let (bits, groupSize) = resolveKVQuantizationParameters(parameters),
            parameters?.quantizedKVStart == 0
        {
            return QuantizedKVCache(groupSize: groupSize, bits: bits)
        }

        return KVCacheSimple()
    }

    private static func resolveKVQuantizationParameters(_ parameters: GenerateParameters?)
        -> (bits: Int, groupSize: Int)?
    {
        if case .affine(let configuration) = parameters?.kvCache?.strategy.storage {
            guard configuration.compressionStart == 0 else { return nil }
            return (configuration.bits, configuration.groupSize)
        }
        if let scheme = parameters?.kvScheme, let resolved = resolveAffineScheme(scheme) {
            return resolved
        }
        if let bits = parameters?.kvBits {
            return (bits, parameters?.kvGroupSize ?? 64)
        }
        return nil
    }
}
