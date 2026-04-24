// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Language

private enum Language {

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]      // cumulative section indices (for half-dim split)
        let mropeSectionRaw: [Int]   // raw section sizes [16, 24, 24] (for full-dim split)
        // Wrapped in a class to hide from Module's weight loading (not a trained weight)
        private class InvFreqBox { let value: MLXArray; init(_ v: MLXArray) { value = v } }
        private let _invFreqBox: InvFreqBox

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                // Raw sections e.g. [16, 24, 24] — used for splitting full-dim cos/sin
                self.mropeSectionRaw = array
                // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        // note the *2
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            // Compute inv_freq for MROPE (same formula as Python)
            // inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
            let freqIndices = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            let base = MLXArray(args.ropeTheta)
            self._invFreqBox = InvFreqBox(1.0 / pow(base, freqIndices / Float(headDim)))

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        /// Compute cos/sin for MROPE from 3D position IDs
        /// Matches Python apply_mrope: start with temporal, overwrite H/W ranges
        private func mropeCosSin(positionIds: MLXArray) -> (MLXArray, MLXArray) {
            // positionIds: [3, batch, seq]
            let invFreqExpanded = _invFreqBox.value.reshaped(1, 1, -1, 1)  // [1, 1, dim/2, 1]
            let posExpanded = positionIds[0..., 0..., .newAxis, 0...].asType(.float32)  // [3, batch, 1, seq]
            var freqs = matmul(invFreqExpanded, posExpanded)  // [3, batch, dim/2, seq]
            freqs = freqs.transposed(0, 1, 3, 2)  // [3, batch, seq, dim/2]

            // Python apply_mrope: start with temporal (freqs[0]), overwrite H/W slices
            // mropeSectionRaw = [16, 24, 24]
            // dims 0..15 → temporal, dims 16..39 → height, dims 40..63 → width
            var result = freqs[0]  // [batch, seq, dim/2] — start with temporal
            var offset = mropeSectionRaw[0]  // 16
            for dim in 1 ..< mropeSectionRaw.count {
                let length = mropeSectionRaw[dim]
                // Replace slice [offset..<offset+length] with freqs from this dimension
                let dimFreqs = freqs[dim]  // [batch, seq, dim/2]
                // Use scatter or manual slice replacement
                let before = result[0..., 0..., 0 ..< offset]
                let middle = dimFreqs[0..., 0..., offset ..< (offset + length)]
                let after: MLXArray
                if offset + length < result.dim(-1) {
                    after = result[0..., 0..., (offset + length)...]
                    result = concatenated([before, middle, after], axis: -1)
                } else {
                    result = concatenated([before, middle], axis: -1)
                }
                offset += length
            }

            let emb = concatenated([result, result], axis: -1)  // [batch, seq, dim]
            return (MLX.cos(emb), MLX.sin(emb))
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            if let positionIds {
                // MROPE path: compute 3D-aware cos/sin from position IDs
                let (cosValues, sinValues) = mropeCosSin(positionIds: positionIds)
                // Apply rotary embedding: expand dims for head broadcasting
                let cos = cosValues[.newAxis, 0..., 0..., 0...]  // [1, batch, seq, dim]
                let sin = sinValues[.newAxis, 0..., 0..., 0...]
                let qEmbed = (queries * cos) + (QwenVL.rotateHalf(queries) * sin)
                let kEmbed = (keys * cos) + (QwenVL.rotateHalf(keys) * sin)
                queries = qEmbed
                keys = kEmbed
            } else {
                // Simple sequential RoPE for autoregressive generation
                let offset = cache?.offset ?? 0
                queries = rotaryEmbedding(queries, offset: offset)
                keys = rotaryEmbedding(keys, offset: offset)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen25Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen25VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen25VLDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen25Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        // MROPE state: stored position IDs from prefill, reused during autoregressive generation
        var _positionIds: MLXArray?
        var _ropeDeltas: MLXArray?

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self.model = Qwen25Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil
        ) -> LMOutput {
            var effectivePositionIds = positionIds ?? _positionIds

            // Clear stored position IDs after first use (prefill done)
            if _positionIds != nil { _positionIds = nil }

            // During autoregressive generation after multimodal prefill:
            // compute position IDs using rope_deltas so MROPE offsets are correct
            if effectivePositionIds == nil, let ropeDeltas = _ropeDeltas, let cache {
                let input = inputs ?? inputEmbedding!
                let batch = input.dim(0)
                let seqLength = input.dim(1)
                let lastCacheOffset = cache.last?.offset ?? 0

                var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                var base = MLXArray(0 ..< seqLength).asType(.int32)
                base = base[.newAxis, 0...]
                base = broadcast(base, to: [batch, seqLength])

                if delta.dim(0) == 1 && batch > 1 {
                    delta = repeated(delta, count: batch, axis: 0)
                }

                base = base + delta

                effectivePositionIds = base[.newAxis, 0..., 0...]
                effectivePositionIds = broadcast(effectivePositionIds!, to: [3, batch, seqLength])
            }

            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding,
                           positionIds: effectivePositionIds)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {

    static fileprivate func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (QwenVL.rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    fileprivate class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    fileprivate class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)

            // Apply attention mask (window or full attention)
            // attentionMask is [1, seqLen, seqLen] boolean — convert to float16 mask for SDPA
            let boolMask = attentionMask[.newAxis, 0..., 0..., 0...]  // [1, 1, seqLen, seqLen]
            let floatMask = MLX.where(boolMask,
                                       MLXArray(Float16(0)),
                                       MLXArray(Float16(-10000)))

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .array(floatMask)
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numHeads)

            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVL.PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullattBlockIndexes: [Int]

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize
            self.windowSize = config.windowSize
            self.patchSize = config.patchSize
            self.spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            self.fullattBlockIndexes = config.fullattBlockIndexes

            self._patchEmbed.wrappedValue = QwenVL.PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDimensions = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = QwenVL.VisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let (gridT, gridH, gridW) = frame.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize

                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW)).reshaped(
                    gridT, llmGridH, llmGridW)

                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize

                // Pad the index
                let indexPadded = padded(
                    index,
                    widths: [[0, 0], [0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100)
                )

                // Reshape and transpose
                let indexReshaped = indexPadded.reshaped(
                    gridT,
                    numWindowsH,
                    vitMergerWindowSize,
                    numWindowsW,
                    vitMergerWindowSize
                )

                let indexTransposed = indexReshaped.transposed(0, 1, 3, 2, 4).reshaped(
                    gridT,
                    numWindowsH * numWindowsW,
                    vitMergerWindowSize,
                    vitMergerWindowSize
                )

                // Calculate sequence lengths
                let seqlens = sum(indexTransposed .!= -100, axes: [2, 3]).reshaped(-1)

                // Get valid indices
                let indexFlattened = indexTransposed.flattened()
                let validIndices = indexFlattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map { $0.offset }

                let validValues = indexFlattened[MLXArray(validIndices)]

                // Add to window index
                windowIndex.append(validValues + windowIndexId)

                // Update cumulative sequence lengths
                let cuSeqlensTmp =
                    cumsum(seqlens, axis: 0) * spatialMergeUnit + cuWindowSeqlens.last!
                cuWindowSeqlens.append(contentsOf: cuSeqlensTmp.asArray(Int.self))

                windowIndexId += gridT * llmGridH * llmGridW
            }

            // Concatenate all window indices
            let combinedWindowIndex = concatenated(windowIndex, axis: 0)
            let cuWindowSeqlensArray = MLXArray(cuWindowSeqlens)

            // Get unique values in cuWindowSeqlens
            var seen = Set<Int>()
            var uniqueIndices = [Int]()

            for (i, value) in cuWindowSeqlens.enumerated() {
                if !seen.contains(value) {
                    seen.insert(value)
                    uniqueIndices.append(i)
                }
            }

            let uniqueCuWindowSeqlens = cuWindowSeqlensArray[MLXArray(uniqueIndices)]

            return (combinedWindowIndex, uniqueCuWindowSeqlens)
        }

        func attentionMask(sequenceLength: Int, cuSeqlens: MLXArray) -> MLXArray {
            // Create attention mask
            let attentionMask = full(
                [1, sequenceLength, sequenceLength],
                values: false)

            // Update mask for each sequence
            let cuSeqlens = cuSeqlens.asArray(Int.self)
            for i in 1 ..< cuSeqlens.count {
                let start = cuSeqlens[i - 1]
                let end = cuSeqlens[i]
                attentionMask[0..., start ..< end, start ..< end] = MLXArray(true)
            }

            return attentionMask
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPosEmb = rotaryPositionEmbedding(frames)

            // Get window indices and sequence lengths
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)

            // prepare attention masks
            let seqLen = hiddenStates.dim(0)
            var cuSeqlens = [0]
            for frame in frames {
                let seqLen = frame.h * frame.w
                cuSeqlens.append(
                    contentsOf: Array(repeating: seqLen, count: frame.t).map {
                        cuSeqlens.last! + $0
                    })
            }
            let cuSeqlensArray = MLXArray(cuSeqlens)

            let fullAttentionMask = attentionMask(sequenceLength: seqLen, cuSeqlens: cuSeqlensArray)
            let windowAttentionMask = attentionMask(
                sequenceLength: seqLen, cuSeqlens: cuWindowSeqlens)

            // Reshape and reindex hidden states
            hiddenStates = hiddenStates.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            hiddenStates = hiddenStates[windowIndex, 0..., 0...]
            hiddenStates = hiddenStates.reshaped(seqLen, -1)

            // Reshape and reindex rotary position embeddings
            var rotaryPosEmbReshaped = rotaryPosEmb.reshaped(
                seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            rotaryPosEmbReshaped = rotaryPosEmbReshaped[windowIndex, 0..., 0...]
            rotaryPosEmbReshaped = rotaryPosEmbReshaped.reshaped(seqLen, -1)

            // Process through blocks
            for (i, block) in blocks.enumerated() {
                // Use full attention for specific blocks, window attention for others
                let attentionMask =
                    fullattBlockIndexes.contains(i) ? fullAttentionMask : windowAttentionMask

                hiddenStates = block(
                    hiddenStates,
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPosEmbReshaped
                )
            }

            // Apply patch merger
            hiddenStates = patchMerger(hiddenStates)

            // Reorder back to original sequence
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// Qwen2.5VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen25VL`` and is typically created by ``VLMModelFactory``.
public struct Qwen25VLProcessor: UserInputProcessor {
    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .lanczos)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // Compute target size from the original image before any processing, matching Python's
        // single-step resize: ratio = max_size / max(w,h); new_w = int(w*ratio)//28*28
        let origSize = images[0].extent.size
        let factor = config.patchSize * config.mergeSize
        let targetSize: CGSize
        if let resize = processing?.resize {
            targetSize = resize
        } else {
            let (h, w) = try QwenVL.targetSize(
                height: Int(origSize.height), width: Int(origSize.width),
                factor: factor, minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
            targetSize = CGSize(width: w, height: h)
        }

        let processedImages = images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map { MediaProcessing.resampleBicubic($0, to: targetSize) }
            .map { MediaProcessing.normalize($0, mean: config.imageMeanTuple, std: config.imageStdTuple) }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })

            if let imageFrames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[MLXArray]]()
            for video in input.videos {

                var resizedSize: CGSize = .zero

                let imageSequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { _ in Double(2) }
                ) { frame in
                    // first apply the user requested resizing, etc. if any
                    let resizedImage = MediaProcessing.apply(
                        frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let size = resizedImage.extent.size
                        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                            height: Int(size.height), width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels, maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
                    }
                    let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
                    return VideoFrame(frame: processedImage, timeStamp: frame.timeStamp)
                }

                videosAsImageSequences.append(imageSequence.frames)
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
}

// MARK: - Model

/// Qwen2.5VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class Qwen25VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen25VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Qwen25VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            // Text-only: reset MROPE state
            languageModel._positionIds = nil
            languageModel._ropeDeltas = nil
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Reset MROPE state when processing new image (matches Python)
        languageModel._positionIds = nil
        languageModel._ropeDeltas = nil

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the output hidden states from the vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        let mergedEmbeds = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)

        // Compute MROPE 3D position IDs for spatial awareness
        let spatialMergeSize = config.visionConfiguration.spatialMergeSize
        let inputIds2D = inputIds.ndim == 1 ? inputIds[.newAxis, 0...] : inputIds
        let (positionIds, ropeDeltas) = Qwen25VL.getRopeIndex(
            inputIds: inputIds2D,
            imageGridTHW: frames,
            videoGridTHW: nil,
            spatialMergeSize: spatialMergeSize,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId,
            visionStartTokenId: config.baseConfiguration.visionStartTokenId)
        languageModel._positionIds = positionIds
        languageModel._ropeDeltas = ropeDeltas

        return mergedEmbeds
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        // Process both images and videos together
        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            if allPixels == nil {
                allPixels = videoPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, videoPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: videoFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    /// Compute MROPE 3D position IDs for image/video spatial encoding
    /// Ported from Qwen3VL.swift getRopeIndex() and Python get_rope_index()
    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
            positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])
            let positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            let zeros = MLXArray.zeros([batchSize], dtype: .int32)
            return (positionIds3D, zeros)
        }

        var positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            let imageNums = ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()).item(Int.self)
            let videoNums = ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()).item(Int.self)

            let inputTokens = batchInputIds.asArray(Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            for _ in 0 ..< (imageNums + videoNums) {
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    stIdx = lastArray.max().item(Int.self) + 1
                } else {
                    stIdx = 0
                }

                // Text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // 3D position IDs for visual tokens (temporal, height, width)
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    stIdx = lastArray.max().item(Int.self) + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            if !llmPosIdsList.isEmpty {
                let llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]

                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                positionIds = newPositions

                let maxPosId = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(
            weights:
                Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        var key = key
                        if !key.contains("vision_tower") {
                            key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                        }
                        if !key.contains("language_model") {
                            key = key.replacingOccurrences(
                                of: "model", with: "language_model.model")
                            key = key.replacingOccurrences(
                                of: "lm_head", with: "language_model.lm_head")
                        }

                        return (key, value)
                    })
        )
    }
}

// MARK: - Configuration

/// Configuration for ``Qwen25VL``
public struct Qwen25VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int { _maxPositionEmbeddings ?? 128000 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
        private let _slidingWindow: Int?
        public var slidingWindow: Int { _slidingWindow ?? 32768 }
        private let _useSlidingWindow: Bool?
        public var useSlidingWindow: Bool { _useSlidingWindow ?? false }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _slidingWindow = "sliding_window"
            case _useSlidingWindow = "use_sliding_window"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        private let _inChans: Int?
        public var inChannels: Int { _inChans ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullattBlockIndexes: [Int]
        public let tokensPerSecond: Int
        private let _skipVision: Bool?
        public var skipVision: Bool { _skipVision ?? false }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "silu" }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case _inChans = "in_chans"
            case _layerNormEps = "layer_norm_eps"  // Added this line
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
            case tokensPerSecond = "tokens_per_second"
            case _skipVision = "skip_vision"
            case _hiddenAct = "hidden_act"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let visionStartTokenId: Int
        public let visionEndTokenId: Int
        public let visionTokenId: Int
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numKeyValueHeads: Int
        public let slidingWindow: Int
        public let useSlidingWindow: Bool
        public let maxWindowLayers: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case visionStartTokenId = "vision_start_token_id"
            case visionEndTokenId = "vision_end_token_id"
            case visionTokenId = "vision_token_id"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numKeyValueHeads = "num_key_value_heads"
            case slidingWindow = "sliding_window"
            case useSlidingWindow = "use_sliding_window"
            case maxWindowLayers = "max_window_layers"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // this is a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // these are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``Qwen25VLProcessor``
public struct Qwen25VLProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var size: Size {
        Size(maxPixels: maxPixels, minPixels: minPixels)
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
    }
}
