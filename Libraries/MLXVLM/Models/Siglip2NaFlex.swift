// Copyright © 2026 Apple Inc.
// Port of transformers' SigLIP2 vision model (models/siglip2/modeling_siglip2.py) and its
// NaFlex patch input. LFM2-VL uses it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// SigLIP2 NaFlex vision tower (no pooling head), as LFM2-VL uses it.
enum Siglip2NaFlex {

    /// Tower hyperparameters. LFM2-VL builds them from its `vision_config`.
    struct Configuration: Sendable, Equatable {
        var hiddenSize: Int
        var intermediateSize: Int
        var numHiddenLayers: Int
        var numAttentionHeads: Int
        var numChannels: Int
        var numPatches: Int
        var patchSize: Int
        var layerNormEps: Float
    }

    final class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(
                dims % numHeads == 0,
                "The input feature dimensions should be divisible by the number of heads")

            self.numHeads = numHeads
            self.scale = pow(Float(dims / numHeads), -0.5)

            self._qProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._kProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._vProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._outProj.wrappedValue = Linear(dims, dims, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let queries = qProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            let keys = kProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale,
                mask: mask.map { .array($0) } ?? .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        init(_ config: Configuration) {
            self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(geluTanh(fc1(x)))
        }
    }

    /// `gelu_pytorch_tanh` as torch computes it: bf16 and fp16 in float32, rounded once. Compiled
    /// with fixed shapes so the casts fuse into one kernel and no float32 array is allocated.
    static let geluTanh: @Sendable (MLXArray) -> MLXArray = compile { x in
        geluApproximate(x.asType(.float32)).asType(x.dtype)
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Configuration) {
            self._selfAttn.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numAttentionHeads, bias: true)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = MLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let h = x + selfAttn(layerNorm1(x), mask: mask)
            return h + mlp(layerNorm2(h))
        }
    }

    final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: Configuration) {
            self.layers = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
        }

        /// The final state, plus the input and every layer output when `outputHiddenStates`.
        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, outputHiddenStates: Bool = false)
            -> (last: MLXArray, hiddenStates: [MLXArray]?)
        {
            var states: [MLXArray]? = outputHiddenStates ? [x] : nil
            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
                states?.append(h)
            }
            return (h, states)
        }
    }

    final class VisionEmbeddings: Module {
        let positionEmbeddingSize: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: Configuration) {
            self.positionEmbeddingSize = Int(Double(config.numPatches).squareRoot())
            self._patchEmbedding.wrappedValue = Linear(
                config.numChannels * config.patchSize * config.patchSize, config.hiddenSize)
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: config.numPatches, dimensions: config.hiddenSize)
        }

        /// `resize_positional_embeddings`: the `[S, S, D]` table resized to each `(h, w)` with
        /// antialiased bilinear in float32, rows past `h * w` filled with the first row.
        static func resizePositionEmbeddings(
            _ table: MLXArray, spatialShapes: [(Int, Int)], maxLength: Int
        ) -> MLXArray {
            let (srcH, srcW, dims) = (table.dim(0), table.dim(1), table.dim(2))
            let sourceType = table.dtype
            let table32 = table.asType(.float32)

            let rows = spatialShapes.map { (h, w) -> MLXArray in
                precondition(h * w <= maxLength, "spatial shape \(h)x\(w) exceeds \(maxLength)")
                var resized = table32
                if (h, w) != (srcH, srcW) {
                    let wx = MLXArray(
                        antialiasedBilinearWeights(inputSize: srcW, outputSize: w), [w, srcW])
                    let wy = MLXArray(
                        antialiasedBilinearWeights(inputSize: srcH, outputSize: h), [h, srcH])
                    let alongWidth = matmul(wx, table32).reshaped(srcH, w * dims)
                    resized = matmul(wy, alongWidth)
                }
                resized = resized.reshaped(h * w, dims).asType(sourceType)
                if h * w < maxLength {
                    let fill = broadcast(resized[0 ..< 1], to: [maxLength - h * w, dims])
                    resized = concatenated([resized, fill], axis: 0)
                }
                return resized
            }
            return stacked(rows, axis: 0)
        }

        func callAsFunction(_ pixels: MLXArray, spatialShapes: [(Int, Int)]) -> MLXArray {
            let patchEmbeds = patchEmbedding(pixels.asType(patchEmbedding.weight.dtype))
            let table = positionEmbedding.weight.reshaped(
                positionEmbeddingSize, positionEmbeddingSize, -1)
            let positions = Self.resizePositionEmbeddings(
                table, spatialShapes: spatialShapes, maxLength: pixels.dim(1))
            return patchEmbeds + positions
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

        init(_ config: Configuration) {
            self.embeddings = VisionEmbeddings(config)
            self.encoder = Encoder(config)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        /// pixels `[B, P, C*p*p]`, mask `[B, P]` (1 = real patch) or nil, one `(h, w)` per row.
        /// Returns the post-layernorm last hidden state `[B, P, D]`, and the encoder input plus
        /// every layer output when `outputHiddenStates`.
        func callAsFunction(
            _ pixels: MLXArray, spatialShapes: [(Int, Int)], mask: MLXArray?,
            outputHiddenStates: Bool = false
        ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
            let x = embeddings(pixels, spatialShapes: spatialShapes)
                .asType(embeddings.patchEmbedding.weight.dtype)

            // Padded keys get -inf so real patches never attend to them.
            let additiveMask = mask.map {
                MLX.where($0.asType(.bool), MLXArray(Float(0)), MLXArray(-Float.infinity))
                    .asType(x.dtype)[0..., .newAxis, .newAxis, 0...]
            }

            let (last, hiddenStates) = encoder(
                x, mask: additiveMask, outputHiddenStates: outputHiddenStates)
            return (postLayernorm(last), hiddenStates)
        }

        /// The last hidden state `[h * w, D]` of each view in `pixels`, which may be padded past
        /// `h * w`. Views with the same patch count run as one batch cut to that count, without a
        /// mask. This is the same math as the padded batch with a mask, with less work.
        func viewStates(_ pixels: MLXArray, spatialShapes: [(Int, Int)]) -> [MLXArray] {
            let lengths = spatialShapes.map { $0.0 * $0.1 }
            var groups: [(length: Int, views: [Int])] = []
            for (view, length) in lengths.enumerated() {
                if let group = groups.firstIndex(where: { $0.length == length }) {
                    groups[group].views.append(view)
                } else {
                    groups.append((length, [view]))
                }
            }

            var states = [MLXArray?](repeating: nil, count: lengths.count)
            for (length, views) in groups {
                var batch = pixels[0..., 0 ..< length]
                if views.count < lengths.count {
                    let (first, last) = (views[0], views[views.count - 1])
                    if last - first + 1 == views.count {
                        batch = batch[first ..< (last + 1)]
                    } else {
                        batch = batch[MLXArray(views.map { Int32($0) })]
                    }
                }
                let hidden = self(
                    batch, spatialShapes: views.map { spatialShapes[$0] }, mask: nil
                ).lastHiddenState
                for (row, view) in views.enumerated() {
                    states[view] = hidden[row]
                }
            }
            return states.map { $0! }
        }
    }

    /// `[views, length]` int32, 1 for the first `h * w` patches of each view and 0 for padding,
    /// like transformers' `pixel_attention_mask`.
    static func patchMask(_ spatialShapes: [(Int, Int)], length: Int) -> MLXArray {
        var mask = [Int32](repeating: 0, count: spatialShapes.count * length)
        for (i, (h, w)) in spatialShapes.enumerated() {
            precondition(h * w <= length, "\(h * w) patches do not fit in \(length)")
            for j in 0 ..< h * w {
                mask[i * length + j] = 1
            }
        }
        return MLXArray(mask, [spatialShapes.count, length])
    }
}

// MARK: - Patches

/// The SigLIP2 NaFlex vision-tower input for a set of views.
struct Siglip2Patches {
    /// `[views, patches, 3 * patch * patch]` float32 in (py, px, c) order, zero padded.
    var pixels: MLXArray
    /// `[views, patches]` int32, 1 for real patches; see `Siglip2NaFlex.patchMask`.
    var mask: MLXArray
    /// `(rows, columns)` of patches per view.
    var spatialShapes: [(Int, Int)]
}

/// Patches for SigLIP2 NaFlex, padded to `maxPatches` per crop. Each crop is cut to whole
/// patches, as `convert_image_to_patches` does. `table[channel * 256 + byte]` normalises each
/// byte, as `LFM2VLImageSettings.normalizationTable()` makes it.
func siglip2Patches(
    _ crops: [RGB8Image], patchSize: Int, maxPatches: Int, table: [Float]
) -> Siglip2Patches {
    precondition(table.count == 3 * 256, "the table needs 768 entries")
    let dimensions = 3 * patchSize * patchSize
    var pixels = [Float](repeating: 0, count: crops.count * maxPatches * dimensions)
    var shapes: [(Int, Int)] = []
    for (i, crop) in crops.enumerated() {
        let rows = crop.height / patchSize
        let columns = crop.width / patchSize
        precondition(
            rows * columns <= maxPatches, "\(rows * columns) patches do not fit in \(maxPatches)")
        crop.bytes.withUnsafeBufferPointer { source in
            pixels.withUnsafeMutableBufferPointer { pixels in
                table.withUnsafeBufferPointer { table in
                    guard let src = source.baseAddress, let dst = pixels.baseAddress else {
                        return
                    }
                    let red = table.baseAddress!
                    let (green, blue) = (red + 256, red + 512)
                    // The view's values in (py, px, y, x, c) order are contiguous.
                    var d = i * maxPatches * dimensions
                    for py in 0 ..< rows {
                        for px in 0 ..< columns {
                            for y in 0 ..< patchSize {
                                var s = ((py * patchSize + y) * crop.width + px * patchSize) * 3
                                let end = s + patchSize * 3
                                while s < end {
                                    dst[d] = red[Int(src[s])]
                                    dst[d + 1] = green[Int(src[s + 1])]
                                    dst[d + 2] = blue[Int(src[s + 2])]
                                    d += 3
                                    s += 3
                                }
                            }
                        }
                    }
                }
            }
        }
        shapes.append((rows, columns))
    }
    return Siglip2Patches(
        pixels: MLXArray(pixels, [crops.count, maxPatches, dimensions]),
        mask: Siglip2NaFlex.patchMask(shapes, length: maxPatches), spatialShapes: shapes)
}
