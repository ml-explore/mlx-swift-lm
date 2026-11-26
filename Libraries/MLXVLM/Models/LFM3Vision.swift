// Shared visual encoder for LiquidAI LFM3-VL and LFM3-Omni.
// Reference: LiquidAI/LFM3-VL-2B, revision 71d8e745d97c224984e0f20412f532a5ce9dfd3e.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LFM3VisionConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let heads: Int
    let kvHeads: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int
    let channels: Int
    let kernelSize: Int
    let normEps: Float
    let layerTypes: [String]
    let rope: Rope
    let patchDimension: Int

    struct Rope: Decodable, Sendable {
        let theta: Float
        let type: String
        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
            case type = "rope_type"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case mergeSize = "spatial_merge_size"
        case channels = "in_channels"
        case kernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case layerTypes = "layer_types"
        case rope = "rope_parameters"
        case layerCount = "num_hidden_layers"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        temporalPatchSize = try c.decode(Int.self, forKey: .temporalPatchSize)
        mergeSize = try c.decode(Int.self, forKey: .mergeSize)
        channels = try c.decode(Int.self, forKey: .channels)
        kernelSize = try c.decode(Int.self, forKey: .kernelSize)
        normEps = try c.decode(Float.self, forKey: .normEps)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        rope = try c.decode(Rope.self, forKey: .rope)
        let count = try c.decode(Int.self, forKey: .layerCount)
        guard hiddenSize > 0, hiddenSize <= Int.max / 3, intermediateSize > 0, heads > 0,
            kvHeads > 0,
            hiddenSize % heads == 0, heads % kvHeads == 0,
            (hiddenSize / heads) % 4 == 0, patchSize > 0, temporalPatchSize > 0,
            mergeSize > 0, channels > 0, kernelSize > 0, kernelSize % 2 == 1,
            normEps.isFinite, normEps > 0, rope.theta.isFinite, rope.theta > 0,
            rope.type == "axial", count > 0, count == layerTypes.count,
            layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid LFM3 vision configuration"))
        }
        var dimension = channels
        for factor in [temporalPatchSize, patchSize, patchSize] {
            let (product, overflow) = dimension.multipliedReportingOverflow(by: factor)
            guard !overflow else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "LFM3 patch dimension overflows"))
            }
            dimension = product
        }
        patchDimension = dimension
    }
}

private final class LFM3VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: RMSNorm
    let heads: Int
    let kvHeads: Int
    let headDim: Int

    init(_ c: LFM3VisionConfiguration) {
        heads = c.heads
        kvHeads = c.kvHeads
        headDim = c.hiddenSize / c.heads
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _key.wrappedValue = Linear(c.hiddenSize, c.kvHeads * headDim, bias: false)
        _value.wrappedValue = Linear(c.hiddenSize, c.kvHeads * headDim, bias: false)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
    }

    private func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let half = headDim / 2
        let rotated = concatenated([-f[.ellipsis, half...], f[.ellipsis, ..<half]], axis: -1)
        return (f * cosine + rotated * sine).asType(x.dtype)
    }

    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let q = rotate(queryNorm(query(x).reshaped(n, heads, headDim)), cosine: cosine, sine: sine)
        let k = rotate(keyNorm(key(x).reshaped(n, kvHeads, headDim)), cosine: cosine, sine: sine)
        let v = value(x).reshaped(n, kvHeads, headDim)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: 1 / sqrt(Float(headDim)), mask: .none)
        return output(attended.transposed(0, 2, 1, 3).reshaped(n, -1))
    }
}

private final class LFM3VisionConvolution: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "spatial_conv") var convolution: Conv2d
    let merge: Int

    init(_ c: LFM3VisionConfiguration) {
        merge = c.mergeSize
        _input.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: false)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _convolution.wrappedValue = Conv2d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: .init(c.kernelSize), padding: .init(c.kernelSize / 2),
            groups: c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let parts = input(x).split(parts: 3, axis: -1)
        let gated = (parts[0] * parts[2])
            .reshaped(height / merge, width / merge, merge, merge, -1)
            .transposed(0, 2, 1, 3, 4).reshaped(1, height, width, -1)
        let filtered = convolution(gated)
            .reshaped(height / merge, merge, width / merge, merge, -1)
            .transposed(0, 2, 1, 3, 4).reshaped(height * width, -1)
        return output(parts[1] * filtered)
    }
}

private final class LFM3VisionMLP: Module {
    let w1: Linear
    let w2: Linear
    let w3: Linear

    init(_ c: LFM3VisionConfiguration) {
        w1 = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        w2 = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        w3 = Linear(c.hiddenSize, c.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private final class LFM3VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3VisionAttention?
    @ModuleInfo(key: "conv") var convolution: LFM3VisionConvolution?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3VisionMLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ c: LFM3VisionConfiguration, kind: String) {
        if kind == "full_attention" {
            _attention.wrappedValue = LFM3VisionAttention(c)
        } else {
            _convolution.wrappedValue = LFM3VisionConvolution(c)
        }
        _feedForward.wrappedValue = LFM3VisionMLP(c)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, height: Int, width: Int,
        cosine: MLXArray, sine: MLXArray
    ) -> MLXArray {
        let normalized = operatorNorm(x)
        let mixed: MLXArray
        if let attention {
            mixed = attention(normalized, cosine: cosine, sine: sine)
        } else {
            mixed = convolution!(normalized, height: height, width: width)
        }
        let residual = x + mixed
        return residual + feedForward(ffnNorm(residual))
    }
}

final class LFM3VisionModel: Module {
    let patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    private let layers: [LFM3VisionLayer]
    let configuration: LFM3VisionConfiguration

    init(_ c: LFM3VisionConfiguration) {
        configuration = c
        patchifier = Linear(
            c.patchDimension,
            c.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        layers = c.layerTypes.map { LFM3VisionLayer(c, kind: $0) }
    }

    func callAsFunction(_ pixels: MLXArray, frames: [THW]) throws -> MLXArray {
        let c = configuration
        guard !frames.isEmpty, pixels.ndim == 2, pixels.dim(1) == c.patchDimension else {
            throw VLMError.imageProcessingFailure("Invalid LFM3 patch grid or pixel shape")
        }
        var remaining = pixels.dim(0)
        for frame in frames {
            guard frame.t > 0, frame.h > 0, frame.w > 0,
                frame.h <= Int(Int32.max), frame.w <= Int(Int32.max),
                frame.h % c.mergeSize == 0, frame.w % c.mergeSize == 0,
                frame.h <= remaining / frame.t / frame.w
            else {
                throw VLMError.imageProcessingFailure("Invalid LFM3 patch grid or pixel shape")
            }
            remaining -= frame.product
        }
        guard remaining == 0 else {
            throw VLMError.imageProcessingFailure("LFM3 patch count does not match the grid")
        }
        let embedded = embeddingNorm(patchifier(pixels.asType(embeddingNorm.weight.dtype)))
        let headDim = c.hiddenSize / c.heads
        let frequencies = pow(
            MLXArray(c.rope.theta),
            -MLXArray(stride(from: 0, to: headDim / 2, by: 2)).asType(.float32) / Float(headDim / 2)
        )
        var offset = 0
        var results = [MLXArray]()
        for frame in frames {
            var rows = [Int32]()
            var columns = [Int32]()
            for blockY in stride(from: 0, to: frame.h, by: c.mergeSize) {
                for blockX in stride(from: 0, to: frame.w, by: c.mergeSize) {
                    for y in 0 ..< c.mergeSize {
                        for x in 0 ..< c.mergeSize {
                            rows.append(Int32(blockY + y))
                            columns.append(Int32(blockX + x))
                        }
                    }
                }
            }
            let angles = concatenated(
                [
                    MLXArray(rows).asType(.float32).expandedDimensions(axis: -1) * frequencies,
                    MLXArray(columns).asType(.float32).expandedDimensions(axis: -1) * frequencies,
                ], axis: -1)
            let doubled = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 1)
            let cosine = cos(doubled)
            let sine = sin(doubled)
            for _ in 0 ..< frame.t {
                let count = frame.h * frame.w
                var hidden = embedded[offset ..< (offset + count)]
                for layer in layers {
                    hidden = layer(
                        hidden, height: frame.h, width: frame.w, cosine: cosine, sine: sine)
                }
                results.append(hidden)
                offset += count
            }
        }
        return concatenated(results)
    }
}
