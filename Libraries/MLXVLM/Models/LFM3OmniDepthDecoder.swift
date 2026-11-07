import Foundation
import MLX
import MLXLMCommon
import MLXNN

private final class LFM3OmniCodebookNorm: Module {
    @ParameterInfo var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, epsilon: Float) {
        self.epsilon = epsilon
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float32)
        return
            (x * rsqrt((x * x).mean(axis: -1, keepDims: true) + epsilon)
            * weight.asType(.float32)).asType(input.dtype)
    }
}

private final class LFM3OmniCodebook: Module {
    let embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var norm: LFM3OmniCodebookNorm

    init(_ config: LFM3OmniConfiguration) {
        embedding = Embedding(
            embeddingCount: config.codebookSize, dimensions: config.depth.hiddenSize)
        _norm.wrappedValue = LFM3OmniCodebookNorm(
            dimensions: config.depth.hiddenSize, epsilon: config.depth.normEps)
    }

    func logits(_ x: MLXArray) -> MLXArray {
        embedding.asLinear(norm(x))
    }
}

final class LFM3OmniDepthDecoder: Module {
    @ModuleInfo(key: "depth_linear") var projection: Linear
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray
    let model: LFM3ModelInner
    @ModuleInfo(key: "depth_embeddings") private var codebookEmbeddings: [LFM3OmniCodebook]
    let config: LFM3OmniConfiguration

    init(_ config: LFM3OmniConfiguration) {
        self.config = config
        _projection.wrappedValue = Linear(
            config.visionLanguage.text.hiddenSize,
            config.codebooks * config.depth.hiddenSize)
        _maskEmbedding.wrappedValue = MLXArray.zeros([config.depth.hiddenSize])
        model = LFM3ModelInner(config.depth, includeTokenEmbedding: false)
        _codebookEmbeddings.wrappedValue = (0 ..< config.codebooks).map { _ in
            LFM3OmniCodebook(config)
        }
    }

    func callAsFunction(hidden: MLXArray, codes: MLXArray, masked: MLXArray) throws -> MLXArray {
        guard hidden.ndim == 2, hidden.dim(0) > 0,
            hidden.dim(1) == config.visionLanguage.text.hiddenSize,
            codes.shape == [hidden.dim(0), config.codebooks], masked.shape == codes.shape,
            codes.dtype == .int32 || codes.dtype == .int64,
            masked.dtype == .bool,
            all((codes .>= 0) .&& (codes .< config.codebookSize)).item(Bool.self)
        else { throw VLMError.processing("Invalid LFM3-Omni depth decoder inputs") }
        let embedded = stacked(
            codebookEmbeddings.enumerated().map { index, codebook in
                codebook.embedding(codes[0..., index])
            }, axis: 1)
        let input =
            projection(hidden).reshaped(hidden.dim(0), config.codebooks, config.depth.hiddenSize)
            + MLX.where(
                masked.expandedDimensions(axis: -1), maskEmbedding.asType(embedded.dtype), embedded)
        let output = model(
            embeddings: input,
            masks: .init(fullAttention: MLXFast.ScaledDotProductAttentionMaskMode.none))
        return stacked(
            codebookEmbeddings.enumerated().map { index, codebook in
                codebook.logits(output[0..., index, 0...])
            }, axis: 1)
    }
}
