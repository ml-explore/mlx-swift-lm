// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct SparkConfiguration: Decodable, Sendable {
    struct RopeParameters: Decodable, Sendable {
        let ropeTheta: Float
        let partialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    let hiddenSize: Int
    let intermediateSize: Int
    let layerCount: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let vocabularySize: Int
    let normEpsilon: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let ropeParameters: [String: RopeParameters]
    let attentionBias: Bool
    let mlpBias: Bool
    let outputGate: Bool
    let gateActivation: String
    let tiedEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerCount = "num_hidden_layers"
        case headCount = "num_attention_heads"
        case kvHeadCount = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case normEpsilon = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case outputGate = "headwise_attn_output_gate"
        case gateActivation = "gate_attn_act_mode"
        case tiedEmbeddings = "tie_word_embeddings"
        case hiddenActivation = "hidden_act"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        guard layerCount > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerCount, in: c, debugDescription: "Layer count must be positive")
        }
        headCount = try c.decode(Int.self, forKey: .headCount)
        kvHeadCount = try c.decodeIfPresent(Int.self, forKey: .kvHeadCount) ?? headCount
        guard headCount > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .headCount, in: c,
                debugDescription: "Attention head count must be positive")
        }
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / headCount
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        normEpsilon = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 0
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: layerCount)
        ropeParameters = try c.decode([String: RopeParameters].self, forKey: .ropeParameters)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        outputGate = try c.decodeIfPresent(Bool.self, forKey: .outputGate) ?? false
        gateActivation = try c.decodeIfPresent(String.self, forKey: .gateActivation) ?? "sigmoid"
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
        let activation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu"
        guard hiddenSize > 0, intermediateSize > 0, layerCount > 0, headCount > 0,
            kvHeadCount > 0, headCount % kvHeadCount == 0, headDim > 0,
            vocabularySize > 0, layerTypes.count == layerCount, activation == "gelu",
            ["sigmoid", "silu"].contains(gateActivation),
            layerTypes.allSatisfy({ type in
                guard ["sliding_attention", "full_attention"].contains(type),
                    let rope = ropeParameters[type]
                else { return false }
                guard rope.partialRotaryFactor.isFinite, rope.partialRotaryFactor > 0,
                    rope.partialRotaryFactor <= 1
                else { return false }
                let dimensions = Int(Float(headDim) * rope.partialRotaryFactor)
                return dimensions > 0 && dimensions <= headDim && dimensions % 2 == 0
                    && rope.ropeTheta.isFinite && rope.ropeTheta > 0
                    && (type != "sliding_attention" || slidingWindow > 0)
            })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath, debugDescription: "Invalid Spark configuration")
            )
        }
    }
}

private class SparkAttention: Module {
    let config: SparkConfiguration
    let rope: RoPE
    let scale: Float
    @ModuleInfo(key: "q_k_v_proj") var qkv: Linear
    @ModuleInfo(key: "g_proj") var gate: Linear?
    @ModuleInfo(key: "out_proj") var output: Linear

    init(_ config: SparkConfiguration, type: String) {
        self.config = config
        let parameters = config.ropeParameters[type]!
        rope = RoPE(
            dimensions: Int(Float(config.headDim) * parameters.partialRotaryFactor),
            traditional: false, base: parameters.ropeTheta)
        scale = pow(Float(config.headDim), -0.5)
        _qkv.wrappedValue = Linear(
            config.hiddenSize,
            (config.headCount + 2 * config.kvHeadCount) * config.headDim,
            bias: config.attentionBias)
        if config.outputGate {
            _gate.wrappedValue = Linear(
                config.hiddenSize, config.headCount, bias: config.attentionBias)
        }
        _output.wrappedValue = Linear(
            config.headCount * config.headDim, config.hiddenSize,
            bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let qSize = config.headCount * config.headDim
        let kvSize = config.kvHeadCount * config.headDim
        let projected = qkv(x)
        var q = projected[.ellipsis, 0 ..< qSize].reshaped(b, n, config.headCount, config.headDim)
            .transposed(0, 2, 1, 3)
        var k = projected[.ellipsis, qSize ..< (qSize + kvSize)]
            .reshaped(b, n, config.kvHeadCount, config.headDim).transposed(0, 2, 1, 3)
        let v = projected[.ellipsis, (qSize + kvSize)...]
            .reshaped(b, n, config.kvHeadCount, config.headDim).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, offset: cache?.ropeOffset)
        k = applyRotaryPosition(rope, to: k, offset: cache?.ropeOffset)
        var result = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache,
            scale: scale, mask: mask)
        if let gate {
            let scores = gate(x).asType(.float32).reshaped(b, n, config.headCount, 1)
                .transposed(0, 2, 1, 3)
            let activation = config.gateActivation == "silu" ? silu(scores) : sigmoid(scores)
            result = result * activation.asType(result.dtype)
        }
        return output(result.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

private class SparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ config: SparkConfiguration) {
        _gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(gelu(gate(x)) * up(x)) }
}

private class SparkBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SparkAttention
    @ModuleInfo var mlp: SparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ config: SparkConfiguration, type: String) {
        _attention.wrappedValue = SparkAttention(config, type: type)
        mlp = SparkMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEpsilon)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEpsilon)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

private class SparkInner: Module {
    @ModuleInfo var embedding: Embedding
    @ModuleInfo var layers: [SparkBlock]
    @ModuleInfo var norm: RMSNorm
    let config: SparkConfiguration
    init(_ config: SparkConfiguration) {
        self.config = config
        embedding = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = config.layerTypes.map { SparkBlock(config, type: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEpsilon)
    }
    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        if tokens.dim(1) > 1, let cache,
            config.layerTypes.indices.contains(where: { i in
                guard config.layerTypes[i] == "full_attention", let limit = cache[i].maxSize
                else { return false }
                return cache[i].offset + tokens.dim(1) > limit
            })
        {
            return concatenated(
                (0 ..< tokens.dim(1)).map { i in
                    callAsFunction(tokens[0..., i ..< i + 1], cache: cache)
                }, axis: 1)
        }
        var h = embedding(tokens)
        for (i, layer) in layers.enumerated() {
            let window: Int?
            if config.layerTypes[i] == "sliding_attention" {
                window = min(config.slidingWindow, cache?[i].maxSize ?? config.slidingWindow)
            } else {
                window = nil
            }
            let mask = createAttentionMask(h: h, cache: cache?[i], windowSize: window)
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

final class SparkModel: Module, LLMModel {
    @ModuleInfo private var model: SparkInner
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?
    let config: SparkConfiguration
    var vocabularySize: Int { config.vocabularySize }
    var toolCallFormat: ToolCallFormat? { .spark }
    var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
    }
    var loraLayers: [Module] { model.layers }
    init(_ config: SparkConfiguration) {
        self.config = config
        model = SparkInner(config)
        if !config.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead.map { $0(h) } ?? model.embedding.asLinear(h)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        filterLMHeadWeights(from: weights, tiedWordEmbeddings: config.tiedEmbeddings)
    }
    func newCache(parameters: GenerateParameters? = nil) throws -> [KVCache] {
        try config.layerTypes.map { type in
            if type == "sliding_attention" {
                return try makeSlidingWindowKVCache(
                    parameters: parameters, window: config.slidingWindow)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State? = nil,
        prefill: PrefillParameters = .init()
    ) throws -> PrepareResult {
        let text = input.text
        let processed = try withPreparedCache(cache, lengths: text.sequenceLengths) {
            let processed = try prefill.forEachChunk(
                total: text.tokens.size, defaultStepSize: 128,
                maximumStepSize: cache.compactMap(\.maxSize).min()
            ) { range in
                _ = model(text[.newAxis, range].tokens, cache: cache.isEmpty ? nil : cache)
                asyncEval(cache)
            }
            if processed > 0 {
                eval(cache)
            }
            return processed
        }
        return .tokens(text[processed...])
    }
}
