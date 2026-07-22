// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Mamba2Configuration: Codable, Sendable {
    var modelType: String = "mamba2"
    var numHeads: Int
    var headDim: Int
    var vocabSize: Int
    var hiddenSize: Int
    var stateSize: Int
    var numHiddenLayers: Int
    var layerNormEpsilon: Float
    var convKernel: Int
    var nGroups: Int
    var useBias: Bool
    var useConvBias: Bool
    var tieWordEmbeddings: Bool
    var timeStepLimit: [Float] = [0.0, Float.infinity]
    var ssmStateSize: Int

    /// `time_step_limit` as a (min, max) pair (defaults to (0, +inf)).
    var timeStepLimitPair: (Float, Float) {
        (timeStepLimit.first ?? 0.0, timeStepLimit.count > 1 ? timeStepLimit[1] : .infinity)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case stateSize = "state_size"
        case numHiddenLayers = "num_hidden_layers"
        case layerNormEpsilon = "layer_norm_epsilon"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case timeStepLimit = "time_step_limit"
        case ssmStateSize = "ssm_state_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mamba2"
        self.numHeads = try c.decode(Int.self, forKey: .numHeads)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.stateSize = try c.decode(Int.self, forKey: .stateSize)
        self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        self.layerNormEpsilon =
            try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5
        self.convKernel = try c.decode(Int.self, forKey: .convKernel)
        self.nGroups = try c.decodeIfPresent(Int.self, forKey: .nGroups) ?? 1
        self.useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        self.useConvBias = try c.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        // ssm_state_size defaults to state_size (see mamba2.py __post_init__).
        self.ssmStateSize = try c.decodeIfPresent(Int.self, forKey: .ssmStateSize) ?? stateSize
        // time_step_limit is [min, max]; default to (0, +inf).
        self.timeStepLimit =
            try c.decodeIfPresent([Float].self, forKey: .timeStepLimit) ?? [0.0, Float.infinity]
    }
}

// Gated RMSNorm: `silu(gate) * x` then a (single-group) RMS norm with a learned weight.
private class Mamba2RMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var h = x
        if let gate {
            h = h * silu(gate)
        }
        return MLXFast.rmsNorm(h, weight: weight, eps: eps)
    }
}

private class Mamba2Mixer: Module {
    let numHeads: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: Mamba2RMSNormGated

    init(_ args: Mamba2Configuration) {
        self.numHeads = args.numHeads
        self.ssmStateSize = args.ssmStateSize
        self.convKernelSize = args.convKernel
        self.intermediateSize = args.numHeads * args.headDim
        self.numGroups = args.nGroups
        self.headDim = args.headDim
        self.timeStepLimit = args.timeStepLimitPair
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(args.hiddenSize, projectionSize, bias: args.useBias)

        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        self._aLog.wrappedValue = MLX.log(MLXArray(1 ... numHeads).asType(.float32))
        self._D.wrappedValue = MLXArray.ones([numHeads])

        self._norm.wrappedValue = Mamba2RMSNormGated(
            dimensions: intermediateSize, eps: args.layerNormEpsilon)
        self._outProj.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: args.useBias)

        super.init()
    }

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input
        if let mask {
            convInput = MLX.where(
                expandedDimensions(mask, axis: -1), convInput, MLXArray.zeros(like: convInput))
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]
        if convState == nil {
            let keep = max(0, convKernelSize - 1)
            convState = MLXArray.zeros([batch, keep, convDim], dtype: dtype)
        }

        let padded = concatenated([convState!, convInput], axis: 1)
        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = contiguous(padded[0..., start ..< end, 0...])
        }

        return silu(conv1d(padded))
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1)

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])
        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: cache?[1],
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
            cache.advance(hiddenStates.dim(1))
        }

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
    }
}

private class Mamba2ResidualBlock: Module {
    @ModuleInfo(key: "mixer") var mixer: Mamba2Mixer
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: Mamba2Configuration) {
        self._mixer.wrappedValue = Mamba2Mixer(args)
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        x + mixer(norm(x), mask: mask, cache: cache)
    }
}

public class Mamba2Backbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    fileprivate let layers: [Mamba2ResidualBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    init(_ args: Mamba2Configuration) {
        precondition(args.vocabSize > 0)
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map { _ in Mamba2ResidualBlock(args) }
        self._normF.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings(inputs)
        let mask = createSSMMask(h: h, cache: cache?.first as? MambaCache)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i] as? MambaCache)
        }
        return normF(h)
    }
}

public class Mamba2Model: Module, LLMModel {
    let config: Mamba2Configuration
    @ModuleInfo(key: "backbone") var backbone: Mamba2Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Mamba2Configuration) {
        self.config = config
        self._backbone.wrappedValue = Mamba2Backbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = backbone(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return backbone.embeddings.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in MambaCache() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                sanitized[key] = value.swappedAxes(1, 2)
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    public var loraLayers: [Module] {
        backbone.layers
    }
}
