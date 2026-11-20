// Copyright © 2026 Apple Inc.
// Port of mlx-vlm's d1-omni model (Blaizzy/mlx-vlm#2467), which follows LiquidAI's reference code.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Configuration

/// The d1-omni `config.json` (`model_type: "d1_omni"`). Missing keys take the Python defaults;
/// `vision_config` must give its model type and sizes.
struct D1OmniConfiguration: Codable, Sendable {

    /// The bidirectional LFM2 trunk.
    struct TextConfiguration: Codable, Sendable {
        var vocabularySize = 65536
        var hiddenSize = 1024
        var intermediateSize = 6656
        var hiddenLayers = 16
        var attentionHeads = 16
        var kvHeads = 8
        /// `"conv"` or `"full_attention"` per layer; all `"conv"` when the config has none.
        var layerTypes = [String](repeating: "conv", count: 16)
        var normEps: Float = 1e-5
        var convLCache = 3
        var blockFFNDimMultiplier: Double? = 1.0
        var blockMultipleOf = 256
        var ropeTheta: Float = 1_000_000

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case convLCache = "conv_L_cache"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockMultipleOf = "block_multiple_of"
            case ropeTheta = "rope_theta"
        }

        init() {}

        init(from decoder: any Decoder) throws {
            self.init()
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
            layerTypes =
                try c.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? Array(repeating: "conv", count: hiddenLayers)
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? normEps
            convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? convLCache
            if c.contains(.blockFFNDimMultiplier) {
                blockFFNDimMultiplier = try c.decodeIfPresent(
                    Double.self, forKey: .blockFFNDimMultiplier)
            }
            blockMultipleOf =
                try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? blockMultipleOf
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        }

        var headDimensions: Int { hiddenSize / attentionHeads }

        /// LFM2's MLP width with `block_auto_adjust_ff_dim` forced on, as d1-omni builds it.
        var feedForwardDimensions: Int {
            var width = Int(2 * Double(intermediateSize) / 3)
            if let blockFFNDimMultiplier {
                width = Int(blockFFNDimMultiplier * Double(width))
            }
            return blockMultipleOf * ((width + blockMultipleOf - 1) / blockMultipleOf)
        }
    }

    var textConfiguration = TextConfiguration()
    /// The SigLIP2 NaFlex image tower, as mlx-vlm reads it with LFM2-VL's `VisionConfig`.
    var visionConfiguration: LFM2VLConfiguration.VisionConfiguration
    var projectorHiddenSize = 2048
    var headLayers = 2
    /// Positions of a text-only row, and of a media prefix plus its text.
    var maxLength = 16384
    /// Text positions after images.
    var imageTextLength = 896
    /// Calibration temperatures keyed `"type:bucket"` or `"type"`.
    var temperatures: [String: Double] = [:]
    var bosTokenId = 1

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case projectorHiddenSize = "projector_hidden_size"
        case headLayers = "head_layers"
        case maxLength = "max_length"
        case imageTextLength = "image_text_length"
        case temperatures
        case bosTokenId = "bos_token_id"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        visionConfiguration = try c.decode(
            LFM2VLConfiguration.VisionConfiguration.self, forKey: .visionConfiguration)
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? textConfiguration
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? projectorHiddenSize
        headLayers = try c.decodeIfPresent(Int.self, forKey: .headLayers) ?? headLayers
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? maxLength
        imageTextLength =
            try c.decodeIfPresent(Int.self, forKey: .imageTextLength) ?? imageTextLength
        temperatures =
            try c.decodeIfPresent([String: Double].self, forKey: .temperatures) ?? temperatures
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? bosTokenId
    }
}

// MARK: - Trunk

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halves = split(x, parts: 2, axis: -1)
    return concatenated([-halves[1], halves[0]], axis: -1)
}

final class D1OmniAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    let headDimensions: Int

    init(_ config: D1OmniConfiguration.TextConfiguration) {
        let dimensions = config.hiddenSize
        headDimensions = config.headDimensions
        _qProj.wrappedValue = Linear(
            dimensions, config.attentionHeads * headDimensions, bias: false)
        _kProj.wrappedValue = Linear(dimensions, config.kvHeads * headDimensions, bias: false)
        _vProj.wrappedValue = Linear(dimensions, config.kvHeads * headDimensions, bias: false)
        _outProj.wrappedValue = Linear(
            config.attentionHeads * headDimensions, dimensions, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: config.normEps)
    }

    /// Rotated keys and values `[B, kvHeads, L, headDimensions]`.
    func keyValues(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> (
        keys: MLXArray, values: MLXArray
    ) {
        let (B, L) = (x.dim(0), x.dim(1))
        let k = kNorm(kProj(x).reshaped(B, L, -1, headDimensions)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, -1, headDimensions).transposed(0, 2, 1, 3)
        return (k * cos + rotateHalf(k) * sin, v)
    }

    /// `keys` and `values` may hold a media prefix in front of the positions of `x`.
    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, keys: MLXArray, values: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, -1, headDimensions)).transposed(0, 2, 1, 3)
        q = q * cos + rotateHalf(q) * sin
        let y = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: pow(Float(headDimensions), -0.5),
            mask: .array(mask))
        return outProj(y.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// `out_proj(C * conv(B * x))` with a centred 3-tap depthwise convolution. The conv module
/// only holds the taps.
final class D1OmniShortConv: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: D1OmniConfiguration.TextConfiguration) {
        let dimensions = config.hiddenSize
        _conv.wrappedValue = Conv1d(
            inputChannels: dimensions, outputChannels: dimensions,
            kernelSize: config.convLCache, groups: dimensions, bias: false)
        _inProj.wrappedValue = Linear(dimensions, 3 * dimensions, bias: false)
        _outProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    /// What the taps read at each position of `x`.
    func tapInput(_ x: MLXArray) -> MLXArray {
        let bcu = split(inProj(x), parts: 3, axis: -1)
        return bcu[0] * bcu[2]
    }

    /// `pad` is 1 on real positions; `keepRight` is 0 where the right tap must not reach
    /// (the last media position). `left` is the `tapInput` `[B, 1, D]` in front of `x` when
    /// `x` is the text after a media prefix; nil reads zero.
    func callAsFunction(
        _ x: MLXArray, pad: MLXArray, keepRight: MLXArray, left: MLXArray? = nil
    ) -> MLXArray {
        let L = x.dim(1)
        let bcu = split(inProj(x * expandedDimensions(pad, axis: -1)), parts: 3, axis: -1)
        let (b, c, u) = (bcu[0], bcu[1], bcu[2])
        let xp =
            left.map { concatenated([$0, b * u, zeros($0.shape, dtype: $0.dtype)], axis: 1) }
            ?? padded(b * u, widths: [0, [1, 1], 0])
        let w = conv.weight[0..., 0..., 0]
        var y = xp[0..., 0 ..< L] * w[0..., 0] + xp[0..., 1 ..< (L + 1)] * w[0..., 1]
        y = y + xp[0..., 2 ..< (L + 2)] * expandedDimensions(keepRight, axis: -1) * w[0..., 2]
        return outProj(c * y)
    }
}

final class D1OmniMLP: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _w1.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _w2.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _w3.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class D1OmniTrunkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1OmniAttention?
    @ModuleInfo var conv: D1OmniShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1OmniMLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ config: D1OmniConfiguration.TextConfiguration, kind: String) {
        if kind == "full_attention" {
            _attention.wrappedValue = D1OmniAttention(config)
        } else {
            _conv.wrappedValue = D1OmniShortConv(config)
        }
        _feedForward.wrappedValue = D1OmniMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.feedForwardDimensions)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }
}

/// What the text after a media prefix reads from it. A request with several questions runs
/// the prefix once and keeps this.
struct D1OmniPrefixCache {
    enum Layer {
        /// The prefix's rotated keys and values `[1, kvHeads, P, headDimensions]`.
        case attention(keys: MLXArray, values: MLXArray)
        /// The convolution's `tapInput` `[1, 1, D]` at the last prefix position, which the
        /// first text position's left tap reads.
        case conv(MLXArray)
    }

    /// The prefix length `P`.
    var length: Int
    /// One entry per trunk layer.
    var layers: [Layer]

    var arrays: [MLXArray] {
        layers.flatMap { layer -> [MLXArray] in
            switch layer {
            case .attention(let keys, let values): [keys, values]
            case .conv(let last): [last]
            }
        }
    }
}

/// Bidirectional LFM2 trunk over a media prefix followed by text.
///
/// Prefix queries never see text keys, and the centred convolution never lets the last
/// prefix position read the first text one, so the prefix does not depend on the text.
final class D1OmniTrunk: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [D1OmniTrunkLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    let ropeTheta: Float
    let headDimensions: Int

    /// Text passes after a shared prefix with this many positions or more wait for each layer,
    /// so only one layer's graph is alive, which lowers peak memory. Smaller passes pipeline.
    static let waitEachLayerPositions = 8192

    init(_ config: D1OmniConfiguration.TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = config.layerTypes.map { D1OmniTrunkLayer(config, kind: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        ropeTheta = config.ropeTheta
        headDimensions = config.headDimensions
    }

    /// RoPE tables computed in float32, then cast. MLX fast RoPE drifts at long lengths.
    func rope(length: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let exponent =
            MLXArray(Array(stride(from: Int32(0), to: Int32(headDimensions), by: 2)))
            .asType(.float32) / Float(headDimensions)
        let inverseFrequencies = MLXArray(Float(1)) / pow(MLXArray(ropeTheta), exponent)
        let frequencies =
            expandedDimensions(MLXArray(Int32(0) ..< Int32(length)).asType(.float32), axis: 1)
            * expandedDimensions(inverseFrequencies, axis: 0)
        let embedding = concatenated([frequencies, frequencies], axis: -1)
        return (MLX.cos(embedding).asType(dtype), MLX.sin(embedding).asType(dtype))
    }

    /// - Parameters:
    ///   - h: embeddings `[B, L, D]`, media prefix then text, right padded
    ///   - pad: `[B, L]`, true on real positions
    ///   - prefix: media length per row, 0 for text
    func callAsFunction(_ h: MLXArray, pad: MLXArray, prefix: [Int]) -> MLXArray {
        run(h, pad: pad, prefix: prefix, keep: false).hidden
    }

    /// Runs one media prefix `[P, D]` alone and keeps what the text after it reads.
    func prefixCache(_ prefix: MLXArray) -> D1OmniPrefixCache {
        let P = prefix.dim(0)
        let layers = run(
            expandedDimensions(prefix, axis: 0), pad: ones([1, P], dtype: .bool),
            prefix: [P], keep: true
        ).layers
        return D1OmniPrefixCache(length: P, layers: layers)
    }

    /// `callAsFunction`, and with `keep` what `prefixCache` keeps per layer, read at the last
    /// position of a single row.
    private func run(_ h: MLXArray, pad: MLXArray, prefix: [Int], keep: Bool) -> (
        hidden: MLXArray, layers: [D1OmniPrefixCache.Layer]
    ) {
        var h = h
        let L = h.dim(1)
        let t = expandedDimensions(MLXArray(Int32(0) ..< Int32(L)), axis: 0)
        let p = expandedDimensions(MLXArray(prefix.map { Int32($0) }), axis: 1)
        var mask = expandedDimensions(pad, axes: [1, 2])
        if prefix.contains(where: { $0 > 0 }) {
            let blocked =
                expandedDimensions(t .< p, axis: 2) .&& expandedDimensions(t .>= p, axis: 1)
            mask = mask .&& expandedDimensions(logicalNot(blocked), axis: 1)
        }
        let keepRight = (t .!= (p - 1)).asType(h.dtype)
        let padf = pad.asType(h.dtype)
        let (cos, sin) = rope(length: L, dtype: h.dtype)
        var kept = [D1OmniPrefixCache.Layer]()
        for layer in layers {
            var x = layer.operatorNorm(h)
            if let attention = layer.attention {
                let (keys, values) = attention.keyValues(x, cos: cos, sin: sin)
                if keep { kept.append(.attention(keys: keys, values: values)) }
                x = attention(x, cos: cos, sin: sin, keys: keys, values: values, mask: mask)
            } else if let conv = layer.conv {
                if keep { kept.append(.conv(conv.tapInput(x[0..., (L - 1)...]))) }
                x = conv(x, pad: padf, keepRight: keepRight)
            }
            h = h + x
            h = h + layer.feedForward(layer.ffnNorm(h))
        }
        return (embeddingNorm(h), kept)
    }

    /// The trunk over text rows `[B, T, D]` that all follow the media prefix of `cache`. It
    /// gives the text positions of `callAsFunction(_:pad:prefix:)` over the prefix and text.
    func callAsFunction(_ h: MLXArray, pad: MLXArray, after cache: D1OmniPrefixCache) -> MLXArray {
        var h = h
        let (B, T, P) = (h.dim(0), h.dim(1), cache.length)
        // Text sits at positions P..<P + T and sees every prefix key.
        let (cos, sin) = rope(length: P + T, dtype: h.dtype)
        let (textCos, textSin) = (cos[P...], sin[P...])
        let mask = expandedDimensions(
            concatenated([ones([B, P], dtype: .bool), pad], axis: 1), axes: [1, 2])
        let padf = pad.asType(h.dtype)
        // No text position is the last media one.
        let keepRight = ones([1, T], dtype: h.dtype)
        let waitEachLayer = B * T >= Self.waitEachLayerPositions
        var previous: MLXArray?
        for (layer, prefix) in zip(layers, cache.layers) {
            var x = layer.operatorNorm(h)
            if let attention = layer.attention, case .attention(let keys, let values) = prefix {
                let text = attention.keyValues(x, cos: textCos, sin: textSin)
                let shape = [B] + keys.shape.dropFirst()
                x = attention(
                    x, cos: textCos, sin: textSin,
                    keys: concatenated([broadcast(keys, to: shape), text.keys], axis: 2),
                    values: concatenated([broadcast(values, to: shape), text.values], axis: 2),
                    mask: mask)
            } else if let conv = layer.conv, case .conv(let last) = prefix {
                x = conv(
                    x, pad: padf, keepRight: keepRight,
                    left: broadcast(last, to: [B, 1, last.dim(2)]))
            }
            h = h + x
            h = h + layer.feedForward(layer.ffnNorm(h))
            if waitEachLayer {
                eval(h)
            } else {
                asyncEval(h)
                if let previous { eval(previous) }
                previous = h
            }
        }
        return embeddingNorm(h)
    }
}

// MARK: - Decision head

/// torch `TransformerEncoderLayer(norm_first=True, activation=relu, dropout=0)`.
final class D1OmniDecisionLayer: Module {
    @ModuleInfo var attention: MultiHeadAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(width: Int) {
        _attention.wrappedValue = MultiHeadAttention(
            dimensions: width, numHeads: width / 64, bias: true)
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _linear1.wrappedValue = Linear(width, 4 * width)
        _linear2.wrappedValue = Linear(4 * width, width)
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray) -> MLXArray {
        let normalized = norm1(h)
        let h = h + attention(normalized, keys: normalized, values: normalized, mask: mask)
        return h + linear2(relu(linear1(norm2(h))))
    }
}

final class D1OmniDecisionLayers: Module {
    let layers: [D1OmniDecisionLayer]

    init(width: Int, count: Int) {
        layers = (0 ..< count).map { _ in D1OmniDecisionLayer(width: width) }
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray) -> MLXArray {
        layers.reduce(h) { $1($0, mask: mask) }
    }
}

/// Scores the hidden state at each option marker.
final class D1OmniDecisionHead: Module {
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo var head: D1OmniDecisionLayers
    @ModuleInfo var scorer: Sequential

    init(width: Int, layers: Int) {
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        _head.wrappedValue = D1OmniDecisionLayers(width: width, count: layers)
        _scorer.wrappedValue = Sequential(
            layers: LayerNorm(dimensions: width), Linear(width, width), GELU(), Linear(width, 1))
    }

    /// Float32 logits `[B, K]`, -1e4 where `markerMask` is false.
    func callAsFunction(
        _ h: MLXArray, pad: MLXArray, markerPositions: MLXArray, markerMask: MLXArray,
        questionTypes: MLXArray
    ) -> MLXArray {
        var h = h + expandedDimensions(typeEmbedding(questionTypes), axis: 1)
        h = head(h, mask: expandedDimensions(pad, axes: [1, 2]))
        let markers = takeAlong(h, expandedDimensions(markerPositions, axis: -1), axis: 1)
        let logits = scorer(markers).squeezed(axis: -1).asType(.float32)
        return which(markerMask, logits, MLXArray(Float(-1e4)))
    }
}

// MARK: - Prompt

/// d1-omni prompts, as in `d1_omni/prompt.py`:
///
///     <bos> <state> state <q> instructions (<opt> <mask> option </opt>)* <decide>
///
/// The head scores the hidden state at every `<mask>`.
enum D1OmniPrompt {
    /// Descriptions of a bool question's answers.
    typealias BoolCriteria = (whenTrue: String?, whenFalse: String?)

    /// How image bool questions were trained.
    static let yesNo: BoolCriteria = (whenTrue: "yes", whenFalse: "no")

    /// The delimiter and marker ids.
    struct Tokens: Equatable, Sendable {
        var bos: Int
        var state: Int
        var question: Int
        var option: Int
        var optionEnd: Int
        var decide: Int
        var marker: Int
    }

    /// Python `QTYPES`: choice 0, score 1, noul (bool) 2.
    static func questionType(_ question: DecisionQuestion) -> Int {
        switch question.criteria {
        case .choice: 0
        case .score: 1
        case .bool: 2
        }
    }

    static func typeName(_ question: DecisionQuestion) -> String {
        switch question.criteria {
        case .choice: "choice"
        case .score: "score"
        case .bool: "noul"
        }
    }

    static func optionCount(_ question: DecisionQuestion) -> Int {
        switch question.criteria {
        case .choice(let options): options.count
        case .score(let levels): levels.count
        case .bool: 2
        }
    }

    /// `type:bucket` with buckets 2, 3-5, 6-10 and 11+.
    static func temperatureKey(_ question: DecisionQuestion) -> String {
        let count = optionCount(question)
        let bucket =
            count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
        return "\(typeName(question)):\(bucket)"
    }

    /// Rewrite `<|name|>` as `<¦name¦>` so caller text cannot forge a delimiter.
    static func escape(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        func isWord(_ scalar: Unicode.Scalar) -> Bool {
            scalar.isASCII
                && (("A" ... "Z").contains(scalar) || ("a" ... "z").contains(scalar)
                    || ("0" ... "9").contains(scalar) || scalar == "_")
        }
        var out = String.UnicodeScalarView()
        var i = 0
        while i < scalars.count {
            if scalars[i] == "<", i + 1 < scalars.count, scalars[i + 1] == "|" {
                var j = i + 2
                while j < scalars.count, isWord(scalars[j]) { j += 1 }
                if j > i + 2, j + 1 < scalars.count, scalars[j] == "|", scalars[j + 1] == ">" {
                    out.append("<")
                    out.append("¦")
                    out.append(contentsOf: scalars[(i + 2) ..< j])
                    out.append("¦")
                    out.append(">")
                    i = j + 2
                    continue
                }
            }
            out.append(scalars[i])
            i += 1
        }
        return String(out)
    }

    /// The state as the model reads it. A missing state is empty.
    static func stateText(_ state: DecisionState?) -> String {
        state?.serialized(indent: nil) ?? ""
    }

    /// Option texts in the model's order; bool reads `[false, true]`.
    static func renderOptions(
        _ question: DecisionQuestion, noulDefault: BoolCriteria? = nil
    ) -> [String] {
        switch question.criteria {
        case .choice(let options):
            return options.map { option in
                let description = option.description.flatMap { $0.isEmpty ? nil : $0 }
                return description.map { "\(option.label): \($0)" } ?? option.label
            }
        case .score(let levels):
            return levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .bool(let whenTrue, let whenFalse):
            // Python `criteria or noul_default or {}`: one given side disables the default.
            let criteria: BoolCriteria =
                whenTrue == nil && whenFalse == nil
                ? (noulDefault ?? (nil, nil)) : (whenTrue, whenFalse)
            let falseText = criteria.whenFalse.flatMap { $0.isEmpty ? nil : $0 }
            let trueText = criteria.whenTrue.flatMap { $0.isEmpty ? nil : $0 }
            return [
                "false: " + (falseText ?? "no, the statement does not hold"),
                "true: " + (trueText ?? "yes, the statement holds"),
            ]
        }
    }

    /// Token ids of caller text, escaped.
    static func encoded(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        tokenizer.encode(text: escape(text), addSpecialTokens: false)
    }

    /// The first `limit` ids of ``encoded(_:tokenizer:)``. A long text is encoded piece by
    /// piece until there are enough, so the cost follows what the model reads. Text with no
    /// cut point (see `startsToken`), such as one long word, is encoded whole.
    static func encoded(_ text: String, limit: Int, tokenizer: any Tokenizer) -> [Int] {
        let scalars = Array(escape(text).unicodeScalars)
        var ids = [Int]()
        var start = 0
        // Characters per token: a guess, then what the text has shown.
        var perToken = 4
        while ids.count < limit, start < scalars.count {
            var end = min(start + perToken * (limit - ids.count) + 64, scalars.count)
            while end < scalars.count, !startsToken(scalars, at: end) {
                end += 1
            }
            let piece = String(String.UnicodeScalarView(scalars[start ..< end]))
            ids += tokenizer.encode(text: piece, addSpecialTokens: false)
            start = end
            perToken = max(perToken, start / max(ids.count, 1) + 1)
        }
        return Array(ids.prefix(limit))
    }

    /// Whether d1-omni's pre-tokenizer always starts a token at `index`, with no token before
    /// it that reads past it: before a space that follows a non-space, or before a non-space
    /// that follows a line break. No added token holds whitespace, so pieces cut there encode
    /// as the whole text does.
    static func startsToken(_ scalars: [Unicode.Scalar], at index: Int) -> Bool {
        let (previous, next) = (scalars[index - 1], scalars[index])
        if previous == "\n" || previous == "\r" {
            return !next.properties.isWhitespace
        }
        return next == " " && !previous.properties.isWhitespace
    }

    /// Token ids of one question over one state, and each option marker's position.
    ///
    /// `stateIds` comes from ``encoded(_:limit:tokenizer:)``, so a request encodes its state
    /// once.
    /// The option block gets `max(96, min(24n + 32, maxLength / 2))` tokens shared evenly;
    /// the state is cut on the right to the room left.
    static func encode(
        tokenizer: any Tokenizer, tokens: Tokens, stateIds: [Int], question: DecisionQuestion,
        maxLength: Int, noulDefault: BoolCriteria? = nil, name: String = ""
    ) throws -> (ids: [Int], markers: [Int]) {
        func encoded(_ text: String) -> [Int] { Self.encoded(text, tokenizer: tokenizer) }
        let options = renderOptions(question, noulDefault: noulDefault)
        let count = options.count
        let budget = max(96, min(count * 24 + 32, maxLength / 2))
        let perOptionTokens = max(2, (budget - 3 * count) / count)
        var ids = Array(
            ([tokens.question] + encoded(question.instructions)).prefix(max(16, budget)))
        var markers = [Int]()
        for text in options {
            markers.append(ids.count + 1)
            ids += [tokens.option, tokens.marker]
            ids += encoded(" " + text).prefix(perOptionTokens)
            ids.append(tokens.optionEnd)
        }
        ids.append(tokens.decide)
        let room = max(0, maxLength - ids.count - 2)
        let state = [tokens.state] + stateIds.prefix(room)
        ids = Array(([tokens.bos] + state + ids).prefix(max(0, maxLength)))
        markers = markers.map { $0 + 1 + state.count }
        if let last = markers.last, last >= maxLength {
            throw DecisionError.optionsDoNotFit(name: name)
        }
        return (ids, markers)
    }
}

extension D1OmniPrompt.Tokens {
    /// Resolve the delimiters by name. BOS falls back to `fallbackBOS`.
    init(_ tokenizer: any Tokenizer, fallbackBOS: Int) throws {
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw DecisionError.missingToken("The tokenizer has no \(token) token")
            }
            return id
        }
        self.init(
            bos: tokenizer.bosToken.flatMap { tokenizer.convertTokenToId($0) } ?? fallbackBOS,
            state: try id("<|reserved_7|>"),
            question: try id("<|reserved_8|>"),
            option: try id("<|reserved_9|>"),
            optionEnd: try id("<|reserved_10|>"),
            decide: try id("<|reserved_11|>"),
            marker: try id("<|mask|>"))
    }
}

// MARK: - Model

/// One question over one request: an optional media prefix `[P, D]`, then text ids.
struct D1OmniRow {
    var prefix: MLXArray?
    var ids: [Int]
    var markers: [Int]
    var question: DecisionQuestion
    var calibrate: Bool

    var prefixLength: Int { prefix?.dim(0) ?? 0 }
    var length: Int { prefixLength + ids.count }
}

/// LiquidAI d1-omni-600M: a bidirectional LFM2 trunk and a decision head that scores option
/// markers, with an optional image prefix.
final class D1Omni: Module, DecisionModel {
    var configuration: D1OmniConfiguration

    @ModuleInfo var encoder: D1OmniTrunk
    @ModuleInfo var head: D1OmniDecisionHead
    @ModuleInfo var vision: D1OmniVision

    init(_ configuration: D1OmniConfiguration) {
        self.configuration = configuration
        let width = configuration.textConfiguration.hiddenSize
        _encoder.wrappedValue = D1OmniTrunk(configuration.textConfiguration)
        _head.wrappedValue = D1OmniDecisionHead(width: width, layers: configuration.headLayers)
        _vision.wrappedValue = D1OmniVision(
            configuration.visionConfiguration,
            projectorHiddenSize: configuration.projectorHiddenSize, outputDimensions: width)
    }

    /// A model for a checkpoint's `config.json`, before its weights are loaded.
    convenience init(configurationData: Data) throws {
        self.init(
            try JSONDecoder.json5().decode(D1OmniConfiguration.self, from: configurationData))
    }

    var decisionCapabilities: DecisionCapabilities {
        DecisionCapabilities(
            questionKinds: [.choice, .score, .bool], media: [.images], maximumScoreLevels: 10)
    }

    /// The checkpoint's audio encoder. This model reads no audio, so it is not loaded.
    var excludedCheckpointNamespaces: [String] { ["audio"] }

    /// Option logits `[B, K]` for right-padded rows of media embeddings then text.
    func callAsFunction(
        _ embeddings: MLXArray, pad: MLXArray, prefix: [Int], lengths: [Int],
        markerPositions: MLXArray, markerMask: MLXArray, questionTypes: MLXArray
    ) -> MLXArray {
        let h = encoder(embeddings, pad: pad, prefix: prefix)
        let T = lengths.max() ?? 0
        let positions = expandedDimensions(MLXArray(Int32(0) ..< Int32(T)), axis: 0)
        let textPad = positions .< expandedDimensions(MLXArray(lengths.map { Int32($0) }), axis: 1)
        let index = minimum(
            expandedDimensions(MLXArray(prefix.map { Int32($0) }), axis: 1) + positions,
            MLXArray(Int32(h.dim(1) - 1)))
        var text = takeAlong(h, expandedDimensions(index, axis: -1), axis: 1)
        text = which(expandedDimensions(textPad, axis: -1), text, MLXArray(0, dtype: text.dtype))
        return head(
            text, pad: textPad, markerPositions: markerPositions, markerMask: markerMask,
            questionTypes: questionTypes)
    }

    // MARK: Requests

    func decisionProbabilities(
        _ request: consuming DecisionRequest, tokenizer: any Tokenizer,
        options: DecisionExecutionOptions
    ) async throws -> DecisionProbabilities {
        let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: configuration.bosTokenId)
        let (prefix, media) = try await mediaPrefix(images: request.images)
        let (rows, inputTokens) = try rows(
            state: request.state, questions: request.questions, prefix: prefix, media: media,
            tokenizer: tokenizer, tokens: tokens)
        let probabilities = try run(rows, tokenBudget: options.tokenBudget)
        return DecisionProbabilities(probabilities: probabilities, inputTokens: inputTokens)
    }

    /// The prefix `[P, D]` every question of a request reads: all images in order, or none for
    /// text.
    func mediaPrefix(
        images: [UserInput.Image]
    ) async throws -> (prefix: MLXArray?, media: DecisionMediaKind?) {
        if !images.isEmpty {
            let prefix = try decisionImageErrors {
                try vision(images) { try OrientedImage($0) }
            }
            return (prefix, .images)
        }
        return (nil, nil)
    }

    /// One row per question of a request, and the positions they hold. The prefix counts
    /// once per question, as mlx-vlm reports it, though `run` reads a shared prefix once.
    func rows(
        state: DecisionState?, questions: [DecisionRequest.NamedQuestion], prefix: MLXArray?,
        media: DecisionMediaKind?, tokenizer: any Tokenizer, tokens: D1OmniPrompt.Tokens
    ) throws -> (rows: [D1OmniRow], inputTokens: Int) {
        let prefixLength = prefix?.dim(0) ?? 0
        var maxLength = media == .images ? configuration.imageTextLength : configuration.maxLength
        maxLength = min(maxLength, configuration.maxLength - prefixLength)
        guard maxLength >= 64 else {
            throw DecisionError.mediaTooLong(
                positions: prefixLength, maximum: configuration.maxLength)
        }
        // No row reads more than `maxLength` state tokens.
        let stateIds = D1OmniPrompt.encoded(
            D1OmniPrompt.stateText(state), limit: maxLength,
            tokenizer: tokenizer)
        let rows = try questions.map { named in
            let (ids, markers) = try D1OmniPrompt.encode(
                tokenizer: tokenizer, tokens: tokens, stateIds: stateIds, question: named.question,
                maxLength: maxLength, noulDefault: media == nil ? nil : D1OmniPrompt.yesNo,
                name: named.name)
            return D1OmniRow(
                prefix: prefix, ids: ids, markers: markers, question: named.question,
                calibrate: media == nil)
        }
        return (rows, rows.reduce(0) { $0 + $1.length })
    }

    /// Probabilities per row, run in the batches that `D1Prompt.chunks` makes. Rows that all
    /// follow one media prefix run it once, and only their text is padded.
    func run(_ rows: [D1OmniRow], tokenBudget: Int) throws -> [[Double]] {
        try Task.checkCancellation()
        let cache = sharedPrefixCache(rows)
        var results = [[Double]](repeating: [], count: rows.count)
        let trunk = cache?.length ?? 0
        for batch in D1Prompt.chunks(
            lengths: rows.map { $0.length - trunk }, trunk: trunk, budget: tokenBudget)
        {
            try Task.checkCancellation()
            let probabilities = forward(batch.map { rows[$0] }, after: cache)
            for (index, probabilities) in zip(batch, probabilities) {
                results[index] = probabilities
            }
        }
        return results
    }

    /// The trunk's pass over the media prefix when several rows all follow the same one.
    /// One row runs with its prefix instead, which is a little faster.
    func sharedPrefixCache(_ rows: [D1OmniRow]) -> D1OmniPrefixCache? {
        guard rows.count > 1, let prefix = rows[0].prefix, prefix.dim(0) > 0,
            rows.allSatisfy({ $0.prefix === prefix })
        else { return nil }
        let cache = encoder.prefixCache(prefix.asType(encoder.embeddingNorm.weight.dtype))
        // Finish the prefix pass before the text passes take memory.
        eval(cache.arrays)
        return cache
    }

    /// Softmax over each row's options, calibrated for text rows.
    func forward(_ rows: [D1OmniRow], after cache: D1OmniPrefixCache? = nil) -> [[Double]] {
        let logits = self.logits(rows, after: cache)
        let temperatures = configuration.temperatures
        let probabilities = rows.enumerated().map { index, row in
            var z = logits[index, 0 ..< D1OmniPrompt.optionCount(row.question)]
            if row.calibrate {
                let temperature =
                    temperatures[D1OmniPrompt.temperatureKey(row.question)]
                    ?? temperatures[D1OmniPrompt.typeName(row.question)] ?? 1.0
                z = z / Float(temperature)
            }
            return softmax(z, axis: -1)
        }
        // One GPU sync for the batch, not one per row.
        eval(probabilities)
        return probabilities.map { $0.asArray(Float.self).map { Double($0) } }
    }

    /// Raw option logits `[B, K]` for a batch of rows. With `cache`, every row follows its
    /// media prefix and only the text runs.
    func logits(_ rows: [D1OmniRow], after cache: D1OmniPrefixCache? = nil) -> MLXArray {
        let dtype = encoder.embeddingNorm.weight.dtype
        let width = configuration.textConfiguration.hiddenSize
        let sequences = rows.map { row in
            let text = encoder.embedTokens(MLXArray(row.ids.map { Int32($0) })).asType(dtype)
            guard cache == nil, let prefix = row.prefix else { return text }
            return concatenated([prefix.asType(dtype), text])
        }
        let totals = sequences.map { $0.dim(0) }
        let L = totals.max() ?? 0
        let h = stacked(
            sequences.map {
                $0.dim(0) == L
                    ? $0 : concatenated([$0, zeros([L - $0.dim(0), width], dtype: dtype)])
            })
        let pad =
            expandedDimensions(MLXArray(Int32(0) ..< Int32(L)), axis: 0)
            .< expandedDimensions(MLXArray(totals.map { Int32($0) }), axis: 1)
        let K = rows.map(\.markers.count).max() ?? 0
        let markerPositions = MLXArray(
            rows.flatMap { row in
                row.markers.map { Int32($0) } + Array(repeating: 0, count: K - row.markers.count)
            }, [rows.count, K])
        let markerMask =
            expandedDimensions(MLXArray(Int32(0) ..< Int32(K)), axis: 0)
            .< expandedDimensions(MLXArray(rows.map { Int32($0.markers.count) }), axis: 1)
        let questionTypes = MLXArray(rows.map { Int32(D1OmniPrompt.questionType($0.question)) })
        guard let cache else {
            return self(
                h, pad: pad, prefix: rows.map(\.prefixLength), lengths: rows.map(\.ids.count),
                markerPositions: markerPositions, markerMask: markerMask,
                questionTypes: questionTypes)
        }
        let text = which(
            expandedDimensions(pad, axis: -1), encoder(h, pad: pad, after: cache),
            MLXArray(0, dtype: dtype))
        return head(
            text, pad: pad, markerPositions: markerPositions, markerMask: markerMask,
            questionTypes: questionTypes)
    }

    // MARK: Weights

    /// Python `Model.sanitize`: accepts the torch checkpoint and MLX names, and is idempotent.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasSuffix("num_batches_tracked") || key.hasSuffix("position_ids") {
                continue
            }
            var key = key.replacingOccurrences(
                of: "vision.tower.vision_model.", with: "vision.tower.")
            var value = value
            if key.hasSuffix(".conv.conv.weight"), value.ndim == 3, value.dim(-1) > value.dim(1) {
                value = value.transposed(0, 2, 1)
            }
            if key.hasPrefix("head.head."), let range = key.range(of: ".self_attn.in_proj_") {
                let prefix = key[..<range.lowerBound]
                let suffix = key[range.upperBound...]
                let parts = split(value, parts: 3, axis: 0)
                for (name, part) in zip(["query_proj", "key_proj", "value_proj"], parts) {
                    result["\(prefix).attention.\(name).\(suffix)"] = part
                }
                continue
            }
            if key.hasPrefix("head.head.") {
                key = key.replacingOccurrences(
                    of: ".self_attn.out_proj.", with: ".attention.out_proj.")
            }
            if key.hasPrefix("head.scorer."), !key.hasPrefix("head.scorer.layers.") {
                key = "head.scorer.layers." + key.dropFirst("head.scorer.".count)
            }
            result[key] = value
        }
        return result
    }
}
