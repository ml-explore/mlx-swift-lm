// Copyright © 2026 Apple Inc.
//
// Port of KRAFTON's Raon-1 mini (`raon1_mini`): https://huggingface.co/KRAFTON/Raon-1-mini
//
// 25 layers on a 4-stream mHC residual. 18 layers use Raon Delta Attention (RDA, a gated delta
// rule with one decay per key dimension), 7 use NoPE multi-head latent attention (MLA) with an
// output gate. Layer 0 has a dense SiTU-GLU MLP; the others a sigmoid-routed MoE with one shared
// expert.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Raon1MiniConfiguration: Codable, Sendable {
    public struct RDA: Codable, Sendable {
        var heads: Int?
        var headDim: Int?
        var convKernelSize: Int
        var fullRankGate: Bool
        /// `nil` uses `-softplus` for the log decay.
        var gateLowerBound: Float?
        var rdaLayers: [Int]
        var mlaLayers: [Int]

        enum CodingKeys: String, CodingKey {
            case heads = "num_heads"
            case headDim = "head_dim"
            case convKernelSize = "conv_kernel_size"
            case fullRankGate = "full_rank_gate"
            case gateLowerBound = "gate_lower_bound"
            case rdaLayers = "rda_layers"
            case mlaLayers = "mla_layers"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            heads = try c.decodeIfPresent(Int.self, forKey: .heads)
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 4
            fullRankGate = try c.decodeIfPresent(Bool.self, forKey: .fullRankGate) ?? true
            gateLowerBound =
                c.contains(.gateLowerBound)
                ? try c.decodeIfPresent(Float.self, forKey: .gateLowerBound) : -5
            rdaLayers = try c.decode([Int].self, forKey: .rdaLayers)
            mlaLayers = try c.decode([Int].self, forKey: .mlaLayers)
        }
    }

    /// Steering vectors added to the MoE output of some layers, scaled by the reasoning effort.
    public struct EffortSteering: Codable, Sendable {
        var alpha: [String: Float]
        var layers: [Int]
        var defaultEffort: String?

        enum CodingKeys: String, CodingKey {
            case alpha, layers
            case defaultEffort = "default"
        }

        /// The alpha of thinking off when the config lists none above 0, as released configs do.
        static let thinkingOffAlpha: Float = 0.1

        /// The alpha of `effort`: the default effort's for `nil` or a name the config does not
        /// list, and for ``ReasoningEffort/off`` the config's `none` alpha when above 0, else
        /// ``thinkingOffAlpha``.
        func alpha(for effort: ReasoningEffort?) -> Float {
            if effort == .off {
                return alpha[ReasoningEffort.off.name].flatMap { $0 > 0 ? $0 : nil }
                    ?? Self.thinkingOffAlpha
            }
            return effort.flatMap { alpha[$0.name] } ?? alpha[defaultEffort ?? "high"] ?? 0
        }
    }

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var mlaLatentNormEps: Float
    var mhcSelectorEps: Float
    var mhcStreams: Int
    var mhcPostFactor: Float
    var hiddenAct: String
    var situBeta: Float
    var situLinearBeta: Float
    var experts: Int
    var expertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedExperts: Int
    var firstKDenseReplace: Int
    var moeLayerFreq: Int
    var routerActivation: String
    var moeRenormalize: Bool
    var routedScalingFactor: Float
    var useGroupedTopK: Bool
    var qLoraRank: Int
    var kvLoraRank: Int
    var qkNopeHeadDim: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var mlaUseNope: Bool
    var mlaUseOutputGate: Bool
    var rda: RDA
    var rdaDecayRank: Int
    var tieWordEmbeddings: Bool
    var globalScaledLinears: [String]
    var effortSteering: EffortSteering?
    var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case mlaLatentNormEps = "mla_latent_norm_eps"
        case mhcSelectorEps = "raon_mhc_selector_eps"
        case mhcStreams = "raon_mhc_streams"
        case mhcPostFactor = "raon_mhc_post_factor"
        case hiddenAct = "hidden_act"
        case situBeta = "activation_situ_beta"
        case situLinearBeta = "activation_situ_linear_beta"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_token"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExperts = "num_shared_experts"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case routerActivation = "moe_router_activation_func"
        case moeRenormalize = "moe_renormalize"
        case routedScalingFactor = "routed_scaling_factor"
        case useGroupedTopK = "use_grouped_topk"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case mlaUseNope = "mla_use_nope"
        case mlaUseOutputGate = "mla_use_output_gate"
        case rda = "rda_config"
        case rdaDecayRank = "rda_decay_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case globalScaledLinears = "raon_global_scaled_linears"
        case effortSteering = "reasoning_effort_steering"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "raon1_mini"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 25
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 40
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 259_840
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12032
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        mlaLatentNormEps = try c.decodeIfPresent(Float.self, forKey: .mlaLatentNormEps) ?? 1e-6
        mhcSelectorEps = try c.decodeIfPresent(Float.self, forKey: .mhcSelectorEps) ?? 1e-5
        mhcStreams = try c.decodeIfPresent(Int.self, forKey: .mhcStreams) ?? 4
        mhcPostFactor = try c.decodeIfPresent(Float.self, forKey: .mhcPostFactor) ?? 2
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "situ"
        situBeta = try c.decodeIfPresent(Float.self, forKey: .situBeta) ?? 4
        situLinearBeta = try c.decodeIfPresent(Float.self, forKey: .situLinearBeta) ?? 25
        experts = try c.decodeIfPresent(Int.self, forKey: .experts) ?? 128
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 6
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1088
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        routerActivation =
            try c.decodeIfPresent(String.self, forKey: .routerActivation) ?? "sigmoid"
        moeRenormalize = try c.decodeIfPresent(Bool.self, forKey: .moeRenormalize) ?? true
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        useGroupedTopK = try c.decodeIfPresent(Bool.self, forKey: .useGroupedTopK) ?? false
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 576
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? true
        mlaUseOutputGate = try c.decodeIfPresent(Bool.self, forKey: .mlaUseOutputGate) ?? true
        rda = try c.decode(RDA.self, forKey: .rda)
        rdaDecayRank = try c.decodeIfPresent(Int.self, forKey: .rdaDecayRank) ?? 128
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        globalScaledLinears =
            try c.decodeIfPresent([String].self, forKey: .globalScaledLinears) ?? []
        effortSteering = try c.decodeIfPresent(EffortSteering.self, forKey: .effortSteering)
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    var rdaHeads: Int { rda.heads ?? attentionHeads }
    var rdaHeadDim: Int { rda.headDim ?? headDim }

    /// Whether layer `index` (from 0) is an RDA layer. The config counts layers from 1.
    func isRDALayer(_ index: Int) -> Bool { rda.rdaLayers.contains(index + 1) }

    func isMoELayer(_ index: Int) -> Bool {
        experts > 0 && index >= firstKDenseReplace && index % moeLayerFreq == 0
    }
}

extension Raon1MiniConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ message: String) throws {
            guard condition else { throw Raon1MiniError.unsupportedConfiguration(message) }
        }
        try require(hiddenLayers > 0, "num_hidden_layers must be positive")
        try require(moeLayerFreq > 0, "moe_layer_freq must be positive")
        try require(hiddenAct == "situ", "hidden_act must be situ")
        try require(routerActivation == "sigmoid", "moe_router_activation_func must be sigmoid")
        try require(!useGroupedTopK, "grouped top-k routing is not supported")
        try require(mlaUseNope, "only NoPE MLA is supported")
        try require(rda.fullRankGate, "only full-rank RDA gates are supported")
        try require(mhcStreams >= 2, "raon_mhc_streams must be at least 2")
        try require(
            (rda.rdaLayers + rda.mlaLayers).sorted() == Array(1 ... hiddenLayers),
            "rda_layers and mla_layers must partition the layers")
        try require(
            experts == 0 || (1 ... experts).contains(expertsPerToken),
            "num_experts_per_token must be between 1 and num_experts")
        if let effortSteering {
            try require(
                effortSteering.layers.allSatisfy { (0 ..< hiddenLayers).contains($0) },
                "reasoning_effort_steering layers must be layer indices")
        }
    }
}

// MARK: - Primitives

/// SiTU-GLU: `beta tanh(gate / beta) sigmoid(gate) * linearBeta tanh(x / linearBeta)`, in float32.
private func makeSiTU(beta: Float, linearBeta: Float) -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: true) { x, gate in
        let g = gate.asType(.float32)
        let v = x.asType(.float32)
        return (beta * tanh(g / beta) * sigmoid(g) * (linearBeta * tanh(v / linearBeta)))
            .asType(x.dtype)
    }
}

/// The compiled arithmetic every layer shares.
struct Raon1Arithmetic: Sendable {
    let situ: @Sendable (MLXArray, MLXArray) -> MLXArray
    let mhc: Raon1MHCFunctions
    let rda: Raon1RDAArithmetic
    let moe: Raon1MoEArithmetic

    init(_ config: Raon1MiniConfiguration) {
        situ = makeSiTU(beta: config.situBeta, linearBeta: config.situLinearBeta)
        mhc = Raon1MHCFunctions(config)
        rda = Raon1RDAArithmetic(config)
        moe = Raon1MoEArithmetic(config)
    }
}

private let scaleOutput: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    y, scale in
    (y.asType(.float32) * scale).asType(y.dtype)
}

/// An NVFP4 linear layer whose float32 per-tensor scale multiplies the output.
///
/// KRAFTON's export keeps the scale outside the E4M3 block scales, so it is not the MLX
/// `global_scale` of ``QuantizedLinear``.
final class Raon1ScaledLinear: Linear, Quantized {
    let groupSize = 16
    let bits = 4
    let mode = QuantizationMode.nvfp4

    private(set) var parameterRevision = 0

    @ParameterInfo var scales: MLXArray
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray

    init(_ inputDims: Int, _ outputDims: Int) {
        self._scales.wrappedValue = MLXArray.zeros([outputDims, inputDims / 16], dtype: .uint8)
        self._globalScale.wrappedValue = MLXArray(Float(1))
        super.init(weight: MLXArray.zeros([outputDims, inputDims / 8], dtype: .uint32))
        freeze()
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { parameterRevision &+= 1 }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        scaleOutput(
            quantizedMM(
                x, weight, scales: scales, biases: nil, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode),
            globalScale)
    }
}

/// Several ``Raon1ScaledLinear`` on one input as a single product, for decode-sized inputs.
///
/// The parts keep row views of the stacked storage, so each weight exists once.
final class Raon1FusedProjection: @unchecked Sendable {
    /// Inputs with at most this many rows take the fused product.
    static let maximumRows = 8

    private let weight: MLXArray
    private let scales: MLXArray
    /// Each output row's per-tensor scale, float32.
    private let columnScale: MLXArray
    private let splits: [Int]

    /// Stacks `parts`, or returns `nil` unless every part is a ``Raon1ScaledLinear``.
    init?(_ parts: [Linear]) {
        guard let parts = parts as? [Raon1ScaledLinear], parts.count > 1,
            Set(parts.map { $0.weight.dim(1) }).count == 1
        else { return nil }
        weight = concatenated(parts.map(\.weight))
        scales = concatenated(parts.map(\.scales))
        columnScale = concatenated(
            parts.map { broadcast($0.globalScale.asType(.float32), to: [$0.weight.dim(0)]) })
        eval(weight, scales, columnScale)

        var offset = 0
        splits = parts.dropLast().map {
            offset += $0.weight.dim(0)
            return offset
        }
        for (part, bounds) in zip(parts, zip([0] + splits, splits + [weight.dim(0)])) {
            let rows = bounds.0 ..< bounds.1
            part.update(
                parameters: ModuleParameters.unflattened([
                    "weight": weight[rows], "scales": scales[rows],
                ]))
        }
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let y = quantizedMM(
            x, weight, scales: scales, biases: nil, transpose: true, groupSize: 16, bits: 4,
            mode: .nvfp4)
        return split(scaleOutput(y, columnScale), indices: splits, axis: -1)
    }
}

/// The fused projection of a module's parts, set by `prepare()`.
///
/// Replacing or updating a part disables fusion until the next preparation.
/// Training always takes the parts.
final class Raon1FusedSlot: @unchecked Sendable {
    private var projection: Raon1FusedProjection?
    private var parts: [ObjectIdentifier] = []
    private var revisions: [Int] = []

    func prepare(_ parts: [Linear]) {
        projection = Raon1FusedProjection(parts)
        self.parts = parts.map(ObjectIdentifier.init)
        revisions = parts.compactMap { ($0 as? Raon1ScaledLinear)?.parameterRevision }
    }

    func invalidate() {
        projection = nil
        parts = []
        revisions = []
    }

    /// The outputs of `parts` on `x`, fused for decode-sized inputs unless `training`.
    func project(_ x: MLXArray, _ parts: [Linear], training: Bool) -> [MLXArray] {
        if let projection, !training, x.size / x.dim(-1) <= Raon1FusedProjection.maximumRows,
            parts.map(ObjectIdentifier.init) == self.parts,
            zip(parts, revisions).allSatisfy({ part, revision in
                (part as? Raon1ScaledLinear)?.parameterRevision == revision
            })
        {
            return projection(x)
        }
        return parts.map { $0(x) }
    }
}

/// The experts of one projection, `[experts, out, in]`, applied to the routed rows.
final class Raon1ExpertProjection: Module, ParameterCountProviding {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray?
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?

    let quantization: BaseConfiguration.Quantization?

    init(
        experts: Int, inputDims: Int, outputDims: Int,
        quantization: BaseConfiguration.Quantization?
    ) {
        self.quantization = quantization
        if let quantization {
            let (groupSize, bits, mode) = quantization.asTuple
            self._weight.wrappedValue = MLXArray.zeros(
                [experts, outputDims, inputDims * bits / 32], dtype: .uint32)
            self._scales.wrappedValue = MLXArray.zeros(
                [experts, outputDims, inputDims / groupSize], dtype: .uint8)
            if mode == .nvfp4 {
                self._globalScale.wrappedValue = MLXArray.ones([experts], dtype: .float32)
            }
        } else {
            self._weight.wrappedValue = MLXArray.zeros([experts, outputDims, inputDims])
        }
        super.init()
        freeze()
    }

    var parameterCount: Int {
        guard let quantization else { return weight.size }
        return weight.size * 32 / quantization.bits
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        guard let quantization, let scales else {
            return gatherMM(
                x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
        }
        let (groupSize, bits, mode) = quantization.asTuple
        return gatherQuantizedMM(
            x, weight, scales: scales, rhsIndices: indices, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode, globalScale: globalScale,
            sortedIndices: sorted)
    }
}

// MARK: - Prepared state

/// A module with inference state derived from its parameters by `prepare()`.
///
/// A parameter or module update drops that state, so the module runs from its parameters until
/// `prepare()` runs again. The state is written only while the loader or an update has exclusive
/// access to the model, as ``LanguageModel/prepare()`` requires, so it needs no lock.
class Raon1PreparedModule: Module {
    /// Drops the state `prepare()` derived.
    func invalidate() {}

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        // An update can throw after changing some tensors, so invalidate in both cases.
        defer { invalidate() }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { invalidate() }
        try super.updateModule(key: key, value)
    }
}

// MARK: - mHC residual streams

final class Raon1StreamMix: Module {
    @ParameterInfo(key: "phi_z") var phiZ: MLXArray
    @ParameterInfo(key: "alpha_z") var alphaZ: MLXArray
    @ParameterInfo(key: "b_z") var bZ: MLXArray
    @ParameterInfo var eta: MLXArray

    init(streams: Int, hiddenSize: Int) {
        let size = (streams - 1) * (streams - 1)
        self._phiZ.wrappedValue = MLXArray.zeros([size, streams * hiddenSize])
        self._alphaZ.wrappedValue = MLXArray(Float(1))
        self._bZ.wrappedValue = MLXArray.zeros([size])
        self._eta.wrappedValue = MLXArray(Float(0))
    }
}

/// The stream coefficients of one mHC site, all float32.
struct Raon1Route {
    /// `[N, n]`: how much of each stream the branch reads.
    let pre: MLXArray
    /// `[N, n]`: how much of the branch output each stream receives.
    let post: MLXArray
    /// `[N, n, n]`: the stream-to-stream residual mix.
    let residual: MLXArray
}

/// A branch output `[N, H]` that the next site mixes into the rows, with the float32 coefficients
/// of its site: `post` `[N, n]` and `residual` `[N, n, n]`.
struct Raon1PendingBranch {
    let output: MLXArray
    let post: MLXArray
    let residual: MLXArray
}

/// The arithmetic of the mHC sites, shared by every site.
///
/// The residual mix is `(1 - γ) I + γ Φ` with `Φ = B + E / (1 + ρ)`, `E = Q Z Qᵀ` for the
/// Helmert basis `Q`, `B = (I + 11ᵀ/n) / 2` and `ρ = max(0, max(-E / B))`.
struct Raon1MHCArithmetic: @unchecked Sendable {
    let streams: Int
    let eps: Float
    let postFactor: Float
    let normEps: Float
    /// `[n, n - 1]`: an orthonormal basis orthogonal to the all-ones vector.
    let helmert: MLXArray
    /// `(I + 11ᵀ/n) / 2`.
    let center: MLXArray
    /// `[(n - 1)², n²]`: maps a chart `Z` to `Q Z Qᵀ`.
    private let chartBasis: MLXArray
    private let eye: MLXArray

    init(_ config: Raon1MiniConfiguration) {
        let n = config.mhcStreams
        let m = n - 1
        streams = n
        eps = config.mhcSelectorEps
        postFactor = config.mhcPostFactor
        normEps = config.rmsNormEps
        var basis = [Float](repeating: 0, count: n * m)
        for k in 1 ..< n {
            let norm = Float(Double(k * (k + 1)).squareRoot())
            for i in 0 ..< k { basis[i * m + k - 1] = 1 / norm }
            basis[k * m + k - 1] = -Float(k) / norm
        }
        var kron = [Float](repeating: 0, count: m * m * n * n)
        for a in 0 ..< m {
            for b in 0 ..< m {
                for i in 0 ..< n {
                    for j in 0 ..< n {
                        kron[(a * m + b) * n * n + i * n + j] = basis[i * m + a] * basis[j * m + b]
                    }
                }
            }
        }
        helmert = MLXArray(basis, [n, m])
        chartBasis = MLXArray(kron, [m * m, n * n])
        eye = MLXArray.eye(n, dtype: .float32)
        center = (eye + MLXArray.ones([n, n], dtype: .float32) / Float(n)) * 0.5
    }

    /// `[rows, branch?, post?, residual?, site parameters..., norm weight]` to
    /// `[rows, post, residual, x]`, mixing the branch first when there is one.
    func enter(_ inputs: [MLXArray], pending: Bool) -> [MLXArray] {
        var rows = inputs[0]
        var next = 1
        if pending {
            rows = mix(rows, branch: inputs[1], post: inputs[2], residual: inputs[3])
            next = 4
        }
        let route = self.route(rows, Array(inputs[next ..< (next + 8)]))
        let x = MLXFast.rmsNorm(aggregate(rows, route.pre), weight: inputs[next + 8], eps: normEps)
        return [rows, route.post, route.residual, x]
    }

    /// The coefficients for the rows `[N, n, H]` from `phi, α_pre, b_pre, α_post, b_post, α_z,
    /// b_z, η`.
    func route(_ rows: MLXArray, _ parameters: [MLXArray]) -> Raon1Route {
        let n = streams
        let f32 = parameters.dropFirst().map { $0.asType(.float32) }
        let (alphaPre, bPre, alphaPost, bPost) = (f32[0], f32[1], f32[2], f32[3])
        let (alphaZ, bZ, eta) = (f32[4], f32[5], f32[6])
        let flat = rows.reshaped(rows.dim(0), -1)
        // The RMS scale follows the selector product in the model dtype, as in training.
        let inverseRMS = rsqrt(
            square(flat.asType(.float32)).mean(axis: -1, keepDims: true) + eps)
        let z = matmul(flat, parameters[0].asType(flat.dtype).T).asType(.float32) * inverseRMS

        let pre = sigmoid(alphaPre * z[0..., ..<n] + bPre)
        let post = postFactor * sigmoid(alphaPost * z[0..., n ..< (2 * n)] + bPost)
        let chart = alphaZ * z[0..., (2 * n)...] + bZ
        let e = matmul(chart, chartBasis).reshaped(-1, n, n)
        let rho = maximum((-e / center).max(axes: [-2, -1]), 0)
        let mixed = center + e / (1 + rho).reshaped(-1, 1, 1)
        let gamma = sigmoid(eta)
        return Raon1Route(pre: pre, post: post, residual: (1 - gamma) * eye + gamma * mixed)
    }

    // The stream sums below are unrolled: batched [n, n] matmuls over every row are far slower
    // than one fused elementwise kernel.

    /// The branch input `[N, H]` from the rows `[N, n, H]`.
    func aggregate(_ rows: MLXArray, _ pre: MLXArray) -> MLXArray {
        (0 ..< streams).map { pre[0..., $0 ..< ($0 + 1)] * rows[0..., $0].asType(.float32) }
            .reduce(MLXArray(Float(0)), +).asType(rows.dtype)
    }

    /// The next rows: the residual mix plus the branch output `[N, H]`, rounded once.
    func mix(_ rows: MLXArray, branch: MLXArray, post: MLXArray, residual: MLXArray) -> MLXArray {
        (0 ..< streams).map {
            residual[0..., 0..., $0 ..< ($0 + 1)] * rows[0..., $0 ..< ($0 + 1)].asType(.float32)
        }
        .reduce(
            post.expandedDimensions(axis: -1) * branch.asType(.float32).expandedDimensions(axis: 1),
            +
        )
        .asType(rows.dtype)
    }

    func mix(_ rows: MLXArray, _ pending: Raon1PendingBranch) -> MLXArray {
        mix(rows, branch: pending.output, post: pending.post, residual: pending.residual)
    }
}

/// ``Raon1MHCArithmetic`` compiled once for every site, with the constants the kernels read.
struct Raon1MHCFunctions: @unchecked Sendable {
    let arithmetic: Raon1MHCArithmetic
    /// The site without a pending branch, before the first layer.
    let enter: @Sendable ([MLXArray]) -> [MLXArray]
    let mixAndEnter: @Sendable ([MLXArray]) -> [MLXArray]
    /// The norm eps as a float32 `[1]`.
    let normEps: MLXArray

    init(_ config: Raon1MiniConfiguration) {
        let arithmetic = Raon1MHCArithmetic(config)
        self.arithmetic = arithmetic
        enter = compile { arithmetic.enter($0, pending: false) }
        mixAndEnter = compile { arithmetic.enter($0, pending: true) }
        normEps = MLXArray([arithmetic.normEps])
    }
}

/// The derived inputs of an mHC site, set by `prepare()`.
private final class Raon1Selector: @unchecked Sendable {
    /// `phi_pre`, `phi_post` and `phi_z` stacked for the selector product.
    var weight: MLXArray?
    /// The site's scalars packed for ``Raon1MHCKernels``.
    var packed: MLXArray?
}

/// One mHC site: selects the branch input from the streams and writes the branch output back.
final class Raon1MHC: Raon1PreparedModule {
    @ParameterInfo(key: "phi_pre") var phiPre: MLXArray
    @ParameterInfo(key: "phi_post") var phiPost: MLXArray
    @ParameterInfo(key: "alpha_pre") var alphaPre: MLXArray
    @ParameterInfo(key: "alpha_post") var alphaPost: MLXArray
    @ParameterInfo(key: "b_pre") var bPre: MLXArray
    @ParameterInfo(key: "b_post") var bPost: MLXArray
    @ModuleInfo var res: Raon1StreamMix

    private let functions: Raon1MHCFunctions
    private let selector = Raon1Selector()

    init(_ config: Raon1MiniConfiguration, functions: Raon1MHCFunctions) {
        let n = config.mhcStreams
        let hidden = config.hiddenSize
        self._phiPre.wrappedValue = MLXArray.zeros([n, n * hidden])
        self._phiPost.wrappedValue = MLXArray.zeros([n, n * hidden])
        self._alphaPre.wrappedValue = MLXArray(Float(1))
        self._alphaPost.wrappedValue = MLXArray(Float(1))
        self._bPre.wrappedValue = MLXArray.zeros([n])
        self._bPost.wrappedValue = MLXArray.zeros([n])
        self._res.wrappedValue = Raon1StreamMix(streams: n, hiddenSize: hidden)
        self.functions = functions
        super.init()
    }

    func prepare() {
        let a = functions.arithmetic
        let weight = concatenated([phiPre, phiPost, res.phiZ])
        let packed = concatenated(
            [
                MLXArray([a.eps, a.postFactor]), alphaPre.reshaped(1), bPre, alphaPost.reshaped(1),
                bPost, res.alphaZ.reshaped(1), res.bZ,
                sigmoid(res.eta.asType(.float32)).reshaped(1),
                a.helmert.flattened(), a.center.flattened(),
            ].map { $0.asType(.float32) })
        eval(weight, packed)
        selector.weight = weight
        selector.packed = packed
    }

    override func invalidate() {
        selector.weight = nil
        selector.packed = nil
    }

    /// The site's parameters as the operations read them, from the live parameters so training
    /// differentiates them.
    private var parameters: [MLXArray] {
        [
            concatenated([phiPre, phiPost, res.phiZ]), alphaPre, bPre, alphaPost, bPost,
            res.alphaZ, res.bZ, res.eta,
        ]
    }

    /// Mixes the pending branch into the rows `[N, n, H]`, routes them and returns the
    /// normalized branch input `[N, H]`.
    func enter(_ rows: MLXArray, pending: Raon1PendingBranch?, norm: RMSNorm)
        -> (rows: MLXArray, x: MLXArray, post: MLXArray, residual: MLXArray)
    {
        if let weight = selector.weight, let packed = selector.packed, !training,
            Raon1MHCKernels.supports(rows)
        {
            return Raon1MHCKernels.enter(
                rows: rows, pending: pending.map { ($0.output, $0.post, $0.residual) },
                selector: weight, parameters: packed, normWeight: norm.weight,
                eps: functions.normEps)
        }
        return enterThroughOperations(rows, pending: pending, norm: norm)
    }

    /// ``enter(_:pending:norm:)`` through the compiled operations, which training differentiates.
    func enterThroughOperations(_ rows: MLXArray, pending: Raon1PendingBranch?, norm: RMSNorm)
        -> (rows: MLXArray, x: MLXArray, post: MLXArray, residual: MLXArray)
    {
        let out =
            if let pending {
                functions.mixAndEnter(
                    [rows, pending.output, pending.post, pending.residual] + parameters
                        + [norm.weight])
            } else {
                functions.enter([rows] + parameters + [norm.weight])
            }
        return (out[0], out[3], out[1], out[2])
    }
}

// MARK: - Raon Delta Attention

final class Raon1RDA: Raon1PreparedModule {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "rda_w_proj") var wProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "qkv_conv1d") var conv: Raon1ShortConvolution
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    private let heads: Int
    private let headDim: Int
    private let lowerBound: Float?
    private let eps: Float
    private let arithmetic: Raon1RDAArithmetic
    private let fused = Raon1FusedSlot()
    private let prepared = Raon1RDAPreparedConstants()

    init(
        _ config: Raon1MiniConfiguration, arithmetic: Raon1RDAArithmetic,
        linear: (String, Int, Int) -> Linear
    ) {
        self.arithmetic = arithmetic
        let hidden = config.hiddenSize
        (heads, headDim) = (config.rdaHeads, config.rdaHeadDim)
        lowerBound = config.rda.gateLowerBound
        eps = config.rmsNormEps
        let size = heads * headDim
        _qProj.wrappedValue = linear("q_proj", hidden, size)
        _kProj.wrappedValue = linear("k_proj", hidden, size)
        _vProj.wrappedValue = linear("v_proj", hidden, size)
        _gProj.wrappedValue = linear("g_proj", hidden, size)
        _fAProj.wrappedValue = linear("f_a_proj", hidden, config.rdaDecayRank)
        _fBProj.wrappedValue = linear("f_b_proj", config.rdaDecayRank, size)
        _bProj.wrappedValue = linear("b_proj", hidden, heads)
        _wProj.wrappedValue = linear("rda_w_proj", hidden, heads)
        _oProj.wrappedValue = linear("o_proj", size, hidden)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _conv.wrappedValue = Raon1ShortConvolution(
            channels: 3 * size, kernelSize: config.rda.convKernelSize)
        _aLog.wrappedValue = MLXArray.zeros([heads], dtype: .float32)
        _dtBias.wrappedValue = MLXArray.zeros([size], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        if mask == nil, !training, let cache, let constants = prepared.constants,
            lowerBound != nil, Raon1RDAKernels.supports(x, headDim: headDim)
        {
            let p = fused.project(x, inputProjections, training: training)
            let step = Raon1RDAKernels.prepare(
                q: p[0], k: p[1], v: p[2],
                state: cache[0]
                    ?? MLXArray.zeros(
                        [B, conv.kernelSize - 1, 3 * heads * headDim], dtype: x.dtype),
                b: p[5], w: p[6], decay: fBProj(p[4]), constants: constants, heads: heads,
                headDim: headDim, kernelSize: conv.kernelSize)
            let (gated, state) = Raon1RDAKernels.recurrence(
                q: step[0], k: step[1], v: step[2], g: step[3], beta: step[4],
                state: cache[1]
                    ?? MLXArray.zeros([B, heads, headDim, headDim], dtype: .float32),
                gate: p[3], constants: constants)
            Self.store(
                convolutionState: step[5], recurrentState: state,
                convolutionInputs: Array(p[0 ..< 3]), recurrence: Array(step[0 ..< 5]),
                in: cache)
            return oProj(gated.reshaped(B, L, -1))
        }

        let x = mask.map { MLX.where($0[.ellipsis, .newAxis], x, 0) } ?? x
        let p = fused.project(x, inputProjections, training: training)
        let keep = conv.kernelSize - 1
        let convInputs = Array(p[0 ..< 3])
        let convState = cache?[0] ?? MLXArray.zeros([B, keep, 3 * heads * headDim], dtype: x.dtype)
        let activations =
            if let constants = prepared.constants, !training, x.dtype == .bfloat16 {
                Raon1RDAKernels.convolution(
                    q: p[0], k: p[1], v: p[2], state: convState, weight: constants.convolution)
            } else {
                split(
                    conv(concatenated(convInputs, axis: -1), state: convState), parts: 3, axis: -1)
            }
        let gates = arithmetic.gates(activations + [p[5], p[6], fBProj(p[4]), dtBias, aLog])
        let (out, state) =
            if mask == nil, !training,
                Raon1RDAKernels.supportsPrompt(
                    q: gates[0], v: gates[2], g: gates[3], state: cache?[1])
            {
                Raon1RDAKernels.prompt(
                    q: gates[0], k: gates[1], v: gates[2], g: gates[3], beta: gates[4],
                    state: cache?[1]
                        ?? MLXArray.zeros([B, heads, headDim, headDim], dtype: .float32))
            } else {
                gatedDeltaRecurrence(
                    q: gates[0], k: gates[1], v: gates[2], g: gates[3], beta: gates[4],
                    state: cache?[1], mask: mask, useKernel: !training)
            }
        if let cache {
            Self.store(
                convolutionState: Raon1ShortConvolution.nextState(
                    convState, inputs: convInputs, keep: keep),
                recurrentState: state, convolutionInputs: convInputs,
                recurrence: Array(gates[0 ..< 5]), in: cache)
        }
        return oProj(arithmetic.output([out, p[3], oNorm.weight])[0].reshaped(B, L, -1))
    }

    /// Stores a call's states in `cache`; a rewindable cache also keeps the call's inputs
    /// `[q, k, v, g, beta]`, so a speculative round can take it back.
    private static func store(
        convolutionState: MLXArray, recurrentState: MLXArray, convolutionInputs: [MLXArray],
        recurrence: [MLXArray], in cache: MambaCache
    ) {
        if let cache = cache as? RewindableGatedDeltaCache {
            cache.update(
                convolutionState: convolutionState, recurrentState: recurrentState,
                convolutionInputs: convolutionInputs, q: recurrence[0], k: recurrence[1],
                v: recurrence[2], g: recurrence[3], beta: recurrence[4])
        } else {
            cache[0] = convolutionState
            cache[1] = recurrentState
            cache.advance(recurrence[0].dim(1))
        }
    }

    /// The projections of the layer input, in the order ``callAsFunction(_:mask:cache:)`` reads.
    var inputProjections: [Linear] { [qProj, kProj, vProj, gProj, fAProj, bProj, wProj] }

    func prepare() {
        fused.prepare(inputProjections)
        let constants = Raon1RDAKernels.Constants(
            convolution: conv.weight.asType(.float32).reshaped(conv.weight.dim(0), -1),
            dtBias: dtBias.asType(.float32).reshaped(heads, headDim),
            decayScale: exp(aLog.asType(.float32)),
            scalars: MLXArray([1 / Float(headDim).squareRoot(), lowerBound ?? 0]),
            normWeight: oNorm.weight.asType(.float32),
            eps: MLXArray([eps]))
        eval(
            constants.convolution, constants.dtBias, constants.decayScale, constants.normWeight)
        prepared.constants = constants
    }

    override func invalidate() {
        fused.invalidate()
        prepared.constants = nil
    }
}

/// The constants of the RDA kernels, set by `prepare()`.
final class Raon1RDAPreparedConstants: @unchecked Sendable {
    var constants: Raon1RDAKernels.Constants?
}

/// The elementwise arithmetic of RDA around its projections and recurrence, compiled once.
final class Raon1RDAArithmetic: @unchecked Sendable {
    /// `[q, k, v, b, w, decay, dt_bias, A_log]` after the convolution to `[q, k, v, g, beta]` for
    /// the recurrence.
    let gates: @Sendable ([MLXArray]) -> [MLXArray]
    /// `[out, gate, o_norm weight]` to the normalized, gated output.
    let output: @Sendable ([MLXArray]) -> [MLXArray]

    init(_ config: Raon1MiniConfiguration) {
        let (heads, headDim) = (config.rdaHeads, config.rdaHeadDim)
        let lowerBound = config.rda.gateLowerBound
        let eps = config.rmsNormEps

        /// `x / max(‖x‖, 1e-6)` along the head dim, in float32.
        @Sendable func l2Normalized(_ x: MLXArray) -> MLXArray {
            let x32 = x.asType(.float32)
            let norm = sqrt(square(x32).sum(axis: -1, keepDims: true))
            return (x32 / maximum(norm, 1e-6)).asType(x.dtype)
        }

        gates = compile { inputs in
            let (b, w, decay, dtBias, aLog) = (
                inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]
            )
            let shape = [inputs[0].dim(0), inputs[0].dim(1), heads, headDim]
            let q = l2Normalized(inputs[0].reshaped(shape)) * (1 / Float(headDim).squareRoot())
            let k = l2Normalized(inputs[1].reshaped(shape))

            // Write gate sigmoid(b + w), erase gate sigmoid(b): the ratio scales the values.
            let betaRaw = b.asType(.float32)
            let ratio = exp(softplus(-betaRaw) - softplus(-betaRaw - w.asType(.float32)))
            let v =
                (inputs[2].reshaped(shape).asType(.float32) * ratio.expandedDimensions(axis: -1))
                .asType(inputs[2].dtype)

            let scaledDecay =
                (decay.reshaped(shape).asType(.float32) + dtBias.reshaped(heads, headDim))
                * exp(aLog).expandedDimensions(axis: -1)
            let logDecay = lowerBound.map { $0 * sigmoid(scaledDecay) } ?? -softplus(scaledDecay)
            return [q, k, v, exp(logDecay), sigmoid(betaRaw).asType(inputs[0].dtype)]
        }

        output = compile { inputs in
            let (out, gate, weight) = (inputs[0], inputs[1], inputs[2])
            let normed = MLXFast.rmsNorm(
                out.asType(.float32), weight: weight.asType(.float32), eps: eps)
            return [
                (normed * sigmoid(gate.reshaped(out.shape).asType(.float32))).asType(gate.dtype)
            ]
        }
    }
}

/// The causal depthwise convolution of q, k and v, followed by SiLU, in float32.
final class Raon1ShortConvolution: Module {
    @ParameterInfo var weight: MLXArray

    let kernelSize: Int

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
    }

    /// The activations of `x` `[B, L, C]` after the raw inputs `state` `[B, K - 1, C]`.
    func callAsFunction(_ x: MLXArray, state: MLXArray) -> MLXArray {
        let signal = concatenated([state, x], axis: 1)
        let out = conv1d(signal.asType(.float32), weight.asType(.float32), groups: x.dim(-1))
        return silu(out).asType(x.dtype)
    }

    /// The last `keep` raw inputs after `state`, with `inputs` concatenated along channels.
    static func nextState(_ state: MLXArray, inputs: [MLXArray], keep: Int) -> MLXArray {
        let length = inputs[0].dim(1)
        if length >= keep {
            return concatenated(inputs.map { $0[0..., (length - keep)...] }, axis: -1)
        }
        let signal = concatenated([state, concatenated(inputs, axis: -1)], axis: 1)
        return contiguous(signal[0..., (-keep)..., 0...])
    }
}

// MARK: - Multi-head latent attention

/// The attention mask of an MLA prompt chunk.
enum Raon1PromptMask {
    /// One token: no mask.
    case none
    /// Causal over a cache that presents every earlier key.
    case causal
    /// Any other mask, `[L, T]` or `[B, 1, L, T]`.
    case array(MLXArray)

    init(_ h: MLXArray, cache: KVCache?) {
        guard h.dim(1) > 1 else {
            self = .none
            return
        }
        if cache.map({ type(of: $0) == KVCacheSimple.self || $0 is QuantizedKVCacheProtocol })
            ?? true
        {
            self = .causal
        } else if case .array(let mask) = createAttentionMask(h: h, cache: cache, returnArray: true)
        {
            self = .array(mask)
        } else {
            self = .causal
        }
    }
}

/// NoPE MLA with a sigmoid output gate.
///
/// The cache holds the normalized latent `[B, 1, T, kv_lora_rank]` as keys and the shared
/// key channels `[B, 1, T, qk_rope_head_dim]` as values. The shared channels are never rotated.
/// Decode attends in the latent space; a prompt expands per-head keys and values and adds the
/// shared-channel scores as an additive mask.
final class Raon1MLA: Raon1PreparedModule {
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: MultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: MultiLinear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let nopeDim: Int
    let sharedDim: Int
    let latentDim: Int
    let scale: Float
    private let fused = Raon1FusedSlot()

    /// The largest shared-channel score block of a prompt, in bytes.
    static let maximumMaskBytes = 256 << 20
    static let minimumBlockRows = 32

    init(_ config: Raon1MiniConfiguration, linear: (String, Int, Int) -> Linear) {
        let hidden = config.hiddenSize
        heads = config.attentionHeads
        nopeDim = config.qkNopeHeadDim
        sharedDim = config.qkRopeHeadDim
        latentDim = config.kvLoraRank
        scale = 1 / Float(nopeDim + sharedDim).squareRoot()
        _qAProj.wrappedValue = linear("q_a_proj", hidden, config.qLoraRank)
        _qANorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.mlaLatentNormEps)
        _qBProj.wrappedValue = linear("q_b_proj", config.qLoraRank, heads * (nopeDim + sharedDim))
        _kvAProj.wrappedValue = linear("kv_a_proj_with_mqa", hidden, latentDim + sharedDim)
        _kvANorm.wrappedValue = RMSNorm(dimensions: latentDim, eps: config.mlaLatentNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: nopeDim, outputDims: latentDim, numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: latentDim, outputDims: config.vHeadDim, numHeads: heads)
        if config.mlaUseOutputGate {
            _gProj.wrappedValue = linear("g_proj", hidden, heads * config.vHeadDim)
        }
        _oProj.wrappedValue = linear("o_proj", heads * config.vHeadDim, hidden)
    }

    func callAsFunction(_ x: MLXArray, mask: Raon1PromptMask, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let p = fused.project(x, inputProjections, training: training)
        let q = qBProj(qANorm(p[0]))
            .reshaped(B, L, heads, nopeDim + sharedDim).transposed(0, 2, 1, 3)
        let qNope = q[.ellipsis, ..<nopeDim]
        let qShared = q[.ellipsis, nopeDim...] * scale

        let kv = p[1]
        var latent = kvANorm(kv[.ellipsis, ..<latentDim]).expandedDimensions(axis: 1)
        var shared = kv[.ellipsis, latentDim...].expandedDimensions(axis: 1)

        var out: MLXArray
        if let quantized = cache as? QuantizedKVCacheProtocol {
            let (qLatent, qSharedKeys) = quantized.updateQuantized(keys: latent, values: shared)
            let (groupSize, bits, mode) = (quantized.groupSize, quantized.bits, quantized.mode)
            if L == 1 {
                let scores =
                    quantizedMM(
                        embedQ(qNope) * scale, qLatent.0, scales: qLatent.1, biases: qLatent.2,
                        transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                    + quantizedMM(
                        qShared, qSharedKeys.0, scales: qSharedKeys.1, biases: qSharedKeys.2,
                        transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let weights = softmax(scores, axis: -1, precise: true)
                out = unembedOut(
                    quantizedMM(
                        weights, qLatent.0, scales: qLatent.1, biases: qLatent.2,
                        transpose: false, groupSize: groupSize, bits: bits, mode: mode))
            } else {
                latent = dequantized(
                    qLatent.0, scales: qLatent.1, biases: qLatent.2, groupSize: groupSize,
                    bits: bits, mode: mode, dtype: x.dtype)
                shared = dequantized(
                    qSharedKeys.0, scales: qSharedKeys.1, biases: qSharedKeys.2,
                    groupSize: groupSize, bits: bits, mode: mode, dtype: x.dtype)
                out = promptAttention(qNope, qShared, latent: latent, shared: shared, mask: mask)
            }
        } else {
            if let cache {
                (latent, shared) = cache.update(keys: latent, values: shared)
            }
            if L == 1 {
                // The cached keys are the latent itself, so the latent serves as both keys and values.
                let scores =
                    matmul(embedQ(qNope) * scale, latent.swappedAxes(-1, -2))
                    + matmul(qShared, shared.swappedAxes(-1, -2))
                let weights = softmax(scores, axis: -1, precise: true)
                out = unembedOut(matmul(weights, latent))
            } else {
                out = promptAttention(qNope, qShared, latent: latent, shared: shared, mask: mask)
            }
        }

        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        if gProj != nil {
            out = out * sigmoid(p[2])
        }
        return oProj(out)
    }

    /// The projections of the layer input, in the order ``callAsFunction(_:mask:cache:)`` reads.
    var inputProjections: [Linear] { [qAProj, kvAProj] + (gProj.map { [$0] } ?? []) }

    func prepare() { fused.prepare(inputProjections) }

    override func invalidate() { fused.invalidate() }

    /// Prompt attention with per-head keys and values expanded from the latent.
    private func promptAttention(
        _ qNope: MLXArray, _ qShared: MLXArray, latent: MLXArray, shared: MLXArray,
        mask: Raon1PromptMask
    ) -> MLXArray {
        let keys = matmul(latent, embedQ.weight)
        let values = unembedOut(latent)
        let sharedKeys = shared.swappedAxes(-1, -2)
        let (length, keyCount) = (qNope.dim(2), keys.dim(2))
        let fill = MLXArray.maskFill(for: qShared.dtype)

        // The shared-channel scores are an additive mask [B, heads, rows, keys]. Query blocks
        // bound it for long contexts.
        let rowsPerBlock = max(
            Self.minimumBlockRows,
            Self.maximumMaskBytes / (qNope.dim(0) * heads * keyCount * qShared.dtype.size))
        var outputs = [MLXArray]()
        for start in stride(from: 0, to: length, by: rowsPerBlock) {
            let rows = start ..< min(start + rowsPerBlock, length)
            var scores = matmul(qShared[0..., 0..., rows], sharedKeys)
            switch mask {
            case .causal:
                // Keys before the chunk are all visible, so only the chunk's own keys are masked.
                let own = keyCount - length
                scores[.ellipsis, own...] = MLX.where(
                    MLXArray(Int32(rows.lowerBound) ..< Int32(rows.upperBound))[0..., .newAxis]
                        .>= MLXArray(Int32(0) ..< Int32(length))[.newAxis],
                    scores[.ellipsis, own...], fill)
            case .array(let mask):
                scores = MLX.where(mask[.ellipsis, rows, 0...], scores, fill)
            case .none:
                break
            }
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: qNope[0..., 0..., rows], keys: keys, values: values, scale: scale,
                    mask: .array(scores)))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }
}

// MARK: - Feed forward

final class Raon1MLP: Raon1PreparedModule, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray
    private let fused = Raon1FusedSlot()

    init(
        hiddenSize: Int, intermediateSize: Int,
        activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray,
        linear: (String, Int, Int) -> Linear
    ) {
        self.activation = activation
        _gateProj.wrappedValue = linear("gate_proj", hiddenSize, intermediateSize)
        _upProj.wrappedValue = linear("up_proj", hiddenSize, intermediateSize)
        _downProj.wrappedValue = linear("down_proj", intermediateSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = fused.project(x, [upProj, gateProj], training: training)
        return downProj(activation(p[0], p[1]))
    }

    func prepare() { fused.prepare([upProj, gateProj]) }

    override func invalidate() { fused.invalidate() }

    /// The projections for ``Raon1MoEKernels``, or `nil` unless each is a ``Raon1ScaledLinear``.
    var kernelProjections: Raon1MoEKernels.Projections? {
        guard let up = upProj as? Raon1ScaledLinear, let gate = gateProj as? Raon1ScaledLinear,
            let down = downProj as? Raon1ScaledLinear
        else { return nil }
        func weights(_ p: Raon1ScaledLinear) -> Raon1MoEKernels.Weights {
            (p.weight, p.scales, p.globalScale.reshaped(1))
        }
        return .init(up: weights(up), gate: weights(gate), down: weights(down))
    }
}

/// Sigmoid router whose expert bias only ranks the experts.
final class Raon1Router: Module {
    /// Float32, cast at load.
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray

    init(_ config: Raon1MiniConfiguration) {
        _weight.wrappedValue = MLXArray.zeros([config.experts, config.hiddenSize], dtype: .float32)
        _bias.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
    }
}

/// The routing and combination arithmetic of the MoE layers, compiled once.
final class Raon1MoEArithmetic: @unchecked Sendable {
    /// `[x, router weight, expert bias]` to the top-k `[indices, weights]`.
    let route: @Sendable ([MLXArray]) -> [MLXArray]
    /// `[routed outputs, weights, shared output]` to the MoE output.
    let combine: @Sendable ([MLXArray]) -> [MLXArray]
    let topK: Int
    let renormalize: Bool
    /// float32 `[SiTU beta, SiTU linear beta, routed scale]` for ``Raon1MoEKernels``.
    let kernelParameters: MLXArray

    init(_ config: Raon1MiniConfiguration) {
        let topK = config.expertsPerToken
        let renormalize = config.moeRenormalize
        let scale = config.routedScalingFactor
        self.topK = topK
        self.renormalize = renormalize
        kernelParameters = MLXArray([config.situBeta, config.situLinearBeta, scale])
        route = compile { inputs in
            let scores = sigmoid(matmul(inputs[0].asType(.float32), inputs[1].T))
            let ranked = scores + inputs[2]
            let indices = argPartition(-ranked, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, indices, axis: -1)
            if renormalize {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            return [indices, weights * scale]
        }
        combine = compile { inputs in
            let (routed, weights, shared) = (inputs[0], inputs[1], inputs[2])
            let sum = (routed.asType(.float32) * weights.expandedDimensions(axis: -1))
                .sum(axis: -2).asType(shared.dtype)
            return [sum + shared]
        }
    }
}

/// Routed SiTU-GLU experts.
final class Raon1SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Raon1ExpertProjection
    @ModuleInfo(key: "up_proj") var upProj: Raon1ExpertProjection
    @ModuleInfo(key: "down_proj") var downProj: Raon1ExpertProjection

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(
        _ config: Raon1MiniConfiguration,
        activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
    ) {
        self.activation = activation
        let (experts, hidden) = (config.experts, config.hiddenSize)
        _gateProj.wrappedValue = Raon1ExpertProjection(
            experts: experts, inputDims: hidden, outputDims: config.moeIntermediateSize,
            quantization: config.quantization)
        _upProj.wrappedValue = Raon1ExpertProjection(
            experts: experts, inputDims: hidden, outputDims: config.moeIntermediateSize,
            quantization: config.quantization)
        _downProj.wrappedValue = Raon1ExpertProjection(
            experts: experts, inputDims: config.moeIntermediateSize, outputDims: hidden,
            quantization: config.quantization)
    }

    /// `[..., H]` rows and `[..., K]` experts to `[..., K, H]`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = x.expandedDimensions(axes: [-2, -3])
        var routes = indices
        var inverseOrder: MLXArray?
        let sorted = indices.size >= 64
        if sorted {
            let order: MLXArray
            (x, routes, order) = gatherSort(x: x, indices: indices)
            inverseOrder = order
        }
        let activated = activation(
            upProj(x, routes, sorted: sorted), gateProj(x, routes, sorted: sorted))
        var y = downProj(activated, routes, sorted: sorted)
        if let inverseOrder {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        return y.squeezed(axis: -2)
    }

    /// The projections for ``Raon1MoEKernels``, or `nil` unless each is NVFP4 with global scales.
    var kernelProjections: Raon1MoEKernels.Projections? {
        func weights(_ p: Raon1ExpertProjection) -> Raon1MoEKernels.Weights? {
            guard let quantization = p.quantization, quantization.mode == .nvfp4,
                quantization.groupSize == 16, quantization.bits == 4, let scales = p.scales,
                let globalScale = p.globalScale
            else { return nil }
            return (p.weight, scales, globalScale)
        }
        guard let up = weights(upProj), let gate = weights(gateProj), let down = weights(downProj)
        else { return nil }
        return .init(up: up, gate: gate, down: down)
    }
}

final class Raon1MoE: Module {
    @ModuleInfo var gate: Raon1Router
    @ModuleInfo(key: "switch_mlp") var switchMLP: Raon1SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Raon1MLP?
    /// The reasoning-effort steering vector at alpha 1, float32.
    @ParameterInfo(key: "reasoning_effort_steering") var steering: MLXArray?

    private let arithmetic: Raon1MoEArithmetic

    init(
        _ config: Raon1MiniConfiguration, steered: Bool, arithmetic: Raon1Arithmetic,
        linear: (String, Int, Int) -> Linear
    ) {
        self.arithmetic = arithmetic.moe
        let activation = arithmetic.situ
        _gate.wrappedValue = Raon1Router(config)
        _switchMLP.wrappedValue = Raon1SwitchGLU(config, activation: activation)
        if config.sharedExperts > 0 {
            _sharedExperts.wrappedValue = Raon1MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.sharedExperts,
                activation: activation, linear: { linear("shared_experts.\($0)", $1, $2) })
        }
        if steered {
            _steering.wrappedValue = MLXArray.zeros([config.hiddenSize], dtype: .float32)
        }
    }

    func callAsFunction(_ x: MLXArray, effort: Float) -> MLXArray {
        let out: MLXArray
        if let kernelOut = kernelOutput(x) {
            out = kernelOut
        } else {
            let route = arithmetic.route([x, gate.weight, gate.bias])
            let shared = sharedExperts?(x) ?? MLXArray.zeros(like: x)
            out = arithmetic.combine([switchMLP(x, route[0]), route[1], shared])[0]
        }
        guard let steering, effort != 0 else { return out }
        return (out.asType(.float32) + effort * steering).asType(out.dtype)
    }

    /// The output through ``Raon1MoEKernels`` for decode-sized inputs, or `nil`.
    private func kernelOutput(_ x: MLXArray) -> MLXArray? {
        guard !training, gate.weight.dtype == .float32, gate.bias.dtype == .float32,
            let routed = switchMLP.kernelProjections,
            Raon1MoEKernels.supports(
                x, experts: gate.weight.dim(0), intermediate: routed.up.weight.dim(1),
                topK: arithmetic.topK)
        else { return nil }
        var shared: Raon1MoEKernels.Projections?
        if let sharedExperts {
            guard let projections = sharedExperts.kernelProjections,
                projections.up.weight.dim(0) == routed.up.weight.dim(1)
            else { return nil }
            shared = projections
        }
        let hidden = x.dim(-1)
        return Raon1MoEKernels.apply(
            x.reshaped(-1, hidden), router: (gate.weight, gate.bias), routed: routed,
            shared: shared, topK: arithmetic.topK, renormalize: arithmetic.renormalize,
            parameters: arithmetic.kernelParameters
        ).reshaped(x.shape)
    }
}

// MARK: - Decoder

final class Raon1DecoderLayer: Module {
    let isRDA: Bool

    @ModuleInfo(key: "self_attn") var attention: Module
    @ModuleInfo(key: "self_attn_raon_mhc") var attentionMHC: Raon1MHC
    @ModuleInfo(key: "mlp_raon_mhc") var mlpMHC: Raon1MHC
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo var mlp: Raon1MLP?
    @ModuleInfo var moe: Raon1MoE?

    init(_ config: Raon1MiniConfiguration, index: Int, arithmetic: Raon1Arithmetic) {
        isRDA = config.isRDALayer(index)
        let activation = arithmetic.situ
        let scaled = Set(config.globalScaledLinears)
        func linear(_ path: String) -> (String, Int, Int) -> Linear {
            { name, input, output in
                scaled.contains("model.layers.\(index).\(path).\(name)")
                    ? Raon1ScaledLinear(input, output) : Linear(input, output, bias: false)
            }
        }

        if isRDA {
            _attention.wrappedValue = Raon1RDA(
                config, arithmetic: arithmetic.rda, linear: linear("self_attn"))
        } else {
            _attention.wrappedValue = Raon1MLA(config, linear: linear("self_attn"))
        }
        _attentionMHC.wrappedValue = Raon1MHC(config, functions: arithmetic.mhc)
        _mlpMHC.wrappedValue = Raon1MHC(config, functions: arithmetic.mhc)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.isMoELayer(index) {
            _moe.wrappedValue = Raon1MoE(
                config, steered: config.effortSteering?.layers.contains(index) ?? false,
                arithmetic: arithmetic, linear: linear("moe"))
        } else {
            _mlp.wrappedValue = Raon1MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                activation: activation, linear: linear("mlp"))
        }
    }

    /// Runs the layer on the streams `[B * L, n, H]`. Each branch output stays pending until the
    /// next site mixes it in, so the arithmetic between two branches is one compiled call.
    ///
    /// Also returns the streams the layer entered: the previous layer's output, `[B * L, n, H]`.
    func callAsFunction(
        _ rows: MLXArray, pending: Raon1PendingBranch?, shape: (B: Int, L: Int),
        ssmMask: MLXArray?, attentionMask: Raon1PromptMask, cache: KVCache?, effort: Float
    ) -> (rows: MLXArray, pending: Raon1PendingBranch, entered: MLXArray) {
        let hidden = rows.dim(-1)
        let attentionSite = attentionMHC.enter(rows, pending: pending, norm: inputLayerNorm)
        let x = attentionSite.x.reshaped(shape.B, shape.L, hidden)
        let attended =
            if isRDA {
                (attention as! Raon1RDA)(x, mask: ssmMask, cache: cache as? MambaCache)
            } else {
                (attention as! Raon1MLA)(x, mask: attentionMask, cache: cache)
            }

        let mlpSite = mlpMHC.enter(
            attentionSite.rows,
            pending: Raon1PendingBranch(
                output: attended.reshaped(-1, hidden), post: attentionSite.post,
                residual: attentionSite.residual),
            norm: postAttentionLayerNorm)
        let y = mlpSite.x.reshaped(shape.B, shape.L, hidden)
        let branch = moe.map { $0(y, effort: effort) } ?? mlp!(y)
        return (
            mlpSite.rows,
            Raon1PendingBranch(
                output: branch.reshaped(-1, hidden), post: mlpSite.post,
                residual: mlpSite.residual),
            attentionSite.rows
        )
    }
}

public class Raon1MiniModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Raon1DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let streams: Int
    let steering: Raon1MiniConfiguration.EffortSteering?
    let rdaIndex: Int?
    let mlaIndex: Int?
    private let arithmetic: Raon1Arithmetic

    init(_ config: Raon1MiniConfiguration) {
        streams = config.mhcStreams
        steering = config.effortSteering
        let arithmetic = Raon1Arithmetic(config)
        self.arithmetic = arithmetic
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Raon1DecoderLayer(config, index: $0, arithmetic: arithmetic)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        rdaIndex = (0 ..< config.hiddenLayers).first(where: config.isRDALayer)
        mlaIndex = (0 ..< config.hiddenLayers).first { !config.isRDALayer($0) }
    }

    /// The final hidden states, with the steering of `effort` added to the steered MoE layers.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, effort: ReasoningEffort?)
        -> MLXArray
    {
        forward(inputs, cache: cache, effort: effort, tapping: []).hidden
    }

    /// The final hidden states and the streams after each layer in `tapping`, in layer order,
    /// concatenated to `[B, L, layers * n * H]`, or `nil` when `tapping` is empty.
    func forward(
        _ inputs: MLXArray, cache: [KVCache]?, effort: ReasoningEffort?, tapping: Set<Int>
    ) -> (hidden: MLXArray, taps: MLXArray?) {
        let effort = steering?.alpha(for: effort) ?? 0
        let h = embedTokens(inputs)
        let (B, L, hidden) = (h.dim(0), h.dim(1), h.dim(2))

        let ssmMask = rdaIndex.flatMap {
            createSSMMask(h: h, cache: cache?[$0] as? MambaCache)
        }
        let attentionMask = Raon1PromptMask(h, cache: mlaIndex.flatMap { cache?[$0] })

        var rows = repeated(h.reshaped(B * L, 1, hidden), count: streams, axis: 1)
        var pending: Raon1PendingBranch?
        var taps = [MLXArray]()
        for (index, layer) in layers.enumerated() {
            let entered: MLXArray
            (rows, pending, entered) = layer(
                rows, pending: pending, shape: (B, L), ssmMask: ssmMask,
                attentionMask: attentionMask, cache: cache?[index], effort: effort)
            if tapping.contains(index - 1) { taps.append(entered.reshaped(B, L, -1)) }
        }
        if let pending {
            rows = arithmetic.mhc.arithmetic.mix(rows, pending)
        }
        if tapping.contains(layers.count - 1) { taps.append(rows.reshaped(B, L, -1)) }
        let merged = rows.asType(.float32).mean(axis: 1).asType(h.dtype)
        return (
            norm(merged.reshaped(B, L, hidden)),
            taps.isEmpty ? nil : concatenated(taps, axis: -1)
        )
    }
}

public class Raon1MiniModel: Module, LLMModel, AdditionalWeightFilesProviding,
    SpeculativeCacheRewindModel, ParameterCountProviding
{
    public let vocabularySize: Int

    public let model: Raon1MiniModelInner
    let configuration: Raon1MiniConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    /// The 2-bit copy of `lm_head` that shortlists the decode candidates, when the checkpoint
    /// ships one. See ``shortlistLogits(_:)``.
    @ModuleInfo(key: "lm_head_stage1") var shortlistHead: QuantizedLinear?

    /// KRAFTON's shortlist head: affine 2-bit, group 64, fitted with GPTQ on calibration
    /// activations.
    static let shortlistFile = "head_stage1_a2g64gm.safetensors"
    /// Rows the shortlist head serves; more rows read the full head more cheaply.
    static let shortlistMaximumRows = 16
    /// Candidates whose logits are exact.
    static let shortlistCandidates = 256
    /// Every other logit is capped at the candidate of this rank.
    static let shortlistTailRank = 64

    public var additionalWeightFiles: [String] { [Self.shortlistFile] }

    package var parameterCount: Int {
        // The shortlist is a second representation of lm_head.
        model.numParameters() + (lmHead?.numParameters() ?? 0)
    }

    public var supportedReasoningEfforts: Set<ReasoningEffort> {
        guard let steering = configuration.effortSteering, !steering.layers.isEmpty else {
            return []
        }
        return Set(steering.alpha.keys.map(ReasoningEffort.init)).union([.off])
    }

    /// Positions a speculative round can take back: the RDA caches replay what they keep.
    public let maximumNativeTargetCacheRewind = 16

    public init(_ config: Raon1MiniConfiguration) {
        configuration = config
        vocabularySize = config.vocabularySize
        model = Raon1MiniModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(hiddenStates(inputs, cache: cache))
    }

    /// Steers with the request's ``reasoningEffortKey`` and returns `state` unchanged.
    ///
    /// When a drafter asks, it also emits the drafter's target features, and the logits of
    /// every position follow the same accuracy policy as plain decoding.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let effort = state?[reasoningEffortKey]
        guard let tapping = Self.drafterTaps(state) else {
            let h = model(input.tokens, cache: cache, effort: effort)
            return .init(logits: projectLogits(h), state: state)
        }
        let output = model.forward(input.tokens, cache: cache, effort: effort, tapping: tapping)
        return .init(
            logits: generationLogits(output.hidden, state: state),
            state: Self.drafterState(state, features: output.taps ?? output.hidden))
    }

    /// Prefills in chunks. When a drafter asks, also emits the drafter's target features of the
    /// last prompt positions it attends, and returns the logits of the last position.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let tapping = Self.drafterTaps(state) else {
            return try prepareInChunks(input, cache: cache, state: state, prefill: prefill)
        }
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let total = tokens.dim(1)
        let context = min(total, state?[mtpTargetContextLengthKey] ?? total)
        let effort = state?[reasoningEffortKey]

        var features = [MLXArray]()
        let processed = try prefill.forEachChunk(total: total) { range in
            // Chunks before the drafter's context need no features.
            let tapped = range.upperBound > total - context
            let output = model.forward(
                tokens[0..., range], cache: cache, effort: effort, tapping: tapped ? tapping : [])
            if tapped { features.append(output.taps ?? output.hidden) }
            asyncEval(cache, features)
        }
        if processed > 0 { eval(cache) }
        let output = model.forward(
            tokens[0..., processed...], cache: cache, effort: effort, tapping: tapping)
        features.append(output.taps ?? output.hidden)
        prefill.progress?(total, total)

        let all = concatenated(features, axis: 1)
        let last = output.hidden[0..., (-1)...]
        return .logits(
            LMOutput(
                logits: generationLogits(last, state: state),
                state: Self.drafterState(state, features: all[0..., (all.dim(1) - context)...])))
    }

    /// The layers whose streams a drafter reads, or `nil` when no drafter asks. Without layers a
    /// drafter reads the final hidden states.
    private static func drafterTaps(_ state: LMOutput.State?) -> Set<Int>? {
        guard state?[mtpEmitFlagKey] ?? false else { return nil }
        return Set(state?[mtpTargetLayerIdsKey] ?? [])
    }

    /// `state` with the drafter's features `[B, L, F]`. The drafter reads no K/V.
    private static func drafterState(_ state: LMOutput.State?, features: MLXArray)
        -> LMOutput.State
    {
        var state = state ?? LMOutput.State()
        state[mtpLastHiddenStatesKey] = features
        state[mtpSharedKVStatesKey] = [:]
        state[mtpSharedKVOffsetsKey] = [:]
        state[mtpSharedKVSourceIndicesKey] = [:]
        return state
    }

    /// Projects only the last position. The shortlist head requires request opt-in.
    /// Scoring without a drafter uses the full head.
    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let h = model(input.tokens, cache: cache, effort: state?[reasoningEffortKey])
        let last = h[0..., (-1)...]
        return .init(logits: generationLogits(last, state: state), state: state)
    }

    private func generationLogits(_ hidden: MLXArray, state: LMOutput.State?) -> MLXArray {
        if state?[logitsAccuracyKey] == .allowApproximate, let logits = shortlistLogits(hidden) {
            return logits
        }
        return projectLogits(hidden)
    }

    /// The logits of the final hidden states `h` `[..., H]` through the shortlist head, or `nil`
    /// without one or for more than ``shortlistMaximumRows`` rows.
    ///
    /// A 2-bit pass scores the vocabulary and the full head recomputes the best
    /// ``shortlistCandidates``, so their logits are exact. Every other logit keeps its 2-bit
    /// score, capped at the exact logit of rank ``shortlistTailRank``. The arg max and any top-k
    /// up to that rank are exact whenever the 2-bit pass ranks the true candidates within the
    /// shortlist.
    ///
    /// Measured over 2,047 teacher-forced tokens: the arg max matched every time, and the
    /// mean KL divergence from the full head was 0.004.
    func shortlistLogits(_ h: MLXArray) -> MLXArray? {
        let hidden = h.dim(-1)
        let rows = h.size / hidden
        guard !training, let shortlistHead, let head = lmHead as? QuantizedLinear,
            head.bias == nil, head.globalScale == nil, rows <= Self.shortlistMaximumRows
        else { return nil }

        let x = h.reshaped(rows, hidden)
        let scores = shortlistHead(x)
        let count = Self.shortlistCandidates
        let candidates = Self.largest(count, of: scores)
        let flat = candidates.reshaped(-1)
        func candidateRows(_ tensor: MLXArray) -> MLXArray {
            tensor.take(flat, axis: 0).reshaped(rows, count, -1)
        }
        let exact = quantizedMM(
            x.expandedDimensions(axis: 1), candidateRows(head.weight),
            scales: candidateRows(head.scales), biases: head.biases.map(candidateRows),
            transpose: true, groupSize: head.groupSize, bits: head.bits, mode: head.mode
        ).reshaped(rows, count)
        let cap = sorted(exact, axis: -1)[0..., (count - Self.shortlistTailRank)...][0..., ..<1]
        let logits = putAlong(minimum(scores, cap), candidates, values: exact, axis: -1)
        return logits.reshaped(Array(h.shape.dropLast()) + [vocabularySize])
    }

    /// The indices of the `count` largest values of each row of `scores` `[N, V]`.
    ///
    /// The `count` 256-value chunks with the largest maxima hold every value of rank up to
    /// `count`, so the second partition reads `count × 256` values instead of `V`.
    static func largest(_ count: Int, of scores: MLXArray) -> MLXArray {
        let (rows, vocabulary) = (scores.dim(0), scores.dim(1))
        let chunks = scores.reshaped(rows, vocabulary / 256, 256)
        let top = argPartition(-chunks.max(axis: -1), kth: count - 1, axis: -1)[0..., ..<count]
        let values = takeAlong(chunks, top.expandedDimensions(axis: -1), axis: 1)
            .reshaped(rows, count * 256)
        let local = argPartition(-values, kth: count - 1, axis: -1)[0..., ..<count]
        return takeAlong(top, floorDivide(local, 256), axis: -1) * 256 + local % 256
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer in
            if layer.isRDA {
                return RewindableGatedDeltaCache(
                    rollbackCapacity: maximumNativeTargetCacheRewind + 1)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    public func prepare() throws {
        for layer in model.layers {
            layer.attentionMHC.prepare()
            layer.mlpMHC.prepare()
            (layer.attention as? Raon1RDA)?.prepare()
            (layer.attention as? Raon1MLA)?.prepare()
            layer.mlp?.prepare()
            layer.moe?.sharedExperts?.prepare()
        }
    }

    /// Attaches the shortlist head when the checkpoint ships a file that matches `lm_head`.
    /// Its tensors are unscoped: `weight`, `scales` and `biases`.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var weights = checkpoint.weights
        let names = ["weight", "scales", "biases"]
        let tensors = names.compactMap { weights.removeValue(forKey: $0) }
        let (vocabulary, hidden) = (configuration.vocabularySize, configuration.hiddenSize)
        if lmHead != nil, tensors.count == names.count,
            vocabulary % 256 == 0, vocabulary / 256 >= Self.shortlistCandidates,
            tensors[0].shape == [vocabulary, hidden / 16], tensors[0].dtype == .uint32,
            tensors[1].shape == [vocabulary, hidden / 64], tensors[2].shape == tensors[1].shape
        {
            let head = QuantizedLinear(
                weight: tensors[0], scales: tensors[1], biases: tensors[2], groupSize: 64,
                bits: 2, mode: .affine)
            try update(
                modules: ModuleChildren.unflattened(["lm_head_stage1": head]), verify: .none)
            for (name, tensor) in zip(names, tensors) {
                weights["lm_head_stage1.\(name)"] = tensor
            }
        }
        checkpoint.weights = try sanitize(weights: weights)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let config = configuration
        // Layers past the last are next-token prediction heads this model does not run.
        var weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.tieWordEmbeddings
        ).filter { key, _ in
            guard key.hasPrefix("model.layers."),
                let index = Int(key.dropFirst("model.layers.".count).prefix { $0 != "." })
            else { return true }
            return index < config.hiddenLayers
        }
        let heads = config.attentionHeads
        let nope = config.qkNopeHeadDim
        let steered = Set(config.effortSteering?.layers ?? [])

        for (index, layer) in model.layers.enumerated() {
            let prefix = "model.layers.\(index)"
            if layer.isRDA {
                // Transformers checkpoints keep one convolution per projection.
                let parts = ["q", "k", "v"].compactMap {
                    weights.removeValue(forKey: "\(prefix).self_attn.\($0)_conv1d.weight")
                }
                if parts.count == 3 {
                    weights["\(prefix).self_attn.qkv_conv1d.weight"] = concatenated(parts, axis: 0)
                }
                let key = "\(prefix).self_attn.qkv_conv1d.weight"
                if let w = weights[key], w.ndim == 2 {
                    weights[key] = w.expandedDimensions(axis: -1)
                }
            } else if let w = weights.removeValue(forKey: "\(prefix).self_attn.kv_b_proj.weight") {
                let w = w.reshaped(heads, nope + config.vHeadDim, config.kvLoraRank)
                weights["\(prefix).self_attn.embed_q.weight"] = w[0..., ..<nope].swappedAxes(-1, -2)
                weights["\(prefix).self_attn.unembed_out.weight"] = w[0..., nope...]
            }

            guard layer.moe != nil else { continue }
            let moe = "\(prefix).moe"
            if let w = weights["\(moe).gate.weight"] {
                weights["\(moe).gate.weight"] = w.asType(.float32)
            }
            // A checkpoint saved after the reference loader merged up and gate, up rows first.
            let experts = "\(moe).switch_mlp"
            for name in ["weight", "scales"] {
                if let merged = weights.removeValue(forKey: "\(experts).gate_up.\(name)") {
                    let halves = split(merged, parts: 2, axis: 1)
                    weights["\(experts).up_proj.\(name)"] = halves[0]
                    weights["\(experts).gate_proj.\(name)"] = halves[1]
                }
            }
            if let scale = weights.removeValue(forKey: "\(experts).gate_up.global_scale") {
                weights["\(experts).up_proj.global_scale"] = scale
                weights["\(experts).gate_proj.global_scale"] = scale
            }
            let vector = "\(moe).reasoning_effort_steering"
            if steered.contains(index) {
                weights[vector] =
                    weights[vector]?.asType(.float32)
                    ?? MLXArray.zeros([config.hiddenSize], dtype: .float32)
            } else {
                weights[vector] = nil
            }
        }
        return weights
    }
}

extension Raon1MiniModel: HiddenStateLanguageModel {
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache, effort: nil)
    }

    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead?(hiddenStates) ?? model.embedTokens.asLinear(hiddenStates)
    }
}

enum Raon1MiniError: LocalizedError {
    case unsupportedConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let message):
            "Raon-1 mini configuration: \(message)"
        }
    }
}

// MARK: - LoRA

extension Raon1MiniModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension Raon1MiniModel {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { Self.reasoning }

    /// `<think>` reasoning, closed at a budget the way the chat template closes it: `\n</think>\n\n`.
    ///
    /// Measured on `Raon-1-mini-mlx` over 20 prompts at budgets of 32, 64 and 128 tokens: forcing
    /// the template's text answered 60 of 60 cuts and never continued reasoning. Forcing
    /// `</think>` alone also answered every cut, but 4 of the 60 answers began with more
    /// reasoning and a second `</think>`.
    static let reasoning = QwenReasoningProtocol.tagged.declaringBudgetTransition(
        ReasoningBudgetTransition(beforeEndDelimiter: "\n", afterEndDelimiter: "\n\n"))
}
