// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Configuration of KRAFTON's DFlash2 drafter for Raon-1 mini, `dflash2/config.json`.
public struct Raon1DFlash2Configuration: Codable, Sendable {
    public struct DFlash: Codable, Sendable {
        let blockSize: Int
        let maskTokenID: Int
        let convolutionTaps: Int
        let convolutionGroupSize: Int
        let selectorRank: Int
        let selectorTopK: Int
        let window: Int
        let targetLayerIDs: [Int]
        let targetStreamsPerLayer: Int

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case maskTokenID = "mask_token_id"
            case convolutionTaps = "conv_kernel_size"
            case convolutionGroupSize = "conv_group_size"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case window = "swa_window_size"
            case targetLayerIDs = "target_layer_ids"
            case targetStreamsPerLayer = "target_streams_per_layer"
        }
    }

    public struct RoPE: Codable, Sendable {
        let theta: Float

        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
        }
    }

    let architectures: [String]
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDimension: Int
    let intermediateSize: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let targetLayerCount: Int
    let rope: RoPE
    let dflash: DFlash

    enum CodingKeys: String, CodingKey {
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case targetLayerCount = "num_target_layers"
        case rope = "rope_parameters"
        case dflash = "dflash_config"
    }

    var groups: Int { hiddenSize / dflash.convolutionGroupSize }
    var featureSize: Int { dflash.targetLayerIDs.count * dflash.targetStreamsPerLayer * hiddenSize }
}

extension Raon1DFlash2Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard architectures.contains("DFlash2DraftModel") else {
            throw ModelFactoryError.invalidConfiguration(
                "DFlash2 requires architecture 'DFlash2DraftModel'.")
        }
        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0, kvHeads > 0,
            intermediateSize > 0, vocabularySize > 0,
            attentionHeads.isMultiple(of: kvHeads), headDimension > 0,
            headDimension.isMultiple(of: 2), dflash.convolutionGroupSize > 0,
            hiddenSize.isMultiple(of: dflash.convolutionGroupSize),
            dflash.blockSize > 1, dflash.convolutionTaps > 0,
            dflash.selectorRank > 0, (1 ... vocabularySize).contains(dflash.selectorTopK),
            dflash.window > dflash.blockSize, dflash.targetStreamsPerLayer > 0,
            !dflash.targetLayerIDs.isEmpty,
            Set(dflash.targetLayerIDs).count == dflash.targetLayerIDs.count,
            (0 ..< vocabularySize).contains(dflash.maskTokenID),
            dflash.targetLayerIDs == dflash.targetLayerIDs.sorted(),
            dflash.targetLayerIDs.allSatisfy({ (0 ..< targetLayerCount).contains($0) })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "DFlash2 dimensions, block size, window and target layers must be consistent.")
        }
    }
}

/// The block-causal grouped convolution around attention and the MLP.
///
/// Each position mixes itself and the `taps - 1` positions before it in the proposal block, with
/// a kernel per channel group: a learned base plus a projection of the position's input.
final class Raon1DFlash2Convolution: Module {
    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    let taps: Int
    let groups: Int

    init(_ config: Raon1DFlash2Configuration) {
        taps = config.dflash.convolutionTaps
        groups = config.groups
        _baseKernel.wrappedValue = MLXArray.zeros([2, taps, config.hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            config.hiddenSize, 2 * taps * groups, bias: false)
    }

    /// The convolution before the branch, and the kernel deltas of the one after it.
    func enter(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (B, n) = (x.dim(0), x.dim(1))
        let deltas = kernelProjection(x).reshaped(B, n, 2, taps, groups)
        return (apply(x, deltas[0..., 0..., 0], baseKernel[0]), deltas[0..., 0..., 1])
    }

    /// The convolution after the branch.
    func exit(_ y: MLXArray, _ deltas: MLXArray) -> MLXArray {
        apply(y, deltas, baseKernel[1])
    }

    /// `x` `[B, n, H]` with deltas `[B, n, taps, groups]` and the base `[taps, H]`.
    private func apply(_ x: MLXArray, _ deltas: MLXArray, _ base: MLXArray) -> MLXArray {
        let (B, n) = (x.dim(0), x.dim(1))
        let blocks = x.reshaped(B, n, groups, -1)
        let kernel =
            base.reshaped(1, 1, taps, groups, -1) + deltas.expandedDimensions(axis: -1)
        var out = kernel[0..., 0..., 0] * blocks
        for tap in 1 ..< min(taps, n) {
            let shifted = padded(
                blocks[0..., ..<(n - tap)],
                widths: [.init((0, 0)), .init((tap, 0)), .init((0, 0)), .init((0, 0))])
            out = out + kernel[0..., 0..., tap] * shifted
        }
        return out.reshaped(B, n, -1)
    }
}

final class Raon1DFlash2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let heads: Int
    let kvHeads: Int
    let headDimension: Int
    let rope: RoPE

    init(_ config: Raon1DFlash2Configuration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDimension = config.headDimension
        let (hidden, d) = (config.hiddenSize, config.headDimension)
        _qProj.wrappedValue = Linear(hidden, heads * d, bias: false)
        _kProj.wrappedValue = Linear(hidden, kvHeads * d, bias: false)
        _vProj.wrappedValue = Linear(hidden, kvHeads * d, bias: false)
        _oProj.wrappedValue = Linear(heads * d, hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.rmsNormEps)
        rope = RoPE(dimensions: d, traditional: false, base: config.rope.theta)
    }

    /// Rotated keys and values `[B, kvHeads, n, d]` of `x` at positions from `position`.
    func keysAndValues(_ x: MLXArray, position: Int) -> (MLXArray, MLXArray) {
        let (B, n) = (x.dim(0), x.dim(1))
        let keys = kNorm(kProj(x).reshaped(B, n, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, n, kvHeads, -1).transposed(0, 2, 1, 3)
        return (rope(keys, offset: position), values)
    }

    /// Non-causal attention of the block over the target context and the block itself.
    func callAsFunction(_ x: MLXArray, position: Int, context: (MLXArray, MLXArray)?)
        -> MLXArray
    {
        let (B, n) = (x.dim(0), x.dim(1))
        let queries = rope(
            qNorm(qProj(x).reshaped(B, n, heads, -1)).transposed(0, 2, 1, 3), offset: position)
        var (keys, values) = keysAndValues(x, position: position)
        if let context {
            keys = concatenated([context.0, keys], axis: 2)
            values = concatenated([context.1, values], axis: 2)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / Float(headDimension).squareRoot(), mask: .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, n, -1))
    }
}

final class Raon1DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Raon1DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Raon1DFlash2Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: Raon1DFlash2Attention
    @ModuleInfo var mlp: Raon1DFlash2MLP
    @ModuleInfo(key: "attention_conv") var attentionConvolution: Raon1DFlash2Convolution
    @ModuleInfo(key: "mlp_conv") var mlpConvolution: Raon1DFlash2Convolution
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Raon1DFlash2Configuration) {
        _attention.wrappedValue = Raon1DFlash2Attention(config)
        _mlp.wrappedValue = Raon1DFlash2MLP(config)
        _attentionConvolution.wrappedValue = Raon1DFlash2Convolution(config)
        _mlpConvolution.wrappedValue = Raon1DFlash2Convolution(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// One layer on the pre-norm residual stream: returns the branch output and the residual.
    func callAsFunction(
        _ x: MLXArray, residual: MLXArray?, position: Int, context: (MLXArray, MLXArray)?
    ) -> (MLXArray, MLXArray) {
        let residual = residual.map { x + $0 } ?? x
        let (attentionInput, attentionDeltas) = attentionConvolution.enter(
            inputLayerNorm(residual))
        let attended = attentionConvolution.exit(
            attention(attentionInput, position: position, context: context), attentionDeltas)
        let next = attended + residual
        let (mlpInput, mlpDeltas) = mlpConvolution.enter(postAttentionLayerNorm(next))
        return (mlpConvolution.exit(mlp(mlpInput), mlpDeltas), next)
    }
}

/// Scores pairs of consecutive candidates, so a proposal is a chain rather than independent picks.
final class Raon1DFlash2Selector: Module {
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear
    @ParameterInfo(key: "predecessor_codebook") var predecessorCodebook: MLXArray
    @ParameterInfo(key: "successor_codebook") var successorCodebook: MLXArray

    init(_ config: Raon1DFlash2Configuration) {
        let rank = config.dflash.selectorRank
        _hiddenProjection.wrappedValue = Linear(config.hiddenSize, rank, bias: false)
        _predecessorCodebook.wrappedValue = MLXArray.zeros([config.vocabularySize, rank])
        _successorCodebook.wrappedValue = MLXArray.zeros([config.vocabularySize, rank])
    }

    /// The scores `[B, n, K, K]` of the candidate `c` after the predecessor `p` at each step:
    /// the candidate's logit plus a low-rank pair term.
    func scores(hidden: MLXArray, candidates: MLXArray, logits: MLXArray, bonus: MLXArray)
        -> MLXArray
    {
        let (B, K) = (candidates.dim(0), candidates.dim(2))
        let predecessors = concatenated(
            [
                broadcast(bonus.reshaped(B, 1, 1).asType(candidates.dtype), to: [B, 1, K]),
                candidates[0..., ..<(candidates.dim(1) - 1)],
            ], axis: 1)
        let pair =
            (predecessorCodebook[predecessors]
            * hiddenProjection(hidden).expandedDimensions(axis: 2))
            .asType(.float32)
        let successors = successorCodebook[candidates].asType(.float32)
        return takeAlong(logits, candidates, axis: -1).expandedDimensions(axis: 2)
            + matmul(pair, successors.swappedAxes(-1, -2))
    }
}

/// KRAFTON's DFlash2 drafter for Raon-1 mini.
///
/// One non-causal forward over the bonus token and mask tokens proposes a whole block. Five
/// sliding-window layers attend to features of the target's mHC streams at five layers and to
/// the block. The drafter's own NVFP4 copy of the output head shortlists candidates at each step,
/// and a selector chains them. The drafter reuses the target's token embeddings.
public final class Raon1DFlash2DraftModel: Module, ResumableMTPDrafterModel {
    public let configuration: Raon1DFlash2Configuration
    public var maximumBlockSize: Int? { configuration.dflash.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { configuration.dflash.targetLayerIDs }
    public var targetContextLength: Int? { configuration.dflash.window }

    /// A proposal stops where the product of the selector's step confidences falls below this.
    static let confidenceCut: Float = 0.2

    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo var layers: [Raon1DFlash2Layer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var selector: Raon1DFlash2Selector
    @ModuleInfo var head: Linear

    public init(_ configuration: Raon1DFlash2Configuration) {
        self.configuration = configuration
        let hidden = configuration.hiddenSize
        _fc.wrappedValue = Linear(configuration.featureSize, hidden, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: configuration.rmsNormEps)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            Raon1DFlash2Layer(configuration)
        }
        _norm.wrappedValue = RMSNorm(dimensions: hidden, eps: configuration.rmsNormEps)
        _selector.wrappedValue = Raon1DFlash2Selector(configuration)
        _head.wrappedValue = Linear(hidden, configuration.vocabularySize, bias: false)
        super.init()
    }

    // MARK: Drafter state

    /// Per layer, the rotated keys `[B, kvHeads, T, d]` and values of the last `window` target
    /// positions, in slots 0 and 1. `nextPosition` is the position after them.
    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in MambaCache() })
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        validateTarget(target)
        state.nextPosition = promptTokens.dim(-1) - targetHidden.dim(1)
        appendContext(targetHidden, state: &state)
    }

    public func commitDrafterState(
        target _: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        appendContext(targetHidden[0..., ..<(acceptedCount + 1)], state: &state)
    }

    public func finalizeDrafterState(
        target _: any LanguageModel, targetBoundaryHidden _: MLXArray,
        targetProcessedTokenCount: Int, discardedTargetTokens: Int,
        positionDeltas _: MLXArray?, state: inout MTPDrafterState
    ) -> Bool {
        guard state.proposalAppended == 0,
            state.nextPosition == targetProcessedTokenCount + discardedTargetTokens
        else { return false }
        if discardedTargetTokens > 0 {
            for cache in state.cache.map(Self.slots) {
                guard let keys = cache[0], let values = cache[1] else { continue }
                guard keys.dim(2) < configuration.dflash.window else { return false }
                let kept = max(0, keys.dim(2) - discardedTargetTokens)
                cache[0] = keys[0..., 0..., ..<kept]
                cache[1] = values[0..., 0..., ..<kept]
            }
            state.nextPosition -= discardedTargetTokens
        }
        return true
    }

    public func resumeDrafterState(
        target: any LanguageModel, suffixTokens: MLXArray, suffixTargetHidden: MLXArray,
        targetBoundaryHidden _: MLXArray, firstBonus _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) -> Bool {
        guard isCompatible(with: target) else { return false }
        let skipped = suffixTokens.dim(-1) - suffixTargetHidden.dim(1)
        if skipped > 0 {
            // A suffix longer than the window replaces the context it pushes out.
            state.cache = makeState(parameters: nil).cache
            state.nextPosition += skipped
        }
        appendContext(suffixTargetHidden, state: &state)
        return true
    }

    /// A layer's context: rotated keys in slot 0, values in slot 1.
    private static func slots(_ cache: KVCache) -> ArraysCache {
        guard let slots = cache as? ArraysCache else {
            preconditionFailure("DFlash2 drafter state must hold its own context caches")
        }
        return slots
    }

    /// Appends the target features `[B, n, F]` of the next `n` positions, keeping `window`.
    private func appendContext(_ features: MLXArray, state: inout MTPDrafterState) {
        let window = configuration.dflash.window
        let n = features.dim(1)
        let skipped = max(0, n - window)
        let projected = hiddenNorm(fc(features[0..., skipped...]))
        let position = state.nextPosition + skipped
        for (layer, cache) in zip(layers, state.cache.map(Self.slots)) {
            let (keys, values) = layer.attention.keysAndValues(projected, position: position)
            let joinedKeys =
                skipped > 0 ? keys : cache[0].map { concatenated([$0, keys], axis: 2) } ?? keys
            let joinedValues =
                skipped > 0
                ? values : cache[1].map { concatenated([$0, values], axis: 2) } ?? values
            let start = max(0, joinedKeys.dim(2) - window)
            cache[0] = joinedKeys[0..., 0..., start...]
            cache[1] = joinedValues[0..., 0..., start...]
        }
        state.nextPosition += n
    }

    // MARK: Drafting

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        preconditionFailure("Raon1DFlash2DraftModel drafts from its iterator-owned state")
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        let target = validateTarget(target)
        let n = min(blockSize, configuration.dflash.blockSize)
        let bonus = normalizedMTPColumn(lastToken).asType(.int32)
        let tokens = concatenated(
            [
                bonus,
                MLXArray.full(
                    [bonus.dim(0), n - 1], values: MLXArray(Int32(configuration.dflash.maskTokenID))
                ),
            ],
            axis: 1)

        // The block attends to the last `window - n` context positions and to itself.
        let keep = configuration.dflash.window - n
        var x = target.model.embedTokens(tokens)
        var residual: MLXArray?
        for (layer, cache) in zip(layers, state.cache.map(Self.slots)) {
            let context = cache[0].flatMap { keys in
                cache[1].map { values in
                    let start = max(0, keys.dim(2) - keep)
                    return (keys[0..., 0..., start...], values[0..., 0..., start...])
                }
            }
            (x, residual) = layer(
                x, residual: residual, position: state.nextPosition, context: context)
        }
        let hidden = norm(x + (residual ?? MLXArray.zeros(like: x)))[0..., 1...]

        let logits = head(hidden).asType(.float32)
        let topK = configuration.dflash.selectorTopK
        let candidates = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let scores = selector.scores(
            hidden: hidden, candidates: candidates, logits: logits, bonus: bonus)
        if n == 2 {
            // The first draft is always kept. Do not synchronize to read its confidence.
            let pick = argMax(scores[0..., 0, 0], axis: -1, keepDims: true)
            return takeAlong(candidates[0..., 0], pick, axis: -1).asType(lastToken.dtype)
        }
        let (proposal, confidences) = chain(scores, candidates)

        // The cut reads the confidences, so the proposal is evaluated here.
        eval(proposal, confidences)
        var kept = 0
        var probability: Float = 1
        for confidence in confidences.asArray(Float.self) {
            probability *= confidence
            if probability < Self.confidenceCut { break }
            kept += 1
        }
        return proposal[0..., ..<max(1, kept)].asType(lastToken.dtype)
    }

    /// Follows the best successor from step to step: the proposal `[B, n]` and each step's
    /// softmax probability of its pick, for the first row.
    private func chain(_ scores: MLXArray, _ candidates: MLXArray) -> (MLXArray, MLXArray) {
        let (B, steps) = (scores.dim(0), scores.dim(1))
        var previous = MLXArray.zeros([B, 1], dtype: .int32)
        var tokens = [MLXArray]()
        var confidences = [MLXArray]()
        for step in 0 ..< steps {
            let row = takeAlong(
                scores[0..., step], previous.reshaped(B, 1, 1), axis: 1
            ).squeezed(axis: 1)
            let pick = argMax(row, axis: -1, keepDims: true).asType(.int32)
            tokens.append(takeAlong(candidates[0..., step], pick, axis: -1))
            confidences.append(
                takeAlong(softmax(row, axis: -1, precise: true), pick, axis: -1)[0])
            previous = pick
        }
        return (concatenated(tokens, axis: 1), concatenated(confidences))
    }

    // MARK: Target

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard isCompatible(with: target) else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Raon1DFlash2DraftModel",
                expected: "a Raon-1 mini target with matching dimensions and mHC streams",
                actual: String(describing: type(of: target)))
        }
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        guard let target = target as? Raon1MiniModel else { return false }
        let config = target.configuration
        return config.hiddenSize == configuration.hiddenSize
            && config.vocabularySize == configuration.vocabularySize
            && config.hiddenLayers == configuration.targetLayerCount
            && config.mhcStreams == configuration.dflash.targetStreamsPerLayer
    }

    @discardableResult
    private func validateTarget(_ target: any LanguageModel) -> Raon1MiniModel {
        guard isCompatible(with: target), let target = target as? Raon1MiniModel else {
            preconditionFailure("Raon-1 mini DFlash2 draft and target are incompatible")
        }
        return target
    }

    // MARK: Checkpoint

    /// KRAFTON stores plain tensors as `w.<name>`, NVFP4 linears as `q.<name>.w` and `.s` with
    /// their format in the metadata `fmt.q.<name>`, and the head as `head.w` and `head.s`.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var weights = [String: MLXArray]()
        var layers = [String: BaseConfiguration.QuantizationOption]()
        func format(_ key: String) throws -> BaseConfiguration.Quantization? {
            guard let json = checkpoint.metadata[key] else { return nil }
            return try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: Data(json.utf8))
        }
        for (key, value) in checkpoint.weights {
            if key.hasPrefix("w.") {
                weights[String(key.dropFirst(2))] = value
            } else if key.hasPrefix("q."), let suffix = key.split(separator: ".").last,
                ["w", "s", "b"].contains(suffix)
            {
                let tensor = String(key.dropFirst(2).dropLast(2))
                let path = String(tensor.dropLast(".weight".count))
                let name = suffix == "w" ? "weight" : suffix == "s" ? "scales" : "biases"
                weights["\(path).\(name)"] = value
                if let quantization = try format("fmt.q.\(tensor)") {
                    layers[path] = .quantize(quantization)
                }
            } else if key == "head.w" || key == "head.s" {
                weights[key == "head.w" ? "head.weight" : "head.scales"] = value
                if let quantization = try format("fmt.head") {
                    layers["head"] = .quantize(quantization)
                }
            } else {
                weights[key] = value
            }
        }
        checkpoint.weights = weights
        if !layers.isEmpty {
            checkpoint.perLayerQuantization = .init(perLayerQuantization: layers)
        }
        return checkpoint
    }
}

/// Registers the DFlash2 drafter that Raon-1 mini checkpoints carry in `dflash2/`.
public enum Raon1DFlash2Registration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3",
            matches: { data in
                guard
                    let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let architectures = object["architectures"] as? [String]
                else { return false }
                return architectures.contains("DFlash2DraftModel")
            },
            creator: { data in
                let configuration = try JSONDecoder.json5().decode(
                    Raon1DFlash2Configuration.self, from: data)
                try configuration.validateModelConfiguration()
                return Raon1DFlash2DraftModel(configuration)
            })
    }
}
