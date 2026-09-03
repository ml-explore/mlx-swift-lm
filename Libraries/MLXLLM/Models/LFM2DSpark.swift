// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Configuration for Liquid AI's LFM2.5 DSpark draft checkpoints.
public struct LFM2DSparkConfiguration: Codable, Sendable {
    public struct DFlashConfiguration: Codable, Sendable {
        let maskTokenID: Int
        let targetLayerIDs: [Int]
        let targetLayerCount: Int

        enum CodingKeys: String, CodingKey {
            case maskTokenID = "mask_token_id"
            case targetLayerIDs = "target_layer_ids"
            case targetLayerCount = "num_target_layers"
        }
    }

    let architectures: [String]
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDimension: Int
    let intermediateSize: Int
    let hiddenActivation: String
    let rmsNormEps: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let maxPositionEmbeddings: Int
    let layerTypes: [String]
    let blockSize: Int
    let dflash: DFlashConfiguration
    let markovRank: Int
    let ropeIsNeoxStyle: Bool
    let enableConfidenceHead: Bool
    let markovHeadType: String

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case intermediateSize = "intermediate_size"
        case hiddenActivation = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case blockSize = "block_size"
        case dflash = "dflash_config"
        case markovRank = "markov_rank"
        case ropeIsNeoxStyle = "rope_is_neox_style"
        case enableConfidenceHead = "enable_confidence_head"
        case markovHeadType = "markov_head_type"
    }
}

extension LFM2DSparkConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard modelType == "qwen3",
            architectures.contains("Lfm2DSparkDraftModel")
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM2 DSpark requires architecture 'Lfm2DSparkDraftModel' and model_type 'qwen3'."
            )
        }
        guard hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0,
            vocabularySize > 0, blockSize > 0, markovRank > 0,
            headDimension > 0, attentionHeads > 0, kvHeads > 0,
            attentionHeads.isMultiple(of: kvHeads),
            attentionHeads * headDimension == hiddenSize,
            rmsNormEps > 0, ropeTheta > 0, maxPositionEmbeddings > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM2 DSpark dimensions, block size, Markov rank, and RoPE parameters must be positive and internally consistent."
            )
        }
        guard hiddenActivation == "silu", markovHeadType == "vanilla",
            enableConfidenceHead
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Released LFM2 DSpark checkpoints require SiLU, the vanilla Markov head, and the confidence head."
            )
        }
        guard !ropeIsNeoxStyle else {
            throw ModelFactoryError.invalidConfiguration(
                "Released LFM2 DSpark checkpoints use interleaved RoPE.")
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "full_attention" }),
            dflash.targetLayerIDs.count == hiddenLayers,
            dflash.targetLayerIDs == dflash.targetLayerIDs.sorted(),
            Set(dflash.targetLayerIDs).count == dflash.targetLayerIDs.count,
            dflash.targetLayerIDs.allSatisfy({ (0 ..< dflash.targetLayerCount).contains($0) }),
            (0 ..< vocabularySize).contains(dflash.maskTokenID)
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM2 DSpark requires one ordered, unique target feature tap per attention-only draft layer and an in-vocabulary mask token."
            )
        }
    }
}

private final class LFM2DSparkAttention: Module {
    private let configuration: LFM2DSparkConfiguration
    private let scale: Float

    @ModuleInfo(key: "q_proj") var qProjection: Linear
    @ModuleInfo(key: "k_proj") var kProjection: Linear
    @ModuleInfo(key: "v_proj") var vProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    private let rope: RoPE

    init(_ configuration: LFM2DSparkConfiguration) {
        self.configuration = configuration
        self.scale = pow(Float(configuration.headDimension), -0.5)
        _qProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDimension,
            bias: false)
        _kProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.kvHeads * configuration.headDimension,
            bias: false)
        _vProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.kvHeads * configuration.headDimension,
            bias: false)
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDimension,
            configuration.hiddenSize,
            bias: false)
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEps)

        // Hugging Face's rope_is_neox_style=false rotates adjacent pairs.
        self.rope = RoPE(
            dimensions: configuration.headDimension,
            traditional: true,
            base: configuration.ropeTheta)
    }

    func appendContext(_ hidden: MLXArray, cache: KVCache) {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        var keys = kProjection(hidden)
        var values = vProjection(hidden)
        keys = keyNorm(
            keys.reshaped(batch, length, configuration.kvHeads, -1)
        ).transposed(0, 2, 1, 3)
        values = values.reshaped(
            batch, length, configuration.kvHeads, -1
        ).transposed(0, 2, 1, 3)
        keys = rope(keys, offset: cache.offset)
        _ = cache.update(keys: keys, values: values)
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        var queries = qProjection(hidden)
        var keys = kProjection(hidden)
        var values = vProjection(hidden)

        queries = queryNorm(
            queries.reshaped(batch, length, configuration.attentionHeads, -1)
        ).transposed(0, 2, 1, 3)
        keys = keyNorm(
            keys.reshaped(batch, length, configuration.kvHeads, -1)
        ).transposed(0, 2, 1, 3)
        values = values.reshaped(
            batch, length, configuration.kvHeads, -1
        ).transposed(0, 2, 1, 3)

        let offset = cache.offset
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // DSpark's parallel backbone is deliberately non-causal inside the
        // proposal block: every mask position attends the complete block and
        // all committed target-context features.
        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1))
    }
}

private final class LFM2DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ configuration: LFM2DSparkConfiguration) {
        _gate.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _up.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _down.wrappedValue = Linear(
            configuration.intermediateSize, configuration.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class LFM2DSparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: LFM2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ configuration: LFM2DSparkConfiguration) {
        _attention.wrappedValue = LFM2DSparkAttention(configuration)
        _mlp.wrappedValue = LFM2DSparkMLP(configuration)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    func appendContext(_ hidden: MLXArray, cache: KVCache) {
        attention.appendContext(hidden, cache: cache)
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache) -> MLXArray {
        let attended = hidden + attention(inputNorm(hidden), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

private final class LFM2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousTokenEmbedding: Embedding
    @ModuleInfo(key: "markov_w2") var vocabularyProjection: Linear

    init(_ configuration: LFM2DSparkConfiguration) {
        _previousTokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.markovRank)
        _vocabularyProjection.wrappedValue = Linear(
            configuration.markovRank, configuration.vocabularySize, bias: false)
    }

    func embedding(_ tokens: MLXArray) -> MLXArray {
        previousTokenEmbedding(tokens)
    }

    func bias(_ tokens: MLXArray) -> MLXArray {
        vocabularyProjection(embedding(tokens))
    }
}

private final class LFM2DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(_ configuration: LFM2DSparkConfiguration) {
        _projection.wrappedValue = Linear(
            configuration.hiddenSize + configuration.markovRank, 1, bias: true)
    }

    func callAsFunction(hidden: MLXArray, markovEmbedding: MLXArray) -> MLXArray {
        projection(concatenated([hidden, markovEmbedding], axis: -1)).squeezed(axis: -1)
    }
}

/// Liquid AI's attention-only DSpark drafter for LFM2.5 dense and MoE targets.
///
/// The model owns only its five-layer backbone, target-feature projection,
/// Markov head, and confidence head. Token embeddings and the vocabulary head
/// are tied to the target checkpoint exactly as in the reference runtime.
public final class LFM2DSparkDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: LFM2DSparkConfiguration
    public var maximumBlockSize: Int? { configuration.blockSize + 1 }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { configuration.dflash.targetLayerIDs }

    @ModuleInfo(key: "layers") fileprivate var layers: [LFM2DSparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "fc") var targetProjection: Linear
    @ModuleInfo(key: "hidden_norm") var targetNorm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: LFM2DSparkMarkovHead
    @ModuleInfo(key: "confidence_head") fileprivate var confidenceHead: LFM2DSparkConfidenceHead

    public init(_ configuration: LFM2DSparkConfiguration) {
        self.configuration = configuration
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            LFM2DSparkDecoderLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _targetProjection.wrappedValue = Linear(
            configuration.hiddenSize * configuration.dflash.targetLayerIDs.count,
            configuration.hiddenSize,
            bias: false)
        _targetNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _markovHead.wrappedValue = LFM2DSparkMarkovHead(configuration)
        _confidenceHead.wrappedValue = LFM2DSparkConfidenceHead(configuration)
        super.init()
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in KVCacheSimple() })
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        validateTarget(target)
        precondition(
            targetHidden.dim(-1)
                == configuration.hiddenSize * configuration.dflash.targetLayerIDs.count,
            "LFM2 DSpark target-feature width does not match dflash_config.target_layer_ids")
        appendTargetContext(targetHidden, state: &state)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: MLXArray.zeros([1, 1, 1]),
            sharedKV: [:],
            positionDeltas: nil,
            queryOffset: 0,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        validateTarget(target)
        let proposalLength = blockSize - 1
        precondition(
            (1 ... configuration.blockSize).contains(proposalLength),
            "LFM2 DSpark proposal length exceeds the checkpoint block size")
        precondition(state.cache.count == layers.count, "LFM2 DSpark cache/layer mismatch")

        let anchor = normalizedMTPColumn(lastToken)
        let draftInput: MLXArray
        if proposalLength == 1 {
            draftInput = anchor
        } else {
            let masks = MLXArray.full(
                [anchor.dim(0), proposalLength - 1],
                values: MLXArray(Int32(configuration.dflash.maskTokenID)),
                dtype: .int32)
            draftInput = concatenated([anchor.asType(.int32), masks], axis: 1)
        }

        let targetEmbedding = targetEmbedding(target)
        var hidden = targetEmbedding(draftInput)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: state.cache[index])
        }
        hidden = norm(hidden)

        // The proposal block is transient. Only projected target features are
        // retained in the drafter cache between rounds.
        let trimmed = trimPromptCache(state.cache, numTokens: proposalLength)
        precondition(trimmed == proposalLength, "LFM2 DSpark failed to discard proposal cache")

        let baseLogits = targetEmbedding.asLinear(hidden)
        var previous = anchor
        var proposals = [MLXArray]()
        proposals.reserveCapacity(proposalLength)
        for index in 0 ..< proposalLength {
            let stepLogits =
                baseLogits[0..., index, 0...] + markovHead.bias(previous)[0..., -1, 0...]
            let next = normalizedMTPColumn(sampler.sample(logits: stepLogits))
            proposals.append(next)
            previous = next
        }
        return concatenated(proposals, axis: 1)
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens _: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        validateTarget(target)
        let committedCount = acceptedCount + 1
        precondition(
            targetHidden.dim(1) >= committedCount,
            "LFM2 target did not emit every committed DSpark context feature")
        appendTargetContext(
            targetHidden[0..., ..<committedCount, 0...], state: &state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    private func appendTargetContext(
        _ targetHidden: MLXArray, state: inout MTPDrafterState
    ) {
        let projected = targetNorm(targetProjection(targetHidden))
        for (index, layer) in layers.enumerated() {
            layer.appendContext(projected, cache: state.cache[index])
        }
        state.nextPosition += targetHidden.dim(1)
    }

    private func targetEmbedding(_ target: any LanguageModel) -> Embedding {
        if let target = target as? LFM2Model {
            return target.model.embedTokens
        }
        if let target = target as? LFM2MoEModel {
            return target.model.embedTokens
        }
        preconditionFailure(
            "LFM2DSparkDraftModel requires an LFM2 or LFM2 MoE target, got \(type(of: target))"
        )
    }

    private func validateTarget(_ target: any LanguageModel) {
        let targetLayerCount: Int
        let targetHiddenSize: Int
        let targetVocabularySize: Int
        let targetRopeTheta: Float
        if let target = target as? LFM2Model {
            targetLayerCount = target.configuration.hiddenLayers
            targetHiddenSize = target.configuration.hiddenSize
            targetVocabularySize = target.configuration.vocabularySize
            targetRopeTheta = target.configuration.ropeTheta
        } else if let target = target as? LFM2MoEModel {
            targetLayerCount = target.configuration.hiddenLayers
            targetHiddenSize = target.configuration.hiddenSize
            targetVocabularySize = target.configuration.vocabularySize
            targetRopeTheta = target.configuration.ropeTheta
        } else {
            preconditionFailure(
                "LFM2DSparkDraftModel requires an LFM2 or LFM2 MoE target, got \(type(of: target))"
            )
        }
        precondition(
            targetVocabularySize == configuration.vocabularySize
                && targetHiddenSize == configuration.hiddenSize,
            "LFM2 DSpark target embedding does not match the draft checkpoint")
        precondition(
            targetLayerCount == configuration.dflash.targetLayerCount,
            "LFM2 DSpark draft checkpoint targets \(configuration.dflash.targetLayerCount) layers, but the verifier has \(targetLayerCount)"
        )
        precondition(
            targetRopeTheta == configuration.ropeTheta,
            "LFM2 DSpark draft and target checkpoints use different RoPE bases")
    }
}
