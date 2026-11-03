//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py
//
// The configuration and backbone are in MLXLMCommon (`Models/LFM3.swift`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == .conv ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(inputs, cache: cache))
    }

    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let hidden = hiddenStates(input.tokens, cache: cache)
        let projection: Module = lmHead ?? model.embedTokens
        return .init(
            logits: projectLogits(
                quantizedVocabularyProjectionInput(hidden, projection: projection)))
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // MLX conversions nest the text model under `language_model`. Keep only the text
        // model, which drops any vision or audio tensors. The same mapping renames the
        // per-layer quantization settings, including the bare `lm_head` module path.
        var checkpoint = try checkpoint.mapNames { name in
            var name = name
            if name.hasPrefix("language_model.") {
                name.removeFirst("language_model.".count)
            }
            let root = name.prefix { $0 != "." }
            return root == "model" || root == "lm_head" ? name : nil
        }
        var weights = filterLMHeadWeights(
            from: checkpoint.weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        // PyTorch stores the depthwise conv as [C, 1, K]; MLX expects [C, K, 1].
        for (name, value) in weights where name.hasSuffix("conv.conv.weight") {
            if value.dim(-1) > value.dim(1) {
                weights[name] = value.transposed(0, 2, 1)
            }
        }
        checkpoint.weights = weights
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.newCache(parameters: parameters)
    }
}

extension LFM3Model: HiddenStateLanguageModel {
    /// Final normed hidden states `[B, L, hidden]` for token ids `[B, L]`.
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    /// Vocabulary logits for final hidden states, tied or untied.
    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        model.projectLogits(hiddenStates, lmHead: lmHead)
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension LFM3Model {
    public var reasoningConfig: ReasoningConfig? { .alwaysOnThinking }
}
