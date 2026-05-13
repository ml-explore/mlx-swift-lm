//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for the `"gemma4"` model_type.
/// This is a thin wrapper around `Gemma4TextConfiguration` that handles the
/// nested `text_config` structure from HuggingFace model configs.
public struct Gemma4Configuration: Codable, Sendable {
    var modelType: String = "gemma4"
    var textConfig: Gemma4TextConfiguration
    var vocabSize: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144

        // If text_config is present, decode from it; otherwise treat entire config as text config
        if let textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
            // Propagate vocab_size into text config
            self.textConfig.vocabSize = self.vocabSize
        } else {
            self.textConfig = try Gemma4TextConfiguration(from: decoder)
        }
    }
}

// MARK: - Model

public class Gemma4Model: Module, LLMModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    @ModuleInfo(key: "language_model") fileprivate var languageModel: Gemma4TextModel

    public init(_ config: Gemma4Configuration) {
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Strip "model." prefix
            let startsWithModel = k.hasPrefix("model.")
            k = k.replacingOccurrences(of: "model.", with: "", options: .anchored)

            // Skip vision/audio weights
            if k.hasPrefix("vision_tower") || k.hasPrefix("multi_modal_projector")
                || k.hasPrefix("audio_tower") || k.hasPrefix("embed_audio")
                || k.hasPrefix("embed_vision")
            {
                continue
            }

            if !startsWithModel {
                sanitized[k] = value
                continue
            }

            // Remap language_model keys
            if k.hasPrefix("language_model") {
                k = k.replacingOccurrences(
                    of: "language_model.", with: "language_model.model.", options: .anchored)
            }

            sanitized[k] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

// MARK: - LoRA

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}
