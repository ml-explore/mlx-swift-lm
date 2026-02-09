//
//  Qwen3_5.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen3_5Configuration: Codable, Sendable {
    var modelType: String = "qwen3_5"
    var textConfig: Qwen3_5TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5"

        if let textConfig = try container.decodeIfPresent(
            Qwen3_5TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen3_5TextConfiguration(from: decoder)
        }
    }
}

public class Qwen3_5Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen3_5TextModel

    public init(_ args: Qwen3_5Configuration) {
        let textModel = Qwen3_5TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func makeCache() -> [KVCache] {
        languageModel.makeCache()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower.") || key.hasPrefix("visual.") {
                continue
            }
            let newKey = key.hasPrefix("language_model.") ? key : "language_model." + key
            sanitized[newKey] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen3_5Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
