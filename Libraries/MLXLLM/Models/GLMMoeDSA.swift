//
//  GLMMoeDSA.swift
//  LLM
//
//  Created for GLM-5.1 (glm_moe_dsa) support.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct GLMMoeDSAConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    
    var indexerNHeads: Int?
    var indexerHeadDim: Int?
    var indexerTopk: Int?
    var indexerRopeInterleave: Bool?
    
    var topkMethod: String?
    var scoringFunc: String?
    var normTopkProb: Bool?
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int?
    var firstKDenseReplace: Int?
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float?
    var ropeScaling: [String: StringOrNumber]?
    var ropeTraditional: Bool?
    var attentionBias: Bool?
    var attentionDropout: Float?
    var partialRotaryFactor: Float?
    var tieWordEmbeddings: Bool?
    var numNextnPredictLayers: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        
        case indexerNHeads = "indexer_n_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerTopk = "indexer_topk"
        case indexerRopeInterleave = "indexer_rope_interleave"
        
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Defaults + Extraction
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        self.kvLoraRank = try container.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        self.qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        self.qkNopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        self.vHeadDim = try container.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        
        self.indexerNHeads = try container.decodeIfPresent(Int.self, forKey: .indexerNHeads)
        self.indexerHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexerHeadDim)
        self.indexerTopk = try container.decodeIfPresent(Int.self, forKey: .indexerTopk)
        self.indexerRopeInterleave = try container.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave)
        
        self.topkMethod = try container.decodeIfPresent(String.self, forKey: .topkMethod)
        self.scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc)
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup)
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq)
        self.firstKDenseReplace = try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace)
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout)
        self.partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
        self.numNextnPredictLayers = try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers)
    }
}
