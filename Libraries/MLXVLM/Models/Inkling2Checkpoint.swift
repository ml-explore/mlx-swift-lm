// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// The layouts of Inkling 2 checkpoints.
///
/// Thinking Machines' own checkpoints keep the text model under `model.llm` and the vision
/// tower under `model.visual`, with their own names. Their routed experts are NVFP4 or MXFP8
/// with separate global-scale factors. A converted checkpoint passes through unchanged.
enum Inkling2Checkpoint {
    /// The vision tower, in either layout.
    static let components: [ModelComponent: [CheckpointComponent]] = [
        .vision: [
            CheckpointComponent(
                name: "vision_tower",
                namespaces: ["vision_tower", "model.vision_tower", "model.visual"],
                destination: "vision_tower")
        ]
    ]

    /// The vocabulary head in Thinking Machines' layout.
    static let originalHead = "model.llm.unembed"

    /// Every namespace of the model except its vocabulary head, in either layout.
    static let namespacesBesideHead = [
        "model.language_model", "model.vision_tower", "vision_tower", "model.visual",
        "model.llm.layers", "model.llm.attn_res", "model.llm.embed", "model.llm.embed_norm",
        "model.llm.norm", "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    ]

    private static let originalNamespaces = ["model.llm.", "model.visual."]

    /// The names of Thinking Machines' layout, in the reference's order, then the experts.
    private static let names = CheckpointNameMapping([
        .replaceMatches(#"^model\.llm\.unembed\."#, with: "lm_head."),
        .replaceMatches(#"^model\.llm\.embed\."#, with: "model.language_model.embed_tokens."),
        .replaceMatches(#"^model\.llm\."#, with: "model.language_model."),
        .replaceMatches(#"\.attn_res\.mlp_queries\.(\d+)$"#, with: ".layers.$1.mlp_input.query"),
        .replaceMatches(#"\.attn_res\.output_query$"#, with: ".final_input.query"),
        .replaceMatches(#"\.attn\.wq_du\."#, with: ".self_attn.q_proj."),
        .replaceMatches(#"\.attn\.wk_dv\."#, with: ".self_attn.k_proj."),
        .replaceMatches(#"\.attn\.wv_dv\."#, with: ".self_attn.v_proj."),
        .replaceMatches(#"\.attn\.wr_du\."#, with: ".self_attn.r_proj."),
        .replaceMatches(#"\.attn\.wo_ud\."#, with: ".self_attn.o_proj."),
        .replaceMatches(#"\.attn\.gate_attn\."#, with: ".self_attn.gate_proj."),
        .replaceMatches(#"\.attn\.attn_sink_proj\."#, with: ".self_attn.sink_proj."),
        .replaceMatches(#"\.controllers\.attn\."#, with: ".q_message_gate."),
        .replaceMatches(#"\.controllers\.(k|v|mlp)\."#, with: ".$1_message_gate."),
        .replaceMatches(#"_message_gate\.down_weight$"#, with: "_message_gate.down_proj.weight"),
        .replaceMatches(
            #"_message_gate\.forget_weight$"#, with: "_message_gate.forget_gate_proj.weight"),
        .replaceMatches(
            #"_message_gate\.gain_weight$"#, with: "_message_gate.message_gate_proj.weight"),
        .replaceMatches(#"\.attn\."#, with: ".self_attn."),
        .replaceMatches(#"\.attn_sconv\."#, with: ".post_attention_convolution."),
        .replaceMatches(#"\.mlp_sconv\."#, with: ".post_mlp_convolution."),
        .replaceMatches(#"sconv\.weight$"#, with: "sconv.conv1d.weight"),
        .replaceMatches(#"convolution\.weight$"#, with: "convolution.conv1d.weight"),
        .replaceMatches(#"sconv\.static_w$"#, with: "sconv.static_weight"),
        .replaceMatches(#"sconv\.weight_proj$"#, with: "sconv.weight_proj.weight"),
        .replaceMatches(#"\.mlp_norm\."#, with: ".post_attention_layernorm."),
        .replaceMatches(#"\.attn_norm\."#, with: ".input_layernorm."),
        .replaceMatches(#"\.(q|k|v)_input_norm\."#, with: ".$1_input_layernorm."),
        .replaceMatches(#"\.mlp\.gate\.bias$"#, with: ".mlp.gate.e_score_correction_bias"),
        .replaceMatches(
            #"^model\.visual\.backbone\.ln_pre\."#, with: "model.vision_tower.pre_layernorm."),
        .replaceMatches(#"^model\.visual\.backbone\."#, with: "model.vision_tower."),
        .replaceMatches(#"^model\.visual\.proj_in\."#, with: "model.vision_tower.merger.linear_1."),
        .replaceMatches(
            #"^model\.visual\.proj_out\."#, with: "model.vision_tower.merger.linear_2."),
        .replaceMatches(
            #"embeddings\.position_embedding$"#, with: "embeddings.position_embedding.weight"),
        .replaceMatches(#"(vision_tower\.layers\.\d+)\.ln_1\."#, with: "$1.layernorm_before."),
        .replaceMatches(#"(vision_tower\.layers\.\d+)\.ln_2\."#, with: "$1.layernorm_after."),
        .replaceMatches(#"(vision_tower\.layers\.\d+)\.mlp\.c_fc\."#, with: "$1.mlp.fc1."),
        .replaceMatches(#"(vision_tower\.layers\.\d+)\.mlp\.c_proj\."#, with: "$1.mlp.fc2."),
        .replaceMatches(#"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$"#, with: "$1.lambda_$2"),
        .replaceMatches(#"\.experts\.w13_weight"#, with: ".experts.gate_up_proj.weight"),
        .replaceMatches(#"\.experts\.w2_weight"#, with: ".experts.down_proj.weight"),
        .replaceMatches(#"\.weight_scale$"#, with: ".scales"),
        .replaceMatches(#"\.weight_(amax|outer_scale_denominator)$"#, with: ".$1"),
    ])

    /// The tensors that are float32 at run time.
    private static let float32Suffixes = [
        ".mlp.gate.weight", ".forget_gate_proj.weight", ".message_gate_proj.weight",
    ]

    /// `checkpoint` in the converted layout, or unchanged when it is converted already.
    static func convertingOriginalLayout(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        guard checkpoint.weights.keys.contains(where: isOriginal) else { return checkpoint }
        var checkpoint = try checkpoint.mapNames(convertedName)
        var weights = checkpoint.weights
        for (name, value) in weights {
            if name.hasSuffix("conv1d.weight") {
                weights[name] = value.swappedAxes(1, 2)
            } else if float32Suffixes.contains(where: name.hasSuffix) {
                weights[name] = value.asType(.float32)
            } else if name.contains(".experts."), name.hasSuffix(".weight"),
                value.dtype == .uint8
            {
                weights[name] = value.view(dtype: .uint32)
            }
        }
        foldGlobalScales(&weights)
        try foldMessageQueries(&weights)
        checkpoint.weights = weights
        return checkpoint
    }

    /// The converted name of a tensor of Thinking Machines' layout.
    static func convertedName(_ name: String) -> String? {
        // Layer 0 reads no attention input, so query i is for layer i + 1.
        let prefix = "model.llm.attn_res.attn_queries."
        if name.hasPrefix(prefix), let index = Int(name.dropFirst(prefix.count)) {
            return "model.language_model.layers.\(index + 1).attention_input.query"
        }
        return names.mapName(name)
    }

    private static func isOriginal(_ name: String) -> Bool {
        originalNamespaces.contains(where: name.hasPrefix)
    }

    /// Each expert projection's NVFP4 global scale, from its amax and outer-scale denominator.
    private static func foldGlobalScales(_ weights: inout [String: MLXArray]) {
        let amaxSuffix = ".amax"
        for name in weights.keys where name.hasSuffix(amaxSuffix) {
            let path = String(name.dropLast(amaxSuffix.count))
            guard let amax = weights.removeValue(forKey: name),
                let denominator = weights.removeValue(forKey: "\(path).outer_scale_denominator")
            else { continue }
            let scale = MLXArray(Float(6 * 448)) / denominator.asType(.float32)
            weights["\(path).global_scale"] = amax.asType(.float32) * scale
        }
        // A pipeline stage can hold one factor without the other.
        for name in weights.keys where name.hasSuffix(".outer_scale_denominator") {
            weights[name] = nil
        }
    }

    /// The loop's message queries: `[query, mlp]` and `[key, value]` rows per looped layer become
    /// `[query, key, value, mlp]`.
    private static func foldMessageQueries(_ weights: inout [String: MLXArray]) throws {
        let queriesName = "model.language_model.middle_cycle_cross_pass_queries"
        let kvName = "model.language_model.middle_cycle_cross_pass_kv_queries"
        let queries = weights.removeValue(forKey: queriesName)
        let kv = weights.removeValue(forKey: kvName)
        guard queries != nil || kv != nil else { return }
        guard let queries, let kv else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 needs both \(queriesName) and \(kvName).")
        }
        let dims = queries.dim(-1)
        let q = queries.reshaped(-1, 2, dims)
        let pairs = kv.reshaped(-1, 2, dims)
        weights["model.language_model.message_input.query"] = concatenated(
            [q[0..., ..<1], pairs, q[0..., 1...]], axis: 1)
    }
}
