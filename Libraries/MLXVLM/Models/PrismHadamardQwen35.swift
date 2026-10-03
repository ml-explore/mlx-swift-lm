// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 vision-language model loaded from a Hadamard-rotated ternary pack
/// (`model_type: prism_hadamard_qwen35`, `base_model_type: qwen3_5`).
///
/// The stock ``Qwen35`` with the manifest's language-model modules replaced by
/// `HadamardQuantizedLinear` / `HadamardQuantizedEmbedding` before the weights
/// load, and a sanitize pass that casts the pack's unpacked float32 tensors to the
/// manifest's activation dtype. The vision tower is not rotated and loads as stored.
public final class PrismHadamardQwen35: Qwen35 {

    /// The pack's manifest, validated for the Qwen3.5 vision class.
    public let checkpoint: HadamardQuantizedCheckpoint

    /// Builds the base model and substitutes the manifest modules.
    ///
    /// - Throws: `HadamardQuantizedCheckpointError` when the manifest is not one this
    ///   loader can honor or names a module the base model does not have.
    public init(_ configuration: Qwen35Configuration, manifest: HadamardQuantizedManifest)
        throws
    {
        checkpoint = try HadamardQuantizedCheckpoint(
            manifest: manifest, baseModelType: "qwen3_5", tensorNamespace: "mlx-vlm-qwen3_5",
            pathPrefix: "language_model.")
        super.init(configuration)
        try checkpoint.substituteModules(in: self)
    }

    public override func sanitize(weights: [String: MLXArray], metadata: [String: String])
        -> [String: MLXArray]
    {
        checkpoint.sanitize(super.sanitize(weights: weights, metadata: metadata))
    }
}

/// Registry creator for `prism_hadamard_qwen35`: the manifest and the base
/// configuration sit in the same `config.json`.
func createPrismHadamardQwen35Model(configuration data: Data) throws -> any LanguageModel {
    let decoder = JSONDecoder.json5()
    let manifest = try decoder.decode(HadamardQuantizedManifest.self, from: data)
    let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
    return try PrismHadamardQwen35(configuration, manifest: manifest)
}
