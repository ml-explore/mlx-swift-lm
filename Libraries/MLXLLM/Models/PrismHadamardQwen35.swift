// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 text model loaded from a Hadamard-rotated ternary pack
/// (`model_type: prism_hadamard_qwen35`, `base_model_type: qwen3_5`).
///
/// The pack stores its language-model weights in MLX affine 2-bit form in a
/// Hadamard-rotated input basis, and its `config.json` carries a
/// `HadamardQuantizedManifest` naming every rotated module. This class is the
/// stock ``Qwen35Model`` with those modules replaced by `HadamardQuantizedLinear`
/// / `HadamardQuantizedEmbedding` before the weights load, and a sanitize pass that
/// casts the pack's unpacked float32 tensors to the manifest's activation dtype.
public final class PrismHadamardQwen35Model: Qwen35Model {

    /// The pack's manifest, validated for the Qwen3.5 text class.
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

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        checkpoint.sanitize(super.sanitize(weights: weights))
    }
}

/// Registry creator for `prism_hadamard_qwen35`: the manifest and the base
/// configuration sit in the same `config.json`.
func createPrismHadamardQwen35Model(configuration data: Data) throws -> any LanguageModel {
    let decoder = JSONDecoder.json5()
    let manifest = try decoder.decode(HadamardQuantizedManifest.self, from: data)
    let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
    return try PrismHadamardQwen35Model(configuration, manifest: manifest)
}
