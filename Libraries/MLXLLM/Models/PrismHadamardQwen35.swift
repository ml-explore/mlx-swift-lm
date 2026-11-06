// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 text model loaded from a Hadamard-rotated ternary pack
/// (`model_type: prism_hadamard_qwen35`, `base_model_type: qwen3_5`).
///
/// The stock ``Qwen35Model`` with the manifest's modules replaced by rotated layers before the
/// weights load. `HadamardQuantizedCheckpointPolicy` prepares the checkpoint after
/// ``Qwen35Model`` has mapped its names.
public final class PrismHadamardQwen35Model: Qwen35Model {

    /// The pack's manifest, validated for the Qwen3.5 text class.
    public let checkpointPolicy: HadamardQuantizedCheckpointPolicy

    /// Builds the base model and substitutes the manifest modules.
    ///
    /// - Throws: `HadamardQuantizedCheckpointError` when the manifest is not one this loader
    ///   can honor or names a module the base model does not have.
    public init(_ configuration: Qwen35Configuration, manifest: HadamardQuantizedManifest)
        throws
    {
        checkpointPolicy = try HadamardQuantizedCheckpointPolicy(
            manifest: manifest, baseModelType: "qwen3_5", tensorNamespace: "mlx-vlm-qwen3_5",
            pathPrefix: "language_model.")
        super.init(configuration)
        try checkpointPolicy.substituteModules(in: self)
    }

    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws
        -> ModelCheckpoint
    {
        try checkpointPolicy.prepare(super.prepareCheckpoint(checkpoint))
    }
}

/// Registry creator for `prism_hadamard_qwen35`: the manifest and the base configuration sit in
/// the same `config.json`.
func createPrismHadamardQwen35Model(configuration data: Data) throws -> any LanguageModel {
    let decoder = JSONDecoder.json5()
    let manifest = try decoder.decode(HadamardQuantizedManifest.self, from: data)
    let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
    return try PrismHadamardQwen35Model(configuration, manifest: manifest)
}
