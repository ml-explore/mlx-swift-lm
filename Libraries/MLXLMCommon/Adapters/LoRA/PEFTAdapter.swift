//  PEFTAdapter.swift
//  mlx-libraries
//
//  Created by Sachin Desai on 5/21/26.
//

// Loader for PEFT-format LoRA adapters (the format used by Hugging Face's
// `peft` library). Converts to the MLX-native `LoRAContainer` representation.

import Foundation
import MLX
import MLXNN

/// PEFT `adapter_config.json` schema. Only the fields we use are decoded.
public struct PEFTAdapterConfiguration: Codable, Sendable {
    public let peftType: String
    public let r: Int
    public let loraAlpha: Float
    public let targetModules: [String]
    public let useDora: Bool

    enum CodingKeys: String, CodingKey {
        case peftType = "peft_type"
        case r = "r"
        case loraAlpha = "lora_alpha"
        case targetModules = "target_modules"
        case useDora = "use_dora"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peftType = try c.decodeIfPresent(String.self, forKey: .peftType) ?? "LORA"
        r = try c.decode(Int.self, forKey: .r)
        loraAlpha = try c.decode(Float.self, forKey: .loraAlpha)
        targetModules = try c.decode([String].self, forKey: .targetModules)
        useDora = try c.decodeIfPresent(Bool.self, forKey: .useDora) ?? false
    }
}

extension LoRAContainer {

    /// Loads a PEFT-format LoRA adapter directory and returns a
    /// ``LoRAContainer`` you can ``LoRAContainer/load(into:)`` on a model.
    ///
    /// PEFT format (Hugging Face `peft` library):
    /// ```
    /// adapter_config.json:
    ///   {
    ///     "peft_type": "LORA",
    ///     "r": 128,
    ///     "lora_alpha": 512,
    ///     "target_modules": ["o_proj"],
    ///     ...
    ///   }
    /// adapter_model.safetensors (one pair per target module per layer):
    ///   base_model.model.<inner-path>.<target_module>.lora_A.weight  shape [r, in]
    ///   base_model.model.<inner-path>.<target_module>.lora_B.weight  shape [out, r]
    /// ```
    ///
    /// MLX-native format (``LoRALinear`` / ``QLoRALinear``):
    /// ```
    /// loraA: [in, r], loraB: [r, out], scale = lora_alpha / r
    /// ```
    ///
    /// Conversion performed by this loader:
    /// - Strip the leading `base_model.model.` prefix.
    /// - Rename `.lora_A.weight` → `.lora_a`, `.lora_B.weight` → `.lora_b`.
    /// - Transpose both tensors to match MLX's `[in, r]` / `[r, out]` convention.
    /// - Compute `scale = lora_alpha / r`.
    ///
    /// - Parameters:
    ///   - directory: Directory containing `adapter_config.json` and
    ///     `adapter_model.safetensors`.
    ///   - numLayers: Apply the adapter to at most the last `numLayers`
    ///     entries of the model's `loraLayers`. Defaults to a large value so
    ///     the suffix slice covers every layer; pass a smaller value to
    ///     restrict adapters to only the upper N transformer blocks.
    /// - Returns: A ``LoRAContainer`` ready to be loaded into any
    ///   ``LoRAModel``-conforming model.
    /// - Throws: ``ModelAdapterError/unsupportedAdapterType(_:)`` if the
    ///   adapter is not LORA, or if `use_dora=true` is set.
    public static func fromPEFT(directory: URL, numLayers: Int = 1024) throws -> LoRAContainer {
        let configURL = directory.appending(component: "adapter_config.json")
        let configData = try Data(contentsOf: configURL)
        let peft = try JSONDecoder().decode(PEFTAdapterConfiguration.self, from: configData)

        guard peft.peftType.uppercased() == "LORA" else {
            throw ModelAdapterError.unsupportedAdapterType(
                "PEFT adapter type \(peft.peftType) (only LORA is supported)")
        }
        if peft.useDora {
            throw ModelAdapterError.unsupportedAdapterType(
                "PEFT LORA with use_dora=true")
        }

        let weightsURL = directory.appending(component: "adapter_model.safetensors")
        let raw = try MLX.loadArrays(url: weightsURL)
        let (renamed, keys) = renamePEFTWeights(raw, targetModules: peft.targetModules)

        let scale = peft.loraAlpha / Float(peft.r)
        let configuration = LoRAConfiguration(
            numLayers: numLayers,
            fineTuneType: .lora,
            loraParameters: LoRAConfiguration.LoRAParameters(
                rank: peft.r, scale: scale, keys: keys))

        return LoRAContainer(
            configuration: configuration,
            parameters: ModuleParameters.unflattened(renamed))
    }

    /// PEFT key → MLX key transformation, plus tensor transposition.
    /// Returns (renamed_weights, mlx_target_keys).
    private static func renamePEFTWeights(
        _ raw: [String: MLXArray], targetModules: [String]
    ) -> ([String: MLXArray], [String]) {
        var renamed: [String: MLXArray] = [:]
        var seenTargetKeys = Set<String>()

        for (key, tensor) in raw {
            // Drop the leading "base_model.model." prefix that PEFT injects.
            var k = key
            if k.hasPrefix("base_model.model.") {
                k = String(k.dropFirst("base_model.model.".count))
            }

            let suffix: String
            let mlxParam: String
            if k.hasSuffix(".lora_A.weight") {
                suffix = ".lora_A.weight"
                mlxParam = ".lora_a"
            } else if k.hasSuffix(".lora_B.weight") {
                suffix = ".lora_B.weight"
                mlxParam = ".lora_b"
            } else {
                // Unknown key — skip rather than fail loudly so PEFT exports
                // with extra bookkeeping tensors still load.
                continue
            }

            let modulePath = String(k.dropLast(suffix.count))
            // Track which path the LoRA layer needs to live at (e.g.
            // "encoder.layers.0.self_attn.o_proj"). LoRAContainer's
            // `keys` filter walks each `loraLayers` entry and matches against
            // the path *relative to that entry*, so we strip the
            // "encoder.layers.<n>." prefix to get the layer-relative key.
            if let relative = stripLayerPrefix(modulePath) {
                seenTargetKeys.insert(relative)
            }

            // PEFT stores lora_A as [r, in] and lora_B as [out, r]. MLX wants
            // lora_a as [in, r] and lora_b as [r, out]. Transpose both.
            renamed[modulePath + mlxParam] = tensor.transposed()
        }

        // Sort for determinism in tests.
        let keys = Array(seenTargetKeys).sorted()
        return (renamed, keys)
    }

    private static func stripLayerPrefix(_ path: String) -> String? {
        // Match "<encoder|model>.layers.<n>." then return the rest. This
        // matches the two common backbone layouts the project uses.
        let parts = path.split(separator: ".", omittingEmptySubsequences: false)
        // Need at least 3 parts to form "<root>.layers.<n>".
        guard parts.count >= 3 else { return nil }
        for i in 0 ... (parts.count - 3) {
            if parts[i + 1] == "layers", Int(parts[i + 2]) != nil {
                let tail = parts[(i + 3)...].joined(separator: ".")
                return tail.isEmpty ? nil : tail
            }
        }
        return nil
    }
}
