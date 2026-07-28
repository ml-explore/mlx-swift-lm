// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// End-to-end regression coverage for mixed-precision quantized checkpoints.
///
/// The mlx-community Gemma 4 QAT uploads (e.g. `gemma-4-12B-it-qat-4bit`,
/// `gemma-4-E4B-it-qat-4bit`) quantize attention and embeddings at 4-bit but
/// every `mlp.{gate,up,down}_proj` at 8-bit, declared in `config.json` as a
/// global `quantization` dict interleaved with per-module overrides:
///
/// ```json
/// "quantization": {
///     "group_size": 64, "bits": 4, "mode": "affine",
///     "language_model.model.layers.0.mlp.gate_proj": {"group_size": 64, "bits": 8},
///     ...
/// }
/// ```
///
/// `loadWeights` must quantize each module with its per-module bits (matching
/// Python mlx-lm's `class_predicate`), and a load that ignores the overrides
/// must fail loudly on the packed-shape mismatch rather than silently
/// mis-decoding 8-bit payloads as 4-bit (see ollama/ollama#16740 for the
/// confusion these checkpoints caused across loaders).
struct MixedPrecisionQuantLoadTests {

    private static let groupSize = 64

    private func tinyUnifiedConfig() throws -> Gemma4UnifiedConfiguration {
        // Text-only Gemma4 Unified sized so every quantizable input dimension
        // is divisible by the quantization group size.
        try JSONDecoder.json5().decode(
            Gemma4UnifiedConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "gemma4_unified",
                  "vocab_size": 64,
                  "image_token_id": 31,
                  "audio_token_id": 30,
                  "video_token_id": 29,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 64,
                    "global_head_dim": 64,
                    "vocab_size": 64,
                    "vocab_size_per_layer_input": 64,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "attention_k_eq_v": true,
                    "use_double_wide_mlp": false,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": true
                  },
                  "vision_config": null,
                  "audio_config": null
                }
                """.utf8))
    }

    /// Quantize the freshly initialized model's own float weights into a
    /// synthetic checkpoint laid out like the published QAT repos: 8-bit for
    /// `mlp.*` projections, 4-bit for every other quantizable module, plain
    /// float arrays for the rest.
    private func makeCheckpoint(model: Module) -> (
        arrays: [String: MLXArray], mlpPaths: [String], fourBitPaths: [String]
    ) {
        var arrays = [String: MLXArray]()
        var mlpPaths = [String]()
        var fourBitPaths = [String]()
        var quantizedModulePaths = Set<String>()

        for (path, module) in model.leafModules().flattened() {
            let weight: MLXArray
            switch module {
            case let linear as Linear:
                weight = linear.weight
            case let embedding as Embedding:
                weight = embedding.weight
            default:
                continue
            }
            let bits: Int
            if path.contains(".mlp.") {
                bits = 8
                mlpPaths.append(path)
            } else {
                bits = 4
                fourBitPaths.append(path)
            }
            let (wq, scales, biases) = quantized(
                weight, groupSize: Self.groupSize, bits: bits)
            arrays["\(path).weight"] = wq
            arrays["\(path).scales"] = scales
            if let biases {
                arrays["\(path).biases"] = biases
            }
            quantizedModulePaths.insert(path)
        }

        for (key, value) in model.parameters().flattened() {
            let modulePath = key.split(separator: ".").dropLast().joined(separator: ".")
            if quantizedModulePaths.contains(modulePath) { continue }
            arrays[key] = value
        }

        return (arrays, mlpPaths, fourBitPaths)
    }

    private func quantizationConfig(mlpPaths: [String]) throws -> BaseConfiguration {
        let overrides = mlpPaths.map {
            "\"\($0)\": {\"group_size\": \(Self.groupSize), \"bits\": 8}"
        }
        .joined(separator: ",\n")
        return try JSONDecoder.json5().decode(
            BaseConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "gemma4_unified",
                  "quantization": {
                    "group_size": \(Self.groupSize),
                    "bits": 4,
                    "mode": "affine",
                    \(overrides)
                  }
                }
                """.utf8))
    }

    private func writeCheckpoint(_ arrays: [String: MLXArray]) throws -> URL {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "mixed-precision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: arrays, url: directory.appending(component: "model.safetensors"))
        return directory
    }

    @Test("Per-module quantization overrides are applied when loading")
    func perModuleOverridesApplied() throws {
        let config = try tinyUnifiedConfig()
        let reference = Gemma4Unified(config)
        let (arrays, mlpPaths, fourBitPaths) = makeCheckpoint(model: reference)
        #expect(!mlpPaths.isEmpty)
        #expect(!fourBitPaths.isEmpty)

        let directory = try writeCheckpoint(arrays)
        defer { try? FileManager.default.removeItem(at: directory) }

        let baseConfig = try quantizationConfig(mlpPaths: mlpPaths)
        let model = Gemma4Unified(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        for path in mlpPaths {
            let quantized = try #require(
                modules[path] as? QuantizedLinear, "\(path) should be quantized")
            #expect(quantized.bits == 8, "\(path) must honor the 8-bit override")
            #expect(quantized.groupSize == Self.groupSize)
        }
        for path in fourBitPaths {
            let quantized = try #require(
                modules[path] as? Quantized, "\(path) should be quantized")
            #expect(quantized.bits == 4, "\(path) must use the global 4-bit default")
            #expect(quantized.groupSize == Self.groupSize)
        }
    }

    @Test("Ignoring the per-module overrides fails loudly, not silently")
    func globalOnlyQuantizationFailsLoudly() throws {
        let config = try tinyUnifiedConfig()
        let reference = Gemma4Unified(config)
        let (arrays, mlpPaths, _) = makeCheckpoint(model: reference)
        #expect(!mlpPaths.isEmpty)

        let directory = try writeCheckpoint(arrays)
        defer { try? FileManager.default.removeItem(at: directory) }

        // A loader that applies only the global 4-bit settings would build
        // 4-bit MLP modules whose packed shapes cannot hold the checkpoint's
        // 8-bit payload. That must surface as a load error — decoding the
        // 8-bit payload as 4-bit would produce garbage logits downstream.
        let model = Gemma4Unified(config)
        #expect(throws: (any Error).self) {
            try loadWeights(
                modelDirectory: directory, model: model,
                quantization: .init(groupSize: Self.groupSize, bits: 4))
        }
    }
}
