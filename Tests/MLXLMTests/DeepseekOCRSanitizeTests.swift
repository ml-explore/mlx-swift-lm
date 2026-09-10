// Copyright © 2026 Apple Inc.
//
// DeepSeek-OCR / Unlimited-OCR checkpoints name their bare-array parameters with
// safetensors leaf names (`image_newline`, `sam_model.pos_embed`, ...). The model
// declares those names with `@ParameterInfo(key:)`, so `sanitize` only re-roots
// prefixes, collapses the Unlimited `view_seperator` typo, and reshapes the CLIP
// position table. A sanitize that renamed leaves onto camelCase properties, or a
// property that lost its key, would fail a strict load with missing and unused keys.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class DeepseekOCRSanitizeTests: XCTestCase {

    /// `(checkpoint key, module parameter key)` for every bare-array parameter, as
    /// Unlimited-OCR packs ship them: rooted at `model.`, with the `view_seperator` typo.
    private static let unlimitedLeafKeys: [(pack: String, module: String)] = [
        ("model.image_newline", "image_newline"),
        ("model.view_seperator", "view_separator"),
        ("model.sam_model.pos_embed", "sam_model.pos_embed"),
        ("model.sam_model.blocks.0.attn.rel_pos_h", "sam_model.layers.0.attn.rel_pos_h"),
        ("model.sam_model.blocks.0.attn.rel_pos_w", "sam_model.layers.0.attn.rel_pos_w"),
        (
            "model.vision_model.embeddings.class_embedding",
            "vision_model.embeddings.class_embedding"
        ),
        (
            "model.vision_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.position_embedding"
        ),
    ]

    /// The same parameters as DeepSeek-OCR packs ship them: root level, `view_separator`.
    private static let deepseekLeafKeys: [(pack: String, module: String)] = [
        ("image_newline", "image_newline"),
        ("view_separator", "view_separator"),
        ("sam_model.pos_embed", "sam_model.pos_embed"),
        ("sam_model.blocks.0.attn.rel_pos_h", "sam_model.layers.0.attn.rel_pos_h"),
        ("sam_model.blocks.0.attn.rel_pos_w", "sam_model.layers.0.attn.rel_pos_w"),
        ("vision_model.embeddings.class_embedding", "vision_model.embeddings.class_embedding"),
        (
            "vision_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.position_embedding"
        ),
    ]

    func testBareArrayParametersUseSafetensorsLeafKeys() throws {
        let keys = Set(try makeModel().parameters().flattened().map(\.0))

        for (_, moduleKey) in Self.deepseekLeafKeys {
            XCTAssertTrue(keys.contains(moduleKey), "missing parameter key \(moduleKey)")
        }
    }

    func testSanitizeMapsUnlimitedPackLeavesOntoParameterKeys() throws {
        try assertSanitizeMapsLeaves(Self.unlimitedLeafKeys)
    }

    func testSanitizeMapsDeepseekPackLeavesOntoParameterKeys() throws {
        try assertSanitizeMapsLeaves(Self.deepseekLeafKeys)
    }

    /// Runs a real checkpoint's tensor names through `sanitize` and requires the result
    /// to be exactly the module's parameter keys (plus quantization companions of them).
    /// Names are read from the safetensors headers of the files the loader selects, not
    /// from `model.safetensors.index.json`, which a converted pack may leave stale.
    /// Packs come from `DEEPSEEK_OCR_PACK_DIRS` (colon-separated directories) and the
    /// Hugging Face cache snapshots of `mlx-community/DeepSeek-OCR-4bit`; skips when none
    /// is present.
    func testSanitizeOnLocalPackTensorNamesYieldsExactlyTheModuleParameterKeys() throws {
        let packs = Self.localPackDirectories()
        guard !packs.isEmpty else {
            throw XCTSkip("no local DeepSeek-OCR checkpoint; set DEEPSEEK_OCR_PACK_DIRS")
        }

        for pack in packs {
            let config = try JSONDecoder().decode(
                DeepseekOCRConfiguration.self,
                from: Data(contentsOf: pack.appending(path: "config.json")))
            let tensorNames = try safetensorWeightURLs(in: pack).flatMap {
                try safetensorSpansInFileOrder(url: $0).map(\.name)
            }
            XCTAssertFalse(tensorNames.isEmpty, "\(pack.lastPathComponent): no tensors")
            let model = DeepseekOCR(config)
            let moduleKeys = Set(model.parameters().flattened().map(\.0))

            // Key mapping does not depend on shapes: sanitize's only shape-sensitive
            // branches (conv layout, CLIP position table) either skip or keep 2-D inputs.
            let weights = Dictionary(
                uniqueKeysWithValues: tensorNames.map { ($0, zeros([1, 1])) })
            let sanitized = Set(model.sanitize(weights: weights).keys)

            // Quantized packs ship `.scales` / `.biases` beside each quantized `.weight`;
            // `quantize(model:)` creates those parameters at load, so they count as
            // reaching the `.weight` they belong to.
            let loadable = Set(
                sanitized.compactMap { key -> String? in
                    for companion in [".scales", ".biases"] where key.hasSuffix(companion) {
                        let weight = String(key.dropLast(companion.count)) + ".weight"
                        return moduleKeys.contains(weight) ? nil : key
                    }
                    return key
                })

            XCTAssertEqual(
                loadable.subtracting(moduleKeys).sorted(), [],
                "\(pack.lastPathComponent): sanitized keys no parameter accepts")
            XCTAssertEqual(
                moduleKeys.subtracting(loadable).sorted(), [],
                "\(pack.lastPathComponent): parameters no checkpoint key reaches")
        }
    }

    private func assertSanitizeMapsLeaves(
        _ leaves: [(pack: String, module: String)],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let model = try makeModel()
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })

        var weights = [String: MLXArray]()
        for (packKey, moduleKey) in leaves {
            let shape = try XCTUnwrap(shapes[moduleKey], moduleKey, file: file, line: line)
            // HF stores the CLIP position table as `Embedding.weight` [N, D].
            let packShape =
                packKey.hasSuffix(".position_embedding.weight") ? Array(shape.dropFirst()) : shape
            weights[packKey] = zeros(packShape)
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), Set(leaves.map(\.module)), file: file, line: line)
        for (_, moduleKey) in leaves {
            XCTAssertEqual(
                sanitized[moduleKey]?.shape, shapes[moduleKey], moduleKey, file: file, line: line)
        }
        // Strict load: every sanitized key must land on a parameter.
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])
    }

    private func makeModel() throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(Self.configJSON.utf8))
        return DeepseekOCR(config)
    }

    private static func localPackDirectories() -> [URL] {
        let environment = ProcessInfo.processInfo.environment
        var candidates = [URL]()
        if let dirs = environment["DEEPSEEK_OCR_PACK_DIRS"] {
            candidates += dirs.split(separator: ":").map { URL(filePath: String($0)) }
        }
        let hub =
            environment["HF_HOME"].map { URL(filePath: $0).appending(path: "hub") }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(
                path: ".cache/huggingface/hub")
        let snapshots = hub.appending(path: "models--mlx-community--DeepSeek-OCR-4bit/snapshots")
        candidates +=
            (try? FileManager.default.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil)) ?? []
        return candidates.filter { pack in
            FileManager.default.fileExists(atPath: pack.appending(path: "config.json").path)
                && !((try? safetensorWeightURLs(in: pack)) ?? []).isEmpty
        }
    }

    /// SAM-base vision geometry over a one-layer language model; the vision tower's
    /// parameter set is what the leaf keys exercise.
    private static let configJSON = #"""
        {
         "model_type": "deepseekocr",
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 32,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "max_position_embeddings": 32
         }
        }
        """#
}
