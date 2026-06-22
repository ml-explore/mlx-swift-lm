import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders

// `swift test` from the CLI can't bundle mlx-swift's Metal library, so every MLX
// op below is scoped to the CPU backend via `Device.withDefaultDevice(.cpu)`
// (identical math). The default device is a task-local, so each test scopes its
// own work and they remain safe to run in parallel.

// Parity fixtures: the real bf16 models resolved from the default Hugging Face hub
// cache (`~/.cache/huggingface`, via `hfSnapshotDir`), plus the HF float32 reference
// exported by Models/mlx/export_swift_ref.py. The parity test is skipped automatically
// when these aren't present locally (CI, or a machine that hasn't downloaded the models);
// it never triggers a download.
private let lfm2EmbeddingDir = hfSnapshotDir(modelId: "mlx-community/LFM2.5-Embedding-350M-bf16")
private let lfm2ColbertDir = hfSnapshotDir(modelId: "mlx-community/LFM2.5-ColBERT-350M-bf16")
private let lfm2RefURL = URL(fileURLWithPath: "/tmp/ref/swift_ref.safetensors")

private let lfm2FixturesAvailable: Bool = {
    guard let lfm2EmbeddingDir, let lfm2ColbertDir else { return false }
    let fm = FileManager.default
    return fm.fileExists(atPath: lfm2RefURL.path)
        && fm.fileExists(atPath: lfm2EmbeddingDir.appendingPathComponent("config.json").path)
        && fm.fileExists(atPath: lfm2ColbertDir.appendingPathComponent("config.json").path)
}()

private let embeddingConfigJSON = """
    {
      "model_type": "lfm2",
      "vocab_size": 64,
      "hidden_size": 32,
      "num_hidden_layers": 4,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "norm_eps": 1e-5,
      "conv_bias": false,
      "conv_L_cache": 3,
      "block_ff_dim": 64,
      "block_multiple_of": 8,
      "block_ffn_dim_multiplier": 1.0,
      "block_auto_adjust_ff_dim": true,
      "rope_theta": 1000000.0,
      "layer_types": ["conv", "full_attention", "conv", "full_attention"],
      "max_position_embeddings": 128000,
      "mlx": { "head": "embedding", "pooling": "cls",
               "prompts": { "query": "query: ", "document": "document: " } }
    }
    """

private let colbertConfigJSON = """
    {
      "model_type": "lfm2",
      "vocab_size": 64,
      "hidden_size": 32,
      "num_hidden_layers": 4,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "norm_eps": 1e-5,
      "conv_bias": false,
      "conv_L_cache": 3,
      "block_ff_dim": 64,
      "block_multiple_of": 8,
      "block_ffn_dim_multiplier": 1.0,
      "block_auto_adjust_ff_dim": true,
      "rope_theta": 1000000.0,
      "layer_types": ["conv", "full_attention", "conv", "full_attention"],
      "mlx": { "head": "colbert", "proj_dim": 16,
               "query_prefix": "[Q] ", "document_prefix": "[D] ",
               "query_length": 32, "document_length": 512 }
    }
    """

private func decode(_ json: String) throws -> LFM2BidirectionalConfiguration {
    try JSONDecoder().decode(LFM2BidirectionalConfiguration.self, from: Data(json.utf8))
}

private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.asType(.float32).reshaped(-1)
    let bf = b.asType(.float32).reshaped(-1)
    let dot = (af * bf).sum()
    let na = sqrt((af * af).sum())
    let nb = sqrt((bf * bf).sum())
    return (dot / (na * nb)).item(Float.self)
}

struct LFM2BidirectionalTests {

    // MARK: - Configuration

    @Test("Embedding config decodes to a CLS-pooled head")
    func testEmbeddingConfigDecode() throws {
        let c = try decode(embeddingConfigJSON)
        #expect(c.mlx.head == .embedding)
        #expect(c.vocabSize == 64)
        #expect(c.numKeyValueHeads == 2)
        // full_attention at indices 1 and 3.
        #expect(c.attnLayerIdxs == [1, 3])
        Device.withDefaultDevice(.cpu) {
            #expect(LFM2BidirectionalModel(c).poolingStrategy == .cls)
        }
    }

    @Test("ColBERT config decodes to a none-pooled projection head")
    func testColbertConfigDecode() throws {
        let c = try decode(colbertConfigJSON)
        #expect(c.mlx.head == .colbert)
        #expect(c.mlx.projDim == 16)
        #expect(c.mlx.queryLength == 32)
        Device.withDefaultDevice(.cpu) {
            #expect(LFM2BidirectionalModel(c).poolingStrategy == Pooling.Strategy.none)
        }
    }

    @Test("Real configs decode with the expected attention layout")
    func testRealConfigAttentionLayout() throws {
        // The 350M models alternate conv/attention with full_attention at these indices.
        let layerTypes = [
            "conv", "conv", "full_attention", "conv", "conv", "full_attention",
            "conv", "conv", "full_attention", "conv", "full_attention", "conv",
            "full_attention", "conv", "full_attention", "conv",
        ]
        let json = """
            {
              "model_type": "lfm2", "vocab_size": 65536, "hidden_size": 1024,
              "num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 8,
              "norm_eps": 1e-5, "conv_L_cache": 3, "block_ff_dim": 6656,
              "block_multiple_of": 256, "block_ffn_dim_multiplier": 1.0,
              "block_auto_adjust_ff_dim": true, "rope_theta": 1000000.0,
              "layer_types": \(layerTypes), "mlx": { "head": "embedding", "pooling": "cls" }
            }
            """
        let c = try decode(json)
        #expect(c.attnLayerIdxs == [2, 5, 8, 10, 12, 14])
        #expect(c.headDim == 64)
    }

    // MARK: - Synthetic forward (shape + invariants, random weights)

    @Test("Embedding head pools the CLS token to a normalized vector")
    func testEmbeddingForwardShape() throws {
        let c = try decode(embeddingConfigJSON)
        Device.withDefaultDevice(.cpu) {
            let model = LFM2BidirectionalModel(c)
            let ids = MLXArray((0 ..< 10).map { Int32($0 % 64) }).reshaped(2, 5)

            let out = model(ids)
            out.hiddenStates!.eval()
            #expect(out.hiddenStates!.shape == [2, 5, 32])

            let pooled = Pooling(strategy: .cls)(out, normalize: true)
            pooled.eval()
            #expect(pooled.shape == [2, 32])
            // L2-normalized rows -> unit norm.
            let norms = sqrt((pooled * pooled).sum(axis: -1))
            #expect(abs(norms[0].item(Float.self) - 1.0) < 1e-3)
        }
    }

    @Test("ColBERT head projects every token to projDim multi-vectors")
    func testColbertForwardShape() throws {
        let c = try decode(colbertConfigJSON)
        Device.withDefaultDevice(.cpu) {
            let model = LFM2BidirectionalModel(c)
            let ids = MLXArray((0 ..< 10).map { Int32($0 % 64) }).reshaped(2, 5)

            let out = model(ids)
            out.hiddenStates!.eval()
            #expect(out.hiddenStates!.shape == [2, 5, 16])

            // .none pooling returns the per-token multi-vectors unchanged.
            let pooled = Pooling(strategy: .none)(out)
            pooled.eval()
            #expect(pooled.shape == [2, 5, 16])
        }
    }

    @Test("A config without an mlx block defaults to a CLS embedding head")
    func testConfigWithoutMLXBlockDefaultsToEmbedding() throws {
        let json = """
            {
              "model_type": "lfm2", "vocab_size": 64, "hidden_size": 32,
              "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "norm_eps": 1e-5, "conv_L_cache": 3, "block_ff_dim": 64,
              "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
              "block_auto_adjust_ff_dim": true, "rope_theta": 1000000.0,
              "layer_types": ["conv", "full_attention"]
            }
            """
        let c = try decode(json)
        #expect(c.mlx.head == .embedding)
        Device.withDefaultDevice(.cpu) {
            #expect(LFM2BidirectionalModel(c).poolingStrategy == .cls)
        }
    }

    @Test("ColBERT returns raw per-token vectors and does not erase masked positions")
    func testColbertKeepsMaskedPositions() throws {
        let c = try decode(colbertConfigJSON)
        Device.withDefaultDevice(.cpu) {
            let model = LFM2BidirectionalModel(c)
            let ids = MLXArray((0 ..< 6).map { Int32($0 % 64) }).reshaped(1, 6)
            let mask = MLXArray([1, 1, 1, 1, 0, 0].map { Int32($0) }).reshaped(1, 6)

            let out = model(ids, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
            let tok = out.hiddenStates!  // (1, 6, projDim)
            tok.eval()
            // Masked positions (4, 5) are NOT zeroed: the encoder returns raw vectors,
            // so ColBERT query-expansion tokens survive for the retrieval layer to keep
            // (documents drop padding + apply the skiplist outside the model).
            let maskedNorm = sqrt((tok[0, 4] * tok[0, 4]).sum()).item(Float.self)
            let realNorm = sqrt((tok[0, 0] * tok[0, 0]).sum()).item(Float.self)
            #expect(maskedNorm > 0)
            #expect(realNorm > 0)
        }
    }

    // MARK: - Parity vs HF float32 reference (gated on local fixtures)

    @Test(
        "LFM2.5 outputs match the HF float32 reference",
        .enabled(if: lfm2FixturesAvailable))
    func testParityAgainstReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let ref = try loadArrays(url: lfm2RefURL)

            // Embedding: compare the raw CLS vector (cosine is scale-invariant).
            // Force-unwrap is safe: `lfm2FixturesAvailable` already proved both dirs exist.
            let emb = try loadLFM2(lfm2EmbeddingDir!)
            var worstEmb: Float = 1
            var i = 0
            while let idsArr = ref["emb_ids_\(i)"] {
                let ids = idsArr.asType(.int32).reshaped(1, -1)
                let cls = emb(ids).hiddenStates![0..., 0, 0...]
                worstEmb = min(worstEmb, cosine(cls, ref["emb_cls_\(i)"]!))
                i += 1
            }
            #expect(i > 0)
            #expect(worstEmb > 0.999, "worst embedding cosine \(worstEmb)")

            // ColBERT: compare the per-token projection.
            let col = try loadLFM2(lfm2ColbertDir!)
            var worstCol: Float = 1
            i = 0
            while let idsArr = ref["col_ids_\(i)"] {
                let ids = idsArr.asType(.int32).reshaped(1, -1)
                let proj = col(ids).hiddenStates!
                worstCol = min(worstCol, cosine(proj, ref["col_proj_\(i)"]!))
                i += 1
            }
            #expect(i > 0)
            #expect(worstCol > 0.999, "worst colbert cosine \(worstCol)")
        }
    }

    /// Build the model from a local dir and load its weights as float32 (max precision
    /// for the parity comparison), mirroring the factory's sanitize + load path.
    private func loadLFM2(_ dir: URL) throws -> LFM2BidirectionalModel {
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder.json5().decode(
            LFM2BidirectionalConfiguration.self, from: configData)
        let model = LFM2BidirectionalModel(config)

        var weights = try loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        weights = model.sanitize(weights: weights)
        weights = weights.mapValues { $0.asType(.float32) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
