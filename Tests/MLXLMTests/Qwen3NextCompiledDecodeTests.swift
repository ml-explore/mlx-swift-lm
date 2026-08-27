import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen3NextCompiledDecodeTests: XCTestCase {
    private struct Snapshot: Equatable {
        let shape: [Int]
        let dtype: DType
        let bytes: Data

        init(_ array: MLXArray) {
            let contents = array.asData(access: .copy)
            self.shape = contents.shape
            self.dtype = contents.dType
            self.bytes = contents.data
        }
    }

    private struct RunResult {
        let logits: [Snapshot]
        let cacheState: [Snapshot]
        let offsets: [Int]
    }

    private func configuration(
        headDim: Int = 8,
        quantizable: Bool = false
    ) throws -> Qwen3NextConfiguration {
        let hiddenSize = quantizable ? 64 : 16
        let intermediateSize = quantizable ? 128 : 32
        let linearHeadDim = quantizable ? 32 : 8
        let expertSize = quantizable ? 64 : 16
        let json = """
            {
                "hidden_size": \(hiddenSize), "num_hidden_layers": 4,
                "intermediate_size": \(intermediateSize),
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": \(headDim),
                "linear_num_value_heads": 2, "linear_num_key_heads": 2,
                "linear_key_head_dim": \(linearHeadDim),
                "linear_value_head_dim": \(linearHeadDim),
                "linear_conv_kernel_dim": 4,
                "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": \(expertSize), "mlp_only_layers": [],
                "moe_intermediate_size": \(expertSize), "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128, "norm_topk_prob": true,
                "tie_word_embeddings": true, "attention_bias": false,
                "full_attention_interval": 2
            }
            """
        return try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: Data(json.utf8))
    }

    private func run(
        _ model: Qwen3NextModel,
        tokens: [Int32],
        compiled: Bool,
        cache makeCache: (Qwen3NextModel) throws -> [KVCache] = {
            try $0.newCache(parameters: nil)
        }
    ) throws -> RunResult {
        let cache = try makeCache(model)
        var logits: [Snapshot] = []
        for token in tokens {
            let output = model.forward(
                MLXArray([token]).reshaped(1, 1),
                cache: cache,
                useCompiledDecode: compiled)
            eval(output)
            logits.append(Snapshot(output))
        }
        return RunResult(
            logits: logits,
            cacheState: cache.flatMap { $0.innerState() }.map(Snapshot.init),
            offsets: cache.map(\.offset))
    }

    func testSharedScheduleSplitsAtAttentionWrites() {
        let segments = CompiledDecodeSegment.schedule(
            linearLayers: [true, false, true, true, false])

        XCTAssertEqual(
            segments,
            [
                .init(linearLayers: [0], attentionPreLayer: 1),
                .init(attentionPostLayer: 1, linearLayers: [2, 3], attentionPreLayer: 4),
                .init(attentionPostLayer: 4),
            ])
    }

    func testDecodeConvolutionMatchesGeneralKernelExactly() throws {
        let config = try configuration()
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(29)
            let gdn = Qwen3NextGatedDeltaNet(config)
            gdn.conv1d.update(
                parameters: gdn.conv1d.parameters().mapValues { $0.asType(dtype) })
            let qkv = MLXRandom.normal([1, 1, gdn.convDim]).asType(dtype)
            let state = MLXRandom.normal(
                [1, gdn.convKernelSize - 1, gdn.convDim]
            ).asType(dtype)

            let fused = gdn.decodeConv(convState: state, qkv: qkv)
            let general = gdn.generalConv(convState: state, qkv: qkv)

            XCTAssertEqual(
                Snapshot(fused.conv), Snapshot(general.conv), "conv differs for \(dtype)")
            XCTAssertEqual(
                Snapshot(fused.state), Snapshot(general.state), "state differs for \(dtype)")
        }
    }

    func testCompiledDecodeMatchesGeneralPathExactly() throws {
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(31)
            let model = Qwen3NextModel(try configuration())
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            let tokens = [Int32(1), 7, 3, 9, 2]

            let general = try run(model, tokens: tokens, compiled: false)
            let compiled = try run(model, tokens: tokens, compiled: true)

            XCTAssertEqual(compiled.logits, general.logits, "logits differ for \(dtype)")
            XCTAssertEqual(
                compiled.cacheState, general.cacheState, "cache state differs for \(dtype)")
            XCTAssertEqual(compiled.offsets, general.offsets)
            XCTAssertEqual(compiled.offsets, [0, 5, 0, 5])
            if dtype == .float16 {
                XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
            } else {
                XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
            }
        }
    }

    func testQuantizedCacheKeepsExactGeneralFallback() throws {
        MLXRandom.seed(37)
        let model = Qwen3NextModel(try configuration(headDim: 32))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        let tokens = [Int32(2), 4, 6]
        let quantizedCache: (Qwen3NextModel) throws -> [KVCache] = { model in
            try model.newCache(parameters: nil).map { cache in
                cache is MambaCache
                    ? cache : QuantizedKVCache(groupSize: 32, bits: 8)
            }
        }

        let general = try run(
            model, tokens: tokens, compiled: false, cache: quantizedCache)
        let fallback = try run(
            model, tokens: tokens, compiled: true, cache: quantizedCache)

        XCTAssertEqual(fallback.logits, general.logits)
        XCTAssertEqual(fallback.cacheState, general.cacheState)
        XCTAssertEqual(fallback.offsets, general.offsets)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testQuantizedWeightsCompileAndMatchExactly() throws {
        MLXRandom.seed(39)
        let model = Qwen3NextModel(
            try configuration(headDim: 32, quantizable: true))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        quantize(model: model, groupSize: 32, bits: 4)
        let tokens = [Int32(3), 8, 1, 5]

        let general = try run(model, tokens: tokens, compiled: false)
        let compiled = try run(model, tokens: tokens, compiled: true)

        XCTAssertEqual(compiled.logits, general.logits)
        XCTAssertEqual(compiled.cacheState, general.cacheState)
        XCTAssertEqual(compiled.offsets, general.offsets)
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
    }

    func testModelDeallocatesAfterCompiledDecode() throws {
        MLXRandom.seed(41)
        var model: Qwen3NextModel? = Qwen3NextModel(try configuration())
        model!.update(parameters: model!.parameters().mapValues { $0.asType(.float16) })
        var cache: [KVCache]? = try model!.newCache(parameters: nil)

        for token in [Int32(1), 2, 3] {
            let output = model!(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(output)
        }

        weak let inner = model!.model
        weak let linearAttention = model!.modules()
            .compactMap { $0 as? Qwen3NextGatedDeltaNet }.first
        XCTAssertNotNil(inner)
        XCTAssertNotNil(linearAttention)
        XCTAssertGreaterThan(model!.model.compiledDecodeSegmentCount, 0)

        model = nil
        cache = nil

        XCTAssertNil(inner, "compiled segments retained the model")
        XCTAssertNil(linearAttention, "compiled segments retained their linear-attention layer")
    }

}
