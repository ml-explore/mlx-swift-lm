// Copyright © 2026 Apple Inc.
//
// The compiled-decode closures are stored on the module they capture, so a
// strong capture would cycle and keep the module — weights and compiled tape
// included — alive after the model is released. These tests pin the
// lifecycle: after compiled decode steps, releasing the model must
// deallocate the blocks.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35CompiledDecodeLifecycleTests: XCTestCase {

    /// Layer 0 GDN, layer 1 full attention, MoE mlp in both — the two block
    /// kinds that install compiled decode closures. `headDim` is a parameter
    /// because the quantized-cache variant needs one the KV group size
    /// divides.
    private func tinyMoEConfiguration(headDim: Int) throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": \(headDim),
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func assertCacheOffsets(
        _ cache: [KVCache], equal expected: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertTrue(
            cache.allSatisfy { $0.offset == expected },
            "expected every cache offset to equal \(expected), got \(cache.map(\.offset))",
            file: file,
            line: line
        )
    }

    /// Runs two decode steps (the first installs the compiled closures, the
    /// second replays them), releases the model, and asserts the
    /// closure-owning blocks deallocate.
    private func assertBlocksDeallocate(
        headDim: Int,
        mapCache: ([KVCache]) -> [KVCache] = { $0 },
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        var model: Qwen35TextModel? = Qwen35TextModel(try tinyMoEConfiguration(headDim: headDim))
        var cache: [KVCache]? = mapCache(model!.newCache(parameters: nil))

        for token in [Int32(1), Int32(2)] {
            let logits = model!(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
        }

        weak var gdn = model!.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first
        weak var moe = model!.modules().compactMap { $0 as? Qwen35SparseMoeBlock }.first
        XCTAssertNotNil(gdn, "expected a GDN layer in the config", file: file, line: line)
        XCTAssertNotNil(moe, "expected a MoE mlp in the config", file: file, line: line)

        model = nil
        cache = nil

        XCTAssertNil(gdn, "Qwen35GatedDeltaNet leaked after model release", file: file, line: line)
        XCTAssertNil(moe, "Qwen35SparseMoeBlock leaked after model release", file: file, line: line)
    }

    func testBlocksDeallocateAfterCompiledDecode() throws {
        try assertBlocksDeallocate(headDim: 8)
    }

    func testBlocksDeallocateAfterQuantizedCacheDecode() throws {
        // A quantized KV cache forces the fallback that installs the MoE
        // block's own compiled closure; GDN layers still take per-layer
        // traces.
        try assertBlocksDeallocate(headDim: 32) { caches in
            caches.map { $0 is MambaCache ? $0 : QuantizedKVCache(groupSize: 32, bits: 8) }
        }
    }

    func testCacheOffsetsAdvanceAcrossPrefillAndWholeStepDecode() throws {
        let model = Qwen35TextModel(try tinyMoEConfiguration(headDim: 8))
        let cache = model.newCache(parameters: nil)
        let prompt = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        eval(model(prompt, cache: cache))
        assertCacheOffsets(cache, equal: 4)

        let nextToken = MLXArray([Int32(5)]).reshaped(1, 1)
        eval(model(nextToken, cache: cache))
        assertCacheOffsets(cache, equal: 5)
    }

    func testCacheOffsetsAdvanceAcrossFallbackDecode() throws {
        let model = Qwen35TextModel(try tinyMoEConfiguration(headDim: 32))
        let cache = model.newCache(parameters: nil).map { cache in
            cache is MambaCache
                ? cache
                : QuantizedKVCache(groupSize: 32, bits: 8)
        }

        // Quantized attention caches make the whole-step decode schedule
        // ineligible. This exercises Qwen35DecoderLayer.decodeLinearLayer.
        for (token, expectedOffset) in [(Int32(1), 1), (Int32(2), 2)] {
            eval(model(MLXArray([token]).reshaped(1, 1), cache: cache))
            assertCacheOffsets(cache, equal: expectedOffset)
        }
    }
}
