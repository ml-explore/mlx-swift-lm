// Copyright © 2026 Apple Inc.
//
// Regression tests for the compiled-decode closures: each is stored on the
// module that owns it while capturing that module, so a strong capture
// cycles (module → closure → CompiledFunction → closure → module) and keeps
// the blocks — weights and compiled mlx tape included — alive after the
// model is released. The captures are `unowned`; these tests pin the
// lifecycle: after compiled decode steps, releasing the model must
// deallocate the blocks.
//
// Coverage spans all three kinds of closures: the plain-cache decode
// installs the per-layer traces (first token, no GDN state yet) and the
// whole-step segment traces (second token); the quantized-cache decode
// forces the fallback that installs the MoE block's own compiled closure —
// the one the original regression was filed against.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35CompiledDecodeLifecycleTests: XCTestCase {

    /// Hybrid layout: layer 0 GDN (linear), layer 1 full attention, MoE mlp
    /// in both — the two block types that lazily install compiled decode
    /// closures. Every omitted field has a config default. `headDim` is
    /// parameterized because the quantized-cache variant needs a head dim
    /// the KV quantizer's group size divides.
    private func tinyMoEConfiguration(headDim: Int = 8) throws -> Qwen35TextConfiguration {
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

    func testBlocksDeallocateAfterCompiledDecode() throws {
        var model: Qwen35TextModel? = Qwen35TextModel(try tinyMoEConfiguration())
        var cache: [KVCache]? = model!.newCache(parameters: nil)

        // Two S==1 steps: the first installs the compiled closures (and
        // exercises the explicit zero-state first-token leg of the GDN
        // trace), the second replays them.
        for token in [Int32(1), Int32(2)] {
            let logits = model!(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
        }

        weak var gdn = model!.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first
        weak var moe = model!.modules().compactMap { $0 as? Qwen35SparseMoeBlock }.first
        XCTAssertNotNil(gdn, "expected a GDN layer in the tiny hybrid config")
        XCTAssertNotNil(moe, "expected a MoE mlp in the tiny hybrid config")

        model = nil
        cache = nil

        XCTAssertNil(
            gdn,
            "Qwen35GatedDeltaNet leaked after model release — the compiled "
                + "decode closure must not retain its module")
        XCTAssertNil(
            moe,
            "Qwen35SparseMoeBlock leaked after model release — the compiled "
                + "decode closure must not retain its module")
    }

    func testBlocksDeallocateAfterQuantizedCacheDecode() throws {
        // A quantized KV cache makes the whole-step schedule and the
        // per-layer attention traces bail, so the FA layer runs the general
        // body and its MoE mlp installs the block's own compiled closure.
        // GDN layers still take the per-layer trace.
        var model: Qwen35TextModel? = Qwen35TextModel(
            try tinyMoEConfiguration(headDim: 32))
        var cache: [KVCache]? = model!.newCache(parameters: nil).map { c in
            c is MambaCache ? c : QuantizedKVCache(groupSize: 32, bits: 8)
        }

        for token in [Int32(1), Int32(2)] {
            let logits = model!(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
        }

        weak var gdn = model!.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first
        weak var moe = model!.modules().compactMap { $0 as? Qwen35SparseMoeBlock }.first
        XCTAssertNotNil(gdn, "expected a GDN layer in the tiny hybrid config")
        XCTAssertNotNil(moe, "expected a MoE mlp in the tiny hybrid config")

        model = nil
        cache = nil

        XCTAssertNil(
            gdn,
            "Qwen35GatedDeltaNet leaked after quantized-cache decode — the "
                + "compiled decode closure must not retain its module")
        XCTAssertNil(
            moe,
            "Qwen35SparseMoeBlock leaked after quantized-cache decode — the "
                + "block's compiled closure must not retain its module")
    }
}
