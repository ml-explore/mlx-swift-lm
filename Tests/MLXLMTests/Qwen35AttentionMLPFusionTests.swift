// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class Qwen35AttentionMLPFusionTests: XCTestCase {

    private func configurationJSON(numExperts: Int = 0) -> String {
        let modelType = numExperts > 0 ? "qwen3_5_moe_text" : "qwen3_5_text"
        return """
            {
                "model_type": "\(modelType)",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": \(numExperts),
                "num_experts_per_tok": \(min(2, numExperts)),
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
            """
    }

    private func llmConfiguration(numExperts: Int = 0) throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(configurationJSON(numExperts: numExperts).utf8))
    }

    private func vlmConfiguration(
        numExperts: Int = 0
    ) throws -> MLXVLM.Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(configurationJSON(numExperts: numExperts).utf8))
    }

    private func assertBitIdentical(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape", file: file, line: line)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        let mismatches = zip(actualValues, expectedValues).lazy.filter {
            $0.bitPattern != $1.bitPattern
        }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(actualValues.count) values differ",
            file: file, line: line)
    }

    private func quantize(_ attn: Qwen35Attention) throws {
        try attn.update(
            modules: ModuleChildren(values: [
                "q_proj": .value(QuantizedLinear(attn.qProj, groupSize: 32, bits: 4)),
                "k_proj": .value(QuantizedLinear(attn.kProj, groupSize: 32, bits: 4)),
                "v_proj": .value(QuantizedLinear(attn.vProj, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    private func quantize(_ attn: Qwen35Language.Attention) throws {
        try attn.update(
            modules: ModuleChildren(values: [
                "q_proj": .value(QuantizedLinear(attn.qProj, groupSize: 32, bits: 4)),
                "k_proj": .value(QuantizedLinear(attn.kProj, groupSize: 32, bits: 4)),
                "v_proj": .value(QuantizedLinear(attn.vProj, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    private func quantize(_ mlp: Qwen35MLP) throws {
        try mlp.update(
            modules: ModuleChildren(values: [
                "gate_proj": .value(QuantizedLinear(mlp.gateProj, groupSize: 32, bits: 4)),
                "up_proj": .value(QuantizedLinear(mlp.upProj, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    private func quantize(_ mlp: Qwen35Language.MLP) throws {
        try mlp.update(
            modules: ModuleChildren(values: [
                "gate_proj": .value(QuantizedLinear(mlp.gateProj, groupSize: 32, bits: 4)),
                "up_proj": .value(QuantizedLinear(mlp.upProj, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    func testLLMFusedQKVAndMLPProjectionsAreBitIdentical() throws {
        XCTAssertTrue(qwen35FourGDNEnabled)
        for (batch, sequence) in [(1, 1), (2, 7)] {
            let attn = Qwen35Attention(try llmConfiguration())
            try quantize(attn)
            let input = MLXRandom.normal([batch, sequence, 64]).asType(.bfloat16)

            XCTAssertFalse(try attn.prepareFusedQKVProjection(enabled: false))
            XCTAssertFalse(attn.hasFusedQKVProjection)

            let mask: MLXFast.ScaledDotProductAttentionMaskMode = sequence > 1 ? .causal : .none
            let refOut = attn(input, mask: mask, cache: KVCacheSimple())
            eval(refOut)

            XCTAssertTrue(try attn.prepareFusedQKVProjection())
            XCTAssertTrue(attn.hasFusedQKVProjection)
            let fusedOut = attn(input, mask: mask, cache: KVCacheSimple())
            eval(fusedOut)
            assertBitIdentical(fusedOut, refOut, "LLM QKV B\(batch) S\(sequence)")

            let mlp = Qwen35MLP(dimensions: 64, hiddenDimensions: 128)
            try quantize(mlp)
            XCTAssertFalse(try mlp.prepareFusedGateUpProjection(enabled: false))
            let refMLP = mlp(input)
            eval(refMLP)

            XCTAssertTrue(try mlp.prepareFusedGateUpProjection())
            XCTAssertTrue(mlp.hasFusedGateUpProjection)
            let fusedMLP = mlp(input)
            eval(fusedMLP)
            assertBitIdentical(fusedMLP, refMLP, "LLM MLP B\(batch) S\(sequence)")
        }

        let moe = Qwen35SparseMoeBlock(try llmConfiguration(numExperts: 4))
        try quantize(moe.sharedExpert)
        let moeInput = MLXRandom.normal([1, 3, 64]).asType(.bfloat16)
        let refMoE = moe(moeInput)
        eval(refMoE)
        XCTAssertTrue(try moe.sharedExpert.prepareFusedGateUpProjection())
        let fusedMoE = moe(moeInput)
        eval(fusedMoE)
        assertBitIdentical(fusedMoE, refMoE, "LLM MoE sharedExpert")
    }

    func testVLMFusedQKVAndMLPProjectionsAreBitIdentical() throws {
        XCTAssertTrue(qwen35FourGDNEnabled)
        for (batch, sequence) in [(1, 1), (1, 9)] {
            let attn = Qwen35Language.Attention(try vlmConfiguration())
            try quantize(attn)
            let input = MLXRandom.normal([batch, sequence, 64]).asType(.bfloat16)
            let positionIds = broadcast(
                MLXArray(0 ..< Int32(sequence))[.newAxis, .newAxis, 0...],
                to: [3, batch, sequence])
            let maskMode = createAttentionMask(
                h: input, cache: nil as KVCache?, returnArray: true)
            let mask: MLXArray? =
                if case .array(let arrayMask) = maskMode { arrayMask } else { nil }

            XCTAssertFalse(try attn.prepareFusedQKVProjection(enabled: false))
            let refOut = attn(input, mask: mask, cache: KVCacheSimple(), positionIds: positionIds)
            eval(refOut)

            XCTAssertTrue(try attn.prepareFusedQKVProjection())
            XCTAssertTrue(attn.hasFusedQKVProjection)
            let fusedOut = attn(input, mask: mask, cache: KVCacheSimple(), positionIds: positionIds)
            eval(fusedOut)
            assertBitIdentical(fusedOut, refOut, "VLM QKV B\(batch) S\(sequence)")

            let mlp = Qwen35Language.MLP(dimensions: 64, hiddenDimensions: 128)
            try quantize(mlp)
            let refMLP = mlp(input)
            eval(refMLP)

            XCTAssertTrue(try mlp.prepareFusedGateUpProjection())
            XCTAssertTrue(mlp.hasFusedGateUpProjection)
            let fusedMLP = mlp(input)
            eval(fusedMLP)
            assertBitIdentical(fusedMLP, refMLP, "VLM MLP B\(batch) S\(sequence)")
        }

        let moe = Qwen35Language.SparseMoeBlock(try vlmConfiguration(numExperts: 4))
        try quantize(moe.sharedExpert)
        let moeInput = MLXRandom.normal([1, 3, 64]).asType(.bfloat16)
        let refMoE = moe(moeInput)
        eval(refMoE)
        XCTAssertTrue(try moe.sharedExpert.prepareFusedGateUpProjection())
        let fusedMoE = moe(moeInput)
        eval(fusedMoE)
        assertBitIdentical(fusedMoE, refMoE, "VLM MoE sharedExpert")
    }

    func testCheckpointTopologyIsPreservedForAttentionAndMLP() throws {
        let attn = Qwen35Attention(try llmConfiguration())
        try quantize(attn)
        let attnKeysBefore = Set(attn.parameters().flattened().map(\.0))
        XCTAssertTrue(try attn.prepareFusedQKVProjection())
        let attnKeysAfter = Set(attn.parameters().flattened().map(\.0))
        XCTAssertEqual(attnKeysAfter, attnKeysBefore)
        XCTAssertTrue(attnKeysAfter.contains("q_proj.weight"))
        XCTAssertTrue(attnKeysAfter.contains("k_proj.weight"))
        XCTAssertTrue(attnKeysAfter.contains("v_proj.weight"))
        XCTAssertFalse(attnKeysAfter.contains { $0.contains("fused") })

        let mlp = Qwen35MLP(dimensions: 64, hiddenDimensions: 128)
        try quantize(mlp)
        let mlpKeysBefore = Set(mlp.parameters().flattened().map(\.0))
        XCTAssertTrue(try mlp.prepareFusedGateUpProjection())
        let mlpKeysAfter = Set(mlp.parameters().flattened().map(\.0))
        XCTAssertEqual(mlpKeysAfter, mlpKeysBefore)
        XCTAssertTrue(mlpKeysAfter.contains("gate_proj.weight"))
        XCTAssertTrue(mlpKeysAfter.contains("up_proj.weight"))
        XCTAssertFalse(mlpKeysAfter.contains { $0.contains("fused") })
    }

    func testPrepareLifecycleCoversDenseAndMoESharedExpert() throws {
        let denseLLM = Qwen35TextModel(try llmConfiguration(numExperts: 0))
        for attn in denseLLM.modules().compactMap({ $0 as? Qwen35Attention }) {
            try quantize(attn)
        }
        for mlp in denseLLM.modules().compactMap({ $0 as? Qwen35MLP }) {
            try quantize(mlp)
        }
        try denseLLM.prepare()
        XCTAssertTrue(
            denseLLM.modules().compactMap { $0 as? Qwen35Attention }
                .allSatisfy(\.hasFusedQKVProjection))
        XCTAssertTrue(
            denseLLM.modules().compactMap { $0 as? Qwen35MLP }
                .allSatisfy(\.hasFusedGateUpProjection))

        let moeLLM = Qwen35TextModel(try llmConfiguration(numExperts: 4))
        let llmSharedExperts = moeLLM.modules().compactMap { $0 as? Qwen35SparseMoeBlock }
            .map(\.sharedExpert)
        XCTAssertFalse(llmSharedExperts.isEmpty)
        for expert in llmSharedExperts {
            try quantize(expert)
        }
        try moeLLM.prepare()
        XCTAssertTrue(llmSharedExperts.allSatisfy(\.hasFusedGateUpProjection))

        let moeVLM = Qwen35Language.Model(try vlmConfiguration(numExperts: 4))
        let vlmSharedExperts = moeVLM.modules().compactMap { $0 as? Qwen35Language.SparseMoeBlock }
            .map(\.sharedExpert)
        XCTAssertFalse(vlmSharedExperts.isEmpty)
        for expert in vlmSharedExperts {
            try quantize(expert)
            XCTAssertTrue(try expert.prepareFusedGateUpProjection())
        }
        XCTAssertTrue(vlmSharedExperts.allSatisfy(\.hasFusedGateUpProjection))
    }

    func testLLMParameterAndModuleUpdatesInvalidateCompiledTraces() throws {
        let config = try llmConfiguration(numExperts: 0)
        let (warm, reference) = withRandomState(MLXRandom.RandomState(seed: 91)) {
            let warm = Qwen35TextModel(config)
            let reference = Qwen35TextModel(config)
            warm.update(parameters: warm.parameters().mapValues { $0.asType(.float16) })
            reference.update(parameters: warm.parameters())
            return (warm, reference)
        }

        for model in [warm, reference] {
            for attn in model.modules().compactMap({ $0 as? Qwen35Attention }) {
                try quantize(attn)
            }
            for mlp in model.modules().compactMap({ $0 as? Qwen35MLP }) {
                try quantize(mlp)
            }
            try model.prepare()
        }

        func decode(_ model: Qwen35TextModel) throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            return [Int32(1), 2, 3].map { token in
                let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                eval(logits)
                return logits
            }
        }

        _ = try decode(warm)
        XCTAssertGreaterThan(warm.model.compiledDecodeSegmentCount, 0)

        let parameters = warm.parameters().flattened()
        let (qScaleKey, qScale) = try XCTUnwrap(
            parameters.first { $0.0.hasSuffix("q_proj.scales") })
        let (gateScaleKey, gateScale) = try XCTUnwrap(
            parameters.first { $0.0.hasSuffix("gate_proj.scales") })
        let update = ModuleParameters.unflattened([
            qScaleKey: qScale * 1.25,
            gateScaleKey: gateScale * 0.75,
        ])
        warm.update(parameters: update)
        reference.update(parameters: update)

        XCTAssertEqual(warm.model.compiledDecodeSegmentCount, 0)
        let actual = try decode(warm)
        let expected = try decode(reference)
        for (idx, (a, e)) in zip(actual, expected).enumerated() {
            assertBitIdentical(a, e, "updated LLM decode step \(idx)")
        }

        // Replacing a projected child via update(modules:) also invalidates fusion and traces.
        try warm.prepare()
        _ = try decode(warm)
        XCTAssertGreaterThan(warm.model.compiledDecodeSegmentCount, 0)
        let firstAttn = try XCTUnwrap(warm.modules().compactMap { $0 as? Qwen35Attention }.first)
        let adapted = try XCTUnwrap(
            LoRALinear.from(linear: firstAttn.qProj, rank: 4, scale: 1) as? Linear)
        try firstAttn.update(
            modules: ModuleChildren(values: ["q_proj": .value(adapted)]), verify: [])
        XCTAssertFalse(firstAttn.hasFusedQKVProjection)
        XCTAssertEqual(warm.model.compiledDecodeSegmentCount, 0)
    }

    func testVLMCompiledDecodeInvalidatesOnAttentionAndMLPUpdates() throws {
        let config = try vlmConfiguration(numExperts: 4)
        let (warm, reference) = withRandomState(MLXRandom.RandomState(seed: 101)) {
            let warm = Qwen35Language.Model(config)
            let reference = Qwen35Language.Model(config)
            warm.update(parameters: warm.parameters().mapValues { $0.asType(.float16) })
            reference.update(parameters: warm.parameters())
            return (warm, reference)
        }

        for model in [warm, reference] {
            for attn in model.modules().compactMap({ $0 as? Qwen35Language.Attention }) {
                try quantize(attn)
                XCTAssertTrue(try attn.prepareFusedQKVProjection())
            }
            for moe in model.modules().compactMap({ $0 as? Qwen35Language.SparseMoeBlock }) {
                try quantize(moe.sharedExpert)
                XCTAssertTrue(try moe.sharedExpert.prepareFusedGateUpProjection())
            }
            model.train(false)
        }

        func decode(_ model: Qwen35Language.Model, quantizedKV: Bool = false) -> [MLXArray] {
            let cache: [KVCache?] = (0 ..< 4).map { idx in
                if idx.isMultiple(of: 2) { return MambaCache() }
                return quantizedKV ? QuantizedKVCache(groupSize: 32, bits: 8) : KVCacheSimple()
            }
            return [Int32(1), 2, 3].map { token in
                let out = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                eval(out)
                return out
            }
        }

        _ = decode(warm)
        XCTAssertGreaterThan(warm.compiledDecodeSegmentCount, 0)

        let parameters = warm.parameters().flattened()
        let (qScaleKey, qScale) = try XCTUnwrap(
            parameters.first { $0.0.hasSuffix("q_proj.scales") })
        let (gateScaleKey, gateScale) = try XCTUnwrap(
            parameters.first { $0.0.hasSuffix("shared_expert.gate_proj.scales") })
        let update = ModuleParameters.unflattened([
            qScaleKey: qScale * 1.25,
            gateScaleKey: gateScale * 0.75,
        ])
        warm.update(parameters: update)
        reference.update(parameters: update)

        XCTAssertEqual(warm.compiledDecodeSegmentCount, 0)
        let actual = decode(warm)
        let expected = decode(reference)
        for (idx, (a, e)) in zip(actual, expected).enumerated() {
            assertBitIdentical(a, e, "updated VLM decode step \(idx)")
        }

        // Quantized KV fallback compiles GDN layer traces on linear layers (0, 2)
        // and SparseMoeBlock.compiledForward on full-attention layers (1, 3).
        for moe in warm.modules().compactMap({ $0 as? Qwen35Language.SparseMoeBlock }) {
            XCTAssertTrue(try moe.sharedExpert.prepareFusedGateUpProjection())
        }
        _ = decode(warm, quantizedKV: true)
        XCTAssertEqual(warm.compiledGDNTraceCount, 2)
        XCTAssertEqual(warm.compiledMoETraceCount, 2)

        warm.update(parameters: ModuleParameters.unflattened([gateScaleKey: gateScale * 0.5]))
        XCTAssertEqual(warm.compiledGDNTraceCount, 1)
        XCTAssertEqual(warm.compiledMoETraceCount, 2)

        let (faMoEScaleKey, faMoEScale) = try XCTUnwrap(
            parameters.first { $0.0 == "layers.1.mlp.shared_expert.gate_proj.scales" })
        warm.update(parameters: ModuleParameters.unflattened([faMoEScaleKey: faMoEScale * 0.5]))
        XCTAssertEqual(warm.compiledMoETraceCount, 1)
    }

    private func settledActiveMemory() -> Int {
        Stream.gpu.synchronize()
        usleep(150_000)
        Memory.clearCache()
        return Memory.activeMemory
    }

    func testCompiledDecodeReleasesFusedAttentionAndMLPWithModel() throws {
        let baseLLM = settledActiveMemory()
        try autoreleasepool {
            let model = Qwen35TextModel(try llmConfiguration(numExperts: 0))
            for attn in model.modules().compactMap({ $0 as? Qwen35Attention }) {
                try quantize(attn)
            }
            for mlp in model.modules().compactMap({ $0 as? Qwen35MLP }) {
                try quantize(mlp)
            }
            try model.prepare()

            let cache = try model.newCache(parameters: nil)
            for token: Int32 in [1, 2, 3] {
                let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                eval(logits)
            }
            XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
            eval(cache.flatMap(\.state))
        }
        let retainedLLM = settledActiveMemory() - baseLLM
        XCTAssertLessThan(
            retainedLLM, 64, "\(retainedLLM) bytes stayed allocated after Qwen35TextModel dropped")

        let baseVLM = settledActiveMemory()
        try autoreleasepool {
            let model = Qwen35Language.Model(try vlmConfiguration(numExperts: 0))
            model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
            for attn in model.modules().compactMap({ $0 as? Qwen35Language.Attention }) {
                try quantize(attn)
                XCTAssertTrue(try attn.prepareFusedQKVProjection())
            }
            for mlp in model.modules().compactMap({ $0 as? Qwen35Language.MLP }) {
                try quantize(mlp)
                XCTAssertTrue(try mlp.prepareFusedGateUpProjection())
            }
            model.train(false)

            let cache: [KVCache?] = [
                MambaCache(), KVCacheSimple(), MambaCache(), KVCacheSimple(),
            ]
            for token: Int32 in [1, 2, 3] {
                let out = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                eval(out)
            }
            XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)
            eval(cache.compactMap { $0 }.flatMap(\.state))
        }
        let retainedVLM = settledActiveMemory() - baseVLM
        XCTAssertLessThan(
            retainedVLM, 64,
            "\(retainedVLM) bytes stayed allocated after Qwen35Language.Model dropped")
    }
}
