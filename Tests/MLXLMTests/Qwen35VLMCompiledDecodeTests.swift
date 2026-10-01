// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class Qwen35VLMCompiledDecodeTests: XCTestCase {
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
        let outputs: [Snapshot]
        let cacheState: [Snapshot]
        let offsets: [Int]
    }

    private let tokens: [Int32] = [1, 7, 3, 9, 2]
    private let positions: [[Int32]] = [
        [0, 0, 0],
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 8],
        [4, 7, 9],
    ]

    private func configuration(numExperts: Int) throws -> Qwen35Configuration.TextConfiguration {
        let modelType = numExperts > 0 ? "qwen3_5_moe" : "qwen3_5"
        let json = """
            {
                "model_type": "\(modelType)",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": \(numExperts),
                "num_experts_per_tok": \(min(2, numExperts)),
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeModel(
        numExperts: Int, seed: UInt64, dtype: DType = .float16
    ) throws -> Qwen35Language.Model {
        let configuration = try configuration(numExperts: numExperts)
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35Language.Model(configuration)
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            model.train(false)
            return model
        }
    }

    private func plainCache() -> [KVCache?] {
        [MambaCache(), KVCacheSimple()]
    }

    private func quantizedCache() -> [KVCache?] {
        [MambaCache(), QuantizedKVCache(groupSize: 32, bits: 8)]
    }

    private func rotatingCache() -> [KVCache?] {
        [MambaCache(), RotatingKVCache(maxSize: 4)]
    }

    private func turboQuantCache() -> [KVCache?] {
        [MambaCache(), TurboQuantKVCache(bits: 4, seed: 67)]
    }

    private func run(
        _ model: Qwen35Language.Model,
        compiled: Bool,
        cache makeCache: () -> [KVCache?]
    ) -> RunResult {
        precondition(tokens.count == positions.count)
        let cache = makeCache()
        var outputs: [Snapshot] = []

        for (token, axes) in zip(tokens, positions) {
            let input = MLXArray([token]).reshaped(1, 1)
            let positionIds = MLXArray(axes).reshaped(3, 1, 1)
            eval(input, positionIds)
            let output = model.forward(
                input,
                cache: cache,
                positionIds: positionIds,
                useCompiledDecode: compiled)
            eval(output)
            outputs.append(Snapshot(output))
        }

        return RunResult(
            outputs: outputs,
            cacheState: cache.compactMap { $0 }.flatMap { $0.innerState() }.map(Snapshot.init),
            offsets: cache.map { $0?.offset ?? -1 })
    }

    private func runTurboQuantAfterPrefill(
        _ model: Qwen35Language.Model,
        compiled: Bool
    ) -> RunResult {
        let cache = turboQuantCache()
        let prefillCount = 2
        var outputs: [Snapshot] = []

        let prefill = MLXArray(Array(tokens.prefix(prefillCount))).reshaped(1, prefillCount)
        let prefillPositions = MLXArray(
            (0 ..< 3).flatMap { axis in
                positions.prefix(prefillCount).map { $0[axis] }
            }
        ).reshaped(3, 1, prefillCount)
        eval(prefill, prefillPositions)
        let prefillOutput = model.forward(
            prefill,
            cache: cache,
            positionIds: prefillPositions,
            useCompiledDecode: false)
        eval(prefillOutput)
        outputs.append(Snapshot(prefillOutput))

        for index in prefillCount ..< tokens.count {
            let input = MLXArray([tokens[index]]).reshaped(1, 1)
            let positionIds = MLXArray(positions[index]).reshaped(3, 1, 1)
            eval(input, positionIds)
            let output = model.forward(
                input,
                cache: cache,
                positionIds: positionIds,
                useCompiledDecode: compiled)
            eval(output)
            outputs.append(Snapshot(output))
        }

        return RunResult(
            outputs: outputs,
            cacheState: cache.compactMap { $0 }.flatMap { $0.innerState() }.map(Snapshot.init),
            offsets: cache.map { $0?.offset ?? -1 })
    }

    private func assertEqual(
        _ compiled: RunResult,
        _ eager: RunResult,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(compiled.outputs, eager.outputs, "outputs differ", file: file, line: line)
        XCTAssertEqual(
            compiled.cacheState, eager.cacheState, "cache state differs", file: file, line: line)
        XCTAssertEqual(
            compiled.offsets, eager.offsets, "cache offsets differ", file: file, line: line)
    }

    func testDensePlainCacheMatchesEagerExactlyWithDynamicMRoPE() throws {
        let model = try makeModel(numExperts: 0, seed: 31)

        let eager = run(model, compiled: false, cache: plainCache)
        let compiled = run(model, compiled: true, cache: plainCache)

        assertEqual(compiled, eager)
        XCTAssertEqual(compiled.offsets, [0, 5])
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)
    }

    func testMoEPlainCacheMatchesEagerExactly() throws {
        let model = try makeModel(numExperts: 4, seed: 37)

        let eager = run(model, compiled: false, cache: plainCache)
        let compiled = run(model, compiled: true, cache: plainCache)

        assertEqual(compiled, eager)
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)
    }

    func testFirstEmptyCacheTokenStaysEager() throws {
        let model = try makeModel(numExperts: 0, seed: 41)
        let cache = plainCache()

        let first = model.forward(
            MLXArray([tokens[0]]).reshaped(1, 1),
            cache: cache,
            positionIds: MLXArray(positions[0]).reshaped(3, 1, 1),
            useCompiledDecode: true)
        eval(first)
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)

        let second = model.forward(
            MLXArray([tokens[1]]).reshaped(1, 1),
            cache: cache,
            positionIds: MLXArray(positions[1]).reshaped(3, 1, 1),
            useCompiledDecode: true)
        eval(second)
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)
    }

    func testQuantizedCacheUsesExactPerLayerFallback() throws {
        let model = try makeModel(numExperts: 4, seed: 43)

        let eager = run(model, compiled: false, cache: quantizedCache)
        let fallback = run(model, compiled: true, cache: quantizedCache)

        assertEqual(fallback, eager)
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)
        XCTAssertGreaterThan(model.compiledGDNTraceCount, 0)
        XCTAssertGreaterThan(model.compiledMoETraceCount, 0)
    }

    func testWrappedRotatingCacheMatchesEagerExactly() throws {
        let model = try makeModel(numExperts: 4, seed: 71)

        let eager = run(model, compiled: false, cache: rotatingCache)
        let compiled = run(model, compiled: true, cache: rotatingCache)

        assertEqual(compiled, eager)
        XCTAssertEqual(compiled.offsets, [0, 5])
        XCTAssertEqual(compiled.cacheState.suffix(2).map { $0.shape[2] }, [4, 4])
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)
    }

    func testTurboQuantCacheUsesExactPerLayerFallback() throws {
        let model = try makeModel(numExperts: 4, seed: 73)

        let eager = runTurboQuantAfterPrefill(model, compiled: false)
        let fallback = runTurboQuantAfterPrefill(model, compiled: true)

        assertEqual(fallback, eager)
        XCTAssertEqual(fallback.offsets, [0, 5])
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)
        XCTAssertGreaterThan(model.compiledGDNTraceCount, 0)
        XCTAssertGreaterThan(model.compiledMoETraceCount, 0)
    }

    func testBFloat16UsesExactCompiledRoutes() throws {
        let segmentModel = try makeModel(numExperts: 4, seed: 79, dtype: .bfloat16)
        let eagerSegment = run(segmentModel, compiled: false, cache: plainCache)
        let compiledSegment = run(segmentModel, compiled: true, cache: plainCache)

        assertEqual(compiledSegment, eagerSegment)
        XCTAssertGreaterThan(segmentModel.compiledDecodeSegmentCount, 0)

        let fallbackModel = try makeModel(numExperts: 4, seed: 83, dtype: .bfloat16)
        let eagerFallback = run(fallbackModel, compiled: false, cache: quantizedCache)
        let compiledFallback = run(fallbackModel, compiled: true, cache: quantizedCache)

        assertEqual(compiledFallback, eagerFallback)
        XCTAssertEqual(fallbackModel.compiledDecodeSegmentCount, 0)
        XCTAssertGreaterThan(fallbackModel.compiledGDNTraceCount, 0)
        XCTAssertGreaterThan(fallbackModel.compiledMoETraceCount, 0)
    }

    func testFloat32StaysOnEagerPathForAllCacheRoutes() throws {
        let segmentModel = try makeModel(numExperts: 4, seed: 89, dtype: .float32)
        let eagerSegment = run(segmentModel, compiled: false, cache: plainCache)
        let fallbackSegment = run(segmentModel, compiled: true, cache: plainCache)

        assertEqual(fallbackSegment, eagerSegment)
        XCTAssertEqual(segmentModel.compiledDecodeSegmentCount, 0)
        XCTAssertEqual(segmentModel.compiledGDNTraceCount, 0)
        XCTAssertEqual(segmentModel.compiledMoETraceCount, 0)

        let layerModel = try makeModel(numExperts: 4, seed: 97, dtype: .float32)
        let eagerLayer = run(layerModel, compiled: false, cache: quantizedCache)
        let fallbackLayer = run(layerModel, compiled: true, cache: quantizedCache)

        assertEqual(fallbackLayer, eagerLayer)
        XCTAssertEqual(layerModel.compiledDecodeSegmentCount, 0)
        XCTAssertEqual(layerModel.compiledGDNTraceCount, 0)
        XCTAssertEqual(layerModel.compiledMoETraceCount, 0)
    }

    func testTrainingGDNStaysOnEagerPath() throws {
        let model = try makeModel(numExperts: 4, seed: 101)
        let gdn = try XCTUnwrap(
            model.modules().compactMap { $0 as? Qwen35Language.GatedDeltaNet }.first)
        gdn.train(true)

        let eager = run(model, compiled: false, cache: plainCache)
        let fallback = run(model, compiled: true, cache: plainCache)

        assertEqual(fallback, eager)
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)
        XCTAssertEqual(model.compiledGDNTraceCount, 0)
        XCTAssertEqual(model.compiledMoETraceCount, 0)
    }

    private func assertWeightUpdatesAreVisible(
        cache makeCache: () -> [KVCache?],
        usesSegments: Bool,
        seed: UInt64,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let warm = try makeModel(numExperts: 4, seed: seed)
        let reference = try makeModel(numExperts: 4, seed: seed + 1)
        reference.update(parameters: warm.parameters())

        let baseline = run(warm, compiled: true, cache: makeCache)
        if usesSegments {
            XCTAssertGreaterThan(warm.compiledDecodeSegmentCount, 0, file: file, line: line)
        } else {
            XCTAssertEqual(warm.compiledDecodeSegmentCount, 0, file: file, line: line)
            XCTAssertGreaterThan(warm.compiledGDNTraceCount, 0, file: file, line: line)
            XCTAssertGreaterThan(warm.compiledMoETraceCount, 0, file: file, line: line)
        }

        let updatedParameters = warm.parameters().mapValues { $0 * 1.5 }
        warm.update(parameters: updatedParameters)
        reference.update(parameters: updatedParameters)

        let updated = run(warm, compiled: true, cache: makeCache)
        let expected = run(reference, compiled: true, cache: makeCache)

        XCTAssertNotEqual(updated.outputs, baseline.outputs, file: file, line: line)
        assertEqual(updated, expected, file: file, line: line)
    }

    func testWholeStepSegmentsObserveParameterUpdates() throws {
        try assertWeightUpdatesAreVisible(
            cache: plainCache, usesSegments: true, seed: 47)
    }

    func testQuantizedFallbackTracesObserveParameterUpdates() throws {
        try assertWeightUpdatesAreVisible(
            cache: quantizedCache, usesSegments: false, seed: 53)
    }

    private func assertModelDeallocates(
        cache makeCache: () -> [KVCache?],
        usesSegments: Bool,
        seed: UInt64,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        var model: Qwen35Language.Model? = try makeModel(numExperts: 4, seed: seed)
        var cache: [KVCache?]? = makeCache()

        for (token, axes) in zip(tokens.prefix(3), positions.prefix(3)) {
            let output = model!.forward(
                MLXArray([token]).reshaped(1, 1),
                cache: cache,
                positionIds: MLXArray(axes).reshaped(3, 1, 1),
                useCompiledDecode: true)
            eval(output)
        }

        if usesSegments {
            XCTAssertGreaterThan(model!.compiledDecodeSegmentCount, 0, file: file, line: line)
        } else {
            XCTAssertGreaterThan(model!.compiledGDNTraceCount, 0, file: file, line: line)
            XCTAssertGreaterThan(model!.compiledMoETraceCount, 0, file: file, line: line)
        }

        weak let inner = model
        weak let gdn = model!.modules()
            .compactMap { $0 as? Qwen35Language.GatedDeltaNet }.first
        weak let moe = model!.modules()
            .compactMap { $0 as? Qwen35Language.SparseMoeBlock }.first
        XCTAssertNotNil(gdn, "expected a GDN layer", file: file, line: line)
        XCTAssertNotNil(moe, "expected a MoE block", file: file, line: line)

        model = nil
        cache = nil

        XCTAssertNil(inner, "compiled traces retained the language model", file: file, line: line)
        XCTAssertNil(gdn, "compiled traces retained the GDN block", file: file, line: line)
        XCTAssertNil(moe, "compiled traces retained the MoE block", file: file, line: line)
    }

    func testModelDeallocatesAfterWholeStepCompiledDecode() throws {
        try assertModelDeallocates(cache: plainCache, usesSegments: true, seed: 59)
    }

    func testModelDeallocatesAfterQuantizedFallbackDecode() throws {
        try assertModelDeallocates(cache: quantizedCache, usesSegments: false, seed: 61)
    }
}
