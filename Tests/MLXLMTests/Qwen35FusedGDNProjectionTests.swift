// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class Qwen35FusedGDNProjectionTests: XCTestCase {

    private let configurationJSON = """
        {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
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
            "num_experts": 0,
            "num_experts_per_tok": 0
        }
        """

    private func llmConfiguration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(configurationJSON.utf8))
    }

    private func vlmConfiguration() throws -> MLXVLM.Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(configurationJSON.utf8))
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

    private func quantize(_ layer: Qwen35GatedDeltaNet) throws {
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(
                    QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(
                    QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(
                    QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(
                    QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    private func quantize(_ layer: Qwen35Language.GatedDeltaNet) throws {
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(
                    QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(
                    QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(
                    QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(
                    QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
    }

    func testLLMFusedProjectionIsBitIdenticalForDecodeAndPrefill() throws {
        for (batch, sequence) in [(1, 1), (2, 7)] {
            let layer = Qwen35GatedDeltaNet(try llmConfiguration())
            try quantize(layer)
            let input = MLXRandom.normal([batch, sequence, 64]).asType(.bfloat16)

            layer.fusedInputProjectionEnabled = false
            let reference = layer.projectInputs(input, batch: batch, sequence: sequence)
            eval(reference.qkv, reference.z, reference.b, reference.a)

            layer.fusedInputProjectionEnabled = true
            XCTAssertTrue(try layer.prepareFusedInputProjection())
            let fused = layer.projectInputs(input, batch: batch, sequence: sequence)
            eval(fused.qkv, fused.z, fused.b, fused.a)

            XCTAssertTrue(layer.hasFusedInputProjection)
            assertBitIdentical(fused.qkv, reference.qkv, "qkv B\(batch) S\(sequence)")
            assertBitIdentical(fused.z, reference.z, "z B\(batch) S\(sequence)")
            assertBitIdentical(fused.b, reference.b, "b B\(batch) S\(sequence)")
            assertBitIdentical(fused.a, reference.a, "a B\(batch) S\(sequence)")
        }
    }

    func testLLMFullGDNForwardIsBitIdentical() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let input = MLXRandom.normal([1, 5, 64]).asType(.bfloat16)

        layer.fusedInputProjectionEnabled = false
        let reference = layer(input)
        eval(reference)

        layer.fusedInputProjectionEnabled = true
        XCTAssertTrue(try layer.prepareFusedInputProjection())
        let fused = layer(input)
        eval(fused)

        XCTAssertTrue(layer.hasFusedInputProjection)
        assertBitIdentical(fused, reference, "full GDN")
    }

    func testVLMFusedProjectionIsBitIdentical() throws {
        let layer = Qwen35Language.GatedDeltaNet(try vlmConfiguration())
        try quantize(layer)
        let input = MLXRandom.normal([1, 7, 64]).asType(.bfloat16)

        layer.fusedInputProjectionEnabled = false
        let reference = layer.projectInputs(input, batch: 1, sequence: 7)
        eval(reference.qkv, reference.z, reference.b, reference.a)

        layer.fusedInputProjectionEnabled = true
        XCTAssertTrue(try layer.prepareFusedInputProjection())
        let fused = layer.projectInputs(input, batch: 1, sequence: 7)
        eval(fused.qkv, fused.z, fused.b, fused.a)

        XCTAssertTrue(layer.hasFusedInputProjection)
        assertBitIdentical(fused.qkv, reference.qkv, "VLM qkv")
        assertBitIdentical(fused.z, reference.z, "VLM z")
        assertBitIdentical(fused.b, reference.b, "VLM b")
        assertBitIdentical(fused.a, reference.a, "VLM a")
    }

    func testForwardDoesNotMutateOrLazilyPrepareFusion() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let modulesBefore = [
            ObjectIdentifier(layer.inProjQKV),
            ObjectIdentifier(layer.inProjZ),
            ObjectIdentifier(layer.inProjB),
            ObjectIdentifier(layer.inProjA),
        ]

        let output = layer(MLXRandom.normal([1, 3, 64]).asType(.bfloat16))
        eval(output)

        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertEqual(
            modulesBefore,
            [
                ObjectIdentifier(layer.inProjQKV),
                ObjectIdentifier(layer.inProjZ),
                ObjectIdentifier(layer.inProjB),
                ObjectIdentifier(layer.inProjA),
            ])
    }

    func testVLMForwardDoesNotLazilyPrepareFusion() throws {
        let layer = Qwen35Language.GatedDeltaNet(try vlmConfiguration())
        try quantize(layer)

        let output = layer(MLXRandom.normal([1, 3, 64]).asType(.bfloat16))
        eval(output)

        XCTAssertFalse(layer.hasFusedInputProjection)
    }

    func testLanguageModelLifecyclePreparesNestedGDNModules() throws {
        let model = Qwen35TextModel(try llmConfiguration())
        let layer = try XCTUnwrap(
            model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first)
        try quantize(layer)
        XCTAssertFalse(layer.hasFusedInputProjection)

        try model.prepare()

        XCTAssertTrue(layer.hasFusedInputProjection)
    }

    func testIncompatibleProjectionPoliciesFallBack() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(
                    QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(
                    QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(
                    QuantizedLinear(layer.inProjB, groupSize: 32, bits: 8)),
                "in_proj_a": .value(
                    QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertFalse(try layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
    }

    func testAdapterProjectionFallsBack() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let adapted = try XCTUnwrap(
            LoRALinear.from(linear: layer.inProjQKV, rank: 4, scale: 1) as? Linear)
        try layer.update(
            modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]), verify: [])

        XCTAssertFalse(try layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(layer.inProjQKV is QLoRALinear)
    }

    func testCheckpointTopologyIsPreserved() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let keysBefore = Set(layer.parameters().flattened().map(\.0))

        XCTAssertTrue(try layer.prepareFusedInputProjection())

        let keysAfter = Set(layer.parameters().flattened().map(\.0))
        XCTAssertEqual(keysAfter, keysBefore)
        XCTAssertTrue(keysAfter.contains("in_proj_qkv.weight"))
        XCTAssertTrue(keysAfter.contains("in_proj_z.weight"))
        XCTAssertTrue(keysAfter.contains("in_proj_b.weight"))
        XCTAssertTrue(keysAfter.contains("in_proj_a.weight"))
        XCTAssertFalse(keysAfter.contains { $0.contains("fused") })
    }

    func testParameterAndModuleUpdatesInvalidateFusion() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        XCTAssertTrue(try layer.prepareFusedInputProjection())

        let replacement =
            layer.inProjQKV.weight
            + MLXArray.zeros(
                layer.inProjQKV.weight.shape, dtype: layer.inProjQKV.weight.dtype)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "in_proj_qkv.weight": replacement
            ]), verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(try layer.prepareFusedInputProjection())

        let adapted = try XCTUnwrap(
            LoRALinear.from(linear: layer.inProjQKV, rank: 4, scale: 1) as? Linear)
        try layer.update(
            modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]), verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertFalse(try layer.prepareFusedInputProjection())
    }

    func testThrowingParameterUpdateInvalidatesPublishedFusion() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        XCTAssertTrue(try layer.prepareFusedInputProjection())

        let replacement =
            layer.inProjQKV.weight
            + MLXArray.zeros(
                layer.inProjQKV.weight.shape, dtype: layer.inProjQKV.weight.dtype)
        XCTAssertThrowsError(
            try layer.update(
                parameters: ModuleParameters.unflattened([
                    "in_proj_qkv.weight": replacement,
                    "unknown.weight": MLXArray.zeros([1]),
                ]), verify: .noUnusedKeys))

        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(try layer.prepareFusedInputProjection())
    }

    func testRejectedModuleUpdateInvalidatesPublishedFusion() throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        XCTAssertTrue(try layer.prepareFusedInputProjection())

        XCTAssertThrowsError(
            try layer.updateModule(
                key: "in_proj_qkv",
                Conv1d(inputChannels: 1, outputChannels: 1, kernelSize: 1)))

        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(try layer.prepareFusedInputProjection())
    }

    func testFailedPreparationDoesNotRetryUntilInvalidated() throws {
        enum ExpectedFailure: Error { case install }

        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let linears = [layer.inProjQKV, layer.inProjZ, layer.inProjB, layer.inProjA]
        let cache = FusedQuantizedLinearProjectionCache()
        var installAttempts = 0
        var installedModules: [[Linear]] = []

        XCTAssertThrowsError(
            try cache.prepare(enabled: true, linears: linears) { modules in
                installAttempts += 1
                installedModules.append(modules)
                if installAttempts == 1 {
                    throw ExpectedFailure.install
                }
            }
        ) { error in
            let preparationError = error as? FusedQuantizedLinearPreparationError
            XCTAssertNotNil(preparationError)
            XCTAssertNil(preparationError?.rollbackError)
        }
        XCTAssertEqual(installAttempts, 2, "the second installation restores the originals")
        XCTAssertEqual(
            installedModules[1].map(ObjectIdentifier.init),
            linears.map(ObjectIdentifier.init))

        XCTAssertFalse(
            try cache.prepare(enabled: true, linears: linears) { _ in
                installAttempts += 1
            })
        XCTAssertEqual(installAttempts, 2)

        cache.invalidate()
        XCTAssertTrue(
            try cache.prepare(enabled: true, linears: linears) { _ in
                installAttempts += 1
            })
        XCTAssertEqual(installAttempts, 3)
    }

    func testRollbackFailureIsIncludedInPreparationError() throws {
        enum ExpectedFailure: Error { case install, rollback }

        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let linears = [layer.inProjQKV, layer.inProjZ, layer.inProjB, layer.inProjA]
        let cache = FusedQuantizedLinearProjectionCache()
        var installAttempts = 0

        XCTAssertThrowsError(
            try cache.prepare(enabled: true, linears: linears) { _ in
                installAttempts += 1
                throw installAttempts == 1
                    ? ExpectedFailure.install : ExpectedFailure.rollback
            }
        ) { error in
            let preparationError = error as? FusedQuantizedLinearPreparationError
            XCTAssertNotNil(preparationError?.rollbackError)
        }
        XCTAssertEqual(installAttempts, 2)
        XCTAssertFalse(cache.isPrepared)
    }

    /// Opt-in paired benchmark for a local Qwen 3.5 checkpoint.
    ///
    /// The same model and materialized weights are used for both paths, with
    /// alternating order, to avoid cross-process Metal-cache and load noise:
    ///
    /// ```sh
    /// MLX_QWEN_GDN_BENCH_MODEL=/path/to/model \
    ///   swift test -c release --filter testRealCheckpointBenchmark
    /// ```
    func testRealCheckpointBenchmark() throws {
        guard
            let path = ProcessInfo.processInfo.environment["MLX_QWEN_GDN_BENCH_MODEL"],
            !path.isEmpty
        else {
            throw XCTSkip("set MLX_QWEN_GDN_BENCH_MODEL to run the real-model benchmark")
        }

        let directory = URL(filePath: path)
        let configurationData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35Configuration.self, from: configurationData)
        let baseConfiguration = try JSONDecoder().decode(
            BaseConfiguration.self, from: configurationData)
        func loadModel(fused: Bool) throws -> Qwen35Model {
            let model = Qwen35Model(configuration)
            try loadWeights(
                modelDirectory: directory,
                model: model,
                perLayerQuantization: baseConfiguration.perLayerQuantization)
            let layers = model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }
            XCTAssertFalse(layers.isEmpty)
            for layer in layers {
                layer.fusedInputProjectionEnabled = fused
                if fused {
                    XCTAssertTrue(try layer.prepareFusedInputProjection())
                }
            }
            return model
        }

        // Two models keep independently compiled decode traces, while sharing
        // one process's Metal pipeline cache and runtime conditions.
        let baselineModel = try loadModel(fused: false)
        let fusedModel = try loadModel(fused: true)
        let fusedLayerCount = fusedModel.modules().compactMap { $0 as? Qwen35GatedDeltaNet }
            .count

        let promptLength = 512
        let prompt = MLXArray((0 ..< promptLength).map { Int32(($0 % 32_000) + 1) })
        let parameters = GenerateParameters(
            maxTokens: 64, temperature: 0,
            prefill: PrefillParameters(stepSize: 128))

        func run(_ model: Qwen35Model) throws -> (
            prefill: Double, decode: Double, tokens: [Int]
        ) {
            var iterator = try TokenIterator(
                input: LMInput(tokens: prompt), model: model, parameters: parameters)
            let start = CFAbsoluteTimeGetCurrent()
            var tokens: [Int] = []
            while let token = iterator.next() {
                tokens.append(token)
            }
            return (
                iterator.promptPrefillTime,
                CFAbsoluteTimeGetCurrent() - start,
                tokens
            )
        }

        // Compile and warm each independent model before alternating AB / BA.
        let baselineWarmup = try run(baselineModel)
        let fusedWarmup = try run(fusedModel)
        XCTAssertEqual(fusedWarmup.tokens, baselineWarmup.tokens)

        var baselinePrefill: [Double] = []
        var fusedPrefill: [Double] = []
        var baselineDecode: [Double] = []
        var fusedDecode: [Double] = []
        var referenceTokens = baselineWarmup.tokens
        for pair in 0 ..< 4 {
            let order = pair.isMultiple(of: 2) ? [false, true] : [true, false]
            for fused in order {
                let result = try run(fused ? fusedModel : baselineModel)
                XCTAssertEqual(result.tokens, referenceTokens)
                referenceTokens = result.tokens
                if fused {
                    fusedPrefill.append(result.prefill)
                    fusedDecode.append(result.decode)
                } else {
                    baselinePrefill.append(result.prefill)
                    baselineDecode.append(result.decode)
                }
            }
        }

        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return (sorted[1] + sorted[2]) / 2
        }

        let baselinePrefillMedian = median(baselinePrefill)
        let fusedPrefillMedian = median(fusedPrefill)
        let baselineDecodeMedian = median(baselineDecode)
        let fusedDecodeMedian = median(fusedDecode)
        let checksum = referenceTokens.reduce(into: UInt64(0)) { checksum, token in
            checksum = checksum &* 1_099_511_628_211 &+ UInt64(bitPattern: Int64(token))
        }
        print(
            String(
                format:
                    "[four-gdn] layers=%d prefill baseline=%.4fs fused=%.4fs speedup=%.2f%% decode baseline=%.4fs fused=%.4fs speedup=%.2f%% baselineTPS=%.2f fusedTPS=%.2f checksum=%llu",
                fusedLayerCount,
                baselinePrefillMedian,
                fusedPrefillMedian,
                100 * (baselinePrefillMedian / fusedPrefillMedian - 1),
                baselineDecodeMedian,
                fusedDecodeMedian,
                100 * (baselineDecodeMedian / fusedDecodeMedian - 1),
                Double(referenceTokens.count) / baselineDecodeMedian,
                Double(referenceTokens.count) / fusedDecodeMedian,
                checksum))
    }
}
