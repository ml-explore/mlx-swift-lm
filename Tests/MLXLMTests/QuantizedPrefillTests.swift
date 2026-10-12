// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct QuantizedPrefillTests {
    private final class ProgressLog: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []
        func append(_ value: Int) { lock.withLock { values.append(value) } }
        var counts: [Int] { lock.withLock { values } }
    }

    private final class OffsetQuantizedLinear: QuantizedLinear {
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            super.callAsFunction(x) + MLXArray(Float(0.25)).asType(x.dtype)
        }
    }

    private final class ProjectionBank: Module {
        @ModuleInfo var projections: [Linear]
        let trace = CompiledTrace<ProjectionBank> { bank, arguments in
            [bank.projections[0](arguments[0])]
        }

        init(_ projections: [Linear]) {
            _projections.wrappedValue = projections
            super.init()
            train(false)
        }
    }

    private final class ImmutableProjection: Module {
        let projection: QuantizedLinear
        init(_ projection: QuantizedLinear) {
            self.projection = projection
            super.init()
            train(false)
        }
    }

    private final class DictionaryBank: Module {
        @ModuleInfo(key: "named_projections") var projections: [String: Linear]
        init(_ projections: [String: Linear]) {
            _projections.wrappedValue = projections
            super.init()
            train(false)
        }
    }

    private final class TupleBank: Module {
        @ModuleInfo var projections: (Linear, Linear, Linear, Linear, Linear)
        init(_ projection: Linear) {
            _projections.wrappedValue = (projection, projection, projection, projection, projection)
            super.init()
            train(false)
        }
    }

    private final class NestedBank: Module {
        @ModuleInfo var projections: [[Linear]]
        @ModuleInfo var optionalProjections: [Linear?]
        init(_ projection: Linear) {
            _projections.wrappedValue = [[projection]]
            _optionalProjections.wrappedValue = [nil, projection]
            super.init()
            train(false)
        }
    }

    private final class FailingBank: Module {
        enum Failure: Error { case rejected }
        @ModuleInfo var a: Linear
        @ModuleInfo var z: Linear
        let mutateBeforeFailure: Bool
        let failure: any Error
        var rejectRefresh = false
        let trace = CompiledTrace<FailingBank> { bank, arguments in [bank.a(arguments[0])] }
        init(
            _ projection: Linear, mutateBeforeFailure: Bool = false,
            failure: any Error = Failure.rejected
        ) {
            self.mutateBeforeFailure = mutateBeforeFailure
            self.failure = failure
            _a.wrappedValue = projection
            _z.wrappedValue = projection
            super.init()
            train(false)
        }
        override func updateModule(key: String, _ value: Any) throws {
            if key == "z" {
                if mutateBeforeFailure { try super.updateModule(key: key, value) }
                throw failure
            }
            try super.updateModule(key: key, value)
        }

        override func update(
            modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
            modulePath: [String] = []
        ) throws -> Self {
            if rejectRefresh && modules.isEmpty { throw Failure.rejected }
            return try super.update(
                modules: modules, verify: verify, path: path, modulePath: modulePath)
        }
    }

    private final class PreparationProbe: Module, LLMModel {
        let model: LlamaModel
        var discoveries = 0
        var loraLayers: [Module] { model.loraLayers }

        init(_ model: LlamaModel) {
            self.model = model
            super.init()
            train(false)
        }

        override func modules() -> [Module] {
            discoveries += 1
            return super.modules()
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try model.newCache(parameters: parameters)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            model(inputs, cache: cache)
        }
    }

    private func makeLinear(
        dtype: DType = .bfloat16, inputs: Int = 1024, outputs: Int = 2048, bias: Bool = false,
        groupSize: Int = 64, bits: Int = 4
    ) -> QuantizedLinear {
        let linear = Linear(inputs, outputs, bias: bias)
        linear.update(parameters: linear.parameters().mapValues { $0.asType(dtype) })
        let quantized = QuantizedLinear(linear, groupSize: groupSize, bits: bits)
        quantized.train(false)
        eval(quantized)
        return quantized
    }

    private func identical(_ a: MLXArray, _ b: MLXArray) {
        #expect(a.shape == b.shape)
        #expect(a.dtype == b.dtype)
        let bitType: DType = a.dtype == .float32 ? .uint32 : .uint16
        #expect(arrayEqual(a.view(dtype: bitType), b.view(dtype: bitType)).item(Bool.self))
    }

    private func numericallyClose(_ a: MLXArray, _ b: MLXArray) {
        #expect(a.shape == b.shape)
        #expect(a.dtype == b.dtype)
        #expect(all(isFinite(a)).item(Bool.self) && all(isFinite(b)).item(Bool.self))
        #expect(allClose(a, b, rtol: 0.02, atol: 0.02).item(Bool.self))
    }

    private func projectionAccuracy(_ linear: QuantizedLinear, _ x: MLXArray, _ actual: MLXArray) {
        let weights = dequantized(
            linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, dtype: .float32)
        let product = matmul(x.asType(.float32), weights.T)
        let reference = linear.bias.map { product + $0.asType(.float32) } ?? product
        let stock = linear(x).asType(.float32)
        #expect(actual.shape == stock.shape && actual.dtype == x.dtype)
        #expect(all(isFinite(reference)).item(Bool.self))
        #expect(all(isFinite(stock)).item(Bool.self) && all(isFinite(actual)).item(Bool.self))
        let stockError = abs(stock - reference).max().item(Float.self)
        let routedError = abs(actual.asType(.float32) - reference).max().item(Float.self)
        // Bound error relative to stock rounding against the FP32 reference.
        #expect(routedError <= 1.5 * stockError + 0.005)
    }

    @Test func policyUsesPrefillWidthAndWorkspaceWithoutKernelGeometry() {
        func dense(_ rows: Int, _ inputs: Int = 1024, _ outputs: Int = 2048, bits: Int = 4) -> Bool
        {
            QuantizedPrefill.prefersDense(
                rows: rows, inputs: inputs, outputs: outputs, bits: bits,
                workspaceLimit: 64 * 1024 * 1024)
        }
        #expect(!dense(383, bits: 2))
        #expect(dense(384, bits: 2))
        for bits in [3, 4, 5, 6, 8] {
            #expect(!dense(2047, bits: bits))
            #expect(dense(2048, bits: bits))
        }
        #expect(!dense(2048, bits: 7))
        #expect(dense(8192))
        #expect(dense(2048, 256, 384))
        #expect(dense(2048, 4096, 128))
        #expect(dense(2048, 1024, 2049))
        #expect(dense(2048, 1024, 32768))
        #expect(!dense(2048, 1024, 32800))
        #expect(!dense(1))
        #expect(!dense(-1))
        #expect(!dense(2048, 0))
        #expect(!dense(2048, 1024, 0))
        #expect(!dense(2048, Int.max))
        #expect(!dense(2048, 1024, Int.max))
    }

    @Test func workspaceBudgetScalesAndRespectsAllocatorHeadroom() {
        let gib = 1024 * 1024 * 1024
        func limit(_ recommended: Int, _ memory: Int, _ active: Int) -> Int {
            QuantizedPrefill.workspaceLimit(
                recommendedBytes: recommended, memoryLimit: memory, activeBytes: active)
        }
        #expect(limit(8 * gib, 12 * gib, 0) == 128 * 1024 * 1024)
        #expect(limit(16 * gib, 24 * gib, 0) == 256 * 1024 * 1024)
        #expect(limit(16 * gib, 8 * gib, 0) == 128 * 1024 * 1024)
        #expect(limit(8 * gib, 12 * gib, 8 * gib - 64 * 1024 * 1024) == 1024 * 1024)
        #expect(limit(16 * gib, 24 * gib, 8 * gib) >= 96 * 1024 * 1024)
        #expect(limit(12 * gib, 18 * gib, 9 * gib) < 96 * 1024 * 1024)
        #expect(limit(8 * gib, 12 * gib, 8 * gib) == 0)
        #expect(limit(8 * gib, 12 * gib, 9 * gib) == 0)
        #expect(limit(0, 12 * gib, 0) == 0)
        #expect(limit(8 * gib, 0, 0) == 0)
        #expect(limit(-1, Int.max, Int.max) == 0)
        #expect(limit(Int.max, Int.max, Int.max) == 0)
    }

    @Test func affinePrecisionsBitWidthsAndGroupsMeetTheFP32AccuracyBound() throws {
        MLXRandom.seed(103)
        for dtype in [DType.float16, .bfloat16] {
            for bits in [2, 3, 4, 5, 6, 8] {
                for groupSize in [32, 64, 128] {
                    let stock = makeLinear(
                        dtype: dtype, inputs: 256, outputs: 384, bias: true,
                        groupSize: groupSize, bits: bits)
                    let rows = bits == 2 ? 384 : 2048
                    #expect(
                        !QuantizedPrefill.usesDense(
                            stock, rows: rows - 1, inputs: 256, dtype: dtype))
                    #expect(
                        QuantizedPrefill.usesDense(stock, rows: rows, inputs: 256, dtype: dtype))
                    #expect(
                        !QuantizedPrefill.usesDense(
                            stock, rows: rows, inputs: 256, dtype: dtype,
                            workspaceLimit: 256 * 384 * 2 - 1))
                    let bank = ProjectionBank([stock])
                    try QuantizedPrefill.prepare(bank, rows: rows)
                    #expect(
                        (bank.projections[0] is PrefillQuantizedLinear)
                            == QuantizedPrefill.isAvailable)
                    let x = MLXRandom.normal([1, rows, 256], dtype: dtype)
                    projectionAccuracy(
                        stock, x, QuantizedPrefill.withPrefill { bank.projections[0](x) })
                }
            }
        }
    }

    @Test func deviceAvailabilityFollowsMLXGPUAndMemorySupport() {
        Device.withDefaultDevice(.cpu) {
            #expect(!QuantizedPrefill.isAvailable)
        }
        #if canImport(Metal)
        Device.withDefaultDevice(.gpu) {
            #expect(
                QuantizedPrefill.isAvailable == ((GPU.maxRecommendedWorkingSetBytes() ?? 0) > 0))
        }
        #endif
    }

    @Test func projectionsBeyondTheOldCapUseTheCurrentBudget() {
        let stock = makeLinear(inputs: 4096, outputs: 12288, bits: 8)
        let bytes = 4096 * 12288 * 2
        #expect(bytes > 64 * 1024 * 1024)
        #expect(
            !QuantizedPrefill.usesDense(
                stock, rows: 2048, inputs: 4096, dtype: .bfloat16,
                workspaceLimit: bytes - 1))
        #expect(
            QuantizedPrefill.usesDense(
                stock, rows: 2048, inputs: 4096, dtype: .bfloat16,
                workspaceLimit: bytes))
        let x = MLXRandom.normal([1, 2048, 4096], dtype: .bfloat16)
        if QuantizedPrefill.isAvailable {
            projectionAccuracy(stock, x, QuantizedPrefill.dense(stock, x))
        }
    }

    @Test func eightBitPrefillPreservesNumericsAndWarmCache() throws {
        let configuration = LlamaConfiguration(
            hiddenSize: 1024, hiddenLayers: 1, intermediateSize: 2048,
            attentionHeads: 32, rmsNormEps: 0.000001, vocabularySize: 16, kvHeads: 2)
        let stock = LlamaModel(configuration)
        stock.update(parameters: stock.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: stock, groupSize: 64, bits: 8)
        let automatic = LlamaModel(configuration)
        automatic.update(parameters: automatic.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: automatic, groupSize: 64, bits: 8)
        automatic.update(parameters: stock.parameters())
        stock.train(false)
        automatic.train(false)
        let a = try stock.newCache(parameters: nil)
        let b = try automatic.newCache(parameters: nil)

        func logits(_ model: LlamaModel, _ cache: [KVCache], _ prompt: MLXArray) throws -> MLXArray
        {
            let prepared = try model.prepare(
                LMInput(tokens: prompt), cache: cache, state: nil,
                prefill: .init(
                    stepSize: 2048, quantizedProjections: model === automatic ? .automatic : .stock)
            )
            switch prepared {
            case .tokens(let remaining):
                return model(remaining.tokens[.newAxis], cache: cache)[0, -1, 0...]
            case .logits(let output):
                return output.logits[0, -1, 0...]
            }
        }
        for length in [4097, 2049] {
            let prompt = MLXArray((0 ..< length).map { Int32($0 % 16) })
            var reference = try logits(stock, a, prompt)
            var candidate = try logits(automatic, b, prompt)
            numericallyClose(reference, candidate)
            for _ in 0 ..< 8 {
                let token = argMax(reference).item(Int.self)
                reference = stock(MLXArray([token])[.newAxis], cache: a)[0, -1, 0...]
                candidate = automatic(MLXArray([token])[.newAxis], cache: b)[0, -1, 0...]
                numericallyClose(reference, candidate)
            }
            #expect(a[0].offset == b[0].offset)
            for (left, right) in zip(a[0].state, b[0].state) { numericallyClose(left, right) }
        }
        if QuantizedPrefill.isAvailable {
            #expect(automatic.leafModules().flattened().contains { $0.1 is PrefillQuantizedLinear })
        }
    }

    @Test func fourBitModelPrefillRoutesWithWarmPrefixesRepeatedRequestsAndDecode() throws {
        guard QuantizedPrefill.isAvailable else { return }
        final class RoutingLog: @unchecked Sendable {
            let lock = NSLock()
            private var decisions: [QuantizedPrefill.Decision] = []
            func append(_ value: QuantizedPrefill.Decision) {
                lock.withLock { decisions.append(value) }
            }
            var values: [QuantizedPrefill.Decision] { lock.withLock { decisions } }
        }
        func accuracy(_ stock: MLXArray, _ routed: MLXArray, _ reference: MLXArray, dtype: DType) {
            #expect(stock.shape == routed.shape && stock.shape == reference.shape)
            #expect(stock.dtype == dtype && routed.dtype == dtype)
            #expect(all(isFinite(stock)).item(Bool.self))
            #expect(
                all(isFinite(routed)).item(Bool.self) && all(isFinite(reference)).item(Bool.self))
            let stockDifference = stock.asType(.float32) - reference
            let routedDifference = routed.asType(.float32) - reference
            let denominator = maximum(sum(square(reference)), MLXArray(Float(1e-30)))
            let stockError = sqrt(sum(square(stockDifference)) / denominator).item(Float.self)
            let routedError = sqrt(sum(square(routedDifference)) / denominator).item(Float.self)
            let epsilon: Float = dtype == .float16 ? 1 / 1024 : 1 / 128
            #expect(routedError <= 1.5 * stockError + epsilon)
            let stockMax = abs(stockDifference).max().item(Float.self)
            let routedMax = abs(routedDifference).max().item(Float.self)
            // Whole-model rounding scales with output magnitude and storage precision.
            // Allow one representable step at the reference peak, not a fixed absolute floor.
            let rounding = epsilon * abs(reference).max().item(Float.self)
            #expect(routedMax <= 1.5 * stockMax + rounding)
        }
        MLXRandom.seed(414)
        let qwen = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {"hidden_size":256,"num_hidden_layers":2,"intermediate_size":512,
                 "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
                 "vocab_size":64,"rms_norm_eps":0.000001,"tie_word_embeddings":true}
                """.utf8))
        let factories: [() -> any LLMModel] = [
            { Qwen3Model(qwen) },
            {
                LlamaModel(
                    .init(
                        hiddenSize: 256, hiddenLayers: 2, intermediateSize: 512,
                        attentionHeads: 8, rmsNormEps: 0.000001, vocabularySize: 64, kvHeads: 2))
            },
        ]
        for dtype in [DType.float16, .bfloat16] {
            for factory in factories {
                let model = factory()
                model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
                quantize(model: model, groupSize: 64, bits: 4)
                model.train(false)
                let reference = factory()
                var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                for (path, module) in model.leafModules().flattened() {
                    guard let quantized = module as? Quantized else { continue }
                    let parameters = Dictionary(
                        uniqueKeysWithValues: module.parameters().flattened())
                    weights[path + ".weight"] = dequantized(
                        parameters["weight"]!, scales: parameters["scales"]!,
                        biases: parameters["biases"],
                        groupSize: quantized.groupSize, bits: quantized.bits, dtype: .float32)
                }
                // Reconstruct the same quantized model in FP32, including embeddings and norms.
                reference.update(
                    parameters: ModuleParameters.unflattened(
                        reference.parameters().flattened().map { key, _ in
                            (key, weights[key]!.asType(.float32))
                        }))
                reference.train(false)
                let caches = [
                    try model.newCache(parameters: nil), try model.newCache(parameters: nil),
                    try reference.newCache(parameters: nil),
                ]
                let prefix = MLXArray((0 ..< 17).map { Int32($0 % 64) })
                for arm in 0 ..< 3 {
                    let runner = arm == 2 ? reference : model
                    eval(runner(prefix[.newAxis], cache: caches[arm]), caches[arm])
                }
                for length in [4097, 2049] {
                    let prompt = MLXArray((0 ..< length).map { Int32($0 % 64) })
                    var outputs = [MLXArray]()
                    for arm in 0 ..< 3 {
                        let runner = arm == 2 ? reference : model
                        runner.train(false)
                        let routes = RoutingLog()
                        let progress = ProgressLog()
                        let observe: @Sendable (QuantizedPrefill.Decision) -> Void = {
                            routes.append($0)
                        }
                        let prepared = try QuantizedPrefill.$observer.withValue(observe) {
                            try runner.prepare(
                                LMInput(tokens: prompt), cache: caches[arm], state: nil,
                                prefill: .init(
                                    stepSize: 2048,
                                    quantizedProjections: arm == 1 ? .automatic : .stock,
                                    progress: { count, _ in progress.append(count) }
                                ))
                        }
                        guard case .tokens(let tail) = prepared else {
                            Issue.record("Expected final token")
                            return
                        }
                        let output = runner(tail.tokens[.newAxis], cache: caches[arm])
                        eval(output, caches[arm])
                        outputs.append(output)
                        #expect(progress.counts == (length == 4097 ? [2048, 4096] : [2048]))
                        #expect(routes.values.contains(.dense) == (arm == 1))
                    }
                    accuracy(outputs[0], outputs[1], outputs[2], dtype: dtype)
                    // Teacher forcing compares the same history. Trained-model benchmarks
                    // separately compare independently selected greedy tokens.
                    for _ in 0 ..< 4 {
                        let token = MLXArray([argMax(outputs[0][0, -1, 0...]).item(Int.self)])
                        for arm in 0 ..< 3 {
                            let runner = arm == 2 ? reference : model
                            runner.train(false)
                            let routes = RoutingLog()
                            let observe: @Sendable (QuantizedPrefill.Decision) -> Void = {
                                routes.append($0)
                            }
                            outputs[arm] = QuantizedPrefill.$observer.withValue(observe) {
                                runner(token[.newAxis], cache: caches[arm])
                            }
                            eval(outputs[arm], caches[arm])
                            #expect(routes.values.isEmpty == (arm == 2))
                            #expect(routes.values.allSatisfy { $0 == .outsidePrefill })
                        }
                        accuracy(outputs[0], outputs[1], outputs[2], dtype: dtype)
                    }
                    for layer in caches[0].indices {
                        let a = caches[0][layer]
                        let b = caches[1][layer]
                        let c = caches[2][layer]
                        #expect(a.offset == b.offset && a.offset == c.offset)
                        for index in a.state.indices {
                            accuracy(a.state[index], b.state[index], c.state[index], dtype: dtype)
                        }
                    }
                }
            }
        }
    }

    @Test func routingIsScopedAndRestoresAfterErrorsAndNestedCalls() throws {
        guard QuantizedPrefill.isAvailable else { return }
        enum Failure: Error { case stopped }
        let stock = makeLinear(inputs: 256, outputs: 384)
        let bank = ProjectionBank([stock])
        #expect(try QuantizedPrefill.prepare(bank, rows: 2048))
        #expect(try QuantizedPrefill.prepare(bank, rows: 2048))
        #expect(!(try QuantizedPrefill.prepare(bank, rows: 512)))
        let x = MLXRandom.normal([1, 2048, 256], dtype: .bfloat16)
        eval(x)
        func isStock() -> Bool {
            graphDescription([bank.projections[0](x)]).contains("QuantizedMatmul")
        }
        #expect(isStock())
        #expect(throws: Failure.self) {
            try QuantizedPrefill.withPrefill {
                #expect(!isStock())
                QuantizedPrefill.withPrefill(enabled: false) { #expect(isStock()) }
                #expect(!isStock())
                throw Failure.stopped
            }
        }
        #expect(isStock())
        identical(stock(x), bank.projections[0](x))
    }

    @Test func compiledForwardsKeepStockRoutingAndCompilationAcrossMemoryChanges() throws {
        guard QuantizedPrefill.isAvailable else { return }
        let bank = ProjectionBank([makeLinear(inputs: 256, outputs: 384)])
        #expect(try QuantizedPrefill.prepare(bank, rows: 2048))
        let x = MLXRandom.normal([1, 2048, 256], dtype: .bfloat16)
        eval(x)
        let first = QuantizedPrefill.withPrefill { bank.trace(bank, x) }
        eval(first)
        #expect(bank.trace.isCompiled)
        identical(first, bank.projections[0](x))
        let outside = bank.trace(bank, x)
        identical(first, outside)
        QuantizedPrefill.withPrefill {
            let compiled = bank.trace(bank, x)
            identical(first, compiled)
            // Leaving the compiled region restores routing for subsequent eager work.
            #expect(!graphDescription([bank.projections[0](x)]).contains("QuantizedMatmul"))
        }
        let previousLimit = Memory.memoryLimit
        let fallback: MLXArray
        do {
            defer { Memory.memoryLimit = previousLimit }
            Memory.memoryLimit = Memory.activeMemory
            #expect(!(try QuantizedPrefill.prepare(bank, rows: 2048)))
            fallback = QuantizedPrefill.withPrefill { bank.trace(bank, x) }
            let eager = QuantizedPrefill.withPrefill { bank.projections[0](x) }
            #expect(graphDescription([eager]).contains("QuantizedMatmul"))
        }
        identical(first, fallback)
        #expect(try QuantizedPrefill.prepare(bank, rows: 2048))
        // Also cover a trace compiled outside prefill before its first scoped call.
        bank.trace.invalidate()
        identical(first, bank.trace(bank, x))
        identical(first, QuantizedPrefill.withPrefill { bank.trace(bank, x) })
        #expect(bank.trace.isCompiled)
    }

    @Test func diagnosticsReportDispatchWithoutEvaluatingTheGraph() throws {
        guard QuantizedPrefill.isAvailable else { return }
        final class Decisions: @unchecked Sendable {
            let lock = NSLock()
            private var storage: [QuantizedPrefill.Decision] = []
            func append(_ value: QuantizedPrefill.Decision) {
                lock.withLock { storage.append(value) }
            }
            var values: [QuantizedPrefill.Decision] { lock.withLock { storage } }
        }
        let bank = ProjectionBank([makeLinear(inputs: 256, outputs: 384)])
        #expect(try QuantizedPrefill.prepare(bank, rows: 2048))
        let decisions = Decisions()
        let x = MLXRandom.normal([1, 2048, 256], dtype: .bfloat16)
        let observe: @Sendable (QuantizedPrefill.Decision) -> Void = { decisions.append($0) }
        QuantizedPrefill.$observer.withValue(observe) {
            _ = bank.projections[0](x)
            QuantizedPrefill.withPrefill {
                _ = bank.projections[0](x)
                _ = bank.projections[0](x[0..., ..<512, 0...])
                _ = bank.projections[0](x[0])
                bank.train(true)
                _ = bank.projections[0](x)
                bank.train(false)
            }
        }
        #expect(
            decisions.values == [
                .outsidePrefill, .dense, .insufficientRows, .unsupportedInput, .trainable,
            ])
        #expect(QuantizedPrefill.observer == nil)
        #expect(
            QuantizedPrefill.decision(
                bank.projections[0], rows: 2048, inputs: 256, dtype: .bfloat16, workspaceLimit: 0
            ) == .insufficientWorkspace)
        #expect(
            QuantizedPrefill.decision(
                bank.projections[0], rows: 2048, inputs: 255, dtype: .bfloat16,
                workspaceLimit: Int.max
            ) == .unsupportedFormat)
    }

    @Test func routedAccuracyRemainsBoundedAcrossSeedsScalesAndShapes() throws {
        guard QuantizedPrefill.isAvailable else { return }
        for seed: UInt64 in [17, 103, 711] {
            MLXRandom.seed(seed)
            for dtype in [DType.float16, .bfloat16] {
                for bits in [2, 4, 8] {
                    for outputs in [128, 257] {
                        let stock = makeLinear(
                            dtype: dtype, inputs: 256, outputs: outputs, bits: bits)
                        let bank = ProjectionBank([stock])
                        let rows = bits == 2 ? 384 : 2048
                        #expect(try QuantizedPrefill.prepare(bank, rows: rows))
                        for scale: Float in [1 / 64, 1, 64] {
                            let x = MLXRandom.normal([1, rows, 256], dtype: dtype) * scale
                            let actual = QuantizedPrefill.withPrefill { bank.projections[0](x) }
                            projectionAccuracy(stock, x, actual)
                            let reference = matmul(
                                x.asType(.float32),
                                dequantized(
                                    stock.weight, scales: stock.scales, biases: stock.biases,
                                    groupSize: stock.groupSize, bits: bits, dtype: .float32
                                ).T)
                            let denominator = maximum(
                                sum(square(reference)), MLXArray(Float(1e-30)))
                            func relativeError(_ value: MLXArray) -> Float {
                                sqrt(sum(square(value.asType(.float32) - reference)) / denominator)
                                    .item(Float.self)
                            }
                            let epsilon: Float = dtype == .float16 ? 1 / 1024 : 1 / 128
                            #expect(
                                relativeError(actual) <= 1.5 * relativeError(stock(x)) + epsilon)
                        }
                    }
                }
            }
        }
    }

    @Test func unpreparedAndTrainableProjectionsNeverTakeTheDenseRoute() {
        let stock = makeLinear(inputs: 256, outputs: 384)
        let unfrozen = QuantizedLinear(
            weight: stock.weight, scales: stock.scales, biases: stock.biases,
            groupSize: 64, bits: 4)
        unfrozen.train(false)
        let routed = PrefillQuantizedLinear(unfrozen)
        let x = MLXRandom.normal([1, 2048, 256], dtype: .bfloat16)
        #expect(!QuantizedPrefill.usesDense(routed, rows: 2048, inputs: 256, dtype: .bfloat16))
        let result = QuantizedPrefill.withPrefill { routed(x) }
        #expect(graphDescription([result]).contains("QuantizedMatmul"))
        identical(stock(x), result)
    }

    @Test func replacementPreservesFrozenKeysAndSkipsTrainableLayers() throws {
        let stock = makeLinear()
        let unfrozen = QuantizedLinear(
            weight: stock.weight, scales: stock.scales,
            biases: stock.biases, groupSize: 64, bits: 4)
        let routed = PrefillQuantizedLinear(unfrozen)
        #expect(routed.noGrad() == unfrozen.noGrad())
        #expect(
            routed.trainableParameters().flattened().count
                == unfrozen.trainableParameters().flattened().count)
        let bank = ProjectionBank([unfrozen])
        try QuantizedPrefill.prepare(bank, rows: 2048)
        #expect(bank.projections[0] === unfrozen)
    }

    @Test func dispatchEligibilityDependsOnCurrentParametersAndPrecision() {
        let up = makeLinear()
        let down = makeLinear(inputs: 2048, outputs: 1024)
        #expect(!QuantizedPrefill.usesDense(up, rows: 512, inputs: 1024, dtype: .bfloat16))
        #expect(QuantizedPrefill.usesDense(up, rows: 2048, inputs: 1024, dtype: .bfloat16))
        #expect(QuantizedPrefill.usesDense(down, rows: 2048, inputs: 2048, dtype: .bfloat16))
        #expect(!QuantizedPrefill.usesDense(up, rows: 2048, inputs: 1024, dtype: .float32))
        #expect(!QuantizedPrefill.usesDense(up, rows: 2048, inputs: 1024, dtype: .float16))
        #expect(!QuantizedPrefill.usesDense(up, rows: 2048, inputs: 2048, dtype: .bfloat16))
        up.train(true)
        #expect(!QuantizedPrefill.usesDense(up, rows: 2048, inputs: 1024, dtype: .bfloat16))
    }

    @Test func projectionParityParametersAndFallbacks() throws {
        MLXRandom.seed(97)
        for dtype in [DType.float16, .bfloat16] {
            let stock = makeLinear(dtype: dtype, bias: true)
            let routed = PrefillQuantizedLinear(stock)
            #expect(
                QuantizedPrefill.usesDense(stock, rows: 2048, inputs: 1024, dtype: dtype))
            let original = Dictionary(uniqueKeysWithValues: stock.parameters().flattened())
            let replacement = Dictionary(uniqueKeysWithValues: routed.parameters().flattened())
            #expect(Set(original.keys) == Set(replacement.keys))
            for key in original.keys { identical(original[key]!, replacement[key]!) }
            #expect(routed.trainableParameters().flattened().isEmpty)
            for dimensions in [
                [1, 512, 1024], [1024], [512, 1024], [2, 512, 1024],
                [1, 1, 1024], [1, 480, 1024], [1, 481, 1024], [1, 500, 1024],
                [1, 511, 1024], [1, 513, 1024], [1, 1024, 1024],
            ] {
                let x = MLXRandom.normal(dimensions, dtype: dtype)
                identical(stock(x), routed(x))
            }
            let strided = MLXRandom.normal([1, 1024, 512], dtype: dtype).transposed(0, 2, 1)
            identical(stock(strided), routed(strided))
            let wideStrided = MLXRandom.normal([1, 1024, 2048], dtype: dtype).transposed(0, 2, 1)
            projectionAccuracy(
                stock, wideStrided, QuantizedPrefill.withPrefill { routed(wideStrided) })
            routed.train(true)
            identical(stock(wideStrided), routed(wideStrided))
            routed.train(false)
        }
    }

    @Test func largeProjectionParityAndCurrentWeights() throws {
        MLXRandom.seed(19)
        for dtype in [DType.float16, .bfloat16] {
            for (inputs, outputs) in [(2560, 9728), (9728, 2560)] {
                let stock = makeLinear(dtype: dtype, inputs: inputs, outputs: outputs)
                let routed = PrefillQuantizedLinear(stock)
                for rows in [512, 2048, 4096] {
                    #expect(
                        QuantizedPrefill.usesDense(stock, rows: rows, inputs: inputs, dtype: dtype)
                            == (rows >= 2048))
                    let x = MLXRandom.normal([1, rows, inputs], dtype: dtype)
                    projectionAccuracy(stock, x, QuantizedPrefill.withPrefill { routed(x) })
                    if QuantizedPrefill.isAvailable,
                        QuantizedPrefill.usesDense(stock, rows: rows, inputs: inputs, dtype: dtype)
                    {
                        projectionAccuracy(stock, x, QuantizedPrefill.dense(stock, x))
                    }
                }
                let x = MLXRandom.normal([1, 2048, inputs], dtype: dtype)
                let before = QuantizedPrefill.withPrefill { routed(x) }
                eval(before)
                let parameters = ModuleParameters.unflattened([
                    "weight": MLXArray.zeros(stock.weight.shape, dtype: .uint32)
                ])
                stock.update(parameters: parameters)
                routed.update(parameters: parameters)
                let after = QuantizedPrefill.withPrefill { routed(x) }
                #expect(!arrayEqual(before, after).item(Bool.self))
                projectionAccuracy(stock, x, after)
            }
        }
    }

    @Test func customSubclassesAndUnsupportedQuantizationStayUntouched() throws {
        let stock = makeLinear()
        let custom = OffsetQuantizedLinear(
            weight: stock.weight, scales: stock.scales,
            biases: stock.biases, groupSize: 64, bits: 4)
        custom.train(false)
        #expect(!QuantizedPrefill.usesDense(custom, rows: 2048, inputs: 1024, dtype: .bfloat16))
        let bank = ProjectionBank([custom])
        try QuantizedPrefill.prepare(bank, rows: 2048)
        #expect(bank.projections[0] === custom)
        let batched = QuantizedLinear(
            weight: .zeros([2, 2048, 1024], dtype: .bfloat16), bias: nil,
            groupSize: 64, bits: 4)
        let batchedBank = ProjectionBank([batched])
        try QuantizedPrefill.prepare(batchedBank, rows: 2048)
        #expect(batchedBank.projections[0] === batched)
        for dtype in [DType.float16, .bfloat16] {
            let other = QuantizedLinear(
                weight: stock.weight, scales: stock.scales.asType(dtype),
                biases: stock.biases?.asType(dtype), groupSize: 64, bits: 4,
                globalScale: dtype == .bfloat16 ? MLXArray(Float(1)) : nil)
            other.train(false)
            #expect(!QuantizedPrefill.usesDense(other, rows: 2048, inputs: 1024, dtype: .bfloat16))
        }
        Device.withDefaultDevice(.cpu) {
            #expect(!QuantizedPrefill.isAvailable)
            let routed = PrefillQuantizedLinear(stock)
            let x = MLXRandom.normal([1, 4, 1024], dtype: .bfloat16)
            identical(stock(x), routed(x))
        }
    }

    @Test func genericPreparationPreservesParametersAndInvalidatesChangedTraces() throws {
        let bank = ProjectionBank((0 ..< 8).map { _ in makeLinear(bias: true) })
        let x = MLXRandom.normal([1, 512, 1024], dtype: .bfloat16)
        let reference = bank.projections[0](x)
        eval(reference)
        let keys = Set(bank.parameters().flattened().map(\.0))
        eval(bank.trace(bank, x))
        #expect(bank.trace.isCompiled)
        let before = bank.projections[0]
        try QuantizedPrefill.prepare(bank, rows: 2048)
        #expect(Set(bank.parameters().flattened().map(\.0)) == keys)
        #expect((bank.projections[0] !== before) == QuantizedPrefill.isAvailable)
        if QuantizedPrefill.isAvailable {
            #expect(bank.projections[0] is PrefillQuantizedLinear)
            #expect(!bank.trace.isCompiled)
        }
        identical(reference, bank.projections[0](x))
        identical(reference, bank.trace(bank, x))
        let installed = bank.projections[0]
        try QuantizedPrefill.prepare(bank, rows: 1024)
        #expect(bank.projections[0] === installed)
        #expect(bank.trace.isCompiled)
    }

    @Test func sparseReplacementPreservesTrailingProjection() throws {
        let eligible = makeLinear()
        let untouched = makeLinear(dtype: .float32)
        let bank = ProjectionBank([eligible, untouched])
        let keys = Set(bank.parameters().flattened().map(\.0))
        try QuantizedPrefill.prepare(bank, rows: 2048)
        #expect(bank.projections.count == 2)
        #expect(Set(bank.parameters().flattened().map(\.0)) == keys)
        if bank.projections.count == 2 { #expect(bank.projections[1] === untouched) }
        #expect((bank.projections[0] is PrefillQuantizedLinear) == QuantizedPrefill.isAvailable)
    }

    @Test func mixedContainersPreserveEveryUntouchedModule() throws {
        let eligible = makeLinear()
        let custom = OffsetQuantizedLinear(
            weight: eligible.weight, scales: eligible.scales,
            biases: eligible.biases, groupSize: 64, bits: 4)
        custom.train(false)
        let ineligible = makeLinear(dtype: .float32)
        for original: [Linear] in [
            [custom, eligible], [eligible, custom],
            [custom, eligible, ineligible, eligible, custom],
        ] {
            let bank = ProjectionBank(original)
            try QuantizedPrefill.prepare(bank, rows: 2048)
            #expect(bank.projections.count == original.count)
            for (before, after) in zip(original, bank.projections) {
                if before === eligible {
                    #expect((after is PrefillQuantizedLinear) == QuantizedPrefill.isAvailable)
                } else {
                    #expect(before === after)
                }
            }
        }
        let bank = DictionaryBank(["eligible": eligible, "custom": custom, "fp32": ineligible])
        let keys = Set(bank.parameters().flattened().map(\.0))
        try QuantizedPrefill.prepare(bank, rows: 2048)
        #expect(Set(bank.projections.keys) == ["eligible", "custom", "fp32"])
        #expect(Set(bank.parameters().flattened().map(\.0)) == keys)
        #expect(bank.projections["custom"] === custom)
        #expect(bank.projections["fp32"] === ineligible)
        #expect(
            (bank.projections["eligible"] is PrefillQuantizedLinear) == QuantizedPrefill.isAvailable
        )
    }

    @Test func nonreplaceablePropertiesStayUntouched() throws {
        let stock = makeLinear()
        let immutable = ImmutableProjection(stock)
        let tuple = TupleBank(stock)
        let nested = NestedBank(stock)
        #expect(throws: UpdateError.self) {
            try immutable.update(
                modules: ModuleChildren(values: [
                    "projection": .value(PrefillQuantizedLinear(stock))
                ]),
                verify: .none)
        }
        try QuantizedPrefill.prepare(immutable, rows: 2048)
        try QuantizedPrefill.prepare(tuple, rows: 2048)
        try QuantizedPrefill.prepare(nested, rows: 2048)
        #expect(immutable.projection === stock)
        #expect(tuple.projections.0 === stock)
        #expect(tuple.projections.4 === stock)
        #expect(nested.projections[0][0] === stock)
        #expect(nested.optionalProjections.count == 2)
        #expect(nested.optionalProjections[0] == nil)
        #expect(nested.optionalProjections[1] === stock)
        let x = MLXRandom.normal([1, 500, 1024], dtype: .bfloat16)
        identical(stock(x), immutable.projection(x))
    }

    @Test func laterReplacementFailureLeavesEarlierChangesCoherent() throws {
        guard QuantizedPrefill.isAvailable else { return }
        let stock = makeLinear()
        let x = MLXRandom.normal([1, 2048, 1024], dtype: .bfloat16)
        for mutateBeforeFailure in [false, true] {
            let bank = FailingBank(stock, mutateBeforeFailure: mutateBeforeFailure)
            let reference = bank.trace(bank, x)
            eval(reference)
            #expect(bank.trace.isCompiled)
            let enabled = try QuantizedPrefill.prepare(bank, rows: 2048)
            #expect(!enabled)
            #expect(bank.a is PrefillQuantizedLinear)
            #expect((bank.z is PrefillQuantizedLinear) == mutateBeforeFailure)
            #expect(!bank.trace.isCompiled)
            #expect(bank.children()["a"]?.flattenedValues().first === bank.a)
            #expect(bank.children()["z"]?.flattenedValues().first === bank.z)
            identical(reference, bank.trace(bank, x))
            identical(stock(x), QuantizedPrefill.withPrefill(enabled: enabled) { bank.z(x) })
        }
    }

    @Test(arguments: [2, 4, 8])
    func stockPolicySkipsDiscoveryBeforeAndAfterAutomaticRequests(bits: Int) throws {
        guard QuantizedPrefill.isAvailable else { return }
        let model = LlamaModel(
            .init(
                hiddenSize: 256, hiddenLayers: 1, intermediateSize: 512,
                attentionHeads: 8, rmsNormEps: 0.000001, vocabularySize: 64, kvHeads: 2))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        quantize(model: model, groupSize: 64, bits: bits)
        let probe = PreparationProbe(model)
        eval(probe)
        #expect(PrefillParameters().quantizedProjections == .stock)
        let step = bits == 2 ? 512 : 2048
        let prompt = MLXArray((0 ... step * 2).map { Int32($0 % 64) })
        var stockOutputs = [MLXArray]()
        var stockCaches = [[KVCache]]()
        for policy: PrefillParameters.QuantizedProjections in [.stock, .automatic, .stock] {
            let cache = try probe.newCache(parameters: nil)
            let routes = ProgressLog()
            let observe: @Sendable (QuantizedPrefill.Decision) -> Void = {
                if $0 == .dense { routes.append(1) }
            }
            probe.discoveries = 0
            let result = try QuantizedPrefill.$observer.withValue(observe) {
                try probe.prepare(
                    LMInput(tokens: prompt), cache: cache, state: nil,
                    prefill: .init(stepSize: step, quantizedProjections: policy))
            }
            guard case .tokens(let tail) = result else {
                Issue.record("Expected token tail")
                return
            }
            let output = probe(tail.tokens[.newAxis], cache: cache)
            eval(output, cache)
            if policy == .stock {
                #expect(probe.discoveries == 0)
                #expect(routes.counts.isEmpty)
                stockOutputs.append(output)
                stockCaches.append(cache)
            } else {
                #expect(probe.discoveries > 0)
                #expect(!routes.counts.isEmpty)
            }
        }
        identical(stockOutputs[0], stockOutputs[1])
        for (a, b) in zip(stockCaches[0], stockCaches[1]) {
            #expect(a.offset == b.offset)
            for (x, y) in zip(a.state, b.state) { identical(x, y) }
        }
    }

    @Test func replacementCancellationRefreshesModulesAndPropagates() throws {
        guard QuantizedPrefill.isAvailable else { return }
        let bank = FailingBank(
            makeLinear(inputs: 256, outputs: 384), mutateBeforeFailure: true,
            failure: CancellationError())
        let x = MLXArray.ones([1, 2048, 256], dtype: .bfloat16)
        eval(bank.trace(bank, x))
        #expect(throws: CancellationError.self) {
            try QuantizedPrefill.prepare(bank, rows: 2048)
        }
        #expect(!bank.trace.isCompiled)
        #expect(bank.children()["z"]?.flattenedValues().first === bank.z)
        #expect(!PrefillExecutionContext.isActive)
    }

    @Test func failedMetadataRefreshDoesNotSilentlyContinue() throws {
        guard QuantizedPrefill.isAvailable else { return }
        let bank = FailingBank(makeLinear(inputs: 256, outputs: 384))
        bank.rejectRefresh = true
        #expect(throws: FailingBank.Failure.self) {
            try QuantizedPrefill.prepare(bank, rows: 2048)
        }
    }

    @Test func shortPromptOutputDoesNotDependOnEarlierPreparation() throws {
        let configuration = LlamaConfiguration(
            hiddenSize: 1024, hiddenLayers: 1, intermediateSize: 2048,
            attentionHeads: 32, rmsNormEps: 0.000001, vocabularySize: 16, kvHeads: 2)
        let fresh = LlamaModel(configuration)
        fresh.update(parameters: fresh.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: fresh, groupSize: 64, bits: 4)
        fresh.train(false)
        let prepared = LlamaModel(configuration)
        prepared.update(parameters: prepared.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: prepared, groupSize: 64, bits: 4)
        prepared.update(parameters: fresh.parameters())
        prepared.train(false)
        try QuantizedPrefill.prepare(prepared, rows: 2048)
        for length in [480, 481, 500, 512, 2048, 2049] {
            let prompt = MLXArray((0 ..< length).map { Int32($0 % 16) })
            let a = try fresh.newCache(parameters: nil)
            let b = try prepared.newCache(parameters: nil)
            let first = try fresh.prepare(
                LMInput(tokens: prompt), cache: a, state: nil, prefill: .init(chunking: .unchunked))
            let second = try prepared.prepare(
                LMInput(tokens: prompt), cache: b, state: nil, prefill: .init(chunking: .unchunked))
            guard case .tokens(let x) = first, case .tokens(let y) = second else {
                Issue.record("Short text prompts must remain unchunked")
                return
            }
            identical(fresh(x.tokens[.newAxis], cache: a), prepared(y.tokens[.newAxis], cache: b))
            for (a, b) in zip(a, b) {
                #expect(a.offset == b.offset)
                for (x, y) in zip(a.state, b.state) { identical(x, y) }
            }
        }
    }

    @Test func cancellationDuringRoutedPrefillRestoresScopeAndFinishesCacheWrites() async throws {
        let task = Task { @MainActor in
            guard QuantizedPrefill.isAvailable else { return }
            let model = LlamaModel(
                .init(
                    hiddenSize: 256, hiddenLayers: 2, intermediateSize: 512,
                    attentionHeads: 8, rmsNormEps: 0.000001, vocabularySize: 64, kvHeads: 2))
            model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
            quantize(model: model, groupSize: 64, bits: 4)
            model.train(false)
            let caches = try model.newCache(parameters: nil)
            let prompt = MLXArray((0 ..< 4097).map { Int32($0 % 64) })
            let routes = ProgressLog()
            let observe: @Sendable (QuantizedPrefill.Decision) -> Void = {
                if $0 == .dense { routes.append(1) }
            }
            let prefill = PrefillParameters(
                stepSize: 2048, quantizedProjections: .automatic,
                progress: { _, _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                })
            #expect(throws: CancellationError.self) {
                try QuantizedPrefill.$observer.withValue(observe) {
                    try model.prepare(
                        LMInput(tokens: prompt), cache: caches, state: nil, prefill: prefill)
                }
            }
            #expect(!routes.counts.isEmpty)
            #expect(!PrefillExecutionContext.isActive && QuantizedPrefill.observer == nil)
            let reference = try model.newCache(parameters: nil)
            _ = QuantizedPrefill.withPrefill {
                model(prompt[..<2048][.newAxis], cache: reference)
            }
            eval(reference)
            for (actual, expected) in zip(caches, reference) {
                #expect(actual.offset == 2048 && expected.offset == 2048)
                #expect(actual.state.allSatisfy { $0.dim(2) == 2048 })
                for (x, y) in zip(actual.state, expected.state) { identical(x, y) }
            }
        }
        try await task.value
    }

    @Test func cancellationAfterSubmissionPreservesOnlyWrittenCacheRows() async throws {
        let task = Task { @MainActor in
            let model = LlamaModel(
                .init(
                    hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64,
                    attentionHeads: 4, rmsNormEps: 0.000001, vocabularySize: 16, kvHeads: 2))
            model.train(false)
            let cache = try model.newCache(parameters: nil)
            let prompt = MLXArray((0 ..< 1537).map { Int32($0 % 16) })
            let prefill = PrefillParameters(progress: { _, _ in
                withUnsafeCurrentTask { $0?.cancel() }
            })
            #expect(throws: CancellationError.self) {
                try model.prepare(
                    LMInput(tokens: prompt), cache: cache, state: nil, prefill: prefill)
            }
            let simple = try #require(cache.first as? KVCacheSimple)
            #expect(simple.offset == 512)
            #expect(simple.keys?.dim(2) == 1792)
            #expect(simple.state.allSatisfy { $0.dim(2) == 512 })
            let reference = try model.newCache(parameters: nil)
            eval(model(prompt[..<512][.newAxis], cache: reference), reference)
            for (x, y) in zip(simple.state, reference[0].state) { identical(x, y) }
            _ = simple.trim(511)
            let row = MLXArray.ones([1, 2, 1, 8])
            _ = simple.update(keys: row, values: row)
            #expect(simple.offset == 2)
            #expect(simple.keys?.dim(2) == 1792)
            #expect(simple.state.allSatisfy { $0.dim(2) == 2 })
            eval(simple)
        }
        try await task.value
    }

    @Test func sharedPreparationPreservesSchedulesAndWarmStateAcrossModelFamilies() throws {
        let qwen = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {"hidden_size":1024,"num_hidden_layers":2,"intermediate_size":2048,
                 "num_attention_heads":32,"num_key_value_heads":2,"head_dim":32,
                 "vocab_size":64,"rms_norm_eps":0.000001,"tie_word_embeddings":true}
                """.utf8))
        let models: [any LLMModel] = [
            Qwen3Model(qwen),
            LlamaModel(
                .init(
                    hiddenSize: 1024, hiddenLayers: 2, intermediateSize: 2048,
                    attentionHeads: 32, rmsNormEps: 0.000001, vocabularySize: 64, kvHeads: 2)),
        ]
        for model in models {
            model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
            quantize(model: model, groupSize: 64, bits: 4)
            eval(model)
            let prompt = MLXArray((0 ..< 1537).map { Int32($0 % 64) })
            for prefill in [
                PrefillParameters(), .init(stepSize: 256),
                .init(stepSize: 512, chunking: .remainder), .init(chunking: .unchunked),
            ] {
                let referenceCache = try model.newCache(parameters: nil)
                let optimizedCache = try model.newCache(parameters: nil)
                for _ in 0 ..< 2 {
                    var outputs = [MLXArray]()
                    var schedules = [[Int]]()
                    for (automatic, cache) in [(false, referenceCache), (true, optimizedCache)] {
                        model.train(false)
                        let log = ProgressLog()
                        var prefill = prefill
                        prefill.quantizedProjections = automatic ? .automatic : .stock
                        prefill.progress = { processed, _ in log.append(processed) }
                        let result = try model.prepare(
                            LMInput(tokens: prompt), cache: cache,
                            state: nil, prefill: prefill)
                        guard case .tokens(let tail) = result else {
                            Issue.record("Text preparation must return a token tail")
                            return
                        }
                        let output = model(tail.tokens[.newAxis], cache: cache)
                        eval(output, cache)
                        outputs.append(output)
                        schedules.append(log.counts)
                    }
                    identical(outputs[0], outputs[1])
                    #expect(schedules[0] == schedules[1])
                    if prefill.chunking == .balanced, prefill.stepSize == nil {
                        #expect(schedules[1] == [512, 1024, 1536])
                    }
                    for (a, b) in zip(referenceCache, optimizedCache) {
                        #expect(a.offset == b.offset)
                        for (x, y) in zip(a.state, b.state) { identical(x, y) }
                    }
                }
            }
        }
    }
}
