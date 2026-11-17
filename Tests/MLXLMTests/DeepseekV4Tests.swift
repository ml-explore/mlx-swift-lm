// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

#if canImport(Metal)
import Metal
#endif

/// DeepSeek-V4 on a 4-layer MXFP4/MXFP8 model built in code, against values from DeepSeek's
/// reference `inference/model.py` written by `tools/deepseek-v4-reference.py`.
final class DeepseekV4Tests: XCTestCase {

    private typealias Reference = DeepseekV4ReferenceValues

    /// Log-probability tolerance against the reference, as `LFM3ReferenceTests`.
    private static let tolerance: Float = 2e-5

    private func configurationData() -> Data {
        Data(Reference.configuration.utf8)
    }

    private func tinyConfiguration() throws -> DeepseekV4Configuration {
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: configurationData())
        try config.validateModelConfiguration()
        return config
    }

    private func skipUnlessLong() throws {
        guard ProcessInfo.processInfo.environment["MLX_DEEPSEEK_V4_LONG"] == "1" else {
            throw XCTSkip("Set MLX_DEEPSEEK_V4_LONG=1 for the 250- and 300-token checks")
        }
    }

    /// The generator's checkpoint as the MLX conversion stores it; same formula as `weights()`
    /// in `tools/deepseek-v4-reference.py`.
    nonisolated(unsafe) private static let checkpoint = makeCheckpoint(experts: 8)

    private static func makeCheckpoint(experts: Int) -> [String: MLXArray] {
        let p = 2_147_483_647
        func hash(_ i: Int, _ j: Int) -> Int {
            let x = (Reference.weightSeed * (1 << 23) + i * (1 << 15) + j) % p
            return (x * x % p) * 48271 % p
        }
        // The e4m3 code of an integer `m`, |m| <= 16 (all exact).
        func e4m3(_ m: Int) -> UInt32 {
            let a = abs(m)
            guard a > 0 else { return 0 }
            let e = Int.bitWidth - 1 - a.leadingZeroBitCount
            return UInt32((m < 0 ? 128 : 0) | (e + 7) << 3 | (((a << 3) >> e) - 8))
        }
        var weights = [String: MLXArray]()
        for (i, parameter) in Reference.parameters.enumerated() {
            let (name, shape, storage) = parameter
            let count = shape.reduce(1, *)
            switch storage {
            case .int32:
                var table = [Int32]()
                for row in 0 ..< shape[0] {
                    var left = Array(0 ..< experts)
                    for column in 0 ..< shape[1] {
                        let pick = hash(i, row * shape[1] + column) % left.count
                        table.append(Int32(left.remove(at: pick)))
                    }
                }
                weights[name] = MLXArray(table, shape)
            case .mxfp4, .mxfp8:
                let bits = storage == .mxfp4 ? 4 : 8
                let codes = (0 ..< count).map {
                    bits == 4 ? UInt32(hash(i, $0) % 16) : e4m3(hash(i, $0) % 33 - 16)
                }
                let perWord = 32 / bits
                let packed = stride(from: 0, to: count, by: perWord).map { start in
                    (0 ..< perWord).reduce(UInt32(0)) { $0 | codes[start + $1] << ($1 * bits) }
                }
                let scales = (0 ..< count / 32).map {
                    UInt8(127 - (bits == 4 ? 5 : 7) + hash(i, count + $0) % 2)
                }
                weights[name] = MLXArray(packed, [shape[0], shape[1] / perWord])
                weights["\(name.dropLast(".weight".count)).scales"] = MLXArray(
                    scales, [shape[0], shape[1] / 32])
            case .float32, .bfloat16:
                let (shift, half, denominator): (Int, Int, Float) =
                    if name.hasSuffix("norm.weight") {
                        (128, 20, 128)
                    } else if name.hasSuffix(".gate.bias") {
                        (0, 20, 256)
                    } else if name.hasSuffix("_scale") {
                        (64, 20, 128)
                    } else {
                        (0, 32, 128)
                    }
                let values = (0 ..< count).map {
                    Float(shift + hash(i, $0) % (2 * half + 1) - half) / denominator
                }
                let value = MLXArray(values, shape)
                weights[name] = storage == .float32 ? value : value.asType(.bfloat16)
            }
        }
        return weights
    }

    /// The checkpoint through `prepareCheckpoint` as `loadWeights` runs it; `float32` casts the
    /// bfloat16 parameters to float32, as the reference runs.
    private func loadTiny(float32: Bool = false) throws -> DeepseekV4Model {
        let config = try tinyConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configurationData())
        let model = DeepseekV4Model(config)
        // The reference's parameter names and shapes, mapped by `sanitize`, are the module tree's.
        let dense = Dictionary(
            uniqueKeysWithValues: Reference.parameters.map { ($0.name, MLXArray.zeros($0.shape)) })
        XCTAssertEqual(
            model.sanitize(weights: dense).mapValues(\.shape),
            Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) }))
        let checkpoint = try model.prepareCheckpoint(
            ModelCheckpoint(
                weights: Self.checkpoint,
                perLayerQuantization: base.perLayerQuantization))
        try applyCheckpoint(checkpoint, to: model)
        materializeModelForInference(model)
        if float32 {
            model.update(
                parameters: model.parameters().mapValues {
                    $0.dtype == .bfloat16 ? $0.asType(.float32) : $0
                })
            eval(model)
        }
        model.train(false)
        return model
    }

    /// Prefill in chunks of `min(step, remaining - 1)`, then one step per entry of `tokens` fed
    /// the previous one (mlx-vlm's `generate_step` plan); returns each step's log-softmax.
    private func teacherForced(
        _ model: DeepseekV4Model, prompt: [Int], step: Int, tokens: [Int]
    ) throws -> [[Float]] {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            let chunk = MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n)
            eval(model(chunk, cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        var input = MLXArray(rest.map(Int32.init)).reshaped(1, 1)
        var logprobs = [[Float]]()
        for token in tokens {
            let logits = model(input, cache: cache)[0, -1].asType(.float32)
            logprobs.append(
                (logits - logSumExp(logits, axis: -1, keepDims: true)).asArray(Float.self))
            input = MLXArray([Int32(token)]).reshaped(1, 1)
        }
        return logprobs
    }

    private static func argmax(_ row: [Float]) -> Int {
        row.indices.max { row[$0] < row[$1] }!
    }

    /// The checkpoint built here is, dtypes and shapes included, the one the generator used.
    func testGeneratedCheckpointMatchesGenerator() throws {
        let names = Reference.parameters.map(\.name)
        XCTAssertEqual(names, names.sorted())
        XCTAssertEqual(try tinyConfiguration().nRoutedExperts, 8)
        var h: UInt64 = 0
        for (name, array) in Self.checkpoint.sorted(by: { $0.key < $1.key }) {
            let (code, values): (UInt64, [UInt64]) =
                switch array.dtype {
                case .bfloat16:
                    (0, array.asType(.float32).asArray(Float.self).map { UInt64($0.bitPattern) })
                case .float32: (1, array.asArray(Float.self).map { UInt64($0.bitPattern) })
                case .int32: (2, array.asArray(Int32.self).map { UInt64(UInt32(bitPattern: $0)) })
                case .uint32: (3, array.asArray(UInt32.self).map(UInt64.init))
                case .uint8: (4, array.asArray(UInt8.self).map(UInt64.init))
                default: (5, [])
                }
            XCTAssertNotEqual(code, 5, name)
            for v in [code] + array.shape.map(UInt64.init) + values {
                h = h &* 1_000_003 &+ v
            }
        }
        XCTAssertEqual(h, Reference.checkpointChecksum)
    }

    /// Exact greedy tokens and the reference's top-k log-probs within tolerance at every step,
    /// teacher-forced in float32 on the CPU, for each prefill chunk size of the run.
    private func assertMatchesReference(_ run: Reference.Run) throws {
        XCTAssertEqual(run.greedy.count, Reference.decode)
        XCTAssertEqual(run.topTokens.count, Reference.decode * Reference.topK)
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(float32: true)
            for step in run.prefillSteps {
                let label = "prompt=\(run.prompt.count) step=\(step)"
                let logprobs = try teacherForced(
                    model, prompt: run.prompt, step: step, tokens: run.greedy)
                XCTAssertEqual(logprobs.map(Self.argmax), run.greedy, label)
                var difference: Float = 0
                for (k, (token, expected)) in zip(run.topTokens, run.topLogprobs).enumerated() {
                    difference = max(
                        difference, abs(logprobs[k / Reference.topK][token] - expected))
                }
                print("deepseek_v4 parity \(label): max |dlogprob| = \(difference)")
                XCTAssertLessThanOrEqual(difference, Self.tolerance, label)
            }
        }
    }

    func testShortPromptMatchesReference() throws {
        // Every reference selection (experts, indexer, argmax) is decided by over 2x tolerance.
        XCTAssertEqual(Reference.selectionMargins.count, 3)
        for (kind, margin) in Reference.selectionMargins {
            XCTAssertGreaterThan(margin, 2 * Self.tolerance, kind)
        }
        try assertMatchesReference(Reference.short)
    }

    func testLongPromptMatchesReference() throws {
        try skipUnlessLong()
        try assertMatchesReference(Reference.long)
    }

    /// Free-running greedy through the default `LLMModel.prepare` and `TokenIterator`, with a
    /// step one short of the prompt, gives the reference's tokens.
    func testTokenIteratorMatchesReferenceThroughDefaultPrepare() throws {
        try Device.withDefaultDevice(.cpu) {
            let run = Reference.short
            let parameters = GenerateParameters(
                maxTokens: Reference.decode, temperature: 0,
                prefill: PrefillParameters(stepSize: run.prompt.count - 1))
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(run.prompt.map(Int32.init))),
                model: try loadTiny(float32: true), parameters: parameters)
            var tokens = [Int]()
            while let token = iterator.next() { tokens.append(token) }
            XCTAssertEqual(tokens, run.greedy)
        }
    }

    /// The fused hyper-connection kernel (GPU) against the op path (CPU) on the streams entering
    /// each block; a bfloat16 collapse may differ by one ulp.
    func testFusedHyperConnectionKernelMatchesOpPathOnGPU() throws {
        #if canImport(Metal)
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil, "no GPU")
        #else
        throw XCTSkip("no GPU")
        #endif
        let prompt = Reference.short.prompt
        for float32 in [false, true] {
            let model = try loadTiny(float32: float32)
            var streams = [MLXArray]()
            try Device.withDefaultDevice(.cpu) {
                let inputs = MLXArray(prompt.prefix(16).map(Int32.init)).reshaped(1, 16)
                let caches = model.makeCache()
                var mask: MLXArray?
                if case .array(let array) = caches[0].makeMask(
                    n: 16, windowSize: model.configuration.slidingWindow, returnArray: true)
                {
                    mask = array
                }
                var h = repeated(
                    expandedDimensions(model.embed(inputs), axis: 2),
                    count: model.configuration.hcMult, axis: 2)
                for (layer, cache) in zip(model.layers, caches) {
                    streams.append(h)
                    h = layer(
                        h, mask: mask, cache: DeepseekV4LayerCache(cache, ratio: layer.attn.ratio),
                        inputIDs: inputs)
                }
                eval(streams)
            }
            var worst: (collapsed: Float, weights: Float) = (0, 0)
            for (layer, h) in zip(model.layers, streams) {
                for connection in [layer.attnHC, layer.ffnHC] {
                    for t in 0 ..< h.dim(1) {
                        let row = h[0..., t ..< (t + 1)]
                        let cpu = Device.withDefaultDevice(.cpu) { connection(row) }
                        let gpu = Device.withDefaultDevice(.gpu) { connection(row) }
                        eval(cpu.0, cpu.1, cpu.2, gpu.0, gpu.1, gpu.2)
                        let (a, b) = (cpu.0.asType(.float32), gpu.0.asType(.float32))
                        let collapsed = abs(a - b).max().item(Float.self)
                        worst.collapsed = max(worst.collapsed, collapsed)
                        worst.weights = max(
                            worst.weights, abs(cpu.1 - gpu.1).max().item(Float.self),
                            abs(cpu.2 - gpu.2).max().item(Float.self))
                        if float32 {
                            XCTAssertLessThanOrEqual(collapsed, 2.9e-5, "collapse")
                        } else {
                            // One bfloat16 ulp is at most 2^-7 of the value.
                            let ulps = abs(a - b) .> abs(a) * Float(1.0 / 128) + 1e-6
                            XCTAssertFalse(any(ulps).item(Bool.self), "bfloat16 collapse")
                        }
                    }
                }
            }
            print(
                "deepseek_v4 fused hc float32=\(float32): max |collapsed| diff = "
                    + "\(worst.collapsed), max |post|,|comb| diff = \(worst.weights)")
            XCTAssertLessThanOrEqual(worst.weights, 3.6e-6, "post and comb, float32=\(float32)")
        }
    }

    /// `nextTokenLogits` drops rows before a quantized head; its last row equals the full forward's.
    func testNextTokenLogitsMatchesFullForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            quantize(model: model) { path, _ in path == "head" ? (32, 4, .affine) : nil }
            eval(model)

            let input = MLXArray((0 ..< 100).map { Int32(2 + $0 % 120) }).reshaped(1, 100)
            let hidden = model.hiddenStates(input, cache: nil)
            XCTAssertLessThan(
                quantizedVocabularyProjectionInput(hidden, projection: model.head).dim(1), 100)

            let full = model(input, cache: try model.newCache(parameters: nil))[0, -1]
            let next = model.nextTokenLogits(
                LMInput.Text(tokens: input), cache: try model.newCache(parameters: nil),
                state: nil
            ).logits[0, -1]
            XCTAssertLessThan(abs(full - next).max().item(Float.self), 1e-4)
        }
    }

    func testCacheTypesFollowCompressRatios() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            let caches = try model.newCache(parameters: nil)
            XCTAssertEqual(caches.count, 4)
            XCTAssertTrue(caches[0] is RotatingKVCache)
            XCTAssertEqual(caches[0].maxSize, 8)
            for (index, ratio) in [(1, 4), (2, 128), (3, 4)] {
                let list = try XCTUnwrap(caches[index] as? CacheList)
                XCTAssertTrue(list[0] is RotatingKVCache)
                XCTAssertEqual(list[0].maxSize, 8)
                XCTAssertEqual((list[1] as? DeepseekV4PoolCache)?.ratio, ratio)
                if ratio == 4 {
                    XCTAssertEqual((list[2] as? DeepseekV4PoolCache)?.ratio, 4)
                }
            }

            // Every cache, pools included, counts the tokens it has seen.
            eval(model(MLXArray((0 ..< 10).map { Int32($0 + 2) }).reshaped(1, 10), cache: caches))
            XCTAssertEqual(caches[0].offset, 10)
            for index in 1 ..< 4 {
                let list = try XCTUnwrap(caches[index] as? CacheList)
                XCTAssertEqual(list[0].offset, 10)
                XCTAssertEqual(list[1].offset, 10)
            }
            XCTAssertEqual((caches[1] as? CacheList)?[2].offset, 10)
            XCTAssertEqual(((caches[1] as? CacheList)?[1] as? DeepseekV4PoolCache)?.pooledCount, 2)
        }
    }

    func testPoolCacheCompletesWindowsAcrossChunksAndDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = DeepseekV4PoolCache(ratio: 4)
            let rows = MLXArray(0 ..< 11).asType(.float32).reshaped(1, 11, 1)
            XCTAssertTrue(cache.overlap)
            // A 6-token chunk completes the first window (nothing precedes it), buffers two rows.
            var ready = cache.accumulateWindows(
                kv: rows[0..., 0 ..< 6], gate: rows[0..., 0 ..< 6])
            XCTAssertEqual(ready.kv.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(ready.base, 0)
            XCTAssertNil(ready.previousKV)
            XCTAssertNil(ready.previousGate)
            XCTAssertEqual(cache.remainder, 2)
            // A 3-token chunk completes the next window, preceded by the previous call's window.
            ready = cache.accumulateWindows(
                kv: rows[0..., 6 ..< 9], gate: rows[0..., 6 ..< 9])
            XCTAssertEqual(ready.kv.asArray(Float.self), [4, 5, 6, 7])
            XCTAssertEqual(ready.base, 4)
            XCTAssertEqual(ready.previousKV?.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(ready.previousGate?.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(cache.remainder, 1)
            _ = cache.appendPooled(MLXArray.zeros([1, 2, 1]))
            // Decode steps complete the next window on its fourth token.
            for t in 9 ..< 11 {
                ready = cache.accumulateWindows(
                    kv: rows[0..., t ..< (t + 1)], gate: rows[0..., t ..< (t + 1)])
                XCTAssertEqual(ready.kv.dim(1), 0)
                XCTAssertNil(ready.previousKV)
            }
            ready = cache.accumulateWindows(
                kv: MLXArray([Float(11)]).reshaped(1, 1, 1),
                gate: MLXArray([Float(11)]).reshaped(1, 1, 1))
            XCTAssertEqual(ready.kv.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(ready.base, 8)
            XCTAssertEqual(ready.previousKV?.asArray(Float.self), [4, 5, 6, 7])
            XCTAssertEqual(ready.previousGate?.asArray(Float.self), [4, 5, 6, 7])
            XCTAssertEqual(cache.previousKV?.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(cache.offset, 12)

            // A multi-token call that completes no window appends all its rows to the buffer.
            let short = DeepseekV4PoolCache(ratio: 4)
            for (start, end) in [(0, 1), (1, 3)] {
                let part = short.accumulateWindows(
                    kv: rows[0..., start ..< end], gate: rows[0..., start ..< end])
                XCTAssertEqual(part.kv.dim(1), 0)
            }
            XCTAssertEqual(short.remainder, 3)
            let joined = short.accumulateWindows(kv: rows[0..., 3 ..< 6], gate: rows[0..., 3 ..< 6])
            XCTAssertEqual(joined.kv.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(joined.gate.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(short.remainder, 2)

            // Ratio-128 pools do not overlap and keep no previous window.
            let wide = DeepseekV4PoolCache(ratio: 128)
            XCTAssertFalse(wide.overlap)
            let wideRows = MLXArray(0 ..< 300).asType(.float32).reshaped(1, 300, 1)
            let wideReady = wide.accumulateWindows(kv: wideRows, gate: wideRows)
            XCTAssertEqual(wideReady.kv.dim(1), 256)
            XCTAssertNil(wideReady.previousKV)
            XCTAssertNil(wide.previousKV)
            XCTAssertNil(wide.accumulateWindows(kv: wideRows, gate: wideRows).previousKV)

            // Two pooled entries: visible to query t iff index < (t + 1) / 4.
            let mask = try XCTUnwrap(cache.poolMask(queries: 6, offset: 3))
            XCTAssertEqual(
                mask.asArray(Bool.self),
                [true, false, true, false, true, false, true, false, true, true, true, true])
            XCTAssertNil(cache.poolMask(queries: 1, offset: 12))

            let copy = try XCTUnwrap(cache.copy() as? DeepseekV4PoolCache)
            XCTAssertEqual(copy.remainder, cache.remainder)
            XCTAssertEqual(copy.pooledCount, 2)
            XCTAssertEqual(copy.offset, 12)
            XCTAssertEqual(copy.previousKV?.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(copy.previousGate?.asArray(Float.self), [8, 9, 10, 11])

            // The state carries the previous window: the restored cache hands it to its next
            // window, as the original does.
            XCTAssertEqual(cache.state.count, 5)
            let restored = DeepseekV4PoolCache(ratio: 4)
            restored.state = cache.state
            restored.metaState = cache.metaState
            XCTAssertEqual(restored.pooledCount, 2)
            XCTAssertEqual(restored.offset, 12)
            let next = MLXArray((12 ..< 16).map(Float.init)).reshaped(1, 4, 1)
            for resumed in [cache, copy, restored] {
                let window = resumed.accumulateWindows(kv: next, gate: next)
                XCTAssertEqual(window.kv.asArray(Float.self), [12, 13, 14, 15])
                XCTAssertEqual(window.base, 12)
                XCTAssertEqual(window.previousKV?.asArray(Float.self), [8, 9, 10, 11])
                XCTAssertEqual(window.previousGate?.asArray(Float.self), [8, 9, 10, 11])
            }
        }
    }

    /// Ratio-4 pooling against `overlap_transform` written out: entry `w` pools the first half of
    /// window `w - 1` and the second half of window `w`, `ape` added to both.
    func testOverlapCompressionMatchesReferenceWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(float32: true)
            for compressor in [
                try XCTUnwrap(model.layers[1].attn.compressor),
                try XCTUnwrap(model.layers[1].attn.indexer?.compressor),
            ] {
                XCTAssertTrue(compressor.overlap)
                let (ratio, d) = (compressor.ratio, compressor.headDim)
                let x = MLXRandom.normal(
                    [1, 23, model.configuration.hiddenSize], key: MLXRandom.key(5))
                let pool = compressor(x, cache: DeepseekV4PoolCache(ratio: ratio))
                XCTAssertEqual(pool.shape, [1, 5, d])

                let kv = compressor.wkv(x)[0]
                let gate = compressor.wgate(x)[0]
                var expected = [MLXArray]()
                for w in 0 ..< 5 {
                    let current = (w * ratio) ..< ((w + 1) * ratio)
                    let previousKV: MLXArray
                    let previousGate: MLXArray
                    if w == 0 {
                        previousKV = MLXArray.zeros([ratio, d])
                        previousGate = MLXArray.full(
                            [ratio, d], values: MLXArray(-Float.infinity), type: Float.self)
                    } else {
                        let before = ((w - 1) * ratio) ..< (w * ratio)
                        previousKV = kv[before, ..<d]
                        previousGate = gate[before, ..<d] + compressor.ape[0..., ..<d]
                    }
                    let rows = concatenated([previousKV, kv[current, d...]], axis: 0)
                    let gates = concatenated(
                        [previousGate, gate[current, d...] + compressor.ape[0..., d...]], axis: 0)
                    expected.append((rows * softmax(gates, axis: 0, precise: true)).sum(axis: 0))
                }
                var reference = compressor.norm(stacked(expected).reshaped(1, 5, d))
                reference = compressor.rope(expandedDimensions(reference, axis: 1), offset: 0)
                    .squeezed(axis: 1)
                let difference = abs(pool - reference).max().item(Float.self)
                print("deepseek_v4 overlap vs reference windows: max |d| = \(difference)")
                XCTAssertLessThanOrEqual(difference, 1e-5)
            }
        }
    }

    /// On the bfloat16 checkpoint, pooling and routing equal exactly the reference's float32
    /// computation cast back where the reference casts.
    func testBFloat16PoolingAndRoutingFollowReferenceFloat32() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            let hidden = model.configuration.hiddenSize
            let x = MLXRandom.normal([1, 23, hidden], key: MLXRandom.key(7)).asType(.bfloat16)
            let x32 = x.asType(.float32)

            for compressor in [
                try XCTUnwrap(model.layers[1].attn.compressor),
                try XCTUnwrap(model.layers[1].attn.indexer?.compressor),
            ] {
                XCTAssertTrue(compressor.overlap)
                // Kept in float32 at load; bfloat16 values, so the expectation below is the same.
                XCTAssertEqual(compressor.wkv.weight.dtype, .float32)
                let (ratio, d) = (compressor.ratio, compressor.headDim)
                let cache = DeepseekV4PoolCache(ratio: ratio)
                let pool = compressor(x, cache: cache)
                XCTAssertEqual(pool.dtype, .bfloat16)
                XCTAssertEqual(cache.previousKV?.dtype, .float32, "buffered rows")

                let kv = matmul(x32, compressor.wkv.weight.asType(.float32).T)[0]
                let gate = matmul(x32, compressor.wgate.weight.asType(.float32).T)[0]
                let ape = compressor.ape.asType(.float32)
                var expected = [MLXArray]()
                for w in 0 ..< 5 {
                    let current = (w * ratio) ..< ((w + 1) * ratio)
                    let previousKV: MLXArray
                    let previousGate: MLXArray
                    if w == 0 {
                        previousKV = MLXArray.zeros([ratio, d])
                        previousGate = MLXArray.full(
                            [ratio, d], values: MLXArray(-Float.infinity), type: Float.self)
                    } else {
                        let before = ((w - 1) * ratio) ..< (w * ratio)
                        previousKV = kv[before, ..<d]
                        previousGate = gate[before, ..<d] + ape[0..., ..<d]
                    }
                    let rows = concatenated([previousKV, kv[current, d...]], axis: 0)
                    let gates = concatenated(
                        [previousGate, gate[current, d...] + ape[0..., d...]], axis: 0)
                    expected.append((rows * softmax(gates, axis: 0, precise: true)).sum(axis: 0))
                }
                var reference = compressor.norm(
                    stacked(expected).reshaped(1, 5, d).asType(.bfloat16))
                reference = compressor.rope(expandedDimensions(reference, axis: 1), offset: 0)
                    .squeezed(axis: 1)
                XCTAssertEqual(reference.dtype, .bfloat16)
                let difference = abs(pool.asType(.float32) - reference.asType(.float32)).max()
                    .item(Float.self)
                print("deepseek_v4 bf16 overlap pooling vs float32: max |d| = \(difference)")
                XCTAssertEqual(difference, 0, "bfloat16 overlap pooling")
            }

            let ids = MLXArray((0 ..< 23).map { Int32(2 + $0 * 5 % 120) }).reshaped(1, 23)
            var routerWorst: Float = 0
            for layer in model.layers {
                let gate = layer.ffn.gate
                let (indices, weights) = gate(x, inputIDs: ids)
                XCTAssertEqual(weights.dtype, .float32)

                let logits = matmul(x32, gate.weight.asType(.float32).T)
                let scores = sqrt(softplus(logits))
                if let tid2eid = gate.tid2eid {
                    XCTAssertEqual(
                        indices.asArray(Int32.self),
                        take(tid2eid, ids, axis: 0).asArray(Int32.self))
                } else {
                    let biased = scores + gate.bias!.asType(.float32)
                    let top = argPartition(-biased, kth: gate.topK - 1, axis: -1)[
                        .ellipsis, ..<gate.topK
                    ].asType(.int32)
                    XCTAssertEqual(
                        sorted(top, axis: -1).asArray(Int32.self),
                        sorted(indices, axis: -1).asArray(Int32.self), "routed experts")
                }
                var expected = takeAlong(scores, indices, axis: -1)
                expected = expected / expected.sum(axis: -1, keepDims: true) * gate.scaling
                let difference = abs(weights - expected).max().item(Float.self)
                routerWorst = max(routerWorst, difference)
                XCTAssertEqual(difference, 0, "bfloat16 router weights")
            }
            print("deepseek_v4 bf16 router weights vs float32 reference: max |d| = \(routerWorst)")
        }
    }

    /// The bfloat16 load keeps the reference's float32 weights in float32; the logits equal
    /// those of casting the weights back to bfloat16.
    func testBFloat16LoadKeepsReferenceFloat32Weights() throws {
        try skipUnlessLong()
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            let parameters = model.parameters().flattened()
            let kept = parameters.filter { DeepseekV4Weights.isFloat32Weight($0.0) }
            // 4 routers, 5 compressors (attention and indexer) with wkv and wgate, the head;
            // norms: 4 per layer (attn, ffn, q, kv), one per compressor, the final one.
            XCTAssertEqual(kept.count, 15 + 22)
            // Stored as float32 in the checkpoint, as in the reference.
            let storedFloat32 = ["_hc.", "hc_head.", ".ape", ".attn_sink", ".gate.bias"]
            for (key, value) in parameters {
                if DeepseekV4Weights.isFloat32Weight(key) {
                    XCTAssertEqual(value.dtype, .float32, key)
                } else if !storedFloat32.contains(where: key.contains) {
                    XCTAssertNotEqual(value.dtype, .float32, key)
                }
            }
            XCTAssertEqual(model.layers[1].attn.indexer?.weightsProj.weight.dtype, .bfloat16)

            let castAtUse = try loadTiny()
            let castBack = castAtUse.parameters().flattened().map { key, value in
                (key, DeepseekV4Weights.isFloat32Weight(key) ? value.asType(.bfloat16) : value)
            }
            castAtUse.update(
                parameters: ModuleParameters.unflattened(
                    Dictionary(uniqueKeysWithValues: castBack)))
            eval(castAtUse)
            XCTAssertEqual(castAtUse.head.weight.dtype, .bfloat16)

            let tokens = MLXArray((0 ..< 300).map { Int32(2 + $0 * 7 % 120) }).reshaped(1, 300)
            var outputs = [[MLXArray]]()
            for m in [model, castAtUse] {
                let cache = m.makeCache()
                var logits = [m(tokens, cache: cache)[0, -1]]
                for token in [5, 77, 31] as [Int32] {
                    logits.append(m(MLXArray([token]).reshaped(1, 1), cache: cache)[0, -1])
                }
                eval(logits)
                outputs.append(logits)
            }
            for (a, b) in zip(outputs[0], outputs[1]) {
                XCTAssertEqual(a.dtype, .float32)
                XCTAssertEqual(abs(a - b).max().item(Float.self), 0, "float32 vs cast-at-use")
            }
        }
    }

    /// A norm weight doubled with `update(parameters:)` after a forward gives exactly the logits
    /// of a fresh model with that weight.
    func testNormWeightUpdateReachesNextForward() throws {
        try Device.withDefaultDevice(.cpu) {
            // 40 tokens complete ratio-4 windows, so the compressor norms take part.
            let tokens = MLXArray((0 ..< 40).map { Int32(2 + $0 * 7 % 120) }).reshaped(1, 40)
            func logits(_ model: DeepseekV4Model) -> MLXArray {
                let out = model(tokens, cache: model.makeCache())
                eval(out)
                return out
            }
            for key in [
                "layers.0.ffn_norm.weight", "layers.1.attn_norm.weight",
                "layers.1.attn.compressor.norm.weight", "norm.weight",
            ] {
                let updated = try loadTiny()
                let before = logits(updated)
                let weight = try XCTUnwrap(
                    updated.parameters().flattened().first { $0.0 == key }?.1, key)
                let doubled = weight * 2
                eval(doubled)
                updated.update(parameters: ModuleParameters.unflattened([key: doubled]))
                eval(updated)
                let after = logits(updated)

                let fresh = try loadTiny()
                fresh.update(parameters: ModuleParameters.unflattened([key: doubled]))
                eval(fresh)
                let expected = logits(fresh)

                XCTAssertTrue(any(before .!= expected).item(Bool.self), "\(key) moves the logits")
                XCTAssertEqual(abs(after - expected).max().item(Float.self), 0, key)
            }
        }
    }

    /// Ratio-4 and ratio-128 pools and continuation logits do not depend on how 300 tokens are
    /// split into calls.
    func testPoolsAndLogitsIndependentOfCallSplit() throws {
        try skipUnlessLong()
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(float32: true)
            XCTAssertEqual(model.layers.map(\.attn.ratio), [0, 4, 128, 4])
            let tokens = (0 ..< 300).map { Int32(2 + $0 * 7 % 120) }
            let continuation: [Int32] = [5, 77, 31, 64]
            let splits: [(String, [Int])] = [
                ("one-shot", [300]),
                ("chunks", [7, 1, 130, 5, 61, 96]),
                ("token-by-token", Array(repeating: 1, count: 300)),
            ]

            var runs = [(pools: [MLXArray], logits: [MLXArray])]()
            for (_, sizes) in splits {
                XCTAssertEqual(sizes.reduce(0, +), tokens.count)
                let caches = model.makeCache()
                var start = 0
                for n in sizes {
                    let chunk = MLXArray(Array(tokens[start ..< (start + n)])).reshaped(1, n)
                    eval(model(chunk, cache: caches))
                    eval(caches)
                    start += n
                }
                var logits = [MLXArray]()
                for token in continuation {
                    let step = model(MLXArray([token]).reshaped(1, 1), cache: caches)[0, -1]
                    eval(step)
                    logits.append(step)
                }
                let pools = try [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)].map {
                    item -> MLXArray in
                    let (layer, slot) = item
                    let list = try XCTUnwrap(caches[layer] as? CacheList)
                    let pool = try XCTUnwrap(list[slot] as? DeepseekV4PoolCache)
                    return try XCTUnwrap(pool.pooled)
                }
                runs.append((pools, logits))
            }

            // 304 tokens: 76 ratio-4 entries, 2 ratio-128 entries.
            XCTAssertEqual(runs[0].pools.map { $0.dim(1) }, [76, 76, 2, 76, 76])
            for (index, (label, _)) in splits.enumerated().dropFirst() {
                for (a, b) in zip(runs[0].pools, runs[index].pools) {
                    XCTAssertEqual(a.shape, b.shape, label)
                    let difference = abs(a - b).max().item(Float.self)
                    print("deepseek_v4 \(label) pool max |d| = \(difference)")
                    XCTAssertLessThanOrEqual(difference, 1e-5, "pool, \(label)")
                }
                for (a, b) in zip(runs[0].logits, runs[index].logits) {
                    let difference = abs(a - b).max().item(Float.self)
                    print("deepseek_v4 \(label) logits max |d| = \(difference)")
                    XCTAssertLessThanOrEqual(difference, 1e-4, "logits, \(label)")
                }
            }
        }
    }

    /// The pool grows `step` entries at a time; every reader sees only the live entries.
    func testPoolGrowsInStepsAndReadsLiveEntries() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = DeepseekV4PoolCache(ratio: 4)
            cache.step = 2
            // One buffered row, so the state carries the window buffers as in a model run.
            _ = cache.accumulateWindows(
                kv: MLXArray.zeros([1, 1, 1]), gate: MLXArray.zeros([1, 1, 1]))
            XCTAssertEqual(cache.appendPooled(MLXArray.zeros([1, 0, 1])).dim(1), 0)
            XCTAssertNil(cache.pooled)
            var all = [Float]()
            for values in [[1], [2, 3], [4], [5, 6, 7]] as [[Float]] {
                all += values
                let pool = cache.appendPooled(MLXArray(values).reshaped(1, values.count, 1))
                XCTAssertEqual(pool.asArray(Float.self), all)
                XCTAssertEqual(cache.pooledCount, all.count)
                XCTAssertEqual(cache.pooled?.asArray(Float.self), all)
                XCTAssertEqual(cache.state.last?.shape, [1, all.count, 1])
            }
            XCTAssertEqual(cache.appendPooled(MLXArray.zeros([1, 0, 1])).asArray(Float.self), all)
            XCTAssertEqual(try XCTUnwrap(cache.poolMask(queries: 2, offset: 30)).shape, [2, 7])

            let copy = try XCTUnwrap(cache.copy() as? DeepseekV4PoolCache)
            XCTAssertEqual(
                copy.appendPooled(MLXArray([Float(8)]).reshaped(1, 1, 1))
                    .asArray(Float.self), all + [8])
            XCTAssertEqual(cache.pooled?.asArray(Float.self), all)

            let restored = DeepseekV4PoolCache(ratio: 4)
            restored.state = cache.state
            restored.metaState = cache.metaState
            XCTAssertEqual(restored.pooledCount, 7)
            XCTAssertEqual(restored.pooled?.asArray(Float.self), all)
        }
    }

    func testCheckpointConfigurationTakesOneRatioPerLayer() throws {
        let ratios = [0, 0] + (0 ..< 20).flatMap { _ in [4, 128] } + [4, 0, 0, 0]
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 43,
             "compress_ratios": \(ratios), "num_hash_layers": 3,
             "rope_scaling": {"type": "yarn", "factor": 16,
               "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.compressRatios.count, 43)
        XCTAssertEqual(config.compressRatios.filter { $0 == 4 }.count, 21)
        XCTAssertEqual(config.compressRatios.filter { $0 == 128 }.count, 20)
        XCTAssertEqual(config.compressRatios.last, 4)
        // The checkpoint sets neither key: exact top-k, as the reference.
        XCTAssertEqual(config.indexBlock, 0)
        XCTAssertEqual(config.indexKeep, 0)
    }

    func testRejectsInconsistentConfiguration() throws {
        func configuration(_ change: (inout [String: Any]) -> Void) throws
            -> DeepseekV4Configuration
        {
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: configurationData()) as? [String: Any])
            change(&json)
            return try JSONDecoder().decode(
                DeepseekV4Configuration.self,
                from: JSONSerialization.data(withJSONObject: json))
        }
        XCTAssertThrowsError(
            try configuration { $0["compress_ratios"] = [0, 4, 2, 4] }
                .validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["compress_ratios"] = [0, 4] }.validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["o_groups"] = 3 }.validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["scoring_func"] = "relu" }.validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["rope_scaling"] = ["type": "linear", "factor": 2] }
                .validateModelConfiguration())
        // There is no default layout: a configuration without compress_ratios does not decode.
        XCTAssertThrowsError(try configuration { $0.removeValue(forKey: "compress_ratios") }) {
            guard case DecodingError.keyNotFound(let key, _) = $0 else {
                return XCTFail("expected keyNotFound, got \($0)")
            }
            XCTAssertEqual(key.stringValue, "compress_ratios")
        }
    }

    /// The exact indexer scored in query tiles picks what one untiled pass picks.
    func testExactIndexerTilesMatchUntiled() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(float32: true)
            let attn = model.layers[1].attn
            let indexer = try XCTUnwrap(attn.indexer)
            let hidden = model.configuration.hiddenSize
            let x = MLXRandom.normal([1, 50, hidden], key: MLXRandom.key(3))
            let qResidual = attn.qNorm(attn.wqA(x))
            let cache = DeepseekV4PoolCache(ratio: 4)
            for (start, end) in [(0, 41), (41, 50)] {
                func pick(tile: Int?) throws -> [Int32] {
                    indexer.exactQueryTile = tile
                    defer { indexer.exactQueryTile = nil }
                    let copy = try XCTUnwrap(cache.copy() as? DeepseekV4PoolCache)
                    let picked = try XCTUnwrap(
                        indexer(
                            x[0..., start ..< end], qResidual: qResidual[0..., start ..< end],
                            rope: attn.rope, cache: copy, offset: start))
                    XCTAssertEqual(picked.shape, [1, end - start, indexer.topK])
                    return picked.asArray(Int32.self)
                }
                let untiled = try pick(tile: nil)
                for tile in [1, 3, 8] {
                    XCTAssertEqual(try pick(tile: tile), untiled, "tile \(tile) from \(start)")
                }
                _ = indexer(
                    x[0..., start ..< end], qResidual: qResidual[0..., start ..< end],
                    rope: attn.rope, cache: cache, offset: start)
                eval(cache.innerState())
            }
        }
    }

    func testSanitizeMapsCheckpointLayout() throws {
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 1, "hidden_size": 64,
             "n_routed_experts": 2, "o_groups": 2, "o_lora_rank": 4, "compress_ratios": [0]}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        let weights: [String: MLXArray] = [
            "layers.0.ffn.experts.0.w1.weight": MLXArray.zeros([3, 5]),
            "layers.0.ffn.experts.1.w1.weight": MLXArray.ones([3, 5]),
            "layers.0.ffn.experts.0.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.ffn.experts.1.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.attn.wo_a.weight": MLXArray.zeros([8, 6]),
            "layers.0.hc_attn_fn": MLXArray.zeros([2]),
            "layers.0.hc_ffn_scale": MLXArray.zeros([3]),
            "layers.0.ffn.gate.tid2eid": MLXArray([Int64(1), 0]),
            "layers.0.ffn.shared_experts.w3.weight": MLXArray.zeros([2]),
            "layers.1.attn.wkv.weight": MLXArray.zeros([2]),
            "mtp.0.block.attn.wkv.weight": MLXArray.zeros([2]),
            "hc_head_fn": MLXArray.zeros([2]),
            "embed.weight": MLXArray.zeros([2]),
        ]
        let out = DeepseekV4Weights.sanitize(weights, configuration: config)
        XCTAssertEqual(
            Set(out.keys),
            [
                "layers.0.ffn.switch_mlp.gate_proj.weight",
                "layers.0.ffn.switch_mlp.down_proj.scales", "layers.0.attn.wo_a.weight",
                "layers.0.attn_hc.fn", "layers.0.ffn_hc.scale", "layers.0.ffn.gate.tid2eid",
                "layers.0.ffn.shared_experts.w3.weight", "hc_head.fn", "embed.weight",
            ])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]?.shape, [2, 3, 5])
        XCTAssertEqual(
            out["layers.0.ffn.switch_mlp.gate_proj.weight"]?[1].sum().item(Float.self), 15)
        XCTAssertEqual(out["layers.0.attn.wo_a.weight"]?.shape, [2, 4, 6])
        XCTAssertEqual(out["layers.0.ffn.gate.tid2eid"]?.dtype, .int32)
    }

    /// The checkpoint saved to a file loads to `loadTiny`'s parameters; a model saved with an
    /// extra 8-bit head reloads to the same outputs.
    func testQuantizedCheckpointRoundTrip() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = try loadTiny()
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("deepseek-v4-\(UUID().uuidString)")
            let stored = directory.appendingPathComponent("checkpoint")
            try FileManager.default.createDirectory(
                at: stored, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }

            let config = try tinyConfiguration()
            try save(
                arrays: Self.checkpoint,
                url: stored.appendingPathComponent("model.safetensors"))
            try configurationData().write(to: stored.appendingPathComponent("config.json"))
            let base = try JSONDecoder().decode(
                BaseConfiguration.self, from: configurationData())
            let fromFiles = DeepseekV4Model(config)
            try loadWeights(
                modelDirectory: stored, model: fromFiles,
                perLayerQuantization: base.perLayerQuantization)
            let expected = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            let parameters = fromFiles.parameters().flattened()
            XCTAssertEqual(parameters.count, expected.count)
            for (key, value) in parameters {
                let other = try XCTUnwrap(expected[key], key)
                XCTAssertEqual(value.dtype, other.dtype, key)
                XCTAssertTrue(arrayEqual(value, other).item(Bool.self), key)
            }

            quantize(model: source) { path, _ in path == "head" ? (32, 8, .affine) : nil }
            eval(source)
            let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            XCTAssertNotNil(weights["head.scales"])
            XCTAssertNotNil(weights["layers.1.ffn.switch_mlp.gate_proj.scales"])
            XCTAssertNotNil(weights["layers.1.attn.wo_a.scales"])
            XCTAssertNil(weights["layers.1.attn.compressor.wkv.scales"])
            XCTAssertEqual(weights["layers.0.ffn.gate.tid2eid"]?.dtype, .int32)
            try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

            var perLayer = try XCTUnwrap(base.perLayerQuantization)
            perLayer.perLayerQuantization["head"] = .quantize(.init(groupSize: 32, bits: 8))
            let loaded = DeepseekV4Model(config)
            try loadWeights(
                modelDirectory: directory, model: loaded, perLayerQuantization: perLayer)
            loaded.train(false)

            let input = MLXArray((0 ..< 40).map { Int32(2 + $0 * 3 % 120) }).reshaped(1, 40)
            let (a, b) = (source(input, cache: nil), loaded(input, cache: nil))
            XCTAssertEqual(abs(a - b).max().item(Float.self), 0)
        }
    }

    func testRegisteredModelType() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configurationData(), modelType: "deepseek_v4")
        XCTAssertTrue(model is DeepseekV4Model)
    }
}
