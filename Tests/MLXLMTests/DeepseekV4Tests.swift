// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// DeepSeek-V4 against DeepSeek's reference `inference/model.py` on a 4-layer fixture
/// (`Resources/deepseek_v4_tiny`, from `tools/deepseek_v4_parity/make_tiny_fixture.py`)
/// covering every V4 path. Comparisons run
/// teacher-forced in float32 on the CPU, except the GPU-only fused hyper-connection kernel check.
final class DeepseekV4Tests: XCTestCase {

    private struct Oracle: Decodable {
        struct Case: Decodable {
            let indexer: String
            let indexBlock: Int
            let indexKeep: Int
            let prefillStep: Int
            let prompt: [Int]
            /// The reference's greedy tokens: what each decode step is fed.
            let greedy: [Int]
            let logprobs: [[Float]]
            /// Per decode step, the sorted routed experts of each score-routed layer.
            let experts: [[[Int]]]
            /// Per decode step, the sorted pooled entries the indexer of each ratio-4 layer picks.
            let indexerSelection: [[[Int]]]

            enum CodingKeys: String, CodingKey {
                case indexer, prompt, greedy, logprobs, experts
                case indexBlock = "index_block"
                case indexKeep = "index_keep"
                case prefillStep = "prefill_step"
                case indexerSelection = "indexer_selection"
            }
        }
        let decode: Int
        let tolerance: Float
        let cases: [Case]
    }

    private var fixture: URL {
        get throws {
            try XCTUnwrap(Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        }
    }

    private func configurationData() throws -> Data {
        try Data(contentsOf: fixture.appendingPathComponent("config.json"))
    }

    private func tinyConfiguration() throws -> DeepseekV4Configuration {
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: configurationData())
        try config.validateModelConfiguration()
        return config
    }

    /// The checkpoint as stored (bfloat16 with MXFP4/MXFP8 layers), or with every floating
    /// parameter in float32 as the oracle runs it (quantized weights and scales unchanged).
    private func loadTiny(indexBlock: Int, indexKeep: Int, float32: Bool = false) throws
        -> DeepseekV4Model
    {
        var config = try tinyConfiguration()
        config.indexBlock = indexBlock
        config.indexKeep = indexKeep
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configurationData())
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: fixture, model: model, perLayerQuantization: base.perLayerQuantization)
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

    private func oracle() throws -> Oracle {
        try JSONDecoder().decode(
            Oracle.self, from: Data(contentsOf: fixture.appendingPathComponent("oracle.json")))
    }

    /// mlx-vlm `generate_step`'s plan: prompt chunks of `min(step, remaining - 1)` while more
    /// than one token is left, then the last token, then one step per entry of `tokens`, each
    /// fed the previous entry. Returns the float32 log-softmax of each step.
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

    /// The same teacher-forced run, block by block, returning each decode step's routed experts
    /// (score-routed layers) and indexer picks (ratio-4 layers, run on a copy of the pool cache).
    private func selections(
        _ model: DeepseekV4Model, prompt: [Int], step: Int, tokens: [Int]
    ) throws -> (experts: [[[Int]]], indexer: [[[Int]]]) {
        let caches = model.makeCache()
        let layerCaches = zip(caches, model.layers).map {
            DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
        }
        func sorted(_ indices: MLXArray) -> [Int] {
            indices.asArray(Int32.self).map(Int.init).sorted()
        }
        func forward(_ ids: [Int]) -> (experts: [[Int]], indexer: [[Int]]) {
            let inputs = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            var h = model.embed(inputs)
            h = repeated(expandedDimensions(h, axis: 2), count: model.configuration.hcMult, axis: 2)
            var mask: MLXArray?
            if ids.count > 1,
                case .array(let array) = layerCaches[0].local.makeMask(
                    n: ids.count, windowSize: model.configuration.slidingWindow, returnArray: true)
            {
                mask = array
            }
            var experts = [[Int]]()
            var indexer = [[Int]]()
            for (layer, cache) in zip(model.layers, layerCaches) {
                let (attnInput, attnPost, attnComb) = layer.attnHC(h)
                let x = layer.attnNorm(attnInput)
                if let index = layer.attn.indexer, let pool = cache.indexPool,
                    let copy = pool.copy() as? DeepseekV4PoolCache,
                    let picked = index(
                        x, qResidual: layer.attn.qNorm(layer.attn.wqA(x)), rope: layer.attn.rope,
                        cache: copy, offset: cache.local.offset)
                {
                    indexer.append(sorted(picked))
                }
                let attnOut = layer.attn(x, mask: mask, cache: cache)
                h = deepseekV4HCExpand([attnOut, h, attnPost, attnComb])[0]
                let (ffnInput, ffnPost, ffnComb) = layer.ffnHC(h)
                let y = layer.ffnNorm(ffnInput)
                if layer.ffn.gate.tid2eid == nil {
                    experts.append(sorted(layer.ffn.gate(y, inputIDs: inputs).0))
                }
                h = deepseekV4HCExpand([layer.ffn(y, inputIDs: inputs), h, ffnPost, ffnComb])[0]
            }
            eval(h)
            eval(caches)
            return (experts, indexer)
        }

        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            _ = forward(Array(rest.prefix(n)))
            rest = rest.dropFirst(n)
        }
        var experts = [[[Int]]]()
        var indexer = [[[Int]]]()
        for input in [rest.first!] + tokens.dropLast() {
            let picked = forward([input])
            experts.append(picked.experts)
            indexer.append(picked.indexer)
        }
        return (experts, indexer)
    }

    private static func maxDifference(_ a: [[Float]], _ b: [[Float]]) -> Float {
        zip(a, b).map { mine, theirs in zip(mine, theirs).map { abs($0 - $1) }.max() ?? 0 }
            .max() ?? 0
    }

    private static func argmax(_ row: [Float]) -> Int {
        row.indices.max { row[$0] < row[$1] }!
    }

    func testTinyCheckpointMatchesReferenceTeacherForced() throws {
        try Device.withDefaultDevice(.cpu) {
            let expected = try oracle()
            XCTAssertEqual(expected.cases.count, 6)
            var models = [String: DeepseekV4Model]()
            var worst: Float = 0
            for item in expected.cases {
                let model =
                    try models[item.indexer]
                    ?? loadTiny(
                        indexBlock: item.indexBlock, indexKeep: item.indexKeep, float32: true)
                models[item.indexer] = model
                let label = "\(item.indexer) prompt=\(item.prompt.count) step=\(item.prefillStep)"
                XCTAssertEqual(item.greedy.count, expected.decode)

                let logprobs = try teacherForced(
                    model, prompt: item.prompt, step: item.prefillStep, tokens: item.greedy)
                // The oracle's argmax margins exceed its threshold, so greedy agrees exactly.
                XCTAssertEqual(logprobs.map(Self.argmax), item.greedy, label)
                let difference = Self.maxDifference(logprobs, item.logprobs)
                worst = max(worst, difference)
                print("deepseek_v4 parity \(label): max |dlogprob| = \(difference)")
                XCTAssertLessThanOrEqual(difference, expected.tolerance, label)

                let picked = try selections(
                    model, prompt: item.prompt, step: item.prefillStep, tokens: item.greedy)
                XCTAssertEqual(picked.experts, item.experts, "routed experts, \(label)")
                XCTAssertEqual(picked.indexer, item.indexerSelection, "indexer top-k, \(label)")
            }
            print("deepseek_v4 parity max |dlogprob| = \(worst) tolerance = \(expected.tolerance)")
        }
    }

    /// The default `LLMModel.prepare` and `TokenIterator` with a step one short of the prompt
    /// (mlx-vlm's plan for a short prompt); free-running greedy must give the oracle's tokens.
    func testTokenIteratorMatchesReferenceThroughDefaultPrepare() throws {
        try Device.withDefaultDevice(.cpu) {
            let expected = try oracle()
            let cases = expected.cases.filter { $0.prefillStep >= $0.prompt.count }
            XCTAssertEqual(cases.count, 2)
            for item in cases {
                let model = try loadTiny(
                    indexBlock: item.indexBlock, indexKeep: item.indexKeep, float32: true)
                let parameters = GenerateParameters(
                    maxTokens: expected.decode, temperature: 0,
                    prefill: PrefillParameters(stepSize: item.prompt.count - 1))
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(item.prompt.map(Int32.init))),
                    model: model, parameters: parameters)
                var tokens = [Int]()
                while let token = iterator.next() { tokens.append(token) }
                XCTAssertEqual(tokens, item.greedy, item.indexer)
            }
        }
    }

    /// The fused hyper-connection kernel (GPU) against the op path (CPU), one token at a time on
    /// the streams entering each block for 16 prompt tokens; a bfloat16 collapse may differ by
    /// one ulp.
    func testFusedHyperConnectionKernelMatchesOpPathOnGPU() throws {
        let prompt = try XCTUnwrap(oracle().cases.first { $0.prompt.count == 45 }).prompt
        for float32 in [false, true] {
            let model = try loadTiny(indexBlock: 4, indexKeep: 2, float32: float32)
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

    /// `nextTokenLogits` drops leading rows before a quantized vocabulary projection; the last
    /// row must equal the full forward's.
    func testNextTokenLogitsMatchesFullForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 4, indexKeep: 2)
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
            let model = try loadTiny(indexBlock: 4, indexKeep: 2)
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
            // A 6-token chunk completes one window and buffers two rows.
            XCTAssertTrue(cache.overlap)
            // A 6-token chunk completes one window and buffers two rows. It is the first window
            // of the sequence: nothing precedes it.
            var ready = cache.accumulateWindows(
                kv: rows[0..., 0 ..< 6], gate: rows[0..., 0 ..< 6])
            XCTAssertEqual(ready.kv.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(ready.base, 0)
            XCTAssertNil(ready.previousKV)
            XCTAssertNil(ready.previousGate)
            XCTAssertEqual(cache.remainder, 2)
            // A 3-token chunk joins the buffered rows: one window from 4, preceded by the window
            // the previous call completed.
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

    /// One-shot ratio-4 pooling against the reference's `overlap_transform` written out per
    /// window: entry `w` pools the first half-channels of window `w - 1` (none for `w == 0`) and
    /// the second half-channels of window `w`, `ape` added to both.
    func testOverlapCompressionMatchesReferenceWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
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

    /// On the bfloat16 checkpoint, ratio-4 pooling and routing keep the reference's float32
    /// placement: the compressor projects, buffers, softmaxes and sums in float32 and casts the
    /// entries back before the norm; the router's matmul, scores and weights are float32. The
    /// expected values are those computations done in float32 and cast back where the reference
    /// casts, so the port must match them exactly.
    func testBFloat16PoolingAndRoutingFollowReferenceFloat32() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
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

    /// The normal load of the bfloat16 checkpoint keeps the reference's float32 weights (logits
    /// head, compressor projections, router) in float32 and every other parameter as stored.
    /// Prefill and decode give exactly what casting those weights at use gives, which is also
    /// the path after a caller casts them back to bfloat16.
    func testBFloat16LoadKeepsReferenceFloat32Weights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            let parameters = model.parameters().flattened()
            let kept = parameters.filter { DeepseekV4Weights.isFloat32Weight($0.0) }
            // 4 routers, 5 compressors (attention and indexer) with wkv and wgate, the head.
            XCTAssertEqual(kept.count, 15)
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

            let castAtUse = try loadTiny(indexBlock: 0, indexKeep: 0)
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

    /// The pools of a ratio-4 layer (attention and indexer) and a ratio-128 layer, and the
    /// logits of a continuation, do not depend on how the tokens are split into calls: one
    /// prefill, uneven chunks (not multiples of either ratio), or one token at a time.
    func testPoolsAndLogitsIndependentOfCallSplit() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
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

    /// The exact indexer path scored in query tiles picks exactly what one untiled pass picks,
    /// for a first prefill chunk and for a later one.
    func testExactIndexerTilesMatchUntiled() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
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

    /// Module-path weights saved from a loaded model reload to the same outputs, with the
    /// checkpoint's MXFP4/MXFP8 layers plus an affine 8-bit head.
    func testQuantizedCheckpointRoundTrip() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = try loadTiny(indexBlock: 4, indexKeep: 2)
            quantize(model: source) { path, _ in path == "head" ? (32, 8, .affine) : nil }
            eval(source)

            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("deepseek-v4-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            XCTAssertNotNil(weights["head.scales"])
            XCTAssertNotNil(weights["layers.1.ffn.switch_mlp.gate_proj.scales"])
            XCTAssertNotNil(weights["layers.1.attn.wo_a.scales"])
            XCTAssertNil(weights["layers.1.attn.compressor.wkv.scales"])
            XCTAssertEqual(weights["layers.0.ffn.gate.tid2eid"]?.dtype, .int32)
            try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

            let base = try JSONDecoder().decode(
                BaseConfiguration.self, from: configurationData())
            var perLayer = try XCTUnwrap(base.perLayerQuantization)
            perLayer.perLayerQuantization["head"] = .quantize(.init(groupSize: 32, bits: 8))
            var config = try tinyConfiguration()
            config.indexBlock = 4
            config.indexKeep = 2
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

    /// A real checkpoint's safetensors headers (names, shapes, quantization modes) against the
    /// module tree after sanitize and per-layer quantization; no weight is read.
    func testCheckpointLayoutMatchesModuleTree() throws {
        guard let path = ProcessInfo.processInfo.environment["DEEPSEEK_V4_CHECKPOINT_DIR"] else {
            throw XCTSkip(
                "Set DEEPSEEK_V4_CHECKPOINT_DIR to an MXFP4 MLX conversion of DeepSeek-V4-Flash")
        }
        try Device.withDefaultDevice(.cpu) {
            let directory = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let dtypes: [String: DType] = [
                "BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32,
                "I64": .int64, "I32": .int32,
            ]
            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(atPath: path)
                .filter { $0.hasSuffix(".safetensors") }
            XCTAssertFalse(files.isEmpty)
            for file in files {
                let handle = try FileHandle(
                    forReadingFrom: directory.appendingPathComponent(file))
                defer { try? handle.close() }
                let size = try XCTUnwrap(handle.read(upToCount: 8)).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt64.self)
                }
                let header = try XCTUnwrap(handle.read(upToCount: Int(size)))
                let entries = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: header) as? [String: Any])
                for (name, value) in entries where name != "__metadata__" {
                    let entry = try XCTUnwrap(value as? [String: Any])
                    let shape = try XCTUnwrap(entry["shape"] as? [Int])
                    let dtype = try XCTUnwrap(dtypes[try XCTUnwrap(entry["dtype"] as? String)])
                    weights[name] = MLXArray.zeros(shape, dtype: dtype)
                }
            }
            let model = DeepseekV4Model(config)
            let sanitized = model.sanitize(weights: weights)
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? base.perLayerQuantization?.quantization(layer: path)?.asTuple : nil
            }
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            let modes = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    key, module -> (String, QuantizationMode)? in
                    switch module {
                    case let q as QuantizedLinear: (key, q.mode)
                    case let q as QuantizedSwitchLinear: (key, q.mode)
                    case let q as QuantizedMultiLinear: (key, q.mode)
                    default: nil
                    }
                })
            XCTAssertEqual(modes["layers.2.ffn.switch_mlp.gate_proj"], .mxfp4)
            XCTAssertEqual(modes["layers.2.ffn.shared_experts.w1"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.wo_a"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.indexer.wq_b"], .mxfp8)
            XCTAssertNil(modes["layers.2.attn.compressor.wkv"])
            XCTAssertNil(modes["head"])
        }
    }
}
