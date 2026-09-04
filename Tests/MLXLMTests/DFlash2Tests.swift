// Copyright © 2026 Apple Inc.

// Synthetic tests for the DFlash2 drafter and its speculative iterator.
// Real-weight parity with the Python reference needs the checkpoints and
// lives with the integration tests.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Configuration

private func configJSON(
    hiddenSize: Int = 64, layers: Int = 2, heads: Int = 4, kvHeads: Int = 2,
    headDim: Int = 16, intermediate: Int = 96, vocab: Int = 64, window: Int = 8,
    topK: Int = 4, rank: Int = 8, layerType: String = "sliding_attention",
    isCausal: String = "false"
) -> String {
    let layerTypes = Array(repeating: "\"\(layerType)\"", count: layers).joined(separator: ", ")
    return """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": \(isCausal),
          "dflash_config": {
            "block_size": 5,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": \(vocab - 2),
            "selector_rank": \(rank),
            "selector_top_k": \(topK),
            "target_layer_ids": [1, 3]
          },
          "head_dim": \(headDim),
          "hidden_size": \(hiddenSize),
          "intermediate_size": \(intermediate),
          "layer_types": [\(layerTypes)],
          "max_position_embeddings": 1024,
          "model_type": "qwen3",
          "num_attention_heads": \(heads),
          "num_hidden_layers": \(layers),
          "num_key_value_heads": \(kvHeads),
          "num_target_layers": 4,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
          "sliding_window": \(window),
          "vocab_size": \(vocab)
        }
        """
}

private func tinyConfig(window: Int = 8, vocab: Int = 64) -> DFlash2Configuration {
    try! JSONDecoder.json5().decode(
        DFlash2Configuration.self, from: Data(configJSON(vocab: vocab, window: window).utf8))
}

@Test
func testDFlash2ConfigurationDecodesReleaseCheckpoint() throws {
    // Mirrors z-lab/Qwen3.8-27B-DFlash2's config.json.
    let json = """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": false,
          "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
          },
          "head_dim": 128,
          "hidden_size": 5120,
          "intermediate_size": 17408,
          "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
          "max_position_embeddings": 262144,
          "model_type": "qwen3",
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "num_target_layers": 64,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
          "sliding_window": 2048,
          "tie_word_embeddings": false,
          "vocab_size": 248320
        }
        """
    let config = try JSONDecoder.json5().decode(
        DFlash2Configuration.self, from: Data(json.utf8))
    #expect(config.hiddenSize == 5120)
    #expect(config.hiddenLayers == 5)
    #expect(config.kvHeads == 8)
    #expect(config.headDim == 128)
    #expect(config.ropeTheta == 10_000_000)
    #expect(config.slidingWindow == 2048)
    #expect(config.contextWindow == 2047)
    #expect(config.numTargetLayers == 64)
    #expect(config.dflash.blockSize == 8)
    #expect(config.dflash.maskTokenId == 248070)
    #expect(config.dflash.selectorTopK == 16)
    #expect(config.dflash.targetLayerIds == [5, 19, 33, 47, 61])
}

@Test
func testDFlash2ConfigurationRejectsUnsupportedShapes() {
    let causal = Data(configJSON(isCausal: "true").utf8)
    #expect(throws: DecodingError.self) {
        try JSONDecoder.json5().decode(DFlash2Configuration.self, from: causal)
    }
    let full = Data(configJSON(layerType: "full_attention").utf8)
    #expect(throws: DecodingError.self) {
        try JSONDecoder.json5().decode(DFlash2Configuration.self, from: full)
    }
}

// MARK: - Context cache

@Test
func testDFlash2ContextCachePlaceholdersResolve() {
    let cache = DFlash2ContextCache(window: 4)
    let rows = MLXArray.ones([1, 1, 3, 2])
    cache.append(keys: rows, values: rows, positions: [0, 1, 2])
    #expect(cache.count == 3)
    #expect(cache.rowValid == [false, false, false])

    cache.resolve(newest: 3, valid: 2)
    #expect(cache.rowValid == [true, true, false])

    let more = MLXArray.ones([1, 1, 2, 2]) * 2
    cache.append(keys: more, values: more, positions: [2, 3])
    cache.resolve(newest: 2, valid: 2)
    #expect(cache.count == 5)
    #expect(cache.rowPositions == [0, 1, 2, 2, 3])
    #expect(cache.rowValid == [true, true, false, true, true])
}

@Test
func testDFlash2ContextCacheCompactionKeepsNewestValidRows() {
    let cache = DFlash2ContextCache(window: 4)
    var tag: Float = 0
    var position: Int32 = 0
    // 86 appends fill 258 of the 260-row buffer; the next one compacts.
    for _ in 0 ..< 86 {
        let rows = MLXArray.full([1, 1, 3, 2], values: MLXArray(tag))
        cache.append(keys: rows, values: rows, positions: [position, position + 1, position + 2])
        // The last row of every append stays a placeholder.
        cache.resolve(newest: 3, valid: 2)
        tag += 1
        position += 3
    }
    // The newest 4 valid rows survive, in order, and the new rows follow.
    let rows = MLXArray.full([1, 1, 3, 2], values: MLXArray(tag))
    cache.append(keys: rows, values: rows, positions: [position, position + 1, position + 2])
    #expect(cache.count == 7)
    #expect(cache.rowValid == [true, true, true, true, false, false, false])
    let values = cache.values![0, 0, 0..., 0].asArray(Float.self)
    #expect(values == [tag - 2, tag - 2, tag - 1, tag - 1, tag, tag, tag])
    #expect(cache.rowPositions.suffix(3) == [position, position + 1, position + 2])
}

// MARK: - Visibility mask

@Test
func testDFlash2VisibilityMaskWindowsAndHidesPlaceholders() {
    let cache = DFlash2ContextCache(window: 8)
    let old = MLXArray.ones([1, 1, 4, 2])
    cache.append(keys: old, values: old, positions: [0, 1, 2, 3])
    cache.resolve(newest: 4, valid: 4)
    let new = MLXArray.ones([1, 1, 2, 2])
    cache.append(keys: new, values: new, positions: [4, 5])

    // One of the two appended rows is valid; the block starts at 5.
    let mask = DFlash2DraftModel.visibilityMask(
        cache: cache, appended: 2, validRows: MLXArray(Int32(1)),
        blockPosition: MLXArray([Int32(5)]), width: 3, window: 3)
    #expect(mask.shape == [3, 9])
    let expected: [[Bool]] = [
        [false, false, false, true, true, false, true, true, true],
        [false, false, false, false, true, false, true, true, true],
        [false, false, false, false, false, false, true, true, true],
    ]
    #expect(mask.asArray(Bool.self) == expected.flatMap { $0 })
}

// MARK: - Dynamic conv

@Test
func testDFlash2DynamicConvMatchesNaive() throws {
    let (hiddenSize, kernelSize, groupSize) = (16, 2, 4)
    let conv = DFlash2DynamicConv(
        hiddenSize: hiddenSize, kernelSize: kernelSize, groupSize: groupSize)

    MLXRandom.seed(7)
    let base = MLXRandom.normal([2, kernelSize, hiddenSize]).asType(.bfloat16)
    let proj = MLXRandom.normal([2 * kernelSize * (hiddenSize / groupSize), hiddenSize])
        .asType(.bfloat16)
    try conv.update(
        parameters: ModuleParameters.unflattened([
            ("base_kernel", base), ("kernel_projection.weight", proj),
        ]),
        verify: [])

    let hidden = MLXRandom.normal([1, 5, hiddenSize]).asType(.bfloat16)
    let (prepared, kernel) = conv.prepare(hidden)
    let finished = conv.finish(hidden, kernel: kernel)
    eval(prepared, finished)

    let fullDyn = conv.kernelProjection(hidden).reshaped(
        1, 5, 2, kernelSize, hiddenSize / groupSize)
    let kernel0 = fullDyn[0..., 0..., 0, 0..., 0...]
    eval(kernel0)

    func naive(_ x: MLXArray, baseRow: Int, dynamic: MLXArray) -> [[Float]] {
        let xF = x.asType(.float32)
        let groups = hiddenSize / groupSize
        let dynF = dynamic.asType(.float32)
        let baseF = base.asType(.float32)
        var out = [[Float]](repeating: [Float](repeating: 0, count: hiddenSize), count: 5)
        for t in 0 ..< 5 {
            for tap in 0 ..< kernelSize {
                let src = t - tap
                guard src >= 0 else { continue }
                for g in 0 ..< groups {
                    let d = dynF[0, t, tap, g].item(Float.self)
                    for c in 0 ..< groupSize {
                        let ch = g * groupSize + c
                        let kVal = baseF[baseRow, tap, ch].item(Float.self) + d
                        out[t][ch] += kVal * xF[0, src, ch].item(Float.self)
                    }
                }
            }
        }
        return out
    }

    // bf16 rounding in the module vs f32 accumulation in the loop.
    let expectedPre = naive(hidden, baseRow: 0, dynamic: kernel0)
    let preparedF = prepared.asType(.float32)
    let expectedFin = naive(hidden, baseRow: 1, dynamic: kernel)
    let finishedF = finished.asType(.float32)
    for t in 0 ..< 5 {
        for ch in 0 ..< hiddenSize {
            #expect(abs(preparedF[0, t, ch].item(Float.self) - expectedPre[t][ch]) < 0.15)
            #expect(abs(finishedF[0, t, ch].item(Float.self) - expectedFin[t][ch]) < 0.15)
        }
    }
}

// MARK: - Selector

@Test
func testDFlash2SelectorGreedyPrefersCoherentPath() throws {
    // vocab 12, topK 3, rank 4, hidden 8. Position 0's top-1 candidate is a
    // decoy; the selector must switch to the candidate the anchor points at.
    let config = try JSONDecoder.json5().decode(
        DFlash2Configuration.self,
        from: Data(
            configJSON(
                hiddenSize: 8, layers: 1, heads: 1, kvHeads: 1, headDim: 8, intermediate: 8,
                vocab: 12, window: 4, topK: 3, rank: 4
            ).utf8))
    let selector = DFlash2CandidateSelector(config)

    var proj = [Float](repeating: 0, count: 4 * 8)
    for i in 0 ..< 4 { proj[i * 8 + i] = 1 }
    try selector.hiddenProjection.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(proj, [4, 8]))]),
        verify: [])
    var pred = [Float](repeating: 0, count: 12 * 4)
    pred[9 * 4 + 0] = 1
    var succ = [Float](repeating: 0, count: 12 * 4)
    succ[5 * 4 + 0] = 50
    try selector.predecessorCodebook.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(pred, [12, 4]))]),
        verify: [])
    try selector.successorCodebook.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(succ, [12, 4]))]),
        verify: [])
    eval(selector.parameters())

    // pos 0: raw top-1 is 7, but 5 wins through the anchor edge; pos 1: 2.
    var logits = [Float](repeating: -30, count: 2 * 12)
    logits[0 * 12 + 7] = 10
    logits[0 * 12 + 5] = 9
    logits[0 * 12 + 1] = 8
    logits[1 * 12 + 2] = 10
    logits[1 * 12 + 3] = 9
    logits[1 * 12 + 4] = 8
    let proposal = selector.select(
        hidden: MLXArray.ones([1, 2, 8]), logits: MLXArray(logits, [1, 2, 12]),
        anchor: MLXArray([Int32(9)]), temperature: 0)
    eval(proposal.tokens, proposal.candidates)
    #expect(proposal.probabilities == nil)
    #expect(proposal.tokens.asArray(Int32.self) == [5, 2])
    #expect(proposal.candidates.shape == [1, 2, 3])

    let sampled = selector.select(
        hidden: MLXArray.ones([1, 2, 8]), logits: MLXArray(logits, [1, 2, 12]),
        anchor: MLXArray([Int32(9)]), temperature: 1)
    #expect(sampled.probabilities?.shape == [1, 2, 3])
}

// MARK: - Mock target and drafter

/// Target with one attention cache and one recurrent cache. Its logits
/// depend only on the input token, `argmax = transition(token)`, so any
/// decoding of the same prompt is deterministic and comparable.
private final class MockTarget: Module, LanguageModel, KVCacheDimensionProvider,
    DFlash2TargetModel
{
    static let vocab = 128
    static let hidden = 64
    static let Hk = 1
    static let Dk = 32
    static let Hv = 2
    static let Dv = 16
    static let K = 4
    static var convDim: Int { 2 * Hk * Dk + Hv * Dv }

    var kvHeads: [Int] { [1] }
    let embedding = Embedding(embeddingCount: vocab, dimensions: hidden)
    private(set) var verifyCalls = 0
    private(set) var verifyWidths: [Int] = []

    /// Per-position token script for the verify rows; nil uses `transition`.
    var script: [Int32]?
    private var scriptIndex = 0

    override init() {
        super.init()
    }

    static func transition(_ token: Int) -> Int { (token * 7 + 3) % vocab }

    private func logits(for tokens: MLXArray) -> MLXArray {
        let ids = tokens.flattened().asArray(Int32.self).map(Int.init)
        var data = [Float](repeating: 0, count: ids.count * Self.vocab)
        for (i, token) in ids.enumerated() {
            let best: Int
            if let script {
                best = scriptIndex < script.count ? Int(script[scriptIndex]) : 0
                scriptIndex += 1
            } else {
                best = Self.transition(token)
            }
            data[i * Self.vocab + best] = 10
            data[i * Self.vocab + (best + 1) % Self.vocab] = 9
            data[i * Self.vocab + (best + 2) % Self.vocab] = 8
        }
        return MLXArray(data, [1, ids.count, Self.vocab])
    }

    private func writeKV(_ cache: [KVCache], positions: Int) {
        for entry in cache {
            if let simple = entry as? KVCacheSimple {
                let kv = MLXArray.zeros([1, 1, positions, 4])
                _ = simple.update(keys: kv, values: kv)
            } else if let mamba = entry as? MambaCache {
                mamba.advance(positions)
                mamba.offset += positions
            }
        }
    }

    // LanguageModel

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if let cache { writeKV(cache, positions: inputs.dim(-1)) }
        return logits(for: inputs)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: self(input.tokens, cache: cache))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let mamba = MambaCache()
        mamba[0] = MLXArray.zeros([1, Self.K - 1, Self.convDim], dtype: .bfloat16)
        mamba[1] = MLXArray.zeros([1, Self.Hv, Self.Dv, Self.Dk], dtype: .float32)
        return [KVCacheSimple(), mamba]
    }

    // DFlash2TargetModel

    var dflash2LayerCount: Int { 4 }
    var dflash2Embedding: Embedding { embedding }
    var dflash2Head: Linear? { nil }

    func dflash2SupportsCache(_ cache: [KVCache]) -> Bool {
        cache.count == 2 && cache[0] is KVCacheSimple && cache[1] is MambaCache
    }

    func dflash2Prefill(
        _ tokens: MLXArray, cache: [KVCache], captureLayers: [Int]
    ) -> (logits: MLXArray, hidden: [MLXArray]) {
        let positions = tokens.dim(-1)
        writeKV(cache, positions: positions)
        return (
            logits(for: tokens),
            captureLayers.map { _ in MLXArray.zeros([1, positions, Self.hidden]) }
        )
    }

    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [KVCache]) -> DFlash2VerifyResult {
        verifyCalls += 1
        let positions = request.tokens.dim(-1)
        verifyWidths.append(positions)
        let attention = cache[0] as! KVCacheSimple
        let kv = MLXArray.zeros([1, 1, positions, 4])
        _ = attention.writeRows(
            keys: kv, values: kv, position: request.position,
            visibleLength: request.positionUpperBound + positions)
        let mamba = cache[1] as! MambaCache
        MLXRandom.seed(UInt64(1000 + verifyCalls))
        let capture = GatedDeltaCapture(
            convInput: MLXRandom.normal([1, Self.K - 1 + positions, Self.convDim])
                .asType(.bfloat16),
            q: MLXRandom.normal([1, positions, Self.Hk, Self.Dk]).asType(.bfloat16),
            k: MLXRandom.normal([1, positions, Self.Hk, Self.Dk]).asType(.bfloat16),
            v: MLXRandom.normal([1, positions, Self.Hv, Self.Dv]).asType(.bfloat16),
            a: MLXRandom.normal([1, positions, Self.Hv]).asType(.bfloat16),
            b: MLXRandom.normal([1, positions, Self.Hv]).asType(.bfloat16),
            aLog: MLXRandom.normal([Self.Hv]).asType(.float32),
            dtBias: MLXRandom.normal([Self.Hv]).asType(.float32),
            initialState: mamba[1]!)
        return DFlash2VerifyResult(
            logits: logits(for: request.tokens),
            hidden: request.captureLayers.map { _ in
                MLXArray.zeros([1, positions, Self.hidden])
            },
            recurrentCaptures: [capture])
    }
}

/// Drafter that proposes from a script, or follows the target's transition
/// with a wrong token every `missEvery` positions.
private final class MockDrafter: Module, DFlash2DrafterModel {
    var script: [[Int32]] = []
    var missEvery = 0
    private(set) var proposeCalls = 0
    private(set) var contextRows: [Int] = []
    private(set) var contextPositions: [Int] = []
    /// Committed context rows visible at each proposal, before its append.
    private(set) var committedRows: [Int] = []

    init(script: [[Int32]] = [], missEvery: Int = 0) {
        self.script = script
        self.missEvery = missEvery
        super.init()
    }

    var blockSize: Int { 4 }
    var maskTokenId: Int { MockTarget.vocab - 1 }
    var targetLayerIds: [Int] { [1, 2] }
    var targetLayerCount: Int { 4 }
    var contextWindow: Int { 16 }

    func makeState() -> DFlash2DrafterState {
        DFlash2DrafterState(contextCaches: [DFlash2ContextCache(window: contextWindow)])
    }

    func propose(
        block: MLXArray, targetHidden: MLXArray, contextPosition: Int, validRows: MLXArray,
        temperature: Float, target: any DFlash2TargetModel, state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        let round = proposeCalls
        proposeCalls += 1
        contextRows.append(targetHidden.dim(1))
        contextPositions.append(contextPosition)
        committedRows.append(state.contextCaches[0].rowValid.filter { $0 }.count)
        let rows = targetHidden.dim(1)
        let kv = MLXArray.zeros([1, 1, rows, 2])
        state.contextCaches[0].append(
            keys: kv, values: kv, positions: (0 ..< rows).map { Int32(contextPosition + $0) })

        let width = block.dim(1) - 1
        var tokens: [Int32]
        if script.isEmpty {
            // The anchor is lazy; sync it the way a real drafter never has to.
            var token = Int(block[0, 0].item(Int32.self))
            tokens = []
            for i in 0 ..< width {
                token = MockTarget.transition(token)
                let miss = missEvery > 0 && (round * width + i) % missEvery == missEvery - 1
                tokens.append(Int32(miss ? (token + 5) % MockTarget.vocab : token))
            }
        } else {
            tokens = Array(script[min(round, script.count - 1)].prefix(width))
        }
        let candidates: [Int32] = tokens.flatMap { [$0, 30, 31, 29] }
        let probabilities: [Float] = tokens.flatMap { _ in [0.9, 0.05, 0.025, 0.025] }
        return DFlash2Proposal(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            candidates: MLXArray(candidates, [1, width, 4]),
            probabilities: MLXArray(probabilities, [1, width, 4]))
    }
}

// MARK: - Iterator

@Test
func testDFlash2IteratorAcceptsPrefixAndTakesBonus() throws {
    // Prefill runs in two chunks (2 + 1); the first token is script[2].
    let target = MockTarget()
    target.script = [
        1, 1, 10,
        20, 21, 55, 0,  // round 1: accept 20, 21; 99 corrected to 55
        30, 77, 0, 0,  // round 2: accept 30; 31 corrected to 77
        77, 77, 77, 77,  // round 3: all accepted
    ]
    let drafter = MockDrafter(script: [[20, 21, 99], [30, 31, 32], [77, 77, 77]])
    var parameters = GenerateParameters(maxTokens: 20)
    parameters.temperature = 0
    var iterator = try DFlash2SpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), mainModel: target,
        drafter: drafter, parameters: parameters)

    var produced: [Int] = []
    while let token = iterator.next() {
        produced.append(token)
    }
    #expect(produced.prefix(10) == [10, 20, 21, 55, 30, 77, 77, 77, 77, 77])
    #expect(produced.count == 20)
    #expect(iterator.acceptedDraftTokens >= 6)
    #expect(iterator.speculativeDecodingTelemetry?.emittedTokenCount == 20)
    // Rounds narrow to what maxTokens leaves.
    #expect(target.verifyWidths.allSatisfy { $0 <= 4 })
    #expect(drafter.contextRows.first == 3)
    #expect(drafter.contextPositions.prefix(3) == [0, 3, 6])
    // The prompt rows are committed before round 1; each round then adds
    // its anchor plus the accepted drafts.
    #expect(drafter.committedRows.prefix(4) == [0, 3, 6, 8])
}

@Test
func testDFlash2IteratorMatchesTokenIteratorWithPenalties() throws {
    // Greedy decoding with a repetition penalty: the target's argmax alone
    // would cycle, so the processed history decides. Both iterators must emit
    // the same stream, whatever the drafter proposes.
    var parameters = GenerateParameters(maxTokens: 40, repetitionPenalty: 1.8)
    parameters.temperature = 0
    parameters.repetitionContextSize = 6
    let prompt = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4, 5]))

    var reference = try TokenIterator(
        input: prompt, model: MockTarget(), parameters: parameters)
    var expected: [Int] = []
    while let token = reference.next() {
        expected.append(token)
    }
    #expect(expected.count == 40)

    for missEvery in [0, 2, 5] {
        var iterator = try DFlash2SpeculativeTokenIterator(
            input: prompt, mainModel: MockTarget(), drafter: MockDrafter(missEvery: missEvery),
            parameters: parameters)
        var produced: [Int] = []
        while let token = iterator.next() {
            produced.append(token)
        }
        #expect(produced == expected, "missEvery \(missEvery)")
    }
}

@Test
func testDFlash2IteratorWarmStartMatchesColdStream() throws {
    let target = MockTarget()
    target.script = [1, 1, 10, 20, 21, 55, 0, 30, 77, 0, 0, 77, 77, 77, 77]
    let drafter = MockDrafter(script: [[20, 21, 99], [30, 31, 32], [77, 77, 77]])
    let cache = target.newCache(parameters: nil)
    _ = target(LMInput.Text(tokens: MLXArray([Int32(1), 2])), cache: cache, state: nil)

    var parameters = GenerateParameters(maxTokens: 8)
    parameters.temperature = 0
    var iterator = try DFlash2SpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), mainModel: target,
        drafter: drafter, mainCache: cache, prefilledPrefixTokens: 2, parameters: parameters)

    var produced: [Int] = []
    while let token = iterator.next() {
        produced.append(token)
    }
    #expect(produced == [10, 20, 21, 55, 30, 77, 77, 77])
    // Only the suffix position is captured; the first context is one row.
    #expect(drafter.contextRows.first == 1)
    #expect(drafter.contextPositions.first == 2)
}

@Test
func testDFlash2IteratorFinalizeRewindsUndrainedDrafts() throws {
    let script: [Int32] = [1, 1, 10, 20, 21, 55, 0, 30, 77, 0, 0, 77, 77, 77, 77]
    let drafts: [[Int32]] = [[20, 21, 99], [30, 31, 32], [77, 77, 77]]
    var parameters = GenerateParameters(maxTokens: 20)
    parameters.temperature = 0

    func run(draining count: Int) throws -> (attention: Int, recurrent: Int) {
        let target = MockTarget()
        target.script = script
        let cache = target.newCache(parameters: nil)
        var iterator = try DFlash2SpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), mainModel: target,
            drafter: MockDrafter(script: drafts), mainCache: cache, parameters: parameters)
        for _ in 0 ..< count { _ = iterator.next() }
        iterator.finalizeGeneration()
        return (cache[0].offset, cache[1].offset)
    }

    // 10 20 21 55 30 drained: 30 sits at position 7, nothing to rewind.
    #expect(try run(draining: 5) == (8, 8))
    // 10 20 drained: 21 at position 5 was committed but never drained.
    #expect(try run(draining: 2) == (5, 5))
    // 10 20 21 55 drained: the bonus has no cache entry yet.
    #expect(try run(draining: 4) == (6, 6))
}

@Test
func testDFlash2IteratorRejectsUnsupportedInputs() throws {
    let target = MockTarget()
    let drafter = MockDrafter()
    let parameters = GenerateParameters(maxTokens: 4)
    #expect(throws: DFlash2SpeculationError.promptTooShort) {
        _ = try DFlash2SpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2])), mainModel: target,
            drafter: drafter, prefilledPrefixTokens: 2, parameters: parameters)
    }
    let image = LMInput(
        text: .init(tokens: MLXArray([Int32(1), 2])),
        image: .init(pixels: MLXArray.zeros([1, 4, 4, 3])))
    #expect(throws: DFlash2SpeculationError.textOnly) {
        _ = try DFlash2SpeculativeTokenIterator(
            input: image, mainModel: target, drafter: drafter, parameters: parameters)
    }
}

// MARK: - Recurrent replay

@Test
func testGatedDeltaCaptureReplayMatchesPrefix() throws {
    let (B, S, Hk, Dk, Hv, Dv, K) = (1, 6, 2, 32, 4, 16, 4)
    let convDim = 2 * Hk * Dk + Hv * Dv
    MLXRandom.seed(11)
    let capture = GatedDeltaCapture(
        convInput: MLXRandom.normal([B, K - 1 + S, convDim]).asType(.bfloat16),
        q: MLXRandom.normal([B, S, Hk, Dk]).asType(.bfloat16),
        k: MLXRandom.normal([B, S, Hk, Dk]).asType(.bfloat16),
        v: MLXRandom.normal([B, S, Hv, Dv]).asType(.bfloat16),
        a: MLXRandom.normal([B, S, Hv]).asType(.bfloat16),
        b: MLXRandom.normal([B, S, Hv]).asType(.bfloat16),
        aLog: MLXRandom.normal([Hv]).asType(.float32),
        dtBias: MLXRandom.normal([Hv]).asType(.float32),
        initialState: MLXRandom.normal([B, Hv, Dv, Dk]).asType(.float32))

    for validCount in [1, 4, S] {
        // Lazy count, as the iterator passes it.
        let count = MLXArray(Int32(validCount - 1)) + Int32(1)
        let (recurrent, conv) = capture.replay(validCount: count)
        let (_, expected) = gatedDeltaUpdate(
            q: capture.q[0..., ..<validCount], k: capture.k[0..., ..<validCount],
            v: capture.v[0..., ..<validCount], a: capture.a[0..., ..<validCount],
            b: capture.b[0..., ..<validCount], aLog: capture.aLog, dtBias: capture.dtBias,
            state: capture.initialState)
        eval(recurrent, expected, conv)
        #expect((recurrent - expected).abs().max().item(Float.self) < 1e-3)
        let expectedConv = capture.convInput[0..., validCount ..< (validCount + K - 1), 0...]
        #expect((conv - expectedConv).abs().max().item(Float.self) == 0)
    }
}

// MARK: - Sampling

@Test
func testDFlash2RejectionSampleFullAgreementAcceptsAll() {
    let (gamma, K, V) = (3, 2, 10)
    let drafts: [Int32] = [2, 3, 4]
    var candidates: [Int32] = []
    var probs: [Float] = []
    var target = [Float](repeating: 0, count: (gamma + 1) * V)
    for (i, token) in drafts.enumerated() {
        candidates += [token, (token + 1) % Int32(V)]
        probs += [1, 0]
        target[i * V + Int(token)] = 1
    }
    target[gamma * V + 7] = 1
    let result = DFlash2SpeculativeTokenIterator.rejectionSample(
        drafts: MLXArray(drafts), targetProbs: MLXArray(target, [gamma + 1, V]),
        draftProbs: MLXArray(probs, [gamma, K]), candidates: MLXArray(candidates, [gamma, K]))
    let packed = result.packed.asArray(Int32.self)
    #expect(packed == [2, 3, 4, 3, 7])
}

@Test
func testDFlash2RejectionSampleZeroProbabilityRejectsFirst() {
    let (gamma, K, V) = (3, 2, 10)
    let drafts: [Int32] = [2, 3, 4]
    let candidates: [Int32] = [2, 5, 3, 5, 4, 5]
    let probs: [Float] = [1, 0, 1, 0, 1, 0]
    var target = [Float](repeating: 0, count: (gamma + 1) * V)
    target[6] = 1  // row 0 gives the draft no mass; all of it sits on 6
    let result = DFlash2SpeculativeTokenIterator.rejectionSample(
        drafts: MLXArray(drafts), targetProbs: MLXArray(target, [gamma + 1, V]),
        draftProbs: MLXArray(probs, [gamma, K]), candidates: MLXArray(candidates, [gamma, K]))
    #expect(result.packed.asArray(Int32.self) == [2, 3, 4, 0, 6])
}

@Test
func testSpeculativeSamplingProbabilitiesMatchTopPSampler() {
    // top-k 2 and top-p leave the two best; temperature reshapes them.
    let logits = MLXArray([Float(0), 1, 2, 3], [1, 4])
    let probs = speculativeSamplingProbabilities(
        logits, temperature: 0.5, topP: 0.9, minP: 0, topK: 2
    ).asArray(Float.self)
    #expect(probs[0] == 0 && probs[1] == 0)
    let e2 = exp(Float(2) / 0.5)
    let e3 = exp(Float(3) / 0.5)
    #expect(abs(probs[2] - e2 / (e2 + e3)) < 1e-5)
    #expect(abs(probs[3] - e3 / (e2 + e3)) < 1e-5)

    // The sampler's own draws stay inside the same support.
    let sampler = TopPSampler(temperature: 0.5, topP: 0.9, topK: 2, minP: 0)
    for _ in 0 ..< 20 {
        let token = sampler.sample(logits: logits).item(Int32.self)
        #expect(token >= 2)
    }
}

// MARK: - Compiled drafter

@Test
func testDFlash2CompiledProposalMatchesEager() throws {
    let config = tinyConfig(window: 24, vocab: MockTarget.vocab)
    MLXRandom.seed(11)
    let compiled = DFlash2DraftModel(config)
    let eager = DFlash2DraftModel(config)
    eager.compiledTracesEnabled = false
    try eager.update(parameters: compiled.parameters(), verify: [])
    let target = MockTarget()

    var stateA = compiled.makeState()
    var stateB = eager.makeState()
    let concatDim = config.dflash.targetLayerIds.count * config.hiddenSize
    // A 20-row first context takes the eager projection; later rounds trace.
    let rounds: [(rows: Int, valid: Int, width: Int)] = [
        (20, 20, 5), (3, 2, 5), (1, 1, 3), (5, 4, 5),
    ]
    var position = 0
    var anchor: Int32 = 7
    for (round, spec) in rounds.enumerated() {
        let targetHidden = MLXRandom.normal([1, spec.rows, concatDim])
        let block = MLXArray(
            [anchor] + Array(repeating: Int32(config.dflash.maskTokenId), count: spec.width - 1)
        ).expandedDimensions(axis: 0)
        let validRows = MLXArray(Int32(spec.valid))
        let a = compiled.propose(
            block: block, targetHidden: targetHidden, contextPosition: position,
            validRows: validRows, temperature: 0, target: target, state: &stateA)
        let b = eager.propose(
            block: block, targetHidden: targetHidden, contextPosition: position,
            validRows: validRows, temperature: 0, target: target, state: &stateB)
        eval(a.tokens, b.tokens, a.candidates, b.candidates)
        #expect(a.tokens.asArray(Int32.self) == b.tokens.asArray(Int32.self), "round \(round)")
        #expect(
            a.candidates.asArray(Int32.self) == b.candidates.asArray(Int32.self),
            "round \(round)")
        for state in [stateA, stateB] {
            state.contextCaches[0].resolve(newest: spec.rows, valid: spec.valid)
        }
        position += spec.valid
        anchor = a.tokens[0, 0].item(Int32.self)
    }
    #expect(compiled.compiledTraceCount >= 6)
    #expect(eager.compiledTraceCount == 0)
}

// MARK: - Same-input stacking

@Test
func testSameInputStackingSkipsQuantizedLinearSubclasses() throws {
    let plain = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
    quantize(model: plain, groupSize: 32, bits: 4)
    let x = MLXRandom.normal([1, 3, 64])
    let before = plain(x)
    #expect(stackSameInputProjections(in: plain) == 1)
    #expect(plain.gateUp != nil)
    let after = plain(x)
    eval(before, after)
    #expect((before - after).abs().max().item(Float.self) == 0)

    let rotated = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
    quantize(
        model: rotated, groupSize: 32, bits: 4,
        apply: { module, groupSize, bits in
            guard let linear = module as? Linear else { return nil }
            return RotateQuantizedLinear(
                inputDims: linear.weight.dim(1), outputDims: linear.weight.dim(0),
                hasBias: false, groupSize: groupSize, bits: bits, krot: 8)
        })
    #expect(stackSameInputProjections(in: rotated) == 0)
    #expect(rotated.gateUp == nil)
    #expect(rotated.gateProj is RotateQuantizedLinear)
}
