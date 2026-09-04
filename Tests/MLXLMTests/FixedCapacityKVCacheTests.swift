import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Regression tests for issue #406.
///
/// When a decode step is traced with ``MLX/compile(inputs:outputs:shapeless:_:)``,
/// every value the trace reads must be threaded through the compile inputs/outputs.
/// ``KVCacheSimple`` keeps its write position as a Swift `Int`, which bakes into
/// the graph as a constant: the compiled step keeps writing the same slot and
/// attending over the same window, so decode falls into repeating token loops.
///
/// ``FixedCapacityKVCache`` tracks the position as an `MLXArray` that rides through
/// compile, so compiled and uncompiled decode stay token-for-token identical.
final class FixedCapacityKVCacheTests: XCTestCase {

    private let hiddenLayers = 2

    /// A tiny deterministic Qwen2 -- enough layers/heads for the mask, RoPE and
    /// GQA paths to matter, cheap enough to run on CI.
    private func makeModel() throws -> Qwen2Model {
        MLXRandom.seed(7)
        let configuration = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: Data(
                """
                {
                    "hidden_size": 16, "num_hidden_layers": \(hiddenLayers),
                    "intermediate_size": 32, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
                    "vocab_size": 64, "tie_word_embeddings": true
                }
                """.utf8))
        return Qwen2Model(configuration)
    }

    private func makeLlamaModel() -> LlamaModel {
        MLXRandom.seed(11)
        return LlamaModel(
            LlamaConfiguration(
                hiddenSize: 16, hiddenLayers: hiddenLayers, intermediateSize: 32,
                attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 64, kvHeads: 1))
    }

    private func simpleCaches() -> [KVCache] {
        (0 ..< hiddenLayers).map { _ in KVCacheSimple() }
    }

    private func compiledCaches(maxTokens: Int = 32) -> [FixedCapacityKVCache] {
        (0 ..< hiddenLayers).map { _ in FixedCapacityKVCache(maxTokens: maxTokens) }
    }

    /// Greedy sample from the last position's logits.
    private func greedy(_ logits: MLXArray) -> Int {
        Int(argMax(logits[0..., -1, 0...]).item(Int32.self))
    }

    /// A single-token model input `[1, 1]`.
    private func tokenInput(_ token: Int) -> MLXArray {
        MLXArray([Int32(token)]).reshaped(1, 1)
    }

    private func assertAllClose(
        _ a: MLXArray, _ b: MLXArray, _ message: String, file: StaticString, line: UInt
    ) {
        XCTAssertTrue(
            allClose(a, b, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            message, file: file, line: line)
    }

    // MARK: - The issue: compiled decode must not freeze

    /// Compiled decode must match the uncompiled reference token-for-token and
    /// keep advancing the write position across steps instead of staying frozen
    /// at the value captured when the graph was traced.
    func testCompiledDecodeMatchesUncompiledTokenForToken() throws {
        let model = try makeModel()
        let prompt = MLXArray([3, 55, 17, 42, 5, 29, 11, 51]).reshaped(1, -1)

        let referenceCaches = simpleCaches()
        let caches = compiledCaches()

        // Eager prefill first: allocates the fixed-capacity buffers before the
        // first compiled call, exactly as the documented usage requires.
        let referencePrefill = model(prompt, cache: referenceCaches)
        let compiledPrefill = model(prompt, cache: caches)
        eval(referencePrefill, compiledPrefill)
        assertAllClose(
            referencePrefill, compiledPrefill, "prefill logits diverged",
            file: #filePath, line: #line)

        let updatables: [any Updatable] = caches
        let compiledDecode = compile(inputs: updatables, outputs: updatables) { token in
            model(token, cache: caches)
        }

        var referenceToken = greedy(referencePrefill)
        var compiledToken = greedy(compiledPrefill)
        XCTAssertEqual(referenceToken, compiledToken, "prefill sample diverged")

        let decodeSteps = 12
        for step in 0 ..< decodeSteps {
            let referenceLogits = model(tokenInput(referenceToken), cache: referenceCaches)
            let compiledLogits = compiledDecode(tokenInput(compiledToken))
            eval(referenceLogits, compiledLogits)

            assertAllClose(
                referenceLogits, compiledLogits, "logits diverged at step \(step)",
                file: #filePath, line: #line)

            referenceToken = greedy(referenceLogits)
            compiledToken = greedy(compiledLogits)
            XCTAssertEqual(
                referenceToken, compiledToken,
                "sampled token diverged at decode step \(step) — the compiled cache "
                    + "write position is not tracking the decode position")
        }

        // The position array must have been threaded through compile: 8 prompt
        // tokens plus one step per decode call. With a frozen Swift-Int offset
        // this stays at the trace-time constant.
        XCTAssertEqual(caches[0].offset, 8 + decodeSteps)

        // And the accumulated cache contents must match the reference exactly.
        for layer in 0 ..< hiddenLayers {
            let referenceState = referenceCaches[layer].state
            let compiledState = caches[layer].state
            XCTAssertEqual(referenceState.count, compiledState.count)
            for index in 0 ..< referenceState.count {
                XCTAssertEqual(
                    referenceState[index].shape, compiledState[index].shape,
                    "layer \(layer) state \(index) shape mismatch")
                assertAllClose(
                    referenceState[index], compiledState[index],
                    "layer \(layer) state \(index) contents diverged",
                    file: #filePath, line: #line)
            }
        }
    }

    /// The second explicitly supported model family gets its own compiled parity
    /// test so protocol conformance remains a checked correctness promise.
    func testLlamaCompiledDecodeMatchesUncompiled() throws {
        let model = makeLlamaModel()
        let prompt = MLXArray([7, 9, 13, 21, 4]).reshaped(1, -1)
        let referenceCaches = simpleCaches()
        let caches = try model.newFixedCapacityCache(maxTokens: 24)

        let referencePrefill = model(prompt, cache: referenceCaches)
        let compiledPrefill = model(prompt, cache: caches)
        eval(referencePrefill, compiledPrefill)
        assertAllClose(
            referencePrefill, compiledPrefill, "Llama prefill logits diverged",
            file: #filePath, line: #line)

        let compiledDecode = compile(inputs: caches, outputs: caches) { token in
            model(token, cache: caches)
        }
        var token = greedy(referencePrefill)
        for step in 0 ..< 8 {
            let reference = model(tokenInput(token), cache: referenceCaches)
            let compiled = compiledDecode(tokenInput(token))
            eval(reference, compiled)
            assertAllClose(
                reference, compiled, "Llama logits diverged at step \(step)",
                file: #filePath, line: #line)
            token = greedy(reference)
        }
        XCTAssertEqual(caches.map(\.offset), [13, 13])
    }

    func testFixedCapacityCacheProviderValidatesCapacity() throws {
        let qwen: any FixedCapacityKVCacheProviding = try makeModel()
        let qwenCaches = try qwen.newFixedCapacityCache(maxTokens: 17)
        XCTAssertEqual(qwenCaches.count, hiddenLayers)
        XCTAssertTrue(qwenCaches.allSatisfy { $0.maxTokens == 17 })

        let llama: any FixedCapacityKVCacheProviding = makeLlamaModel()
        XCTAssertEqual(try llama.newFixedCapacityCache(maxTokens: 9).count, hiddenLayers)

        XCTAssertThrowsError(try qwen.newFixedCapacityCache(maxTokens: 0)) { error in
            XCTAssertEqual(error as? KVCacheConfigurationError, .invalidCapacity(0))
        }
    }

    func testCompiledDecodeSessionOwnsPrefillAndEnforcesCapacity() throws {
        let model = try makeModel()
        let prompt = MLXArray([3, 55, 17]).reshaped(1, -1)
        let referenceCaches = simpleCaches()
        let referencePrefill = model(prompt, cache: referenceCaches)

        let session = try CompiledDecodeSession(model: model, prompt: prompt, capacity: 5)
        eval(referencePrefill, session.prefillLogits)
        assertAllClose(
            referencePrefill, session.prefillLogits, "session prefill logits diverged",
            file: #filePath, line: #line)
        XCTAssertEqual(session.processedTokenCount, 3)
        XCTAssertEqual(session.remainingCapacity, 2)

        var token = greedy(referencePrefill)
        for step in 0 ..< 2 {
            let reference = model(tokenInput(token), cache: referenceCaches)
            let compiled = try session.step(tokenInput(token))
            eval(reference, compiled)
            assertAllClose(
                reference, compiled, "session logits diverged at step \(step)",
                file: #filePath, line: #line)
            token = greedy(reference)
        }

        XCTAssertEqual(session.processedTokenCount, 5)
        XCTAssertEqual(session.remainingCapacity, 0)
        XCTAssertEqual(session.cacheOffsets, [5, 5])
        XCTAssertThrowsError(try session.step(tokenInput(token))) { error in
            XCTAssertEqual(
                error as? CompiledDecodeSessionError,
                .capacityExceeded(capacity: 5))
        }
        XCTAssertEqual(session.cacheOffsets, [5, 5], "rejected step must not mutate cache")
    }

    func testCompiledDecodeSessionValidatesShapes() throws {
        let model = try makeModel()

        XCTAssertThrowsError(
            try CompiledDecodeSession(
                model: model, prompt: MLXArray([1, 2]).reshaped(1, 2), capacity: 0)
        ) { error in
            XCTAssertEqual(error as? CompiledDecodeSessionError, .invalidCapacity(0))
        }

        XCTAssertThrowsError(
            try CompiledDecodeSession(model: model, prompt: MLXArray([1, 2]), capacity: 4)
        ) { error in
            XCTAssertEqual(
                error as? CompiledDecodeSessionError,
                .invalidPromptShape([2]))
        }

        let prompt = MLXArray([1, 2]).reshaped(1, 2)
        let session = try CompiledDecodeSession(model: model, prompt: prompt, capacity: 4)
        XCTAssertThrowsError(try session.step(MLXArray([3, 4]).reshaped(1, 2))) { error in
            XCTAssertEqual(
                error as? CompiledDecodeSessionError,
                .invalidTokenShape([1, 2], expectedBatchSize: 1))
        }
        XCTAssertEqual(session.processedTokenCount, 2)
    }

    /// Opt-in regression against the checkpoint from issue #406. Set
    /// `MLX_QWEN25_7B_PATH` to a local Qwen2.5-7B-Instruct-4bit directory.
    func testQwen25SevenBCompiledDecodeParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_QWEN25_7B_PATH"] else {
            throw XCTSkip("set MLX_QWEN25_7B_PATH to run the Qwen2.5-7B regression")
        }

        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let configurationData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder.json5().decode(
            Qwen2Configuration.self, from: configurationData)
        let baseConfiguration = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: configurationData)
        let model = Qwen2Model(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfiguration.perLayerQuantization)

        let prompt = MLXArray([151644, 8948, 198, 2610, 525, 498, 30]).reshaped(1, -1)
        let referenceCaches = try model.newCache(parameters: nil)
        let referencePrefill = model(prompt, cache: referenceCaches)
        let session = try CompiledDecodeSession(
            model: model, prompt: prompt, capacity: 4_096)
        eval(referencePrefill, session.prefillLogits)
        assertAllClose(
            referencePrefill, session.prefillLogits, "Qwen2.5-7B prefill logits diverged",
            file: #filePath, line: #line)

        var token = greedy(referencePrefill)
        var eagerSeconds = 0.0
        var compiledSeconds = 0.0
        var firstCompiledSeconds = 0.0
        for step in 0 ..< 16 {
            let eagerStart = Date()
            let reference = model(tokenInput(token), cache: referenceCaches)
            eval(reference)
            eagerSeconds += Date().timeIntervalSince(eagerStart)

            let compiledStart = Date()
            let compiled = try session.step(tokenInput(token))
            eval(compiled)
            let compiledStepSeconds = Date().timeIntervalSince(compiledStart)
            compiledSeconds += compiledStepSeconds
            if step == 0 {
                firstCompiledSeconds = compiledStepSeconds
            }
            assertAllClose(
                reference, compiled, "Qwen2.5-7B logits diverged at step \(step)",
                file: #filePath, line: #line)
            let referenceToken = greedy(reference)
            XCTAssertEqual(
                referenceToken, greedy(compiled),
                "Qwen2.5-7B sampled token diverged at step \(step)")
            token = referenceToken
        }
        XCTAssertEqual(session.processedTokenCount, prompt.dim(1) + 16)
        let steadyCompiledSeconds = compiledSeconds - firstCompiledSeconds
        print(
            String(
                format:
                    "[QWEN2.5-7B] eager %.1f tok/s; compiled steady %.1f tok/s; first compiled step %.1f ms",
                16 / eagerSeconds, 15 / steadyCompiledSeconds, firstCompiledSeconds * 1_000))
    }

    /// Prefill on the fixed-capacity cache matches ``KVCacheSimple``: the mask
    /// keeps unwritten slots out of attention.
    func testPrefillMatchesSimpleCache() throws {
        let model = try makeModel()
        let prompt = MLXArray([3, 55, 17, 42, 5, 29, 11, 51]).reshaped(1, -1)

        let reference = simpleCaches()
        let caches = compiledCaches()

        let referenceLogits = model(prompt, cache: reference)
        let logits = model(prompt, cache: caches)
        eval(referenceLogits, logits)

        assertAllClose(
            referenceLogits, logits, "prefill logits diverged",
            file: #filePath, line: #line)
        XCTAssertEqual(caches.map(\.offset), [8, 8])
    }

    /// A continuation chunk after the first prompt exercises the causal mask
    /// inside the chunk with a nonzero write position.
    func testContinuedPrefillMatchesSimpleCache() throws {
        let model = try makeModel()
        let prompt = MLXArray([3, 55, 17, 42, 5]).reshaped(1, -1)
        let continuation = MLXArray([29, 11, 51]).reshaped(1, -1)

        let reference = simpleCaches()
        let caches = compiledCaches()

        eval(model(prompt, cache: reference), model(prompt, cache: caches))

        let referenceLogits = model(continuation, cache: reference)
        let logits = model(continuation, cache: caches)
        eval(referenceLogits, logits)

        assertAllClose(
            referenceLogits, logits, "continuation logits diverged",
            file: #filePath, line: #line)
        XCTAssertEqual(caches.map(\.offset), [8, 8])
    }

    // MARK: - Mask construction

    func assertMaskValues(
        _ cache: FixedCapacityKVCache, n: Int, windowSize: Int?, queryPosition: Int,
        expected: (Int, Int) -> Bool, file: StaticString = #filePath, line: UInt = #line
    ) {
        guard
            case .array(let mask) = cache.makeMask(
                n: n, windowSize: windowSize, returnArray: false)
        else {
            XCTFail("expected an array mask", file: file, line: line)
            return
        }
        eval(mask)
        XCTAssertEqual(mask.shape, [1, 1, n, cache.maxTokens], file: file, line: line)

        for query in 0 ..< n {
            for slot in 0 ..< cache.maxTokens {
                XCTAssertEqual(
                    mask[0, 0, query, slot].item(Bool.self),
                    expected(queryPosition + query, slot),
                    "query \(query) (position \(queryPosition + query)) slot \(slot)",
                    file: file, line: line)
            }
        }
    }

    /// Single-token decode mask: slots up to and including the slot the update
    /// writes into are attendable; the rest of the capacity is not.
    func testDecodeMaskCoversOnlyWrittenSlots() throws {
        let cache = FixedCapacityKVCache(maxTokens: 8)
        let chunk = MLXArray.zeros([1, 1, 3, 4])
        cache.update(keys: chunk, values: chunk)
        XCTAssertEqual(cache.offset, 3)

        // Decode query at position 3; slots 0...3 are written.
        assertMaskValues(cache, n: 1, windowSize: nil, queryPosition: 3) { position, slot in
            slot <= position
        }
    }

    /// The legacy array-based helper receives caches as protocol existentials;
    /// this pins witness-table dispatch for requiresAttentionMask.
    func testRequiredMaskDispatchesThroughKVCacheExistential() throws {
        let concrete = FixedCapacityKVCache(maxTokens: 8)
        let chunk = MLXArray.zeros([1, 1, 3, 4])
        concrete.update(keys: chunk, values: chunk)
        let caches: [KVCache] = [concrete]

        XCTAssertTrue(caches[0].requiresAttentionMask)
        let hidden = MLXArray.zeros([1, 1, 4])
        let optionalMask: MLXArray? = createAttentionMask(h: hidden, cache: caches)
        let mask = try XCTUnwrap(optionalMask)
        eval(mask)
        XCTAssertEqual(mask.shape, [1, 1, 1, 8])
        XCTAssertEqual(
            mask.asArray(Bool.self),
            [true, true, true, true, false, false, false, false])
    }

    /// Multi-token chunk mask: causal within the chunk against the full capacity.
    func testChunkMaskIsCausalAgainstCapacity() throws {
        let cache = FixedCapacityKVCache(maxTokens: 8)
        let chunk = MLXArray.zeros([1, 1, 3, 4])
        cache.update(keys: chunk, values: chunk)

        assertMaskValues(cache, n: 2, windowSize: nil, queryPosition: 3) { position, slot in
            slot <= position
        }
    }

    /// A fresh cache masks from position zero.
    func testEmptyCacheMaskIsCausalFromZero() throws {
        let cache = FixedCapacityKVCache(maxTokens: 8)
        assertMaskValues(cache, n: 3, windowSize: nil, queryPosition: 0) { position, slot in
            slot <= position
        }
    }

    /// windowSize bounds the mask to a trailing window of attendable slots.
    func testMaskHonorsWindowSize() throws {
        let cache = FixedCapacityKVCache(maxTokens: 8)
        let chunk = MLXArray.zeros([1, 1, 5, 4])
        cache.update(keys: chunk, values: chunk)

        assertMaskValues(cache, n: 1, windowSize: 2, queryPosition: 5) { position, slot in
            slot <= position && slot > position - 2
        }
    }

    // MARK: - Cache bookkeeping

    func testOffsetTrimCopyAndState() throws {
        let cache = FixedCapacityKVCache(maxTokens: 8)
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.maxSize, 8)
        XCTAssertTrue(cache.isTrimmable)
        XCTAssertTrue(cache.innerState().isEmpty)
        XCTAssertEqual(cache.state.count, 0)

        func chunk(_ n: Int) -> MLXArray { MLXArray.zeros([1, 1, n, 4]) }

        cache.update(keys: chunk(3), values: chunk(3))
        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(cache.state[0].shape, [1, 1, 3, 4])
        XCTAssertEqual(cache.innerState().map(\.shape).last, [1], "position rides in innerState")

        cache.update(keys: chunk(2), values: chunk(2))
        XCTAssertEqual(cache.offset, 5)

        let copy = cache.copy()
        cache.update(keys: chunk(1), values: chunk(1))
        XCTAssertEqual(cache.offset, 6)
        XCTAssertEqual(copy.offset, 5, "copy must be independent")

        XCTAssertEqual(cache.trim(4), 4)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(cache.trim(10), 2)
        XCTAssertEqual(cache.offset, 0)

        // Restoring trimmed state must pad back to full capacity so the cache
        // stays usable inside compiled traces.
        let restored = FixedCapacityKVCache(maxTokens: 8)
        restored.state = copy.state
        XCTAssertEqual(restored.offset, 5)
        XCTAssertEqual(restored.state[0].shape, [1, 1, 5, 4])
        XCTAssertEqual(restored.innerState()[0].shape, [1, 1, 8, 4])
        XCTAssertEqual(restored.metaState, ["8"])
    }

    func testStateRestorePreservesAsymmetricKeyValueDimensions() throws {
        let keys = MLXArray.ones([1, 2, 3, 6])
        let values = MLXArray.ones([1, 2, 3, 4]) * 2
        let cache = FixedCapacityKVCache(maxTokens: 8)

        cache.state = [keys, values]
        let inner = cache.innerState()
        XCTAssertEqual(inner[0].shape, [1, 2, 8, 6])
        XCTAssertEqual(inner[1].shape, [1, 2, 8, 4])
        XCTAssertEqual(cache.state[0].shape, keys.shape)
        XCTAssertEqual(cache.state[1].shape, values.shape)
        assertAllClose(cache.state[0], keys, "restored keys diverged", file: #filePath, line: #line)
        assertAllClose(
            cache.state[1], values, "restored values diverged", file: #filePath, line: #line)
    }
}
