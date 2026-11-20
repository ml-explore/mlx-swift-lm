// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpMTPTests: XCTestCase {

    // MARK: - Helpers

    /// The tiny reference config, with one MTP layer and `vocabularySize` tokens.
    private func configurationData(vocabularySize: Int = 10000) throws -> Data {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(Qwen4ExpReferenceValues.configuration.utf8))
                as? [String: Any])
        var text = try XCTUnwrap(root["text_config"] as? [String: Any])
        text["vocab_size"] = vocabularySize
        text["mtp_num_hidden_layers"] = 1
        root["text_config"] = text
        return try JSONSerialization.data(withJSONObject: root)
    }

    private func configuration(vocabularySize: Int = 10000) throws -> Qwen4ExpConfiguration {
        try JSONDecoder.json5().decode(
            Qwen4ExpConfiguration.self, from: configurationData(vocabularySize: vocabularySize))
    }

    /// Fill `module` with fixed values; `salt` keeps two modules' weights apart.
    private func fill(_ module: Module, salt: Int = 0) throws {
        var weights = [String: MLXArray]()
        let parameters = module.parameters().flattened().sorted { $0.0 < $1.0 }
        for (i, (name, parameter)) in parameters.enumerated() {
            let indices = MLXArray(Int32(0) ..< Int32(parameter.size))
            var values =
                ((indices * 7 + Int32(i * 13 + salt)) % 97 - 48).asType(.float32) / 256
            if parameter.ndim == 1 && name.hasSuffix(".weight") {
                values = values + 1
            }
            weights[name] = values.reshaped(parameter.shape)
        }
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(module)
    }

    private func target(vocabularySize: Int = 10000) throws -> Qwen4ExpModel {
        let model = Qwen4ExpModel(try configuration(vocabularySize: vocabularySize))
        try fill(model)
        return model
    }

    private func drafter(vocabularySize: Int = 10000) throws -> Qwen4ExpMTPDraftModel {
        let drafter = try Qwen4ExpMTPDraftModel(
            try configuration(vocabularySize: vocabularySize))
        try fill(drafter, salt: 5)
        return drafter
    }

    private var tokens: MLXArray {
        MLXArray(Qwen4ExpReferenceValues.tokens[0]).reshaped(1, -1)
    }

    private func emitState(checkpointAfter: Int? = nil) -> LMOutput.State {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpCacheCheckpointIndexKey] = checkpointAfter
        return state
    }

    // MARK: - Target

    func testEmitReturnsTheStreamsBeforeTheMixer() throws {
        let model = try target()
        let plain = model(tokens, cache: try model.newCache(parameters: nil))
        let output = model(
            LMInput.Text(tokens: tokens), cache: try model.newCache(parameters: nil),
            state: emitState())

        XCTAssertTrue(arrayEqual(output.logits, plain).item(Bool.self))
        let hyper = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
        XCTAssertEqual(hyper.shape, [1, tokens.dim(1), 4 * 64])
    }

    func testLinearLayersUseCheckpointedMambaCaches() throws {
        let model = try target()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.map { $0 is MambaCache }, [true, true, true, false])
        XCTAssertTrue(model.maximumNativeTargetCacheRewind >= 1)
    }

    /// A verify pass over two tokens, rewound by one, leaves every cache as a run over
    /// the first token alone: attention keys, indexer keys, deltanet, PLE conv and n-grams.
    func testRewoundVerifyPassMatchesThePrefix() throws {
        let model = try target()
        let prefix = tokens[0..., ..<9]

        let speculative = try model.newCache(parameters: nil)
        _ = model(prefix, cache: speculative)
        let verify = model(
            LMInput.Text(tokens: tokens[0..., 9 ..< 11]), cache: speculative,
            state: emitState(checkpointAfter: 1))
        eval(verify.logits)
        XCTAssertEqual(rewindSpeculativePromptCache(speculative, numTokens: 1), 1)

        let reference = try model.newCache(parameters: nil)
        _ = model(tokens[0..., ..<10], cache: reference)

        for (layer, (actual, expected)) in zip(speculative, reference).enumerated() {
            // A `CacheList` reports no offset of its own, so compare its leaves.
            XCTAssertEqual(
                KVCacheTree.leaves(in: [actual]).map(\.cache.offset),
                KVCacheTree.leaves(in: [expected]).map(\.cache.offset), "layer \(layer) offset")
            let actualState = actual.state
            let expectedState = expected.state
            XCTAssertEqual(actualState.count, expectedState.count, "layer \(layer) slots")
            for (slot, (a, e)) in zip(actualState, expectedState).enumerated() {
                XCTAssertEqual(a.shape, e.shape, "layer \(layer), slot \(slot)")
                if a.dtype.isFloatingPoint {
                    assertClose(
                        a, e, atol: 1e-4, reducedAtol: Double(MatmulPrecision.splitTolerance),
                        "layer \(layer), slot \(slot)")
                } else {
                    XCTAssertTrue(arrayEqual(a, e).item(Bool.self), "layer \(layer), slot \(slot)")
                }
            }
        }

        // The next token sees the same past.
        let next = tokens[0..., 11 ..< 12]
        assertClose(
            model(next, cache: speculative), model(next, cache: reference), atol: 1e-4,
            reducedAtol: Double(MatmulPrecision.splitTolerance))
    }

    // MARK: - Drafter

    func testHeadReturnsTheMixedStateAndNewStreams() throws {
        let drafter = try drafter()
        let model = try target()
        let hyper = MLXRandom.normal([1, 3, 4 * 64], key: MLXRandom.key(1))
        let ids = tokens[0..., ..<3]
        let (mixed, next) = drafter.mtp(
            embeddings: model.embedTokens(ids), hyper: hyper, ids: ids,
            cache: drafter.mtp.newCache())
        XCTAssertEqual(mixed.shape, [1, 3, 64])
        XCTAssertEqual(next.shape, [1, 3, 4 * 64])
        XCTAssertTrue(all(isFinite(mixed)).item(Bool.self))
    }

    /// Greedy speculative decoding gives the tokens of greedy decoding, whatever the
    /// drafts: accepted ones keep the verify pass, rejected ones rewind it. It holds for a
    /// prompt prefilled in one forward and in chunks.
    func testSpeculativeGreedyMatchesPlainGreedy() throws {
        // A small vocabulary makes some drafts match.
        let model = try target(vocabularySize: 16)
        let drafter = try drafter(vocabularySize: 16)
        let prompt = LMInput(tokens: MLXArray([Int32(3), 7, 5, 11, 4, 9, 13, 6]))

        for stepSize in [nil, 3] as [Int?] {
            var parameters = GenerateParameters(maxTokens: 24, temperature: 0)
            parameters.prefill.stepSize = stepSize
            let label = "prefill step \(stepSize.map(String.init) ?? "default")"

            var plain = try TokenIterator(input: prompt, model: model, parameters: parameters)
            var expected: [Int] = []
            while let token = plain.next() {
                expected.append(token)
            }

            var speculative = try MTPSpeculativeTokenIterator(
                input: prompt, mainModel: model, drafter: drafter, parameters: parameters,
                blockSize: 2)
            var actual: [Int] = []
            while let token = speculative.next() {
                actual.append(token)
            }

            XCTAssertEqual(actual, expected, label)
            XCTAssertNil(speculative.passthroughReason, label)
            // Both paths ran: some drafts were accepted, others rewound.
            XCTAssertGreaterThan(speculative.acceptedCount, 0, label)
            XCTAssertLessThan(speculative.acceptedCount, speculative.proposedCount, label)

            // The end of the stream keeps the head's state for the next turn.
            speculative.finalizeGeneration()
            XCTAssertNil(speculative.passthroughReason, label)
            XCTAssertNotNil(speculative.mtpDrafterContinuation, label)
        }
    }

    /// A chunked prefill returns the hidden rows of one forward, and logits for the last
    /// position alone.
    func testChunkedPrefillGathersEveryHiddenRow() throws {
        let model = try target()
        let reference = model(
            LMInput.Text(tokens: tokens), cache: try model.newCache(parameters: nil),
            state: emitState())

        let result = try model.prepare(
            LMInput(tokens: tokens[0]), cache: try model.newCache(parameters: nil),
            state: emitState(), prefill: PrefillParameters(stepSize: 5))
        guard case .logits(let output) = result else {
            return XCTFail("a chunked MTP prefill returns logits")
        }
        XCTAssertEqual(output.logits.dim(1), 1)
        let hidden = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
        let expected = try XCTUnwrap(reference.state?[mtpLastHiddenStatesKey])
        XCTAssertEqual(hidden.shape, expected.shape)
        assertClose(
            hidden, expected, atol: 1e-4, reducedAtol: Double(MatmulPrecision.splitTolerance))
        assertClose(
            output.logits[0..., -1, 0...], reference.logits[0..., -1, 0...], atol: 1e-4,
            reducedAtol: Double(MatmulPrecision.splitTolerance))
    }

    /// The head prefills a prompt longer than one chunk as one forward would.
    func testHeadPrefillsALongPromptInChunks() throws {
        let model = try target()
        let drafter = try drafter()
        let length = 600
        let ids = ((MLXArray(Int32(0) ..< Int32(length)) * 37) % 9973).reshaped(1, -1)
        let hyper = MLXRandom.normal([1, length, 4 * 64], key: MLXRandom.key(2))
        let bonus = MLXArray([Int32(4)])

        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: model, promptTokens: ids, targetHidden: hyper, firstBonus: bonus,
            positionDeltas: nil, state: &state,
            sampler: GenerateParameters(temperature: 0).sampler())

        let shifted = concatenated([ids[0..., 1...], bonus.reshaped(1, 1)], axis: 1)
        let cache = drafter.mtp.newCache()
        let (mixed, _) = drafter.mtp(
            embeddings: model.embedTokens(shifted), hyper: hyper, ids: shifted, cache: cache)
        let seed = argMax(model.project(mixed[0..., (-1)..., 0...])[0, -1])

        XCTAssertEqual(state.nextPosition, length)
        XCTAssertEqual(try XCTUnwrap(state.seedToken).item(Int.self), seed.item(Int.self))
        let actualState = state.cache.flatMap(\.state)
        let expectedState = cache.flatMap(\.state)
        XCTAssertEqual(actualState.count, expectedState.count)
        for (slot, (a, e)) in zip(actualState, expectedState).enumerated() {
            XCTAssertEqual(a.shape, e.shape, "slot \(slot)")
            assertClose(
                a, e, atol: 1e-4, reducedAtol: Double(MatmulPrecision.splitTolerance),
                "slot \(slot)")
        }
    }

    /// A stream that cannot speculate frees the head's state, which holds the prompt's rows.
    func testPassthroughReleasesTheHeadState() throws {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(3), 7, 5, 11])),
            mainModel: try target(vocabularySize: 16), drafter: try drafter(vocabularySize: 16),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0.7), blockSize: 2)
        XCTAssertNotNil(iterator.passthroughReason)
        XCTAssertNil(iterator.drafterState)
        var count = 0
        while iterator.next() != nil {
            count += 1
        }
        XCTAssertEqual(count, 4)
    }

    /// Finalize leaves the head one position behind the target, and resume carries on.
    func testDrafterFinalizesAndResumesAtTheBoundary() throws {
        let model = try target()
        let drafter = try drafter()
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = tokens[0..., ..<3]
        let hidden = try XCTUnwrap(
            model(
                LMInput.Text(tokens: prompt), cache: try model.newCache(parameters: nil),
                state: emitState()
            ).state?[mtpLastHiddenStatesKey])
        // The attention keys and the indexer keys move together.
        func offsets(_ state: MTPDrafterState) -> [Int] {
            let list = state.cache[0] as! CacheList
            return [list[0].offset, list[1].offset]
        }

        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: model, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(4)]), positionDeltas: nil, state: &state,
            sampler: sampler)
        XCTAssertEqual(state.nextPosition, 3)
        XCTAssertEqual(offsets(state), [3, 3])

        let boundary = hidden[0..., (-1)..., 0...]
        XCTAssertTrue(
            drafter.finalizeDrafterState(
                target: model, targetBoundaryHidden: boundary, targetProcessedTokenCount: 3,
                discardedTargetTokens: 0, positionDeltas: nil, state: &state))
        XCTAssertEqual(state.nextPosition, 2)
        XCTAssertEqual(offsets(state), [2, 2])

        XCTAssertTrue(
            drafter.resumeDrafterState(
                target: model, suffixTokens: tokens[0..., 3 ..< 5],
                suffixTargetHidden: MLXArray.zeros([1, 2, 4 * 64]),
                targetBoundaryHidden: boundary, firstBonus: MLXArray([Int32(8)]),
                positionDeltas: nil, state: &state, sampler: sampler))
        XCTAssertEqual(state.nextPosition, 5)
        XCTAssertEqual(offsets(state), [5, 5])
    }

    func testCheckpointKeepsTheHeadFoldsNormsAndMergesBlockInjects() throws {
        let drafter = try drafter()
        let expected = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())

        // The published layout: norms as `w`, block injects apart, the target's tensors too.
        var stored = [String: MLXArray]()
        for (key, value) in expected {
            if key.hasSuffix("_hyper_connection.input_mix_weight_down.weight") {
                let rows = value.dim(0) - 4
                let prefix = key.dropLast("input_mix_weight_down.weight".count)
                stored[key] = value[..<rows]
                stored["\(prefix)block_inject_weight.weight"] = value[rows...]
            } else if value.ndim == 1, key.contains("norm") {
                stored[key] = value - 1
            } else {
                stored[key] = value
            }
        }
        stored["model.layers.0.mlp.gate.weight"] = MLXArray.zeros([8, 64])
        stored["lm_head.weight"] = MLXArray.zeros([10, 64])

        let prepared = try drafter.prepareCheckpoint(.init(weights: stored)).weights
        XCTAssertEqual(Set(prepared.keys), Set(expected.keys))
        for (key, value) in expected {
            let actual = try XCTUnwrap(prepared[key], key)
            XCTAssertEqual(actual.shape, value.shape, key)
            XCTAssertTrue(allClose(actual, value, atol: 1e-6).item(Bool.self), key)
        }

        // A file that declares `1 + w` norms is not folded again.
        let norm = "mtp.pre_fc_norm_hidden.weight"
        let declared = try drafter.prepareCheckpoint(
            .init(
                weights: [norm: expected[norm]!],
                weightMetadata: [norm: [Qwen4ExpMTPDraftModel.normMetadataKey: "scale"]]))
        XCTAssertTrue(
            arrayEqual(try XCTUnwrap(declared.weights[norm]), expected[norm]!).item(Bool.self))
    }

    /// The loader reads only the head from a full checkpoint, in each layout of the target.
    func testFullCheckpointReadsOnlyTheHead() throws {
        let drafter = try drafter()
        let read = CheckpointNameMapping(
            drafter.excludedCheckpointNamespaces.map { .excludePrefix($0) })

        for key in try target().parameters().flattened().map(\.0) {
            var stored = [key, "language_model.\(key)"]
            if key.hasPrefix("model.") {
                stored.append("model.language_model.\(key.dropFirst("model.".count))")
            }
            for name in stored {
                XCTAssertNil(read.mapName(name), name)
            }
        }
        for key in drafter.parameters().flattened().map(\.0) {
            XCTAssertNotNil(read.mapName(key), key)
            XCTAssertNotNil(read.mapName("model.\(key)"), key)
        }
    }

    func testRegistryCreatesTheHeadOfAnMTPCheckpoint() async throws {
        await Qwen4ExpMTPRegistration.register()
        let drafter = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: try configurationData(), modelType: "qwen4_exp")
        XCTAssertTrue(drafter is Qwen4ExpMTPDraftModel)
        XCTAssertEqual(drafter.maximumBlockSize, 2)
        XCTAssertTrue(drafter.isCompatible(with: try target()))
        XCTAssertFalse(drafter.isCompatible(with: try target(vocabularySize: 16)))

        // The text model's config has no MTP layer.
        XCTAssertThrowsError(
            try Qwen4ExpMTPDraftModel(
                JSONDecoder.json5().decode(
                    Qwen4ExpConfiguration.self,
                    from: Data(Qwen4ExpReferenceValues.configuration.utf8))))
    }
}
