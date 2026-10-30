// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// Tests of the SigLIP2 NaFlex tower and its patch input. Exact comparisons run on the CPU; see
// LFM2BidirectionalTests.swift. Expected values come from tools/lfm2_vl_test_vectors.py
// (transformers and torch).

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// Deterministic, exactly representable table values shared with the generator.
private func positionTable() -> MLXArray {
    let values = (0 ..< 16 * 16 * 4).map { (i: Int) -> Float in Float((i * 37) % 101) / 8 }
    return MLXArray(values, [16, 16, 4])
}

private func randomPixels(_ shape: [Int], seed: UInt64) -> MLXArray {
    withRandomState(MLXRandom.RandomState(seed: seed)) { MLXRandom.normal(shape) }
}

struct Siglip2NaFlexTests {

    // MARK: - Position embeddings

    @Test("Bilinear antialias weights match torch F.interpolate bit for bit")
    func bilinearWeights() {
        // (input, output, row) -> the row's nonzero weights in torch.
        let cases: [(Int, Int, Int, [Int: Float])] = [
            (16, 14, 0, [0: 0.8333333134651184, 1: 0.16666671633720398]),
            (16, 14, 1, [1: 0.722222089767456, 2: 0.27777794003486633]),
            (
                16, 2, 0,
                [
                    0: 0.0803571417927742, 1: 0.0982142835855484, 2: 0.1160714253783226,
                    3: 0.1339285671710968, 4: 0.1339285671710968, 5: 0.1160714253783226,
                    6: 0.0982142835855484, 7: 0.0803571417927742, 8: 0.0625,
                    9: 0.0446428582072258, 10: 0.02678571455180645, 11: 0.008928571827709675,
                ]
            ),
            (16, 32, 0, [0: 1]),
            (16, 32, 1, [0: 0.75, 1: 0.25]),
            (16, 32, 31, [15: 1]),
            (16, 26, 1, [0: 0.5769230127334595, 1: 0.4230769872665405]),
        ]
        for (input, output, row, expected) in cases {
            let weights = antialiasedBilinearWeights(inputSize: input, outputSize: output)
            let values = (0 ..< input).map { weights[row * input + $0] }
            #expect(
                values == (0 ..< input).map { expected[$0] ?? 0 }, "\(input)->\(output) row \(row)")
        }

        let same = antialiasedBilinearWeights(inputSize: 16, outputSize: 16)
        #expect(same == (0 ..< 256).map { $0 / 16 == $0 % 16 ? 1 : 0 })

        for (n, m) in [(16, 3), (16, 7), (16, 38), (16, 46), (5, 64)] {
            let w = antialiasedBilinearWeights(inputSize: n, outputSize: m)
            for i in 0 ..< m {
                let total = w[(i * n) ..< ((i + 1) * n)].reduce(0, +)
                #expect(abs(total - 1) < 1e-5, "\(n)->\(m) row \(i) sums to \(total)")
            }
        }
    }

    @Test("Resized position embeddings match transformers resize_positional_embeddings")
    func resizedPositionEmbeddings() {
        Device.withDefaultDevice(.cpu) {
            let table = positionTable()
            let out = Siglip2NaFlex.VisionEmbeddings.resizePositionEmbeddings(
                table, spatialShapes: [(14, 20), (2, 40), (32, 32), (16, 16)], maxLength: 1024)
            #expect(out.shape == [4, 1024, 4])
            #expect(out.dtype == .float32)

            // `Siglip2VisionEmbeddings.resize_positional_embeddings` on the same table.
            let expected: [(Int, Int, [Float])] = [
                (
                    0, 0,
                    [0.9375002980232239, 5.562500476837158, 8.083333015441895, 2.187500238418579]
                ),
                (
                    0, 21,
                    [8.845138549804688, 6.035417079925537, 5.4701385498046875, 3.712500810623169]
                ),
                (
                    0, 279,
                    [7.437501430511475, 1.5416656732559204, 6.166666030883789, 8.687501907348633]
                ),
                (
                    1, 0,
                    [5.745535373687744, 5.861606597900391, 6.4285712242126465, 6.882812023162842]
                ),
                (
                    1, 57,
                    [5.8409600257873535, 6.5770087242126465, 6.4676337242126465, 5.907365798950195]
                ),
                (2, 33, [2.875, 6.7109375, 6.6015625, 3.3359375]),
            ]
            for (row, position, values) in expected {
                #expect(maxAbsDiff(out[row, position], MLXArray(values)) < 2e-5)
            }
            let sum0 = out[0, 0 ..< 280].asType(.float32).sum().item(Float.self)
            #expect(abs(sum0 - 6991.90597897768) < 1e-2)
            let sum1 = out[1, 0 ..< 80].sum().item(Float.self)
            #expect(abs(sum1 - 1998.5016255378723) < 1e-2)
            let sum2 = out[2].sum().item(Float.self)
            #expect(abs(sum2 - 25570.0) < 1e-2)

            // Same size is the table itself; padded rows repeat the first row.
            #expect(maxAbsDiff(out[3, 0 ..< 256], table.reshaped(256, 4)) == 0)
            #expect(maxAbsDiff(out[0, 280...], broadcast(out[0, 0 ..< 1], to: [744, 4])) == 0)
            #expect(maxAbsDiff(out[1, 80...], broadcast(out[1, 0 ..< 1], to: [944, 4])) == 0)
        }
    }

    @Test("Position embeddings are resized in float32 and cast back")
    func resizeKeepsSourceType() {
        Device.withDefaultDevice(.cpu) {
            let table = positionTable()
            let half = Siglip2NaFlex.VisionEmbeddings.resizePositionEmbeddings(
                table.asType(.bfloat16), spatialShapes: [(14, 20)], maxLength: 300)
            let full = Siglip2NaFlex.VisionEmbeddings.resizePositionEmbeddings(
                table.asType(.bfloat16).asType(.float32), spatialShapes: [(14, 20)],
                maxLength: 300)
            #expect(half.dtype == .bfloat16)
            #expect(maxAbsDiff(half, full.asType(.bfloat16)) == 0)
        }
    }

    // MARK: - Tower

    private func tinyTower() throws -> Siglip2NaFlex.VisionModel {
        let config = Siglip2NaFlex.Configuration(
            hiddenSize: 16, intermediateSize: 32, numHiddenLayers: 2, numAttentionHeads: 2,
            numChannels: 3, numPatches: 16, patchSize: 2, layerNormEps: 1e-6)
        let tower = withRandomState(MLXRandom.RandomState(seed: 0)) {
            Siglip2NaFlex.VisionModel(config)
        }
        eval(tower)
        return tower
    }

    @Test("geluTanh rounds a float32 GELU once and keeps the input type")
    func geluTanhRoundsOnce() {
        let x = MLXArray(stride(from: Float(-8), through: 8, by: 1.0 / 64).map { $0 })
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                for dtype in [DType.bfloat16, .float16] {
                    let h = x.asType(dtype)
                    let y = Siglip2NaFlex.geluTanh(h)
                    #expect(y.dtype == dtype)
                    #expect(maxAbsDiff(y, geluApproximate(h.asType(.float32)).asType(dtype)) == 0)
                }
                #expect(maxAbsDiff(Siglip2NaFlex.geluTanh(x), geluApproximate(x)) == 0)
            }
        }
    }

    @Test("Views cut to their patches give the same features as padded, masked views")
    func viewStatesMatchPaddedAndMasked() throws {
        try Device.withDefaultDevice(.cpu) {
            let tower = try tinyTower()
            // The views of 12 patches share a batch; the 4x4 view runs alone.
            let shapes = [(3, 4), (4, 4), (2, 6)]
            let length = 20
            // Non-zero padding, so a missing mask would change the result.
            let pixels = randomPixels([3, length, 12], seed: 1)
            let mask = Siglip2NaFlex.patchMask(shapes, length: length)

            let padded = tower(pixels, spatialShapes: shapes, mask: mask).lastHiddenState
            #expect(padded.shape == [3, length, 16])

            let states = tower.viewStates(pixels, spatialShapes: shapes)
            for (i, (h, w)) in shapes.enumerated() {
                #expect(states[i].shape == [h * w, 16])
                #expect(maxAbsDiff(padded[i, 0 ..< (h * w)], states[i]) < 1e-5)
                let alone = tower(
                    pixels[i ..< (i + 1), 0 ..< (h * w)], spatialShapes: [(h, w)], mask: nil
                ).lastHiddenState
                #expect(maxAbsDiff(alone[0], states[i]) < 1e-5)
            }

            let unmasked = tower(pixels, spatialShapes: shapes, mask: nil).lastHiddenState
            #expect(maxAbsDiff(unmasked[0, 0 ..< 12], padded[0, 0 ..< 12]) > 1e-3)
        }
    }

    @Test("The encoder returns its final state without hidden states")
    func encoderReturnsFinalState() throws {
        try Device.withDefaultDevice(.cpu) {
            let tower = try tinyTower()
            let shapes = [(4, 4)]
            let pixels = randomPixels([1, 16, 12], seed: 2)

            let (last, hiddenStates) = tower(pixels, spatialShapes: shapes, mask: nil)
            #expect(hiddenStates == nil)

            let embeddings = tower.embeddings(pixels, spatialShapes: shapes)
            var x = embeddings
            for layer in tower.encoder.layers {
                x = layer(x)
            }
            #expect(maxAbsDiff(last, tower.postLayernorm(x)) < 1e-6)
            #expect(maxAbsDiff(last, tower.postLayernorm(embeddings)) > 1e-2)

            let (lastWithStates, states) = tower(
                pixels, spatialShapes: shapes, mask: nil, outputHiddenStates: true)
            let all = try #require(states)
            #expect(all.count == 3)
            #expect(maxAbsDiff(all[0], embeddings) == 0)
            #expect(maxAbsDiff(lastWithStates, last) == 0)
            #expect(maxAbsDiff(tower.postLayernorm(all[2]), last) == 0)
        }
    }

    // MARK: - Patches

    @Test("patchMask marks the first h * w patches of each view")
    func patchMask() {
        let mask = Siglip2NaFlex.patchMask([(1, 2), (2, 2), (1, 1)], length: 4)
        #expect(mask.shape == [3, 4])
        #expect(mask.dtype == .int32)
        #expect(mask.asArray(Int32.self) == [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    }

    @Test func patchesAreRowMajorWithRowColumnChannelOrder() throws {
        // A 4x2 crop with 2x2 patches: two patches side by side.
        var bytes = [UInt8](repeating: 0, count: 4 * 2 * 3)
        for y in 0 ..< 2 {
            for x in 0 ..< 4 {
                for c in 0 ..< 3 { bytes[(y * 4 + x) * 3 + c] = UInt8(y * 100 + x * 10 + c) }
            }
        }
        let crop = RGB8Image(width: 4, height: 2, bytes: bytes)
        // A 5x3 crop is cut to whole patches, as transformers does.
        let ragged = crop.cropped(x: 0, y: 0, width: 5, height: 3)
        try Device.withDefaultDevice(.cpu) { () throws in
            let patches = siglip2Patches(
                [crop, ragged], patchSize: 2, maxPatches: 3,
                table: (0 ..< 768).map { Float($0 % 256) })
            #expect(patches.pixels.shape == [2, 3, 12])
            #expect(patches.spatialShapes.map { [$0.0, $0.1] } == [[1, 2], [1, 2]])
            #expect(patches.mask.asArray(Int32.self) == [1, 1, 0, 1, 1, 0])
            let values = patches.pixels.asArray(Float.self).map { Int($0) }
            #expect(Array(values[0 ..< 12]) == [0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112])
            #expect(
                Array(values[12 ..< 24]) == [20, 21, 22, 30, 31, 32, 120, 121, 122, 130, 131, 132])
            #expect(Array(values[24 ..< 36]) == Array(repeating: 0, count: 12))
            #expect(Array(values[36 ..< 72]) == Array(values[0 ..< 36]))
        }
    }

    @Test func tableVariantUsesEachChannelsEntry() throws {
        let crop = RGB8Image(width: 1, height: 1, bytes: [5, 5, 5])
        let table = (0 ..< 768).map { Float($0) }
        try Device.withDefaultDevice(.cpu) { () throws in
            let pixels = siglip2Patches([crop], patchSize: 1, maxPatches: 1, table: table).pixels
            #expect(pixels.asArray(Float.self) == [5, 261, 517])
        }
    }
}
