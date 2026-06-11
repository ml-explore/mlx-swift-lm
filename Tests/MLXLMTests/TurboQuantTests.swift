// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class TurboQuantTests: XCTestCase {

    // MARK: - Scheme resolution

    func testResolveTurboScheme() {
        XCTAssertNil(resolveTurboScheme(nil))
        XCTAssertNil(resolveTurboScheme("affine4"))
        XCTAssertNil(resolveTurboScheme("nonsense"))

        // Raw-key schemes are the validated surface; keyBits 0 = FP16 keys.
        XCTAssertEqual(resolveTurboScheme("turbo0v4")?.keyBits, 0)
        XCTAssertEqual(resolveTurboScheme("turbo0v4")?.valueBits, 4)
        XCTAssertEqual(resolveTurboScheme("turbo0v2")?.valueBits, 2)
        // Key-quantizing schemes are withheld pending the key-quality fix.
        XCTAssertNil(resolveTurboScheme("turbo4"))
        XCTAssertNil(resolveTurboScheme("turbo8"))
    }

    // MARK: - Rotation

    func testWHTRotationIsOrthogonal() {
        // Power-of-2 dim takes the WHT path. R·Rᵀ ≈ I means attention scores
        // are preserved when both Q and K pass through the rotation.
        let dim = 64
        let codec = MSECodec(dim: dim, bits: 4)
        let r = codec.rotation
        let identity = MLXArray.eye(dim)
        let product = r.matmul(r.transposed())
        let maxErr = abs(product - identity).max().item(Float.self)
        XCTAssertLessThan(
            maxErr, 1e-4, "WHT rotation is not orthogonal (max |R·Rᵀ − I| = \(maxErr))")
    }

    func testDenseRotationIsOrthogonal() {
        // Non-power-of-2 dim falls back to the QR-derived dense rotation.
        // (Must be a multiple of the scale-group size.)
        let dim = 48
        let codec = MSECodec(dim: dim, bits: 4)
        let r = codec.rotation
        let identity = MLXArray.eye(dim)
        let product = r.matmul(r.transposed())
        let maxErr = abs(product - identity).max().item(Float.self)
        XCTAssertLessThan(
            maxErr, 1e-3, "dense rotation is not orthogonal (max |R·Rᵀ − I| = \(maxErr))")
    }

    // MARK: - Codec round trip

    func testCodecRoundTripBoundedError() {
        // Unit-norm-ish gaussian vectors: 4-bit Lloyd-Max round trip should
        // reconstruct within a coarse per-element bound and preserve norms
        // exactly (norms are stored, not quantized).
        let dim = 128
        let n = 256
        let codec = MSECodec(dim: dim, bits: 4)
        let vectors = MLXRandom.normal([n, dim], key: MLXRandom.key(7))

        let state = codec.encode(vectors)
        XCTAssertEqual(state.tokenCount, n)
        let decoded = codec.decode(state)
        XCTAssertEqual(decoded.shape, vectors.shape)

        // The WHT path stores raw norms and reconstructs from quantized unit
        // vectors whose norm is slightly under 1, so reconstruction norms sit
        // a few percent low — bound the RELATIVE error.
        let origNorms = sqrt((vectors * vectors).sum(axis: -1))
        let decNorms = sqrt((decoded * decoded).sum(axis: -1))
        let relNormErr = (abs(origNorms - decNorms) / origNorms).max().item(Float.self)
        XCTAssertLessThan(relNormErr, 0.15, "norms drifted (max relative err \(relNormErr))")

        // Cosine similarity between original and reconstruction — 4-bit
        // codebooks on rotated gaussians sit well above 0.9.
        let dots = (vectors * decoded).sum(axis: -1)
        let cos = (dots / (origNorms * decNorms)).min().item(Float.self)
        XCTAssertGreaterThan(cos, 0.9, "round-trip cosine too low (\(cos))")
    }

    // MARK: - Cache behavior

    func testCachePrefillThenDecodeOffsets() {
        let cache = TurboQuantKVCache(bits: 4)
        let (b, h, d) = (1, 2, 64)

        // Prefill: raw phase
        let k0 = MLXRandom.normal([b, h, 16, d], key: MLXRandom.key(1))
        let v0 = MLXRandom.normal([b, h, 16, d], key: MLXRandom.key(2))
        let (ck, cv) = cache.update(keys: k0, values: v0)
        XCTAssertEqual(cache.offset, 16)
        XCTAssertEqual(ck.dim(2), 16)
        XCTAssertEqual(cv.dim(2), 16)

        // Decode steps
        for i in 0 ..< 4 {
            let k = MLXRandom.normal([b, h, 1, d], key: MLXRandom.key(UInt64(10 + i)))
            let v = MLXRandom.normal([b, h, 1, d], key: MLXRandom.key(UInt64(20 + i)))
            _ = cache.update(keys: k, values: v)
        }
        XCTAssertEqual(cache.offset, 20)

        // Trim back
        let trimmed = cache.trim(5)
        XCTAssertEqual(trimmed, 5)
        XCTAssertEqual(cache.offset, 15)

        // Full trim resets
        _ = cache.trim(100)
        XCTAssertEqual(cache.offset, 0)
    }

    func testCompressedAttentionMatchesFP16Reference() throws {
        // The correctness gate: TurboQuant decode attention vs exact FP16
        // SDPA over the same K/V. 4-bit on smooth inputs should agree to a
        // loose-but-meaningful tolerance; garbage (NaN/zeros) fails loudly.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let prefill = 32

        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(3)) * 0.3
        let values = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(4)) * 0.3
        _ = cache.update(keys: keys, values: values)

        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(5)) * 0.3
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(6)) * 0.3
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(8)) * 0.3
        let scale = 1.0 / Float(d).squareRoot()

        let turboOut = cache.compressedAttention(
            queries: q, keys: newK, values: newV, scale: scale)

        let refK = concatenated([keys, newK], axis: 2)
        let refV = concatenated([values, newV], axis: 2)
        let refOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: refK, values: refV, scale: scale, mask: .none)

        XCTAssertEqual(turboOut.shape, refOut.shape)
        let err = abs(turboOut - refOut).max().item(Float.self)
        XCTAssertFalse(err.isNaN, "TurboQuant attention produced NaN")
        XCTAssertLessThan(
            err, 0.15, "TurboQuant attention diverges from FP16 reference (max err \(err))")
    }

    // MARK: - End-to-end generation

    func testEndToEndGenerationWithTurboScheme() throws {
        // Tiny random-weight Llama: prefill, then a short greedy decode loop
        // with the turbo4 scheme applied after each step — the same call
        // pattern as TokenIterator.step(). Catches integration breaks
        // (cache conversion, attention routing, NaN logits) without a
        // checkpoint.
        let config = LlamaConfiguration(
            hiddenSize: 128, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        func decode(scheme: String?) -> [Int] {
            var cache = model.newCache(parameters: nil)
            let prompt = MLXArray([1, 7, 3, 12, 5, 9, 2, 8])[.newAxis, .ellipsis]
            var logits = model(prompt, cache: cache)
            var tokens: [Int] = []
            for _ in 0 ..< 8 {
                maybeQuantizeKVCache(
                    cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: scheme)
                let next = argMax(logits[0..., -1, 0...], axis: -1)
                let token = next.item(Int.self)
                tokens.append(token)
                logits = model(next[.newAxis, .ellipsis], cache: cache)
                let bad = MLX.isNaN(logits).any().item(Bool.self)
                XCTAssertFalse(bad, "NaN logits during \(scheme ?? "fp16") decode")
            }
            if scheme != nil {
                XCTAssertTrue(
                    cache.allSatisfy { $0 is TurboQuantKVCache },
                    "caches not converted to TurboQuantKVCache")
            }
            return tokens
        }

        let turbo = decode(scheme: "turbo0v4")
        let reference = decode(scheme: nil)
        XCTAssertEqual(turbo.count, 8)
        // The first decode token is computed from the still-raw cache, so it
        // must match the FP16 run exactly; later tokens may diverge within
        // 4-bit quantization error.
        XCTAssertEqual(turbo[0], reference[0], "first decode token diverged before compression")
    }

    func testRawKeyModeAttentionMatchesFP16Reference() throws {
        // turbo0v4: raw FP16 keys + 4-bit values — should be the HIGHEST
        // quality scheme, so hold it to a tighter bound than turbo4.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let prefill = 32

        let cache = TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4)
        let keys = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(3)) * 0.3
        let values = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(4)) * 0.3
        _ = cache.update(keys: keys, values: values)

        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(5)) * 0.3
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(6)) * 0.3
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(8)) * 0.3
        let scale = 1.0 / Float(d).squareRoot()

        let turboOut = cache.compressedAttention(
            queries: q, keys: newK, values: newV, scale: scale)

        let refK = concatenated([keys, newK], axis: 2)
        let refV = concatenated([values, newV], axis: 2)
        let refOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: refK, values: refV, scale: scale, mask: .none)

        XCTAssertEqual(turboOut.shape, refOut.shape)
        let err = abs(turboOut - refOut).max().item(Float.self)
        XCTAssertFalse(err.isNaN, "rawKeyMode attention produced NaN")
        XCTAssertLessThan(err, 0.1, "rawKeyMode attention diverges (max err \(err))")
    }

    func testConvertedCacheAttentionMatchesFP16() throws {
        // The REAL runtime route: prefill into KVCacheSimple, convert via
        // maybeTurboQuantizeKVCache (the kvScheme path), then decode through
        // the converted cache — vs FP16 reference.
        let (b, hq, hkv, d) = (1, 12, 2, 128)
        let prefill = 64
        let scale = 1.0 / Float(d).squareRoot()

        for (keyBits, valueBits) in [(4, 4), (4, 2)] {
            let simple = KVCacheSimple()
            let refK = (MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(3)) * 0.3)
                .asType(.float16)
            let refV = (MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(4)) * 0.3)
                .asType(.float16)
            _ = simple.update(keys: refK, values: refV)

            var caches: [KVCache] = [simple]
            maybeTurboQuantizeKVCache(
                cache: &caches, keyBits: keyBits, valueBits: valueBits, quantizedKVStart: 0)
            guard let turbo = caches[0] as? TurboQuantKVCache else {
                XCTFail("not converted")
                return
            }
            XCTAssertEqual(turbo.offset, prefill, "offset after conversion")

            let newK = (MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(50)) * 0.3)
                .asType(.float16)
            let newV = (MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(70)) * 0.3)
                .asType(.float16)
            let q = (MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(90)) * 0.3)
                .asType(.float16)

            let turboOut = turbo.compressedAttention(
                queries: q, keys: newK, values: newV, scale: scale)

            let allK = concatenated([refK, newK], axis: 2)
            let allV = concatenated([refV, newV], axis: 2)
            let refOut = MLXFast.scaledDotProductAttention(
                queries: q, keys: allK, values: allV, scale: scale, mask: .none)

            let err = abs(turboOut.asType(.float32) - refOut.asType(.float32)).max()
                .item(Float.self)
            XCTAssertFalse(err.isNaN, "k\(keyBits)v\(valueBits): NaN")
            XCTAssertLessThan(err, 0.1, "k\(keyBits)v\(valueBits): max err \(err)")
        }
    }

    func testRawKeyModeGQAHeadMapping() throws {
        // Give each KV head a DISTINCT value signature (+5 vs -5): if the
        // CPU-side score path and the kernel value-sum disagree on the
        // q-head → kv-head mapping, outputs land near the wrong head's
        // signature and the error is O(10), unmissable. iid random heads
        // cannot catch this.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let prefill = 16
        let scale = 1.0 / Float(d).squareRoot()

        let cache = TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4)
        let keys = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(3)) * 0.3
        // head 0 values centered +5, head 1 centered -5
        let base = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(4)) * 0.1
        let signs = MLXArray([Float(5), Float(-5)]).reshaped([1, hkv, 1, 1])
        let values = base + signs
        _ = cache.update(keys: keys, values: values)

        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(5)) * 0.3
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(6)) * 0.1 + signs
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(8)) * 0.3

        let turboOut = cache.compressedAttention(
            queries: q, keys: newK, values: newV, scale: scale)

        let refK = concatenated([keys, newK], axis: 2)
        let refV = concatenated([values, newV], axis: 2)
        let refOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: refK, values: refV, scale: scale, mask: .none)

        // Per-q-head mean output — sign tells which kv head fed it.
        for h in 0 ..< hq {
            let turboMean = turboOut[0, h].mean().item(Float.self)
            let refMean = refOut[0, h].mean().item(Float.self)
            print("HEADMAP q\(h): turbo mean \(turboMean), ref mean \(refMean)")
            XCTAssertEqual(
                turboMean.sign == .minus, refMean.sign == .minus,
                "q-head \(h) fed by the wrong kv head")
        }
        let err = abs(turboOut - refOut).max().item(Float.self)
        XCTAssertLessThan(err, 1.0, "rawKeyMode GQA output error \(err)")
    }

    // MARK: - kvScheme routing

    func testMaybeTurboQuantizeConvertsEligibleLayers() {
        let (b, h, d, t) = (1, 2, 64, 8)
        let simple = KVCacheSimple()
        let keys = MLXRandom.normal([b, h, t, d], key: MLXRandom.key(9))
        let values = MLXRandom.normal([b, h, t, d], key: MLXRandom.key(10))
        _ = simple.update(keys: keys, values: values)

        var caches: [KVCache] = [simple]
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 4, valueBits: 4, quantizedKVStart: 0)

        XCTAssertTrue(caches[0] is TurboQuantKVCache, "eligible layer not converted")
        XCTAssertEqual(caches[0].offset, t, "offset lost in conversion")

        // Below the start threshold: untouched.
        let early = KVCacheSimple()
        _ = early.update(
            keys: MLXRandom.normal([b, h, 2, d], key: MLXRandom.key(11)),
            values: MLXRandom.normal([b, h, 2, d], key: MLXRandom.key(12)))
        var earlyCaches: [KVCache] = [early]
        maybeTurboQuantizeKVCache(
            cache: &earlyCaches, keyBits: 4, valueBits: 4, quantizedKVStart: 100)
        XCTAssertTrue(earlyCaches[0] is KVCacheSimple, "layer below threshold was converted")
    }
}
