// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXGuidedGeneration

/// Microbenchmark for the per-token grammar-mask materialization.
///
/// `bitmaskToMLXArray` currently runs on the critical path inside
/// `applyMaskAndSample` (top of the loop, right after the previous `eval`).
/// The maintainer asked whether it could instead be built in the eval loop's
/// CPU/GPU overlap window and passed in. That relocation is only worth the
/// signature change + test churn if the build is a measurable share of
/// per-token cost, so this measures it directly. No model required.
///
/// Two views per vocab size:
///  - `build`: cost of `bitmaskToMLXArray` alone — the CPU work (alloc +
///    bit-unpack loop + MLXArray host copy) that relocation would hide behind
///    the forward pass.
///  - `sample`: `applyMaskAndSample` with vs without a mask. Both include the
///    argmax + `item()` device sync; the delta is the mask's marginal
///    sample-time cost. The build is now hoisted into the loop's overlap
///    window, so this delta excludes it (dense add only).
///
/// Opt-in (prints timings, asserts nothing):
///   GUIDED_GEN_BENCH=1 xcodebuild test -scheme mlx-swift-lm-Package \
///     -destination 'platform=macOS' \
///     -only-testing:MLXGuidedGenerationTests/MaskBuildBenchmark
@Suite(.serialized)
struct MaskBuildBenchmark {

    /// Representative model logit dimensions: Qwen-small, Llama-3, Qwen-2.5, Gemma.
    static let vocabSizes = [32_000, 128_256, 151_936, 256_000]

    private static func makeMaskWords(vocab: Int, allowed: Int) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: (vocab + 31) / 32)
        // Spread `allowed` set bits across the vocab. Sparsity barely affects
        // the O(vocab) build cost; this just avoids an all-zero mask.
        let stride = Swift.max(1, vocab / Swift.max(1, allowed))
        var i = 0
        while i < vocab {
            words[i / 32] |= (UInt32(1) << (UInt32(i) % 32))
            i += stride
        }
        return words
    }

    private static func microsPerCall(_ elapsed: Duration, iters: Int) -> Double {
        let c = elapsed.components
        let totalMicros =
            Double(c.seconds) * 1_000_000 + Double(c.attoseconds) / 1_000_000_000_000
        return totalMicros / Double(iters)
    }

    private static func f(_ x: Double) -> String { String(format: "%.1f", x) }

    /// Buffer-reuse variant of `bitmaskToMLXArray`: refills a caller-owned
    /// `[Float]` instead of allocating a fresh one each call. Isolates how much
    /// of the build is recoverable allocation vs the unavoidable O(vocab) refill
    /// + MLXArray host copy. (Local to the benchmark — not a production API,
    /// since buffer-reuse may not be the chosen approach.)
    private static func buildReusing(
        _ maskPtr: UnsafePointer<UInt32>, vocab: Int, into buffer: inout [Float]
    ) -> MLXArray {
        buffer.withUnsafeMutableBufferPointer { b in
            b.update(repeating: -.infinity)
            for i in 0 ..< vocab where (maskPtr[i / 32] >> (UInt32(i) % 32)) & 1 == 1 {
                b[i] = 0.0
            }
        }
        return MLXArray(buffer)
    }

    @Test
    func maskBuildCost() {
        let clock = ContinuousClock()
        let warmup = 50
        let iters = 500

        print("[MASKBENCH] === bitmaskToMLXArray build cost (relocatable CPU work) ===")
        print("[MASKBENCH] vocab | build_us/call | us/200tok | us/500tok")
        for vocab in Self.vocabSizes {
            let words = Self.makeMaskWords(vocab: vocab, allowed: 256)
            var sink: MLXArray?
            words.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!
                for _ in 0 ..< warmup {
                    sink = GuidedGenerationLoop.bitmaskToMLXArray(
                        ptr, maskBitCount: vocab, totalCount: vocab)
                }
                let start = clock.now
                for _ in 0 ..< iters {
                    sink = GuidedGenerationLoop.bitmaskToMLXArray(
                        ptr, maskBitCount: vocab, totalCount: vocab)
                }
                let per = Self.microsPerCall(clock.now - start, iters: iters)
                _ = sink?.shape
                print(
                    "[MASKBENCH][build] \(vocab) | \(Self.f(per)) | \(Self.f(per * 200)) | \(Self.f(per * 500))"
                )
            }
        }
    }

    @Test
    func maskBuildCostReuse() {
        let clock = ContinuousClock()
        let warmup = 50
        let iters = 500

        print("[MASKBENCH] === bitmaskToMLXArray with a reused [Float] buffer ===")
        print("[MASKBENCH] vocab | reuse_us/call | us/200tok")
        for vocab in Self.vocabSizes {
            let words = Self.makeMaskWords(vocab: vocab, allowed: 256)
            var buffer = [Float](repeating: -.infinity, count: vocab)
            var sink: MLXArray?
            words.withUnsafeBufferPointer { maskBuf in
                let ptr = maskBuf.baseAddress!
                for _ in 0 ..< warmup {
                    sink = Self.buildReusing(ptr, vocab: vocab, into: &buffer)
                }
                let start = clock.now
                for _ in 0 ..< iters {
                    sink = Self.buildReusing(ptr, vocab: vocab, into: &buffer)
                }
                let per = Self.microsPerCall(clock.now - start, iters: iters)
                _ = sink?.shape
                print("[MASKBENCH][reuse] \(vocab) | \(Self.f(per)) | \(Self.f(per * 200))")
            }
        }
    }

    @Test
    func sampleTimeCost() {
        let clock = ContinuousClock()
        let warmup = 20
        let iters = 200

        print("[MASKBENCH] === applyMaskAndSample per-call (add + argmax/item; mask prebuilt) ===")
        print("[MASKBENCH] vocab | withMask_us | nilMask_us | delta_us")
        for vocab in Self.vocabSizes {
            let logits = MLXArray([Float](repeating: 0, count: vocab))[.newAxis, .newAxis, 0...]
            let words = Self.makeMaskWords(vocab: vocab, allowed: 256)
            let maskArray = words.withUnsafeBufferPointer {
                GuidedGenerationLoop.bitmaskToMLXArray(
                    $0.baseAddress!, maskBitCount: vocab, totalCount: vocab)
            }
            var tok: UInt32 = 0

            for _ in 0 ..< warmup {
                tok = GuidedGenerationLoop.applyMaskAndSample(logits: logits, maskArray: maskArray)
            }
            var start = clock.now
            for _ in 0 ..< iters {
                tok = GuidedGenerationLoop.applyMaskAndSample(logits: logits, maskArray: maskArray)
            }
            let withMask = Self.microsPerCall(clock.now - start, iters: iters)

            for _ in 0 ..< warmup {
                tok = GuidedGenerationLoop.applyMaskAndSample(logits: logits, maskArray: nil)
            }
            start = clock.now
            for _ in 0 ..< iters {
                tok = GuidedGenerationLoop.applyMaskAndSample(logits: logits, maskArray: nil)
            }
            let nilMask = Self.microsPerCall(clock.now - start, iters: iters)
            _ = tok
            print(
                "[MASKBENCH][sample] \(vocab) | \(Self.f(withMask)) | \(Self.f(nilMask)) | \(Self.f(withMask - nilMask))"
            )
        }
    }
}
