// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

public class GatedDeltaTests: XCTestCase {

    private struct Inputs {
        let q, k, v, a, b, aLog, dtBias: MLXArray
    }

    /// Build deterministic bf16 inputs shaped for the GDN entry points.
    /// Hk/Hv/Dk/Dv stay tiny so the kernel dispatches but the test runs in ms.
    private func makeInputs(
        B: Int = 1, T: Int = 16, Hk: Int = 2, Dk: Int = 32,
        Hv: Int = 4, Dv: Int = 16, seed: UInt64 = 42
    ) -> Inputs {
        // Task-local rather than MLXRandom.seed: parallel tests must not
        // share (or perturb) the global random stream.
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let dtype = DType.bfloat16
            let q = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
            let k = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
            let v = MLXRandom.normal([B, T, Hv, Dv]).asType(dtype)
            let a = MLXRandom.normal([B, T, Hv]).asType(dtype)
            let b = MLXRandom.normal([B, T, Hv]).asType(dtype)
            let aLog = (MLXRandom.normal([Hv]) * MLXArray(0.1)).asType(dtype)
            let dtBias = MLXRandom.normal([Hv]).asType(dtype)
            return Inputs(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias)
        }
    }

    /// Multi-chunk prefill must match single-chunk prefill at the same total T.
    ///
    /// Regression for the fp32-state fix. Pre-fix, `gatedDeltaKernel` wrote
    /// `state_out` as `InT` (bf16) and `gatedDeltaUpdate` defaulted state to
    /// `q.dtype` (bf16). When a second-chunk prefill reloaded that state, it
    /// arrived bf16-quantized; the fp32 scratch recurrence then ran from a
    /// degraded starting point. With this test's inputs the cross-chunk drift
    /// vs a single full-length prefill is >10 max abs. Post-fix, state crosses
    /// the chunk boundary as fp32 and the two paths match within bf16 input
    /// quantization noise.
    func testGatedDeltaMultiChunkMatchesSingleChunk() throws {
        let T = 16
        let inputs = makeInputs(T: T)

        let (ySingle, _) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        eval(ySingle)

        let mid = T / 2
        let (y1, state1) = gatedDeltaUpdate(
            q: inputs.q[0..., ..<mid], k: inputs.k[0..., ..<mid], v: inputs.v[0..., ..<mid],
            a: inputs.a[0..., ..<mid], b: inputs.b[0..., ..<mid],
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        let (y2, _) = gatedDeltaUpdate(
            q: inputs.q[0..., mid...], k: inputs.k[0..., mid...], v: inputs.v[0..., mid...],
            a: inputs.a[0..., mid...], b: inputs.b[0..., mid...],
            aLog: inputs.aLog, dtBias: inputs.dtBias,
            state: state1
        )
        let yMulti = concatenated([y1, y2], axis: 1)
        eval(yMulti)

        let diff = abs(ySingle.asType(.float32) - yMulti.asType(.float32)).max()
        eval(diff)
        let maxDiff = diff.item(Float.self)

        // Pre-fix: bf16-cast at chunk boundary diverges by >10 max abs.
        // Post-fix: fp32 state across the boundary leaves only bf16 input noise.
        XCTAssertLessThan(
            maxDiff, 1e-2,
            "Multi-chunk GDN prefill diverged from single-chunk by \(maxDiff) max abs. "
                + "Cross-chunk state must persist in fp32; bf16 cast loses precision."
        )
    }

    /// A key head dim that is not a multiple of 32 must use every dim.
    ///
    /// The fused kernel distributes Dk over exactly 32 simd lanes
    /// (`n_per_t = Dk / 32`, integer-truncating), so for Dk = 48 it covers only
    /// dims 0..<32 and silently drops dims 32..<48. `gatedDeltaUpdate` guards
    /// against this by routing any non-multiple-of-32 Dk to the ops fallback.
    /// Proof: perturbing only the trailing `Dk % 32` dims of q/k must change the
    /// output. Pre-guard the kernel ignores those dims (output unchanged); with
    /// the guard the ops path uses them (output changes).
    func testGatedDeltaNonMultipleOf32KeyDimUsesAllDims() throws {
        let inputs = makeInputs(Dk: 48)

        let (yBase, _) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )

        // Perturb only the trailing dims the truncating kernel would drop.
        let qP = inputs.q
        let kP = inputs.k
        qP[0..., 0..., 0..., 32...] = qP[0..., 0..., 0..., 32...] + MLXArray(1).asType(.bfloat16)
        kP[0..., 0..., 0..., 32...] = kP[0..., 0..., 0..., 32...] + MLXArray(1).asType(.bfloat16)

        let (yPerturbed, _) = gatedDeltaUpdate(
            q: qP, k: kP, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )

        let diff = abs(yBase.asType(.float32) - yPerturbed.asType(.float32)).max()
        eval(diff)
        let maxDiff = diff.item(Float.self)

        // Pre-guard: kernel drops dims 32..<48, so the perturbation is invisible
        // (maxDiff ≈ 0). With the guard: ops fallback uses all dims, output moves.
        XCTAssertGreaterThan(
            maxDiff, 1e-3,
            "Perturbing trailing Dk dims (32..<48) left GDN output unchanged "
                + "(\(maxDiff) max abs) — Dk % 32 != 0 was routed to the truncating "
                + "kernel instead of the ops fallback."
        )
    }

    /// The recurrent state update must retain low-order terms in its k-state dot product.
    ///
    /// Each SIMD lane accumulates one large value followed by five unit values. A naive
    /// FP32 sum drops all five units, shifting the recurrent state by one ULP. Compensated
    /// summation recovers them and matches the correctly rounded reference result.
    func testGatedDeltaCompensatesRecurrentDotProduct() throws {
        let keyDimension = 192
        let laneState: [Float] = [100_000_000, 1, 1, 1, 1, 1]
        let stateValues = (0 ..< 32).flatMap { _ in laneState }

        let q = MLXArray.zeros([1, 1, 1, keyDimension], dtype: .bfloat16)
        let k = MLXArray.ones([1, 1, 1, keyDimension], dtype: .bfloat16)
        let v = MLXArray.zeros([1, 1, 1, 1], dtype: .bfloat16)
        let a = MLXArray.zeros([1, 1, 1], dtype: .bfloat16)
        let b = MLXArray.zeros([1, 1, 1], dtype: .bfloat16)
        let aLog = MLXArray([-100] as [Float])
        let dtBias = MLXArray.zeros([1], dtype: .bfloat16)
        let state = MLXArray(stateValues).reshaped(1, 1, 1, keyDimension)

        let (_, nextState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b,
            aLog: aLog, dtBias: dtBias, state: state)

        let actual = nextState[0, 0, 0, 0].item(Float.self)
        let exactDotProduct = 32.0 * (100_000_000.0 + 5.0)
        let expected = Float(100_000_000.0 - 0.5 * exactDotProduct)

        XCTAssertEqual(
            actual.bitPattern, expected.bitPattern,
            "GDN recurrent dot product rounded to \(actual), expected \(expected). "
                + "Naive FP32 accumulation loses the unit terms and returns -1.5e9."
        )
    }

    /// A model in training mode must be able to differentiate the recurrence.
    ///
    /// The fused kernel is a `CustomKernel`, which has no VJP, so the first
    /// backward pass of a LoRA fine-tune of Qwen3.5 or Qwen3-Next died with
    /// `[Primitive::vjp] Not implemented for CustomKernel` - three of every
    /// four layers in those models are gated-delta layers, so this was every
    /// fine-tune of them rather than an edge case. `useKernel: false` takes
    /// the ops path, which is differentiable. The Python model spells the
    /// same thing `use_kernel=not self.training`.
    ///
    /// T spans more than one recompute chunk on purpose: the ops path runs
    /// the recurrence in chunks through a custom function whose backward
    /// recomputes the chunk, and a gradient that stopped at a chunk boundary
    /// would still pass at T <= 16.
    func testGatedDeltaOpsPathIsDifferentiable() throws {
        let inputs = makeInputs(T: 40)
        let loss: ([MLXArray]) -> [MLXArray] = { arrays in
            let (y, _) = gatedDeltaUpdate(
                q: arrays[0], k: arrays[1], v: arrays[2],
                a: inputs.a, b: inputs.b,
                aLog: inputs.aLog, dtBias: inputs.dtBias,
                useKernel: false
            )
            return [y.asType(.float32).square().sum()]
        }
        let (value, gradients) = valueAndGrad(loss, argumentNumbers: [0, 1, 2])(
            [inputs.q, inputs.k, inputs.v])
        eval(value)
        eval(gradients)

        XCTAssertEqual(gradients.count, 3)
        for (name, gradient) in zip(["q", "k", "v"], gradients) {
            let magnitude = abs(gradient.asType(.float32)).max()
            eval(magnitude)
            let largest = magnitude.item(Float.self)
            XCTAssertTrue(
                largest.isFinite,
                "d(loss)/d\(name) is \(largest); the recurrence produced a "
                    + "non-finite gradient."
            )
            XCTAssertGreaterThan(
                largest, 0,
                "d(loss)/d\(name) is all zero, so no gradient reached the "
                    + "input through the recurrence."
            )
        }
    }

    /// Turning the kernel off must not change the answer.
    ///
    /// Training takes the ops path and inference takes the kernel, so a model
    /// would be fine-tuned against arithmetic it never runs at generation
    /// time if these two disagreed. T spans three recompute chunks, which is
    /// where a mistake in carrying state across a chunk boundary would show.
    ///
    /// **Compared relative to the state, not absolutely.** These inputs are
    /// random, so the recurrent state compounds: it reaches ~1.2e6 by T = 40,
    /// where the two paths differ by 1.25 - about one part in a million, and
    /// the same one part in a million as at T = 16, which is a single chunk
    /// and has no boundary to get wrong. An absolute bound would therefore
    /// be a test of how long the sequence is rather than of the arithmetic.
    func testGatedDeltaOpsPathMatchesTheKernel() throws {
        let inputs = makeInputs(T: 40)
        let (yKernel, stateKernel) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias,
            useKernel: true
        )
        let (yOps, stateOps) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias,
            useKernel: false
        )

        /// Largest disagreement as a fraction of the largest value involved.
        func relativeDrift(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
            let left = lhs.asType(.float32)
            let right = rhs.asType(.float32)
            let drift = abs(left - right).max()
            let scale = maximum(abs(left).max(), abs(right).max())
            eval(drift)
            eval(scale)
            let magnitude = scale.item(Float.self)
            return magnitude > 0 ? drift.item(Float.self) / magnitude : drift.item(Float.self)
        }

        XCTAssertLessThan(
            relativeDrift(yKernel, yOps), 1e-4,
            "The differentiable ops path and the fused kernel disagree on the "
                + "output, so training and inference would run different maths."
        )
        XCTAssertLessThan(
            relativeDrift(stateKernel, stateOps), 1e-4,
            "The two paths disagree on the carried recurrent state."
        )
    }

}
