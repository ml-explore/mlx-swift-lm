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
        MLXRandom.seed(seed)
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
        var qP = inputs.q
        var kP = inputs.k
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

}
