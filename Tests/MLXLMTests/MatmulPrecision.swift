// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing
import XCTest

/// The precision MLX uses for float32 matmuls on the machine running the tests, measured once.
///
/// On hardware with neural accelerators (M5 and later) MLX runs float32 matmuls as TF32 unless
/// `MLX_ENABLE_TF32=0`. That costs about three decimal digits, which matters for every test
/// that computes one prompt two ways — whole against windowed, cold against warm — and asserts
/// the logits agree. Those differences are ~1e-6 with IEEE float32 and ~3e-3 with TF32, so a
/// fixed bound sized for one machine fails on the other for reasons that have nothing to do
/// with the code under test.
///
/// Pinning `MLX_ENABLE_TF32=0` for the test process would remove the variable, but it would
/// also stop testing the path that ships by default on current hardware, and it is read once
/// per process at MLX's first use, so it would have to be set before any test runs. Measure
/// instead.
enum MatmulPrecision {

    /// True when float32 matmuls lose precision (TF32 on the neural accelerators).
    ///
    /// Measured rather than inferred from the device or `MLX_ENABLE_TF32`: the same float32
    /// inputs are multiplied on the GPU and on the CPU, which always uses IEEE float32. The
    /// gap is ~1e-7 relative when the GPU does the same, and ~1e-3 when it uses TF32 — three
    /// orders of magnitude apart, so the threshold between them needs no tuning.
    static let isReducedFloat32: Bool = relativeMatmulError > 1e-5

    /// Relative difference between a GPU and a CPU float32 matmul of the same inputs.
    static let relativeMatmulError: Float = {
        let a = withRandomState(MLXRandom.RandomState(seed: 0)) { MLXRandom.normal([64, 512]) }
        let b = withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([512, 64]) }
        let gpu = matmul(a, b, stream: .gpu)
        let cpu = matmul(a, b, stream: .cpu)
        return abs(gpu - cpu).max().item(Float.self) / abs(cpu).max().item(Float.self)
    }()

    /// A tolerance sized for IEEE float32, and the value to use instead when this machine
    /// computes float32 matmuls as TF32.
    ///
    /// Two explicit numbers rather than one scale factor: what reduced precision costs a
    /// comparison depends on how many matmuls feed it and whether the difference is relative
    /// or absolute, so each call site states what it measured rather than inheriting a guess.
    static func tolerance(float32: Double, reduced: Double) -> Double {
        isReducedFloat32 ? reduced : float32
    }

    static func tolerance(float32: Float, reduced: Float) -> Float {
        isReducedFloat32 ? reduced : float32
    }

    /// The bound for asserting that two executions of the same prompt produce the same logits.
    ///
    /// `1e-3` is what these tests have always used, and it stays exactly that on hardware
    /// that computes float32 matmuls in float32. TF32 raises the noise of splitting a forward
    /// to a few times 1e-3; 2e-2 clears that with margin while staying far below the O(1)
    /// shift a real positioning error produces.
    static let splitTolerance: Float = tolerance(float32: 1e-3, reduced: 2e-2)

    /// Nil when `|actual - expected| <= atol + rtol * |expected|` elementwise, else a message
    /// with the worst difference. `atol` applies with IEEE float32, `reducedAtol` with TF32.
    static func mismatch(
        _ actual: MLXArray, _ expected: MLXArray, rtol: Double, atol: Double, reducedAtol: Double
    ) -> String? {
        guard actual.shape == expected.shape else {
            return "shape \(actual.shape) != \(expected.shape)"
        }
        let atol = tolerance(float32: atol, reduced: reducedAtol)
        if allClose(actual, expected, rtol: rtol, atol: atol).item(Bool.self) {
            return nil
        }
        let worst = abs(actual - expected).max().item(Float.self)
        let precision = isReducedFloat32 ? "TF32" : "float32"
        return "values differ by up to \(worst) (rtol \(rtol), atol \(atol), \(precision))"
    }
}

/// Expect two float32 results to agree when they come from different matmul shapes or kernels
/// (chunked, cached, quantized, or an external reference). See ``MatmulPrecision``.
func expectClose(
    _ actual: MLXArray, _ expected: MLXArray, rtol: Double = 0, atol: Double,
    reducedAtol: Double, sourceLocation: SourceLocation = #_sourceLocation
) {
    let mismatch = MatmulPrecision.mismatch(
        actual, expected, rtol: rtol, atol: atol, reducedAtol: reducedAtol)
    #expect(mismatch == nil, "\(mismatch ?? "")", sourceLocation: sourceLocation)
}

/// ``expectClose(_:_:rtol:atol:reducedAtol:sourceLocation:)`` against reference values.
func expectClose(
    _ actual: MLXArray, _ expected: [Float], rtol: Double = 0, atol: Double,
    reducedAtol: Double, sourceLocation: SourceLocation = #_sourceLocation
) {
    guard actual.size == expected.count else {
        Issue.record(
            "\(actual.size) values, expected \(expected.count)", sourceLocation: sourceLocation)
        return
    }
    expectClose(
        actual, MLXArray(expected, actual.shape), rtol: rtol, atol: atol,
        reducedAtol: reducedAtol, sourceLocation: sourceLocation)
}

/// The XCTest form of ``expectClose(_:_:rtol:atol:reducedAtol:sourceLocation:)``.
func assertClose(
    _ actual: MLXArray, _ expected: MLXArray, rtol: Double = 0, atol: Double,
    reducedAtol: Double, _ message: String = "", file: StaticString = #filePath,
    line: UInt = #line
) {
    if let mismatch = MatmulPrecision.mismatch(
        actual, expected, rtol: rtol, atol: atol, reducedAtol: reducedAtol)
    {
        XCTFail(
            [message, mismatch].filter { !$0.isEmpty }.joined(separator: ": "), file: file,
            line: line)
    }
}

final class MatmulPrecisionTests: XCTestCase {

    /// The probe decides how strict every tolerance derived from it is, so check it on every
    /// machine rather than only on the rare one that sets `MLX_ENABLE_TF32=0`.
    ///
    /// What can be asserted anywhere is that the measurement is unambiguous: float32 lands near
    /// 1e-7 and TF32 near 1e-3, so a reading between 1e-5 and 1e-4 means the two regimes are no
    /// longer cleanly separated and the threshold needs a human. On pre-M5 hardware — CI, and
    /// most developer machines — this is the assertion that catches a probe that has started
    /// reporting reduced precision and quietly loosening every bound.
    ///
    /// There is no converse check that accelerator hardware *must* measure as reduced: nothing
    /// exposes "this GPU has neural accelerators", and MLX is free to change when it uses them.
    func testPrecisionProbeLandsClearlyInOneRegime() {
        let error = MatmulPrecision.relativeMatmulError
        XCTAssertTrue(
            error < 1e-5 || error > 1e-4,
            """
            float32 matmul relative error \(error) falls between the float32 (~1e-7) and \
            TF32 (~1e-3) regimes; MatmulPrecision.isReducedFloat32 can no longer tell them apart
            """)

        // When TF32 is disabled the answer is known, whatever the hardware.
        if ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" {
            XCTAssertFalse(
                MatmulPrecision.isReducedFloat32,
                "float32 matmuls measured as reduced precision despite MLX_ENABLE_TF32=0")
        }
    }
}
