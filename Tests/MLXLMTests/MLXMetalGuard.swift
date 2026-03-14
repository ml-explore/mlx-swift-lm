// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

/// Checks whether the MLX Metal backend is functional (i.e., the metallib is loaded).
///
/// In SPM debug builds (`swift test`), the Metal shader library (`.metallib`) is not
/// bundled, causing any GPU evaluation to fail. Tests that require Metal evaluation
/// should call `try skipIfMetalUnavailable()` at the top of their test body so they
/// are gracefully skipped instead of crashing the test runner.
///
/// When running through Xcode (which correctly bundles the metallib), all tests
/// execute normally.
enum MLXMetalGuard {
    /// Cached result so we only probe once per process.
    private static let _isAvailable: Bool = {
        // Use withError to install the error handler BEFORE any MLX operations.
        // This converts the C-level mlx_error (which by default calls exit(-1))
        // into a Swift throw, allowing graceful detection.
        do {
            try withError {
                let probe = MLXArray([1])
                eval(probe)
            }
            return true
        } catch {
            return false
        }
    }()

    /// `true` when MLX Metal evaluation works.
    static var isAvailable: Bool { _isAvailable }
}

/// Call at the top of any XCTest method that requires MLX Metal evaluation.
///
/// Usage:
/// ```swift
/// func testSomethingWithMetal() throws {
///     try skipIfMetalUnavailable()
///     // … test body using .item(), eval(), etc.
/// }
/// ```
func skipIfMetalUnavailable() throws {
    try XCTSkipUnless(
        MLXMetalGuard.isAvailable,
        "MLX Metal library unavailable (SPM debug build) — skipping"
    )
}
