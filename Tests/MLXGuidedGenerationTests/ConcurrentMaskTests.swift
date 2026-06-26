// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXGuidedGeneration

/// Tests for concurrent mask computation in GuidedGenerationLoop.
///
/// The loop pre-computes the grammar mask while the GPU forward pass runs,
/// overlapping CPU and GPU work. These tests verify the correctness of that
/// overlap.
@Suite
struct ConcurrentMaskTests {

    // MARK: - applyMaskAndSample Tests

    @Test
    func applyMaskAndSampleSelectsAllowedToken() throws {
        // Synthetic logits: token 123 ('{') has the highest logit value.
        // Build a mask that only allows token 123.
        var floats = [Float](repeating: Float(0.0), count: 256)
        floats[123] = 10.0  // '{' gets high logit
        floats[65] = 20.0  // 'A' gets even higher, but will be masked
        let logits = MLXArray(floats)

        // Build a bitmask allowing only token 123
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[123 / 32] |= (1 << (123 % 32))

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...], maskArray: maskArray)

        #expect(result == 123, "Should select token 123 -- the only allowed token")
    }

    @Test
    func applyMaskAndSampleWithNilMaskSelectsArgmax() throws {
        // When sampleMask is nil (unconditional splice), argmax of raw logits
        var floats = [Float](repeating: Float(0.0), count: 256)
        floats[42] = 100.0
        let logits = MLXArray(floats)

        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...], maskArray: nil)

        #expect(result == 42, "Should select argmax token when no mask applied")
    }

    @Test
    func applyMaskAndSampleHandlesMultipleAllowedTokens() throws {
        // Multiple allowed tokens: argmax of (logit + mask) picks highest
        var floats = [Float](repeating: Float(0.0), count: 256)
        floats[48] = 5.0  // '0'
        floats[49] = 10.0  // '1'
        floats[50] = 3.0  // '2'
        let logits = MLXArray(floats)

        // Allow tokens 48, 49, 50
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[48 / 32] |= (1 << (48 % 32))
        maskWords[49 / 32] |= (1 << (49 % 32))
        maskWords[50 / 32] |= (1 << (50 % 32))

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...], maskArray: maskArray)

        #expect(result == 49, "Should select token 49 -- highest logit among allowed tokens")
    }
}
