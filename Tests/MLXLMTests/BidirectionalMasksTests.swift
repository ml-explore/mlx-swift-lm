// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Locate `tools/fixtures/masks/` relative to this test file. Walks up the
/// directory tree from `#filePath` until it finds the `tools/fixtures/masks`
/// subdirectory. Tests skip (with a recorded warning) if the fixtures aren't
/// present — supports running the suite in environments without the fixture
/// generation step having been run.
private func fixturesDir(file: String = #filePath) -> URL? {
    var dir = URL(fileURLWithPath: file).deletingLastPathComponent()
    let fs = FileManager.default
    for _ in 0 ..< 10 {
        let candidate = dir.appendingPathComponent("tools/fixtures/masks")
        if fs.fileExists(atPath: candidate.path) {
            return candidate
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return nil
}

private func loadFixtureMask(_ name: String) throws -> MLXArray? {
    guard let dir = fixturesDir() else { return nil }
    let url = dir.appendingPathComponent(name)
    let arrays = try MLX.loadArrays(url: url)
    return arrays["mask"]
}

@Test
func testBidirectionalMaskMatchesFixtureQ1Kv8() throws {
    guard let reference = try loadFixtureMask("bidirectional_q1_kv8.safetensors") else {
        Issue.record("fixture not found: tools/fixtures/masks/bidirectional_q1_kv8.safetensors")
        return
    }
    let swift = createBidirectionalMask(queryLen: 1, kvLen: 8, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalMaskMatchesFixtureQ2Kv16() throws {
    guard let reference = try loadFixtureMask("bidirectional_q2_kv16.safetensors") else {
        Issue.record("fixture not found: tools/fixtures/masks/bidirectional_q2_kv16.safetensors")
        return
    }
    let swift = createBidirectionalMask(queryLen: 2, kvLen: 16, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalMaskMatchesFixtureQ4Kv4() throws {
    guard let reference = try loadFixtureMask("bidirectional_q4_kv4.safetensors") else {
        Issue.record("fixture not found: tools/fixtures/masks/bidirectional_q4_kv4.safetensors")
        return
    }
    let swift = createBidirectionalMask(queryLen: 4, kvLen: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ1Kv8W4() throws {
    guard let reference = try loadFixtureMask("bidirectional_swa_q1_kv8_w4.safetensors") else {
        Issue.record(
            "fixture not found: tools/fixtures/masks/bidirectional_swa_q1_kv8_w4.safetensors")
        return
    }
    let swift = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 8, windowSize: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    // Use array comparison that tolerates -inf (allClose treats inf vs inf as equal)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ1Kv16W4() throws {
    guard let reference = try loadFixtureMask("bidirectional_swa_q1_kv16_w4.safetensors") else {
        Issue.record(
            "fixture not found: tools/fixtures/masks/bidirectional_swa_q1_kv16_w4.safetensors")
        return
    }
    let swift = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 16, windowSize: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ2Kv16W4() throws {
    guard let reference = try loadFixtureMask("bidirectional_swa_q2_kv16_w4.safetensors") else {
        Issue.record(
            "fixture not found: tools/fixtures/masks/bidirectional_swa_q2_kv16_w4.safetensors")
        return
    }
    let swift = createBidirectionalSlidingWindowMask(
        queryLen: 2, kvLen: 16, windowSize: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskDegenerateLargeWindow() {
    // When windowSize >= kvLen, all positions attend → matches the bidirectional
    // full mask. No fixture needed; semantic invariant.
    let swa = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 4, windowSize: 8, dtype: .float32)
    let full = createBidirectionalMask(queryLen: 1, kvLen: 4, dtype: .float32)
    #expect(allClose(swa, full).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskZeroWindow() {
    // windowSize == 0 should mask everything.
    let swa = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 4, windowSize: 0, dtype: .float32)
    #expect(swa.shape == [1, 4])
    // All entries should be -inf.
    let firstRow = swa.asArray(Float.self)
    #expect(firstRow.allSatisfy { $0 == -Float.infinity })
}
