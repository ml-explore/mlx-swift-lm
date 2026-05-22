// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import Testing

/// Pinned revision of the `angelsbrood/gemma4-mtp-fixtures` HF dataset for
/// byte-exact fixture parity. Bump when the dataset is regenerated.
private let fixturesRevision = "152a8ea4cd9e58da11b0c4b39542d3ad347fce06"

private func loadFixtureMaskOrSkip(_ name: String) async -> MLXArray? {
    let dir: URL
    do {
        dir = try await downloadDataset(
            repo: "angelsbrood/gemma4-mtp-fixtures",
            revision: fixturesRevision,
            matching: ["masks/*.safetensors"]
        )
    } catch {
        Issue.record(
            "masks/\(name) unavailable (dataset fetch failed): \(error.localizedDescription)")
        return nil
    }
    let url = dir.appendingPathComponent("masks").appendingPathComponent(name)
    do {
        let arrays = try MLX.loadArrays(url: url)
        guard let mask = arrays["mask"] else {
            Issue.record("fixture missing 'mask' tensor: masks/\(name)")
            return nil
        }
        return mask
    } catch {
        Issue.record("masks/\(name) load failed: \(error.localizedDescription)")
        return nil
    }
}

@Test
func testBidirectionalMaskMatchesFixtureQ1Kv8() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_q1_kv8.safetensors") else {
        return
    }
    let swift = createBidirectionalMask(queryLen: 1, kvLen: 8, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalMaskMatchesFixtureQ2Kv16() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_q2_kv16.safetensors") else {
        return
    }
    let swift = createBidirectionalMask(queryLen: 2, kvLen: 16, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalMaskMatchesFixtureQ4Kv4() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_q4_kv4.safetensors") else {
        return
    }
    let swift = createBidirectionalMask(queryLen: 4, kvLen: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ1Kv8W4() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_swa_q1_kv8_w4.safetensors")
    else { return }
    let swift = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 8, windowSize: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ1Kv16W4() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_swa_q1_kv16_w4.safetensors")
    else { return }
    let swift = createBidirectionalSlidingWindowMask(
        queryLen: 1, kvLen: 16, windowSize: 4, dtype: .float32)
    #expect(swift.shape == reference.shape)
    #expect(allClose(swift, reference).item(Bool.self))
}

@Test
func testBidirectionalSWAMaskMatchesFixtureQ2Kv16W4() async {
    guard let reference = await loadFixtureMaskOrSkip("bidirectional_swa_q2_kv16_w4.safetensors")
    else { return }
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
    let firstRow = swa.asArray(Float.self)
    #expect(firstRow.allSatisfy { $0 == -Float.infinity })
}
