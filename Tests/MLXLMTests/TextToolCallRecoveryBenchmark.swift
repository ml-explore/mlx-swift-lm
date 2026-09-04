// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct TextToolCallRecoveryBenchmark {
    private func nanoseconds(_ elapsed: Duration) -> Double {
        let components = elapsed.components
        return Double(components.seconds) * 1_000_000_000
            + Double(components.attoseconds) / 1_000_000_000
    }

    private func nanosecondsPerChunk(_ elapsed: Duration, iterations: Int) -> Double {
        nanoseconds(elapsed) / Double(iterations)
    }

    @Test("Ordinary text recovery overhead")
    func ordinaryTextOverhead() {
        let schemas: [[String: any Sendable]] = (0 ..< 16).map { index in
            ["function": ["name": "tool_\(index)"] as [String: any Sendable]]
        }
        let disabled = ToolCallProcessor(format: .lfm2)
        let enabled = ToolCallProcessor(format: .lfm2, tools: schemas)
        let chunk = "An ordinary response fragment with no protocol markers. "
        let iterations = 50_000
        let clock = ContinuousClock()
        var consumed = 0

        for _ in 0 ..< 1_000 {
            consumed += disabled.processChunk(chunk)?.utf8.count ?? 0
            consumed += enabled.processChunk(chunk)?.utf8.count ?? 0
        }

        var start = clock.now
        for _ in 0 ..< iterations {
            consumed += disabled.processChunk(chunk)?.utf8.count ?? 0
        }
        let native = nanosecondsPerChunk(clock.now - start, iterations: iterations)

        start = clock.now
        for _ in 0 ..< iterations {
            consumed += enabled.processChunk(chunk)?.utf8.count ?? 0
        }
        let recovery = nanosecondsPerChunk(clock.now - start, iterations: iterations)

        #expect(consumed > 0)
        print(
            String(
                format:
                    "[TOOLHEALBENCH] native %.1f ns/chunk | recovery %.1f ns/chunk | %.2fx",
                native, recovery, recovery / native))
    }

    @Test("Incomplete candidate processing scales near-linearly")
    func incompleteCandidateScaling() {
        let tools: [[String: any Sendable]] = [
            ["function": ["name": "weather"] as [String: any Sendable]]
        ]
        let clock = ContinuousClock()

        func elapsed(for scalarCount: Int) -> Double {
            let payload =
                "<function=weather><parameter=city>"
                + String(repeating: "x", count: scalarCount)
            let characters = Array(payload)
            let chunks = stride(from: 0, to: characters.count, by: 16).map { start in
                String(characters[start ..< min(start + 16, characters.count)])
            }
            let processor = ToolCallProcessor(format: .json, tools: tools)
            let start = clock.now
            for chunk in chunks {
                _ = processor.processChunk(chunk)
            }
            return nanoseconds(clock.now - start)
        }

        // Warm the scanner and allocator before measuring the scaling ratio.
        _ = elapsed(for: 2_000)
        let small = elapsed(for: 16_000)
        let large = elapsed(for: 32_000)

        #expect(large < small * 3.5)
        print(
            String(
                format: "[TOOLHEALBENCH] incomplete 16K %.2f ms | 32K %.2f ms | %.2fx",
                small / 1_000_000, large / 1_000_000, large / small))
    }
}
