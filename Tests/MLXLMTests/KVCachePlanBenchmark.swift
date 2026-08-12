// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Microbenchmark for dynamic KV-cache application on the decode hot path.
///
/// The traversal measurement intentionally runs only the minimal conversion
/// walk. The previous implementation also collected before/after leaves,
/// built a runtime report, counted outcomes, and allocated a result, so this
/// is a conservative lower bound for the work removed by terminal plans.
@Suite(.serialized)
struct KVCachePlanBenchmark {
    private func nanosecondsPerCall(_ elapsed: Duration, iterations: Int) -> Double {
        let components = elapsed.components
        let nanoseconds =
            Double(components.seconds) * 1_000_000_000
            + Double(components.attoseconds) / 1_000_000_000
        return nanoseconds / Double(iterations)
    }

    @Test func completedApplicationCost() {
        let configuration = KVCacheConfiguration(
            strategy: .affine(.fourBit), compatibility: .allowPartial)
        let plan = KVCachePlan(configuration: configuration)
        let caches: [KVCache] = (0 ..< 32).map { _ in
            QuantizedKVCache(groupSize: 64, bits: 4)
        }
        let storage = KVCacheStorage(caches, plan: plan)
        plan.apply(to: storage)
        #expect(storage.isApplicationTerminal)

        let clock = ContinuousClock()
        let warmup = 1_000
        let iterations = 100_000

        for _ in 0 ..< warmup { plan.apply(to: storage) }
        var start = clock.now
        for _ in 0 ..< iterations { plan.apply(to: storage) }
        let terminal = nanosecondsPerCall(clock.now - start, iterations: iterations)

        var traversed = caches
        for _ in 0 ..< warmup {
            _ = applyKVCacheConfigurationFast(
                cache: &traversed, configuration: configuration)
        }
        start = clock.now
        for _ in 0 ..< iterations {
            _ = applyKVCacheConfigurationFast(
                cache: &traversed, configuration: configuration)
        }
        let traversal = nanosecondsPerCall(clock.now - start, iterations: iterations)

        print(
            String(
                format:
                    "[KVCACHEBENCH] terminal %.1f ns/call | traversal floor %.1f ns/call | %.1fx faster",
                terminal, traversal, traversal / terminal))
    }

    @Test func sharedProgressReadAvoidsLayerScan() {
        let caches: [KVCache] = (0 ..< 64).map { _ in
            let cache = KVCacheSimple()
            cache.offset = 1_024
            return cache
        }
        let storage = KVCacheStorage(
            caches, plan: .disabled, processedTokenCount: 1_024)
        let iterations = 100_000
        let clock = ContinuousClock()
        var directMatches = 0
        var scannedMatches = 0

        var start = clock.now
        for _ in 0 ..< iterations {
            if storage.processedTokenCount == 1_024 { directMatches += 1 }
        }
        let direct = nanosecondsPerCall(clock.now - start, iterations: iterations)

        start = clock.now
        for _ in 0 ..< iterations {
            if caches.allSatisfy({ $0.offset == 1_024 }) { scannedMatches += 1 }
        }
        let scan = nanosecondsPerCall(clock.now - start, iterations: iterations)

        #expect(directMatches == iterations)
        #expect(scannedMatches == iterations)
        print(
            String(
                format:
                    "[KVCACHEBENCH] progress %.1f ns/read | 64-layer scan %.1f ns/read | %.1fx faster",
                direct, scan, scan / direct))
    }
}
