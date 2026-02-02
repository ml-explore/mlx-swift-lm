// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

final class WiredMemoryIntegrationTests: XCTestCase {
    enum TestError: Error {
        case missingInfo
        case timeout
        case missingBaseline
    }

    struct DoubleSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +) * 2
        }
    }

    /// Allows only a single active ticket at a time.
    struct SingleActivePolicy: WiredMemoryPolicy, Hashable, Sendable {
        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +)
        }

        func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
            activeSizes.isEmpty
        }
    }

    /// Models policies that share a common weight budget without double counting it.
    struct SharedWeightsPolicy: WiredMemoryPolicy, Hashable, Sendable {
        let name: String
        let sharedBytes: Int

        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + sharedBytes + activeSizes.reduce(0, +)
        }

        func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
            // Admission uses the shared weight budget so each policy stays within
            // its own view of total demand.
            let projected = sharedBytes + activeSizes.reduce(0, +) + max(0, newSize)
            return projected >= 0
        }
    }

    /// Verifies that the default, no-ticket path still supports concurrent inference.
    func testConcurrentInferencesDefaultWiredMemory() async throws {
        let container = try await IntegrationTestModels.shared.llmContainer()
        let parameters = GenerateParameters(maxTokens: 16)
        let prompt = "What is 2+2? Reply with just the number."

        let infos = try await withThrowingTaskGroup(of: GenerateCompletionInfo.self) { group in
            for _ in 0 ..< 2 {
                group.addTask {
                    let input = UserInput(prompt: prompt)
                    let prepared = try await container.prepare(input: input)
                    let stream = try await container.generate(
                        input: prepared,
                        parameters: parameters
                    )

                    var finalInfo: GenerateCompletionInfo?
                    for await generation in stream {
                        if case .info(let info) = generation {
                            finalInfo = info
                        }
                    }

                    guard let finalInfo else {
                        throw TestError.missingInfo
                    }
                    return finalInfo
                }
            }

            var results: [GenerateCompletionInfo] = []
            for try await info in group {
                results.append(info)
            }
            return results
        }

        XCTAssertEqual(infos.count, 2)
        XCTAssertTrue(infos.allSatisfy { $0.generationTokenCount > 0 })
    }

    func testWiredMemoryPolicyStackingAndCustomPolicies() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
        guard maxBytes > 0 else {
            throw XCTSkip("No recommended working set size available.")
        }

        let manager = WiredMemoryManager()
        let sumPolicy = WiredSumPolicy()
        let customPolicy = DoubleSumPolicy()

        let mib = 1024 * 1024
        let sumTicket1 = WiredMemoryTicket(size: 64 * mib, policy: sumPolicy, manager: manager)
        let sumTicket2 = WiredMemoryTicket(size: 64 * mib, policy: sumPolicy, manager: manager)
        let customTicket = WiredMemoryTicket(size: 96 * mib, policy: customPolicy, manager: manager)

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored
        }

        _ = await sumTicket1.start()
        _ = await sumTicket2.start()
        _ = await customTicket.start()
        _ = await customTicket.end()
        _ = await sumTicket2.end()
        _ = await sumTicket1.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        enum PolicyKind {
            case sum
            case custom
        }

        let policyByTicket: [UUID: PolicyKind] = [
            sumTicket1.id: .sum,
            sumTicket2.id: .sum,
            customTicket.id: .custom,
        ]

        var active: [UUID: Int] = [:]
        var sawCustomDominant = false
        var sawTwoPoliciesActive = false

        for event in events {
            switch event.kind {
            case .ticketStarted:
                guard let id = event.ticketID, let size = event.size else { continue }
                active[id] = size
            case .ticketEnded:
                if let id = event.ticketID {
                    active.removeValue(forKey: id)
                }
            case .limitApplied:
                let sumSizes = active.reduce(0) { partial, entry in
                    guard let kind = policyByTicket[entry.key], kind == .sum else { return partial }
                    return partial + entry.value
                }
                let customSizes = active.reduce(0) { partial, entry in
                    guard let kind = policyByTicket[entry.key], kind == .custom else {
                        return partial
                    }
                    return partial + entry.value
                }

                if sumSizes > 0 && customSizes > 0 {
                    sawTwoPoliciesActive = true
                }

                let sumLimit = baseline + sumSizes
                let customLimit = baseline + (customSizes * 2)
                let expected = max(sumLimit, customLimit)

                if customSizes > 0 && expected == customLimit && customLimit > sumLimit {
                    sawCustomDominant = true
                }

                XCTAssertEqual(event.appliedLimit, expected)
            default:
                break
            }
        }

        XCTAssertTrue(sawCustomDominant, "Expected custom policy to influence the applied limit.")
        XCTAssertTrue(sawTwoPoliciesActive, "Expected tickets from two policies to overlap.")
    }

    /// Verifies that passing a wired memory ticket to inference results in
    /// ticket lifecycle events and limit updates.
    func testGenerateEmitsTicketLifecycleEvents() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let container = try await IntegrationTestModels.shared.llmContainer()
        let parameters = GenerateParameters(maxTokens: 8)
        let prompt = "Write one short sentence about the ocean."

        let manager = WiredMemoryManager()
        let policy = WiredSumPolicy()
        let ticket = WiredMemoryTicket(
            size: 64 * 1024 * 1024,
            policy: policy,
            manager: manager,
            kind: .active
        )

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        let input = UserInput(prompt: prompt)
        let prepared = try await container.prepare(input: input)
        let genStream = try await container.generate(
            input: prepared,
            parameters: parameters,
            wiredMemoryTicket: ticket
        )

        for await _ in genStream {}

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        let sawStart = events.contains { $0.kind == .ticketStarted && $0.ticketID == ticket.id }
        let sawEnd = events.contains { $0.kind == .ticketEnded && $0.ticketID == ticket.id }
        XCTAssertTrue(sawStart, "Expected ticket to start during inference.")
        XCTAssertTrue(sawEnd, "Expected ticket to end after inference completes.")
    }

    /// Ensures admission control can serialize inference when a policy denies
    /// concurrent tickets.
    func testAdmissionGatingSerializesInference() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let container = try await IntegrationTestModels.shared.llmContainer()
        let parameters = GenerateParameters(maxTokens: 32)
        let prompt = "Explain why tests exist in one sentence."

        let manager = WiredMemoryManager()
        let policy = SingleActivePolicy()

        let ticketA = WiredMemoryTicket(
            id: UUID(),
            size: 64 * 1024 * 1024,
            policy: policy,
            manager: manager,
            kind: .active
        )
        let ticketB = WiredMemoryTicket(
            id: UUID(),
            size: 64 * 1024 * 1024,
            policy: policy,
            manager: manager,
            kind: .active
        )

        let stream = await manager.events()
        let gateStream = await manager.events()

        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        let runInference: (WiredMemoryTicket) async throws -> Void = { ticket in
            let input = UserInput(prompt: prompt)
            let prepared = try await container.prepare(input: input)
            let genStream = try await container.generate(
                input: prepared,
                parameters: parameters,
                wiredMemoryTicket: ticket
            )
            for await _ in genStream {}
        }

        async let first = runInference(ticketA)

        // Wait until the first ticket becomes active before launching the second.
        _ = try await Self.collectEvents(stream: gateStream) { event in
            event.kind == .ticketStarted && event.ticketID == ticketA.id
        }
        try await Task.sleep(nanoseconds: 50_000_000)

        async let second = runInference(ticketB)

        _ = try await (first, second)

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        let sawAdmissionWait = events.contains {
            $0.kind == .admissionWait && $0.ticketID == ticketB.id
        }
        XCTAssertTrue(sawAdmissionWait, "Expected second ticket to wait for admission.")
    }

    /// Demonstrates shared embedding weights across two policies without double counting.
    func testSharedWeightsAcrossPoliciesUsesMaxNotSum() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
        guard maxBytes > 0 else {
            throw XCTSkip("No recommended working set size available.")
        }

        let manager = WiredMemoryManager()
        let mib = 1024 * 1024
        let sharedWeights = 256 * mib

        let embeddingPolicy = SharedWeightsPolicy(name: "embedding", sharedBytes: sharedWeights)
        let chatPolicy = SharedWeightsPolicy(name: "chat", sharedBytes: sharedWeights)

        let embedTicket = WiredMemoryTicket(
            size: 64 * mib,
            policy: embeddingPolicy,
            manager: manager,
            kind: .active
        )
        let chatTicket = WiredMemoryTicket(
            size: 192 * mib,
            policy: chatPolicy,
            manager: manager,
            kind: .active
        )

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await embedTicket.start()
        _ = await chatTicket.start()
        _ = await chatTicket.end()
        _ = await embedTicket.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        // Each policy includes the shared weights; the manager selects the max.
        let embedLimit = baseline + sharedWeights + (64 * mib)
        let chatLimit = baseline + sharedWeights + (192 * mib)
        let summedLimit = baseline + sharedWeights + (64 * mib) + (192 * mib)

        let sawEmbed = events.contains { $0.kind == .limitApplied && $0.appliedLimit == embedLimit }
        let sawChat = events.contains { $0.kind == .limitApplied && $0.appliedLimit == chatLimit }
        let sawSummed = events.contains {
            $0.kind == .limitApplied && $0.appliedLimit == summedLimit
        }

        XCTAssertTrue(sawEmbed, "Expected embedding policy limit to be applied.")
        XCTAssertTrue(sawChat, "Expected chat policy limit to be applied.")
        XCTAssertFalse(sawSummed, "Expected shared weights not to be double counted.")
    }

    /// Measurement helpers should return a budget at least as large as the observed peak
    /// and not exceed a reasonable threshold beyond it.
    ///
    /// ## Methodology
    /// - Model weights: extremely stable, derived from unchanging nbytes
    /// - KV: pretty stable with deterministic chunk size, measurable by nbytes
    /// - Attn workspace: high variability, changes with kernel code updates
    ///
    /// This test ensures the utilities for estimating model weights are accurate, such that you
    /// can reliably use `.weightBytes` as a `.reservation` ticket, and the remaining
    /// usages for `.active` tickets.
    ///
    /// We confirm that attention workspace comes out nearly the same with two different input
    /// length prefills to confirm extraction for attention is accurate.
    func testMeasurementBudgetNotBelowPeak() async throws {
        let container = try await IntegrationTestModels.shared.llmContainer()
        let smallTokenCount = 128
        let largeTokenCount = 512
        let prefillStepSize = 128
        let smallParameters = GenerateParameters(maxTokens: 1, prefillStepSize: prefillStepSize)
        let largeParameters = GenerateParameters(maxTokens: 1, prefillStepSize: prefillStepSize)

        let smallResult = try await container.perform { context in
            Memory.clearCache()
            let previousPeak = Memory.peakMemory
            Memory.peakMemory = 0
            _ = previousPeak
            return try await WiredMemoryUtils.tune(
                context: context,
                tokenCount: smallTokenCount,
                parameters: smallParameters,
                resetPeakMemory: false
            )
        }

        let largeResult = try await container.perform { context in
            Memory.clearCache()
            let previousPeak = Memory.peakMemory
            Memory.peakMemory = 0
            _ = previousPeak
            return try await WiredMemoryUtils.tune(
                context: context,
                tokenCount: largeTokenCount,
                parameters: largeParameters,
                resetPeakMemory: false
            )
        }

        XCTAssertGreaterThan(smallResult.weightBytes, 0)
        XCTAssertGreaterThan(smallResult.kvBytes, 0)
        XCTAssertGreaterThan(smallResult.peakActiveBytes, 0)
        XCTAssertGreaterThanOrEqual(smallResult.totalBytes, smallResult.peakActiveBytes)

        XCTAssertGreaterThan(largeResult.weightBytes, 0)
        XCTAssertGreaterThan(largeResult.kvBytes, 0)
        XCTAssertGreaterThan(largeResult.peakActiveBytes, 0)
        XCTAssertGreaterThanOrEqual(largeResult.totalBytes, largeResult.peakActiveBytes)

        XCTAssertGreaterThanOrEqual(largeResult.kvBytes, smallResult.kvBytes)
        XCTAssertGreaterThanOrEqual(largeResult.peakActiveBytes, smallResult.peakActiveBytes)

        let mib = 1024 * 1024
        let tolerance = 1 * mib
        XCTAssertLessThanOrEqual(
            smallResult.totalBytes,
            smallResult.peakActiveBytes + tolerance,
            "Expected measured budget to stay near the observed peak (small)."
        )
        XCTAssertLessThanOrEqual(
            largeResult.totalBytes,
            largeResult.peakActiveBytes + tolerance,
            "Expected measured budget to stay near the observed peak (large)."
        )

        let workspaceDelta = abs(smallResult.workspaceBytes - largeResult.workspaceBytes)
        XCTAssertLessThanOrEqual(
            workspaceDelta,
            tolerance,
            "Expected workspace estimate to be stable across identical prefill sizes."
        )
    }

    /// Collects events until the predicate matches or a timeout fires.
    private static func collectEvents(
        stream: AsyncStream<WiredMemoryEvent>,
        until predicate: @Sendable @escaping (WiredMemoryEvent) -> Bool,
        timeout: TimeInterval = 10
    ) async throws -> [WiredMemoryEvent] {
        return try await withThrowingTaskGroup(of: [WiredMemoryEvent].self) { group in
            group.addTask {
                var events: [WiredMemoryEvent] = []
                for await event in stream {
                    events.append(event)
                    if predicate(event) {
                        break
                    }
                }
                return events
            }

            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw TestError.timeout
            }

            let result = try await group.next()
            group.cancelAll()
            return result ?? []
        }
    }
}
