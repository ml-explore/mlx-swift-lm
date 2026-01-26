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
