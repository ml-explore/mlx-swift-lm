// Copyright © 2024 Apple Inc.

import Darwin
import Dispatch
import Foundation
import MLX

@_silgen_name("mlx_set_wired_limit")
private func mlx_set_wired_limit(_ res: UnsafeMutablePointer<size_t>, _ limit: size_t) -> Int32

private enum WiredMemoryBackend {
    static var isSupported: Bool {
        Device.defaultDevice().deviceType == .gpu
    }

    static func readCurrentLimit() -> Int? {
        guard isSupported else { return nil }
        var previous: size_t = 0
        let result = mlx_set_wired_limit(&previous, 0)
        guard result == 0 else { return nil }
        var tmp: size_t = 0
        _ = mlx_set_wired_limit(&tmp, previous)
        return Int(previous)
    }

    static func applyLimit(_ limit: Int) -> Bool {
        guard isSupported else { return false }
        guard limit >= 0 else { return false }
        var previous: size_t = 0
        let result = mlx_set_wired_limit(&previous, size_t(limit))
        return result == 0
    }
}

private func maxRecommendedWorkingSetBytes() -> Int? {
    let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
    guard maxBytes > 0 else { return nil }
    if maxBytes > UInt64(Int.max) {
        return Int.max
    }
    return Int(maxBytes)
}

/// Debug event emitted by ``WiredMemoryManager`` when coordinating wired memory changes.
public struct WiredMemoryEvent: Sendable {
    public enum Kind: String, Sendable {
        case baselineCaptured
        case admissionWait
        case admissionGranted
        case admissionCancelled
        case ticketStarted
        case ticketStartIgnored
        case ticketEnded
        case ticketEndIgnored
        case limitComputed
        case limitApplied
        case limitApplyFailed
        case baselineRestored
    }

    public let sequence: UInt64
    public let timestamp: Date
    public let kind: Kind
    public let ticketID: UUID?
    public let size: Int?
    public let policy: String?
    public let baseline: Int?
    public let desiredLimit: Int?
    public let appliedLimit: Int?
    public let activeCount: Int
    public let waiterCount: Int

    public init(
        sequence: UInt64,
        timestamp: Date,
        kind: Kind,
        ticketID: UUID? = nil,
        size: Int? = nil,
        policy: String? = nil,
        baseline: Int? = nil,
        desiredLimit: Int? = nil,
        appliedLimit: Int? = nil,
        activeCount: Int,
        waiterCount: Int
    ) {
        self.sequence = sequence
        self.timestamp = timestamp
        self.kind = kind
        self.ticketID = ticketID
        self.size = size
        self.policy = policy
        self.baseline = baseline
        self.desiredLimit = desiredLimit
        self.appliedLimit = appliedLimit
        self.activeCount = activeCount
        self.waiterCount = waiterCount
    }
}

/// Policy for computing a process-global wired memory limit.
///
/// - Note: To group tickets correctly, policies should be pure and either Hashable
///   or reference types. Non-hashable value types are treated as distinct policies.
public protocol WiredMemoryPolicy: Sendable {
    /// Compute the desired wired limit in bytes for the current active set.
    func limit(baseline: Int, activeSizes: [Int]) -> Int

    /// Decide whether a new ticket can be admitted. Defaults to allowing all tickets.
    func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool
}

extension WiredMemoryPolicy {
    public func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        true
    }
}

/// Sum policy: baseline + sum(activeSizes), optionally capped.
public struct WiredSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
    public let cap: Int?

    public init(cap: Int? = nil) {
        self.cap = cap
    }

    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        let sum = activeSizes.reduce(0, +)
        return clamp(baseline + sum)
    }

    public func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        let projected = baseline + activeSizes.reduce(0, +) + max(0, newSize)
        return clamp(projected) == projected
    }

    private func clamp(_ value: Int) -> Int {
        if let cap {
            return min(value, max(0, cap))
        }
        if let maxBytes = maxRecommendedWorkingSetBytes() {
            return min(value, maxBytes)
        }
        return value
    }
}

/// Max policy: max(baseline, max(activeSizes)).
public struct WiredMaxPolicy: WiredMemoryPolicy, Hashable, Sendable {
    public init() {}

    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        let maxSize = activeSizes.max() ?? 0
        return max(baseline, maxSize)
    }
}

/// Fixed policy: limit is constant while any ticket is active.
public struct WiredFixedPolicy: WiredMemoryPolicy, Hashable, Sendable {
    public let bytes: Int

    public init(limit: Int) {
        self.bytes = limit
    }

    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        bytes
    }
}

/// Handle for coordinating wired memory changes across concurrent work.
public struct WiredMemoryTicket: Sendable, Identifiable {
    public let id: UUID
    public let size: Int
    public let manager: WiredMemoryManager
    public let policy: any WiredMemoryPolicy

    public init(
        id: UUID = UUID(),
        size: Int,
        policy: any WiredMemoryPolicy,
        manager: WiredMemoryManager = .shared
    ) {
        self.id = id
        self.size = size
        self.policy = policy
        self.manager = manager
    }

    public func start() async -> Int {
        await manager.start(id: id, size: size, policy: policy)
    }

    public func end() async -> Int {
        await manager.end(id: id, policy: policy)
    }
}

extension WiredMemoryTicket {
    public static func withWiredLimit<R>(
        _ ticket: WiredMemoryTicket,
        _ body: () async throws -> R
    ) async rethrows -> R {
        _ = await ticket.start()
        return try await withTaskCancellationHandler {
            do {
                let result = try await body()
                _ = await ticket.end()
                return result
            } catch {
                _ = await ticket.end()
                throw error
            }
        } onCancel: {
            Task { _ = await ticket.end() }
        }
    }

    public func withWiredLimit<R>(_ body: () async throws -> R) async rethrows -> R {
        try await Self.withWiredLimit(self, body)
    }

    public static func withWiredLimitSync<R>(
        _ ticket: WiredMemoryTicket,
        _ body: () throws -> R
    ) rethrows -> R {
        let startSemaphore = DispatchSemaphore(value: 0)
        Task {
            _ = await ticket.start()
            startSemaphore.signal()
        }
        startSemaphore.wait()

        defer {
            let endSemaphore = DispatchSemaphore(value: 0)
            Task {
                _ = await ticket.end()
                endSemaphore.signal()
            }
            endSemaphore.wait()
        }

        return try body()
    }
}

public actor WiredMemoryManager {
    public static let shared = WiredMemoryManager()

    #if DEBUG
        private var eventContinuations: [UUID: AsyncStream<WiredMemoryEvent>.Continuation] = [:]
        private var eventSequence: UInt64 = 0
    #endif

    private struct TicketState {
        let size: Int
        let policyKey: PolicyKey
        let policyLabel: String
    }

    private enum PolicyKey: Hashable {
        case hash(AnyHashable)
        case object(ObjectIdentifier)
        case unique(UUID)
    }

    private var baseline: Int?
    private var active: [UUID: TicketState] = [:]
    private var policies: [PolicyKey: any WiredMemoryPolicy] = [:]
    private var currentLimit: Int?
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    public init() {}

    public func events() -> AsyncStream<WiredMemoryEvent> {
        #if DEBUG
            return AsyncStream { continuation in
                let id = UUID()
                eventContinuations[id] = continuation
                continuation.onTermination = { _ in
                    Task { await self.removeEventContinuation(id: id) }
                }
            }
        #else
            return AsyncStream { continuation in
                continuation.finish()
            }
        #endif
    }

    public func start(id: UUID, size: Int, policy: any WiredMemoryPolicy) async -> Int {
        let normalizedSize = max(0, size)
        if !WiredMemoryBackend.isSupported {
            if baseline == nil {
                baseline = 0
            }
            return baseline ?? 0
        }

        var baselineValue = ensureBaseline()
        if active[id] != nil {
            emit(
                kind: .ticketStartIgnored,
                ticketID: id,
                size: normalizedSize,
                baseline: baselineValue
            )
            return currentLimit ?? baselineValue
        }

        let key = policyKey(for: policy)
        let label = policyLabel(for: policy, key: key)

        while !policy.canAdmit(
            baseline: baselineValue,
            activeSizes: activeSizes(for: key),
            newSize: normalizedSize
        ) {
            emit(
                kind: .admissionWait,
                ticketID: id,
                size: normalizedSize,
                policy: label,
                baseline: baselineValue
            )
            if Task.isCancelled {
                return currentLimit ?? baselineValue
            }

            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    waiters[id] = continuation
                }
            } onCancel: { [id] in
                Task { await self.cancelWaiter(id: id) }
            }

            baselineValue = ensureBaseline()
            if Task.isCancelled {
                return currentLimit ?? baselineValue
            }
        }

        emit(
            kind: .admissionGranted,
            ticketID: id,
            size: normalizedSize,
            policy: label,
            baseline: baselineValue
        )
        active[id] = TicketState(size: normalizedSize, policyKey: key, policyLabel: label)
        policies[key] = policy
        emit(
            kind: .ticketStarted,
            ticketID: id,
            size: normalizedSize,
            policy: label,
            baseline: baselineValue
        )
        applyCurrentLimit()
        return currentLimit ?? baselineValue
    }

    public func end(id: UUID, policy: any WiredMemoryPolicy) async -> Int {
        if let waiter = waiters.removeValue(forKey: id) {
            waiter.resume()
        }

        guard WiredMemoryBackend.isSupported else {
            if baseline == nil {
                baseline = 0
            }
            return baseline ?? 0
        }

        guard let state = active.removeValue(forKey: id) else {
            emit(kind: .ticketEndIgnored, ticketID: id)
            return currentLimit ?? baseline ?? 0
        }

        emit(
            kind: .ticketEnded,
            ticketID: id,
            size: state.size,
            policy: state.policyLabel,
            baseline: baseline
        )
        if !active.values.contains(where: { $0.policyKey == state.policyKey }) {
            policies.removeValue(forKey: state.policyKey)
        }

        if active.isEmpty {
            let baselineValue = baseline ?? 0
            applyLimitIfNeeded(baselineValue)
            emit(
                kind: .baselineRestored,
                baseline: baselineValue,
                appliedLimit: currentLimit
            )
            baseline = nil
            currentLimit = nil
            resumeWaiters()
            return baselineValue
        }

        applyCurrentLimit()
        resumeWaiters()
        return currentLimit ?? baseline ?? 0
    }

    private func policyKey(for policy: any WiredMemoryPolicy) -> PolicyKey {
        if let hashable = policy as? any Hashable {
            return .hash(AnyHashable(hashable))
        }

        if type(of: policy) is AnyObject.Type, let object = policy as AnyObject? {
            return .object(ObjectIdentifier(object))
        }

        return .unique(UUID())
    }

    private func policyLabel(for policy: any WiredMemoryPolicy, key: PolicyKey) -> String {
        let typeName = String(describing: type(of: policy))
        let suffix: String
        switch key {
        case .hash(let hash):
            suffix = "hash=\(hash.hashValue)"
        case .object(let identifier):
            suffix = "object=\(String(describing: identifier))"
        case .unique(let uuid):
            suffix = "uuid=\(uuid.uuidString)"
        }
        return "\(typeName)#\(suffix)"
    }

    private func activeSizes(for key: PolicyKey) -> [Int] {
        active.values.compactMap { state in
            state.policyKey == key ? state.size : nil
        }
    }

    private func emit(
        kind: WiredMemoryEvent.Kind,
        ticketID: UUID? = nil,
        size: Int? = nil,
        policy: String? = nil,
        baseline: Int? = nil,
        desiredLimit: Int? = nil,
        appliedLimit: Int? = nil,
        activeCount: Int? = nil,
        waiterCount: Int? = nil
    ) {
        #if DEBUG
            guard !eventContinuations.isEmpty else { return }
            eventSequence &+= 1
            let event = WiredMemoryEvent(
                sequence: eventSequence,
                timestamp: Date(),
                kind: kind,
                ticketID: ticketID,
                size: size,
                policy: policy,
                baseline: baseline,
                desiredLimit: desiredLimit,
                appliedLimit: appliedLimit,
                activeCount: activeCount ?? active.count,
                waiterCount: waiterCount ?? waiters.count
            )
            for (_, continuation) in eventContinuations {
                _ = continuation.yield(event)
            }
        #endif
    }

    private func ensureBaseline() -> Int {
        if let baseline {
            return baseline
        }
        let current = WiredMemoryBackend.readCurrentLimit() ?? 0
        baseline = current
        emit(kind: .baselineCaptured, baseline: current, appliedLimit: current)
        return current
    }

    private func desiredLimit() -> Int? {
        guard let baseline else { return nil }
        if active.isEmpty {
            return baseline
        }

        var sizesByPolicy: [PolicyKey: [Int]] = [:]
        for state in active.values {
            sizesByPolicy[state.policyKey, default: []].append(state.size)
        }

        var desired: Int?
        for (key, sizes) in sizesByPolicy {
            guard let policy = policies[key] else { continue }
            let limit = policy.limit(baseline: baseline, activeSizes: sizes)
            desired = max(desired ?? limit, limit)
        }

        return desired ?? baseline
    }

    private func applyCurrentLimit() {
        guard let desired = desiredLimit() else { return }
        emit(
            kind: .limitComputed,
            baseline: baseline,
            desiredLimit: desired
        )
        applyLimitIfNeeded(desired)
    }

    private func applyLimitIfNeeded(_ limit: Int) {
        if currentLimit == limit {
            return
        }

        if WiredMemoryBackend.isSupported {
            if WiredMemoryBackend.applyLimit(limit) {
                currentLimit = limit
                emit(
                    kind: .limitApplied,
                    baseline: baseline,
                    desiredLimit: limit,
                    appliedLimit: currentLimit
                )
            } else {
                emit(
                    kind: .limitApplyFailed,
                    baseline: baseline,
                    desiredLimit: limit,
                    appliedLimit: currentLimit
                )
            }
        } else {
            currentLimit = limit
            emit(
                kind: .limitApplied,
                baseline: baseline,
                desiredLimit: limit,
                appliedLimit: currentLimit
            )
        }
    }

    private func resumeWaiters() {
        guard !waiters.isEmpty else { return }
        let pending = waiters
        waiters.removeAll()
        for (_, continuation) in pending {
            continuation.resume()
        }
    }

    private func cancelWaiter(id: UUID) {
        if let waiter = waiters.removeValue(forKey: id) {
            waiter.resume()
        }
        emit(kind: .admissionCancelled, ticketID: id)
    }

    #if DEBUG
        private func removeEventContinuation(id: UUID) {
            eventContinuations.removeValue(forKey: id)
        }
    #endif
}

extension WiredMemoryPolicy {
    public func ticket(
        size: Int,
        manager: WiredMemoryManager = .shared
    ) -> WiredMemoryTicket {
        WiredMemoryTicket(size: size, policy: self, manager: manager)
    }
}
