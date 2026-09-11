// Copyright © 2026 Apple Inc.

import Foundation

/// The outcome of steering instructions accepted by a chat session.
public enum SteeringEvent: Sendable, Equatable {
    /// Instructions included in the next model step's prefill, in acceptance order.
    case applied([UUID])

    /// Instructions that could not be applied. The response continues normally.
    case failed(SteeringFailure)
}

/// When a steering instruction may enter the model's context.
public enum SteeringPolicy: Sendable {
    /// End decoding at the first supported text boundary. Prefill, tools,
    /// reasoning models, and framed protocols finish their step.
    case nextSafeBoundary
    /// Allow the current model step and dispatched tools to finish.
    case nextStepBoundary
}

/// Limits on steering input pending for one response.
///
/// Exceeding either limit throws ``SteeringError/queueFull``. The pending batch
/// is cleared when a model step applies it, so a response can accept more
/// instructions after each ``SteeringEvent/applied(_:)``.
public enum SteeringLimits: Sendable {
    /// Instructions that may be pending at once.
    public static let maxPendingInstructions = 32
    /// UTF-8 bytes that may be pending at once.
    public static let maxPendingBytes = 64 * 1024
}

/// Why a steering instruction was not accepted or not applied.
///
/// ``ChatSession/steer(_:policy:response:)`` throws the acceptance failures. Application
/// failures arrive as ``SteeringEvent/failed(_:)`` and never end the
/// response that was running.
public enum SteeringError: LocalizedError, Sendable, Equatable {
    /// Instructions must contain non-whitespace text.
    case emptyInstruction
    /// The pending instruction count or UTF-8 byte limit would be exceeded.
    case queueFull
    /// The session has no response accepting instructions.
    case noActiveResponse
    /// The response ended before the instruction could be applied.
    case responseEnded
    /// This response cannot be steered. A session restored from a raw cache has
    /// no transcript to continue from.
    case notSteerable
    /// The model step produced no assistant output to continue from: it was
    /// empty, cancelled, or produced only a rejected tool call.
    case emptyResponse
    /// Supply `toolDispatch` to continue tool calls within a response.
    case toolResultsRequired

    public var errorDescription: String? {
        switch self {
        case .emptyInstruction: "A steering instruction must contain text."
        case .queueFull: "The pending steering queue is full."
        case .noActiveResponse: "This session has no response accepting steering."
        case .responseEnded: "The response ended before this instruction was applied."
        case .notSteerable: "Steering requires a structured conversation history."
        case .emptyResponse: "The model produced no assistant output to continue from."
        case .toolResultsRequired:
            "Configure toolDispatch to continue tool calls within this response."
        }
    }
}

/// A steering instruction that was accepted but not applied.
///
/// Carries the original text so a client can resubmit it. Delivered through
/// ``SteeringEvent/failed(_:)``; the response itself continues.
public struct SteeringFailure: Sendable, Equatable {
    /// Acceptance IDs returned by ``ChatSession/steer(_:policy:response:)``.
    public let ids: [UUID]
    /// The instruction text, in acceptance order.
    public let instructions: [String]
    /// Why the instructions were not applied.
    public let reason: SteeringError

    package init(ids: [UUID], instructions: [String], reason: SteeringError) {
        self.ids = ids
        self.instructions = instructions
        self.reason = reason
    }
}

/// All mutable state is protected by `lock`; no MLX arrays cross this mailbox.
/// Only the session runner drains it. Task cancellation and joining happen outside the lock.
final class SteeringControl: @unchecked Sendable {
    struct Request: Sendable {
        let id: UUID
        let text: String
    }

    /// Stable identity for the response this mailbox belongs to.
    let responseID = ChatSession.ResponseID(rawValue: UUID())

    private let lock = NSLock()
    private var pending: [Request] = []
    private var pendingBytes = 0
    private var early = false
    private var closed = false
    private var cancelled = false
    private var finished = false
    private var steerable = true
    private var worker: Task<Void, Never>?

    /// Accepts an instruction, or throws the reason it cannot be accepted.
    ///
    /// The caller has already rejected whitespace-only text so that an invalid
    /// instruction fails the same way whatever the session is doing.
    func enqueue(_ text: String, policy: SteeringPolicy) throws -> UUID {
        let bytes = text.utf8.count
        return try lock.withLock {
            guard !closed else { throw SteeringError.responseEnded }
            guard steerable else { throw SteeringError.notSteerable }
            guard pending.count < SteeringLimits.maxPendingInstructions,
                bytes <= SteeringLimits.maxPendingBytes - pendingBytes
            else {
                throw SteeringError.queueFull
            }
            let id = UUID()
            pending.append(Request(id: id, text: text))
            pendingBytes += bytes
            if case .nextSafeBoundary = policy { early = true }
            return id
        }
    }

    var requestsEarlyBoundary: Bool {
        lock.withLock { early && !closed }
    }

    /// Whether this response can still accept instructions.
    var isOpen: Bool {
        lock.withLock { !closed && steerable }
    }

    /// Whether this response has completed, failed, or been cancelled.
    var isClosed: Bool {
        lock.withLock { closed }
    }

    /// Records whether this response has the structured history steering needs.
    /// Returns instructions accepted before the answer was known so the runner
    /// can report them as failed.
    func setSteerable(_ value: Bool) -> [Request] {
        lock.withLock {
            steerable = value
            guard !value else { return [] }
            let orphaned = pending
            pending.removeAll(keepingCapacity: true)
            pendingBytes = 0
            early = false
            return orphaned
        }
    }

    /// Atomically resolve the last-input/completion race. A producer either
    /// joins the next step or observes a closed response.
    func take(closeIfEmpty: Bool) throws -> [Request] {
        try lock.withLock {
            if cancelled { throw CancellationError() }
            let batch = pending
            pending.removeAll(keepingCapacity: true)
            pendingBytes = 0
            early = false
            if closeIfEmpty && batch.isEmpty { closed = true }
            return batch
        }
    }

    /// Close acceptance and return remaining instructions in one operation.
    func closeAndTake() throws -> [Request] {
        try lock.withLock {
            if cancelled { throw CancellationError() }
            closed = true
            let batch = pending
            pending.removeAll()
            pendingBytes = 0
            early = false
            return batch
        }
    }

    func setTask(_ task: Task<Void, Never>) {
        let shouldCancel = lock.withLock {
            if !finished { worker = task }
            return cancelled
        }
        if shouldCancel { task.cancel() }
    }

    func cancel() {
        let task = lock.withLock {
            guard !closed else { return nil as Task<Void, Never>? }
            cancelled = true
            closed = true
            pending.removeAll()
            pendingBytes = 0
            early = false
            return worker
        }
        task?.cancel()
    }

    func synchronize() async {
        let task = lock.withLock { worker }
        await task?.value
    }

    func finish() {
        lock.withLock {
            closed = true
            pending.removeAll()
            pendingBytes = 0
            early = false
            finished = true
            worker = nil
        }
    }
}

/// Tracks requests without acquiring the cache lock held during inference.
/// All mutable state is protected by `lock`. Never calls into `SteeringControl`
/// while holding `lock`, so the two locks cannot deadlock.
final class SessionSteering: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [SteeringControl] = []
    private var active: SteeringControl?

    func register(_ control: SteeringControl) {
        lock.withLock { requests.append(control) }
    }

    func start(_ control: SteeringControl) {
        lock.withLock { active = control }
    }

    func remove(_ control: SteeringControl) {
        lock.withLock {
            requests.removeAll { $0 === control }
            if active === control { active = nil }
        }
    }

    /// The most recently created response, whatever its state.
    var latestResponse: ChatSession.ResponseID? {
        lock.withLock { requests.last?.responseID }
    }

    /// Answers the same question `steer` does: whether the response it would
    /// select accepts instructions.
    func canSteer(_ response: ChatSession.ResponseID?) -> Bool {
        candidates(for: response).first { !$0.isClosed }?.isOpen ?? false
    }

    func steer(_ text: String, policy: SteeringPolicy, response: ChatSession.ResponseID?) throws
        -> UUID
    {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SteeringError.emptyInstruction
        }
        var sawEndedResponse = false
        for control in candidates(for: response) {
            do {
                return try control.enqueue(text, policy: policy)
            } catch SteeringError.responseEnded {
                // A cancelled or completed response is removed by its runner, which
                // can lag the event that closed it. Skip it rather than reject input
                // the session can still deliver.
                sawEndedResponse = true
            }
        }
        throw sawEndedResponse ? SteeringError.responseEnded : SteeringError.noActiveResponse
    }

    func synchronize() async {
        let snapshot = lock.withLock { requests }
        for control in snapshot { await control.synchronize() }
    }

    /// One named response, or the cache owner followed by registered responses
    /// in call order.
    private func candidates(for response: ChatSession.ResponseID?) -> [SteeringControl] {
        lock.withLock {
            if let response {
                return requests.filter { $0.responseID == response }
            }
            guard let active else { return requests }
            return [active] + requests.filter { $0 !== active }
        }
    }
}
