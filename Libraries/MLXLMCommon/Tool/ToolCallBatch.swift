// Copyright © 2026 Apple Inc.

/// The tool calls of one model step, run under ``ToolDispatchOptions``.
///
/// Collecting the results consumes the batch. Dropping it cancels the calls
/// still running, so an abandoned step never leaves tools behind.
struct ToolCallBatch: ~Copyable {
    typealias Dispatch = @Sendable (ToolCall) async throws -> String

    private let options: ToolDispatchOptions
    private let dispatch: Dispatch
    private(set) var calls: [ToolCall] = []
    private var runs: [Task<String, Error>] = []
    private var isCancelled = false

    init(options: ToolDispatchOptions, dispatch: @escaping Dispatch) {
        self.options = options
        self.dispatch = dispatch
    }

    var isEmpty: Bool { calls.isEmpty }

    /// Adds a call the model finished writing.
    mutating func append(_ call: ToolCall) {
        calls.append(call)
        if options.contains(.whenWritten) { startPendingCalls() }
    }

    /// Cancels the running calls and starts no more.
    mutating func cancel() {
        isCancelled = true
        for run in runs { run.cancel() }
    }

    /// Runs the calls not yet started and delivers each result in call order.
    ///
    /// The first call in call order that throws ends the batch. Results before it
    /// are delivered, and the calls after it are cancelled if still running. A
    /// cancelled batch throws `CancellationError`.
    consuming func results(_ deliver: (Chat.Message) async throws -> Void) async throws {
        if isCancelled { throw CancellationError() }
        startPendingCalls()
        let calls = calls
        let runs = runs
        defer { for run in runs { run.cancel() } }
        try await withTaskCancellationHandler {
            for (call, run) in zip(calls, runs) {
                let output = try await run.value
                try await deliver(.tool(output, id: call.id, name: call.function.name))
            }
        } onCancel: {
            for run in runs { run.cancel() }
        }
    }

    private mutating func startPendingCalls() {
        guard !isCancelled else { return }
        for call in calls.dropFirst(runs.count) {
            let previous = options.contains(.concurrent) ? nil : runs.last
            runs.append(
                Task { [dispatch] in
                    _ = try await previous?.value
                    try Task.checkCancellation()
                    return try await dispatch(call)
                })
        }
    }

    deinit {
        for run in runs { run.cancel() }
    }
}
