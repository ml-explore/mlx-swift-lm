// Copyright © 2025 Apple Inc.

actor AsyncMutex {
    private var isLocked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func lock() async {
        if !isLocked {
            isLocked = true
            return
        }

        await withCheckedContinuation { cont in
            waiters.append(cont)
        }
    }

    func unlock() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()
        } else {
            isLocked = false
        }
    }

    func withLock<T>(_ body: () async throws -> sending T) async rethrows -> sending T {
        await lock()
        defer { unlock() }
        return try await body()
    }
}

final public class SerialAccessContainer<T>: @unchecked Sendable {

    private var value: T
    private let lock = AsyncMutex()

    public init(_ value: consuming T) {
        self.value = consume value
    }

    public func read<R>(_ body: @Sendable (T) async throws -> sending R) async rethrows -> sending R
    {
        try await lock.withLock {
            try await body(value)
        }
    }

    public func update<R>(_ body: @Sendable (inout T) async throws -> sending R) async rethrows
        -> sending R
    {
        try await lock.withLock {
            try await body(&value)
        }
    }

}

final class SendableBox<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: consuming T) {
        self.value = consume value
    }

    consuming func consume() -> T {
        guard let value else {
            fatalError("value already consumed")
        }
        self.value = nil
        return value
    }
}
