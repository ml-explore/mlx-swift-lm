// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

private let log = MLXLogger(label: "TestModelManager")

/// Keeps at most one model in memory across all integration tests.
///
/// Some integration models are tens of gigabytes, so two of them can't be
/// resident at once. Load every model through ``load(_:_:onUnload:_:)`` or
/// ``load(_:onUnload:_:)``. A request
/// for the resident key returns the cached value, so tests that use the same
/// model share it, even across suites. A request for any other key unloads the
/// resident model first.
///
/// A test that needs several models at once, such as a target and its drafter,
/// loads them together under one key.
///
/// ```swift
/// let container = try await TestModelManager.shared.load("llm", configuration) {
///     try await LLMModelFactory.shared.loadContainer(
///         from: downloader, using: tokenizerLoader, configuration: configuration)
/// }
/// ```
///
/// Note: this is meant to be used by integration tests in `mlx-swift-lm`.  Those
/// tests are required to run serialized under `ModelLoadingSuite`.
public actor TestModelManager {

    public static let shared = TestModelManager()

    /// Holds a loaded value of any type. The manager hands it out only to
    /// serialized tests, so the value is never used from two places at once.
    private final class Box: @unchecked Sendable {
        let value: Any
        init(_ value: Any) { self.value = value }
    }

    private struct Resident {
        let key: String
        let configuration: ModelConfiguration?
        let box: Box
        let onUnload: (@Sendable () async -> Void)?
    }

    private var resident: Resident?

    // A FIFO lock so a load and the unload before it run as one step.
    private var busy = false
    private var waiters: [(id: Int, continuation: CheckedContinuation<Void, any Error>)] = []
    private var nextWaiterID = 0

    private init() {}

    /// Returns the value cached for `configuration`, or unloads the resident model
    /// and calls `loader`.
    ///
    /// A cached value is reused only if its configuration is `==` to
    /// `configuration`, so overrides such as `toolCallFormat` force a fresh load.
    ///
    /// - Parameters:
    ///   - kind: What `loader` returns, such as `"llm"` or `"vlm-context"`.
    ///   - configuration: The configuration `loader` loads.
    ///   - onUnload: Called when this value is unloaded.
    ///   - loader: Loads the value. It must not call back into the manager.
    public nonisolated func load<T>(
        _ kind: String,
        _ configuration: ModelConfiguration,
        onUnload: (@Sendable () async -> Void)? = nil,
        _ loader: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        let key = "\(Self.baseKey(kind, configuration)) [\(T.self)]"
        let box = try await loadBox(key, configuration: configuration, onUnload: onUnload) {
            Box(try await loader())
        }
        return box.value as! T
    }

    /// Returns the value cached under `key`, or unloads the resident model and
    /// calls `loader`.
    ///
    /// Prefer ``load(_:_:onUnload:_:)`` when the value comes from one configuration.
    ///
    /// - Parameters:
    ///   - key: Identifies the model. Include the revision and anything else
    ///     that changes what `loader` returns. See ``key(_:_:)``.
    ///   - onUnload: Called when this value is unloaded. Use it to drop other
    ///     caches that hold the model, such as the `MLXLanguageModel` cache.
    ///   - loader: Loads the value. It must not call back into the manager.
    public nonisolated func load<T>(
        _ key: String,
        onUnload: (@Sendable () async -> Void)? = nil,
        _ loader: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        // Including the type keeps two loaders that share a key from colliding.
        let box = try await loadBox("\(key) [\(T.self)]", configuration: nil, onUnload: onUnload) {
            Box(try await loader())
        }
        return box.value as! T
    }

    /// Unloads the resident model, if any.
    ///
    /// Call this before loading a model outside the manager, for example when a
    /// test measures memory across fresh loads.
    public func unload() async {
        // Only fails if the task is cancelled while waiting; skip the unload then.
        guard (try? await lock()) != nil else { return }
        defer { unlock() }
        await unloadResident()
    }

    /// A cache key for a plain configuration: `kind`, the model name, and the
    /// revision. Use it to build keys for values loaded from several models.
    ///
    /// The key ignores overrides such as `toolCallFormat`, so `configuration` must
    /// not have any. Use ``load(_:_:onUnload:_:)`` for configurations with overrides.
    public static func key(_ kind: String, _ configuration: ModelConfiguration) -> String {
        let plain =
            switch configuration.id {
            case .id(let id, let revision): ModelConfiguration(id: id, revision: revision)
            case .directory(let url): ModelConfiguration(directory: url)
            }
        precondition(
            configuration == plain,
            "TestModelManager.key ignores overrides in \(configuration.name); use load(_:_:)")
        return baseKey(kind, configuration)
    }

    private static func baseKey(_ kind: String, _ configuration: ModelConfiguration) -> String {
        switch configuration.id {
        case .id(let id, let revision): "\(kind):\(id)@\(revision)"
        case .directory(let url): "\(kind):\(url.path)"
        }
    }

    private func logMemory(_ message: String) {
        let snapshot = Memory.snapshot()
        func scale(_ value: Int) -> String {
            "\(value / 1024 / 1024)M"
        }
        log.info(
            "\(message): limit \(scale(Memory.memoryLimit)) active \(scale(snapshot.activeMemory)) peak \(scale(snapshot.peakMemory))"
        )
    }

    private func loadBox(
        _ key: String,
        configuration: ModelConfiguration?,
        onUnload: (@Sendable () async -> Void)?,
        _ loader: @Sendable () async throws -> Box
    ) async throws -> Box {
        try await lock()
        defer { unlock() }
        // Don't start a load, possibly a large one, for a test that is gone.
        try Task.checkCancellation()

        if let resident, resident.key == key, resident.configuration == configuration {
            return resident.box
        }
        await unloadResident()

        // Also frees memory left by loads made outside the manager.
        Stream.gpu.synchronize()
        Memory.clearCache()

        log.info("loading \(key)")
        let box = try await loader()
        logMemory("loaded \(key)")
        resident = Resident(
            key: key, configuration: configuration, box: box, onUnload: onUnload)
        return box
    }

    private func unloadResident() async {
        guard let key = resident?.key else { return }
        logMemory("end \(key)")
        log.info("unloading \(key)")
        let onUnload = resident?.onUnload
        // Drop the only reference here so ARC frees the weights before clearCache().
        resident = nil
        await onUnload?()
        Stream.gpu.synchronize()
        Memory.clearCache()
        logMemory("unloaded \(key)")
        Memory.peakMemory = 0
    }

    /// Waits for the lock. Throws `CancellationError` if the task is cancelled
    /// while waiting, so one stuck load doesn't hold every later test.
    private func lock() async throws {
        guard busy else {
            busy = true
            return
        }
        let id = nextWaiterID
        nextWaiterID += 1
        try await withTaskCancellationHandler {
            // The explicit type avoids a compiler crash inferring `Void` here.
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    waiters.append((id, continuation))
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: Int) {
        // Absent if `unlock()` already handed this waiter the lock.
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }

    private func unlock() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }
}
