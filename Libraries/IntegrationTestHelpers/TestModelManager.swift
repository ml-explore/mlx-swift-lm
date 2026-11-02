// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Keeps at most one model in memory across all integration tests.
///
/// Some integration models are tens of gigabytes, so two of them can't be
/// resident at once. Load every model through ``load(_:onUnload:_:)``. A request
/// for the resident key returns the cached value, so tests that use the same
/// model share it, even across suites. A request for any other key unloads the
/// resident model first.
///
/// A test that needs several models at once, such as a target and its drafter,
/// loads them together under one key.
///
/// ```swift
/// let container = try await TestModelManager.shared.load(
///     TestModelManager.key("llm", configuration)
/// ) {
///     try await LLMModelFactory.shared.loadContainer(
///         from: downloader, using: tokenizerLoader, configuration: configuration)
/// }
/// ```
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
        let box: Box
        let onUnload: (@Sendable () async -> Void)?
    }

    private var resident: Resident?

    // A FIFO lock so a load and the unload before it run as one step.
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() {}

    /// Returns the value cached under `key`, or unloads the resident model and
    /// calls `loader`.
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
        let box = try await loadBox("\(key) [\(T.self)]", onUnload: onUnload) {
            Box(try await loader())
        }
        return box.value as! T
    }

    /// Unloads the resident model, if any.
    ///
    /// Call this before loading a model outside the manager, for example when a
    /// test measures memory across fresh loads.
    public func unload() async {
        await lock()
        defer { unlock() }
        await unloadResident()
    }

    /// A cache key for `configuration`: `kind`, the model name, and the revision.
    public static func key(_ kind: String, _ configuration: ModelConfiguration) -> String {
        switch configuration.id {
        case .id(let id, let revision): "\(kind):\(id)@\(revision)"
        case .directory(let url): "\(kind):\(url.path)"
        }
    }

    private func loadBox(
        _ key: String,
        onUnload: (@Sendable () async -> Void)?,
        _ loader: @Sendable () async throws -> Box
    ) async throws -> Box {
        await lock()
        defer { unlock() }

        if resident?.key == key, let box = resident?.box {
            return box
        }
        await unloadResident()

        // Also frees memory left by loads made outside the manager.
        Stream.gpu.synchronize()
        Memory.clearCache()

        print("TestModelManager: loading \(key)")
        let box = try await loader()
        resident = Resident(key: key, box: box, onUnload: onUnload)
        return box
    }

    private func unloadResident() async {
        guard let key = resident?.key else { return }
        print("TestModelManager: unloading \(key)")
        let onUnload = resident?.onUnload
        // Drop the only reference here so ARC frees the weights before clearCache().
        resident = nil
        await onUnload?()
        Stream.gpu.synchronize()
        Memory.clearCache()
    }

    private func lock() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    private func unlock() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
