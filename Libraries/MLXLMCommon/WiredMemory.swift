// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Control wired memory behavior for generation.
///
/// This is opt-in and no-ops on unsupported platforms or non-GPU devices.
public enum WiredMemoryLimit: Sendable, Equatable {
    /// Leave the wired memory limit unchanged.
    case `default`
    /// Set a fixed wired limit in bytes for the duration of a block.
    case fixed(bytes: Int)
    /// Use the device recommended maximum working set size.
    case max

    /// Resolve the requested limit to bytes, or nil if no change should be applied.
    public func resolvedLimitBytes() -> Int? {
        switch self {
        case .default:
            return nil
        case .fixed(let bytes):
            return bytes > 0 ? bytes : nil
        case .max:
            let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
            guard maxBytes > 0 else { return nil }
            if maxBytes > UInt64(Int.max) {
                return Int.max
            }
            return Int(maxBytes)
        }
    }

    /// Execute a block while applying the requested wired memory limit, if supported.
    ///
    /// - Note: This affects a global MLX setting, so concurrent callers should avoid
    ///   changing the limit at the same time.
    public func withWiredMemory<R>(_ body: () async throws -> R) async rethrows -> R {
        guard Device.defaultDevice().deviceType == .gpu else {
            return try await body()
        }

        guard let limit = resolvedLimitBytes() else {
            return try await body()
        }

        #if os(macOS) || os(iOS) || os(tvOS)
            if #available(macOS 15, iOS 18, tvOS 18, *) {
                return try await Memory.withWiredLimit(limit, body)
            } else {
                return try await body()
            }
        #else
            return try await body()
        #endif
    }
}
