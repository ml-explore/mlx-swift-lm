// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Default-on rollback switch for Qwen 3.5/3.6 fused quantized projections.
package let qwen35FourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw != "0" && raw != "false" && raw != "off"
}()

/// A fused quantized projection and checkpoint-shaped views into its storage.
///
/// The views let a model keep its public/checkpoint module topology without
/// retaining a second physical copy of the quantized weights.
package struct FusedQuantizedLinearProjection {
    package let fused: QuantizedLinear
    package let sourceViews: [QuantizedLinear]
}

/// A fused projection could not replace its source modules atomically.
///
/// `rollbackError` is non-nil only when restoring the original source modules
/// also failed; callers can then surface both failures while retaining the
/// conservative unfused execution path.
package struct FusedQuantizedLinearPreparationError: Error, CustomStringConvertible {
    package let installationError: any Error
    package let rollbackError: (any Error)?

    package var description: String {
        if let rollbackError {
            return "unable to install fused projection views (\(installationError)); "
                + "restoring the original projections also failed (\(rollbackError))"
        }
        return "unable to install fused projection views (\(installationError)); "
            + "the original projections were restored"
    }
}

/// Lifecycle state for a physical projection derived from several registered
/// quantized linears.
///
/// Inference reads ``fused`` without locking after ``prepare(enabled:linears:installSourceModules:)``
/// finishes during model setup. Trace registration is synchronized because segment
/// and per-layer traces compile lazily under separate locks.
package final class FusedQuantizedLinearProjectionCache {
    private enum State {
        case unprepared
        case preparing
        case ready
        case ineligible
    }

    private struct TraceEntry {
        weak var value: (any CompiledTraceInvalidating)?
    }

    private var state = State.unprepared
    package private(set) var fused: QuantizedLinear?
    private let tracesLock = NSLock()
    private var traces: [ObjectIdentifier: TraceEntry] = [:]

    package init() {}

    package var isPrepared: Bool {
        state == .ready && fused != nil
    }

    /// The fused projection as compile state for traces that read it.
    package var traceState: [Module] {
        fused.map { [$0] } ?? []
    }

    package func registerTrace(_ trace: any CompiledTraceInvalidating) {
        tracesLock.withLock {
            traces[ObjectIdentifier(trace)] = TraceEntry(value: trace)
        }
    }

    package func traceState(registering trace: any CompiledTraceInvalidating) -> [Module] {
        registerTrace(trace)
        return traceState
    }

    /// Drop derived state and dependent traces after any source update.
    /// Invalidations triggered while installing storage-sharing views in ``prepare``
    /// are ignored.
    package func invalidate() {
        guard state != .preparing else { return }
        fused = nil
        state = .unprepared
        let registered = tracesLock.withLock {
            let current = traces.values.compactMap(\.value)
            traces.removeAll()
            return current
        }
        // Trace setup acquires its own lock first, so invalidate after unlocking.
        for trace in registered {
            trace.invalidate()
        }
    }

    package func invalidateIfReplacing(parameters: ModuleParameters, keys: Set<String>) {
        let replacesSource = parameters.flattened().contains { key, _ in
            guard
                let root = key.split(
                    separator: ".", maxSplits: 1, omittingEmptySubsequences: false
                ).first
            else {
                return false
            }
            return keys.contains(String(root))
        }
        if replacesSource {
            invalidate()
        }
    }

    package func invalidateIfReplacing(moduleKey: String, keys: Set<String>) {
        if keys.contains(moduleKey) {
            invalidate()
        }
    }

    /// Build and publish the fused projection once for the current source
    /// topology. A failed or ineligible attempt is terminal until a subsequent
    /// source update calls ``invalidate()``.
    @discardableResult
    package func prepare(
        enabled: Bool,
        linears: [Linear],
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> Bool {
        guard enabled else { return false }

        switch state {
        case .ready:
            return fused != nil
        case .preparing, .ineligible:
            return false
        case .unprepared:
            break
        }

        state = .preparing
        guard let projection = fuseQuantizedLinearProjections(linears),
            projection.sourceViews.count == linears.count
        else {
            state = .ineligible
            return false
        }

        do {
            try installSourceModules(projection.sourceViews)
        } catch let installationError {
            let rollbackError: (any Error)?
            do {
                try installSourceModules(linears)
                rollbackError = nil
            } catch let error {
                rollbackError = error
            }
            fused = nil
            state = .ineligible
            throw FusedQuantizedLinearPreparationError(
                installationError: installationError,
                rollbackError: rollbackError)
        }

        fused = projection.fused
        state = .ready
        return true
    }
}

/// Coalesce compatible quantized linears along their output dimension.
///
/// This is intentionally stricter than merely casting to `QuantizedLinear`:
/// custom subclasses may add behavior, adapters, or state that cannot be
/// represented by one stock quantized matmul. Incompatible inputs return
/// `nil` and callers keep their original projections.
package func fuseQuantizedLinearProjections(
    _ linears: [Linear]
) -> FusedQuantizedLinearProjection? {
    guard linears.count > 1 else { return nil }

    let projections = linears.compactMap { $0 as? QuantizedLinear }
    guard projections.count == linears.count,
        zip(linears, projections).allSatisfy({ linear, projection in
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                && linear === projection
        }),
        let first = projections.first,
        first.bias == nil,
        first.globalScale == nil,
        first.weight.ndim == 2,
        first.weight.dim(0) == first.shape.0,
        first.scales.ndim == 2,
        first.scales.dim(0) == first.shape.0,
        first.biases == nil || first.biases?.shape == first.scales.shape
    else {
        return nil
    }

    let hasQuantizationBiases = first.biases != nil
    guard
        projections.allSatisfy({ projection in
            projection.bias == nil
                && projection.globalScale == nil
                && projection.bits == first.bits
                && projection.groupSize == first.groupSize
                && projection.mode == first.mode
                && projection.shape.1 == first.shape.1
                && projection.weight.ndim == 2
                && projection.weight.dim(0) == projection.shape.0
                && projection.weight.dim(1) == first.weight.dim(1)
                && projection.weight.dtype == first.weight.dtype
                && projection.scales.ndim == 2
                && projection.scales.dim(0) == projection.shape.0
                && projection.scales.dim(1) == first.scales.dim(1)
                && projection.scales.dtype == first.scales.dtype
                && (projection.biases != nil) == hasQuantizationBiases
                && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                && projection.biases?.dtype == first.biases?.dtype
        })
    else {
        return nil
    }

    let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
    let fusedScales = concatenated(projections.map(\.scales), axis: 0)
    let fusedBiases =
        hasQuantizationBiases
        ? concatenated(projections.compactMap(\.biases), axis: 0)
        : nil

    // Realize the concatenations once. The fused projection and the named
    // source views below then share this materialized backing storage.
    eval(fusedWeight, fusedScales)
    if let fusedBiases {
        eval(fusedBiases)
    }

    let fused = QuantizedLinear(
        weight: fusedWeight,
        bias: nil,
        scales: fusedScales,
        biases: fusedBiases,
        groupSize: first.groupSize,
        bits: first.bits,
        mode: first.mode)
    fused.freeze()

    var start = 0
    let sourceViews = projections.map { projection in
        let end = start + projection.shape.0
        defer { start = end }

        let rows = start ..< end
        let view = QuantizedLinear(
            weight: fusedWeight[rows],
            bias: nil,
            scales: fusedScales[rows],
            biases: fusedBiases.map { $0[rows] },
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode)
        view.freeze()
        return view
    }

    return FusedQuantizedLinearProjection(fused: fused, sourceViews: sourceViews)
}
