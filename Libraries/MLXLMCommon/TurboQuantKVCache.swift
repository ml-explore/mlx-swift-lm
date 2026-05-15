// Copyright © 2026 Schtack.

import Foundation
import MLX

public typealias TurboQuantPreset = MLX.TurboQuantPreset
public typealias TurboQuantBackend = MLX.TurboQuantBackend
public typealias TurboQuantKernelAvailability = MLX.TurboQuantKernelAvailability

public enum KVCacheStrategy: String, Codable, Sendable, CaseIterable {
    case none
    case mlxAffine
    case turboQuant
}

public struct TurboQuantKVCacheDiagnostics: Equatable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var requestedBackend: TurboQuantBackend
    public var activeBackend: TurboQuantBackend
    public var fallbackReason: String?
    public var metalCodecAvailable: Bool
    public var groupSize: Int
    public var bits: Int
    public var maxSize: Int?
}

public final class TurboQuantKVCache: QuantizedKVCache {
    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) {
        self.preset = preset
        self.requestedBackend = backend
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        super.init(groupSize: groupSize, bits: preset.effectiveBits, mode: mode)
    }

    public override var metaState: [String] {
        get { super.metaState + [preset.rawValue, requestedBackend.rawValue] }
        set {
            super.metaState = Array(newValue.prefix(4))
        }
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec,
            groupSize: groupSize,
            bits: bits,
            maxSize: nil
        )
    }

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend
        )
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }
}

public final class RotatingTurboQuantKVCache: BaseKVCache, QuantizedKVCacheProtocol,
    CustomDebugStringConvertible
{
    private let rawCache: RotatingKVCache
    private var packedKeys: TurboQuantPackedTensor?
    private var packedValues: TurboQuantPackedTensor?

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public override var maxSize: Int? { rawCache.maxSize }
    public override var isTrimmable: Bool { rawCache.isTrimmable }
    public var ropeOffset: RoPEOffset { rawCache.ropeOffset }

    public init(
        maxSize: Int,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) {
        self.rawCache = RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
        self.preset = preset
        self.requestedBackend = backend
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        self.groupSize = groupSize
        self.bits = preset.effectiveBits
        self.mode = mode
        super.init()
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let (cachedKeys, cachedValues) = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset, role: .key, groupSize: groupSize, mode: mode, backend: activeBackend)
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset, role: .value, groupSize: groupSize, mode: mode, backend: activeBackend)
        packedKeys = turboQuantized(cachedKeys, configuration: keyConfiguration)
        packedValues = turboQuantized(cachedValues, configuration: valueConfiguration)

        return (
            (packedKeys!.weight, packedKeys!.scales, packedKeys!.biases),
            (packedValues!.weight, packedValues!.scales, packedValues!.biases)
        )
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let packedKeys, let packedValues else { return nil }
        return (
            (packedKeys.weight, packedKeys.scales, packedKeys.biases),
            (packedValues.weight, packedValues.scales, packedValues.biases)
        )
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        return result
    }

    public override var state: [MLXArray] {
        get { rawCache.state }
        set {
            rawCache.state = newValue
            offset = rawCache.offset
            packedKeys = nil
            packedValues = nil
        }
    }

    public override var metaState: [String] {
        get { rawCache.metaState + [preset.rawValue, String(groupSize), requestedBackend.rawValue] }
        set {
            rawCache.metaState = Array(newValue.prefix(5))
            offset = rawCache.offset
        }
    }

    public override func innerState() -> [MLXArray] {
        rawCache.innerState()
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        rawCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = rawCache.trim(n)
        offset = rawCache.offset
        packedKeys = nil
        packedValues = nil
        return trimmed
    }

    public override func copy() -> any KVCache {
        guard let maxSize else {
            fatalError("RotatingTurboQuantKVCache requires maxSize")
        }
        let new = RotatingTurboQuantKVCache(
            maxSize: maxSize,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec,
            groupSize: groupSize,
            bits: bits,
            maxSize: maxSize
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxSize?.description ?? "-"), preset: \(preset.rawValue), backend: \(activeBackend.rawValue)"
    }
}

public extension KVCacheSimple {
    func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) -> TurboQuantKVCache {
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend
        )
        cache.offset = self.offset

        let currentState = self.state
        if currentState.count == 2 {
            let keyConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend
            )
            let valueConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .value,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend
            )
            let keys = turboQuantized(currentState[0], configuration: keyConfiguration)
            let values = turboQuantized(currentState[1], configuration: valueConfiguration)
            cache.state = [
                keys.weight, keys.scales, keys.biases,
                values.weight, values.scales, values.biases,
            ].compactMap { $0 }
        }

        return cache
    }
}
