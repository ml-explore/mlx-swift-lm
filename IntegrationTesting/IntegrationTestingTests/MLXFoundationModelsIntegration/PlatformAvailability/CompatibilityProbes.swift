// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFoundationModels
import Testing

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Asymmetric, tier-aware compatibility probes.
///
/// Every probe runs identically on all three devices but asserts a
/// *tier-appropriate* outcome (see ``DeviceTier``). A probe that "throws" or
/// "is unavailable" is not a generic pass — each asserts a specific positive
/// fact for its tier and a tripwire if it reaches code that should be
/// unreachable on that tier. The goal is no false greens: if a future change
/// accidentally exposes the FM surface below OS 27, the partial/absent tiers go
/// red here.
@Suite("Platform Compatibility Probes")
struct PlatformCompatibilityProbes {

    /// The unforgeable launch-safety signal.
    ///
    /// Reaching the body of *any* test means the test-runner process loaded and
    /// began executing — i.e. dyld did not fault on a weak-null FoundationModels
    /// conformance record (`MLXLanguageModel: LanguageModel`,
    /// `Executor: LanguageModelExecutor`, `StringResponse: Generable`) during the
    /// `__swift5_proto` scan at image load. On the ABSENT tier (iOS 18.5, FM
    /// framework absent) this is the whole ballgame: if the binary launches, the
    /// `@available` + auto-weak-linking story held.
    @Test("probe suite launches on this tier")
    func binaryLaunches() {
        print("[PlatformCompatibility] DeviceTier.current = \(DeviceTier.current)")
        #expect(Bool(true))
    }

    /// Liveness / anti-false-green. Pure MLX, zero FoundationModels.
    ///
    /// Forces a Metal compute dispatch and reads the scalar back from the GPU.
    /// Must pass on every tier (the package is not FM-only). A no-op submission
    /// would read 0, not 9, so the read-back proves the kernel actually ran.
    @Test("pure-MLX eval works on every tier")
    func rawMLXInferenceWorks() {
        let a = MLXArray([Float(1), Float(2), Float(3)])
        let b = MLXArray([Float(4), Float(5), Float(6)])
        let c = a + b
        eval(c)
        let result: Float = c[2].item()
        #expect(result == 9.0, "MLX scalar add expected 9.0, got \(result)")
    }

    /// The `FoundationModels` framework is present on full + partial, absent below.
    ///
    /// `SystemLanguageModel` shipped in OS 26, so `#available(... 26, *)` is the
    /// runtime proxy for "framework present". Because ``DeviceTier/current`` is
    /// derived from the reported OS version, this assertion also cross-checks the
    /// two against each other.
    @Test("FM framework presence matches tier")
    func fmFrameworkPresenceMatchesTier() {
        var fmPresent = false
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) { fmPresent = true }
        let expected = (DeviceTier.current != .absent)
        #expect(
            fmPresent == expected,
            "FM-26 availability (\(fmPresent)) should match (tier != absent)=\(expected) for \(DeviceTier.current)"
        )
    }

    /// The `LanguageModel` protocol surface (OS 27) is reachable only on full.
    ///
    /// On partial/absent the `#available(... 27, *)` block is skipped entirely,
    /// so the conformance surface is never touched — which is exactly the
    /// graceful-degradation contract.
    @Test("LanguageModel protocol availability matches tier")
    func languageModelProtocolMatchesTier() {
        var lmAvailable = false
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            lmAvailable = true
            #if canImport(FoundationModels, _version: 2)
                // Touch the OS-27 surface to prove it is genuinely reachable here.
                _ = LanguageModelCapabilities(capabilities: [])
                _ = (any LanguageModel).self
            #endif
        }
        let expected = (DeviceTier.current == .full)
        #expect(
            lmAvailable == expected,
            "LanguageModel(27) availability (\(lmAvailable)) should match (tier == full)=\(expected) for \(DeviceTier.current)"
        )
    }

    /// Our own `MLXLanguageModel` adapter type is gated to the full tier.
    @Test("MLXLanguageModel type is gated to the full tier")
    func mlxLanguageModelGatedCorrectly() {
        var typeReachable = false
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            #if canImport(FoundationModels, _version: 2)
                _ = MLXLanguageModel.self
                typeReachable = true
            #endif
        }
        #expect(
            typeReachable == (DeviceTier.current == .full),
            "MLXLanguageModel reachability (\(typeReachable)) should match (tier == full) for \(DeviceTier.current)"
        )
    }

    /// `#available` must agree with the reported OS version.
    ///
    /// Pre-release OS builds can decouple marketing version from feature-set
    /// version; if `#available(27)` and `operatingSystemVersion.major >= 27`
    /// disagree, the build's availability metadata is skewed and every other
    /// probe's verdict is suspect — so the disagreement is itself a failure.
    @Test("#available agrees with reported OS version")
    func availabilityAgreesWithOSVersion() {
        let major = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        var avail27 = false
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) { avail27 = true }
        #expect(
            avail27 == (major >= 27),
            "#available(27)=\(avail27) disagrees with OS major \(major) — pre-release version skew"
        )
    }
}
