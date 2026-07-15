// Copyright © 2026 Apple Inc.

import Foundation

/// Which FoundationModels capability tier the current OS provides at runtime.
///
/// This package ships a single binary that must run across three OS tiers with
/// graceful degradation of FoundationModels (FM) features:
///
/// - ``full`` — OS >= 27: the `FoundationModels.LanguageModel` protocol is
///   public, so the full `MLXLanguageModel` adapter + `LanguageModelSession`
///   pipeline is available.
/// - ``partial`` — OS == 26: the `FoundationModels` framework is present (it
///   shipped in 26), but the `LanguageModel` protocol surface is gated off by
///   `@available(... 27, *)`. Non-FM MLX paths still work.
/// - ``absent`` — OS < 26: no `FoundationModels` framework on the OS at all.
///   The binary must still launch (FM is weak-linked) and non-FM MLX paths
///   still work.
///
/// Classification deliberately uses `ProcessInfo.operatingSystemVersion` rather
/// than `#available`: a single binary built against the 27 SDK has
/// `#if canImport(FoundationModels)` compile-time-true even when it runs on an
/// FM-absent OS, and `#available(... 27, *)` cannot distinguish OS 26 (partial)
/// from OS 18 (absent) — both are simply "< 27". The reported OS version is the
/// only signal that separates all three tiers. Probes then cross-check
/// `#available` *against* this version so a pre-release build where the two
/// disagree surfaces as its own failure.
enum DeviceTier: CustomStringConvertible {
    case full
    case partial
    case absent

    static var current: DeviceTier {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        if v.majorVersion >= 27 { return .full }
        if v.majorVersion >= 26 { return .partial }
        return .absent
    }

    var description: String {
        switch self {
        case .full: return "full (OS >= 27)"
        case .partial: return "partial (OS 26)"
        case .absent: return "absent (OS < 26)"
        }
    }
}
