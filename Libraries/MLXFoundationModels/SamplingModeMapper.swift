// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
import MLXLMCommon

/// Sampling-strategy selection for the adapter, resolved to the
/// `GenerateParameters` fields MLX's sampler consumes.
///
/// The adapter translates the FoundationModels `GenerationOptions.SamplingMode`
/// into this enum at dispatch (dropping the best-effort `seed`, which MLX's
/// samplers cannot honor) and applies the result to `GenerateParameters` via
/// ``resolveSamplingParameters(mode:clampedTemperature:)``.
public enum MLXSamplingMode: Sendable, Equatable {
    /// Deterministic decoding — always pick the most likely token.
    case greedy

    /// Top-k sampling. `k <= 0` disables the filter: MLX has no expression for a
    /// non-positive top-k, so the provider default (no top-k) stands.
    case topK(Int)

    /// Nucleus (top-p) sampling. `p <= 0` ("smallest possible pool") is treated
    /// as greedy; `p >= 1` keeps the full distribution (MLX normalizes a `topP`
    /// outside `(0, 1)` to "no top-p filter").
    case nucleus(Double)
}

/// The sampling fields a resolved ``MLXSamplingMode`` contributes to
/// `GenerateParameters`. A `nil` field means "leave the provider default in
/// place." The resolver never emits a concrete temperature default, because that
/// would collapse the unset-vs-explicit-zero distinction the explicit-zero-wins
/// rule relies on (`GenerateParameters.temperature` defaults to a sampling value).
public struct ResolvedSamplingParameters: Sendable, Equatable {
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?

    public init(temperature: Float? = nil, topP: Float? = nil, topK: Int? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    /// Apply only the fields this resolution sets, leaving every other
    /// `GenerateParameters` field (including `minP` and the temperature default)
    /// untouched.
    public func apply(to parameters: inout GenerateParameters) {
        if let temperature { parameters.temperature = temperature }
        if let topP { parameters.topP = topP }
        if let topK { parameters.topK = topK }
    }
}

/// Translate a sampling mode plus the caller's already-clamped temperature into
/// the `GenerateParameters` fields to set.
///
/// Precedence ladder (matches AFM's behavior at the value level —
/// `GenerativeModelInferenceSession`):
/// 1. An explicit `clampedTemperature == 0` forces argmax, before the mode is
///    consulted (an explicit zero is a deliberate determinism signal).
/// 2. `.greedy` — and a degenerate `.nucleus(p <= 0)`, whose "smallest pool"
///    intent is deterministic — forces argmax, overriding the default temperature.
/// 3. Otherwise the mode's filter is applied at the caller's-or-default temperature.
///
/// `GenerateParameters.temperature` defaults to `0.6` (a sampling value), so for
/// top-k / nucleus a `nil` temperature output deliberately leaves that default in
/// place — emitting `0` would route `sampler()` to argmax and silently ignore the
/// filter. The resolver does not clamp large top-k; MLX's `applyTopK` guards
/// `k >= vocab` downstream.
public func resolveSamplingParameters(
    mode: MLXSamplingMode?,
    clampedTemperature: Float?
) -> ResolvedSamplingParameters {
    var topP: Float?
    var topK: Int?
    var forcesGreedy = false

    switch mode {
    case .none:
        break
    case .greedy:
        forcesGreedy = true
    case .topK(let k):
        topK = k >= 1 ? k : nil
    case .nucleus(let p):
        if p <= 0 {
            forcesGreedy = true  // smallest possible pool ≈ deterministic
        } else {
            topP = Float(p)  // MLX normalizes p >= 1 to "no filter" (full distribution)
        }
    }

    let explicitZero = clampedTemperature.map { $0 == 0 } ?? false
    let temperature: Float? = (explicitZero || forcesGreedy) ? 0 : clampedTemperature

    return ResolvedSamplingParameters(temperature: temperature, topP: topP, topK: topK)
}
#endif
