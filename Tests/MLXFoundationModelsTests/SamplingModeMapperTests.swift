// Copyright © 2026 Apple Inc.

import Testing

#if FoundationModelsIntegration
    @testable import MLXLMCommon
    import MLXFoundationModels

    /// Coverage for the `samplingMode` → `GenerateParameters` resolver. Mirrors
    /// `SampleTests.testGenerateParametersCreatesExpectedSampler` by asserting both
    /// the resolved triple and the resulting `.sampler()` type: a parameter-only
    /// assertion would miss that e.g. `topK == 0` is inert.
    @Suite
    struct SamplingModeMapperTests {

        /// Build a sampler the way the bridge will: start from provider defaults,
        /// apply the resolution, then ask `GenerateParameters` for its sampler.
        private func sampler(
            for mode: MLXSamplingMode?, clampedTemperature: Float?
        ) -> LogitSampler {
            var params = GenerateParameters()
            resolveSamplingParameters(mode: mode, clampedTemperature: clampedTemperature)
                .apply(to: &params)
            return params.sampler()
        }

        private func resolve(
            _ mode: MLXSamplingMode?, _ temperature: Float?
        ) -> ResolvedSamplingParameters {
            resolveSamplingParameters(mode: mode, clampedTemperature: temperature)
        }

        // MARK: nil mode — provider defaults, byte-identical to today

        @Test func nilModeNilTempIsNoOp() {
            #expect(resolve(nil, nil) == ResolvedSamplingParameters())
            #expect(sampler(for: nil, clampedTemperature: nil) is CategoricalSampler)
        }

        @Test func nilModeKeepsCallerTemperature() {
            let r = resolve(nil, 0.7)
            #expect(r.temperature == 0.7)
            #expect(r.topP == nil)
            #expect(r.topK == nil)
            #expect(sampler(for: nil, clampedTemperature: 0.7) is CategoricalSampler)
        }

        // MARK: greedy — forces argmax, overrides temperature

        @Test func greedyForcesArgmax() {
            #expect(resolve(.greedy, nil).temperature == 0)
            #expect(sampler(for: .greedy, clampedTemperature: nil) is ArgMaxSampler)
        }

        @Test func greedyOverridesNonzeroTemperature() {
            #expect(resolve(.greedy, 0.8).temperature == 0)  // stomp, not 0.8
            #expect(sampler(for: .greedy, clampedTemperature: 0.8) is ArgMaxSampler)
        }

        // MARK: top-k

        @Test func topKEngagesWithDefaultTemperature() {
            let r = resolve(.topK(40), nil)
            #expect(r.temperature == nil)  // leave the 0.6 default so the filter engages
            #expect(r.topK == 40)
            #expect(sampler(for: .topK(40), clampedTemperature: nil) is TopPSampler)
        }

        @Test func topKKeepsCallerTemperature() {
            let r = resolve(.topK(40), 0.7)
            #expect(r.temperature == 0.7)
            #expect(r.topK == 40)
            #expect(sampler(for: .topK(40), clampedTemperature: 0.7) is TopPSampler)
        }

        @Test func topKOfOneIsValidNotGreedy() {
            #expect(resolve(.topK(1), nil).topK == 1)
            #expect(sampler(for: .topK(1), clampedTemperature: nil) is TopPSampler)
        }

        @Test func nonPositiveTopKDisablesFilterWithoutGoingGreedy() {
            #expect(resolve(.topK(0), nil).topK == nil)
            #expect(resolve(.topK(-5), nil).topK == nil)
            #expect(resolve(.topK(0), nil).temperature == nil)  // default temp, not 0
            #expect(sampler(for: .topK(0), clampedTemperature: nil) is CategoricalSampler)
        }

        @Test func largeTopKPassesThroughResolverDoesNotClamp() {
            // The resolver does not clamp; MLX's `applyTopK` guards `k >= vocab` downstream.
            #expect(resolve(.topK(1_000_000), nil).topK == 1_000_000)
        }

        // MARK: nucleus

        @Test func nucleusInRangeEngagesTopP() {
            let r = resolve(.nucleus(0.9), nil)
            #expect(r.topP == Float(0.9))
            #expect(r.topK == nil)
            #expect(sampler(for: .nucleus(0.9), clampedTemperature: nil) is TopPSampler)
        }

        @Test func nucleusAtOrAboveOneIsFullDistribution() {
            #expect(resolve(.nucleus(1.0), nil).topP == Float(1.0))
            #expect(sampler(for: .nucleus(1.0), clampedTemperature: nil) is CategoricalSampler)
            // 100.0 is SDK-emitted (GenerationOptionsTests.outOfBoundsValues) — tolerated, not an error.
            #expect(sampler(for: .nucleus(100.0), clampedTemperature: nil) is CategoricalSampler)
        }

        @Test func nucleusAtOrBelowZeroIsGreedy() {
            // "smallest possible pool" ≈ deterministic — argmax, not full-distribution sampling.
            #expect(resolve(.nucleus(0.0), nil).temperature == 0)
            #expect(resolve(.nucleus(0.0), nil).topP == nil)
            #expect(sampler(for: .nucleus(0.0), clampedTemperature: nil) is ArgMaxSampler)
            #expect(sampler(for: .nucleus(-0.5), clampedTemperature: nil) is ArgMaxSampler)
        }

        @Test func nucleusFloatNarrowingBoundary() {
            // Pin the observed narrowing: Float(0.9999999) stays < 1, so this remains a
            // real nucleus filter and does not silently collapse to full-distribution.
            #expect(Float(0.9999999) < 1)
            #expect(sampler(for: .nucleus(0.9999999), clampedTemperature: nil) is TopPSampler)
        }

        // MARK: explicit-zero-wins

        @Test func explicitZeroTemperatureBeatsTopK() {
            let r = resolve(.topK(40), 0)
            #expect(r.temperature == 0)
            #expect(r.topK == 40)  // present but inert under argmax
            #expect(sampler(for: .topK(40), clampedTemperature: 0) is ArgMaxSampler)
        }

        @Test func explicitZeroTemperatureBeatsNucleus() {
            let r = resolve(.nucleus(0.9), 0)
            #expect(r.temperature == 0)
            #expect(r.topP == Float(0.9))  // present but inert under argmax
            #expect(sampler(for: .nucleus(0.9), clampedTemperature: 0) is ArgMaxSampler)
        }

        // MARK: invariants

        @Test func resolverNeverEngagesMinP() {
            // No `SamplingMode` case maps to min-p; applying any resolution must leave
            // `minP` at its provider default.
            let modes: [MLXSamplingMode?] = [
                nil, .greedy, .topK(40), .topK(0), .nucleus(0.9), .nucleus(1.5), .nucleus(0.0),
            ]
            for mode in modes {
                var params = GenerateParameters()
                resolveSamplingParameters(mode: mode, clampedTemperature: nil).apply(to: &params)
                #expect(params.minP == 0.0)
            }
        }
    }
#endif
