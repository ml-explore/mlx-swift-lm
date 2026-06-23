// Compile-only probe. Proves at COMPILE time, against the macOS-27 SDK, that:
//   1. `@testable import MLXFoundationModels` resolves from this xcodeproj
//      test target against the local SwiftPM package.
//   2. An `internal` symbol (`MLXLanguageModel.Executor.samplingMode(from:)`)
//      is reachable, i.e. the package was built with testability enabled and
//      the FoundationModelsIntegration trait came in enabled (the module is
//      not the empty trait-disabled variant).
//
// If this file COMPILES, the gate is green. It is never executed — the
// function is unreferenced and `@available`-gated.

#if FoundationModelsIntegration
    import FoundationModels
    @testable import MLXFoundationModels

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    func _testabilityProbe() {
        // Internal static on the bridge-local Executor — only visible via @testable.
        _ = MLXLanguageModel.Executor.samplingMode(from: nil)
    }
#endif
