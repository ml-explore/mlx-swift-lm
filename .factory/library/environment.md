# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Platform Requirements

- macOS 14+ / iOS 17+ (Apple Silicon required for MLX)
- Swift 5.12+
- Xcode (for mlx-swift-examples repo)

## Dependencies

- `mlx-swift` 0.30.6+ (MLX framework for Apple Silicon)
- `swift-transformers` 1.2.0+ (HuggingFace tokenizer support)

## Build Notes

- StrictConcurrency is enabled for all targets
- Metal library loading may show warnings in test environments without GPU — this is expected and doesn't affect test results
- The mlx-swift-examples repo uses an Xcode project (.xcodeproj) and references mlx-swift-lm as a remote SPM dependency

## Test Notes

- Unit tests: `swift test --filter MLXLMTests` (no model downloads)
- Integration tests require model downloads and are not run in this mission
- Benchmarks in `Tests/Benchmarks/` are separate from unit tests

## Known Environment Limitation: MLX Metal Library in SPM Builds

`swift test` shows "Failed to load the default metallib" error. This is a pre-existing issue affecting ALL MLX-dependent tests. Tests that call array evaluation operations (.item(), eval(), allClose(), etc.) cannot fully execute in SPM debug builds. The test harness still reports exit code 0.

Workarounds:
- Tests run correctly in Xcode (which loads Metal libraries properly)
- `swift test` still validates compilation and non-MLX test logic
- Workers should write tests that verify as much as possible through structure
- The `swift test` exit code 0 is the acceptance criterion

### Reusable test guard pattern

- `Tests/MLXLMTests/MLXMetalGuard.swift` provides `MLXMetalGuard.isAvailable` and `skipIfMetalUnavailable()` for XCTest-based suites.
- Swift Testing suites can gate Metal-dependent cases with `.enabled(if: MLXMetalGuard.isAvailable)`.
- Reuse this helper instead of open-coding metallib checks in new MLX-dependent tests.
