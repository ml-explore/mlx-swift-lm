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
