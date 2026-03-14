# MLX Validation

- `swift test --filter MLXLMTests` is a fast smoke check in this repo, but MLX-backed assertions can skip in SwiftPM debug builds when `MLXMetalGuard` detects that the debug Metal library is unavailable.
- For scheduler batching, cache migration, or other runtime MLX behaviors, prefer targeted `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MLXLMTests/<TestClass or test>` runs because that path loads Metal and exercises the real MLX execution path.
- Treat passing `swift build` and `swift test` as baseline validation only; they do not by themselves prove MLX-backed scheduler upgrade behavior.
