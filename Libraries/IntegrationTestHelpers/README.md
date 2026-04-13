# Integration Test Helpers

`IntegrationTestHelpers` and `BenchmarkHelpers` provide shared test logic for verifying end-to-end model loading, inference, tokenizer performance, and download performance. They are designed to be used by integration packages that supply their own `Downloader` and `TokenizerLoader` implementations.

## Running integration tests locally

The `IntegrationTesting/IntegrationTesting.xcodeproj` Xcode project in this repo uses [Swift Hugging Face](https://github.com/huggingface/swift-huggingface) and [Swift Transformers](https://github.com/huggingface/swift-transformers) via the `MLXHuggingFace` macros to provide `Downloader` and `TokenizerLoader` implementations. Models are downloaded from Hugging Face Hub on first run and cached in `~/.cache/huggingface/`.

To run integration tests, open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests (requires macOS with Metal)
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS'

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

These tests do not run in CI.

## External integration packages

Integration tests and benchmarks can also be run from external packages:

- [Swift Tokenizers MLX](https://github.com/DePasqualeOrg/swift-tokenizers-mlx): Uses [Swift Tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) and [Swift HF API](https://github.com/DePasqualeOrg/swift-hf-api)
- [Swift Transformers MLX](https://github.com/DePasqualeOrg/swift-transformers-mlx): Uses [Swift Transformers](https://github.com/huggingface/swift-transformers) and [Swift Hugging Face](https://github.com/huggingface/swift-huggingface)
