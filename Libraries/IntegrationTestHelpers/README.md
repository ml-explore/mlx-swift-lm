# Integration Test Helpers

`IntegrationTestHelpers` and `BenchmarkHelpers` provide shared test logic for verifying end-to-end model loading, inference, tokenizer performance, and download performance. They are designed to be used by integration packages that supply their own `Downloader` and `TokenizerLoader` implementations.

## Running integration tests locally

The `MLXLMIntegrationTests` test target in this repo uses [Swift Hugging Face](https://github.com/huggingface/swift-huggingface) and [Swift Transformers](https://github.com/huggingface/swift-transformers) via the `MLXHuggingFace` macros to provide `Downloader` and `TokenizerLoader` implementations. Models are downloaded from Hugging Face Hub on first run and cached in `~/.cache/huggingface/`.

```bash
# Run all integration tests (requires macOS with Metal)
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXLMIntegrationTests

# Run a single model's tests
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXLMIntegrationTests/ToolCallIntegrationTests/qwen35FormatAutoDetection
```

## External integration packages

Integration tests and benchmarks can also be run from external packages:

- [Swift Tokenizers MLX](https://github.com/DePasqualeOrg/swift-tokenizers-mlx): Uses [Swift Tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) and [Swift HF API](https://github.com/DePasqualeOrg/swift-hf-api)
- [Swift Transformers MLX](https://github.com/DePasqualeOrg/swift-transformers-mlx): Uses [Swift Transformers](https://github.com/huggingface/swift-transformers) and [Swift Hugging Face](https://github.com/huggingface/swift-huggingface)
