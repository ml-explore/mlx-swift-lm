# Integration Test Helpers

`IntegrationTestHelpers` and `BenchmarkHelpers` provide shared test logic for verifying end-to-end model loading, inference, and performance. They are designed to be used by integration packages that supply their own `Downloader` and `TokenizerLoader` implementations.

## Integration packages

- [Swift Tokenizers MLX](https://github.com/DePasqualeOrg/swift-tokenizers-mlx): Uses [Swift Tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) and [Swift HF API](https://github.com/DePasqualeOrg/swift-hf-api)
- [Swift Transformers MLX](https://github.com/DePasqualeOrg/swift-transformers-mlx): Uses [Swift Transformers](https://github.com/huggingface/swift-transformers) and [Swift Hugging Face](https://github.com/huggingface/swift-huggingface)

Integration tests and benchmarks are run from those packages.
