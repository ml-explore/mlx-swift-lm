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
  -only-testing:IntegrationTestingTests/ModelLoadingSuite/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

These tests do not run in CI.

The reranker integration tests download BGE v2 M3, Qwen3 Reranker 0.6B, and Jina
Reranker v3 checkpoints. To run only those reference checks:

```bash
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/ModelLoadingSuite/RerankerModelIntegrationTests
```

The decision model tests download LiquidAI d1-3B (6.2 GB) and d1-omni-600M (2.3 GB), so they
are skipped unless `MLX_RUN_DECISION_INTEGRATION=1`. With `xcodebuild`, prefix the variable
with `TEST_RUNNER_`:

```bash
TEST_RUNNER_MLX_RUN_DECISION_INTEGRATION=1 xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/DecisionModelIntegrationTests
```

They load both models through `DecisionModelFactory` and check the README examples (a
support ticket, the two cats photo, and for d1-omni a speech clip) against reference
probabilities on the same pinned checkpoints: d1-3B's PyTorch reference and mlx-vlm for
d1-omni. The photo and the clip come from pinned Hugging Face datasets. Each checkpoint runs
in the precision it ships in:

- d1-omni-600M (float32) must match to 1e-4, and to 0.02 for the photo, because Core Image
  decodes JPEG a few levels away from libjpeg-turbo.
- d1-3B (bfloat16) must match to 0.05. swift-transformers before 1.4.0, which the test
  project still allows, encodes some d1-3B text differently from Hugging Face tokenizers, for
  example a word followed by `"\n\n"`, so the d1-3B token counts are not compared.

A further test checks that the generic `loadModelContainer` still loads d1-3B as a chat
model.

`Tests/MLXLMTests` also has gated suites that compare these models with their references
more tightly, from local checkpoints and fixture files; see
[Tests/MLXLMTests/README.md](../../Tests/MLXLMTests/README.md).
