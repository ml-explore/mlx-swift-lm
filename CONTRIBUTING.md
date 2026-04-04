# Contributing to MLX Swift Examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests run without any special hardware and do not download models:

```
swift test
```

Integration tests verify end-to-end model loading and generation. They require macOS with Metal and download models from Hugging Face Hub on first run:

```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:MLXLMIntegrationTests
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift Examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
