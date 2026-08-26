# Contributing to MLX Swift LM

We want to make contributing to this project as easy and transparent as
possible.

## AI Usage Policy

AI-generated code is allowed. What is not allowed is submitting code you do not
understand. You are 100% responsible for every line, however it was produced,
and must explicitly disclose the manner in which AI was employed.

AI-drafted prose is allowed too — commit messages, pull request descriptions,
issues, discussions, and replies to reviewers. The condition is that you read
every word before it is submitted and confirm it says what you mean. By
contributing you agree that you have done so, and the pull request template asks
you to confirm it for the description. Please don't ask a reviewer to read prose
you did not read yourself.

Agents working in this repo should follow [AGENTS.md](AGENTS.md), which encodes
these rules.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications IntegrationTesting
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests run without any special hardware and do not download models.
Note: `swift test` [does not work yet](https://github.com/ml-explore/mlx-swift?tab=readme-ov-file#xcodebuild) — use `xcodebuild` instead:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation
```

Integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. They
are not part of the pull request checks, so they never block a merge. In
`ml-explore/mlx-swift-lm` they run nightly on a self-hosted macOS runner
(`.github/workflows/integration_tests.yml`), and failures are reported on a
tracking issue labeled `ci-failure`. Because nothing runs them on your branch,
run them locally when you change model loading, generation, or tokenizer
behavior.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

CI also verifies that DocC documentation builds without warnings for every
library target. Run the same check locally with:

```bash
scripts/verify-docs.sh
```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift LM, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
