# User Testing

Testing surface, resource cost classification, and validation approach.

**What belongs here:** Testing surface findings, validation tools, resource costs, runtime constraints.

---

## Validation Surface

This is a Swift Package library — no web UI. Validation is through:

1. **`swift test --filter MLXLMTests`** — All unit tests (existing + new batching tests)
2. **`swift build`** — Clean build verification
3. **CLI execution** (Milestone 5 only) — `llm-tool batch` subcommand in mlx-swift-examples
4. **`xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS,arch=arm64' ...`** — Required when MLX-backed tests need real Metal execution; unlike `swift test`, this path loads the Metal library and runs the MLX assertions instead of skipping them.

Primary testing tool: `swift test` (XCTest framework)

## Validation Concurrency

- **Machine:** 32GB RAM, 10 CPU cores (Apple Silicon)
- **`swift test` surface:** Each test run uses 1-3 CPU cores for compilation + test execution
- **Max concurrent validators:** 3 (conservative, since Swift builds are CPU-intensive)
- **Rationale:** Swift compilation peaks at ~8GB RAM and saturates available cores. Running 3 concurrent validators uses ~24GB peak, leaving headroom for OS.
- **Current batch-kv-cache decision:** Use **1 concurrent validator per repo checkout**. `swift test` writes to shared `.build` state, so validators must either run serially in the main checkout or use isolated scratch paths / working copies.
- **Current example-app decision:** Use **at most 1 validator in `mlx-swift-lm` and 1 validator in `mlx-swift-examples` concurrently**. The repos are independent, but each validator must use its own DerivedData/build location because `xcodebuild` and SwiftPM build products are not safe to share during parallel validation.

## Testing Patterns

- All batching tests use mock models (no model downloads)
- Mock models return deterministic outputs for verifiable behavior
- KV cache tests use synthetic tensors with known values
- Scheduler tests use MLX-backed mock models and the real scheduler path, with `skipIfMetalUnavailable()` guarding the MLX assertions that SwiftPM skips when the Metal library is unavailable
- Scheduler-test liveness caveat: `Tests/MLXLMTests/TestTokenizer.swift` treats token `0` as EOS/unknown, and common scheduler mocks such as `RotatingCacheMockModel` advance tokens modulo 32. A high `maxTokens` value alone therefore does **not** guarantee a request stays active long enough to trigger single→batch upgrade; use explicit synchronization or a mock token schedule that cannot wrap to EOS during the setup window.
- Existing tests must continue passing (regression safety)
- `swift test` is still useful for fast smoke checks, but MLX-dependent tests may all skip under SPM because `MLXMetalGuard` detects the missing Metal library.
- For milestone `batch-kv-cache`, direct user-validation evidence came from `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MLXLMTests/<TestClass>`.
- For milestone `batch-engine`, direct user-validation evidence came from targeted `xcodebuild` runs: `BatchTokenIteratorTests` can run as a class, while sampler assertions are safer to isolate per test (`testPerRequestSamplerIndependentBehavior`, `testConcurrentInsertAndNextSafety`, `testBatchVsSingleOutputMatchesWithArgMax`, `testPerRequestProcessorIndependentState`) because broader combined sampler runs can crash in the MLX concatenate path.
- For milestone `prompt-cache`, `PromptCacheBatchIntegrationTests` may need targeted `-only-testing` reruns for assigned assertions because the broader class run can fail on unrelated `testExactCacheMatchSkipsPrefill`; keep both the broad run log and the isolated rerun log as evidence when that happens.
- For milestone `post-review`, direct user-validation evidence came from targeted `xcodebuild` runs: `InferenceSchedulerTests` covers the stream-metadata assertions (`testThirdRequestJoinsExistingBatch`, `testBatchedInfoReportsCorrectPromptTokenCount`, `testFirstRequestPromptTimePreservedAfterUpgrade`, `testThirdRequestHasAccuratePromptTime`), `ModelContainerIntegrationTests` covers the prompt-cache / ChatSession assertions, and the rotating-cache type-preservation assertion lives in `BatchSamplingAndCorrectnessTests/testMakeBatchCachePreservesRotatingKVCacheType` rather than `BatchTokenIteratorTests`.
- For milestone `post-review-followup`, direct user-validation evidence came from targeted `xcodebuild test-without-building` reruns against existing followup build products: `BatchKVCacheTests/testMakeMaskBeforeUpdate` + `testMakeMaskLeftPaddingDecode` cover `VAL-FIX-010`, and `PromptCacheBatchIntegrationTests/testMixedDepthCachedPrefillIntegration` covers `VAL-FIX-011`.
- Some `xcodebuild` runs emit non-fatal `com.apple.metal` `flock failed to lock list file` warnings; record them as friction, but if the run still ends with `** TEST SUCCEEDED **` they do not block assertion validation.

## Flow Validator Guidance: swift-test

- Surface: SwiftPM/XCTest via `swift test` in the repo root.
- Isolation boundary: do not edit source files; only write artifacts under `.factory/validation/<milestone>/user-testing/flows/` and mission evidence directories.
- For parallel execution, each validator must use its own scratch/build directory (for example under `/tmp`) or its own checkout. Do not share `.build` writes across concurrent validators.
- Capture the exact `swift test --filter ...` command, exit code, and the assertion IDs covered by that run in the flow report.
- If Metal-backed MLX tests skip because the debug Metal library is unavailable, treat the skip as part of the observed behavior and report whether the targeted assertion still received direct evidence from the test run.
- When MLX assertions require direct runtime evidence, prefer `xcodebuild test` on the Swift package (`mlx-swift-lm-Package`, destination `platform=macOS,arch=arm64`) and use `swift test` only as supplemental evidence.
- If SwiftPM manifest linking fails in the default temp area with `errno=28` / `No space left on device`, retry with `TMPDIR` redirected to a validator-owned writable directory (for example under the evidence directory).

## Flow Validator Guidance: xcodebuild-test

- Surface: Xcode package tests via `xcodebuild test` against scheme `mlx-swift-lm-Package` on destination `platform=macOS,arch=arm64`.
- Isolation boundary: do not edit source files; only write artifacts under `.factory/validation/<milestone>/user-testing/flows/` and mission evidence directories.
- Use a validator-specific DerivedData path (for example `/tmp/mlx-swift-lm-<milestone>-<group>/DerivedData`) so concurrent or repeated runs do not reuse stale build products.
- For milestone `scheduler`, use `.factory/services.yaml` command `test-scheduler-runtime` or the equivalent `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MLXLMTests/InferenceSchedulerTests -only-testing:MLXLMTests/ModelContainerIntegrationTests`.
- If a fresh `xcodebuild test` attempt fails before execution with `errno=28` / `No space left on device`, and an already-built validator-owned DerivedData tree for the same revision exists, prefer a targeted `xcodebuild test-without-building` rerun against that existing DerivedData rather than reusing shared workspace build products blindly.
- Capture the exact `xcodebuild test` command, exit code, assertion IDs covered, and notable test counts / failure lines in the flow report.
- Save the raw xcodebuild log under the assigned evidence directory so later reruns can inspect the exact runtime output.

## Flow Validator Guidance: llm-tool-cli

- Surface: the `llm-tool` command-line app in `/Users/ronaldmannak/Developer/Projects/Pico AI Homelab/mlx-swift-examples`.
- Isolation boundary: do not edit source files; only write artifacts under `.factory/validation/<milestone>/user-testing/flows/` and mission evidence directories.
- Build with a validator-specific DerivedData path, for example `xcodebuild build -scheme llm-tool -destination 'platform=macOS,arch=arm64' ONLY_ACTIVE_ARCH=YES ARCHS=arm64 -derivedDataPath /tmp/mlx-swift-examples-<milestone>-<group>/DerivedData`.
- After building, run the produced binary directly from DerivedData (for example `/tmp/.../DerivedData/Build/Products/Debug/llm-tool --help` and `... llm-tool batch --help`) so the evidence reflects the real shipped CLI surface.
- For runtime generation validation, only use an **already-present absolute local model directory** via `--model /absolute/path`. Do **not** trigger Hugging Face downloads during validation for this mission. If no local model assets are available, record the runtime assertion as blocked with that reason.
- As of `2026-03-14`, `/Users/ronaldmannak/Documents/huggingface/models` only contained `.safetensors` weights for embedding models (`nomic-ai/nomic-embed-text-v1.5` and `TaylorAI/bge-micro-v2`); the inspected `mlx-community` text-generation directories only had config/tokenizer files, so offline `llm-tool batch` runtime validation remains blocked unless a usable local generative MLX model is staged first.
- Capture the exact build/help/runtime commands, exit codes, notable output lines, and any blocked-runtime reason in the flow report. Save raw build logs under the assigned evidence directory.
