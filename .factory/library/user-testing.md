# User Testing

Testing surface, resource cost classification, and validation approach.

**What belongs here:** Testing surface findings, validation tools, resource costs, runtime constraints.

---

## Validation Surface

This is a Swift Package library — no web UI. Validation is through:

1. **`swift test --filter MLXLMTests`** — All unit tests (existing + new batching tests)
2. **`swift build`** — Clean build verification
3. **CLI execution** (Milestone 5 only) — `llm-tool batch` subcommand in mlx-swift-examples

Primary testing tool: `swift test` (XCTest framework)

## Validation Concurrency

- **Machine:** 32GB RAM, 10 CPU cores (Apple Silicon)
- **`swift test` surface:** Each test run uses 1-3 CPU cores for compilation + test execution
- **Max concurrent validators:** 3 (conservative, since Swift builds are CPU-intensive)
- **Rationale:** Swift compilation peaks at ~8GB RAM and saturates available cores. Running 3 concurrent validators uses ~24GB peak, leaving headroom for OS.

## Testing Patterns

- All batching tests use mock models (no model downloads)
- Mock models return deterministic outputs for verifiable behavior
- KV cache tests use synthetic tensors with known values
- Scheduler tests use mock TokenIterator/BatchTokenIterator stubs
- Existing tests must continue passing (regression safety)
