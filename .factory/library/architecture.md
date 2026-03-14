# Architecture

Architectural decisions, patterns, and knowledge discovered during the mission.

**What belongs here:** Architectural decisions, patterns discovered, module boundaries, key abstractions.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Project Structure

- `Libraries/MLXLMCommon/` — Core shared library (generation, KV cache, model protocols, chat session)
- `Libraries/MLXLLM/` — LLM model implementations (~55 models)
- `Libraries/MLXVLM/` — VLM model implementations
- `Libraries/MLXEmbedders/` — Embedding models
- `Tests/MLXLMTests/` — Unit tests
- `Tests/MLXLMIntegrationTests/` — Integration tests (require model downloads)

## New Batching Code Location

All new batching code goes in `Libraries/MLXLMCommon/Batching/`:
- `BatchKVCache.swift` — Batch-aware KV cache with left-padding
- `BatchRotatingKVCache.swift` — Sliding window variant
- `BatchPositionedCache.swift` — Protocol for batch-aware RoPE
- `BatchTokenIterator.swift` — Core batch generation engine
- `InferenceScheduler.swift` — Scheduler with single-to-batch upgrade
- `LRUPromptCache.swift` — Trie-based prompt cache

## Key Design Decisions

### Single-First Upgrade Pattern
Single requests use the existing `TokenIterator` path. Only when a second concurrent request arrives does the system upgrade to batching. This ensures zero overhead for the common single-request case.

### BatchPositionedKVCache Protocol
A protocol abstraction that lets models call `applyRotaryPosition(rope, to: x, cache: cache)` instead of `rope(x, offset: cache.offset)`. This keeps per-model changes to ~4 lines while supporting both single (Int offset) and batch (MLXArray offset) modes.

### Left-Padding Strategy
Variable-length sequences are left-padded with zeros. `BatchKVCache` tracks per-sequence `leftPadding` and adjusts attention masks accordingly. This matches the Python mlx-lm approach.

## Existing Infrastructure Used

- RoPE with MLXArray offsets: All RoPE implementations already support `callAsFunction(_ x: MLXArray, offset: MLXArray)` via `ArrayOffsetLayer` protocol
- `createCausalMask` already has a `lengths: MLXArray?` parameter for per-sequence masking
- KV cache tensors already have batch dimension `[B, H, S, D]`
- `ModelContainer` has `SerialAccessContainer` for thread-safe model access
- `WiredMemoryPolicies` for memory coordination

## Python mlx-lm Architecture Mapping

| Python | Swift |
|--------|-------|
| `BatchGenerator` | `BatchTokenIterator` |
| `Batch` dataclass | `ActiveBatch` struct |
| `BatchKVCache` | `BatchKVCache` |
| `ResponseGenerator` | `InferenceScheduler` |
| `LRUPromptCache` | `LRUPromptCache` |
