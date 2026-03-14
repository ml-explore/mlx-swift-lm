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

### TokenIterator Upgrade Constraint — Cooperative Handoff
`TokenIterator` in `Libraries/MLXLMCommon/Evaluate.swift` is a mutable value type (`struct`) whose decode state lives in fields like `y`, `cache`, and `tokenCount`. The scheduler's actor state stores a copy at submission time, but as the single-request Task advances its own copy diverges. Reading the actor copy during upgrade would yield stale KV cache state.

**Solution**: The `UpgradeFlag` class mediates a cooperative handoff. When a second request arrives:
1. `upgradeToBatch()` sets `upgradeFlag.upgradeRequested = true` and suspends via `withCheckedContinuation`.
2. The single-request task detects `upgradeRequested` between decode steps, captures its live `TokenIterator` state (`LiveIteratorState`), and resumes the continuation via `depositLiveState()`.
3. The scheduler uses the live cache/y/tokenCount to build the `ActiveBatch`.
4. The first request's `onTermination` handler is rebound to remove its UID from `BatchTokenIterator` (not cancel the defunct single task).

### BatchPositionedKVCache Protocol
A protocol abstraction that lets models call `applyRotaryPosition(rope, to: x, cache: cache)` instead of `rope(x, offset: cache.offset)`. This keeps per-model changes to ~4 lines while supporting both single (Int offset) and batch (MLXArray offset) modes.

### Left-Padding Strategy
Variable-length sequences are left-padded with zeros. `BatchKVCache` tracks per-sequence `leftPadding` and adjusts attention masks accordingly. This matches the Python mlx-lm approach.

### Mask Before Cache Update
Attention-mask creation uses the cache's pre-update position. `makeAttentionMask` / `createAttentionMask` call `cache.makeMask(...)` before the layer appends the current keys and values, so batch cache masking must use the current `_idx` / offset rather than subtracting `n` as if the cache had already been updated.

### Rotating cache keep semantics
The repo's existing max-KV path preserves a fixed prefix when it creates `RotatingKVCache(maxSize: maxKVSize, keep: 4)` in `Libraries/MLXLMCommon/LanguageModel.swift`. Any batch rotating-cache implementation needs to preserve and round-trip nonzero `keep` values instead of assuming the default `keep = 0`.

### Rotating Cache Cached-Prompt Prefill
Batch rotating-cache cached-prefill uses a `prepare(... rightPadding:)` / `finalize()` lifecycle. During mixed-length cached prompt prefill, sequences temporarily switch to right-padding so concatenation and trimming operate on aligned suffixes, then `finalize()` rolls the data back into the normal left-padded layout used for decode.

### Rotating Cache Overflow Extraction
During active sliding-window decode, `BatchRotatingKVCache` can drive per-sequence `leftPadding` below zero as wrapped tokens replace old window positions. Extraction must clamp that value back to `max(0, leftPadding)` before slicing, otherwise overflowed batch caches can slice from a negative start and drop the preserved `[keep-prefix | window]` contents during merge → overflow → extract round-trips.

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
