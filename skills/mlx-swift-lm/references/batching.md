# Batched Inference & Prompt Caching

## Overview

The batching system enables transparent continuous batching of multiple concurrent inference requests through a single model. It uses a single-first upgrade strategy: the first request runs the existing fast `TokenIterator` path, and when a second concurrent request arrives, the scheduler upgrades to a `BatchTokenIterator` by migrating KV caches.

**Files:**
- `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift`
- `Libraries/MLXLMCommon/Batching/BatchTokenIterator.swift`
- `Libraries/MLXLMCommon/Batching/BatchKVCache.swift`
- `Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift`
- `Libraries/MLXLMCommon/Batching/BatchPositionedCache.swift`
- `Libraries/MLXLMCommon/Batching/LRUPromptCache.swift`
- `Libraries/MLXLMCommon/Batching/SchedulerTokenHandler.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `InferenceScheduler` | Actor managing request lifecycle with single-first upgrade strategy |
| `BatchTokenIterator` | Batch prefill/decode engine for multiple sequences |
| `BatchKVCache` | Batched KV cache `[B, nHeads, seqLen, headDim]` with left-padding |
| `BatchRotatingKVCache` | Batched sliding-window KV cache for `maxKVSize` models |
| `BatchPositionedKVCache` | Protocol for caches that provide per-sequence positional offsets |
| `LRUPromptCache` | Trie-based LRU cache for reusing prefill KV state across requests |
| `PendingPrompt` | Struct describing a request waiting to join a batch |
| `ActiveBatch` | Mutable state for the currently-running batch |
| `applyRotaryPosition()` | Helper that dispatches RoPE to batch or scalar offset |
| `isBatchCompatible()` | Check whether caches support batch merge/extend |

## Enabling Batching

### Via ModelContainer (Recommended)

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// Enable batching
container.scheduler = InferenceScheduler()

// Optional: enable prompt caching
container.promptCache = LRUPromptCache(maxSize: 10)

// Use normally — batching is transparent
let stream = try await container.generate(input: lmInput, parameters: params)
```

When `scheduler` is set on `ModelContainer`:
- `generate()` routes through `InferenceScheduler.submit()` (decoded text)
- `generateTokens()` routes through `InferenceScheduler.submitTokens()` (raw tokens)
- VLM models bypass the scheduler (not yet batch-compatible)

### Direct Scheduler Usage

```swift
let scheduler = InferenceScheduler()

let stream = try await scheduler.submit(
    input: lmInput,
    parameters: params,
    model: model,
    cache: nil,
    tokenizer: tokenizer,
    configuration: config
)

for await generation in stream {
    switch generation {
    case .chunk(let text): print(text, terminator: "")
    case .toolCall(let call): print("Tool: \(call.function.name)")
    case .info(let info): print("\nDone: \(info.tokensPerSecond) tok/s")
    }
}
```

### Raw Token Batching

```swift
let tokenStream = try await scheduler.submitTokens(
    input: lmInput,
    parameters: params,
    model: model,
    cache: nil,
    tokenizer: tokenizer,
    configuration: config,
    includeStopToken: false
)

for await event in tokenStream {
    switch event {
    case .token(let tokenID): print(tokenID)
    case .info(let info): print("stop=\(info.stopReason)")
    }
}
```

## InferenceScheduler State Machine

The scheduler is a Swift actor with three main states:

```
.idle → .single → .batched
                ↗
         .pendingUpgrade → .upgrading
```

- **`.idle`**: No active generation.
- **`.single`**: First request running via `TokenIterator` (fast path, zero batch overhead).
- **`.pendingUpgrade`**: Second request arrived; waiting for wired-memory admission.
- **`.upgrading`**: Migrating KV caches from single to batch. Additional requests during this phase run independently on the single path.
- **`.batched`**: Multiple requests active via `BatchTokenIterator`.

### Upgrade Flow

1. First request starts → state = `.single`
2. Second compatible request arrives → state = `.pendingUpgrade`
3. Scheduler signals the single-request task to capture its live `TokenIterator` state
4. Live state (KV cache, current token, samplers) deposited → state = `.upgrading`
5. Scheduler builds `BatchTokenIterator` from both requests → state = `.batched`
6. When all batch requests complete → state = `.idle`

### Batch Compatibility

Not all requests can be batched together. Incompatible requests run independently on the single path:

```swift
// Check cache compatibility
InferenceScheduler.isBatchCompatible(model: model, cache: cache)

// Returns false for:
// - CacheList (hybrid models like Jamba)
// - MambaCache (SSM state-space caches)
// - QuantizedKVCache (quantized tuples)
// - Multimodal models (VLMs)
```

## BatchKVCache

Batched version of `KVCacheSimple`. Stores keys/values in `[B, nHeads, seqLen, headDim]` layout with left-padding for sequences of different lengths.

```swift
// Created from single caches during upgrade
let batchCache = BatchKVCache(leftPadding: [0, 5, 3])  // per-sequence padding

// Key properties
batchCache.batchSize        // Number of sequences
batchCache.batchOffset      // Per-sequence position offsets [B]
batchCache.isEmpty          // True if no KV state stored

// Batch operations
batchCache.filter(batchIndices: [0, 2])   // Remove completed sequences
batchCache.extend(other: newBatchCache)   // Add new sequences to batch
batchCache.extract(idx: 1)               // Extract single KVCacheSimple
batchCache.toSingle()                    // Convert B=1 batch to KVCacheSimple

// Cached-prompt prefill lifecycle
batchCache.prepare(rightPadding: padding)  // Set up for cached prefill
batchCache.finalize()                      // Trim padding after prefill
```

## BatchRotatingKVCache

Batched sliding-window cache for models using `maxKVSize`:

```swift
let batchCache = BatchRotatingKVCache(
    maxSize: 4096,
    leftPadding: [0, 5],
    keep: 4  // Tokens to always keep at start
)
```

Same batch operations as `BatchKVCache` (`filter`, `extend`, `extract`, `toSingle`).

## BatchPositionedKVCache Protocol

Protocol for batch-aware KV caches that provide per-sequence positional offsets:

```swift
public protocol BatchPositionedKVCache: KVCache {
    var batchOffset: MLXArray { get }  // Shape [B], per-sequence offsets
}
```

Both `BatchKVCache` and `BatchRotatingKVCache` conform to this protocol.

## applyRotaryPosition Helper

Use this in model implementations instead of direct `rope(x, offset:)` calls to support both single and batch paths:

```swift
public func applyRotaryPosition<R: RoPELayer>(
    _ rope: R, to x: MLXArray, cache: KVCache?
) -> MLXArray

// In model attention:
queries = applyRotaryPosition(rope, to: queries, cache: cache)
keys = applyRotaryPosition(rope, to: keys, cache: cache)
```

- For `BatchPositionedKVCache`: uses `rope(x, offset: batchOffset)` with per-sequence `MLXArray` offsets
- For single caches: uses `rope(x, offset: cache.offset)` with scalar `Int` offset
- For `nil` cache: uses offset 0

## BatchTokenIterator

The batch prefill/decode engine. Manages pending prompts, active batch state, and per-sequence sampling.

```swift
let batchIterator = BatchTokenIterator(
    model: model,
    stopTokens: eosTokenIds,
    defaultSampler: params.sampler(),
    completionBatchSize: 8,   // Max sequences in decode
    prefillBatchSize: 4,      // Max sequences prefilled at once
    prefillStepSize: 512      // Prompt chunk size
)

// Insert a request
let uid = batchIterator.allocateUID()
batchIterator.insert(
    uid: uid,
    tokens: tokenArray,
    maxTokens: 1000,
    sampler: customSampler,
    processor: customProcessor,
    cachedKVState: cachedCache
)

// Decode loop
while let responses = batchIterator.next() {
    for response in responses {
        // response.uid — which sequence
        // response.token — generated token ID
        // response.finishReason — nil while generating, .stop/.length/.cancelled when done
        // response.finalCache — extracted per-layer KV cache when finished
    }
}
```

### PendingPrompt

Describes a request waiting to be prefilled:

```swift
public struct PendingPrompt: @unchecked Sendable {
    public let uid: Int
    public let tokens: [Int]
    public let maxTokens: Int
    public let sampler: (any LogitSampler)?
    public let processor: LogitProcessor?
    public let cachedKVState: [KVCache]?
    public var effectiveLength: Int { tokens.count }
}
```

### ActiveBatch

Mutable state for the currently-running batch:

```swift
public class ActiveBatch {
    public var uids: [Int]
    public var y: MLXArray           // Current tokens [B, 1]
    public var cache: [KVCache]      // Per-layer batch caches
    public var samplers: [LogitSampler?]
    public var processors: [LogitProcessor?]
    public var maxTokens: [Int]
    public var numTokens: [Int]
    public var tokens: [MLXArray]    // Per-sequence generated tokens
    public var count: Int { uids.count }

    public func filter(keepIndices: [Int])
    public func extend(other: ActiveBatch)
}
```

## LRUPromptCache

Trie-based LRU cache that stores KV state for reuse across requests with matching prompt prefixes:

```swift
let promptCache = LRUPromptCache(
    maxSize: 10,           // Max cached sequences
    maxBytes: Int.max      // Max total bytes
)

// Fetch nearest cached prefix
let (cachedKVState, uncachedTokens) = promptCache.fetchNearestCache(
    model: "Qwen3-4B",
    tokens: inputTokenIds
)

// Store KV state after generation
promptCache.insertCache(
    model: "Qwen3-4B",
    tokens: fullTokenSequence,
    cache: kvCacheLayers
)

// Eviction
promptCache.trimTo(nSequences: 5)
promptCache.trimTo(nBytes: 1_000_000_000)

// Properties
promptCache.count   // Number of cached sequences
promptCache.nbytes  // Total bytes in cache
```

When used with `ModelContainer`, prompt caching is automatic:
```swift
container.promptCache = LRUPromptCache(maxSize: 10)
// All subsequent generate() calls check cache before prefill
```

## Known Limitations

### RoPE Position Limitation
Models use `cache.offset: Int` for single sequences. For batch with left-padding, the decode token can get wrong RoPE by `leftPadding[i]` positions for different-length sequences. The `applyRotaryPosition()` helper with `BatchPositionedKVCache.batchOffset` addresses this for models that have been migrated.

### Attention Mask Limitation
Models using the deprecated `createAttentionMask(h:cache:[KVCache]?)` return `nil` for single-token decode, but `BatchKVCache.makeMask()` produces correct masks. Models should use `cache.makeMask(n:windowSize:returnArray:)` or the non-deprecated single-cache API.

### VLM Not Supported
Vision-Language Models bypass the scheduler. Multimodal inputs are not yet batch-compatible.

### Incompatible Cache Types
Quantized KV caches, Mamba/SSM caches, and composite `CacheList` caches cannot be batched.

## Best Practices

```swift
// DO: Enable both scheduler and prompt cache together
container.scheduler = InferenceScheduler()
container.promptCache = LRUPromptCache(maxSize: 10)

// DO: Use applyRotaryPosition() in model implementations
queries = applyRotaryPosition(rope, to: queries, cache: cache)

// DO: Use cache.makeMask() for attention masks in models
let mask = cache.makeMask(n: h.dim(1), windowSize: nil, returnArray: false)

// DON'T: Use scalar rope offset in batched models
// rope(x, offset: cache.offset)  // Wrong for batch

// DON'T: Expect batching with VLMs
// Scheduler is bypassed for multimodal models
```
