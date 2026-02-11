# KV Cache System

## Overview

The KV (Key-Value) cache stores attention key and value tensors from previous tokens, enabling efficient autoregressive generation. Different cache types trade off between memory usage, context length, and performance.

## Quick Reference

| Type | Use Case | Memory | Max Context |
|------|----------|--------|-------------|
| `KVCacheSimple` | Default, unbounded | Grows with context | Unlimited |
| `RotatingKVCache` | Long contexts | Fixed | `maxKVSize` |
| `QuantizedKVCache` | Memory-constrained | 4-8x less | Unlimited |
| `ChunkedKVCache` | Large prompt processing | Controlled | Chunked |
| `MambaCache` | Mamba/SSM models | Fixed state | N/A |

**File:** `Libraries/MLXLMCommon/KVCache.swift`

## Cache Types

### KVCacheSimple (Default)

Unbounded cache that grows with context length:

```swift
// Created automatically when no maxKVSize specified
let params = GenerateParameters()  // Uses KVCacheSimple
let cache = KVCacheSimple()

// Properties
cache.offset      // Current position in cache
cache.state       // [keys, values] for serialization
cache.isTrimmable // true
```

### RotatingKVCache (Sliding Window)

Fixed-size cache with sliding window attention:

```swift
// Enable via GenerateParameters
let params = GenerateParameters(maxKVSize: 4096)

// Or create directly
let cache = RotatingKVCache(
    maxSize: 4096,  // Window size
    keep: 4,        // Tokens to always keep at start
    step: 256       // Allocation step size
)

// Behavior:
// - First 4 tokens always kept
// - After hitting maxSize, oldest tokens (except kept) are overwritten
// - Offset continues growing, but actual cache size is capped
```

### QuantizedKVCache

Memory-efficient cache using 4-bit or 8-bit quantization:

```swift
// Enable via GenerateParameters
let params = GenerateParameters(
    kvBits: 4,           // 4 or 8 bits
    kvGroupSize: 64,     // Quantization group size
    quantizedKVStart: 0  // Start quantizing after N tokens
)

// Or create directly
let cache = QuantizedKVCache(
    groupSize: 64,
    bits: 4,
    mode: .affine
)

// Use updateQuantized() instead of update()
let (qKeys, qValues) = cache.updateQuantized(keys: keys, values: values)
// qKeys = (weight, scales, biases?)
// qValues = (weight, scales, biases?)
```

### Dynamic Cache Quantization

Caches can be converted during generation:

```swift
// Simple cache converts to quantized after threshold
var cache: [KVCache] = model.newCache(parameters: nil)

// This happens automatically inside TokenIterator when:
// - kvBits is set
// - cache offset > quantizedKVStart
maybeQuantizeKVCache(
    cache: &cache,
    kvBits: 4,
    kvGroupSize: 64,
    quantizedKVStart: 0
)

// Manual conversion (KVCacheSimple only)
let simpleCache = KVCacheSimple()
// ... use cache ...
let quantizedCache = simpleCache.toQuantized(groupSize: 64, bits: 4)

// Convert back
let simpleAgain = quantizedCache.toUnquantized()
```

**Important:** `RotatingKVCache.toQuantized()` is **not implemented** and will `fatalError()`. The temporal ordering of a rotating cache makes quantization complex. If you need both sliding window and quantization, use `KVCacheSimple` with quantization and manage context length manually.

## Creating Caches

### Via Model

```swift
// Models create appropriate cache for their architecture
let cache = model.newCache(parameters: generateParameters)
```

### Via Utility Functions

```swift
// From model (recommended)
let cache = makePromptCache(model: model, parameters: params)

// With known layer count
let cache = makePromptCacheWithLayerCount(
    numLayers: 32,
    maxKVSize: 4096  // nil for unbounded
)
```

## Cache Operations

### Trimming

Remove tokens from the end of cache:

```swift
// Check if trimmable
if canTrimPromptCache(cache) {
    // Trim last 10 tokens
    let trimmed = trimPromptCache(cache, numTokens: 10)
}

// Direct trim
cache.first?.trim(10)
```

### Serialization

Save and load prompt cache for reuse:

```swift
// Save
try savePromptCache(
    url: fileURL,
    cache: cache,
    metadata: ["prompt": "My cached prompt"]
)

// Load
let (loadedCache, metadata) = try loadPromptCache(url: fileURL)
```

Cache files are `.safetensors` format with metadata.

## Attention Masks

Caches create appropriate attention masks:

```swift
// Modern API - cache creates its own mask
let mask = cache.makeMask(
    n: sequenceLength,
    windowSize: nil,  // Or specific window
    returnArray: false  // .causal vs .array
)

// Helper function
let mask = makeAttentionMask(
    n: n,
    cache: cache,
    windowSize: nil,
    returnArray: false
)

// Returns MLXFast.ScaledDotProductAttentionMaskMode:
// .none - no mask needed (single token)
// .causal - symbolic causal mask
// .array(MLXArray) - explicit mask array
```

## Memory Considerations

### Memory Usage by Cache Type

| Cache Type | Memory per Token | Example (8K context, 32 layers) |
|------------|------------------|--------------------------------|
| KVCacheSimple (fp16) | Full | ~512MB |
| RotatingKVCache | Fixed at maxKVSize | Capped at maxKVSize |
| QuantizedKVCache (4-bit) | ~1/4 of fp16 | ~128MB |

### Best Practices

```swift
// For chat applications with long history
let params = GenerateParameters(
    maxKVSize: 4096,  // Sliding window
    kvBits: 4         // Quantized
)

// For short interactions (no memory pressure)
let params = GenerateParameters()  // Simple unbounded cache

// Clear cache when conversation resets
await session.clear()
```

## Quantized Attention

Use with QuantizedKVCache for efficient attention:

```swift
if let qCache = cache as? QuantizedKVCacheProtocol {
    let (qKeys, qValues) = qCache.updateQuantized(keys: keys, values: values)

    let output = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: qKeys,
        quantizedValues: qValues,
        scale: scale,
        mask: .none,
        groupSize: qCache.groupSize,
        bits: qCache.bits
    )
}
```

## State Space Model Caches

### MambaCache

For Mamba/SSM architecture models:

```swift
let cache = MambaCache(leftPadding: nil)

// Access via subscript
cache[0] = convState
cache[1] = ssmState

// Create mask
let mask = cache.makeMask(N: sequenceLength)
```

### CacheList

Composite cache for hybrid architectures:

```swift
let cache = CacheList(kvCache, mambaCache)
let kv = cache[0] as! KVCacheSimple
let mamba = cache[1] as! MambaCache
```

## Deprecated Patterns

### Old createAttentionMask signature

```swift
// DEPRECATED: Array of caches
func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool)

// USE INSTEAD: Single cache with windowSize
func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,       // Single cache
    windowSize: Int?,      // Explicit window
    returnArray: Bool
) -> MLXFast.ScaledDotProductAttentionMaskMode

// Or use cache's method directly
cache.makeMask(n: n, windowSize: windowSize, returnArray: false)
```

### Direct cache.update() on QuantizedKVCache

```swift
// WRONG: QuantizedKVCache.update() will fatalError
let (k, v) = quantizedCache.update(keys: keys, values: values)  // Crashes!

// CORRECT: Use updateQuantized()
let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
```
