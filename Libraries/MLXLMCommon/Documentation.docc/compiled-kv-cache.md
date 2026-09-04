# Compiled KV-cache decoding

Use ``CompiledDecodeSession`` when repeatedly invoking a compatible text model
from an `MLX.compile` decode closure.

## Why this is a separate cache

``KVCacheSimple`` is optimized for eager generation. It grows storage in chunks
and returns only the written prefix, so attention work tracks the actual context
length. Its write position is a Swift `Int`, however, and dynamic slices derived
from that value become constants when MLX traces a compiled function.

``FixedCapacityKVCache`` has the opposite tradeoff. It allocates its final shape
before compilation, carries position as an `MLXArray`, scatters new rows using
tensor indices, and masks unwritten capacity. This makes the graph reusable and
correct across decode steps, but attention spans the configured capacity even
when only part of it has been written. Keeping the implementations separate
preserves the efficient default for eager generation.

## Supported models

Only models conforming to ``FixedCapacityKVCacheProviding`` may use this cache.
The capability means the model has been audited and parity-tested so every
position-dependent forward operation remains in the MLX graph.

The initial supported implementations are:

- `Qwen2Model`, including Qwen 2 and Qwen 2.5 text checkpoints.
- `LlamaModel`, including Llama and the Mistral text checkpoints backed by that
  implementation.

Do not substitute this cache into vision-language, sliding-window, recurrent,
hybrid, or model-specific cache layouts. In particular, a similarly named model
is not implicitly compatible: `Mistral3TextModel` is distinct from `LlamaModel`.
Unsupported models continue using their standard eager cache.

## Usage

Size capacity for the prompt and the complete generation. The session performs
prefill before creating the compiled closure and rejects decode after capacity:

```swift
guard let cacheProvider = model as? any FixedCapacityKVCacheProviding else {
    // Fall back to the normal eager generation path.
    return
}

let maxNewTokens = 256
let capacity = promptTokens.dim(1) + maxNewTokens
let session = try CompiledDecodeSession(
    model: cacheProvider, prompt: promptTokens, capacity: capacity)

var token = nextToken(from: session.prefillLogits)
emit(token)
for _ in 1 ..< maxNewTokens {
    let logits = try session.step(token.reshaped(1, 1))
    token = nextToken(from: logits)
    emit(token)
}
```

``CompiledDecodeSession`` is the safe public execution path. It owns the host
token count because a compiled graph cannot raise a Swift error from a
tensor-valued position without synchronizing it to the CPU. Advanced callers
can construct ``FixedCapacityKVCache`` directly, but then they own prefill order
and capacity admission.
