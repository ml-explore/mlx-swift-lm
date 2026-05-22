# Continuous Batching

`BatchGenerator` provides a low-level continuous-batching engine for token
generation. It batches prompt prefill work, keeps decode rows in a shared
batched cache, and admits new prompts as existing rows finish.

Use it when you need explicit control over request admission and per-row
streaming responses:

```swift
let generator = BatchGenerator(
    model: model,
    eosTokens: [[eosToken]],
    defaultMaxTokens: 128,
    prefillBatchSize: 8,
    completionBatchSize: 32
)

let ids = generator.insert(prompts: [[1, 2, 3], [4, 5]])

while generator.hasWork {
    for response in generator.next() {
        print(response.uid, response.token, response.finishReason as Any)
    }
}
```

The engine is intentionally stateful and should be driven from one execution
context at a time. Each call to `next()` may prefill queued prompts, run one
decode step for active rows, and return one response per active row. A response
with a non-`nil` `finishReason` is the final response for that UID.

Call `cancel(uid:)` to remove a queued or active row. The method returns `true`
when it found the UID and filtered that row out of the generator state.

## Custom Sampling

Pass per-row `RowSampler` values to `insert(prompts:maxTokens:samplers:)` to
mix greedy and probabilistic decoding inside the same generation batch. Use
`makeRowSampler(temperature:topP:topK:seed:)` for OpenAI-style temperature,
top-p, top-k, and seeded categorical sampling.

## Stop Sequences

`SequenceStateMachine` detects single- or multi-token stop sequences per row.
The `BatchGenerator` initializer accepts default EOS token sequences, and
`insert(prompts:maxTokens:samplers:stateMachines:)` can override stop logic per
request.

## Cache Model

Continuous batching uses `BatchKVCache` for attention layers and `ArraysCache`
or `MambaCache` for array-backed state-space layers. `BatchKVCache` keeps rows
right-aligned so requests with different prompt lengths can share one cache and
attention mask while still preserving per-row positions for RoPE.
