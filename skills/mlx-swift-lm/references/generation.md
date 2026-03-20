# Generation APIs

## Overview

Generation APIs in `MLXLMCommon` support two output modes:

- Decoded output (`Generation`): text chunks, tool calls, and final completion info.
- Raw token output (`TokenGeneration`): token IDs plus final completion info.

Primary implementation lives in `Libraries/MLXLMCommon/Evaluate.swift`.

## API Matrix

### Free Functions (Evaluate.swift)

| API | Output | Task Handle | wiredMemoryTicket | Typical Use |
|-----|--------|-------------|-------------------|-------------|
| `generate(input:cache:parameters:context:)` | `AsyncStream<Generation>` | No | Yes | Standard decoded streaming |
| `generateTask(...)` | `AsyncStream<Generation>` | Yes | Yes | Early stop + deterministic cleanup |
| `generateTokens(input:cache:parameters:context:includeStopToken:)` | `AsyncStream<TokenGeneration>` | No | Yes | Raw token parsers |
| `generateTokensTask(...)` | `AsyncStream<TokenGeneration>` | Yes | Yes | Raw token parsing with cleanup control |
| `generateTokenTask(...)` | `AsyncStream<TokenGeneration>` | Yes | Yes | Low-level custom iterator pipelines |

### ModelContainer Methods

| API | Output | Routes Through Scheduler | Typical Use |
|-----|--------|--------------------------|-------------|
| `container.generate(input:parameters:wiredMemoryTicket:)` | `AsyncStream<Generation>` | Yes (when scheduler set) | High-level decoded streaming |
| `container.generateTokens(input:parameters:includeStopToken:wiredMemoryTicket:)` | `AsyncStream<TokenGeneration>` | Yes (when scheduler set) | High-level raw token streaming |

### InferenceScheduler Methods

| API | Output | Typical Use |
|-----|--------|-------------|
| `scheduler.submit(input:parameters:model:cache:tokenizer:configuration:...)` | `AsyncStream<Generation>` | Batched decoded streaming |
| `scheduler.submitTokens(input:parameters:model:cache:tokenizer:configuration:...)` | `AsyncStream<TokenGeneration>` | Batched raw token streaming |

## Decoded Text/Tool Streaming

```swift
import MLXLLM
import MLXLMCommon

let context = try await LLMModelFactory.shared.load(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let userInput = UserInput(prompt: "Explain actor isolation in Swift.")
let lmInput = try await context.processor.prepare(input: userInput)
let params = GenerateParameters(maxTokens: 300, temperature: 0.6)

let stream = try generate(input: lmInput, parameters: params, context: context)
for await event in stream {
    switch event {
    case .chunk(let text):
        print(text, terminator: "")
    case .toolCall(let call):
        print("\nTool requested: \(call.function.name)")
    case .info(let info):
        print("\nstop=\(info.stopReason) tok/s=\(info.tokensPerSecond)")
    }
}
```

## Task-Handle Pattern for Early Stop

If you break out of a stream loop early, underlying generation may continue for a short period.
Use `generateTask(...)` and await `task.value` for deterministic completion.

```swift
let iterator = try TokenIterator(
    input: lmInput,
    model: context.model,
    cache: nil,
    parameters: params
)

let (stream, task) = generateTask(
    promptTokenCount: lmInput.text.tokens.size,
    modelConfiguration: context.configuration,
    tokenizer: context.tokenizer,
    iterator: iterator
)

for await event in stream {
    if shouldStopEarly, case .chunk = event {
        break
    }
}

await task.value
```

## Raw Token Streaming

Use raw token APIs for custom parsers or token-level instrumentation.

```swift
let (tokenStream, tokenTask) = try generateTokensTask(
    input: lmInput,
    parameters: params,
    context: context,
    includeStopToken: false
)

for await event in tokenStream {
    switch event {
    case .token(let tokenID):
        print(tokenID)
    case .info(let info):
        print("stop=\(info.stopReason)")
    }
}

await tokenTask.value
```

### With Wired Memory Coordination

```swift
// With wired memory coordination:
let ticket = WiredSumPolicy().ticket(size: estimatedBytes, kind: .active)
let (tokenStream, tokenTask) = try generateTokensTask(
    input: lmInput,
    parameters: params,
    context: context,
    wiredMemoryTicket: ticket
)
```

## Stop Reasons

`GenerateStopReason` in final `.info` can be:

- `.stop`: EOS or stop token encountered.
- `.length`: `maxTokens` reached.
- `.cancelled`: task cancellation or early termination path.

## Throwing vs Non-Throwing Behavior

- API creation is `throws` (for example bad input/model state).
- Iteration over returned `AsyncStream` is non-throwing.
- `ChatSession.streamResponse(...)` is different: it returns `AsyncThrowingStream` and requires `for try await`.

## Batched Generation

When `ModelContainer.scheduler` is set, both `generate()` and `generateTokens()` transparently route through the `InferenceScheduler`, enabling continuous batching of concurrent requests.

```swift
// Enable batching on the container
container.scheduler = InferenceScheduler()
container.promptCache = LRUPromptCache(maxSize: 10)

// Multiple concurrent requests are automatically batched
async let stream1 = container.generate(input: input1, parameters: params)
async let stream2 = container.generate(input: input2, parameters: params)

// Raw token batching also supported
async let tokens1 = container.generateTokens(input: input1, parameters: params)
async let tokens2 = container.generateTokens(input: input2, parameters: params)
```

The scheduler can also be used directly:

```swift
let scheduler = InferenceScheduler()
let stream = try await scheduler.submit(
    input: lmInput, parameters: params,
    model: model, cache: nil,
    tokenizer: tokenizer, configuration: config
)
```

See [batching.md](batching.md) for full details on the scheduler state machine, batch caches, and prompt caching.

## Practical Defaults

- Prefer `ChatSession` for standard chat UX.
- Prefer `generateTask`/`generateTokensTask` when consumers may stop early.
- Use raw token APIs only when you need token IDs directly.
- Set `container.scheduler` when serving multiple concurrent users/requests.
