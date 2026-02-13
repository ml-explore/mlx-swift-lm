# Generation APIs

## Overview

Generation APIs in `MLXLMCommon` support two output modes:

- Decoded output (`Generation`): text chunks, tool calls, and final completion info.
- Raw token output (`TokenGeneration`): token IDs plus final completion info.

Primary implementation lives in `Libraries/MLXLMCommon/Evaluate.swift`.

## API Matrix

| API | Output | Task Handle | Typical Use |
|-----|--------|-------------|-------------|
| `generate(input:cache:parameters:context:)` | `AsyncStream<Generation>` | No | Standard decoded streaming |
| `generateTask(...)` | `AsyncStream<Generation>` | Yes | Early stop + deterministic cleanup |
| `generateTokens(input:cache:parameters:context:includeStopToken:)` | `AsyncStream<TokenGeneration>` | No | Raw token parsers |
| `generateTokensTask(...)` | `AsyncStream<TokenGeneration>` | Yes | Raw token parsing with cleanup control |
| `generateTokenTask(...)` | `AsyncStream<TokenGeneration>` | Yes | Low-level custom iterator pipelines |

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

## Stop Reasons

`GenerateStopReason` in final `.info` can be:

- `.stop`: EOS or stop token encountered.
- `.length`: `maxTokens` reached.
- `.cancelled`: task cancellation or early termination path.

## Throwing vs Non-Throwing Behavior

- API creation is `throws` (for example bad input/model state).
- Iteration over returned `AsyncStream` is non-throwing.
- `ChatSession.streamResponse(...)` is different: it returns `AsyncThrowingStream` and requires `for try await`.

## Practical Defaults

- Prefer `ChatSession` for standard chat UX.
- Prefer `generateTask`/`generateTokensTask` when consumers may stop early.
- Use raw token APIs only when you need token IDs directly.
