# Concurrency Patterns

## Overview

mlx-swift-lm uses Swift concurrency with specialized utilities to handle the unique constraints of ML workloads: non-Sendable `MLXArray` types, long-running computations, and thread-safe model access.  There is a `MaterializedArray` that is a subclass of MLXArray and it is sendable.  `MaterializedModule` can wrap a `Module` and provide Sendable access as well.

**File:** `Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `SerialAccessContainer<T>` | Exclusive async access to wrapped state |
| `AsyncMutex` | Lock that works with async blocks |
| `SendableBox<T>` | Transfer non-Sendable values across isolation |
| `MaterializedArray` | `Sendable` snapshot of an `MLXArray` (immutable, evaluated) |
| `MaterializedModule` | `Sendable` wrapper of a `Module` (materialized, sealed) |
| `ModelContext` / `EmbedderModelContext` | `Sendable` model bundles (model is a `MaterializedModule`) |
| `ChatSession` | NOT thread-safe (single task only) |

## SerialAccessContainer

Provides exclusive access to state across `async` calls:

```swift
// Unlike actors, this guarantees exclusive access for entire async operation
final class SerialAccessContainer<T>: @unchecked Sendable {
    func read<R>(_ body: (T) async throws -> R) async rethrows -> R
    func update<R>(_ body: (inout T) async throws -> R) async rethrows -> R
}
```

### Why Not Actor?

Actors release isolation at `await` points. `SerialAccessContainer` maintains the lock:

```swift
// Actor example - isolation released at await
actor MyActor {
    var state: Int = 0
    func process() async {
        state = 1
        await someAsyncWork()  // Another caller can modify state here!
        state = 2
    }
}

// SerialAccessContainer - exclusive for entire async operation
let container = SerialAccessContainer(0)
await container.update { state in
    state = 1
    await someAsyncWork()  // Exclusive access maintained
    state = 2
}
```

### Usage Pattern

```swift
let container = SerialAccessContainer(MyState())

// Read access
let value = await container.read { state in
    return state.someProperty
}

// Update access
await container.update { state in
    state.modify()
    await asyncOperation()  // Lock held through await
}
```

## SendableBox

Transfer non-Sendable values across isolation boundaries:

```swift
// Problem: LMInput is not Sendable
let iterator: IteratorProtocol = ...
Task {
    use(iterator)  // Compiler error!
}

// Solution: Use SendableBox
let box = SendableBox(iterator)
Task {
    let iterator = box.consume()  // Transfer ownership
    use(iterator)
}
```

### Important: Single Consume

```swift
let box = SendableBox(value)
let v1 = box.consume()  // OK
let v2 = box.consume()  // fatalError: "value already consumed"
```

## ChatSession Thread Safety

`ChatSession` is NOT thread-safe (it is a class and has mutable properties). Use from a single task:

```swift
// WRONG: Multiple tasks using same session
let session = ChatSession(context)
Task { await session.respond(to: "A") }  // Race condition!
Task { await session.respond(to: "B") }

// CORRECT: Single task per session
let session = ChatSession(context)
let r1 = await session.respond(to: "A")
let r2 = await session.respond(to: "B")

// Or: Separate sessions per task (ModelContext is Sendable, so share it freely)
Task {
    let session = ChatSession(context)  // Own session
    await session.respond(to: "...")
}
```

## AsyncStream Patterns

### Creating Generation Streams

```swift
// API creation can throw; stream iteration itself is non-throwing.
let stream = try generate(
    input: input,
    parameters: params,
    context: context
)

for await generation in stream {
    switch generation {
    case .chunk(let text): print(text)
    case .info(let info): print(info.tokensPerSecond)
    case .toolCall(let call): handleTool(call)
    }
}
```

### Throwing Boundaries

```swift
// ChatSession produces AsyncThrowingStream
for try await chunk in session.streamResponse(to: prompt) {
    print(chunk, terminator: "")
}

// Evaluate.generate produces AsyncStream
let stream = try generate(input: input, parameters: params, context: context)
for await event in stream {
    // non-throwing iteration
}
```

### Early Termination

```swift
// Breaking early still allows generation to continue briefly
// Use generateTask() for clean shutdown
let (stream, task) = generateTask(
    promptTokenCount: count,
    modelConfiguration: config,
    tokenizer: tokenizer,
    iterator: iterator
)

for await item in stream {
    if shouldStop {
        break
    }
}

// Wait for generation to fully stop
await task.value
```

### Raw Token Task Flow

```swift
let (tokenStream, tokenTask) = try generateTokensTask(
    input: input,
    parameters: params,
    context: context,
    includeStopToken: false
)

for await event in tokenStream {
    switch event {
    case .token(let tokenID):
        print(tokenID)
    case .info(let info):
        print("stop: \(info.stopReason)")
    }
}

await tokenTask.value
```

### Cancellation Handling

```swift
// Inside generation loops, check cancellation
for token in iterator {
    if Task.isCancelled {
        break
    }
    // process token
}

// Stream cancellation propagates
let task = Task {
    for await chunk in stream { ... }
}
task.cancel()  // Stream terminates
```

## MLXArray and Sendable

`MLXArray` is NOT `Sendable`. Strategies:

### 1. Use item() to extract a value:

```swift
let result = context.model(input)
eval(result)                 // Evaluate before crossing boundary
let value = result.item(Float.self)  // Return a primitive (Sendable)
```

### 2. Use MaterializedArray for Transfer

```swift
let snapshot = array.materialized()  // MaterializedArray: a Sendable snapshot
Task {
    // Use snapshot safely across the isolation boundary.
    // Operations on it still produce ordinary lazy MLXArray results.
    let doubled = snapshot * 2
    eval(doubled)
}
```

### 3. Keep Arrays Within Isolation

```swift
// Keep all array operations in the same isolation domain
let a = context.model(input1)
let b = context.model(input2)
let combined = a + b
eval(combined)
let value = combined.item(Float.self)
```

## MaterializedArray and MaterializedModule

These two `@unchecked Sendable` types are what let a `ModelContext` cross
isolation boundaries safely, so you can pass it directly instead of hiding a
model behind a `ModelContainer` actor.

### MaterializedArray

A subclass of `MLXArray` that holds a fully-evaluated, immutable snapshot.

```swift
// Create from an existing array
let snapshot = array.materialized()   // or: materialize(array)

// Usable anywhere an MLXArray is expected
let y = snapshot + 1                   // produces an ordinary lazy MLXArray
```

- It is `Sendable`, so it can be captured by another task/actor.
- It is immutable: attempting to mutate it in place traps.
- Operations on it produce ordinary lazy `MLXArray` results (only the snapshot
  itself is materialized).

### MaterializedModule

A `Sendable` wrapper around a `Module` (`MaterializedModule<LayerType: Module>`).

```swift
let sealed = MaterializedModule(module)
```

On init it evaluates and materializes all of the module's parameters and then
SEALS the module against mutation: `update`, `apply`, `freeze`, and `train`
all trap. Protocol conformances are added via extensions constrained on
`LayerType` (e.g. `extension MaterializedModule: LanguageModel where LayerType: LanguageModel`),
so a sealed language model is still usable for inference.

### Why this matters for ModelContext

`ModelContext` and `EmbedderModelContext` are now `Sendable` because their model
is a `MaterializedModule`. You use them directly across tasks and actors — no
`ModelContainer` actor and no `perform { }` block:

```swift
let context = try await LLMModelFactory.shared.load(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let input = try context.processor.prepare(input: UserInput(prompt: "Hello"))
let stream = try generate(input: input, parameters: .init(), context: context)
```

Because the model is sealed, you cannot mutate/train it or call
`context.model.update(parameters:)` (it traps). To modify weights or apply LoRA
adapters, load a mutable `TrainableModelContext` via `loadTrainable(...)` instead.

## Async Evaluation

MLX uses lazy evaluation. Force evaluation at boundaries:

```swift
// asyncEval() for pipelining
asyncEval(nextToken)  // Starts computation, doesn't wait

// eval() for immediate evaluation
eval(result)  // Waits for completion

// Stream synchronize
Stream().synchronize()  // Wait for all pending operations
```

## Task Cancellation Best Practices

```swift
// In generation loops
for try await generation in stream {
    // Check at each iteration
    if Task.isCancelled { break }

    // Process generation
}

// In custom iterators
mutating func next() -> Int? {
    // Guard against runaway generation
    if let maxTokens, tokenCount >= maxTokens {
        return nil
    }
    // ...
}
```

## Deprecated Patterns

### Callback-based generate()

```swift
// DEPRECATED: Callback-based generation
generate(
    input: input,
    parameters: params,
    context: context,
    didGenerate: { token in
        // handle token
        return .more  // or .stop
    }
)

// USE INSTEAD: AsyncStream-based
let stream = try generate(input: input, parameters: params, context: context)
for await generation in stream {
    // handle generation
}
```

The callback API:
- Blocks the calling thread
- Harder to cancel cleanly
- Less idiomatic Swift concurrency

### Old generate() without task handle

```swift
// DEPRECATED: No way to wait for cleanup
let stream = generate(input: input, context: context, iterator: iterator)

// USE INSTEAD: generateTask() returns both stream and task
let (stream, task) = generateTask(
    promptTokenCount: count,
    modelConfiguration: config,
    tokenizer: tokenizer,
    iterator: iterator
)

// Can await task completion
await task.value
```
