# Concurrency Patterns

## Overview

mlx-swift-lm uses Swift concurrency with specialized utilities to handle the unique constraints of ML workloads: non-Sendable `MLXArray` types, long-running computations, and thread-safe model access.

**File:** `Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `SerialAccessContainer<T>` | Exclusive async access to wrapped state |
| `AsyncMutex` | Lock that works with async blocks |
| `SendableBox<T>` | Transfer non-Sendable values across isolation |
| `ModelContainer` | Thread-safe model wrapper (uses SerialAccessContainer) |
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
let input: LMInput = ...
Task {
    use(input)  // Compiler error!
}

// Solution: Use SendableBox
let box = SendableBox(input)
Task {
    let input = box.consume()  // Transfer ownership
    use(input)
}
```

### Pattern: Consuming Parameters

```swift
func processAsync(input: consuming LMInput) async throws -> Result {
    let boxed = SendableBox(input)

    return try await container.read { context in
        let input = boxed.consume()  // Consume inside closure
        return try process(input, context: context)
    }
}
```

### Important: Single Consume

```swift
let box = SendableBox(value)
let v1 = box.consume()  // OK
let v2 = box.consume()  // fatalError: "value already consumed"
```

## ModelContainer Thread Safety

`ModelContainer` uses `SerialAccessContainer` internally:

```swift
public final class ModelContainer: Sendable {
    private let context: SerialAccessContainer<ModelContext>

    // Thread-safe access
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> R
    ) async rethrows -> R
}
```

### Safe Usage

```swift
// Multiple tasks can call perform() safely
let container = try await loadModelContainer()

Task {
    await container.perform { context in
        // Exclusive access to model
    }
}

Task {
    await container.perform { context in
        // Waits for first task to complete
    }
}
```

## ChatSession Thread Safety

`ChatSession` is NOT thread-safe. Use from a single task:

```swift
// WRONG: Multiple tasks using same session
let session = ChatSession(container)
Task { await session.respond(to: "A") }  // Race condition!
Task { await session.respond(to: "B") }

// CORRECT: Single task per session
let session = ChatSession(container)
let r1 = await session.respond(to: "A")
let r2 = await session.respond(to: "B")

// Or: Separate sessions per task
Task {
    let session = ChatSession(container)  // Own session
    await session.respond(to: "...")
}
```

## AsyncStream Patterns

### Creating Generation Streams

```swift
// Generate returns an AsyncStream
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

### 1. Eval Before Returning

```swift
await container.perform { context in
    let result = context.model(input)
    eval(result)  // Evaluate before crossing boundary
    return result.item(Float.self)  // Return primitive
}
```

### 2. Use SendableBox for Transfer

```swift
let box = SendableBox(array)
Task {
    let array = box.consume()
    // Use array in this task only
}
```

### 3. Keep Arrays Within Isolation

```swift
// All array operations in same perform block
await container.perform { context in
    let a = model(input1)
    let b = model(input2)
    let combined = a + b
    eval(combined)
    return combined.item()
}
```

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
for await generation in generate(input: input, parameters: params, context: context) {
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
