# Wired Memory Policies and Tickets

## Overview

`MLXLMCommon` adds LLM-oriented wired-memory helpers on top of MLX manager/ticket primitives:

- Policies: `WiredSumPolicy`, `WiredMaxPolicy`, `WiredFixedPolicy`, `WiredBudgetPolicy`
- Measurement helpers: `WiredMemoryUtils` and `WiredMemoryMeasurement`
- Integration point: `wiredMemoryTicket` parameter on generation APIs

Core implementation:

- `Libraries/MLXLMCommon/WiredMemoryPolicies.swift`
- `Libraries/MLXLMCommon/WiredMemoryUtils.swift`
- `Libraries/MLXLMCommon/Evaluate.swift`
- `Libraries/MLXLMCommon/ModelContainer.swift`

## Ticket Flow

1. Choose a policy.
2. Create a ticket with estimated bytes.
3. Pass ticket to generation (`wiredMemoryTicket:`) or wrap work with `WiredMemoryTicket.withWiredLimit`.
4. Let manager coordinate limits/admission across concurrent work.

```swift
let policy = WiredSumPolicy(cap: 12 * 1024 * 1024 * 1024)
let ticket = policy.ticket(size: estimatedBytes, kind: .active)

let lmInput = try await modelContainer.prepare(input: UserInput(prompt: "Summarize this"))
let stream = try await modelContainer.generate(
    input: lmInput,
    parameters: GenerateParameters(),
    wiredMemoryTicket: ticket
)
```

## Active vs Reservation Tickets

- `.active`: contributes while inference is actively running.
- `.reservation`: tracks long-lived budgets (for example model weights) without keeping limit elevated when no active inference exists.

```swift
let reservation = policy.ticket(size: weightBytes, kind: .reservation)
let inference = policy.ticket(size: kvAndWorkspaceBytes, kind: .active)
```

## Policy Selection

- `WiredSumPolicy`: baseline + sum(activeSizes); common default for concurrent inference.
- `WiredMaxPolicy`: baseline vs largest active request.
- `WiredFixedPolicy`: constant limit when any work is active.
- `WiredBudgetPolicy`: baseline + base budget + active sizes; useful for measured budgets.

```swift
let fixed = WiredFixedPolicy(limit: 8 * 1024 * 1024 * 1024)
let maxPolicy = WiredMaxPolicy()
let budget = WiredBudgetPolicy(baseBytes: measuredWeightsPlusWorkspace, cap: nil)
```

## Measurement-Driven Budgeting

Use `WiredMemoryUtils.tune(...)` to measure real runtime costs and then size policy/tickets.

### Text-only Measurement

```swift
let context = try await LLMModelFactory.shared.load(configuration: config)
let parameters = GenerateParameters(maxTokens: 128, prefillStepSize: 512)

let measurement = try await WiredMemoryUtils.tune(
    context: context,
    tokenCount: 2048,
    parameters: parameters
)

let baseBytes = measurement.weightBytes + measurement.workspaceBytes
let policy = WiredBudgetPolicy(baseBytes: baseBytes)
let ticket = policy.ticket(size: measurement.kvBytes, kind: .active)
```

### Multimodal Measurement

For VLMs, include real media input so image/video tensors are counted.

```swift
let measurement = try await WiredMemoryUtils.tune(
    userInput: userInput,
    context: context,
    parameters: parameters
)
```

## CPU and Unsupported Backends

When wired limit control is unavailable, keep policy math and admission active:

```swift
await WiredMemoryManager.shared.updateConfiguration { configuration in
    configuration.policyOnlyWhenUnsupported = true
}
```

## Debug Event Stream

In DEBUG builds, observe manager events for policy stacking and limit changes.
In release builds, stream is a no-op.

```swift
Task {
    for await event in WiredMemoryManager.shared.events() {
        print(event)
    }
}
```

## Practical Guidance

- Keep weights, KV cache, and transient workspace as separate budgeting components.
- Start with measured values, then add a small safety margin.
- Use task-handle generation APIs when consumers may stop early under load.
