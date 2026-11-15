# Steering an active response

Add instructions while a response runs.

Keep your normal streaming loop:

```swift
let response = session.streamResponse(to: "Explain this implementation")
let consumer = Task {
    for try await text in response {
        print(text, terminator: "")
    }
}

// From a later UI action, on the session's owning actor:
try session.steer("Focus on memory usage")
try session.steer("Include an example", policy: .nextStepBoundary)
```

``ChatSession/steer(_:policy:response:)`` also works with `respond`,
`streamDetails`, and `observe`. `ChatSession` remains owned by the caller's isolation domain; it
is not Sendable. The response stream can be passed to its consumer task.

## Acceptance and application

`steer` returns an acceptance UUID immediately, without waiting for inference.
Acceptance means the instruction is queued, not that the model has acted on it.

Acceptance fails, and throws, when the session has no response accepting input
(``SteeringError/noActiveResponse``), when the named response has already ended
(``SteeringError/responseEnded``), when the response cannot be steered
(``SteeringError/notSteerable``), or when the queue is full
(``SteeringError/queueFull``). A throw from `steer` never affects a running
response. Use ``ChatSession/canSteer(_:)`` to enable UI without catching.

By default `steer` targets the response that owns the session cache, then the
oldest response still accepting input. A session that can have more than one
response outstanding should name its target, which is never retargeted:

```swift
let stream = session.streamDetails(to: "Explain this implementation")
let response = session.latestResponse
// ...
try session.steer("Focus on memory usage", response: response)
```

Use `streamDetails` when you need to know what happened to an instruction.
`observe` streams deliver the same events through
``GenerationProviding/generation``:

```swift
for try await event in session.streamDetails(to: "Explain this implementation") {
    switch event {
    case .steering(.applied(let ids)):
        acknowledge(ids)
    case .steering(.failed(let failure)):
        report(failure.instructions, failure.reason)
    case .chunk(let text):
        display(text)
    case .reasoning(let text):
        displayThinking(text)
    case .info(let info):
        recordStatistics(info)
    case .toolCall(let call):
        handleTool(call)
    case .rejectedToolCall(let rejection):
        handleRejection(rejection)
    }
}
```

``SteeringEvent/applied(_:)`` identifies instructions included in a
successful successor prefill. It confirms delivery to the model, not that the
model followed the instruction. ``SteeringEvent/failed(_:)`` reports
instructions the response accepted but could not apply, with the original text so
you can resubmit them. **A steering failure never ends the response**: the output
already generated is delivered and the stream finishes normally. Only a model,
tool, or preparation error ends a response. If the stream throws or is cancelled,
accepted instructions without an outcome must be treated as unresolved; they are
not replayed automatically.

Stream completion ends the whole response; `.info` is emitted separately for each
model step.

## Scheduling

- ``SteeringPolicy/nextSafeBoundary`` is the default. Decoding stops at the next
  safe point, in the answer or inside a thought: after complete text that does
  not end in whitespace, which chat templates commonly trim. Partial Unicode,
  stop-string prefixes, partial reasoning delimiters, and incomplete tool payloads
  delay the handoff. The step reports ``GenerateStopReason/steered``.
- ``SteeringPolicy/nextStepBoundary`` lets the current model step finish
  naturally. Use it for custom reasoning or output grammars without declared
  protocol metadata.

Prefill and dispatched tools finish before an instruction applies, even with
`.nextSafeBoundary`. Framed Harmony and Onyx protocols stop inside an analysis,
private, or visible frame, never in a header or a tool frame. Steering does not
preempt GPU work or undo tool effects. Already emitted text is retained.

Instructions are applied in acceptance order. A pending batch is joined with
blank lines into one user message to support alternating-role templates. A
`.nextSafeBoundary` request promotes the whole pending batch. The queue holds at
most ``SteeringLimits/maxPendingInstructions`` instructions and
``SteeringLimits/maxPendingBytes`` bytes.

Generation limits apply per model step, as they do for the tool loop. A response
steered twice can generate up to three times `maxTokens`.

## Interrupting a response

Cancellation discards the turn: the prompt and the partial reply leave the
transcript, and the next call prefills the conversation again. To stop a response
and keep what it wrote, interrupt it instead:

```swift
try session.interrupt()
```

The response ends at its next safe point, as a steered step would, and its stream
finishes normally. The step it ends reports ``GenerateStopReason/interrupted``,
and instructions it did not apply are reported as failed. The transcript keeps the
reply up to that point, so the next call continues the same cache. With
`toolDispatch`, calls the model already made run first, and the step that
answers them stops at its first safe point.

## Tools, cancellation, and cache reuse

With `toolDispatch`, steering waits for tool results and the next assistant
output before adding the instruction. This keeps tool results and user instructions
from becoming consecutive user turns in strict templates. Automatically dispatched calls
remain hidden from the public stream, as in ordinary generation. Without a
dispatcher, ordinary tool handling still works; a pending instruction that needs
tool results is reported as ``SteeringError/toolResultsRequired`` and the
response completes with its tool calls. Submit the results with the instruction
through the normal session message API.

Cancel the stream's consumer task to cancel generation and pending instructions.
Await ``ChatSession/synchronize()`` to join registered runners and GPU cleanup.
Tools must cooperate with Swift task cancellation.

The model stays loaded. The session reuses a verified cache prefix and prefills
the changed suffix when the template and cache permit it. A thought that steering
or an interrupt cuts short is recorded as the model wrote it, so a template that
replays reasoning renders what the cache holds. Speculative lookahead is
finalized before reconciliation. Rewritten templates, unrewindable caches, and
models that carry per-call state (M-RoPE VLMs) require a rebuild instead.
Inspect ``GenerateCompletionInfo/cachedPromptTokenCount`` for actual reuse.

Steering accepts text and requires a structured conversation. Ordinary generation
from a raw restored cache still works; that response reports
``SteeringError/notSteerable`` and rejects later instructions up front. A model
step that produced no recordable assistant output cannot establish a new message
boundary and reports ``SteeringError/emptyResponse``. Local models follow
instructions through their normal chat templates and weights.

Existing string consumers need no changes. Exhaustive switches over `Generation`
need the new `.steering` case, and those over `GenerateStopReason` need
`.steered` and `.interrupted`.
