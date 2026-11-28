# Dispatching tool calls

Choose when and how a chat session runs the tools a model calls.

With `toolDispatch`, ``ChatSession`` runs each accepted tool call, adds the
results to the conversation, and lets the model answer them. By default it runs
the calls of a model step one at a time, after the step ends.
``ToolCallPolicy/dispatch`` changes that:

```swift
let session = ChatSession(
    model,
    generateParameters: .init(toolCallPolicy: .init(dispatch: [.whenWritten, .concurrent])),
    tools: [weatherTool, newsTool],
    toolDispatch: { call in try await runTool(call) })
```

## Starting calls early

``ToolDispatchOptions/whenWritten`` starts a call as soon as the model finishes
writing it, while the step keeps generating. When a step makes several calls,
the first ones run while the model writes the rest. A tool that finishes before
the step ends adds no wait at all.

The call starts when the parser has the whole call. Formats that close each call
with an end marker, such as `<tool_call>…</tool_call>`, deliver it right away.
A format without one delivers it when the step ends, as it would without the
option.

A tool that starts early runs while the GPU decodes. A tool that runs its own
MLX work, such as an embedding model, competes with decoding for the GPU.

## Running calls together

``ToolDispatchOptions/concurrent`` runs the calls of a step at the same time, so
the step waits for the slowest call instead of the sum of all calls. Use it when
no call depends on the effects of another. A model can write a call that relies
on an earlier one in the same step, so tools with side effects are safer without
it.

The options combine. `[.whenWritten]` alone keeps calls one at a time and in
order, and still overlaps them with generation.

## Prefilling results early

The step that answers tool results has to prefill them first, and a long result
can take longer to prefill than the tools took to run.
``ToolDispatchOptions/prefillResult`` prefills each result as soon as it
returns, while the later calls still run, so that step prefills only the last
result.

The session renders the conversation twice: once ending with the results so far,
and once with a placeholder for the next call. Only the tokens the two renders
share are prefilled, so a later result cannot change them. The same cache-reuse
rules as a normal step decide whether this is safe. The session skips the early
prefill, and the results prefill after the step as before, when:

- the response format keeps tokens in the cache that the template does not
  render, as Harmony and Onyx do;
- the conversation carries images, videos, or audio, or the model carries
  per-call state;
- the session uses speculative decoding;
- the KV cache could not rewind the prefilled tokens, for example a rotating
  cache that has already wrapped.

The early prefill uses the GPU while tools run, so tools that run their own MLX
work compete with it.

### Running tools yourself

An application that runs tools itself, without `toolDispatch`, gets the same
early prefill from ``ChatSession/prefill(toolResults:before:)``. Call it after
each result except the last, with the results so far and the next call still
running, then send all the results as usual:

```swift
// The response ended with calls; the application runs them.
let rome = Chat.Message.tool(romeOutput, id: calls[0].id, name: calls[0].function.name)
try await session.prefill(toolResults: [rome], before: calls[1])

// Later, when every call has returned:
for try await event in session.streamDetails(to: [rome, paris]) { … }
```

Send the same result messages in the same order, or the response trims what
the prefill added. The method returns the number of tokens it prefilled, and 0
when it skips for the reasons above.

## Results and the KV cache

Results always enter the conversation in call order, after the step that wrote
the calls, whatever order the tools finish in. Templates that pair results with
calls by position stay correct, and the next step prefills only the results: the
cache never has to rewind to place a late result.

## Failures and cancellation

The first call in call order that throws ends the response, as before. Results
of the calls before it stay in the transcript. The calls after it are cancelled
if they are still running, and their results are discarded. With
``ToolDispatchOptions/whenWritten`` or ``ToolDispatchOptions/concurrent``, a
later call may already have finished, so its side effects may have happened.

A step with a rejected call throws ``RejectedToolCallError``. By default none of
its calls run. With ``ToolDispatchOptions/whenWritten`` the rejection cancels the
calls already running and starts no more, but a call may have done some work by
then. Set ``ToolCallPolicy/validation`` to ``ToolCallValidationPolicy/strict`` to
reject calls whose arguments violate their schema before they start.

Cancelling the response cancels its running calls. Tools must cooperate with
Swift task cancellation.
