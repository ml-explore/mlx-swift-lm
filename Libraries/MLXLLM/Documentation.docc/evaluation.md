#  Evaluation

The simplified LLM/VLM API allows you to load a model and evaluate prompts with only a few lines of code.

For example, this loads a model and asks a question and a follow-on question:

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace  // macros: #hubDownloader / #huggingFaceTokenizerLoader
import HuggingFace
import Tokenizers

let model = try await loadModel(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

The second question actually refers to information (the location) from the first
question -- this context is maintained inside the `ChatSession` object.

If you need a one-shot prompt/response simply create a `ChatSession`, evaluate
the prompt and discard.  Multiple `ChatSession` instances could also be used
(at the cost of the memory in the `KVCache`) to handle multiple streams of
context.

## Prefill Optimization

Chunked text prefill reserves standard KV-cache capacity for the prompt and cached
prefix, avoiding repeated growth and copies. Subsequent growth is proportional
and bounded. Cache positions, attention masks, and saved state are unchanged;
custom, rotating, compressed, and recurrent caches keep their allocation paths.

Projection routing is opt-in. The default `.stock` policy skips module discovery
and replacement, including on a model previously used with `.automatic`:

```swift
let parameters = GenerateParameters(
    prefill: .init(stepSize: 4096, quantizedProjections: .automatic)
)
```

On Apple Silicon GPUs, shared text preparation can route standard frozen FP16
and BF16 affine projections through temporary dequantization and dense GEMM.
MLX selects the device kernels for both routes, including quantized NAX where
available; routing has no GPU-family filter, runtime calibration, or retained
dense weights. The route requires at least
384 rows for two-bit weights and 2048 rows for three-, four-, five-, six-, and
eight-bit weights, with groups of 32, 64, or 128.
Each dense projection is limited to 1/64 of the remaining allocator budget,
using the smaller of Metal's recommended working set and the MLX memory limit.
This is an allowance at graph construction, not a total transient-memory bound;
pending graphs and other processes can also allocate memory. CPU execution,
other formats, training, and decoding use stock projection kernels.
Source views owned by an existing fused projection keep their fused route.
Routing is scoped to the shared driver's prefill forwards. Retained wrappers use
stock kernels outside that scope, including later unchunked requests. Managed
compiled regions retain stock projections and their existing fusion. Dynamic
routing outside those regions checks memory admission on each forward.

With `.automatic`, eligible calls may route through dense GEMM. The 512-token
default ceiling and caller-selected chunk schedules remain unchanged; at that
ceiling, only two-bit projections can meet the row threshold. Wider chunks can be
requested through `PrefillParameters.stepSize` or `GenerateParameters.prefillStepSize`.
Spare cache capacity and temporary workspace can increase memory use;
performance gains depend on the model, device, and actual chunk length. Measure
your workload before enabling routing. The row thresholds are heuristics, not a
guarantee of a speedup on every Apple Silicon generation.

A rejected module replacement disables routing for that request after refreshing
module metadata and invalidating affected traces. Earlier coherent wrappers may
remain installed and use stock kernels. Cancellation still propagates. If a custom
module also rejects the metadata refresh, preparation throws rather than continuing
with stale module state. Custom setters must leave the model usable when they throw.

Admission checks current allocator headroom, which excludes future lazy allocations
such as reserved cache storage. Use `.stock` to avoid dense workspace under memory
pressure. Cancelling prefill leaves only written tokens in the live cache, but its
reserved backing capacity can remain allocated until that cache is released.

Dense and quantized kernels can round differently. Validation compares projection
error against an FP32 dequantization-and-matmul reference, using a bound
of `1.5 * stockMaxError + 0.005`. These are projection regression limits,
not a bound on accumulated model error. Multi-layer tests also compare logits
and cache state against an FP32 reconstruction, with error scaled by output
magnitude and precision. Bit-identical logits, KV-cache values, and
generated tokens are not guaranteed; token choices near a tie can change.

## Streaming Output

The previous example produced the entire response in one call.  Often
users want to see the text as it is generated -- you can do this with
a stream:

```swift
let model = try await loadModel(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)

for try await item in session.streamResponse(to: "Why is the sky blue?") {
    print(item, terminator: "")
}
print()
```

## Structured Chat Continuation

`ChatSession` can also continue from structured `Chat.Message` values. This
is useful for agent loops that consume tool calls from `streamDetails(to:role:images:videos:)`
and then append one or more `.tool` messages without rebuilding the whole
conversation history:

```swift
var pendingToolCalls: [ToolCall] = []
var rejectedToolCall: RejectedToolCall?

for try await item in session.streamDetails(
    to: "What is the weather in Paris?",
    images: [],
    videos: []
) {
    if case .toolCall(let toolCall) = item {
        pendingToolCalls.append(toolCall)
    }
    if case .rejectedToolCall(let rejection) = item {
        rejectedToolCall = rejection
    }
}

if let rejectedToolCall {
    throw RejectedToolCallError(rejectedToolCall)
}

var toolResults: [Chat.Message] = []
for toolCall in pendingToolCalls {
    let toolResult = try await callTool(toolCall)
    toolResults.append(.tool(toolResult))
}

if !toolResults.isEmpty {
    let answer = try await session.respond(to: toolResults)
    print(answer)
}
```

When tool schemas are supplied, `Generation.toolCall` contains only parsed and
authorized calls that may be considered for dispatch. Tool-call-shaped output
that is malformed, incomplete, exceeds the parser's bounded safety limit,
or names an undeclared function is emitted separately as
`Generation.rejectedToolCall`; rejected protocol is never returned as a normal
response chunk. `rawTextPreview` is bounded for diagnostics but can contain
sensitive argument values, so applications should not log or persist it
automatically.

Cross-dialect recovery defaults to `ToolCallRecoveryPolicy.conservative`. It
accepts only structurally complete calls naming an exactly declared tool and
keeps syntax inside reasoning spans, Markdown code, and ordinary JSON data
inert. Set `GenerateParameters.toolCallPolicy.recovery` to `.disabled` to permit
only the selected native dialect, or `.permissive` to allow the documented
end-of-stream outer-close repair. `GenerateCompletionInfo` reports both
`recoveredToolCallCount` and `rejectedToolCallCount` for production telemetry.

`GenerateParameters.toolCallPolicy.validation` defaults to `.permissive`, leaving
schema validation to the application. Set it to `.strict` to reject proven schema
violations, including missing required arguments. Declared-tool authorization
and native parser requirements apply in both modes.

The example buffers accepted calls until the generation finishes. This makes
dispatch atomic at the turn level: if a later call in the same model output is
rejected, no earlier call has already caused an external side effect.
`ChatSession` automatic dispatch also fails closed when tool schemas are absent
or empty: no model-emitted call can reach the dispatch callback without an
exactly matching declaration.

When `ChatSession` builds its cache from messages, it retains the structured
transcript and renders the complete conversation for every continuation, as
required by conversation-aware chat templates. When the rendered tokens extend
the tokens already represented by the session's KV cache, only the new suffix
is prefilled. Both the string-and-role overloads and the structured-message
overloads use this same retained-conversation and cache-reuse path. If a
template rewrites an earlier part of the prompt, the session rewinds to a
verified common prefix when the cache and input can be trimmed safely.
Otherwise, it rebuilds the cache rather than combining stale model state with a
mismatched prompt.

The low-level initializers that accept an existing raw KV cache cannot recover
the messages used to create it. Those initializers preserve fragment-based
continuation behavior; use the history initializer when a structured
conversation must be resumed.

When a session is initialized with history, the first generation must prefill
that history to create a KV cache. Reuse the same `ChatSession` so later tool
turns can take the suffix-only fast path.

## VLMs (Vision Language Models)

This same API supports VLMs as well.  Simply present the image or video
to the `ChatSession`:

```swift
let model = try await loadModel(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
)
let session = ChatSession(model)

let answer1 = try await session.respond(
    to: "what kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
)
print(answer1)

// we can ask a followup question referring back to the previous image
let answer2 = try await session.respond(
    to: "What is behind the dog?"
)
print(answer2)
```

## Advanced Usage

The `ChatSession` has a number of parameters you can supply when creating it:

- **instructions**: optional instructions to the chat session, e.g. describing what type of responses to give
    - for example you might instruct the language model to respond in rhyme or
        talking like a famous character from a movie
    - or that the responses should be very brief
- **generateParameters**: parameters that control the generation of output, e.g. token limits and temperature
    - see `GenerateParameters`
- **processing**: optional media processing instructions
