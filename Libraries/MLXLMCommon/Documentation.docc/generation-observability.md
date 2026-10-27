# Observing token log probabilities

Compose a token handler to observe generation without changing ordinary text or tool-call events.

## Cached chat sessions

Observe a ``ChatSession`` response while retaining its transcript and KV cache.
The `observe` method accepts a ``GenerationTransformer`` that configures
this request and wraps a fresh text/tool handler for every generation pass:

```swift
let session = ChatSession(modelContainer)
for try await event in session.observe(to: "Explain gravity.", using: .logProbabilities(topCount: 5)) {
    if let generation = event.generation, let text = generation.chunk {
        print(text, terminator: "")
    }
    if let values = event.logProbabilities {
        print(values.chosen.token, values.chosen.logProbability)
    }
}
```

`topCount` is the number of top candidates; `0` reports only the chosen token.
``LogProbabilityGenerationTransformer`` configures a copy of the session's
parameters. This applies to the entire request, including automatic tool restarts,
without changing `generateParameters`, later calls, or forks.

## Custom transformers

A transformer's ``GenerationTransformer/configure(_:)`` enables the computation
its handler needs. Its ``GenerationTransformer/makeHandler(wrapping:)`` creates
fresh decoding state for each pass. The output can be any
``GenerationProviding`` type, including custom events unrelated to probabilities.
Observation-only events return `nil` from ``GenerationProviding/generation``.
``TextGenerationTransformer`` passes through the standard handler and parameters.

Wrap the supplied ``TextToolTokenLoopHandler`` and forward each base event exactly
once, in order. Preserve its stop-token properties, dispositions, final output,
and completion metadata. The session uses those events to update its transcript
and cache and dispatch tools. Automatically dispatched tool-call events are
consumed by the session; other transformed events reach the caller unchanged.
Cached-prompt counts are attached before the transformer sees completion events.

Probability events precede the token's decoded events. They include reasoning and
tool syntax and omit excluded stop tokens and unused lookahead. A completion
event ends each pass before automatic tool dispatch starts the next pass, so
observations can be grouped without a separate callback channel.

The ordinary `streamDetails` overload, `respond`, and `streamResponse` retain their
output and handler through the text transformer. Leave `GenerateParameters.logProbabilities` as `nil` when
unused to avoid probability computation and readback.

Requesting log probabilities with active speculative decoding throws
``ChatSession/ObservationError/unsupportedSpeculativeDecoding``. A configuration that falls back to standard decoding can still report
probabilities. Observation does not silently change the decoding strategy.

## Overview

``Generation`` and ``TokenGeneration`` keep their existing cases. For token-level
observations, wrap a ``TextToolTokenLoopHandler`` or ``RawTokenLoopHandler`` in
``LogProbabilityTokenLoopHandler`` and pass it to ``generateLoopTask(promptTokenCount:modelConfiguration:tokenizer:iterator:wiredMemoryTicket:includeStopToken:handler:)``.

Enable computation with ``GenerateParameters/logProbabilities`` when constructing
the ``TokenIterator``. `nil` disables reporting, `0` reports the selected token,
and a positive value also reports that many top candidates. The ordinary
`generate` and `generateTokens` streams do not emit probability events.

```swift
let input = try await modelContainer.prepare(input: UserInput(prompt: "Explain gravity."))
let parameters = GenerateParameters(maxTokens: 128, logProbabilities: 5)

let (events, task) = try await modelContainer.perform(nonSendable: input) { context, input in
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    let handler = LogProbabilityTokenLoopHandler(
        TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            stopStrings: context.configuration.effectiveStopStrings,
            format: context.configuration.toolCallFormat ?? .json))
    return generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer, iterator: iterator, handler: handler)
}

for await event in events {
    switch event {
    case .generation(let generation):
        if let text = generation.chunk { print(text, terminator: "") }
    case .probability(let values):
        print(values.chosen.token, values.chosen.logProbability)
    }
}
await task.value
```

Probability events follow individual tokens, including tokens used by reasoning
and tool-call parsers. Text chunks may span multiple tokens. Excluded stop tokens
do not produce probability events. The wrapper preserves the base handler's
stop policy, final output, and completion metadata.

A handler receives each token's values as ``DeferredTokenLogProbabilities``, which
stay on the GPU until the handler calls ``DeferredTokenLogProbabilities/materialize()``.
Only handlers that read them, such as ``LogProbabilityTokenLoopHandler``, pay for the
GPU-to-host copy.

Log probabilities normalize logits after processors and before temperature or
sampling filters, matching Python MLX-LM. They describe the processed token
distribution, rather than confidence that an answer is correct. Top candidates
can include the selected token.

Reporting adds normalization where the sampler does not already need it and
candidate extraction to each decode step. Keep it disabled when unused.
Speculative iterators do not provide log probabilities; use ``TokenIterator``
for observation.

``GenerateCompletionInfo`` also reports ``GenerateCompletionInfo/evictedTokenCount``.
Its ``GenerateCompletionInfo/reasoningTokenCount`` and
``GenerateCompletionInfo/answerTokenCount`` are available when a thinking budget
was configured. These counts and token probabilities do not expose internal
layer activations.
