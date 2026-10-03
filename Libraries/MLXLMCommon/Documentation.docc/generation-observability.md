# Observing token log probabilities

Compose a token handler to observe generation without changing ordinary text or tool-call events.

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

Log probabilities normalize logits after processors and before temperature or
sampling filters, matching Python MLX-LM. They describe the processed token
distribution, rather than confidence that an answer is correct. Top candidates
can include the selected token.

Reporting adds normalization where the sampler does not already need it,
candidate extraction, and GPU-to-host transfers. Keep it disabled when unused.
Direct speculative iterators do not provide log probabilities; use ``TokenIterator``
for observation.

``GenerateCompletionInfo`` also reports ``GenerateCompletionInfo/evictedTokenCount``.
Its ``GenerateCompletionInfo/reasoningTokenCount`` and
``GenerateCompletionInfo/answerTokenCount`` are available when a thinking budget
was configured. These counts and token probabilities do not expose internal
layer activations.
