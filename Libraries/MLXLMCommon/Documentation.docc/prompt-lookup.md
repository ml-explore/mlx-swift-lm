# Prompt lookup decoding

Set ``GenerateParameters/promptLookup`` to let streaming generation and ``ChatSession`` propose repeated tokens from the recent prompt. The target model verifies proposals; no draft model is loaded.

```swift
let parameters = GenerateParameters(
    maxTokens: 512, temperature: 0,
    promptLookup: .init(maxDraftTokens: 8))
```

``PromptLookupConfiguration`` bounds the indexed context and draft length and sets the minimum occurrence and confidence thresholds. The index allocates its memory up front, about 6 MiB with the defaults. ``PromptLookupTokenIterator`` also accepts the full `history` when its input is only an uncached suffix. Recurrent or nested caches and media inputs use ordinary decoding.

If you drive ``PromptLookupTokenIterator`` directly and stop before it returns `nil`, call ``PromptLookupTokenIterator/finish()`` before you reuse its cache. It removes verified tokens the iterator has not returned yet.

A rejected proposal rewinds the cache but not ``LMOutput/State``. Use prompt lookup only with models whose state a decode step does not rewrite, as with ``SpeculativeTokenIterator``.

Performance depends on how often the target accepts drafts. It helps most when the output repeats text from the prompt, such as returning an edited file. Batched greedy verification can differ numerically from single-token decoding near tied logits.
