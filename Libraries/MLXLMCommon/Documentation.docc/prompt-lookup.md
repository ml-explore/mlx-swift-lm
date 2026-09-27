# Prompt lookup decoding

Set ``GenerateParameters/promptLookup`` to let streaming generation and ``ChatSession`` propose repeated tokens from the recent prompt. The target model verifies proposals; no draft model is loaded.

```swift
let parameters = GenerateParameters(
    maxTokens: 512, temperature: 0,
    promptLookup: .init(maxDraftTokens: 8))
```

``PromptLookupConfiguration`` bounds the indexed context and draft length and sets the minimum occurrence and confidence thresholds. ``PromptLookupTokenIterator`` also accepts the full `history` when its input is only an uncached suffix. Recurrent or nested caches and media inputs use ordinary decoding.

Performance depends on how often the target accepts drafts. Batched greedy verification can differ numerically from single-token decoding near tied logits.
