# ``MLXScriptedLM``

Deterministic stand-ins for tokenizers and language models, so code built on
mlx-swift-lm can run without downloaded weights.

## Overview

Code that drives `ChatSession` or the generation APIs is hard to test against a real
model: weights are large, output varies, and the interesting cases (a tool call, a
reasoning block, a stop string) happen only when the model chooses them. This library
provides pieces you control instead.

- term ``ScriptedTokenizer``: A tokenizer built from your test's own text. Encoding is
  deterministic and lossless, so text survives the decode and re-encode that
  `ChatSession` performs on each turn.
- term ``PseudoWordTokenizer``: A tokenizer for random-weight models. Every id decodes to a
  word, and there is no end-of-sequence token, so generation runs until `maxTokens`.
- term ``ScriptedChatTemplate``: The chat template protocol. ``MinimalChatTemplate``
  keeps prompts short and readable; ``ChatMLTemplate`` renders plain ChatML.
- term ``Transcript``: The conversation to script, written in code or loaded from JSON.
- term ``ScriptedLogitsModel``: Wraps a small random-weight model and boosts the scripted
  token in its logits, so generation follows the transcript while the cache, samplers, and
  processors stay real.
- term ``ScriptedScenario``: Builds the tokenizer, model, and `ModelContext` from a
  transcript.

```swift
import MLXScriptedLM

let tokenizer = ScriptedTokenizer(
    corpus: ["how are you?", "fine, you?"],
    template: ChatMLTemplate())

let prompt = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "how are you?"]])
```

## Topics

### Tokenizers

- ``ScriptedTokenizer``
- ``PseudoWordTokenizer``

### Chat Templates

- ``ScriptedChatTemplate``
- ``MinimalChatTemplate``
- ``ChatMLTemplate``
- ``ScriptedTemplateMessage``
- ``ScriptedTemplateError``

### Scripted Models

- ``ScriptedScenario``
- ``Transcript``
- ``ScriptedLogitsModel``
- ``TurnScripts``
- ``ForwardCallLog``
- ``ForwardCall``
- ``ForwardingLanguageModel``
- ``ScriptedInputProcessor``

### Model Support

- ``TranscriptKVCache``
- ``TranscriptKVCacheError``

### Vocabulary

- ``VocabularyBuilder``
- ``ScriptedVocabulary``
