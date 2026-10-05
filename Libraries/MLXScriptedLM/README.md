# MLXScriptedLM

Deterministic stand-ins for tokenizers and language models, so code built on
mlx-swift-lm can run without downloaded weights.  This is most useful
for writing unit tests that require interaction with tokenizers and
models.

Developers can create Tokenizers, LanguageModels and ToolCalls
from a simple script (as in movie script!).  The result will
be objects that can be injected into any place these types are used.
Fully functional language models and related types with no
weight downloads -- perfect for tests.

It is available as a library so that downstream consumers
can make use of it as well.

Status: Tokenizers are built and integrated into the local tests.

Planned: Scripted LanguageModels that wrap real (random weight) models.
This will allow you to construct tests that have real model behavior in
terms of KVCache, but require no weight downloads.  Scripted tool
integration is also planned.

## Pieces

### Tokenizer and Vocabulary

- `ScriptedTokenizer` — a `Tokenizer` using longest-match encoding over a
  `ScriptedVocabulary`, with specials matched first.
- `VocabularyBuilder` / `ScriptedVocabulary` — a vocabulary built from the
  scenario corpus: a reserved block of atomic specials, 256 byte-fallback
  tokens, then corpus pieces. `decode(encode(s)) == s` for every string.
- `PseudoWordTokenizer` — for random-weight models: every id decodes to a word and
  there is no EOS, so generation runs until `maxTokens`.
  
### Chat Templates
  
- `ScriptedChatTemplate` — the template protocol. A template renders to a
  list of strings, each encoded on its own, and declares the tokens it needs
  (`specialTokens`, `textMarkers`, `corpus`). `ScriptedTokenizer` adds those to
  the vocabulary, so scenarios list only their own text.
- `MinimalChatTemplate` — one marker per role (`<|user|>` … `<|end|>`). Not a
  real family; the default, for mechanics tests where readability matters.
- `ChatMLTemplate` — plain ChatML (`<|im_start|>user\n` … `<|im_end|>`). Tools
  and tool calls throw `unsupported` until family variants arrive.

## Example

```swift
// create the tokenizer from a corpus
let tokenizer = ScriptedTokenizer(
    corpus: ["how are you?", "fine, you?"],
    textMarkers: ["<think>"])

// the tokenizer handles tokens along the lines of a BPE tokenizer
// and has fallback tokens for words that it doesn't know
let prompt = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "how are you doing?"]])
let reply = tokenizer.encode(text: "fine, you?", addSpecialTokens: false)
```

## Implementation Notes

- Words longer than `maxPieceLength` (default 4 bytes) are always multi-token.
- Non-ASCII characters use byte fallback by default, so streaming
  detokenization sees partial UTF-8.
- A marker is atomic (one special token) unless it is registered as a text
  marker, in which case it encodes to two or more ordinary tokens.
- The end-of-turn marker (`<|end|>`, `<|im_end|>`) is the default EOS. So a
  turn's prompt plus its generated tokens is a prefix of the next turn's
  rendering.
