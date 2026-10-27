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
Scripted models follow plain-text transcripts.

Planned: tool calls and reasoning in transcripts, and speculative
decoding scenarios.

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

### Scripted Models

- `Transcript` — the conversation: system, user, and assistant turns. Write
  it as an array or load it from JSON. Each assistant turn ends with EOS
  unless its terminator is `none`.
- `ScriptedLogitsModel` — wraps a small random-weight model and raises the
  scripted token above each logits row's maximum. The wrapped model does the
  real cache and prefill work, and samplers still see a full distribution.
  `Margin.dominant` keeps the script under any sampler; `.plausible` lets
  processors override it.
- `TurnScripts` — finds the turn and position. Each assistant turn starts
  after the template's generation prompt, and the script is indexed by
  position from there. The prompt's last user message picks the turn, so a
  turn that `ChatSession` drops (empty, cancelled) does not replay. After the
  script, stop tokens and the turn marker are blocked until the turn ends. A
  turn past the end of the transcript emits EOS at once.
- `TranscriptKVCache` — the tokens the model has processed, kept as the last
  cache entry. Generation trims and copies it with the rest of the cache, so
  the position stays right after rewinds and speculative rejection.
- `ForwardCallLog` — every call: cache offset, inputs, boosted tokens, and
  any input that differed from the script.
- `ForwardingLanguageModel` — forwards every `LanguageModel` requirement, for
  wrappers that change one behavior. Its parameters are the wrapped model's,
  under the same keys, so weights load and quantize through it.
- `ScriptedScenario` — builds the tokenizer, model, and `ModelContext` from a
  transcript. You supply the wrapped model, sized to the vocabulary.

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

```swift
@_spi(Testing) import MLXLLM

let scenario = try ScriptedScenario(
    transcript: Transcript([.user("hello"), .assistant("hi there")])
) { vocabularySize in
    // any small random-weight model with this vocabulary size
    Gemma3TextModel(.smallConfiguration(vocabularySize: vocabularySize))
}
let session = ChatSession(scenario.context)
let reply = try await session.respond(to: "hello")  // "hi there"
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
