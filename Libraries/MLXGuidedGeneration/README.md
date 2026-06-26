# MLXGuidedGeneration

Guided (constrained) generation for MLX. It forces a language model's output to conform to a JSON Schema, an EBNF grammar, or a structural tag by masking the token logits at every decoding step, so the result is always structurally valid. It works with any MLX language model and has no FoundationModels dependency, down to the package's macOS 14 / iOS 17 floor.

# Why it is bundled this way

The engine is backed by [XGrammar](https://github.com/mlc-ai/xgrammar), which we vendor in-repo and compile here rather than depend on the official XGrammar Swift package. Compiling it ourselves lets us rename its C++ namespace so our copy cannot collide with any other XGrammar linked into the same binary. That matters because others in the community may also depend on XGrammar, and we do not want to break them.

# Usage

## Built-in with FoundationModels (macOS / iOS 27+)

When you drive an MLX model through FoundationModels, guided generation is automatic: ask `respond` for a `@Generable` type and the response is constrained to its schema for you.

```swift
import FoundationModels
import MLXFoundationModels

@Generable
struct City {
    let name: String
    let country: String
}

// `model` is an MLXLanguageModel (see MLXLanguageModel docs for construction).
let session = LanguageModelSession(model: model, tools: [], instructions: nil)
let response = try await session.respond(
    to: "Name a city to visit in Japan.",
    generating: City.self
)
print(response.content)  // a City, guaranteed to match the schema
```

## Standalone on any MLX model

On older OS versions, or with any MLX model you load yourself, drive the library directly. Starting from a loaded `ModelContext`, build a grammar constraint from a JSON Schema string and run the guided loop.

```swift
import MLXGuidedGeneration
import MLXLMCommon

// `context: ModelContext` comes from standard MLXLLM loading
// (e.g. loadModelContainer + ModelContainer.perform).
let tokenizer = context.tokenizer

// 1. Extract the vocab in the shape xgrammar expects.
let grammarVocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)

// 2. Build a grammar tokenizer.
let grammarTokenizer = try GrammarTokenizer(
    vocab: grammarVocab.vocab,
    vocabType: grammarVocab.vocabType,
    eosTokenId: Int32(tokenizer.eosTokenId ?? 0)
)

// 3. Compile a JSON Schema into a constraint.
let schema = #"{"type":"object","properties":{"name":{"type":"string"},"country":{"type":"string"}}}"#
let constraint = try GrammarConstraint(
    tokenizer: grammarTokenizer,
    jsonSchema: schema,
    fastForward: true,
    hostTokenizer: tokenizer
)

// 4. Run the guided loop, collecting the constrained output.
let input = try await context.processor.prepare(
    input: UserInput(prompt: "Name a city to visit in Japan, as JSON.")
)
var output = ""
try GuidedGenerationLoop.run(
    input: input,
    context: context,
    constraint: constraint,
    maxTokens: 256,
    vocabSize: grammarTokenizer.vocabSize
) { delta in
    output += delta
    return true
}
print(output)  // valid JSON matching `schema`
```
