# MLXGuidedGeneration

Guided (constrained) generation for MLX. It forces a language model's output to conform to a JSON Schema, an EBNF grammar, or an XGrammar structural tag by masking the token logits at every decoding step, so the result is always structurally valid. It works with any MLX language model and runs on macOS 14 / iOS 17 and later.

## When to use it

Use guided generation whenever a model's output needs to be data that your code can rely on, such as the fields of a form or the arguments for a tool call.

## Usage

### Built-in with `MLXFoundationModels`

With an MLX model running through `MLXFoundationModels`, guided generation is automatic: pass a `@Generable` type to `respond`, and the response is constrained to match its schema.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable
struct Neighborhood {
    let name: String
    let knownFor: String
}

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = #huggingFaceLanguageModel(
        configuration: LLMRegistry.gemma3_1B_qat_4bit,
        capabilities: [.guidedGeneration])
    let session = LanguageModelSession(model: model)
    let response = try await session.respond(
        to: "Suggest a Chicago neighborhood to explore.",
        generating: Neighborhood.self)
    print(response.content)  // a Neighborhood, guaranteed to match the schema
}
```

Learn more about how [`MLXFoundationModels`](../MLXFoundationModels/README.md) integrates `mlx-swift-lm` with Apple's `FoundationModels` framework.

### Standalone on any MLX model

MLXGuidedGeneration constrains any MLX model's output to a JSON Schema, just as in the `@Generable` example above. It supports macOS 14 / iOS 17 and later.

```swift
import HuggingFace
import MLXGuidedGeneration
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// Load any MLX model yourself; here the same gemma model as above.
let container = try await #huggingFaceLoadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit)

let output = try await container.perform { context in
    let tokenizer = context.tokenizer

    // 1. Extract the vocab in the shape XGrammar expects.
    let grammarVocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)

    // 2. Build a grammar tokenizer.
    let grammarTokenizer = try GrammarTokenizer(
        vocab: grammarVocab.vocab,
        vocabType: grammarVocab.vocabType,
        eosTokenId: Int32(tokenizer.eosTokenId ?? 0))

    // 3. Compile a JSON Schema into a constraint.
    let schema = #"{"type":"object","properties":{"name":{"type":"string"},"knownFor":{"type":"string"}}}"#
    let constraint = try GrammarConstraint(
        tokenizer: grammarTokenizer,
        jsonSchema: schema,
        fastForward: true,
        hostTokenizer: tokenizer)

    // 4. Run the guided loop, collecting the constrained output.
    let input = try await context.processor.prepare(
        input: UserInput(prompt: "Suggest a Chicago neighborhood to explore, as JSON."))
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
    return output
}
print(output)  // valid JSON matching `schema`
```

> [!WARNING]
> `GuidedGenerationLoop.run` can block for hundreds of milliseconds on a cold
> grammar compile: the first call for a given schema/grammar and tokenizer
> compiles the grammar and builds its token mask, and neither step yields. Don't
> call it from `@MainActor` — run it in `Task.detached` or on a background
> executor. Later calls that reuse the same compiled grammar and tokenizer skip
> the compile. Pre-warming the expected schema with a throwaway
> `GrammarConstraint` from a background task before the user-visible request
> removes the blocking window entirely.

## Why it is bundled this way

The engine is backed by [XGrammar](https://github.com/mlc-ai/xgrammar), which we vendor in-repo and compile here rather than depend on the official XGrammar Swift package. Compiling it ourselves lets us rename its C++ namespace so our copy cannot collide with any other XGrammar linked into the same binary. Anyone else who depends on XGrammar can link their own copy alongside ours, each working independently.
