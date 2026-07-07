# MLXGuidedGeneration

Guided (constrained) generation for MLX. It forces a language model's output to conform to a JSON Schema, an EBNF grammar, or a structural tag by masking the token logits at every decoding step, so the result is always structurally valid. It works with any MLX language model and needs no FoundationModels dependency, so it runs on macOS 14 / iOS 17 and later.

## When to reach for it

Reach for guided generation whenever a model's output needs to be data your code can rely on: pulling fields out of freeform text, filling in a form, or building the arguments for a tool call. Because the structure is guaranteed, you decode the result and move on, instead of writing defensive parsing or retrying until the output happens to come back valid.

## Usage

### Built-in with FoundationModels

When you drive an MLX model through FoundationModels, guided generation is automatic: ask `respond` for a `@Generable` type and the response is constrained to its schema for you.

```swift
import FoundationModels
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM

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

Guided output is just the start. Learn how [`MLXFoundationModels`](../MLXFoundationModels/README.md) makes any MLX model a drop-in for Apple's `SystemLanguageModel`, adding tool calling, reasoning, and vision through the same `LanguageModelSession`.

### Standalone on any MLX model

You don't need FoundationModels to get guaranteed-valid output. MLXGuidedGeneration constrains any MLX model you load yourself, back to macOS 14 / iOS 17: describe the shape you want as a JSON Schema, and every response conforms to it, the same guarantee as the `@Generable` path above, without the dependency.

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

    // 1. Extract the vocab in the shape xgrammar expects.
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

## Why it is bundled this way

The engine is backed by [XGrammar](https://github.com/mlc-ai/xgrammar), which we vendor in-repo and compile here rather than depend on the official XGrammar Swift package. Compiling it ourselves lets us rename its C++ namespace so our copy cannot collide with any other XGrammar linked into the same binary. Anyone else who depends on XGrammar can link their own copy alongside ours, each working independently.
