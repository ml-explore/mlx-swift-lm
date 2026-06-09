# ModelContext & Model Loading

## Overview

`ModelContext` is the primary bundle for a loaded language model: it holds the
model, tokenizer, processor, and configuration. It is `Sendable` (its `model` is
`any LanguageModel & Sendable`, wrapped in a `MaterializedModule`), so you pass it
directly across tasks and actors — no wrapper or `perform { }` block needed.
`ModelConfiguration` describes model identity and settings. Factory classes handle
model instantiation from HuggingFace or local directories.

> `MaterializedModule` (a Sendable, sealed wrapper around a `Module`) and
> `MaterializedArray` (a Sendable snapshot of an `MLXArray`) are what make
> `ModelContext` Sendable. See [concurrency.md](concurrency.md) for details.

`ModelContainer` — the old thread-safe actor wrapper — is **deprecated**; use
`ModelContext` directly.

## Quick Reference

| Type | Purpose | File |
|------|---------|------|
| `ModelContext` | **Primary** Sendable model + tokenizer + processor bundle | `MLXLMCommon/ModelFactory.swift` |
| `TrainableModelContext` | Mutable model bundle for training / applying adapters | `MLXLMCommon/ModelFactory.swift` |
| `ModelContainer` | Deprecated thread-safe wrapper (use `ModelContext`) | `MLXLMCommon/ModelContainer.swift` |
| `ModelConfiguration` | Model ID, EOS tokens, settings | `MLXLMCommon/ModelConfiguration.swift` |
| `LLMModelFactory` | Load text-only LLMs | `MLXLLM/LLMModelFactory.swift` |
| `VLMModelFactory` | Load vision-language models | `MLXVLM/VLMModelFactory.swift` |
| `LLMRegistry` | Pre-configured LLM models | `MLXLLM/LLMModelFactory.swift` |
| `VLMRegistry` | Pre-configured VLM models | `MLXVLM/VLMModelFactory.swift` |
| `LLMTypeRegistry` | Model type -> init mapping | `MLXLLM/LLMModelFactory.swift` |
| `VLMTypeRegistry` | VLM type -> init mapping | `MLXVLM/VLMModelFactory.swift` |

## ModelContext

### Loading a ModelContext

```swift
// Via factory (recommended). The factory method is `load` (NOT `loadModel`).
let context = try await LLMModelFactory.shared.load(
    from: HubClient.default,
    using: TokenizersLoader(),  // TokenizersLoader() from MLXLMTokenizers (swift-tokenizers-mlx)
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// Or the free function
let context = try await loadModel(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// With custom hub (from MLXLMHuggingFace)
let hub = HubClient(token: "hf_...")
let context = try await LLMModelFactory.shared.load(
    from: hub,
    using: TokenizersLoader(),
    configuration: .init(id: "private/model")
)

// With progress tracking
let context = try await LLMModelFactory.shared.load(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: config,
    progressHandler: { progress in
        print("Downloaded: \(progress.fractionCompleted)")
    }
)
```

You can also load via the `#huggingFaceLoadModel(configuration:)` macro.

### Using ModelContext

Because `ModelContext` is `Sendable`, access its members directly — no actor,
no `perform { }`:

```swift
let config = context.configuration      // ModelConfiguration
let tokenizer = context.tokenizer       // Tokenizer
let processor = context.processor       // UserInputProcessor
let model = context.model               // any LanguageModel & Sendable (sealed)

let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
```

### Generation

```swift
// Prepare input for generation
let lmInput = try context.processor.prepare(input: userInput)

// Generate with streaming (from Evaluate.swift)
let stream = try generate(input: lmInput, parameters: params, context: context)

// Generate with wired-memory coordination
let ticket = WiredSumPolicy().ticket(size: estimatedBytes, kind: .active)
let streamWithTicket = try generate(
    input: lmInput,
    parameters: params,
    context: context,
    wiredMemoryTicket: ticket
)

// Encode/decode via the tokenizer
let tokens = context.tokenizer.encode(text: "Hello world")
let text = context.tokenizer.decode(tokens: [1, 2, 3])
```

### Generation + Wired Memory

`generate(...)` accepts an optional `wiredMemoryTicket` so callers can coordinate
process-wide wired-memory policy with active inference work.

```swift
let policy = WiredBudgetPolicy(baseBytes: measuredWeightPlusWorkspaceBytes)
let inferenceTicket = policy.ticket(size: measuredKVBytes, kind: .active)

let userInput = UserInput(prompt: "Summarize this article")
let lmInput = try context.processor.prepare(input: userInput)

let stream = try generate(
    input: lmInput,
    parameters: GenerateParameters(maxTokens: 400),
    context: context,
    wiredMemoryTicket: inferenceTicket
)

for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## ModelConfiguration

### Creating Configurations

```swift
// From HuggingFace model ID
let config = ModelConfiguration(
    id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    defaultPrompt: "Hello",
    extraEOSTokens: ["<|eot_id|>"]
)

// With specific revision
let config = ModelConfiguration(
    id: "mlx-community/model",
    revision: "v1.0"
)

// From local directory
let config = ModelConfiguration(
    directory: URL(filePath: "/path/to/model"),
    extraEOSTokens: ["</s>"]
)

// With tool call format
let config = ModelConfiguration(
    id: "mlx-community/GLM-4-9B-0414-4bit",
    toolCallFormat: .glm4
)
```

### Configuration Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `Identifier` | `.id(String)` or `.directory(URL)` |
| `name` | `String` | Human-readable name |
| `tokenizerId` | `String?` | Pull tokenizer from different repo |
| `overrideTokenizer` | `String?` | Force tokenizer class |
| `defaultPrompt` | `String` | Default prompt for testing |
| `extraEOSTokens` | `Set<String>` | Additional stop tokens (as strings) |
| `eosTokenIds` | `Set<Int>` | EOS token IDs (loaded from config) |
| `toolCallFormat` | `ToolCallFormat?` | Tool calling format |

## Model Factories

### LLMModelFactory

```swift
// Shared instance
let factory = LLMModelFactory.shared

// Load a Sendable ModelContext
let context = try await factory.load(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: LLMRegistry.llama3_2_3B_4bit
)

// Custom factory with registries
let customFactory = LLMModelFactory(
    typeRegistry: LLMTypeRegistry.shared,
    modelRegistry: LLMRegistry.shared
)
```

### VLMModelFactory

```swift
let factory = VLMModelFactory.shared

let context = try await factory.load(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: VLMRegistry.qwen2VL2BInstruct4Bit
)
```

### Model Registries

Pre-configured models for quick loading:

```swift
// LLM examples
LLMRegistry.llama3_2_3B_4bit
LLMRegistry.qwen3_4b_4bit
LLMRegistry.gemma3_1B_qat_4bit
LLMRegistry.phi3_5_4bit
LLMRegistry.mistral7B4bit

// VLM examples
VLMRegistry.qwen2VL2BInstruct4Bit
VLMRegistry.gemma3_4B_qat_4bit
VLMRegistry.paligemma3bMix448_8bit
```

## Type Registries

Map `model_type` from config.json to model initializers:

```swift
// LLMTypeRegistry supports (partial list):
// "llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3",
// "phi", "phi3", "deepseek_v3", "glm4", "lfm2", ...

// VLMTypeRegistry supports:
// "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "paligemma", "gemma3",
// "idefics3", "smolvlm", "pixtral", "mistral3", ...
```

## Loading Flow

1. **Download**: Model weights fetched from HuggingFace (cached locally)
2. **Parse config.json**: Determine `model_type` and configuration
3. **Create model**: TypeRegistry maps type to initializer
4. **Load weights**: `.safetensors` files loaded into model
5. **Load tokenizer**: From `tokenizer.json` / `tokenizer_config.json`
6. **Load EOS tokens**: From `generation_config.json` (overrides config.json)
7. **Create processor**: For input preparation

```swift
// Download location
let resolved = try await resolve(
    configuration: configuration,
    from: HubClient.default,
    progressHandler: { _ in }
)
let modelDir = resolved.modelDirectory
// ~/.cache/huggingface/hub/models--mlx-community--Model-Name/...
```

## Memory Management

```swift
// Models are loaded fully into memory
// Quantized models (4-bit) use ~4x less memory than fp16

// Memory estimate: ~0.5GB per 1B parameters for 4-bit quantized
// Example: 7B 4-bit model ~ 3.5GB

// To unload, release all references to the ModelContext
context = nil  // Model memory freed
```

## Modifying Model Weights / Adapters

A `ModelContext`'s model is **materialized and sealed** — you cannot mutate or
train it, and calling `context.model.update(parameters:)` traps. To modify
weights or apply LoRA adapters, load a **mutable** `TrainableModelContext`
instead of a `ModelContext`:

```swift
// Trainable load returns a MUTABLE TrainableModelContext
let trainable = try await LLMModelFactory.shared.loadTrainable(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: .init(id: "mlx-community/Llama-3.2-3B-Instruct-4bit")
)

// trainable.model is a TrainableLanguageModel — adapters/updates apply directly
let adapter = try LoRAContainer.from(directory: adapterDir)
try adapter.load(into: trainable.model)

// When done, convert to an inference (Sendable, materialized) ModelContext
let context = ModelContext(trainable)
```

`loadTrainable(...)` is also available as a factory method
(`factory.loadTrainable(...)`) and as the `#huggingFaceLoadTrainableModel(...)`
macro (note the spelling "Trainabled"). `TrainableModelContext` exposes
`configuration`, `model` (`any TrainableLanguageModel`), `processor`, and
`tokenizer`.

## Deprecated: ModelContainer

`ModelContainer` (and `EmbedderModelContainer`) is retained only for
back-compat. It is marked `@available(*, deprecated, message: "use ModelContext
instead")`. Because `ModelContext` is now `Sendable`, the actor wrapper is no
longer needed — prefer loading a `ModelContext` directly and using its members.

Deprecated loading aliases (map to the current API):

| Deprecated | Use instead |
|------------|-------------|
| `factory.loadContainer(...)` | `factory.load(...)` |
| `loadModelContainer(...)` | `loadModel(...)` |
| `#huggingFaceLoadModelContainer(...)` | `#huggingFaceLoadModel(...)` |

The deprecated `ModelContainer` API surface:

```swift
// DEPRECATED: async property accessors
let config = await container.configuration
let tokenizer = await container.tokenizer
let processor = await container.processor

// DEPRECATED: serialized access via perform()
let result = try await container.perform { context in
    // context.model, context.tokenizer, context.processor, context.configuration
    return try context.tokenizer.applyChatTemplate(messages: messages)
}

// DEPRECATED: mutation via update()
await container.update { context in
    // ...
}

// DEPRECATED: convenience generation/encode/decode on the container
let lmInput = try await container.prepare(input: userInput)
let stream = try await container.generate(input: lmInput, parameters: params)
```

### Old perform() signatures

```swift
// DEPRECATED: perform with (model, tokenizer)
await container.perform { model, tokenizer in
    // ...
}

// DEPRECATED: perform with (model, tokenizer, values)
await container.perform(values: myData) { model, tokenizer, data in
    // ...
}

// DEPRECATED: perform with ModelContext (the whole ModelContainer is deprecated)
await container.perform { context in
    // context.model, context.tokenizer, context.processor, context.configuration
}

// USE INSTEAD: access the Sendable ModelContext directly
let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
```

### ModelRegistry typealias

```swift
// DEPRECATED
import MLXLLM
let config = ModelRegistry.llama3_2_3B_4bit  // ambiguous

// USE INSTEAD
import MLXLLM
let config = LLMRegistry.llama3_2_3B_4bit

// or for VLM
import MLXVLM
let config = VLMRegistry.qwen2VL2BInstruct4Bit
```

The `ModelRegistry` typealias still exists for backwards compatibility but is deprecated. Use `LLMRegistry` or `VLMRegistry` explicitly.
