# ModelContainer & Model Loading

## Overview

`ModelContainer` is the thread-safe wrapper for language models, providing exclusive access to model resources during inference. `ModelConfiguration` describes model identity and settings. Factory classes handle model instantiation from HuggingFace or local directories.

## Quick Reference

| Type | Purpose | File |
|------|---------|------|
| `ModelContainer` | Thread-safe model wrapper | `MLXLMCommon/ModelContainer.swift` |
| `ModelContext` | Model + tokenizer + processor bundle | `MLXLMCommon/ModelFactory.swift` |
| `ModelConfiguration` | Model ID, EOS tokens, settings | `MLXLMCommon/ModelConfiguration.swift` |
| `LLMModelFactory` | Load text-only LLMs | `MLXLLM/LLMModelFactory.swift` |
| `VLMModelFactory` | Load vision-language models | `MLXVLM/VLMModelFactory.swift` |
| `LLMRegistry` | Pre-configured LLM models | `MLXLLM/LLMModelFactory.swift` |
| `VLMRegistry` | Pre-configured VLM models | `MLXVLM/VLMModelFactory.swift` |
| `LLMTypeRegistry` | Model type -> init mapping | `MLXLLM/LLMModelFactory.swift` |
| `VLMTypeRegistry` | VLM type -> init mapping | `MLXVLM/VLMModelFactory.swift` |

## ModelContainer

### Creating a ModelContainer

```swift
// Via factory (recommended)
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// With custom hub
let hub = HubApi(hfToken: "your_token")
let container = try await LLMModelFactory.shared.loadContainer(
    hub: hub,
    configuration: .init(id: "private/model")
)

// With progress tracking
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: config
) { progress in
    print("Downloaded: \(progress.fractionCompleted)")
}
```

### Using ModelContainer

```swift
// Access configuration (async property)
let config = await container.configuration

// Access tokenizer
let tokenizer = await container.tokenizer

// Access processor
let processor = await container.processor

// Thread-safe model access via perform()
let result = try await container.perform { context in
    // context.model - the LanguageModel
    // context.tokenizer - the Tokenizer
    // context.processor - the UserInputProcessor
    // context.configuration - ModelConfiguration

    let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
    return tokens
}
```

### Convenience Methods

```swift
// Prepare input for generation
let lmInput = try await container.prepare(input: userInput)

// Generate with streaming
let stream = try await container.generate(input: lmInput, parameters: params)

// Encode/decode
let tokens = await container.encode("Hello world")
let text = await container.decode(tokens: [1, 2, 3])

// Apply chat template
let tokens = try await container.applyChatTemplate(messages: [
    ["role": "user", "content": "Hello"]
])
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

// Load container
let container = try await factory.loadContainer(
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

let container = try await factory.loadContainer(
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
let modelDir = configuration.modelDirectory(hub: HubApi())
// ~/.cache/huggingface/hub/models--mlx-community--Model-Name/...
```

## Memory Management

```swift
// Models are loaded fully into memory
// Quantized models (4-bit) use ~4x less memory than fp16

// Memory estimate: ~0.5GB per 1B parameters for 4-bit quantized
// Example: 7B 4-bit model ~ 3.5GB

// To unload, release all references to ModelContainer
container = nil  // Model memory freed
```

## Updating Model Parameters

```swift
// Load adapters into model
await container.update { context in
    try context.model.update(
        parameters: adapterParams,
        verify: .noUnusedKeys
    )
}
```

## Deprecated Patterns

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

// USE INSTEAD: perform with ModelContext
await container.perform { context in
    // context.model, context.tokenizer, context.processor, context.configuration
}
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
