---
name: swift-mlx-lm
description: MLX Swift LM - Run LLMs and VLMs on Apple Silicon using MLX. Covers local inference, streaming, tool calling, LoRA fine-tuning, embeddings, and porting models from Python MLX-LM.
triggers:
  - mlx
  - mlx-swift
  - mlx-lm
  - apple silicon llm
  - local llm swift
  - vision language model swift
  - lora training swift
  - port model
  - model porting
  - add model support
---

# mlx-swift-lm Skill

## 1. Overview & Triggers

mlx-swift-lm is a Swift package for running Large Language Models (LLMs) and Vision-Language Models (VLMs) on Apple Silicon using MLX. It supports local inference, fine-tuning via LoRA/DoRA, and embeddings generation.

### When to Use This Skill
- Running LLM/VLM inference on macOS/iOS with Apple Silicon
- Streaming text generation from local models
- Vision tasks with images/video (VLMs)
- Tool calling / function calling with models
- LoRA adapter training and fine-tuning
- Text embeddings for RAG/semantic search
- Porting model architectures from Python MLX-LM to Swift

### Architecture Overview
```
MLXLMCommon     - Core infrastructure (ModelContainer, ChatSession, KVCache, etc.)
MLXLLM          - Text-only LLM support (Llama, Qwen, Gemma, Phi, DeepSeek, etc. - examples, not exhaustive)
MLXVLM          - Vision-Language Models (Qwen2-VL, PaliGemma, Gemma3, etc. - examples, not exhaustive)
Embedders       - Embedding models (BGE, Nomic, MiniLM)
```

## 2. Key File Reference

| Purpose | File Path |
|---------|-----------|
| Thread-safe model wrapper | `Libraries/MLXLMCommon/ModelContainer.swift` |
| Simplified chat API | `Libraries/MLXLMCommon/ChatSession.swift` |
| Generation & streaming | `Libraries/MLXLMCommon/Evaluate.swift` |
| KV cache types | `Libraries/MLXLMCommon/KVCache.swift` |
| Model configuration | `Libraries/MLXLMCommon/ModelConfiguration.swift` |
| Chat message types | `Libraries/MLXLMCommon/Chat.swift` |
| Tool call processing | `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift` |
| Concurrency utilities | `Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift` |
| LLM factory & registry | `Libraries/MLXLLM/LLMModelFactory.swift` |
| VLM factory & registry | `Libraries/MLXVLM/VLMModelFactory.swift` |
| LoRA configuration | `Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift` |
| LoRA training | `Libraries/MLXLLM/LoraTrain.swift` |

## 3. Quick Start

### LLM Chat (Simplest API)

```swift
import MLXLLM
import MLXLMCommon

// Load model (downloads from HuggingFace automatically)
let modelContainer = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// Create chat session
let session = ChatSession(modelContainer)

// Single response
let response = try await session.respond(to: "What is Swift?")
print(response)

// Streaming response
for try await chunk in session.streamResponse(to: "Explain concurrency") {
    print(chunk, terminator: "")
}
```

### VLM with Image

```swift
import MLXVLM
import MLXLMCommon

let modelContainer = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen2-VL-2B-Instruct-4bit")
)

let session = ChatSession(modelContainer)

// With image (video is also an optional parameter)
let image = UserInput.Image.url(imageURL)
let response = try await session.respond(
    to: "Describe this image",
    image: image,
    video: nil  // Optional video parameter
)
```

### Embeddings

```swift
import Embedders

// Note: Embedders uses loadModelContainer() helper (not a factory pattern)
let container = try await loadModelContainer(
    configuration: ModelConfiguration(id: "mlx-community/bge-small-en-v1.5-mlx")
)

let embeddings = await container.perform { model, tokenizer, pooler in
    let tokens = tokenizer.encode(text: "Hello world")
    let input = MLXArray(tokens).expandedDimensions(axis: 0)
    let output = model(input)
    let pooled = pooler(output, normalize: true)
    eval(pooled)
    return pooled
}
```

## 4. Primary Workflow: LLM Inference

### ChatSession API (Recommended)

`ChatSession` manages conversation history and KV cache automatically:

```swift
let session = ChatSession(
    modelContainer,
    instructions: "You are a helpful assistant",  // System prompt
    generateParameters: GenerateParameters(
        maxTokens: 500,
        temperature: 0.7
    )
)

// Multi-turn conversation (history preserved automatically)
let r1 = try await session.respond(to: "What is 2+2?")
let r2 = try await session.respond(to: "And if you multiply that by 3?")

// Clear session to start fresh
await session.clear()
```

### Streaming with generate()

For lower-level control, use `generate()` directly:

```swift
let input = try await modelContainer.prepare(input: UserInput(prompt: .text("Hello")))
let stream = try await modelContainer.generate(input: input, parameters: GenerateParameters())

for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")
    case .info(let info):
        print("\n\(info.tokensPerSecond) tok/s")
    case .toolCall(let call):
        // Handle tool call
        break
    }
}
```

### Tool Calling

```swift
// 1. Define tool
struct WeatherInput: Codable { let location: String }
struct WeatherOutput: Codable { let temperature: Double; let conditions: String }

let weatherTool = Tool<WeatherInput, WeatherOutput>(
    name: "get_weather",
    description: "Get current weather",
    parameters: [.required("location", type: .string, description: "City name")]
) { input in
    WeatherOutput(temperature: 22.0, conditions: "Sunny")
}

// 2. Include tool schema in request
let input = UserInput(
    prompt: .text("What's the weather in Tokyo?"),
    tools: [weatherTool.schema]
)

// 3. Handle tool calls in generation stream
for await generation in try await modelContainer.generate(input: input, parameters: params) {
    switch generation {
    case .chunk(let text): print(text)
    case .toolCall(let call):
        let result = try await call.execute(with: weatherTool)
        print("Weather: \(result.conditions)")
    case .info: break
    }
}
```

See [references/tool-calling.md](references/tool-calling.md) for multi-turn and feeding results back.

### GenerateParameters

```swift
let params = GenerateParameters(
    maxTokens: 1000,           // nil = unlimited
    maxKVSize: 4096,           // Sliding window (uses RotatingKVCache)
    kvBits: 4,                 // Quantized cache (4 or 8 bit)
    temperature: 0.7,          // 0 = greedy/argmax
    topP: 0.9,                 // Nucleus sampling
    repetitionPenalty: 1.1,    // Penalize repeats
    repetitionContextSize: 20  // Window for penalty
)
```

### Prompt Caching / History Re-hydration

Restore chat from persisted history:

```swift
let history: [Chat.Message] = [
    .system("You are helpful"),
    .user("Hello"),
    .assistant("Hi there!")
]

let session = ChatSession(
    modelContainer,
    history: history
)
// Continues from this point
```

## 5. Secondary Workflow: VLM Inference

### Image Input Types

```swift
// From URL (file or remote)
let image = UserInput.Image.url(fileURL)

// From CIImage
let image = UserInput.Image.ciImage(ciImage)

// From MLXArray directly
let image = UserInput.Image.array(mlxArray)
```

### Video Input

```swift
// From URL (file or remote)
let video = UserInput.Video.url(videoURL)

// From AVFoundation asset
let video = UserInput.Video.avAsset(avAsset)

// From pre-extracted frames
let video = UserInput.Video.frames(videoFrames)

let response = try await session.respond(
    to: "What happens in this video?",
    video: video
)
```

### Multiple Images

```swift
let images: [UserInput.Image] = [
    .url(url1),
    .url(url2)
]

let response = try await session.respond(
    to: "Compare these two images",
    images: images,
    videos: []
)
```

### VLM-Specific Processing

```swift
let session = ChatSession(
    modelContainer,
    processing: UserInput.Processing(
        resize: CGSize(width: 512, height: 512)  // Resize images
    )
)
```

## 6. Best Practices

### DO

```swift
// DO: Use ChatSession for multi-turn conversations
let session = ChatSession(modelContainer)

// DO: Use AsyncStream APIs (modern, Swift concurrency)
for try await chunk in session.streamResponse(to: prompt) { ... }

// DO: Check Task.isCancelled in long-running loops
for try await generation in stream {
    if Task.isCancelled { break }
    // process generation
}

// DO: Use ModelContainer.perform() for thread-safe access
await modelContainer.perform { context in
    // Access model, tokenizer safely
    let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
    return tokens
}

// DO: When breaking early from generation, use generateTask() to get a task handle
// This is the lower-level API used internally by ChatSession
let (stream, task) = generateTask(...)  // Returns (AsyncStream, Task)

for await item in stream {
    if shouldStop { break }
}
await task.value  // Ensures KV cache cleanup before next generation
```

> `generateTask()` is defined in `Evaluate.swift`. Most users should use `ChatSession` which handles this internally.

### DON'T

```swift
// DON'T: Share MLXArray across tasks (not Sendable)
let array = MLXArray(...)
Task { array.sum() }  // Wrong!

// DON'T: Use deprecated callback-based generation
// Old:
generate(input: input, parameters: params) { tokens in ... }  // Deprecated
// New:
for await generation in try generate(input: input, parameters: params, context: context) { ... }

// DON'T: Use old perform(model, tokenizer) signature
// Old:
modelContainer.perform { model, tokenizer in ... }  // Deprecated
// New:
modelContainer.perform { context in ... }

// DON'T: Forget to eval() MLXArrays before returning from perform()
await modelContainer.perform { context in
    let result = context.model(input)
    eval(result)  // Required before returning
    return result.item(Float.self)
}
```

### Thread Safety

- `ModelContainer` is `Sendable` and thread-safe
- `ChatSession` is NOT thread-safe (use from single task)
- `MLXArray` is NOT `Sendable` - don't pass across isolation boundaries
- Use `SendableBox` for transferring non-Sendable data in consuming contexts

### Memory Management

```swift
// For long contexts, use sliding window cache
let params = GenerateParameters(maxKVSize: 4096)

// For memory efficiency, use quantized cache
let params = GenerateParameters(kvBits: 4)  // or 8

// Clear session cache when done
await session.clear()
```

## 7. Reference Links

For detailed documentation on specific topics, see:

| Reference | When to Use |
|-----------|-------------|
| [references/model-container.md](references/model-container.md) | Loading models, ModelContainer API, ModelConfiguration |
| [references/kv-cache.md](references/kv-cache.md) | Cache types, memory optimization, cache serialization |
| [references/concurrency.md](references/concurrency.md) | Thread safety, SerialAccessContainer, async patterns |
| [references/tool-calling.md](references/tool-calling.md) | Function calling, tool formats, ToolCallProcessor |
| [references/tokenizer-chat.md](references/tokenizer-chat.md) | Tokenizer, Chat.Message, EOS tokens |
| [references/supported-models.md](references/supported-models.md) | Model families, registries, model-specific config |
| [references/lora-adapters.md](references/lora-adapters.md) | LoRA/DoRA/QLoRA, loading adapters |
| [references/training.md](references/training.md) | LoRATrain API, fine-tuning |
| [references/embeddings.md](references/embeddings.md) | EmbeddingModel, pooling, use cases |
| [references/model-porting.md](references/model-porting.md) | Porting models from Python MLX-LM to Swift |

## 8. Deprecated Patterns Summary

Most common migrations (see individual reference files for topic-specific deprecations):

| If you see... | Use instead... |
|---------------|----------------|
| `generate(... didGenerate:)` callback | `generate(...) -> AsyncStream` |
| `perform { model, tokenizer in }` | `perform { context in }` |
| `TokenIterator(prompt: MLXArray)` | `TokenIterator(input: LMInput)` |
| `ModelRegistry` typealias | `LLMRegistry` or `VLMRegistry` |
| `createAttentionMask(h:cache:[KVCache]?)` | `createAttentionMask(h:cache:KVCache?)` |

Each reference file contains a "Deprecated Patterns" section with topic-specific migrations.

## 9. Automatic vs Manual Configuration

### Automatic Behaviors (NO developer action needed)

The framework handles these automatically:

| Feature | Details |
|---------|---------|
| **EOS token loading** | Loaded from `config.json` |
| **EOS token override** | Priority: `generation_config.json` > `config.json` > defaults |
| **EOS token merging** | All sources merged at generation time |
| **EOS token detection** | Stops generation automatically when EOS encountered |
| **Chat template application** | Applied automatically via `applyChatTemplate()` |
| **Tool call format detection** | Inferred from `model_type` in `config.json` |
| **Cache type selection** | Based on GenerateParameters (`maxKVSize`, `kvBits`) |
| **Tokenizer loading** | Loaded from `tokenizer.json` automatically |
| **Model weights loading** | Downloaded and loaded from HuggingFace |

### Optional Configuration (Developer MAY configure)

| Feature | When to Configure |
|---------|-------------------|
| `extraEOSTokens` | Only if model has unlisted stop tokens |
| `toolCallFormat` | Only to override auto-detection |
| `maxKVSize` | To enable sliding window cache |
| `kvBits` | To enable quantized cache (4 or 8 bit) |
| `maxTokens` | To limit output length |
