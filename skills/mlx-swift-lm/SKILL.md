---
name: swift-mlx-lm
description: MLX Swift LM - Run LLMs and VLMs on Apple Silicon using MLX. Covers local inference, streaming, wired memory coordination, tool calling, LoRA fine-tuning, embeddings, and model porting.
triggers:
  - mlx
  - mlx-swift
  - mlx-lm
  - apple silicon llm
  - local llm swift
  - vision language model swift
  - lora training swift
  - wired memory
  - wiredmemory
  - wired memory ticket
  - model porting
  - add model support
---

# mlx-swift-lm Skill

## 1. Overview & Triggers

mlx-swift-lm is a Swift package for running Large Language Models (LLMs) and Vision-Language Models (VLMs) on Apple Silicon using MLX. It supports local inference, streaming generation, wired-memory coordination, tool calling, LoRA/DoRA fine-tuning, and embeddings.

### When to Use This Skill
- Running LLM/VLM inference on macOS/iOS with Apple Silicon
- Streaming text generation from local models
- Coordinating concurrent inference with wired-memory policies and tickets
- Tool calling / function calling with models
- LoRA adapter training and fine-tuning
- Text embeddings for RAG/semantic search
- Porting model architectures from Python MLX-LM to Swift

### Architecture Overview
```
MLXLMCommon     - Core infra (ModelContainer, ChatSession, Evaluate, KVCache, wired memory helpers)
MLXLLM          - Text-only LLM support (Llama, Qwen, Gemma, Phi, DeepSeek, etc.)
MLXVLM          - Vision-Language Models (Qwen-VL, PaliGemma, Gemma3, etc.)
MLXEmbedders    - Embedding models and pooling utilities
```

## 2. Key File Reference

| Purpose | File Path |
|---------|-----------|
| Thread-safe model wrapper | `Libraries/MLXLMCommon/ModelContainer.swift` |
| Simplified chat API | `Libraries/MLXLMCommon/ChatSession.swift` |
| Generation & streaming APIs | `Libraries/MLXLMCommon/Evaluate.swift` |
| KV cache types | `Libraries/MLXLMCommon/KVCache.swift` |
| Wired-memory policies | `Libraries/MLXLMCommon/WiredMemoryPolicies.swift` |
| Wired-memory measurement helpers | `Libraries/MLXLMCommon/WiredMemoryUtils.swift` |
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

let modelContainer = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(modelContainer)

let response = try await session.respond(to: "What is Swift?")
print(response)

for try await chunk in session.streamResponse(to: "Explain structured concurrency") {
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
let image = UserInput.Image.url(imageURL)

let response = try await session.respond(
    to: "Describe this image",
    image: image,
    video: nil
)
```

### Embeddings

```swift
import Embedders

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
    instructions: "You are a helpful assistant",
    generateParameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
)

let r1 = try await session.respond(to: "What is 2+2?")
let r2 = try await session.respond(to: "And if you multiply that by 3?")

await session.clear()
```

### Streaming with `ModelContainer.generate(...)`

For lower-level control, prepare `UserInput` and generate directly:

```swift
let userInput = UserInput(prompt: "Hello")
let lmInput = try await modelContainer.prepare(input: userInput)

let stream = try await modelContainer.generate(
    input: lmInput,
    parameters: GenerateParameters()
)

for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")
    case .toolCall(let call):
        print("Tool call: \(call.function.name)")
    case .info(let info):
        print("\nStop reason: \(info.stopReason)")
        print("\(info.tokensPerSecond) tok/s")
    }
}
```

### Generation API Surface (Evaluate.swift)

Use these depending on your control needs:

- `generate(input:..., context:...) -> AsyncStream<Generation>`: decoded text + tool calls.
- `generateTask(...) -> (AsyncStream<Generation>, Task<Void, Never>)`: same output, plus task handle for deterministic cleanup when consumers stop early.
- `generateTokens(...) -> AsyncStream<TokenGeneration>`: raw token IDs.
- `generateTokensTask(...) -> (AsyncStream<TokenGeneration>, Task<Void, Never>)`: raw tokens + task handle.
- `GenerateStopReason`: `.stop`, `.length`, `.cancelled` in final `.info`.

See [references/generation.md](references/generation.md) for full patterns.

### Tool Calling

```swift
struct WeatherInput: Codable { let location: String }
struct WeatherOutput: Codable { let temperature: Double; let conditions: String }

let weatherTool = Tool<WeatherInput, WeatherOutput>(
    name: "get_weather",
    description: "Get current weather",
    parameters: [.required("location", type: .string, description: "City name")]
) { _ in
    WeatherOutput(temperature: 22.0, conditions: "Sunny")
}

let userInput = UserInput(
    prompt: .text("What's the weather in Tokyo?"),
    tools: [weatherTool.schema]
)

let lmInput = try await modelContainer.prepare(input: userInput)
let stream = try await modelContainer.generate(input: lmInput, parameters: GenerateParameters())

for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")
    case .toolCall(let call):
        let result = try await call.execute(with: weatherTool)
        print("\nWeather: \(result.conditions)")
    case .info:
        break
    }
}
```

See [references/tool-calling.md](references/tool-calling.md) for multi-turn tool loops.

### GenerateParameters

```swift
let params = GenerateParameters(
    maxTokens: 1000,            // nil = unlimited
    maxKVSize: 4096,            // Sliding window (RotatingKVCache)
    kvBits: 4,                  // Quantized cache (4 or 8)
    kvGroupSize: 64,            // Quantization group size
    quantizedKVStart: 0,        // Token index to start KV quantization
    temperature: 0.7,           // 0 = greedy / argmax
    topP: 0.9,                  // Nucleus sampling
    repetitionPenalty: 1.1,     // Penalize repeats
    repetitionContextSize: 20,  // Penalty window
    prefillStepSize: 512        // Prompt prefill chunk size
)
```

### Wired Memory (Optional)

Use policy tickets to coordinate concurrent inference memory:

```swift
let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes, kind: .active)

let userInput = UserInput(prompt: "Summarize this text")
let lmInput = try await modelContainer.prepare(input: userInput)

let stream = try await modelContainer.generate(
    input: lmInput,
    parameters: GenerateParameters(),
    wiredMemoryTicket: ticket
)

for await generation in stream {
    if case .chunk(let text) = generation {
        print(text, terminator: "")
    }
}
```

For policy selection, reservations, and measurement-based budgeting, see [references/wired-memory.md](references/wired-memory.md).

### Prompt Caching / History Re-hydration

```swift
let history: [Chat.Message] = [
    .system("You are helpful"),
    .user("Hello"),
    .assistant("Hi there!")
]

let session = ChatSession(modelContainer, history: history)
```

## 5. Secondary Workflow: VLM Inference

### Image Input Types

```swift
let imageFromURL = UserInput.Image.url(fileURL)
let imageFromCI = UserInput.Image.ciImage(ciImage)
let imageFromArray = UserInput.Image.array(mlxArray)
```

### Video Input

```swift
let videoFromURL = UserInput.Video.url(videoURL)
let videoFromAsset = UserInput.Video.avAsset(avAsset)
let videoFromFrames = UserInput.Video.frames(videoFrames)

let response = try await session.respond(to: "What happens in this video?", video: videoFromURL)
```

### Multiple Images

```swift
let images: [UserInput.Image] = [.url(url1), .url(url2)]
let response = try await session.respond(to: "Compare these two images", images: images, videos: [])
```

### VLM-Specific Processing

```swift
let session = ChatSession(
    modelContainer,
    processing: UserInput.Processing(resize: CGSize(width: 512, height: 512))
)
```

## 6. Best Practices

### DO

```swift
// DO: Prefer ChatSession for multi-turn chat UX
let session = ChatSession(modelContainer)

// DO: Prepare UserInput before container-level generation
let userInput = UserInput(prompt: "Hello")
let lmInput = try await modelContainer.prepare(input: userInput)

// DO: Use task-handle variants for early-stop scenarios
let (stream, task) = generateTask(
    promptTokenCount: lmInput.text.tokens.size,
    modelConfiguration: context.configuration,
    tokenizer: context.tokenizer,
    iterator: iterator
)
for await item in stream {
    if shouldStop { break }
}
await task.value

// DO: Use wired tickets when coordinating concurrent workloads
let ticket = WiredSumPolicy().ticket(size: estimatedBytes)
let _ = try await modelContainer.generate(input: lmInput, parameters: params, wiredMemoryTicket: ticket)
```

### DON'T

```swift
// DON'T: Skip prepare(input:) before container-level generation.
// ModelContainer.generate expects LMInput, not UserInput.

// DON'T: Share MLXArray across tasks (not Sendable)
let array = MLXArray(...)
Task { _ = array.sum() } // wrong

// DON'T: Ignore task completion after early-break on low-level streams
for await item in stream {
    if shouldStop { break }
}
// await task.value is required for deterministic cleanup
```

### Thread Safety

- `ModelContainer` is `Sendable` and thread-safe.
- `ChatSession` is not thread-safe; use one session per task/flow.
- `MLXArray` is not `Sendable`; keep it inside one isolation domain or use `SendableBox` transfer patterns.

### Memory Management

```swift
let slidingWindow = GenerateParameters(maxKVSize: 4096)
let quantizedKV = GenerateParameters(kvBits: 4, kvGroupSize: 64)
await session.clear()
```

## 7. Reference Links

| Reference | When to Use |
|-----------|-------------|
| [references/model-container.md](references/model-container.md) | Loading models, ModelContainer API, ModelConfiguration |
| [references/generation.md](references/generation.md) | `generate`, `generateTask`, raw token streaming APIs |
| [references/wired-memory.md](references/wired-memory.md) | Wired tickets, policies, budgeting, reservations |
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

| If you see... | Use instead... |
|---------------|----------------|
| `generate(... didGenerate:)` callback | AsyncStream-based generation APIs |
| `perform { model, tokenizer in }` | `perform { context in }` |
| `TokenIterator(prompt: MLXArray)` | `TokenIterator(input: LMInput)` |
| `ModelRegistry` typealias | `LLMRegistry` or `VLMRegistry` |
| `createAttentionMask(h:cache:[KVCache]?)` | `createAttentionMask(h:cache:KVCache?)` |

## 9. Automatic vs Manual Configuration

### Automatic Behaviors

| Feature | Details |
|---------|---------|
| EOS token loading | Loaded from `config.json` |
| EOS override | `generation_config.json` > `config.json` > defaults |
| EOS merging | All sources merged at generation time |
| EOS detection | Stops generation when EOS encountered |
| Chat template application | Applied by tokenizer / processor path |
| Tool call format detection | Inferred from `model_type` in `config.json` |
| Cache type selection | Driven by `GenerateParameters` (`maxKVSize`, `kvBits`) |
| Tokenizer loading | Loaded automatically from model assets |
| Model weight loading | Downloaded and loaded from Hugging Face/local directory |

### Optional Configuration

| Feature | When to Configure |
|---------|-------------------|
| `extraEOSTokens` | Model has unlisted stop tokens |
| `toolCallFormat` | Override auto-detected tool parser format |
| `maxKVSize` | Enable sliding window cache |
| `kvBits`, `kvGroupSize`, `quantizedKVStart` | Enable and tune KV quantization |
| `prefillStepSize` | Tune prompt prefill chunking/perf tradeoff |
| `wiredMemoryTicket` | Coordinate policy-based wired-memory limits |
