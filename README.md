# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large
language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key features include:

- Integration with the Hugging Face Hub to easily use thousands of LLMs with a single command.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM check out
the [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

# Using MLX Swift LM

The MLXLLM, MLXVLM, MLXLMCommon, and MLXEmbedders libraries are available
as Swift Packages.

Add the following dependency to your Package.swift:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
```

or use the latest release:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
```

Then add one or more libraries to the target as a dependency:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-lm/` to the
`Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in
Xcode.

# Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). You can get started with a wide
variety of open weights LLMs and VLMs using this simplified API:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Or use the underlying API to control every aspect of the evaluation.

# Continuous Batching

Continuous batching lets a single model serve multiple concurrent requests
efficiently by interleaving their token generation in a shared decode loop.
This is an opt-out feature with zero overhead for single requests.

## How It Works

Assign an `InferenceScheduler` to `ModelContainer.scheduler` to enable batching:

```swift
let container = ModelContainer(context: context)
container.scheduler = InferenceScheduler()
```

When only one request is active, the scheduler uses the existing `TokenIterator`
path — no batch overhead at all. When a second request arrives while the first is
still generating, the scheduler automatically upgrades to a `BatchTokenIterator`,
migrating the in-flight KV cache into a batched layout. Third and subsequent
requests join the existing batch on the fly.

## Usage

Callers use the same `ModelContainer.generate(input:parameters:)` API regardless
of whether batching is enabled. Concurrent requests are scheduled transparently:

```swift
let container = ModelContainer(context: context)
container.scheduler = InferenceScheduler()

// Fire two requests concurrently — the scheduler batches them automatically
async let stream1 = container.generate(
    input: try await container.prepare(input: .init(prompt: "Tell me a joke")),
    parameters: .init()
)
async let stream2 = container.generate(
    input: try await container.prepare(input: .init(prompt: "Explain gravity")),
    parameters: .init()
)

for await event in try await stream1 { /* handle events */ }
for await event in try await stream2 { /* handle events */ }
```

## Compatibility

Continuous batching supports standard transformer-based LLMs. The following
request types automatically fall back to the sequential `TokenIterator` path:

- **VLMs** (inputs containing images or video)
- **Hybrid SSM models** (e.g. Mamba-based architectures)
- **Quantized KV caches** (`kvBits` parameter)

No code changes are needed — incompatible requests are detected and routed to
the single-request path automatically.

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
