# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large
language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key features include:

- Provider-agnostic model loading via the `Downloader` and `TokenizerLoader` protocols, with optional integration packages for Hugging Face Hub and tokenizer implementations.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM check out
the [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

# Using MLX Swift LM

The MLXLLM, MLXVLM, MLXLMCommon, and MLXEmbedders libraries are available
as Swift Packages. mlx-swift-lm has no external dependencies beyond
[mlx-swift](https://github.com/ml-explore/mlx-swift) – tokenizer and
downloader implementations are provided by separate integration packages:

| Package | Modules | Purpose |
|---|---|---|
| [swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) | `MLXLMTokenizers` | Bridges [Swift Tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) to the `TokenizerLoader` protocol |
| [swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) | `MLXLMTransformers` | Bridges [Swift Transformers](https://github.com/huggingface/swift-transformers) to the `TokenizerLoader` protocol |
| [swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) | `MLXLMHuggingFace`, `MLXEmbeddersHuggingFace` | Bridges [Hugging Face Hub](https://github.com/huggingface/swift-huggingface) to the `Downloader` protocol, with convenience loading functions |

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
```

Then add one or more integration packages:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", branch: "main"),
.package(url: "https://github.com/DePasqualeOrg/swift-huggingface-mlx/", branch: "main"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHuggingFace", package: "swift-huggingface-mlx"),
    ]),
```

# Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). You can get started with a wide
variety of open weights LLMs and VLMs using this simplified API:

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let model = try await loadModel(
    from: HubClient.default,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Integration packages may provide convenience overloads that omit the
`from:` or `using:` parameter – see their documentation for details.

More loading scenarios:

Load from a local directory:

```swift
import MLXLLM
import MLXLMTokenizers

let modelDirectory = URL(filePath: "/path/to/model")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

Use a custom Hugging Face client:

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let hub = HubClient(token: "hf_...")
let container = try await loadModelContainer(
    from: hub,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
```

Use a custom downloader:

```swift
import MLXLLM
import MLXLMCommon
import MLXLMTokenizers

struct S3Downloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Download files and return a local directory URL.
        return URL(filePath: "/tmp/model")
    }
}

let container = try await loadModelContainer(
    from: S3Downloader(),
    using: TokenizersLoader(),
    id: "my-bucket/my-model"
)
```

Or use the underlying API to control every aspect of the evaluation.

# Migrating to Version 3

Version 3 decouples the tokenizer and downloader from mlx-swift-lm. This package no longer bundles a tokenizer or Hugging Face Hub client – you choose them via separate integration packages. This allows for downloading from a variety of sources as well as different tokenizer implementations.

## New dependencies

Add integration packages alongside mlx-swift-lm:

```swift
// Before (2.x) – single dependency
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x) – core + integration packages
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "3.0.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "1.0.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-huggingface-mlx/", from: "1.0.0"),
```

And add their products to your target:

```swift
.product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXLMHuggingFace", package: "swift-huggingface-mlx"),
```

See [Using MLX Swift LM](#using-mlx-swift-lm) for full setup details.

## New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.0)
import MLXLLM
import MLXLMHuggingFace    // for HubClient + Downloader conformance
import MLXLMTokenizers      // for TokenizersLoader
```

## Loading API changes

All loading functions now require an explicit `using:` parameter for the tokenizer loader. The `hub:` parameter has been replaced by `from:` with a `Downloader` conformance.

```swift
// Before (2.x) – hub defaulted to HubApi()
let container = try await loadModelContainer(
    id: "mlx-community/Qwen3-4B-4bit"
)

// After (3.0) – downloader and tokenizer loader are explicit
let container = try await loadModelContainer(
    from: HubClient.default,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
```

Integration packages may provide convenience overloads that omit the `from:` or `using:` parameter – see their documentation for details.

Loading from a local directory:

```swift
// Before (2.x)
let container = try await loadModelContainer(directory: modelDirectory)

// After (3.0)
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

## Renamed methods

`decode(tokens:)` is renamed to `decode(tokenIds:)` to align with the `transformers` library in Python, where "tokens" refers to string representations and "token IDs" refers to integer representations:

```swift
// Before (2.x)
let text = tokenizer.decode(tokens: ids)

// After (3.0)
let text = tokenizer.decode(tokenIds: ids)
```

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations
