# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/upgrade) for detailed instructions on upgrading.
>
> If that page shows a 404 you can view the source:
> [upgrading](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/upgrade.md) 
> and [using](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md)

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations
- [MLXFoundationModels](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxfoundationmodel): Bridge MLX models into Apple's `FoundationModels.LanguageModel` so they can plug into `LanguageModelSession`. Requires the macOS/iOS 27.0 SDK. Gated by the `FoundationModelsIntegration` package trait (the adapter types; default on). Grammar-constrained generation comes from the separate `MLXGuidedGeneration` library, which this adapter always uses.

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).

## Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose an [integration package for downloaders and tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


## Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). The simplest way to get started is using the `MLXHuggingFace` macros, which provide a default Hugging Face downloader and tokenizer integration.

## Package.swift

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
    .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
],
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
]
```

## Usage

```swift
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let model = try await #huggingFaceLoadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

For alternative integration approaches (custom downloaders, alternative tokenizer packages, local-only weights), see the [using documentation](Libraries/MLXLMCommon/Documentation.docc/using.md).

### MLXFoundationModels: drop-in for `LanguageModelSession`

If you're building on top of Apple's `FoundationModels` framework and want
to swap `SystemLanguageModel` for an MLX-backed model (Qwen, Llama, Gemma,
Phi), depend on `MLXFoundationModels` and pass an `MLXLanguageModel` to
`LanguageModelSession`. Requires the macOS/iOS 27.0 SDK.

```swift
import MLXFoundationModels
import MLXHuggingFace
import FoundationModels
import Hub

let model = MLXLanguageModel(
    modelIdentifier: "mlx-community/Qwen3-4B-4bit",
    capabilities: LanguageModelCapabilities(
        capabilities: [.guidedGeneration, .toolCalling]),
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    locatedBy: { id in HubApi.shared.localRepoLocation(HubApi.Repo(id: id)) }
)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain MLX in one sentence."))
```

Pass a `GenerationSchema` to `respond(to:schema:)` for grammar-constrained
output. The constraint is enforced via the vendored xgrammar library, which
ships in the separate `MLXGuidedGeneration` product and is always available
when the `MLXFoundationModels` adapter is compiled in.

#### Trait matrix

`MLXFoundationModels` exposes one SwiftPM trait, default-on:

| Trait | Gates |
|---|---|
| `FoundationModelsIntegration` | The `MLXLanguageModel` / `MLXLanguageModel.Executor` adapter types that bridge to `FoundationModels.LanguageModel`. Requires the 27.0 SDK to compile. Disabling it compiles `MLXFoundationModels` down to `MLXDownloadProgress` alone. |

Grammar-constrained ("guided") generation lives in the separate
`MLXGuidedGeneration` product. `MLXFoundationModels` always uses it when the
adapter is compiled in, so guided output and tool calling are always available
there. To use guided generation without FoundationModels (older OS floors),
depend on `MLXGuidedGeneration` directly:

```swift
.package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    from: "3.33.0"
)
```

`FoundationModelsIntegration` is default-on; disable it with
`.disableDefaultTraits` (or by not enabling it) for iOS-17-era consumers that
want `MLXLLM` / `MLXLMCommon` without the FoundationModels adapter.
