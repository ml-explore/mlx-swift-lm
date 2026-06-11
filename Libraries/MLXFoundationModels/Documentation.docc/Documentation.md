# ``MLXFoundationModels``

Bridge Apple's `FoundationModels` framework to MLX-powered on-device inference.

## Overview

`MLXFoundationModels` implements `FoundationModels.LanguageModel` using MLX
for the forward pass. This lets any `LanguageModelSession` consumer swap
between Apple's `SystemLanguageModel` and a community MLX model (Qwen,
Llama, Gemma, Phi, etc.) with a one-line constructor change.

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

## Requirements

`MLXFoundationModels` builds against the public `FoundationModels`
framework. The `LanguageModel` protocol and related types this library
conforms to are public on the SDK shipped with the platforms targeted
by this package.

The rest of mlx-swift-lm (MLXLLM, MLXVLM, MLXLMCommon, etc.) is
unaffected and builds alongside on stock Xcode.

To register MLX model architectures with the loader, depend on `MLXLLM`
in your own target alongside `MLXFoundationModels`. `MLXLLM` registers
`TrampolineModelFactory` at module initialization, which is what
`loadModelContainer` consults to pick a backend for a given model
identifier.

## Package traits

`MLXFoundationModels` is gated by two orthogonal SwiftPM traits, both
default-on:

- `FoundationModelsIntegration` controls the `MLXLanguageModel` /
  `MLXLanguageModel.Executor` surface. Disabling it compiles this target
  down to just ``MLXDownloadProgress``.
- `GuidedGenerationSupport` controls grammar-constrained generation via
  vendored xgrammar. Disabling it skips compiling the xgrammar C++
  sources and makes `respond(to:schema:)` / tool-calling paths throw
  `MLXLanguageModelError.guidedGenerationDisabled`.

Consumer configurations:

| Traits enabled | MLXLanguageModel | Guided generation | Chat / tools |
|---|---|---|---|
| Both (default) | Yes | Yes | Yes |
| `FoundationModelsIntegration` only | Yes | No (throws) | Chat yes, tools throw |
| `GuidedGenerationSupport` only | No (symbol absent) | Yes (direct API) | Caller's responsibility |
| Neither | No | No | Only `MLXDownloadProgress` remains |

Select a subset in your `Package.swift`:

```swift
.package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    from: "3.33.0",
    traits: ["GuidedGenerationSupport"]
)
```

## Topics

### Essentials

- ``MLXLanguageModel``
- ``MLXLanguageModel/Executor``
- ``MLXLanguageModel/Availability``

### Download progress

- ``MLXDownloadProgress``

### Guided generation

- <doc:guided-generation>

### Availability and pre-flight

- <doc:availability>
