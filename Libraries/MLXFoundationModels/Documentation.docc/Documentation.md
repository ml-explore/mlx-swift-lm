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
import MLXLMCommon
import FoundationModels
import HuggingFace
import Tokenizers

let model = MLXLanguageModel(
    configuration: ModelConfiguration(id: "mlx-community/Qwen3-4B-4bit"),
    capabilities: [.guidedGeneration, .toolCalling],
    weightsLocation: { id in
        // Resolve against the same HubClient cache the loader downloads into.
        let cache = HubCache.default
        guard let repo = Repo.ID(rawValue: id) else { return cache.cacheDirectory }
        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
            let snapshot = try? cache.snapshotPath(
                repo: repo, kind: .model, commitHash: commit)
        {
            return snapshot
        }
        return cache.repoDirectory(repo: repo, kind: .model)
    },
    load: { configuration, progressHandler in
        try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration,
            progressHandler: progressHandler)
    })
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain MLX in one sentence."))
```

## Learn More

To learn more about the `LanguageModel` protocol this adapter conforms to, see *[Bring an LLM provider to the Foundation Models framework](https://www.youtube.com/watch?v=u06ZVpSl0J4)* from WWDC26.

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

`MLXFoundationModels` is gated by one SwiftPM trait, default-on:

- `FoundationModelsIntegration` controls the `MLXLanguageModel` /
  `MLXLanguageModel.Executor` surface. Disabling it compiles this target
  to an empty module.

Grammar-constrained generation lives in the separate `MLXGuidedGeneration`
product, which this target always uses when the adapter is compiled in.

Consumer configurations:

| `FoundationModelsIntegration` | MLXLanguageModel | Guided generation | Chat / tools |
|---|---|---|---|
| On (default) | Yes | Yes | Yes |
| Off | No (symbol absent) | Use `MLXGuidedGeneration` directly | Empty module |

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
