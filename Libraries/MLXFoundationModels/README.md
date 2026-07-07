# MLXFoundationModels

An MLX-backed drop-in for Apple's `FoundationModels.LanguageModel`. It lets you swap `SystemLanguageModel` for a locally run MLX model and plug it straight into `LanguageModelSession`, so existing FoundationModels code (guided `@Generable` output, tool calling, streaming) works unchanged. Requires the macOS/iOS/visionOS 27.0 SDK.

## Usage

Build an `MLXLanguageModel` with the `#huggingFaceLanguageModel` macro and pass it to a `LanguageModelSession`.

```swift
import FoundationModels
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = #huggingFaceLanguageModel(
        configuration: LLMRegistry.deepSeekR1_7B_4bit,
        capabilities: [.guidedGeneration, .reasoning])
    let session = LanguageModelSession(model: model)

    let answer = try await session.respond(
        to: "I have three hours near the Loop in Chicago. Is the Art Institute or the Field Museum the better use of my time?")
    print(answer.content)
}
```

The macro synthesizes the `weightsLocation:` and `load:` wiring (Hugging Face download plus tokenizer loading) that you would otherwise pass to the `MLXLanguageModel` initializer by hand. Call the initializer directly when you need a custom weights location or loader.

### Direct initializer

The macro above expands to the call below. Reach for the initializer directly to point `weightsLocation:` at your own on-disk directory or to swap `load:` for a different downloader or tokenizer.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = MLXLanguageModel(
        configuration: LLMRegistry.deepSeekR1_7B_4bit,
        capabilities: [.guidedGeneration, .reasoning],
        weightsLocation: { id in
            let cache = HubCache.default
            guard let repo = Repo.ID(rawValue: id) else { return cache.cacheDirectory }
            if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
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
}
```

## Capabilities

Declare what a model may do with the `capabilities:` list at construction. Declaration is explicit: the adapter does not infer capabilities from the model id, it defaults to `[.guidedGeneration]`, and a request that exceeds what was declared fails with a typed error rather than silently degrading.

| Capability | What it enables |
|---|---|
| `.guidedGeneration` | Grammar-constrained output. Pass a `GenerationSchema` to `respond(to:schema:)` or a `@Generable` type to `respond(to:generating:)`, and the result always matches the schema. Enforced by the vendored xgrammar engine in [`MLXGuidedGeneration`](../MLXGuidedGeneration/README.md), which is always compiled in alongside the adapter. |
| `.toolCalling` | Expose Swift `Tool`s to the model and let it invoke them through the standard `LanguageModelSession` tool loop. |
| `.reasoning` | Run "thinking" models that emit a reasoning trace. Left undeclared, a reasoning request throws instead of leaking the trace into the answer. |
| `.vision` | Accept image inputs. Passing an image to a model that did not declare `.vision` throws, so image support stays opt-in and explicit. |

## Trait matrix

`FoundationModelsIntegration` is the single SwiftPM trait that turns on the adapter, and it is enabled by default. It is the integration point: with it on, `MLXFoundationModels` vends the `MLXLanguageModel` bridge to `FoundationModels.LanguageModel`.

The full capabilities also require the macOS/iOS/visionOS 27.0 SDK. The bridge is guarded by both the trait and the SDK, so anything short of "trait on, 27.0 SDK" compiles `MLXFoundationModels` down to an empty module.

| Trait | SDK | What you get |
|---|---|---|
| On (default) | 27.0 | The full `MLXLanguageModel` adapter bridging to `FoundationModels.LanguageModel`. |
| On (default) | Older | Nothing; the adapter (and its download-progress observable) is compiled out. |
| Off (`.disableDefaultTraits`) | Any | Nothing compiled in. Use this for iOS-17-era consumers that want `MLXLLM` / `MLXLMCommon` without the adapter. |

You can use guided generation outside of FoundationModels through [`MLXGuidedGeneration`](../MLXGuidedGeneration/README.md) directly.
