# MLXFoundationModels

An MLX adapter conforming to Apple's `FoundationModels.LanguageModel`. It provides `MLXLanguageModel` (analogous to `SystemLanguageModel`), usable directly with `LanguageModelSession`, so existing FoundationModels code (guided `@Generable` output, tool calling, streaming) works unchanged. Requires the macOS/iOS/visionOS 27.0 SDK.

## Usage

Build an `MLXLanguageModel` with the `#huggingFaceLanguageModel` macro and pass it to a `LanguageModelSession`.

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
    let model = #huggingFaceLanguageModel(
        configuration: LLMRegistry.qwen3_0_6b_4bit,
        capabilities: [.reasoning])
    let session = LanguageModelSession(model: model)

    let answer = try await session.respond(
        to: "I have three hours near the Loop in Chicago. Is the Art Institute or the Field Museum the better use of my time?")
    print(answer.content)
}
```

The macro synthesizes the `weightsLocation:` and `load:` parameters (Hugging Face download plus tokenizer loading) that you would otherwise pass to the `MLXLanguageModel` initializer by hand.

### Direct initializer

The macro above expands to the call below. Use the initializer directly to point `weightsLocation:` at your own on-disk directory or to swap `load:` for a different downloader or tokenizer.

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
        configuration: LLMRegistry.qwen3_0_6b_4bit,
        capabilities: [.reasoning],
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

Declare what a model may do with the `capabilities:` list at construction. Declaration is explicit: the adapter does not infer capabilities from the model id; it defaults to `[.guidedGeneration]`. A request that exceeds the declared capabilities fails with a typed error.

| Capability | What it enables |
|---|---|
| `.guidedGeneration` | Grammar-constrained output. Pass a `GenerationSchema` to `respond(to:schema:)` or a `@Generable` type to `respond(to:generating:)`, and the result always matches the schema. See [`MLXGuidedGeneration`](../MLXGuidedGeneration/README.md). |
| `.toolCalling` | Expose Swift `Tool`s to the model. |
| `.reasoning` | Run "thinking" models that emit a reasoning trace. |
| `.vision` | Accept image inputs. |

## Availability

`MLXLanguageModel` exposes an `availability` property — `.available`, `.downloading`, `.unavailable(...)` — for gating on model and download state.

## SwiftPM trait

`FoundationModelsIntegration` is the single SwiftPM trait that turns on the adapter, and it is enabled by default. It is the integration point: with it on, `MLXFoundationModels` provides the `MLXLanguageModel` bridge to `FoundationModels.LanguageModel`.

The full capabilities also require the macOS/iOS/visionOS 27.0 SDK. The bridge is guarded by both the trait and the SDK, so anything short of "trait on, 27.0 SDK" compiles `MLXFoundationModels` down to an empty module.

| Trait | SDK | What you get |
|---|---|---|
| On (default) | 27.0 | The full `MLXLanguageModel` adapter bridging to `FoundationModels.LanguageModel`. |
| On (default) | Older | Nothing; the adapter (and its download-progress observable) is compiled out. |
| Off (`.disableDefaultTraits`) | Any | Nothing compiled in. Use this for iOS-17-era consumers that want `MLXLLM` / `MLXLMCommon` without the adapter. |

## See also

To learn more about the `LanguageModel` protocol this adapter conforms to, see [Bring an LLM provider to the Foundation Models framework](https://www.youtube.com/watch?v=u06ZVpSl0J4) from WWDC26.
