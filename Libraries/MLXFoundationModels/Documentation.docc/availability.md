# Availability and pre-flight checks

Resolve where weights live, gate UI on download state, and check disk
space before kicking off a download.

## Overview

Three things must be true for an `MLXLanguageModel` to serve a request:
the device has a Metal GPU, the model weights exist on disk at the
configured location, and no in-flight download is already running.
``MLXLanguageModel/availability`` rolls all three into a single value
suitable for driving UI affordances ("Download", "Downloading...",
"Ready").

`.downloading` always means bytes are actively being fetched. A background
``MLXLanguageModel/Executor/prewarm(model:transcript:)`` (via
`session.prewarm()`) of an *already-downloaded* model deliberately does not
flip an `.available` model to `.downloading` — only a genuine in-flight
fetch reports it. Don't treat `.downloading` as a proxy for "any loading
activity"; a prewarm's shader warmup happens silently while the state stays
`.available`.

```swift
switch await model.availability {
case .available:
    button.title = "Ask"
case .downloading:
    button.title = "Downloading..."
case .unavailable(.modelNotDownloaded):
    button.title = "Download (\(humanReadable(remoteSizeBytes)))"
case .unavailable(.downloadFailed):
    button.title = "Retry"
case .unavailable(.deviceNotCapable):
    button.title = "Not supported"
}
```

`availability` is fast: it inspects local on-disk state and the
in-process model cache without any network I/O.

## The weights-location closure

`MLXLanguageModel` doesn't assume Hugging Face. The on-disk location for
a given model identifier comes from the closure you supply at init:

```swift
public init(
    configuration: ModelConfiguration,
    capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration],
    configurationResolver: any ModelConfigurationResolver = DefaultConfigurationResolver(),
    weightsLocation: @Sendable @escaping (String) -> URL,
    load: @escaping ContainerLoader
)
```

For Hugging Face Hub-backed weights, `MLXHuggingFace` exports a free
function you can pass directly:

```swift
import MLXHuggingFace
import Hub

let model = MLXLanguageModel(
    modelID: "mlx-community/Qwen3-4B-4bit",
    capabilities: LanguageModelCapabilities(
        capabilities: [.guidedGeneration, .toolCalling]),
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    locatedBy: { id in HubApi.shared.localRepoLocation(HubApi.Repo(id: id)) }
)
```

For a private CDN, custom on-disk layout, or shared cache:

```swift
let model = MLXLanguageModel(
    configuration: ModelConfiguration(id: "internal/MyModel-v3"),
    capabilities: [.guidedGeneration, .toolCalling],
    weightsLocation: { id in
        URL(fileURLWithPath: "/Volumes/SharedCache/models/\(id)")
    },
    load: { configuration, progressHandler in
        try await loadModelContainer(
            from: corpDownloader,
            using: corpTokenizerLoader,
            configuration: configuration,
            progressHandler: progressHandler)
    })
```

## Disk-space pre-flight

Before kicking off a download, check the on-disk free space. Sum the
sibling file sizes from the `Hub` client of your choice, then compare
against `freeDiskSpaceBytes`:

```swift
import Hub

let metadata = try await HubApi.shared.getFileMetadata(from: HubApi.Repo(id: id))
let remote = metadata.reduce(Int64(0)) { $0 + Int64($1.size ?? 0) }
if let free = model.freeDiskSpaceBytes,
   free < remote + safetyMargin {
    showDiskSpaceWarning(needed: remote, free: free)
    return
}
try await model.preload()
```

`HubApi.getFileMetadata(from:)` issues a HEAD request per sibling file
in the repo and returns the sizes; it requires network.
``MLXLanguageModel/freeDiskSpaceBytes`` is a synchronous
`URLResourceValues` lookup against the volume hosting
`weightsLocation(modelID)`.

If your weights live on a custom CDN, expose your own remote-size helper
and feed its result into the same comparison.
