# Upgrade From 2.x Release

Notes on upgrading from mlx-swift-lm 2.x releases.

## Introduction

mlx-swift-lm 3.x has breaking API changes from 2.x:

- Download and Tokenizers are protocols and require concrete implementations
- MLXEmbedders now uses the same download/load infrastructure as MLXLMCommon

See <doc:using> for more information.

This was done for several reasons:

- break the hard dependency on the HuggingFace Hub and Tokenizer implementations
    - this allows other implementations with other design constraints, such as performance optimizations
    - see <doc:using#Integration-Packages>
- provide a mechanism to separate the download of weights and the load of weights

## Selecting a Downloader and Tokenizer

See <doc:using> for details on selecting a Downloader and a Tokenizer and
how to hook these up.

### Using MLXHuggingFace Macros

If using the <doc:using#MLXHuggingFace-Macros>, if you had code like this:

```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(configuration: modelConfiguration)

...
```

you would convert that like this:

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

import HuggingFace
import Tokenizers

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await #huggingFaceLoadModelContainer(
    configuration: modelConfiguration
)

...
```

If you want a little more control over the downloader or the tokenizer loader, that
expands to this:

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

import HuggingFace
import Tokenizers

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: modelConfiguration
)

...
```

### Using Integration Packages

If you are using an <doc:using#Integration-Packages> you would do something similar:

```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(configuration: modelConfiguration)

...
```

becomes:

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: modelConfiguration
)

...
```

## MLXEmbedders

MLXEmbedders requires the same <doc:#Selecting-a-Downloader-and-Tokenizer>.  Additionally,
there are some changes to type names and methods -- these now use the same structure
and mechanism as MLXLMCommon / MLXLLM.

Previously the download and load of the model was done like this:

```swift
import MLXEmbedders

let defaultModelConfiguration = ModelConfiguration.nomic_text_v1_5
let container = try await MLXEmbedders.loadModelContainer(
    hub: HubApi(),
    configuration: configuration
)

// use it ...
```

now, using the <doc:#Using-MLXHuggingFace-Macros> (see 
<doc:#Using-Integration-Packages> for the pattern using other tokenizer
packages):

```swift
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace

import HuggingFace
import Tokenizers

// ModelConfiguration -> EmbedderRegistry
let defaultModelConfiguration = EmbedderRegistry.nomic_text_v1_5

let hub = #hubDownloader()
let loader = #huggingFaceTokenizerLoader()

// MLXEmbedders.loadModelContainer (free function) -> EmbedderModelFactory.shared.loadContainer
let container = try await EmbedderModelFactory.shared.loadContainer(
    from: hub,
    using: loader,
    configuration: defaultModelConfiguration
)

// use it ...
```

These types are removed or replaced:

- `ModelConfiguration` -> use MLXLMCommon
- `ModelConfiguration.nomic_text_v1_5` -> `EmbedderRegistry.nomic_text_v1_5`
- `BaseConfiguration` -> use MLXLMCommon
- `ModelType` - removed
- `ModelContainer` -> EmbedderModelContainer and EmbedderModelContext (matches LLM/VLM concepts)
- `load()` free functions -> EmbedderModelFactory

## Release Notes

Detailed release notes.

### Loading API changes

The core APIs now include a `from:` parameter of type `URL` or `any Downloader` as well as a `using:` parameter for the tokenizer loader. Tokenizer integration packages may supply convenience methods with a default tokenizer loader, allowing you to omit the `using:` parameter.

The most visible call-site changes are:

- `hub:` → `from:`: Models are now loaded from a directory `URL` or  `Downloader`.
- `HubApi` → `HubClient`: A new implementation of the Hugging Face Hub client is used.

Example when downloading from Hugging Face:

```swift
// Before (2.x) – hub defaulted to HubApi()
let container = try await loadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)

// After (3.x) – Using HuggingFace integration macros
import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let model = try await #huggingFaceLoadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)
```

At the lower-level core API, you can still pass any `Downloader` and any `TokenizerLoader` explicitly.

Loading from a local directory:

```swift
// Before (2.x)
let container = try await loadModelContainer(directory: modelDirectory)

// After (3.x)
let container = try await loadModelContainer(
    from: modelDirectory,
    using: #huggingFaceTokenizerLoader()
)
```

Loading with a model factory:

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: modelConfiguration
)
```

Loading an embedder:

```swift
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let container = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: EmbedderRegistry.minilm_l6
)
```

### Renamed methods

`decode(tokens:)` is renamed to `decode(tokenIds:)` to align with the `transformers` library in Python:

```swift
// Before (2.x)
let text = tokenizer.decode(tokens: ids)

// After (3.0)
let text = tokenizer.decode(tokenIds: ids)
```

## Breaking Changes

### Loading API

The `hub` parameter (previously `HubApi`) has been replaced with `from` (any `Downloader` or `URL` for a local directory). Functions that previously defaulted to `defaultHubApi` no longer have a default – callers must pass a `Downloader` explicitly (for example the `#hubDownloader()` macro from `MLXHuggingFace`) or use the `#huggingFaceLoadModelContainer` / `#huggingFaceLoadModel` convenience macros.

For most users who were using the default Hub client, adding `import MLXHuggingFace` (together with `import HuggingFace` and `import Tokenizers`) and using `#hubDownloader()` + `#huggingFaceTokenizerLoader()` — or the `#huggingFaceLoadModelContainer` macro — is sufficient.

Users who were passing a custom `HubApi` instance should create a `HuggingFace.HubClient` instead and wrap it with `#hubDownloader(_:)` to pass as the `from` parameter.

### `ModelConfiguration`

- `tokenizerId` and `overrideTokenizer` have been replaced by `tokenizerSource: TokenizerSource?`, which supports `.id(String)` for remote sources and `.directory(URL)` for local paths.
- `preparePrompt` has been removed. This shouldn't be used anyway, since support for chat templates is available.
- `modelDirectory(hub:)` has been removed. For local directories, pass the `URL` directly to the loading functions. For remote models, the `Downloader` protocol handles resolution.

### Tokenizer loading

`loadTokenizer(configuration:hub:)` has been removed. Tokenizer loading is now done by the `TokenizerLoader` passed to the load functions. `#huggingFaceTokenizerLoader()` uses `AutoTokenizer.from(modelFolder:)` from Swift Transformers.

`replacementTokenizers` (the `TokenizerReplacementRegistry`) has been removed. To load a tokenizer class that `AutoTokenizer` does not support, pass your own `TokenizerLoader`.

### `defaultHubApi`

The `defaultHubApi` global has been removed. Hugging Face Hub access is now provided by the `#hubDownloader()` macro from `MLXHuggingFace`, which wraps a default `HuggingFace.HubClient`.

### Low-level APIs

- `downloadModel(hub:configuration:progressHandler:)` → `Downloader.download(id:revision:matching:useLatest:progressHandler:)`
- `loadTokenizerConfig(configuration:hub:)` → `TokenizerLoader.load(from:)`
- `ModelFactory._load(hub:configuration:progressHandler:)` → `_load(configuration: ResolvedModelConfiguration)`
- `ModelFactory._loadContainer`: removed (base `loadContainer` now builds the container from `_load`)

### `UserInput.Image`, `UserInput.Video` and `UserInput.Audio`

These three media types were enums with these cases:

- `UserInput.Image`: `ciImage`, `url` and `array`.
- `UserInput.Video`: `avAsset`, `url` and `frames`.
- `UserInput.Audio`: `url` and `array`.

Each type is now a struct with a nested `Source` enum. `Source` has the old cases and their payloads. The struct also has a `source` property and an `init(source:)`.

`UserInput.Image` also has an optional `label`. A vision message generator writes the label into the prompt as `[label]`, immediately before the image.

Each old case is now a static function with the same name and argument labels. So the code that creates a media value still compiles:

```swift
let image = UserInput.Image.url(imageURL)
let videos = urls.map(UserInput.Video.url)
let audio: UserInput.Audio = .array(samples)
```

A `switch`, `if case`, `guard case` or `for case` that matches a media value against a case no longer compiles. Match against the `source` of the media value instead:

```swift
// Before
switch video {
case .url(let url): print(url)
default: break
}

// After
switch video.source {
case .url(let url): print(url)
default: break
}
```

