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

If you are using an <doc:using#Integration-Packages>, such as [https://github.com/DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx), you would do something similar:

```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(configuration: modelConfiguration)

...
```

becomes:

```swift
import MLXLLM
import MLXLMCommon

import MLXLMHFAPI
import MLXLMTokenizers

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
let model = try await loadModelContainer(
    from: HubClient(),
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
    configuration: configuration
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

### New dependencies

Add your preferred tokenizer and downloader adapters:

```swift
// Before (2.x) – single dependency
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x) – core + adapters
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "3.0.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.2.0", traits: ["Swift"]),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.2.0"),
```

And add their products to your target:

```swift
.product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),

// If you use MLXEmbedders:
.product(name: "MLXEmbeddersTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXEmbeddersHFAPI", package: "swift-hf-api-mlx"),
```

### New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.x)
import MLXLLM
import MLXLMHFAPI      // Downloader adapter
import MLXLMTokenizers // Tokenizer adapter
```

If you use MLXEmbedders:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI      // Downloader adapter
import MLXEmbeddersTokenizers // Tokenizer adapter
```

### Loading API changes

The core APIs now include a `from:` parameter of type `URL` or `any Downloader` as well as a `using:` parameter for the tokenizer loader. Tokenizer integration packages may supply convenience methods with a default tokenizer loader, allowing you to omit the `using:` parameter.

The most visible call-site changes are:

- `hub:` → `from:`: Models are now loaded from a directory `URL` or  `Downloader`.
- `HubApi` → `HubClient`: A new implementation of the Hugging Face Hub client is used.

Example when downloading from Hugging Face:

```swift
// Before (2.x) – hub defaulted to HubApi()
let container = try await loadModelContainer(
    id: "mlx-community/Qwen3-4B-4bit"
)

// After (3.x) – Using Swift Hugging Face + Swift Tokenizers
let container = try await loadModelContainer(
    from: HubClient.default,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

At the lower-level core API, you can still pass any `Downloader` and any `TokenizerLoader` explicitly.

Loading from a local directory:

```swift
// Before (2.x)
let container = try await loadModelContainer(directory: modelDirectory)

// After (3.x)
let container = try await loadModelContainer(from: modelDirectory)
```

Loading with a model factory:

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    from: HubClient.default,
    configuration: modelConfiguration
)
```

Loading an embedder:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI
import MLXEmbeddersTokenizers

let container = try await loadModelContainer(
    from: HubClient.default,
    configuration: .configuration(id: "sentence-transformers/all-MiniLM-L6-v2")
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

The `hub` parameter (previously `HubApi`) has been replaced with `from` (any `Downloader` or `URL` for a local directory). Functions that previously defaulted to `defaultHubApi` no longer have a default – callers must either pass a `Downloader` explicitly or use the convenience methods in `MLXLMHuggingFace` / `MLXEmbeddersHuggingFace`, which default to `HubClient.default`.

For most users who were using the default Hub client, adding `import MLXLMHuggingFace` or `import MLXEmbeddersHuggingFace` and using the convenience overloads is sufficient.

Users who were passing a custom `HubApi` instance should create a `HubClient` instead and pass it as the `from` parameter. `HubClient` conforms to `Downloader` via `MLXLMHuggingFace`.

### `ModelConfiguration`

- `tokenizerId` and `overrideTokenizer` have been replaced by `tokenizerSource: TokenizerSource?`, which supports `.id(String)` for remote sources and `.directory(URL)` for local paths.
- `preparePrompt` has been removed. This shouldn't be used anyway, since support for chat templates is available.
- `modelDirectory(hub:)` has been removed. For local directories, pass the `URL` directly to the loading functions. For remote models, the `Downloader` protocol handles resolution.

### Tokenizer loading

`loadTokenizer(configuration:hub:)` has been removed. Tokenizer loading now uses `AutoTokenizer.from(directory:)` from Swift Tokenizers directly.

`replacementTokenizers` (the `TokenizerReplacementRegistry`) has been removed. Use `AutoTokenizer.register(_:for:)` from Swift Tokenizers instead.

### `defaultHubApi`

The `defaultHubApi` global has been removed. Hugging Face Hub access is now provided by `HubClient.default` from the `HuggingFace` module.

### Low-level APIs

- `downloadModel(hub:configuration:progressHandler:)` → `Downloader.download(id:revision:matching:useLatest:progressHandler:)`
- `loadTokenizerConfig(configuration:hub:)` → `AutoTokenizer.from(directory:)`
- `ModelFactory._load(hub:configuration:progressHandler:)` → `_load(configuration: ResolvedModelConfiguration)`
- `ModelFactory._loadContainer`: removed (base `loadContainer` now builds the container from `_load`)

