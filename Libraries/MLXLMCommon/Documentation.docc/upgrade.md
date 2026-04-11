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
