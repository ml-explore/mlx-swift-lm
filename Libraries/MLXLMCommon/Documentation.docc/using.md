# Using mlx-swift-lm

How to use mlx-swift-lm in your own tools and applications

## Overview

Using mlx-swift-lm to add LLM, VLM or text embedding capabilities to your own
software is straightforward:

- add a depdendency on `mlx-swift-lm`
- add a dependency on a _Downloader_ and _Tokenizer_
- adapt the API of the Downloader and Tokenizer to conform to the protocols

Then make use of the model:

```swift
import MLXLMCommon
    
let downloader: any Downloader = ...
let tokenizerLoader: any TokenizerLoader = ...

let model = try await loadModel(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

## Downloaders and Tokenizers

There are 3 general ways to select and use concrete Downloader and Tokenizer implementations:

- implementing protocols
- using an integration package
- using [MLXHuggingFace](MLXHuggingFace) macros

If you are <doc:upgrade> from mlx-swift-lm 2.x the macros will be the
simplest way, but consider <doc:#Integration-Packages> as there are alternate
implementations that may provide features and capabilities that you want.

### Implementing Protocols

The other two methods use exactly this technique to wrap concrete 
implementations in the mlx-swift-lm protocol.  You can do this yourself
if you have custom code or simply wish to see how it works.

mlx-swift-lm requires implementation of at least the two tokenizer protocols:

- ``Downloader`` -- required if you need to download weights.  Not needed if you have local weights.
- ``Tokenizer`` -- adapt the concrete tokenizers to the mlx-swift-lm protocol.
- ``TokenizerLoader`` -- factory for ``Tokenizer`` implementations.

You can look at <doc:#Integration-Packages> implementations for examples
of how to write these -- there are only a few properties and methods
and they typically have trivial mappings to the concrete implementation.

This example shows adapting `HuggingFace.HubClient` to the `Downloader` protocol:

```swift
import HuggingFace
import MLXLMCommon

struct HubDownloader: MLXLMCommon.Downloader {
    private let upstream: HubClient

    init(_ upstream: HubClient) {
        self.upstream = upstream
    }
    
    init() {
        self.upstream = HubClient()
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {                        
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }                    
}

// now you can use it
let downloader = HubDownloader()
let tokenizerLoader: any TokenizerLoader = ...

let model = try await loadModel(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

### Integration Packages

Integration packages provide an adapter that encapsulates a concrete
implementation.  Adding a dependency on the adapter will transitively
add a dependency on the implementation.

So which adapter do you chose?

- `huggingface/swift-transformers`
    - this is the package that mlx-swift-lm originally integrated with
- `DePasqualeOrg/swift-tokenizers`
    - Swift Tokenizers is a streamlined and optimized fork of Swift Transformers that focuses solely on tokenizer functionality, with an optional Rust backend for even better performance.

You need a downloader package if you want to download weights at runtime -- this isn't
required if you have some other way to get weights into a local directory.

| Downloader package (implementation) | Adapter                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DePasqualeOrg/swift-hf-api](https://github.com/DePasqualeOrg/swift-hf-api) | [DePasqualeOrg/swift-hf-api-mlx](https://github.com/DePasqualeOrg/swift-hf-api-mlx) |
| [huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface) | [DePasqualeOrg/swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) |

The tokenizer package translates Strings into tokens for model consumption and back:

| Tokenizer package (implementation) | Adapter                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DePasqualeOrg/swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | [DePasqualeOrg/swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) |

See <doc:#Xcode-projects> for information about how to hook it up.

### MLXHuggingFace Macros

To provide parity with mlx-swift-lm 2.x there is a built in integration with
the HuggingFace downloader and tokenizer implementations using macros.

Add these dependencies to your project (see <doc:#Xcode-projects>):

- [https://github.com/huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface)
- [https://github.com/huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)

and add `HuggingFace`, `Tokenizers`, `MLXLLM`, `MLXLMCommon` and `MLXHuggingFace` as libraries that your project links.

You can use the integration like this:

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

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

or if you prefer more explicit downloader and tokenizer loading for more
control:

```swift
import HuggingFace
import Tokenizers

import MLXLLM
import MLXLMCommon
import MLXHuggingFace

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

let model = try await LLMModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

`#hubDownloader()` provides an integration just like what is shown in <doc:#Implementing-Protocols> and `#huggingFaceTokenizerLoader()`
provides something similar to load the tokenizers.

See <doc:upgrade> for more information on upgrading from a 2.x release.

## Xcode projects

You can read the [Xcode documentation](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app).

Click on your project (the top item in the Xcode navigator) and select the **Project** (top item).  Then select **Package Dependencies** and click `+` to add a new dependency.

For all integration methods you will need to add:

- [https://github.com/ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

Beyond that, chose one of the 3 integration methods and add either the adapter packages OR the implementation packages if using macros/local implemenentation.  See <doc#Integration-Packages>.

## Package.swift / SwiftPM

In your Package.swift add a reference to mlx-swift-lm, chosing either the `main` branch or something that tracks versions:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Beyond that, chose one of the 3 integration methods and add either the adapter packages OR the implementation packages if using macros/local implemenentation.  See <doc#Integration-Packages>.

You can use the <doc:#MLXHuggingFace-Macros> like this:

```swift
.package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
```

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]),
```

or one of the integration packages:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.2.0", traits: ["Swift"]),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.2.0"),
```

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),
    ]),
```
