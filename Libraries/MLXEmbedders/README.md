#  MLXEmbedders

This directory contains ports of popular Encoders / Embedding Models. 

## Usage Example

```swift
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers
import MLXLMHuggingFace

let context = try await EmbedderModelFactory.shared.load(
    from: HubClient.default,
    using: TokenizersLoader(),
    configuration: EmbedderRegistry.nomic_text_v1_5
)
let searchInputs = [
    "search_query: Animals in Tropical Climates.",
    "search_document: Elephants",
    "search_document: Horses",
    "search_document: Polar Bears",
]

// Generate embeddings
let model = context.model
let tokenizer = context.tokenizer
let pooling = context.pooling

let inputs = searchInputs.map {
    tokenizer.encode(text: $0, addSpecialTokens: true)
}
// Pad to longest
let maxLength = inputs.reduce(into: 16) { acc, elem in
    acc = max(acc, elem.count)
}

let padded = stacked(
    inputs.map { elem in
        MLXArray(
            elem
                + Array(
                    repeating: tokenizer.eosTokenId ?? 0,
                    count: maxLength - elem.count))
    })
let mask = (padded .!= tokenizer.eosTokenId ?? 0)
let tokenTypes = MLXArray.zeros(like: padded)
let result = pooling(
    model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
    normalize: true, applyLayerNorm: true
)
result.eval()
let resultEmbeddings = result.map { $0.asArray(Float.self) }
```

Load from a local directory:

```swift
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers

let modelDirectory = URL(filePath: "/path/to/embedder")
let context = try await EmbedderModelFactory.shared.load(
    from: modelDirectory,
    using: TokenizersLoader(),
    configuration: EmbedderRegistry.nomic_text_v1_5
)
```

Use a custom Hugging Face client:

```swift
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers
import MLXLMHuggingFace

let hub = HubClient(token: "hf_...")
let context = try await EmbedderModelFactory.shared.load(
    from: hub,
    using: TokenizersLoader(),
    configuration: EmbedderRegistry.nomic_text_v1_5
)
```

Use a custom downloader:

```swift
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers

struct S3Downloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Download files and return a local directory URL.
        return URL(filePath: "/tmp/embedder")
    }
}

let context = try await EmbedderModelFactory.shared.load(
    from: S3Downloader(),
    using: TokenizersLoader(),
    configuration: .init(id: "my-bucket/my-embedder")
)
```


Ported to swift from [taylorai/mlx_embedding_models](https://github.com/taylorai/mlx_embedding_models/tree/main)[^1]

[^1]: Modified by [CodebyCR](https://github.com/CodebyCR) to match test case.
