#  MLXEmbedders

This directory contains ports of popular Encoders / Embedding Models. 

## Usage Example

```swift
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: EmbedderRegistry.nomic_text_v1_5
)
let searchInputs = [
    "search_query: Animals in Tropical Climates.",
    "search_document: Elephants",
    "search_document: Horses",
    "search_document: Polar Bears",
]

// Generate embeddings
let resultEmbeddings = await modelContainer.perform { context -> [[Float]] in
    let inputs = searchInputs.map {
        context.tokenizer.encode(text: $0, addSpecialTokens: true)
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
                        repeating: context.tokenizer.eosTokenId ?? 0,
                        count: maxLength - elem.count))
        })
    // Mask from the real lengths: some tokenizers end every text with a real EOS token.
    let mask = stacked(
        inputs.map { elem in
            MLXArray(
                Array(repeating: true, count: elem.count)
                    + Array(repeating: false, count: maxLength - elem.count))
        })
    let tokenTypes = MLXArray.zeros(like: padded)
    let result = context.pooling(
        context.model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
    )
    result.eval()
    // MLXArray is not Sendable; convert before returning from perform
    return result.map { $0.asArray(Float.self) }
}
```

Load from a local directory:

```swift
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

let modelDirectory = URL(filePath: "/path/to/embedder")
let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
    from: modelDirectory,
    using: #huggingFaceTokenizerLoader()
)
```

Use a custom Hugging Face client:

```swift
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

// HubClient comes from the HuggingFace module; wrap it with #hubDownloader(_:).
let hub = HubClient(host: HubClient.defaultHost, bearerToken: "hf_...")
let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(hub),
    using: #huggingFaceTokenizerLoader(),
    configuration: EmbedderRegistry.nomic_text_v1_5
)
```

Use a custom downloader:

```swift
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

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

let modelContainer = try await EmbedderModelFactory.shared.loadContainer(
    from: S3Downloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: .init(id: "my-bucket/my-embedder")
)
```


Ported to swift from [taylorai/mlx_embedding_models](https://github.com/taylorai/mlx_embedding_models/tree/main)[^1]

[^1]: Modified by [CodebyCR](https://github.com/CodebyCR) to match test case.


## EmbeddingGemma 2

`EmbedderModelFactory` loads `embedding_gemma2` and `embedding_gemma2_text`
checkpoints through the usual local-directory or downloader APIs:

```swift
let container = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
    configuration: EmbedderRegistry.embeddinggemma2)

let vector = try await container.perform { context in
    let text = "task: search result | query: What causes the northern lights?"
    let ids = context.tokenizer.encode(text: text)
    let output = context.model(
        MLXArray(ids)[.newAxis], positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
    let pooled = context.pooling(output, normalize: true)
    try MLX.checkedEval(pooled)
    return pooled.asArray(Float.self)
}
```

The model returns a projected, normalized vector in `pooledOutput`; the factory
uses `.none` pooling even when the checkpoint includes Sentence Transformers'
mean-pooling configuration. For padded batches, pass the padding mask to the
model. The token-level API truncates sequences to 8,192 tokens. It does not add
task prefixes: use `task: search result | query: ` for retrieval queries and
`title: none | text: ` for documents without titles.

The factory retains every encoder present in `config.json`. Its token-level
`EmbeddingModel` interface embeds text. For prepared media tensors, the shared
`MLXLMCommon.EmbeddingGemma2` model also exposes `imageFeatures`, `audioFeatures`,
and `embed(inputIds:attentionMask:softTokens:)`. For media decoding, prompting,
and interleaving, use `MLXVLM.EmbeddingGemma2Embedding`:

```swift
let embeddings = try await EmbeddingGemma2Embedding(
    modelDirectory: directory, tokenizerLoader: #huggingFaceTokenizerLoader())
let vector = try await embeddings.embed(
    .init([.text("Product demo: "), .image(.url(imageURL)), .audio(.url(audioURL))]),
    task: .document, dimensions: 256)
```

`dimensions` accepts 128, 256, 512, or the checkpoint's full output size (768 for
the released model). Truncated vectors are normalized again. Task options include
search, documents, classification, clustering, code retrieval, question answering,
fact checking, sentence similarity, and `.none` for unprompted input. Media-only
inputs receive no task prefix.

To reduce memory, construct the actor with `loadVision: false` or
`loadAudio: false`; unused encoders are omitted before loading weights. Inspect
`supportedModalities` before preparing input. Required processor configuration
errors fail loading. Unsupported input fails before any media encoder runs.

The actor processes batch items sequentially to bound working memory. Images and
video use the checkpoint's token budgets. Video defaults to 1 FPS and at most 32
frames, without its audio track; predecoded frames keep their order before the
frame cap. Audio is mono at 16 kHz and keeps the first 30 seconds. Text truncation
preserves complete media blocks; media that exceeds the shared 8,192-token budget
throws `contextExceeded`. Float16 checkpoint tensors are promoted to float32
before inference; bfloat16 and float32 checkpoints retain their precision.
