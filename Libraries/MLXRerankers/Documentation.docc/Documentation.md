# ``MLXRerankers``

Load and run local text rerankers with one architecture-neutral API.

`RerankerModelFactory` inspects the checkpoint configuration and selects the matching
encoder, causal classifier, or listwise implementation. The initial model support covers
BGE v2 sequence classifiers, Qwen3 causal rerankers, and Jina reranker v3 MLX checkpoints.

## Loading A Reranker

Provide the same downloader and tokenizer loader used by the other mlx-swift-lm model
factories:

```swift
import MLXLMCommon
import MLXRerankers

let reranker = try await RerankerModelFactory.shared.loadContainer(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-Reranker-0.6B-4bit"
)
```

The returned `RerankerContainer` hides the model architecture. Loading
`BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B`, or
`jinaai/jina-reranker-v3-mlx` uses the same calling code when compatible MLX weights are
available.

The downloader is provider-neutral. Applications can pass an internal implementation of
`Downloader` for private registries or managed caches, or resolve a model themselves and use
the local-directory overload.

Automatic loading accepts checkpoints whose identifier declares that they are rerankers and
Jina checkpoints with the explicit `JinaForRanking` architecture. A trusted private checkpoint
with an opaque identifier can opt in explicitly:

```swift
let reranker = try await RerankerModelFactory.shared.loadContainer(
    from: downloader,
    using: tokenizerLoader,
    id: "lampo/private-ranking-model",
    allowUnverifiedModel: true
)
```

Do not enable this option for an arbitrary language or sequence-classification model. Those
architectures can produce valid tensors without having been trained for relevance ranking.

## Reranking Documents

Use `rerank` to return results sorted by descending relevance:

```swift
let response = try await reranker.rerank(
    query: "How does Swift structured concurrency work?",
    documents: candidates,
    topK: 5
)

for result in response.results {
    print(result.score, candidates[result.index])
}
```

Use `scores` when the output must remain aligned with the original document order:

```swift
let response = try await reranker.scores(
    query: query,
    documents: candidates
)
```

Use `RerankDocument` to preserve application identifiers and string metadata without exposing
them to the model:

```swift
let documents = candidates.map {
    RerankDocument(
        id: $0.id,
        text: $0.content,
        metadata: ["source": $0.source]
    )
}

let response = try await reranker.rerank(
    query: query,
    documents: documents,
    topK: 10
)

for result in response.results {
    print(result.document.id, result.score)
}
```

Document identifiers must be unique within a request. Metadata is carried through unchanged
and does not affect scoring.

Each response declares its score semantics through `scoreKind`. Qwen and single-logit BGE
rerankers return model-specific relevance scores normalized to `0...1`, while Jina reranker
v3 returns cosine similarities. A normalized relevance score is not necessarily a calibrated
probability. Do not compare scores or reuse thresholds across models, revisions, quantizations,
prompts, or instructions without application-level evaluation.

## Execution Limits

`RerankExecutionOptions` bounds batch size and token allocation. Pairwise inputs are sorted
by encoded length, micro-batched under both limits, then restored to their original order.
`maxBatchTokens` is a hard forward-pass ceiling: it limits padded pairwise batches and the
complete prompt of a listwise model.
Choose `.error` truncation when silently shortening a candidate is not acceptable:

```swift
let options = RerankExecutionOptions(
    maxBatchSize: 8,
    maxBatchTokens: 4_096,
    truncation: .error
)

let response = try await reranker.rerank(
    RerankRequest(query: query, documents: candidates, topK: 10),
    options: options
)
```

Jina reranker v3 accepts at most 64 documents in one listwise request. Pairwise BGE and
Qwen rerankers use token-budgeted micro-batches and check task cancellation between input
encoding and model batches.

## Model Compatibility

Single-logit encoder rerankers use a sigmoid-normalized relevance score. Multi-label encoder checkpoints
must provide `id2label` or `label2id` metadata that identifies a positive class such as
`relevant`, `positive`, `yes`, or `LABEL_1`; ambiguous classifier heads are rejected. Qwen3
rerankers use the official yes/no logit margin, and Jina reranker v3 uses its listwise marker
representations and cosine similarity.

The integration test project contains revision-pinned checkpoint checks for BGE v2 M3, Qwen3
Reranker 0.6B, and Jina reranker v3. They download large model weights and therefore do not
run as part of the package's normal CI suite.

## Topics

### Model Loading

- ``RerankerModelFactory``
