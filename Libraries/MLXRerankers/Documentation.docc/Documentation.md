# ``MLXRerankers``

Load and run local text rerankers with one architecture-neutral API.

`RerankerModelFactory` inspects the checkpoint configuration and selects the matching
encoder, causal classifier, or listwise implementation. Supported families include BGE v2
sequence classifiers, official Qwen3 causal rerankers, Zerank-2, ContextualAI v2 multilingual
1B, and Jina reranker v3 and v3.5 checkpoints.

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

Automatic loading selects a supported prompt protocol from the model family in its identifier,
then validates the checkpoint configuration. Sharing the Qwen3 backbone or having `rerank` in
the name does not establish protocol compatibility. Conversions keep their source name as a
prefix, so `mlx-community/zerank-2-4bit` resolves to Zerank-2.

A renamed or fine-tuned checkpoint that keeps its base model's protocol declares it with
``RerankerFamily``. The family must match the checkpoint architecture:

```swift
let reranker = try await RerankerModelFactory.shared.loadContainer(
    from: downloader,
    using: tokenizerLoader,
    id: "lampo/private-ranking-model",
    family: .zerank2
)
```

`allowUnverifiedModel: true` opts a trusted custom checkpoint in to the official
Qwen3-Reranker protocol, or for `JinaForRanking` to v3 or v3.5 depending on whether its
configuration enables sliding-window attention. Do not enable this option for an arbitrary
language or sequence-classification model. Those architectures can produce valid tensors
without having been trained for relevance ranking. Neither option bypasses malformed or
incompatible scoring metadata.

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

Each response declares its score semantics through `scoreKind`. Official Qwen3 and single-logit
BGE rerankers return model-specific relevance scores normalized to `0...1`. Zerank-2 and
ContextualAI return raw logits (`.logit`); Jina v3 and v3.5 return cosine similarities.
A normalized relevance score is not necessarily a calibrated probability. Do not compare
scores or reuse thresholds across models, revisions, quantizations,
prompts, or instructions without application-level evaluation.

## Execution Limits

`RerankExecutionOptions` bounds batch size and token allocation. Pairwise inputs are sorted
by encoded length, micro-batched under both limits, then restored to their original order.
`maxBatchTokens` is a hard forward-pass ceiling: it limits padded pairwise batches, the
complete Jina v3 prompt, and each Jina prefill step.
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
causal rerankers use token-budgeted micro-batches and check task cancellation between input
encoding and model batches.

Jina v3.5 uses dual query markers, interleaved sliding attention, and weighted query fusion
across blocks of at most 125 documents. Its reference token limits are 1,984 query tokens,
8,191 tokens per document, and the model context (131,072 tokens) per block. Blocks follow
this reference geometry regardless of execution options, so scores do not depend on
`maxBatchTokens`; each block is prefilled in steps of at most `prefillStepSize` and
`maxBatchTokens` tokens. `.error` rejects content that requires truncation.

Qwen3 projects only each row's final valid hidden state when scoring a batch, avoiding
vocabulary logits for the other input tokens. Singleton requests retain cached prefill.

## Model Compatibility

Single-logit encoder rerankers use a sigmoid-normalized relevance score. Multi-label encoder checkpoints
must provide `id2label` or `label2id` metadata that identifies a positive class such as
`relevant`, `positive`, `yes`, or `LABEL_1`; ambiguous classifier heads are rejected. Qwen3
rerankers use the official yes/no logit margin. Zerank-2 renders a query/document chat and
reads the raw `Yes` logit; ContextualAI renders its document/query prompt and reads the raw
bfloat16 logit at token 0. Jina rerankers use listwise marker representations and cosine similarity.
For Zerank-2, include any instructions in the query; its published chat template ignores
separate system instructions.

When present, `1_LogitScore/config.json` supplies the classifier token IDs. Its score shape,
input name, and token IDs must match the selected protocol. Unknown causal reranker families
are rejected rather than silently receiving the official Qwen3 prompt.

The integration test project contains revision-pinned checkpoint checks for BGE v2 M3, Qwen3
Reranker 0.6B, Zerank-2, ContextualAI, and Jina v3 and v3.5. The v3.5 checks include a prompt
longer than its sliding window. They download large model weights and therefore do not
run as part of the package's normal CI suite.

## Topics

### Model Loading

- ``RerankerModelFactory``
- ``RerankerFamily``
