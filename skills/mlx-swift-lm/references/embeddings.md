# Embedding Models

## Overview

The Embedders library provides text embedding models for semantic search, RAG, clustering, and similarity tasks. It supports BERT-family models with various pooling strategies.

**Files:**
- `Libraries/Embedders/EmbeddingModel.swift`
- `Libraries/Embedders/Pooling.swift`
- `Libraries/Embedders/Models.swift`
- `Libraries/Embedders/Load.swift`
- `Libraries/Embedders/Bert.swift`
- `Libraries/Embedders/NomicBert.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `EmbeddingModel` | Protocol for embedding models |
| `ModelContainer` | Thread-safe embedding model wrapper |
| `ModelConfiguration` | Model ID and settings |
| `Pooling` | Pooling strategies for embeddings |
| `Pooling.Strategy` | mean, cls, first, last, max, none |
| `EmbeddingModelOutput` | Hidden states and pooled output |

## Pre-registered Models

| Model | ID | Size |
|-------|-----|------|
| BGE Micro v2 | `TaylorAI/bge-micro-v2` | ~17M |
| GTE Tiny | `TaylorAI/gte-tiny` | ~20M |
| MiniLM L6 | `sentence-transformers/all-MiniLM-L6-v2` | ~22M |
| MiniLM L12 | `sentence-transformers/all-MiniLM-L12-v2` | ~33M |
| BGE Small | `BAAI/bge-small-en-v1.5` | ~33M |
| BGE Base | `BAAI/bge-base-en-v1.5` | ~110M |
| BGE Large | `BAAI/bge-large-en-v1.5` | ~335M |
| Nomic v1 | `nomic-ai/nomic-embed-text-v1` | ~137M |
| Nomic v1.5 | `nomic-ai/nomic-embed-text-v1.5` | ~137M |
| Multilingual E5 | `intfloat/multilingual-e5-small` | ~118M |
| Snowflake XS | `Snowflake/snowflake-arctic-embed-xs` | ~22M |
| Snowflake L | `Snowflake/snowflake-arctic-embed-l` | ~335M |
| BGE M3 | `BAAI/bge-m3` | ~568M |
| Mixedbread Large | `mixedbread-ai/mxbai-embed-large-v1` | ~335M |
| Qwen3 Embedding | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | ~600M |

## Loading Models

### Using Pre-registered Configuration

```swift
import Embedders

let config = ModelConfiguration.bge_small
let container = try await loadModelContainer(configuration: config)
```

### Using Custom Model ID

```swift
let config = ModelConfiguration(id: "BAAI/bge-small-en-v1.5")
let container = try await loadModelContainer(configuration: config)
```

### From Local Directory

```swift
let config = ModelConfiguration(directory: localModelURL)
let container = try await loadModelContainer(configuration: config)
```

### With Progress Tracking

```swift
let container = try await loadModelContainer(
    configuration: config
) { progress in
    print("Download progress: \(progress.fractionCompleted)")
}
```

## Generating Embeddings

### Basic Usage

```swift
let container: ModelContainer = ...

let embedding = await container.perform { model, tokenizer, pooler in
    // Encode text
    let tokens = tokenizer.encode(text: "Hello world")
    let input = MLXArray(tokens).expandedDimensions(axis: 0)

    // Get model output
    let output = model(input)

    // Pool to single vector
    let pooled = pooler(output, normalize: true)

    eval(pooled)
    return pooled
}
```

### Batch Embeddings

```swift
let texts = ["First text", "Second text", "Third text"]

let embeddings = await container.perform { model, tokenizer, pooler in
    // Encode all texts
    let tokensList = texts.map { tokenizer.encode(text: $0) }
    let maxLen = tokensList.map { $0.count }.max() ?? 0

    // Pad to same length
    var padded = [[Int]]()
    var mask = [[Float]]()
    for tokens in tokensList {
        let padding = Array(repeating: 0, count: maxLen - tokens.count)
        padded.append(tokens + padding)
        mask.append(Array(repeating: 1.0, count: tokens.count) +
                   Array(repeating: 0.0, count: padding.count))
    }

    let input = MLXArray(padded)
    let attentionMask = MLXArray(mask)

    // Forward pass
    let output = model(input, attentionMask: attentionMask)

    // Pool
    let pooled = pooler(output, mask: attentionMask, normalize: true)

    eval(pooled)
    return pooled
}
```

## Pooling Strategies

### Available Strategies

```swift
public enum Strategy {
    case mean   // Average all token embeddings (default for most models)
    case cls    // Use CLS token (first token)
    case first  // Use first token
    case last   // Use last token
    case max    // Max over sequence dimension
    case none   // Return raw hidden states
}
```

### Custom Pooling

```swift
// Create custom pooler
let pooler = Pooling(strategy: .mean, dimension: 384)

// Or from configuration file
let pooler = loadPooling(modelDirectory: modelDir)
// Reads from 1_Pooling/config.json
```

### Pooling Options

```swift
let pooled = pooler(
    output,
    mask: attentionMask,      // Attention mask for variable length
    normalize: true,           // L2 normalize output
    applyLayerNorm: false     // Apply layer norm before pooling
)
```

## EmbeddingModel Protocol

```swift
public protocol EmbeddingModel: Module {
    var vocabularySize: Int { get }

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

public struct EmbeddingModelOutput {
    let hiddenStates: MLXArray?  // [batch, seq_len, hidden]
    let pooledOutput: MLXArray?  // [batch, hidden] (CLS pooling)
}
```

## Use Cases

### Semantic Search

```swift
// Encode query and documents
let queryEmb = await embed(container, "What is machine learning?")
let docEmbs = await embedBatch(container, documents)

// Compute similarities
let similarities = matmul(queryEmb, docEmbs.T)
let topK = argSort(similarities, axis: -1)[0..., (-k)...]
```

### RAG (Retrieval-Augmented Generation)

```swift
// 1. Index documents
let docEmbeddings = await embedBatch(container, documents)
// Store in vector database...

// 2. Retrieve relevant docs
let queryEmb = await embed(container, userQuery)
let relevantDocs = vectorDB.search(queryEmb, topK: 5)

// 3. Generate with context
let context = relevantDocs.map { $0.text }.joined(separator: "\n")
let prompt = "Context: \(context)\n\nQuestion: \(userQuery)"
let response = try await llmSession.respond(to: prompt)
```

### Clustering

```swift
let embeddings = await embedBatch(container, texts)
// Use with clustering algorithms (k-means, DBSCAN, etc.)
```

### Similarity Scoring

```swift
let emb1 = await embed(container, "The cat sat on the mat")
let emb2 = await embed(container, "A cat was sitting on a rug")

// Cosine similarity (embeddings already normalized)
let similarity = sum(emb1 * emb2).item(Float.self)
print("Similarity: \(similarity)")  // ~0.85
```

## Model Configuration

```swift
public struct ModelConfiguration: Sendable {
    public enum Identifier: Sendable {
        case id(String)          // HuggingFace ID
        case directory(URL)      // Local path
    }

    public var id: Identifier
    public var name: String
    public let tokenizerId: String?       // Alternate tokenizer
    public let overrideTokenizer: String? // Override tokenizer class
}
```

### Registry

```swift
// Get registered model
let config = await ModelConfiguration.configuration(id: "bge-small")

// List all models
let models = await ModelConfiguration.models

// Register custom model
await ModelConfiguration.register(configurations: [myConfig])
```

## Supported Architectures

| Architecture | model_type |
|-------------|------------|
| BERT | `bert` |
| Nomic BERT | `nomic_bert` |
| Qwen3 | `qwen3` |

## Memory Considerations

| Model Size | Approximate Memory |
|------------|-------------------|
| Micro (~17M) | ~70MB |
| Small (~33M) | ~130MB |
| Base (~110M) | ~440MB |
| Large (~335M) | ~1.3GB |

## Deprecated Patterns

### quantization property

```swift
// DEPRECATED
baseConfig.quantization

// USE INSTEAD
baseConfig.perLayerQuantization
```

No major deprecations specific to embedding functionality. The embedding API is stable.
